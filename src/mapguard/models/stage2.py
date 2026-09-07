"""Stage 2 - the changeset classifier. Five branches, masked attention fusion.

Written to build under **both Keras 2 and Keras 3**, because the two machines this runs on
disagree. Gate 1d verified the architecture locally on Keras 3.15 / TF 2.21, but the SVKM
cluster's `tensorflow_pbs:23.07-tf2-py3` image is NGC 23.07, which ships TF 2.13 and Keras
2.x. Discovering that in the middle of a training run would cost a day, so:

  - everything goes through `tf.keras`, never a bare `import keras`
  - tensor ops are `tf.*`, never `keras.ops.*` (Keras 3 only)
  - no `Input(..., dtype=...)` tricks or Keras-3-only layer kwargs

The branches, and the gap each closes (PROJECT_PLAN section 5.3):

    B1  geometry     2-channel before/after raster per edit      -> G1
    B2  text         char-CNN, multi-width parallel kernels      -> G2
    B3  edit seq     BiLSTM over sampled edits in commit order   -> G4
    B4  user history GRU over point-in-time prior changesets     -> G3
    B5  tabular      MLP over changeset-level features           -> parity

Fusion is multi-head attention pooling over the K sampled edits. The mask is not optional
decoration: a model that attends to padding trains fine, scores fine, and is wrong, and
the failure presents as "geometry did not help". `verify_masking` below is the same check
Gate 1d runs, exposed so the training script can assert it on the real model.

Three outputs, because a reviewer needs to know where to look:
    p(damaging) - per-edit attention weights - branch attribution
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def pairwise_mask(mask: tf.Tensor) -> tf.Tensor:
    """(N, K) key mask -> (N, K, K) attention mask.

    A padded edit must be unable to attend and unable to be attended to; masking only one
    direction still lets padding influence the output through the softmax denominator.
    """
    m = tf.cast(mask, tf.bool)
    return tf.logical_and(m[:, :, None], m[:, None, :])


class MaskedMeanPool(layers.Layer):
    """Mean over the real edits only.

    `GlobalAveragePooling1D(mask=...)` exists in both Keras versions but its keyword
    handling differs, so the arithmetic is written out. Dividing by the true count rather
    than by K also stops a changeset with 3 real edits reading as one with 253 zeros.
    """

    def call(self, x, mask):
        m = tf.cast(mask, x.dtype)[:, :, None]
        return tf.reduce_sum(x * m, axis=1) / tf.maximum(tf.reduce_sum(m, axis=1), 1e-6)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


def build_model(
    *,
    max_edits: int,
    edit_features: int,
    text_len: int,
    vocab_size: int,
    user_steps: int,
    user_features: int,
    tabular_features: int,
    raster: int | None = None,
    embed: int = 32,
    heads: int = 4,
    dropout: float = 0.2,
    branches: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5"),
) -> keras.Model:
    """The full model, or any subset of branches for the ablation grid.

    `branches` is what makes the ablation table cheap: every row in PROJECT_PLAN section 11
    is this function with a different tuple, so the configurations cannot drift apart the
    way parallel hand-written models do.

    `raster=None` runs B1 on scalar geometry features only. The rasters are the expensive
    part and Gate 1b already forced a geometry subset, so scalar-only is the honest default
    until raster coverage is measured.
    """
    inputs: dict[str, keras.Input] = {}
    heads_out: list[tf.Tensor] = []

    edit_in = keras.Input(shape=(max_edits, edit_features), name="edit_features")
    mask_in = keras.Input(shape=(max_edits,), name="edit_mask")
    inputs["edit_features"] = edit_in
    inputs["edit_mask"] = mask_in

    per_edit = [edit_in]

    # --- B1 geometry -------------------------------------------------------------
    if "B1" in branches and raster:
        geom_in = keras.Input(shape=(max_edits, raster, raster, 2), name="geometry")
        inputs["geometry"] = geom_in
        g = layers.TimeDistributed(layers.Conv2D(8, 3, activation="relu", padding="same"))(geom_in)
        g = layers.TimeDistributed(layers.MaxPooling2D(2))(g)
        g = layers.TimeDistributed(layers.Conv2D(16, 3, activation="relu", padding="same"))(g)
        g = layers.TimeDistributed(layers.GlobalAveragePooling2D())(g)
        per_edit.append(g)

    # --- B2 text: multi-width parallel kernels, the Inception idea ---------------
    if "B2" in branches:
        text_in = keras.Input(shape=(text_len,), name="text")
        inputs["text"] = text_in
        e = layers.Embedding(vocab_size, 24, name="char_embed")(text_in)
        conv = [
            layers.GlobalMaxPooling1D()(
                layers.Conv1D(24, w, activation="relu", padding="same", name=f"char_conv{w}")(e)
            )
            for w in (3, 5, 7)
        ]
        heads_out.append(layers.Concatenate(name="b2_text")(conv))

    # --- per-edit trunk, shared by B3 and the attention pooling ------------------
    x = layers.Concatenate(axis=-1)(per_edit) if len(per_edit) > 1 else per_edit[0]
    x = layers.TimeDistributed(layers.Dense(embed, activation="relu"), name="edit_encoder")(x)

    if "B3" in branches:
        x = layers.Bidirectional(
            layers.LSTM(embed // 2, return_sequences=True), name="b3_edit_sequence"
        )(x, mask=tf.cast(mask_in, tf.bool))

    attn_layer = layers.MultiHeadAttention(
        num_heads=heads, key_dim=max(8, embed // heads), name="edit_attention"
    )
    attended, attn_scores = attn_layer(
        x, x, attention_mask=pairwise_mask(mask_in), return_attention_scores=True
    )
    pooled = MaskedMeanPool(name="masked_pool")(attended, mask_in)
    heads_out.append(pooled)

    # Per-edit suspicion: how much the rest of the changeset attends to each edit,
    # averaged over heads and queries. This is output 2, and it costs nothing extra.
    edit_weight = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=[1, 2]), name="edit_attention_weight"
    )(attn_scores)

    # --- B4 user history ---------------------------------------------------------
    if "B4" in branches:
        user_in = keras.Input(shape=(user_steps, user_features), name="user_sequence")
        umask_in = keras.Input(shape=(user_steps,), name="user_mask")
        inputs["user_sequence"] = user_in
        inputs["user_mask"] = umask_in
        u = layers.GRU(32, name="b4_user_history")(user_in, mask=tf.cast(umask_in, tf.bool))
        heads_out.append(u)

    # --- B5 tabular --------------------------------------------------------------
    if "B5" in branches:
        tab_in = keras.Input(shape=(tabular_features,), name="tabular")
        inputs["tabular"] = tab_in
        t = layers.Dense(64, activation="relu")(tab_in)
        t = layers.Dropout(dropout)(t)
        heads_out.append(layers.Dense(32, activation="relu", name="b5_tabular")(t))

    fused = layers.Concatenate(name="fusion")(heads_out) if len(heads_out) > 1 else heads_out[0]
    h = layers.Dense(64, activation="relu")(fused)
    h = layers.Dropout(dropout)(h)
    h = layers.LayerNormalization()(h)
    out = layers.Dense(1, activation="sigmoid", name="p_damaging")(h)

    model = keras.Model(
        inputs=[inputs[k] for k in sorted(inputs)],
        outputs=[out, edit_weight],
        name="mapguard_stage2",
    )
    model.branch_config = tuple(branches)      # recorded in every results file
    return model


def focal_loss(gamma: float = 2.0, alpha: float = 0.5):
    """Focal loss (Lin et al.). Down-weights easy examples so the rare hard ones matter.

    Compared against plain BCE in the ablation - PROJECT_PLAN section 11. It is not
    assumed to help; imbalance handling is one of the things being measured.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), p, 1 - p)
        w = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -tf.reduce_mean(w * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return loss


def verify_masking(model: keras.Model, inputs: dict, mask_key: str = "edit_mask") -> dict:
    """Assert padded edits cannot change the output. The Gate 1d check, on the real model.

    Run this once per training script, not once per project. The architecture changes, the
    branches get toggled for ablations, and the failure is silent every time.
    """
    order = [t.name.split(":")[0] for t in model.inputs]
    base = {k: np.asarray(v) for k, v in inputs.items()}
    before = model.predict([base[k] for k in order], verbose=0)[0]

    rng = np.random.default_rng(0)
    pad = 1.0 - base[mask_key]
    perturbed = {k: v.copy() for k, v in base.items()}
    for key, arr in perturbed.items():
        if key in (mask_key, "text", "tabular", "user_sequence", "user_mask"):
            continue
        if arr.ndim >= 2 and arr.shape[1] == pad.shape[1]:
            shape = (1,) * 0 + pad.shape + (1,) * (arr.ndim - pad.ndim)
            perturbed[key] = arr + pad.reshape(shape) * rng.normal(size=arr.shape) * 50.0

    after = model.predict([perturbed[k] for k in order], verbose=0)[0]
    delta = float(np.max(np.abs(before - after)))
    return {
        "max_delta": delta,
        "masking_holds": bool(delta < 1e-4),
        "note": "padded edits were perturbed by 50x noise; a correct model does not move",
    }
