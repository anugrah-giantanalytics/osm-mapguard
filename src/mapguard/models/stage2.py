"""Stage 2 - the changeset classifier. Five branches, masked attention fusion.

Must build and train under **both Keras 2 and Keras 3**. The SVKM cluster image
(`tensorflow_pbs:23.07-tf2-py3`) is TF 2.13 / Keras 2; a current TensorFlow on a laptop
or an M4 is Keras 3. The two disagree in one place that matters here:

    Keras 3 refuses raw TensorFlow ops on symbolic tensors while a model is being built.

An earlier version called `tf.cast` and `tf.logical_and` directly on `keras.Input`
tensors. Keras 2 quietly wrapped those; Keras 3 raises. So every tensor operation in the
graph now lives inside a small `Layer` subclass, where it runs on real tensors at call
time. That is the only pattern that behaves identically under both.

The branches, and the gap each closes (PROJECT_PLAN section 5.3):

    B1  geometry     scalar per-edit geometry, optional raster CNN  -> G1
    B2  text         char-CNN, multi-width parallel kernels         -> G2
    B3  edit seq     BiLSTM over sampled edits in commit order      -> G4
    B4  user history GRU over point-in-time prior changesets        -> G3
    B5  tabular      MLP over changeset-level features              -> parity
    B6  edit meaning tokens of what each edit did (+leisure=park, -highway,
                     ~name) and its old->new text, per edit            -> v2 (2026-09-22)

Pooling over edits (v2): a changeset is a *bag* of edits and one bad edit makes it bad -
multiple-instance learning. Round 1 averaged the edits, which dilutes one bad edit among
200 good ones, and its attention came out near-uniform (~1/n). v2 pools with gated attention
(Ilse, Tomczak & Welling 2018) *and* a masked max, so a single strong edit survives, and the
attention weights become the "which edit" explanation. `pooling="mean"` keeps the round-1
behaviour for the ablation that proves the difference.

Outputs:
    p_damaging              (N, 1)  the score
    edit_attention_weight   (N, K)  which edit the model attended to - output 2 in the plan
Branch attribution (output 3) is computed at evaluation time by occlusion; see
`train/common/attribution.py`. It needs no extra graph.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ALL_BRANCHES = ("B1", "B2", "B3", "B4", "B5", "B6")


# ------------------------------------------------------------------- mask layers


class ToBool(layers.Layer):
    """float mask -> bool mask, as a layer so Keras 3 accepts it."""

    # The argument must not be called `mask`: Keras reserves that name for mask propagation,
    # and Keras 3 then passes nothing as the input (measured: wiring check, Keras 3.15).
    def call(self, inputs):
        return tf.cast(inputs, tf.bool)

    def compute_output_shape(self, input_shape):
        return input_shape


class PairwiseMask(layers.Layer):
    """(N, K) key mask -> (N, K, K) attention mask.

    A padded edit must be unable to attend *and* unable to be attended to; masking only one
    direction still lets padding leak in through the softmax denominator.
    """

    def call(self, inputs):
        m = tf.cast(inputs, tf.bool)
        return tf.logical_and(m[:, :, None], m[:, None, :])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])


class MaskedMeanPool(layers.Layer):
    """Mean over the real edits only.

    Takes `[values, mask]` as one list input rather than a `mask=` argument: `mask` is a
    reserved keyword in Keras's call machinery and the two versions treat it differently.
    Dividing by the true count also stops 3 real edits reading like 253 zeros.
    """

    def call(self, inputs):
        x, mask = inputs
        keep = tf.cast(mask, tf.bool)[:, :, None]
        # where, not multiply: a padded row that overflowed in float16 is inf/NaN, and
        # NaN * 0 is NaN - it would leak through a multiplicative mask.
        x = tf.where(keep, x, tf.zeros_like(x))
        m = tf.cast(keep, x.dtype)
        return tf.reduce_sum(x, axis=1) / tf.maximum(tf.reduce_sum(m, axis=1), 1.0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


class AttentionReceived(layers.Layer):
    """(N, heads, K, K) scores -> (N, K): attention each edit received, over real queries.

    Averaging over *all* queries would let padded query rows dilute the weight, so the
    mean is taken over real queries only, and padded keys read exactly 0.
    """

    def call(self, inputs):
        scores, mask = inputs
        m = tf.cast(mask, scores.dtype)                     # (N, K)
        per_query = tf.reduce_mean(scores, axis=1)          # (N, K, K) mean over heads
        real_q = tf.cast(mask, tf.bool)[:, :, None]
        weighted = tf.where(real_q, per_query, tf.zeros_like(per_query))  # drop padded queries
        received = tf.reduce_sum(weighted, axis=1)          # (N, K)
        n_real = tf.maximum(tf.reduce_sum(m, axis=1, keepdims=True), 1.0)
        return tf.cast((received / n_real) * m, "float32")

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


class MaskedTokenMean(layers.Layer):
    """[(N, K, T, D) embeddings, (N, K, T) ids] -> (N, K, D): mean over real tokens (id > 0)."""

    def call(self, inputs):
        emb, ids = inputs
        m = tf.cast(tf.greater(ids, 0), emb.dtype)[..., None]
        return tf.reduce_sum(emb * m, axis=2) / tf.maximum(tf.reduce_sum(m, axis=2), 1.0)

    def compute_output_shape(self, input_shape):
        s = input_shape[0]
        return (s[0], s[1], s[3])


class MaskedMax(layers.Layer):
    """[(N, K, D), (N, K) mask] -> (N, D): strongest real edit per feature. Padding can never
    win: it is replaced by a large negative before the max."""

    def call(self, inputs):
        x, mask = inputs
        m = tf.cast(mask, tf.bool)[:, :, None]
        return tf.reduce_max(tf.where(m, x, tf.cast(-1e4, x.dtype) * tf.ones_like(x)), axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


class GatedAttentionPool(layers.Layer):
    """Gated attention MIL pooling (Ilse et al. 2018) over real edits.

    a_k = softmax_k( w^T (tanh(V h_k) * sigmoid(U h_k)) ),  z = sum_k a_k h_k
    Returns [z (N, D), a (N, K) float32]; padded edits get exactly zero weight.
    """

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.v = layers.Dense(units, activation="tanh", name="mil_v")
        self.u = layers.Dense(units, activation="sigmoid", name="mil_u")
        self.w = layers.Dense(1, name="mil_w")

    def call(self, inputs):
        h, mask = inputs
        m = tf.cast(mask, tf.bool)
        h = tf.where(m[:, :, None], h, tf.zeros_like(h))   # padded rows may be inf/NaN in fp16
        logits = tf.squeeze(self.w(self.v(h) * self.u(h)), axis=-1)
        logits = tf.where(m, logits, tf.cast(-1e4, logits.dtype) * tf.ones_like(logits))
        a = tf.nn.softmax(logits, axis=1) * tf.cast(m, logits.dtype)
        a = a / tf.maximum(tf.reduce_sum(a, axis=1, keepdims=True), tf.cast(1e-6, a.dtype))
        z = tf.reduce_sum(h * a[:, :, None], axis=1)
        return [z, tf.cast(a, "float32")]

    def compute_output_shape(self, input_shape):
        s = input_shape[0]
        return [(s[0], s[2]), (s[0], s[1])]

    def get_config(self):
        return {**super().get_config(), "units": self.units}


# ------------------------------------------------------------------------ model


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
    embed: int = 64,
    heads: int = 4,
    dropout: float = 0.2,
    branches: tuple[str, ...] = ALL_BRANCHES,
    edit_tokens_len: int = 12,
    edit_text_len: int = 24,
    token_buckets: int = 1 << 14,
    pooling: str = "mil",
) -> keras.Model:
    """The full model, or any subset of branches for the ablation grid.

    Every row of the ablation table is this function with a different `branches` tuple,
    so the configurations cannot drift apart the way parallel hand-written models do.

    The edit trunk is always present: attention pooling over edits is the fusion
    mechanism, and "B1 off" means the geometry columns are removed before they reach it,
    not that edits disappear. Column selection happens in `train/common/preprocess.py`.

    **Precondition:** every row must have at least one real edit in `edit_mask`. A fully
    masked row gives an all-false attention row, softmax over all -inf, and NaN - which
    poisons the whole batch. `train/common/preprocess.ensure_nonempty` enforces it.
    """
    branches = tuple(branches)
    unknown = set(branches) - set(ALL_BRANCHES)
    if unknown:
        raise ValueError(f"unknown branches {sorted(unknown)}; expected a subset of {ALL_BRANCHES}")

    inputs: dict = {}
    fusion: list = []

    edit_in = keras.Input(shape=(max_edits, edit_features), name="edit_features")
    mask_in = keras.Input(shape=(max_edits,), name="edit_mask")
    inputs["edit_features"] = edit_in
    inputs["edit_mask"] = mask_in
    edit_bool = ToBool(name="edit_mask_bool")(mask_in)

    per_edit = [edit_in]

    # --- B1 geometry raster (optional; scalar geometry already sits in edit_features)
    if "B1" in branches and raster:
        geom_in = keras.Input(shape=(max_edits, raster, raster, 2), name="geometry")
        inputs["geometry"] = geom_in
        g = layers.TimeDistributed(layers.Conv2D(8, 3, activation="relu", padding="same"))(geom_in)
        g = layers.TimeDistributed(layers.MaxPooling2D(2))(g)
        g = layers.TimeDistributed(layers.Conv2D(16, 3, activation="relu", padding="same"))(g)
        g = layers.TimeDistributed(layers.GlobalAveragePooling2D())(g)
        per_edit.append(g)

    # --- per-edit trunk. Named so Stage 1 pretrained weights load into it by name.
    x = layers.Concatenate(axis=-1)(per_edit) if len(per_edit) > 1 else per_edit[0]
    x = layers.TimeDistributed(layers.Dense(embed, activation="relu"), name="edit_encoder")(x)

    # --- B6 edit meaning: what each edit did (tokens) and the text it changed (old>new).
    # Joined *after* edit_encoder, so Stage 1 encoder weights still transfer unchanged.
    if "B6" in branches:
        tok_in = keras.Input(shape=(max_edits, edit_tokens_len), name="edit_tokens", dtype="int32")
        etx_in = keras.Input(shape=(max_edits, edit_text_len), name="edit_text", dtype="int32")
        inputs["edit_tokens"] = tok_in
        inputs["edit_text"] = etx_in
        tok = layers.Embedding(token_buckets, 32, name="edit_token_embed")(tok_in)
        tok = MaskedTokenMean(name="edit_token_mean")([tok, tok_in])
        ch = layers.Embedding(vocab_size, 16, name="edit_char_embed")(etx_in)
        ch = layers.TimeDistributed(layers.Conv1D(32, 3, activation="relu", padding="same"),
                                    name="edit_char_conv")(ch)
        ch = layers.TimeDistributed(layers.GlobalMaxPooling1D(), name="edit_char_pool")(ch)
        x = layers.Concatenate(axis=-1, name="edit_meaning")([x, tok, ch])
        x = layers.TimeDistributed(layers.Dense(embed, activation="relu"), name="edit_fuse")(x)
        x = layers.Dropout(dropout, name="edit_fuse_dropout")(x)

    # --- B3 edit sequence
    if "B3" in branches:
        x = layers.Bidirectional(
            layers.LSTM(embed // 2, return_sequences=True), name="b3_edit_sequence"
        )(x, mask=edit_bool)

    # --- attention fusion over edits
    attended, scores = layers.MultiHeadAttention(
        num_heads=heads, key_dim=max(8, embed // heads), name="edit_attention"
    )(x, x, attention_mask=PairwiseMask(name="edit_pair_mask")(mask_in),
      return_attention_scores=True)
    if pooling == "mil":
        z_att, a = GatedAttentionPool(32, name="edit_mil_attention")([attended, mask_in])
        z_max = MaskedMax(name="edit_max")([attended, mask_in])
        fusion.append(layers.Concatenate(name="edit_pool")([z_att, z_max]))
        edit_weight = layers.Activation("linear", dtype="float32", name="edit_attention_weight")(a)
    elif pooling == "mean":
        fusion.append(MaskedMeanPool(name="edit_pool")([attended, mask_in]))
        edit_weight = AttentionReceived(name="edit_attention_weight")([scores, mask_in])
    else:
        raise ValueError(f"pooling must be 'mil' or 'mean', got {pooling!r}")

    # --- B2 text
    if "B2" in branches:
        text_in = keras.Input(shape=(text_len,), name="text", dtype="int32")
        inputs["text"] = text_in
        e = layers.Embedding(vocab_size, 24, name="char_embed")(text_in)
        conv = [
            layers.GlobalMaxPooling1D()(
                layers.Conv1D(24, w, activation="relu", padding="same", name=f"char_conv{w}")(e)
            )
            for w in (3, 5, 7)
        ]
        fusion.append(layers.Concatenate(name="b2_text")(conv))

    # --- B4 user history
    if "B4" in branches:
        user_in = keras.Input(shape=(user_steps, user_features), name="user_sequence")
        umask_in = keras.Input(shape=(user_steps,), name="user_mask")
        inputs["user_sequence"] = user_in
        inputs["user_mask"] = umask_in
        u = layers.GRU(32, name="b4_user_history")(
            user_in, mask=ToBool(name="user_mask_bool")(umask_in))
        fusion.append(u)

    # --- B5 tabular
    if "B5" in branches:
        tab_in = keras.Input(shape=(tabular_features,), name="tabular")
        inputs["tabular"] = tab_in
        t = layers.Dense(64, activation="relu")(tab_in)
        t = layers.Dropout(dropout)(t)
        fusion.append(layers.Dense(32, activation="relu", name="b5_tabular")(t))

    fused = layers.Concatenate(name="fusion")(fusion) if len(fusion) > 1 else fusion[0]
    h = layers.Dense(64, activation="relu")(fused)
    h = layers.Dropout(dropout)(h)
    h = layers.LayerNormalization()(h)
    # float32 even under a mixed-precision policy: the loss and the score need full range
    out = layers.Dense(1, activation="sigmoid", name="p_damaging", dtype="float32")(h)

    return keras.Model(
        inputs=[inputs[k] for k in sorted(inputs)],
        outputs=[out, edit_weight],
        name="mapguard_stage2",
    )


def input_names(model: keras.Model) -> list[str]:
    """Input names in model order. Keras 2 names end in ':0', Keras 3 names do not."""
    return [t.name.split(":")[0] for t in model.inputs]


def pack(model: keras.Model, arrays: dict, rows=None) -> list:
    """Dict of arrays -> list in the model's input order, optionally row-selected.

    A list rather than a dict because dict inputs to a list-built functional model are
    matched differently by Keras 2 and Keras 3.
    """
    out = []
    for name in input_names(model):
        if name not in arrays:
            raise KeyError(f"model needs input '{name}' but it was not provided")
        a = np.asarray(arrays[name])
        out.append(a if rows is None else a[rows])
    return out


def scoring_model(model: keras.Model) -> keras.Model:
    """Single-output view sharing the same weights, used for fit().

    Training the two-output model would need a loss or a dummy target for the attention
    output, and the two Keras versions disagree on how an unlabelled output is handled.
    A single-output view avoids the question: gradients flow into exactly the same
    weights, and the validation metric is named `val_pr_auc` under both versions.
    """
    return keras.Model(model.inputs, model.outputs[0], name="mapguard_stage2_scoring")


def focal_loss(gamma: float = 2.0, alpha: float = 0.5):
    """Focal loss (Lin et al.). Compared against BCE in the ablation, not assumed to help.

    Returns one loss **per sample** (shape (batch,)), so Keras can apply `sample_weight`.
    Round 2 returned the batch mean (a scalar); Keras then multiplied that scalar by the mean
    weight of the batch (~1), so author weighting had no effect on the deep model (measured:
    weighted and unweighted runs had near-identical losses, 0.10702 vs 0.10709 at epoch 1).
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)), y_pred.dtype)
        p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1.0), p, 1 - p)
        w = tf.where(tf.equal(y_true, 1.0), alpha, 1 - alpha)
        return -tf.reduce_mean(w * tf.pow(1 - pt, gamma) * tf.math.log(pt), axis=-1)

    loss.__name__ = "focal_loss"
    return loss


def verify_masking(model: keras.Model, arrays: dict, n: int = 64) -> dict:
    """Assert padded edits cannot change the output. The Gate 1d check, on the real model.

    Run once per training run. Architectures change and branches get toggled for
    ablations, and a masking failure is silent every time.
    """
    rng = np.random.default_rng(0)
    base = {k: np.asarray(v)[:n] for k, v in arrays.items()}
    before = model.predict(pack(model, base), verbose=0)[0]

    pad = 1.0 - base["edit_mask"]
    perturbed = dict(base)
    # 50x noise in float32. Under mixed_float16 the same noise overflows padded rows to inf
    # (float16 max is 65504) inside layers that never see padding in real data, so use a
    # scale that is still far outside the data (features are standardised) but finite.
    scale = 3.0 if str(getattr(model, "compute_dtype", "float32")) == "float16" else 50.0
    for key in ("edit_features", "geometry"):
        if key in base:
            arr = base[key]
            shape = pad.shape + (1,) * (arr.ndim - pad.ndim)
            perturbed[key] = (arr + pad.reshape(shape) * rng.normal(size=arr.shape) * scale).astype(arr.dtype)
    for key, high in (("edit_tokens", 1 << 14), ("edit_text", 50)):
        if key in base:
            arr = base[key]
            noise = rng.integers(1, high, size=arr.shape).astype(arr.dtype)
            perturbed[key] = np.where(pad[..., None] > 0, noise, arr)

    after = model.predict(pack(model, perturbed), verbose=0)[0]
    delta = float(np.max(np.abs(before - after)))
    return {
        "max_delta": delta,
        "masking_holds": bool(delta < 1e-4),
        "rows_checked": int(len(before)),
        "noise_scale": scale,
        "note": "padded edits perturbed with noise; a correct model does not move",
    }
