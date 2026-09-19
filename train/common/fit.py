"""The Stage 2 training core, shared by every stage that fits the classifier.

Supervised training (02), the ablation grid (03), label efficiency (05) and temporal
evaluation (06) all fit the same model. If each had its own loop they would drift -
different early stopping, a different threshold rule, a different normaliser - and the
ablation table would compare training procedures instead of branches. So there is one.

Order of operations, and why:

 1. fit the normaliser on training rows only (or load Stage 1's edit statistics, because
    pretrained encoder weights are only meaningful on the inputs they were trained on)
 2. build the model and, if configured, load the pretrained edit encoder
 3. **check masking before training** - a broken mask should cost seconds, not a run
 4. train a single-output view with early stopping on validation PR-AUC
 5. **check masking again after training**, on real rows
 6. choose the threshold on validation, report on test, measure calibration
 7. occlusion attribution on test
 8. save weights, model config, normaliser and predictions into the run directory

Weights are saved, not whole models. Custom layers and loss functions serialise
differently between Keras 2 and Keras 3; rebuilding from `model_config.json` and loading
weights works on both.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import scoring
from .attribution import branch_attribution, top_examples
from .preprocess import Normaliser
from .run import keras_version

ENCODER_FILE = "edit_encoder.npz"

GPU_BATCH = 256     # fills a GPU step; the train sets are 5k-18k rows, so larger stops helping
CPU_BATCH = 128


def resolve_batch_size(value) -> int:
    """`auto` (the default) picks GPU_BATCH when a GPU is visible, else CPU_BATCH."""
    if value not in (None, "auto"):
        return int(value)
    import tensorflow as tf

    return GPU_BATCH if tf.config.list_physical_devices("GPU") else CPU_BATCH


def model_kwargs(prep, cfg: dict) -> dict:
    m = cfg.get("model", {})
    s = prep.shapes()
    return {
        **s,
        "raster": m.get("raster"),
        "embed": int(m.get("embed", 64)),
        "heads": int(m.get("heads", 4)),
        "dropout": float(m.get("dropout", 0.2)),
        "branches": list(m.get("branches", ["B1", "B2", "B3", "B4", "B5"])),
    }


def build(kwargs: dict):
    from mapguard.models.stage2 import build_model

    kw = dict(kwargs)
    kw["branches"] = tuple(kw["branches"])
    return build_model(**kw)


def load_pretrained_encoder(model, encoder_dir: Path, prep, embed: int) -> dict:
    """Load Stage 1 edit-encoder weights into the `edit_encoder` layer.

    Refuses on any mismatch. A pretrained encoder fed differently-ordered or differently
    scaled columns is not pretrained - it is a random initialisation with a misleading name,
    and R4 would report a comparison that never happened.
    """
    blob = np.load(encoder_dir / ENCODER_FILE, allow_pickle=False)
    cols = [str(c) for c in blob["edit_columns"]]
    if cols != list(prep.edit_columns):
        raise SystemExit(
            "pretrained encoder was trained on different edit columns:\n"
            f"  encoder : {cols}\n  model   : {list(prep.edit_columns)}"
        )
    kernel, bias = blob["kernel"], blob["bias"]
    if kernel.shape != (len(cols), embed):
        raise SystemExit(f"encoder kernel {kernel.shape} does not match embed={embed}")
    layer = model.get_layer("edit_encoder")
    layer.set_weights([kernel, bias])
    return {"loaded_from": str(encoder_dir), "kernel_shape": list(kernel.shape)}


def check_encoder_split(enc_dir: Path, cfg: dict) -> None:
    """The encoder must come from pretraining on the SAME frozen split.

    Stage 1 excludes the validation and test folds of the split it was run on. An encoder
    pretrained under a different task, split or split seed was free to see this run's test
    rows, and the resulting "pretrained" number would carry that leak.
    """
    enc_cfg = json.loads((enc_dir / "config.json").read_text(encoding="utf-8"))
    mine = (cfg.get("task"), cfg.get("split"), int(cfg.get("split_seed", cfg.get("seed", 0))))
    theirs = (enc_cfg.get("task"), enc_cfg.get("split"), int(enc_cfg.get("seed", 0)))
    if mine != theirs:
        raise SystemExit(
            f"pretrained encoder was built for (task, split, seed) = {theirs} but this run "
            f"uses {mine}. Its pretraining could include this run's test rows. Pretrain on "
            "the same split, or train from scratch."
        )


def train_stage2(cfg: dict, prep_raw, masks: dict, run, *, train_rows=None) -> dict:
    import tensorflow as tf
    from tensorflow import keras

    from .run import configure_precision

    precision = configure_precision(cfg.get("train", {}).get("mixed_precision"))
    batch_size = resolve_batch_size(cfg.get("train", {}).get("batch_size"))

    from mapguard.models.stage2 import focal_loss, pack, scoring_model, verify_masking

    t_cfg = cfg.get("train", {})
    init = cfg.get("init", {}) or {}
    train_rows = masks["train"] if train_rows is None else train_rows
    val_rows, test_rows = masks["val"], masks["test"]

    # 1. normaliser -------------------------------------------------------------
    enc_dir = Path(init["pretrained_encoder"]) if init.get("pretrained_encoder") else None
    if enc_dir is not None:
        check_encoder_split(enc_dir, cfg)
    norm = Normaliser()
    if enc_dir is not None:
        norm = Normaliser.load(enc_dir / "normaliser.json", groups=("edit",))
        norm.fit(prep_raw, train_rows, groups=("user", "tabular"))
    else:
        norm.fit(prep_raw, train_rows)
    prep = norm.apply(prep_raw)
    norm.save(run.path("normaliser.json"))
    a, y = prep.arrays, prep.y

    # 2. model --------------------------------------------------------------------
    kwargs = model_kwargs(prep, cfg)
    if enc_dir is not None and kwargs["raster"]:
        raise SystemExit("pretrained encoder transfer needs raster=None - the raster CNN "
                         "changes the encoder's input width")
    model = build(kwargs)
    encoder_info = None
    if enc_dir is not None:
        encoder_info = load_pretrained_encoder(model, enc_dir, prep, kwargs["embed"])
        if init.get("freeze_encoder"):
            model.get_layer("edit_encoder").trainable = False

    run.write_json("model_config.json", {
        "build_kwargs": kwargs,
        "edit_columns": list(prep.edit_columns),
        "tabular_columns": list(prep.tabular_columns),
        "pretrained_encoder": encoder_info,
    })

    # 3. masking before -------------------------------------------------------------
    pre_check = verify_masking(model, {k: v[val_rows] for k, v in a.items()})
    if not pre_check["masking_holds"]:
        raise SystemExit(f"masking check failed before training: {pre_check}")

    # 4. train ----------------------------------------------------------------------
    loss_name = t_cfg.get("loss", "focal")
    loss = (focal_loss(float(t_cfg.get("focal_gamma", 2.0)), float(t_cfg.get("focal_alpha", 0.5)))
            if loss_name == "focal" else keras.losses.BinaryCrossentropy())
    scorer = scoring_model(model)
    scorer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(t_cfg.get("lr", 1e-3)),
                                        clipnorm=float(t_cfg.get("clipnorm", 1.0))),
        loss=loss,
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"),
                 keras.metrics.AUC(curve="ROC", name="roc_auc")],
    )

    class_weight = None
    if t_cfg.get("class_weight") == "balanced":
        ytr = y[train_rows]
        pos = max(1.0, float(ytr.sum()))
        neg = max(1.0, float(len(ytr) - ytr.sum()))
        class_weight = {0: len(ytr) / (2 * neg), 1: len(ytr) / (2 * pos)}

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(run.path("best.weights.h5")), monitor="val_pr_auc", mode="max",
            save_best_only=True, save_weights_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=int(t_cfg.get("patience", 5)),
            restore_best_weights=True),
        keras.callbacks.CSVLogger(str(run.path("training_log.csv"))),
    ]

    history = scorer.fit(
        pack(model, a, train_rows), y[train_rows],
        validation_data=(pack(model, a, val_rows), y[val_rows]),
        epochs=int(t_cfg.get("epochs", 40)),
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=int(t_cfg.get("verbose", 2)),
    )
    # Keras 2 restores the best weights only when EarlyStopping actually fires; a run that
    # uses every epoch would keep the last epoch. Reload the checkpoint explicitly so both
    # versions evaluate the epoch that was selected on validation.
    if run.path("best.weights.h5").exists():
        scorer.load_weights(str(run.path("best.weights.h5")))
    model.save_weights(str(run.path("model.weights.h5")))

    # 5. masking after --------------------------------------------------------------
    post_check = verify_masking(model, {k: v[test_rows] for k, v in a.items()})
    if not post_check["masking_holds"]:
        raise SystemExit(f"masking check failed after training: {post_check}")

    # 6. evaluate -------------------------------------------------------------------
    bs = batch_size * 4
    p_val, _ = model.predict(pack(model, a, val_rows), batch_size=bs, verbose=0)
    p_test, w_test = model.predict(pack(model, a, test_rows), batch_size=bs, verbose=0)
    rep = scoring.report(y[val_rows], p_val, y[test_rows], p_test, seed=int(cfg.get("seed", 0)))

    np.savez_compressed(
        run.path("predictions.npz"),
        val_cids=prep.cids[val_rows], val_y=y[val_rows], val_p=p_val.ravel(),
        test_cids=prep.cids[test_rows], test_y=y[test_rows], test_p=p_test.ravel(),
    )

    # 7. attribution ----------------------------------------------------------------
    attribution, top = None, None
    if cfg.get("evaluate", {}).get("attribution", True):
        test_arrays = {k: v[test_rows] for k, v in a.items()}
        attribution, per_row, _ = branch_attribution(
            model, test_arrays, branches=kwargs["branches"],
            edit_columns=prep.edit_columns, tabular_columns=prep.tabular_columns,
            y=y[test_rows], batch_size=bs, seed=int(cfg.get("seed", 0)))
        top = top_examples(prep.cids[test_rows], y[test_rows], p_test, w_test,
                           test_arrays["edit_mask"], per_row,
                           k=int(cfg.get("evaluate", {}).get("top_k", 20)))

    hist = history.history
    best_epoch = int(np.argmax(hist["val_pr_auc"])) + 1 if hist.get("val_pr_auc") else None
    return {
        "headline": scoring.headline(rep),
        "evaluation": rep,
        "attribution": attribution,
        "top_test_examples": top,
        "masking_checks": {"before_training": pre_check, "after_training": post_check},
        "training": {
            "epochs_run": len(hist.get("loss", [])),
            "best_epoch": best_epoch,
            "best_val_pr_auc": round(float(max(hist["val_pr_auc"])), 4) if hist.get("val_pr_auc") else None,
            "n_train": int(np.asarray(train_rows).sum()),
            "n_val": int(val_rows.sum()), "n_test": int(test_rows.sum()),
            "class_weight": class_weight, "loss": loss_name,
            "params": int(model.count_params()),
            "tensorflow": tf.__version__, "keras": keras_version(), "precision": precision,
            "batch_size": batch_size,
        },
        "branches": kwargs["branches"],
        "pretrained_encoder": encoder_info,
        "notes": prep.notes,
    }
