"""[TEXT EMBEDDINGS · v3 prep] Pretrained multilingual sentence embeddings of each changeset's text.

Run order 8 - before 01 and 02, which use the result when it exists. GPU if available.

    python train/08_text_embeddings/embed_text.py --feature-store data/interim

Why. Text is one of the two branches that pay (ablations, rounds 2-2c), and OSM text is in
every language. The char-CNN learns from 18k labelled changesets; a pretrained multilingual
encoder has read billions of sentences, so "test", "asdf", insults or spam in Hindi or Russian
already land somewhere meaningful. The encoder is frozen: embeddings are computed once and fed
to the hybrid (out-of-fold logistic score) and to the deep model (input `text_emb`).

Two texts per changeset, embedded separately:
    comment   the changeset comment (first 256 characters)
    tags      the edited tag text (`text_blob`, first 512 characters)

Writes <feature-store>/text_embeddings.npz: cids, comment (N, d) float16, tags (N, d) float16,
and text_embeddings.json (model, dimension, versions, timing). Nothing is trained here, no
labels are read, so there is no leakage path: every row is embedded the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")               # keep transformers away from the image's TF
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feature-store", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    store = Path(args.feature_store)
    # a --limit run writes beside the real file, never over it
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    out = store / f"text_embeddings{suffix}.npz"
    meta_path = store / f"text_embeddings{suffix}.json"
    if out.exists() and meta_path.exists() and args.limit is None:
        meta = json.loads(meta_path.read_text())
        n_store = pd.read_parquet(store / "changesets.parquet", columns=["cid"]).shape[0]
        if meta.get("model") == args.model and meta.get("limit") is None and meta.get("rows") == n_store:
            print(f"already embedded with {args.model}: {out}")
            return 0
        print("existing embeddings are stale (model or row count differs) - recomputing")

    import torch
    from sentence_transformers import SentenceTransformer

    cs = pd.read_parquet(store / "changesets.parquet", columns=["cid", "comment", "text_blob"])
    if args.limit:
        cs = cs.head(args.limit)
    comments = cs["comment"].fillna("").astype(str).str.slice(0, 256).tolist()
    tags = cs["text_blob"].fillna("").astype(str).str.slice(0, 512).tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(cs):,} changesets | model {args.model} | device {device}", flush=True)

    t0 = time.time()
    model = SentenceTransformer(args.model, device=device)
    enc = lambda xs: model.encode(xs, batch_size=args.batch, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)
    e_comment = enc(comments).astype("float16")
    print(f"comments embedded in {time.time() - t0:.0f}s", flush=True)
    e_tags = enc(tags).astype("float16")
    secs = time.time() - t0

    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez_compressed(tmp, cids=cs["cid"].to_numpy(dtype="int64"), comment=e_comment, tags=e_tags)
    os.replace(tmp, out)                 # a killed job never leaves a truncated file in place
    meta = {"model": args.model, "dim": int(e_comment.shape[1]), "rows": int(len(cs)),
            "device": device, "seconds": round(secs, 1),
            "empty_comments": int(sum(1 for c in comments if not c.strip())),
            "empty_tags": int(sum(1 for c in tags if not c.strip())),
            "torch": torch.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
            "limit": args.limit}
    meta_tmp = meta_path.with_suffix(".partial")
    meta_tmp.write_text(json.dumps(meta, indent=1))
    os.replace(meta_tmp, meta_path)      # meta last: its presence marks the npz complete
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
