"""Build auxiliary training pool from ARC (open, real labels) + JevBench train.

ARC items are science multiple-choice; format matches our embedding layout:
premise = question stem, options = 'label: text'. Output .npz identical in
schema to extract_embeddings.py output (ids/tiers/qtypes/expected/context/
options/mask/label_index). JevBench TEST is never included.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def read_parquet(repo: str, path: str) -> list[dict]:
    import pyarrow.parquet as pq

    local = hf_hub_download(repo, path, repo_type="dataset")
    table = pq.read_table(local)
    return table.to_pylist()


@torch.no_grad()
def encode(texts: list[str], tok, model, batch_size: int = 16) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt",
                  padding=True, truncation=True, max_length=256)
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        cls /= (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12)
        vecs.append(cls)
        if (i // batch_size) % 20 == 0:
            print(f"  encoded {i}/{len(texts)}", flush=True)
    return np.concatenate(vecs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-large")
    ap.add_argument("--max-arc", type=int, default=3000)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    premises: list[str] = []
    opt_lists: list[list[str]] = []
    label_idx: list[int] = []
    ids: list[str] = []

    # 1. ARC-Easy + ARC-Challenge train (real labels, open licence for eval use).
    n_arc = 0
    for subset, path in (("easy", "ARC-Easy/train-00000-of-00001.parquet"),
                         ("challenge", "ARC-Challenge/train-00000-of-00001.parquet")):
        for row in read_parquet("allenai/ai2_arc", path):
            if n_arc >= args.max_arc:
                break
            stem = row["question"]
            labels = list(row["choices"]["label"])
            texts = list(row["choices"]["text"])
            ans = row["answerKey"]
            if ans not in labels:
                continue
            premises.append(stem)
            opt_lists.append([f"{l}: {t}" for l, t in zip(labels, texts)])
            label_idx.append(labels.index(ans))
            ids.append(f"arc-{subset}-{row['id']}")
            n_arc += 1
    print(f"ARC items: {n_arc}")

    # 2. JevBench TRAIN split only (test/dev never included).
    items: dict[str, dict] = {}
    data_dir = Path(args.data_dir)
    for f in ("easy", "original", "hard"):
        for line in open(data_dir / f"{f}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            items[d["id"]] = d
    train_ids = json.load(open(args.splits, encoding="utf-8"))["splits"]["train"]
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from adapters import describe_item, score_levels

    n_jb = 0
    for _id in train_ids:
        item = items[_id]
        prompt, labels, qtype, exp = describe_item(item)
        q = item["question"]
        premise = f"{q.get('instructions', '')}\n{item.get('state', '')}"
        if qtype == "choice":
            crit = q.get("criteria", {}) or {}
            otexts = [f"{lab}: {crit.get(lab, lab)}" for lab in labels]
        elif qtype == "noul":
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from adapters import noul_option_texts
            labels, otexts = noul_option_texts(item)
        else:
            levels = score_levels(item)
            otexts = [f"{lab}: {desc}" for lab, desc in zip(labels, levels)]
        premises.append(premise)
        opt_lists.append(otexts)
        label_idx.append(labels.index(exp))
        ids.append(_id)
        n_jb += 1
    print(f"JB-train items: {n_jb}")

    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModel.from_pretrained(args.backbone, dtype="float32")
    model.eval()
    ctx = encode(premises, tok, model)
    flat = [t for sub in opt_lists for t in sub]
    flat_enc = encode(flat, tok, model)
    n_opts = [len(o) for o in opt_lists]
    cmax, hidden = max(n_opts), ctx.shape[1]
    opts = np.zeros((len(ids), cmax, hidden), dtype=np.float32)
    mask = np.zeros((len(ids), cmax), dtype=np.float32)
    pos = 0
    for i, n in enumerate(n_opts):
        opts[i, :n] = flat_enc[pos:pos + n]
        mask[i, :n] = 1.0
        pos += n

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, ids=np.array(ids),
        tiers=np.array(["aux"] * n_arc + ["jb-train"] * n_jb),
        qtypes=np.array(["choice"] * n_arc + ["mixed"] * n_jb),
        expected=np.array([""] * len(ids)),
        context=ctx, options=opts, mask=mask,
        label_index=np.array(label_idx, dtype=np.int64),
        backbone=np.array([args.backbone]),
    )
    print(f"wrote {out}: n={len(ids)}")


if __name__ == "__main__":
    main()
