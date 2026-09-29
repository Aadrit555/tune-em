"""Frozen-backbone embedding extraction (premise + option texts -> CLS vectors).

Writes one .npz per split with L2-normalized vectors. Extract TEST only at
final scoring time; train/dev are used for fitting and early stopping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adapters import describe_item, score_levels


def option_texts(item: dict, qtype: str, labels: list[str]) -> list[str]:
    q = item["question"]
    if qtype == "choice":
        crit = q.get("criteria", {}) or {}
        return [f"{lab}: {crit.get(lab, lab)}" for lab in labels]
    if qtype == "noul":
        # Dataset label order (usually ['no', 'yes']); label_index follows it.
        from adapters import noul_option_texts

        _labels, texts = noul_option_texts(item)
        assert _labels == labels, f"label order drift for {item.get('id')}"
        return texts
    if qtype == "score":
        levels = score_levels(item)
        return [f"{lab}: {desc}" for lab, desc in zip(labels, levels)]
    raise ValueError(qtype)


@torch.no_grad()
def encode(texts: list[str], tok, model, batch_size: int = 8, max_length: int = 512,
           pool: str = "mean") -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt",
                  padding=True, truncation=True, max_length=max_length)
        out = model(**enc)
        h = out.last_hidden_state.float()
        if pool == "mean":
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        else:
            v = h[:, 0, :]
        v = v.cpu().numpy()
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        vecs.append(v)
    return np.concatenate(vecs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-large")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModel.from_pretrained(args.backbone, dtype="float32")
    model.eval()

    items: dict[str, dict] = {}
    data_dir = Path(args.data_dir)
    for f in ("easy", "original", "hard"):
        for line in open(data_dir / f"{f}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            d["_tier"] = f
            items[d["id"]] = d
    wanted = json.load(open(args.splits, encoding="utf-8"))["splits"][args.split]

    premises, opt_lists, label_idx, n_opts, ids, tiers, qtypes, expected = [], [], [], [], [], [], [], []
    for _id in wanted:
        item = items[_id]
        prompt, labels, qtype, exp = describe_item(item)
        # Premise: instructions + state (NOT the options block; options encoded separately).
        q = item["question"]
        premise = f"{q.get('instructions', '')}\n{item.get('state', '')}"
        premises.append(premise)
        otexts = option_texts(item, qtype, labels)
        opt_lists.append(otexts)
        label_idx.append(labels.index(exp))
        n_opts.append(len(labels))
        ids.append(_id)
        tiers.append(item["_tier"])
        qtypes.append(qtype)
        expected.append(exp)

    ctx = encode(premises, tok, model)
    # Joint premise+option encoding (option-marker style): the pair CLS carries
    # the match signal; premise CLS stays the context representation.
    pair_texts = []
    for premise, opts in zip(premises, opt_lists):
        for otext in opts:
            pair_texts.append(f"{premise}\nCandidate: {otext}")
    flat_enc = encode(pair_texts, tok, model)
    cmax = max(n_opts)
    hidden = ctx.shape[1]
    opts = np.zeros((len(ids), cmax, hidden), dtype=np.float32)
    mask = np.zeros((len(ids), cmax), dtype=np.float32)
    pos = 0
    for i, n in enumerate(n_opts):
        opts[i, :n] = flat_enc[pos:pos + n]
        mask[i, :n] = 1.0
        pos += n

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(ids), tiers=np.array(tiers),
                        qtypes=np.array(qtypes), expected=np.array(expected),
                        context=ctx, options=opts, mask=mask,
                        label_index=np.array(label_idx, dtype=np.int64),
                        backbone=np.array([args.backbone]))
    print(f"wrote {out}: n={len(ids)} hidden={hidden} cmax={cmax}")


if __name__ == "__main__":
    main()
