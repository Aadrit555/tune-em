"""Entrant ours-head: trained bilinear head on frozen embeddings -> artifact."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anydecision.models.decision_head import NonAutoregressiveDecisionHead


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    head = NonAutoregressiveDecisionHead(
        hidden_dim=int(ckpt["hidden_dim"]),
        projection_dim=int(ckpt.get("projection_dim", 256)))
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    do_center = bool(ckpt.get("center", False))
    ctx_mean = np.asarray(ckpt.get("ctx_mean", 0.0), dtype=np.float32)
    opt_mean = np.asarray(ckpt.get("opt_mean", 0.0), dtype=np.float32)

    z = np.load(args.npz)
    ids = [str(i) for i in z["ids"]]
    tiers = [str(i) for i in z["tiers"]]
    qtypes = [str(i) for i in z["qtypes"]]
    expected_raw = [str(i) for i in z["expected"]]
    label_index = [int(i) for i in z["label_index"]]
    mask = z["mask"]
    id2pos = {i: p for p, i in enumerate(ids)}
    wanted = json.load(open(args.splits, encoding="utf-8"))["splits"][args.split]

    # Labels/expected resolved from source data (npz 'expected' is blank for aux).
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from adapters import describe_item
    _items: dict[str, dict] = {}
    _ddir = Path(args.data_dir)
    for _f in ("easy", "original", "hard"):
        for _line in open(_ddir / f"{_f}.jsonl", encoding="utf-8"):
            _d = json.loads(_line)
            _d["_tier"] = _f
            _items[_d["id"]] = _d
    records = []
    import time
    import torch.nn.functional as F
    with torch.no_grad():
        for _id in wanted:
            if _id not in id2pos or _id not in _items:
                continue
            p = id2pos[_id]
            _prompt, _labels, _qtype, _exp = describe_item(_items[_id])
            n_opts = int(mask[p].sum())
            assert n_opts == len(_labels), f"option order drift for {_id}"
            ctx = torch.from_numpy(z["context"][p].astype(np.float32))
            opts = torch.from_numpy(z["options"][p, :n_opts].astype(np.float32))
            if do_center:
                ctx = ctx - torch.from_numpy(ctx_mean)
                opts = opts - torch.from_numpy(opt_mean)
            t0 = time.perf_counter()
            logits = head(ctx.unsqueeze(0), opts)[0]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            pred_idx = int(np.argmax(probs))
            records.append({
                "id": _id, "tier": _items[_id]["_tier"], "qtype": _qtype,
                "expected": _exp, "pred": _labels[pred_idx],
                "probs": {lab: float(pr) for lab, pr in zip(_labels, probs)},
                "latency_ms": lat_ms, "error": None,
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "jevbench_shootout",
        "entrant": "ours-head",
        "entrant_detail": {
            "head": str(args.head), "backbone": str(ckpt.get("backbone")),
            "center": do_center, "objective": "CE+0.5*Brier",
        },
        "split": args.split, "seed": 0,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": git_commit(),
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
