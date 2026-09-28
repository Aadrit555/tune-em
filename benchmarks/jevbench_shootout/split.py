"""Build locked train/dev/test splits of JevBench public items (stratified by file)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FILES = ("easy", "original", "hard")


def load_items(data_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in FILES:
        path = data_dir / f"{f}.jsonl"
        out[f] = [json.loads(line) for line in open(path, encoding="utf-8")]
    return out


def build_splits(items: dict[str, list[dict]], seed: int = 0) -> dict[str, list[str]]:
    rng = np.random.RandomState(seed)
    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for f in FILES:
        ids = [d["id"] for d in items[f]]
        order = rng.permutation(len(ids))
        n = len(ids)
        n_test = max(1, int(round(0.20 * n)))
        n_dev = max(1, int(round(0.20 * n)))
        test_idx = set(order[:n_test].tolist())
        dev_idx = set(order[n_test:n_test + n_dev].tolist())
        for i, _id in enumerate(ids):
            if i in test_idx:
                splits["test"].append(_id)
            elif i in dev_idx:
                splits["dev"].append(_id)
            else:
                splits["train"].append(_id)
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    items = load_items(Path(args.data_dir))
    splits = build_splits(items, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"seed": args.seed, "counts": {k: len(v) for k, v in splits.items()}, "splits": splits},
        indent=2,
    ), encoding="utf-8")
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
