"""Entrant ours-lm: anydecision L0 zero-shot + real causal LM on a split."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

from adapters import describe_item, expected_to_binary


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


PROMPT_TEMPLATE = "v1:instructions+state[+options-block]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()

    from anydecision import DecisionEngine, Question

    engine = DecisionEngine(model=args.model)
    try:
        model_rev = str(engine.metadata.model_revision)
    except Exception:
        model_rev = "main"

    items: dict[str, dict] = {}
    data_dir = Path(args.data_dir)
    for f in ("easy", "original", "hard"):
        for line in open(data_dir / f"{f}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            d["_tier"] = f
            items[d["id"]] = d
    wanted = json.load(open(args.splits, encoding="utf-8"))["splits"][args.split]
    if args.max_items is not None:
        wanted = wanted[:args.max_items]

    records = []
    for _id in wanted:
        item = items[_id]
        prompt, labels, qtype, expected = describe_item(item)
        t0 = time.perf_counter()
        try:
            if qtype == "choice":
                q = Question.choice(prompt, choices=labels)
                dec = engine.decide(q, level="L0")
                pred = str(dec.answer)
            elif qtype == "noul":
                q = Question.binary(prompt)
                dec = engine.decide(q, level="L0")
                pred = expected_to_binary(str(dec.answer))
            elif qtype == "score":
                q = Question.ordinal(prompt, levels=labels)
                dec = engine.decide(q, level="L0")
                pred = str(dec.answer)
            else:
                raise ValueError(f"unknown qtype {qtype}")
            probs = {str(k): float(v) for k, v in dec.probabilities.items()}
            exp_val = dec.expected_value
            error = None
        except Exception as e:
            pred, probs, exp_val, error = "__error__", {}, None, f"{type(e).__name__}: {e}"
        lat_ms = (time.perf_counter() - t0) * 1000.0
        records.append({
            "id": _id, "tier": item["_tier"], "qtype": qtype,
            "expected": expected, "pred": pred, "probs": probs,
            "expected_value": exp_val, "latency_ms": lat_ms, "error": error,
        })
        print(f"{_id}: pred={pred} expected={expected} ({lat_ms:.0f} ms)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "jevbench_shootout",
        "entrant": "ours-lm",
        "entrant_detail": {
            "model": args.model, "model_revision": model_rev,
            "backend": "transformers", "level": "L0",
            "prompt_template": PROMPT_TEMPLATE,
        },
        "split": args.split, "seed": 0,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": git_commit(),
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
