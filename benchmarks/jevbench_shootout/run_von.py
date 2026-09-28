"""Entrant von-1.3: native von-sdk on a split. Usage: run_von.py --split test ..."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

from adapters import describe_item, expected_to_binary, score_levels, von_choices


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()

    import von
    import von as von_pkg

    try:
        sdk_version = getattr(von_pkg, "__version__", "unknown")
    except Exception:
        sdk_version = "unknown"

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
                ans = von.decide(
                    state=item["state"],
                    choices=von_choices(item),
                    instructions=item["question"].get("instructions", ""),
                )
                pred, probs = str(ans.choice), {str(k): float(v) for k, v in ans.probabilities.items()}
            elif qtype == "noul":
                p_true = float(von.judge(
                    state=item["state"],
                    instructions=item["question"].get("instructions", ""),
                ))
                pred = "true" if p_true >= 0.5 else "false"
                probs = {"true": p_true, "false": 1.0 - p_true}
            elif qtype == "score":
                ans = von.rate(
                    state=item["state"],
                    criteria=score_levels(item),
                    instructions=item["question"].get("instructions", ""),
                )
                probs = {str(k): float(v) for k, v in ans.probabilities.items()}
                pred = max(probs, key=probs.get)  # type: ignore[arg-type]
            else:
                raise ValueError(f"unknown qtype {qtype}")
            error = None
        except Exception as e:  # never invent a prediction on failure
            pred, probs, error = "__error__", {}, f"{type(e).__name__}: {e}"
        lat_ms = (time.perf_counter() - t0) * 1000.0
        records.append({
            "id": _id, "tier": item["_tier"], "qtype": qtype,
            "expected": expected, "pred": pred, "probs": probs,
            "latency_ms": lat_ms, "error": error,
        })
        print(f"{_id}: pred={pred} expected={expected} ({lat_ms:.0f} ms)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "jevbench_shootout",
        "entrant": "von-1.3",
        "entrant_detail": {"sdk": f"von-sdk {sdk_version}", "chains": "default(on)", "weights": "wfzyx/von"},
        "split": args.split, "seed": 0,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": git_commit(),
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
