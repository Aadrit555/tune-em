"""Score entrant artifacts: accuracy, ECE, Brier, latency, McNemar + MDE.

Usage: score.py --ref von_dev.json --cmp ours_lm_dev.json [--out scored.json]
Both artifacts share the record schema {id, tier, qtype, expected, pred,
probs, latency_ms, error}. Below-MDE or p>=0.05 deltas are UNRESOLVABLE.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_records(path: str) -> dict[str, dict]:
    d = json.load(open(path, encoding="utf-8"))
    recs = d.get("records", [])
    return {r["id"]: r for r in recs}, d


def accuracy(recs: dict[str, dict], ids: list[str]) -> tuple[float, int]:
    hits = sum(1 for i in ids if recs.get(i, {}).get("pred") == recs.get(i, {}).get("expected")
               and not recs.get(i, {}).get("error"))
    return hits / max(1, len(ids)), len(ids)


def ece_brier(recs: dict[str, dict], ids: list[str]) -> tuple[float, float, int]:
    n_bins, Taco = 10, 0
    bin_c = [0] * n_bins
    bin_a = [0.0] * n_bins
    bin_p = [0.0] * n_bins
    brier, n = 0.0, 0
    for i in ids:
        r = recs.get(i)
        if not r or r.get("error") or not r.get("probs"):
            continue
        probs = {str(k): float(v) for k, v in r["probs"].items()}
        exp = str(r["expected"])
        top = max(probs, key=probs.get)  # type: ignore[arg-type]
        conf = probs[top]
        correct = 1.0 if top == exp else 0.0
        b = min(n_bins - 1, int(conf * n_bins))
        bin_c[b] += 1
        bin_a[b] += correct
        bin_p[b] += conf
        oh = {k: 1.0 if k == exp else 0.0 for k in probs}
        brier += sum((probs[k] - oh[k]) ** 2 for k in probs)
        n += 1
    ece = sum((bin_c[b] / n) * abs(bin_a[b] / bin_c[b] - bin_p[b] / bin_c[b])
              for b in range(n_bins) if bin_c[b] and n) if n else 0.0
    return float(ece), float(brier / n) if n else 0.0, n


def mean_latency(recs: dict[str, dict], ids: list[str]) -> float:
    vals = [float(recs[i]["latency_ms"]) for i in ids if i in recs and not recs[i].get("error")]
    return float(sum(vals) / len(vals)) if vals else 0.0


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for discordant counts (ref-only=b, cmp-only=c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def mde(n: int, discordant: int, alpha: float = 0.05) -> float | None:
    """Smallest accuracy delta significant at alpha given discordant count.

    Normal approximation on the discordant-pair sign test: the critical
    asymmetry is z*sqrt(n_disc)/2 correct-vote margin, i.e. delta of
    z*sqrt(n_disc)/n_total.
    """
    if discordant == 0 or n == 0:
        return None
    z = 1.96 if alpha == 0.05 else 2.576
    return float(z * math.sqrt(discordant) / n)


def compare(ref_recs: dict[str, dict], cmp_recs: dict[str, dict]) -> dict:
    ids = sorted(set(ref_recs) & set(cmp_recs))
    ref_ok = {i: (ref_recs[i].get("pred") == ref_recs[i].get("expected") and not ref_recs[i].get("error")) for i in ids}
    cmp_ok = {i: (cmp_recs[i].get("pred") == cmp_recs[i].get("expected") and not cmp_recs[i].get("error")) for i in ids}
    b = sum(1 for i in ids if ref_ok[i] and not cmp_ok[i])
    c = sum(1 for i in ids if cmp_ok[i] and not ref_ok[i])
    p = mcnemar_exact(b, c)
    acc_r, _ = accuracy(ref_recs, ids)
    acc_c, _ = accuracy(cmp_recs, ids)
    delta = acc_c - acc_r
    return {
        "n": len(ids), "ref_acc": acc_r, "cmp_acc": acc_c, "delta": delta,
        "discordant": {"ref_only": b, "cmp_only": c},
        "mcnemar_p": p, "mde": mde(len(ids), b + c),
        "verdict": ("cmp wins" if (p < 0.05 and delta > 0) else
                    ("ref wins" if (p < 0.05 and delta < 0) else "UNRESOLVABLE")),
    }


def summarize(path: str) -> dict:
    recs, meta = load_records(path)
    ids = sorted(recs)
    tiers: dict[str, list[str]] = {}
    for i in ids:
        tiers.setdefault(str(recs[i].get("tier", "?")), []).append(i)
    acc, _ = accuracy(recs, ids)
    ece, brier, n_scored = ece_brier(recs, ids)
    return {
        "entrant": meta.get("entrant"), "split": meta.get("split"),
        "accuracy": acc, "ece": ece, "brier": brier, "n_scored": n_scored,
        "mean_latency_ms": mean_latency(recs, ids),
        "by_tier": {t: {"accuracy": accuracy(recs, v)[0], "n": len(v)} for t, v in tiers.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref-name", default="ref")
    ap.add_argument("--cmp-name", default="cmp")
    args = ap.parse_args()
    ref_recs, _ = load_records(args.ref)
    cmp_recs, _ = load_records(args.cmp)
    out = {
        args.ref_name: summarize(args.ref),
        args.cmp_name: summarize(args.cmp),
        "head_to_head": compare(ref_recs, cmp_recs),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
