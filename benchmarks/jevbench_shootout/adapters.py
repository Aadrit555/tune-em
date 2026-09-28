"""Shared item mapping: JevBench item -> von call / anydecision Question.

Single source of truth so both entrants see identical state/instructions/labels.
"""

from __future__ import annotations

from typing import Any


def describe_item(item: dict[str, Any]) -> tuple[str, list[str], str, str]:
    """Return (prompt_text, labels, qtype, expected)."""
    q = item["question"]
    labels = list(item["labels"])
    instructions = q.get("instructions", "")
    state = item.get("state", "")
    qtype = q.get("type", "choice")
    if qtype == "choice":
        crit = q.get("criteria", {}) or {}
        lines = [f"- {lab}: {crit.get(lab, lab)}" for lab in labels]
        prompt = f"{instructions}\n{state}\nOptions:\n" + "\n".join(lines)
    else:
        prompt = f"{instructions}\n{state}"
    return prompt, labels, qtype, str(item["expected"])


def von_choices(item: dict[str, Any]) -> dict[str, str]:
    """choices dict for von.decide: {label: description}."""
    crit = item["question"].get("criteria", {}) or {}
    return {lab: str(crit.get(lab, lab)) for lab in item["labels"]}


def expected_to_binary(expected: str) -> str:
    # JevBench noul items use yes/no labels natively; true/false only as fallback.
    return {"true": "yes", "false": "no"}.get(expected, expected)


def noul_option_texts(item: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Option (labels, texts) for noul items in dataset label order."""
    crit = item["question"].get("criteria", {}) or {}
    labels = list(item["labels"]) or ["no", "yes"]
    texts = []
    for lab in labels:
        key = {"yes": "true", "no": "false"}.get(lab, lab)
        texts.append(f"{lab}: {crit.get(key, crit.get(lab, lab))}")
    return labels, texts


def score_levels(item: dict[str, Any]) -> list[str]:
    """Ordered level descriptions for score items."""
    crit = item["question"].get("criteria", []) or []
    return [str(c) for c in crit]
