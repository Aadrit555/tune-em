"""Recordable product showcase: deterministic, non-interactive, mock backend.

Runs five acts with rich panels (no prompts, no network, no GPU):
  1. One-line typed decision (candidate-conditional vs raw vocab probability)
  2. L0 vs L1 debiasing + live temperature fit on synthetic validation (ECE delta)
  3. Utility-gated action + abstention under uncertainty
  4. Adaptive routing with hard call budget and token accounting
  5. Calibration artifact save -> tamper check -> reload

Everything is labeled mock/synthetic where applicable. Use --fast to skip
pacing sleeps (e.g. for CI), --games to append a 1-episode ViZDoom run.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rich.box import DOUBLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from anydecision import DecisionEngine, Question
from anydecision.calibration.metrics import compute_ece
from anydecision.theory.utility import UtilityMatrix


def _act(console: Console, n: int, title: str, body: str) -> None:
    console.print(Panel(f"[bold white]{body}[/bold white]", title=f"[bold cyan]Act {n}: {title}[/bold cyan]", box=DOUBLE))


def run_showcase(fast: bool = False, games: bool = False) -> Dict[str, Any]:
    console = Console()
    pause = 0.0 if fast else 0.8
    results: Dict[str, Any] = {}

    console.print("\n[bold cyan]anydecision :: typed probabilistic decision runtime (showcase)[/bold cyan]")
    console.print("[dim]Backend: mock (deterministic test fixtures, not model evaluation) | seed: 0[/dim]\n")
    engine = DecisionEngine(model="mock")

    # Act 1: one-liner with probability anatomy.
    _act(console, 1, "One-line decision",
         "A question plus candidates returns a typed Decision — no text generation,\n"
         "no regex parsing. Candidate-conditional, raw-vocab, margin, and entropy\nare distinct fields.")
    d1 = engine.choose("What is the primary vulnerability here?",
                       ["SQL Injection", "XSS", "CSRF", "Buffer Overflow"])
    t = Table(box=DOUBLE)
    t.add_column("Field", style="bold white")
    t.add_column("Value", style="bold yellow")
    t.add_row("answer", str(d1.answer))
    t.add_row("choice_prob P(cand|prompt,set)", f"{(d1.choice_probability or 0.0):.4f}")
    t.add_row("raw vocab P(token|prompt)", f"{(d1.model_token_probability or 0.0):.4f}")
    t.add_row("margin / entropy", f"{(d1.choice_margin or 0.0):.4f} / {(d1.predictive_entropy or 0.0):.4f} nats")
    console.print(t)
    results["act1_answer"] = d1.answer
    time.sleep(pause)

    # Act 2: L1 debias + live calibration fit with measured ECE delta.
    _act(console, 2, "Debias + calibrate, with receipts",
         "L1 averages deterministic option permutations (order invariance).\n"
         "Then we fit temperature scaling on 30 synthetic labeled items and\nmeasure ECE on 30 held-out ones. Synthetic data: machinery demo only.")
    rng = np.random.RandomState(0)
    topics = ["billing", "technical", "sales", "general"]
    pool = [Question.choice(f"Route inquiry #{i}: {rng.choice(['invoice', 'outage', 'pricing', 'login'])}",
                            topics) for i in range(60)]
    # Pseudo-labels from a frozen engine snapshot keep the demo deterministic.
    frozen = DecisionEngine(model="mock")
    labeled = [{"question": q, "label": str(frozen.decide(q, level="L0").answer)} for q in pool]
    train, held = labeled[:30], labeled[30:]
    l0 = [engine.decide(s["question"], level="L0") for s in held]
    ece_before = compute_ece(np.array([d.confidence for d in l0]),
                             np.array([1.0 if d.answer == s["label"] else 0.0 for d, s in zip(l0, held)]))
    engine.calibrate(train, method="temperature")
    l1 = engine.decide(pool[0], level="L1")
    l2 = [engine.decide(s["question"], level="L2") for s in held]
    ece_after = compute_ece(np.array([d.confidence for d in l2]),
                            np.array([1.0 if d.answer == s["label"] else 0.0 for d, s in zip(l2, held)]))
    t2 = Table(box=DOUBLE)
    t2.add_column("Metric", style="bold white")
    t2.add_column("Value", style="bold yellow")
    t2.add_row("L1 permutation agreement", f"{(l1.diagnostics.template_agreement or 0.0):.3f}" if l1.diagnostics else "n/a")
    t2.add_row("ECE before (L0)", f"{ece_before:.4f}")
    t2.add_row("ECE after (L2, fitted)", f"{ece_after:.4f}")
    console.print(t2)
    results.update(ece_before=ece_before, ece_after=ece_after)
    time.sleep(pause)

    # Act 3: utility-gated action + abstention.
    _act(console, 3, "Decide what to DO, and when to abstain",
         "Probabilities feed an explicit cost matrix (fraud -> freeze, never name\n"
         "guessing). Low confidence triggers abstention instead of a forced answer.")
    fraud_matrix = UtilityMatrix(
        actions=["freeze_account", "approve_transfer", "step_up_2fa"],
        states=["fraud", "legitimate"],
        matrix={
            "freeze_account": {"fraud": 20.0, "legitimate": -25.0},
            "approve_transfer": {"fraud": -200.0, "legitimate": 15.0},
            "step_up_2fa": {"fraud": 8.0, "legitimate": 6.0},
        },
    )
    q3 = Question.choice(" $14,800 transfer to a new overseas account?", ["fraud", "legitimate"])
    d3 = engine.decide(q3, utility_matrix=fraud_matrix)
    d3_abs = engine.decide(q3, min_confidence=0.9999)
    t3 = Table(box=DOUBLE)
    t3.add_column("Metric", style="bold white")
    t3.add_column("Value", style="bold yellow")
    t3.add_row("selected action", str(d3.selected_action))
    t3.add_row("EU(freeze)", f"{d3.expected_utilities.get('freeze_account', 0.0):+.2f}")
    t3.add_row("abstained @0.9999", f"{d3_abs.abstained} ({d3_abs.reason})")
    console.print(t3)
    results.update(selected_action=d3.selected_action, abstained=d3_abs.abstained)
    time.sleep(pause)

    # Act 4: adaptive routing under a hard budget.
    _act(console, 4, "Adaptive compute with a hard budget",
         "L0 -> L1 -> L2 escalation stops early on confidence; max_backend_calls\n"
         "is enforced, tokens are exact-or-labeled-estimated, exit reason recorded.")
    from anydecision.adaptive.router import AdaptiveComputeConfig

    d4 = engine.decide_adaptive(
        Question.choice("Is this login anomalous?", ["yes", "no"]),
        adaptive_config=AdaptiveComputeConfig(early_exit_l0_confidence=0.9999, max_backend_calls=16),
    )
    t4 = Table(box=DOUBLE)
    t4.add_column("Metric", style="bold white")
    t4.add_column("Value", style="bold yellow")
    t4.add_row("compute path", "+".join(d4.compute_path))
    t4.add_row("backend calls (budget 16)", str(d4.backend_calls))
    t4.add_row("tokens", f"{d4.tokens_processed} ({'estimated' if d4.tokens_estimated else 'exact'})")
    t4.add_row("exit reason", str(d4.exit_reason))
    console.print(t4)
    results.update(compute_path=d4.compute_path, backend_calls=d4.backend_calls)
    time.sleep(pause)

    # Act 5: artifact round-trip with tamper check.
    _act(console, 5, "Tamper-evident calibration artifact",
         "SHA-256 integrity + model-compatibility checks. Ed25519 signatures are\n"
         "optional for authenticity; hashing alone never claims it.")
    from anydecision.artifacts.saver import load_calibration_artifact

    with tempfile.TemporaryDirectory() as td:
        apath = str(Path(td) / "demo_head.json")
        engine.save_calibration(apath)
        cal, manifest = load_calibration_artifact(apath, expected_model="mock")
        t5 = Table(box=DOUBLE)
        t5.add_column("Check", style="bold white")
        t5.add_column("Result", style="bold green")
        t5.add_row("SHA-256 integrity", "PASS")
        t5.add_row("model compatibility", manifest.model)
        t5.add_row("head type", manifest.head)
        console.print(t5)
        results["artifact_head"] = manifest.head
    time.sleep(pause)

    if games:
        _act(console, 6, "Live engine policy (ViZDoom, 1 episode)",
             "Optional: the same Decision.selected_action drives real game actions.")
        try:
            from anydecision.games.vizdoom_env import ViZDoomDecisionRunner

            rep = ViZDoomDecisionRunner.run_simulation(
                engine=DecisionEngine(model="mock"), scenario="basic", num_episodes=1,
                render_console=False, policy="anydecision", seed=0)
            console.print(f"[bold green]basic/1ep: kills={rep.total_kills} "
                          f"deaths={rep.total_deaths} decisions={rep.total_decisions}[/bold green]")
            results["vizdoom_kills"] = rep.total_kills
        except Exception as e:
            console.print(f"[bold yellow]games skipped: {e}[/bold yellow]")

    console.print("\n[bold green]Showcase complete. Every number above was computed live.[/bold green]")
    console.print("[dim]See docs/CLAIMS.md for the claim-to-evidence matrix.[/dim]\n")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--games", action="store_true")
    args = ap.parse_args()
    run_showcase(fast=args.fast, games=args.games)
