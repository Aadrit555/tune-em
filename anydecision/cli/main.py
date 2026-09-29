"""Polished Command Line Interface (CLI) for anydecision / tune-em."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.box import DOUBLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from anydecision import __version__
from anydecision.artifacts.saver import load_calibration_artifact
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.core.types import DecisionLevel
from anydecision.evaluation.benchmark import BenchmarkRunner
from anydecision.evaluation.datasets import create_customer_escalation_benchmark

app = typer.Typer(
    name="anydecision",
    help="A typed, uncertainty-aware decision runtime extracting decisions directly from model probabilities.",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Show the anydecision version."""
    console.print(f"[bold green]anydecision / tune-em[/bold green] version [cyan]{__version__}[/cyan]")


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q", help="Question text to evaluate"),
    choices: List[str] = typer.Option(..., "--choices", "-c", help="Valid candidate answer options"),
    model: str = typer.Option("mock", "--model", "-m", help="Model name or backend (e.g. 'mock' or HF repo)"),
    level: str = typer.Option("L0", "--level", "-l", help="Decision level: L0, L1, or L2"),
    min_confidence: Optional[float] = typer.Option(None, "--min-confidence", help="Minimum required confidence"),
    target_error: Optional[float] = typer.Option(None, "--target-error", help="Target maximum posterior risk"),
    trace: bool = typer.Option(False, "--trace", help="Print complete step-by-step decision trace"),
) -> None:
    """Execute a single typed decision directly from model probabilities."""
    engine = DecisionEngine(model=model)
    q = Question.choice(text=question, choices=choices)

    dec_level = DecisionLevel(level.upper())
    res = engine.decide(
        q,
        level=dec_level,
        min_confidence=min_confidence,
        target_error=target_error,
        trace=trace,
    )

    # Render beautiful Rich output
    table = Table(title=f"Decision Summary (Level: {res.level})", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="bold")

    ans_str = f"[bold red]ABSTAINED ({res.reason})[/bold red]" if res.abstained else f"[bold green]{res.answer}[/bold green]"
    table.add_row("Decision", ans_str)
    table.add_row("Confidence", f"{res.confidence * 100:.2f}%")
    table.add_row("Uncertainty", f"{res.uncertainty:.4f}")
    table.add_row("Risk", f"{res.risk:.4f}")
    table.add_row("Method", res.method)
    table.add_row("Calibrated", str(res.calibrated))

    console.print(table)

    prob_table = Table(title="Candidate Probabilities", show_header=True)
    prob_table.add_column("Choice", style="magenta")
    prob_table.add_column("Probability", style="yellow")
    for opt, prob in res.probabilities.items():
        bar = "█" * int(prob * 30)
        prob_table.add_row(opt, f"{prob * 100:6.2f}%  {bar}")

    console.print(prob_table)

    if dec_level == DecisionLevel.L1 and res.diagnostics:
        diag = res.diagnostics
        console.print(
            Panel(
                f"Permutations Evaluated: [cyan]{diag.number_of_permutations}[/cyan] | "
                f"Permutation Agreement: [green]{diag.template_agreement * 100:.1f}%[/green] | "
                f"Position Bias: [yellow]{diag.option_order_sensitivity:.4f}[/yellow] | "
                f"Entropy: [blue]{diag.entropy:.3f} nats[/blue] | "
                f"Latency: [white]{diag.latency_ms:.2f} ms[/white]",
                title="L1 Invariance & Diagnostics",
            )
        )

    if trace and res.trace:
        console.print("\n[bold]Decision Execution Trace:[/bold]")
        for step in res.trace.steps:
            console.print(f"  ➜ [cyan]{step.step_name}[/cyan]: {step.description}")


@app.command()
def benchmark(
    model: str = typer.Option("mock", "--model", "-m", help="Model name or backend"),
    samples: int = typer.Option(20, "--samples", "-n", help="Number of benchmark samples"),
    level: str = typer.Option("L0", "--level", "-l", help="Decision level (L0, L1)"),
) -> None:
    """Run an automated calibration and decision benchmark."""
    console.print(f"[bold green]Starting benchmark on {model} (level: {level})...[/bold green]")
    engine = DecisionEngine(model=model)
    data = create_customer_escalation_benchmark(n_samples=samples)

    runner = BenchmarkRunner(engine)
    res = runner.run(data, name="Escalation Benchmark", level=DecisionLevel(level.upper()))
    console.print(Panel(res.summary(), title="Benchmark Results"))


@app.command("inspect-artifact")
def inspect_artifact(
    filepath: Path = typer.Argument(..., help="Path to .json calibration artifact file"),
) -> None:
    """Inspect and verify a saved calibration artifact."""
    if not filepath.exists():
        console.print(f"[bold red]File not found: {filepath}[/bold red]")
        raise typer.Exit(1)

    try:
        calibrator, manifest = load_calibration_artifact(filepath, verify_integrity=True)
        table = Table(title="Calibration Artifact Manifest", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="bold")
        table.add_row("Model", manifest.model)
        table.add_row("Model Revision", manifest.model_revision)
        table.add_row("Backend", manifest.backend)
        table.add_row("Head", manifest.head)
        table.add_row("Created At", manifest.created_at)
        table.add_row("Data Hash", manifest.training_data_hash[:16] + "...")
        table.add_row("Checksum", manifest.integrity.sha256[:16] + "... [VALID]")
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Artifact verification failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Service host"),
    port: int = typer.Option(8000, "--port", "-p", help="Service port"),
    model: str = typer.Option("mock", "--model", "-m", help="Underlying model"),
) -> None:
    """Start the FastAPI HTTP decision runtime service."""
    import uvicorn
    from anydecision.serving.app import create_app

    engine = DecisionEngine(model=model)
    service_app = create_app(engine=engine)
    console.print(f"[bold green]Starting anydecision service on http://{host}:{port}...[/bold green]")
    uvicorn.run(service_app, host=host, port=port)


@app.command()
def demo(
    port: int = typer.Option(7860, "--port", "-p", help="Demo port"),
    share: bool = typer.Option(False, "--share", help="Share public Gradio link"),
) -> None:
    """Launch the interactive Gradio research demo."""
    from anydecision.demo.app import launch_demo
    launch_demo(server_port=port, share=share)


@app.command()
def tui(
    model: str = typer.Option("mock", "--model", "-m", help="Underlying model or backend ('mock', HF repo, or vLLM)"),
) -> None:
    """Launch the interactive Linux terminal-based UI (TUI)."""
    from anydecision.tui.app import run_tui
    run_tui(model=model)


@app.command()
def showcase(
    fast: bool = typer.Option(False, "--fast", help="Skip pacing sleeps (CI-friendly)"),
    games: bool = typer.Option(False, "--games", help="Append a 1-episode live ViZDoom run"),
) -> None:
    """Run the recordable product showcase (deterministic, mock backend, no prompts)."""
    from anydecision.demo.showcase import run_showcase
    run_showcase(fast=fast, games=games)


@app.command()
def doom(
    episodes: int = typer.Option(3, "--episodes", "-e", help="Number of DOOM combat episodes to simulate"),
    difficulty: str = typer.Option("medium", "--difficulty", "-d", help="Difficulty: medium, hard, or boss"),
    model: str = typer.Option("mock", "--model", "-m", help="Decision model backend"),
) -> None:
    """Run the SYNTHETIC toy-combat simulator (deterministic unit-test env, not real DOOM)."""
    from anydecision.games.doom import DoomCombatBenchmarkRunner
    engine = DecisionEngine(model=model)
    console.print(f"[bold red]Initializing DOOM Tactical AI Combat Benchmark ({difficulty.upper()} difficulty)...[/bold red]")
    stats = DoomCombatBenchmarkRunner.run_simulation(
        engine=engine,
        num_episodes=episodes,
        difficulty=difficulty,
        render_console=True,
    )
    console.print("\n[bold green]=== DOOM COMBAT BENCHMARK RESULTS ===[/bold green]")
    console.print(f"Episodes Survived:     {stats['num_episodes'] * stats['survival_rate']:.0f}/{stats['num_episodes']} ({stats['survival_rate']*100:.1f}%)")
    console.print(f"Total Demons Slain:    {stats['total_kills']}")
    console.print(f"Mean Decision Latency: {stats['mean_latency_ms']:.2f} ms")
    console.print(f"Throughput:            {stats['decisions_per_second']:.1f} decisions / sec")


@app.command()
def compare_von(
    data_dir: str = typer.Option(..., "--data-dir", help="JevBench datasets/public dir (easy/original/hard.jsonl)"),
    split: str = typer.Option("dev", "--split", help="Split to evaluate (train/dev only; test stays locked)"),
    model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="Our entrant's causal LM"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Score JSON artifact path"),
) -> None:
    """Real head-to-head: native von-sdk 1.3 vs anydecision on identical JevBench items.

    Runs both entrants (same state/instructions/labels), then scores accuracy,
    ECE, Brier, latency, and McNemar+MDE. Below-MDE deltas are UNRESOLVABLE.
    """
    import subprocess
    import sys
    from pathlib import Path

    if split not in ("train", "dev"):
        raise typer.BadParameter("--split must be 'train' or 'dev' (test stays locked; see PROTOCOL.md)")
    try:
        import von  # noqa: F401
    except ImportError:
        console.print("[bold red]compare-von requires 'von-sdk>=1.3.1' (pip install von-sdk).[/bold red]")
        raise typer.Exit(code=2)
    for f in ("easy.jsonl", "original.jsonl", "hard.jsonl"):
        if not Path(data_dir, f).exists():
            console.print(f"[bold red]Missing {f} in --data-dir. Clone https://github.com/fstandhartinger/jevbench and pass datasets/public.[/bold red]")
            raise typer.Exit(code=2)

    harness = Path(__file__).resolve().parents[2] / "benchmarks" / "jevbench_shootout"
    tmp = Path.cwd() / ".compare_von_tmp"
    tmp.mkdir(exist_ok=True)
    von_out = tmp / f"von_{split}.json"
    ours_out = tmp / f"ours_{split}.json"
    splits = harness / "splits.json"
    if not splits.exists():
        r = subprocess.run([sys.executable, str(harness / "split.py"), "--data-dir", data_dir,
                            "--out", str(splits), "--seed", "0"])
        if r.returncode != 0:
            raise typer.Exit(code=r.returncode)
    for cmd in (
        [sys.executable, str(harness / "run_von.py"), "--data-dir", data_dir,
         "--splits", str(splits), "--split", split, "--out", str(von_out)],
        [sys.executable, str(harness / "run_ours_lm.py"), "--data-dir", data_dir,
         "--splits", str(splits), "--split", split, "--out", str(ours_out),
         "--model", model],
    ):
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise typer.Exit(code=r.returncode)

    sys.path.insert(0, str(harness))
    from score import compare, summarize, load_records
    ref_recs, _ = load_records(str(von_out))
    cmp_recs, _ = load_records(str(ours_out))
    ref_s, cmp_s = summarize(str(von_out)), summarize(str(ours_out))
    h2h = compare(ref_recs, cmp_recs)
    if output:
        Path(output).write_text(json.dumps(
            {"von": ref_s, "ours": cmp_s, "head_to_head": h2h}, indent=2), encoding="utf-8")

    table = Table(box=DOUBLE, title=f"anydecision vs von-1.3 ({split} split, n={h2h['n']})")
    table.add_column("Entrant", style="bold white")
    table.add_column("Accuracy", style="bold yellow")
    table.add_column("ECE", style="bold yellow")
    table.add_column("Brier", style="bold yellow")
    table.add_column("Mean latency", style="bold cyan")
    for name, s in (("von-1.3 (native)", ref_s), ("anydecision+" + model.split("/")[-1], cmp_s)):
        table.add_row(name, f"{s['accuracy']:.3f}", f"{s['ece']:.4f}",
                      f"{s['brier']:.4f}", f"{s['mean_latency_ms']:.0f} ms")
    console.print(table)
    console.print(f"McNemar p={h2h['mcnemar_p']:.4f} MDE={h2h['mde']:.3f} "
                  f"discordant={h2h['discordant']} -> [bold]{h2h['verdict']}[/bold]")


@app.command("real-doom")
def real_doom(
    map_code: str = typer.Option("E1M1", "--map", "-m", help="Map to evaluate (e.g. E1M1, E1M8, E2M8)"),
    skill: int = typer.Option(3, "--skill", "-s", help="Skill level: 1 (Easy) to 5 (Nightmare)"),
    wad_path: Optional[str] = typer.Option(None, "--wad", "-w", help="Path to genuine DOOM.WAD (required)"),
    model: str = typer.Option("mock", "--model", help="Decision model backend"),
    episodes: int = typer.Option(1, "--episodes", "-e", help="Number of live engine episodes"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Deterministic seed (recorded in artifact)"),
    policy: str = typer.Option("anydecision", "--policy", help="Policy: anydecision, random, scripted"),
    observation_mode: str = typer.Option("HYBRID", "--observation-mode", help="Observation mode: STATE, VISION, HYBRID"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Machine-readable JSON artifact path"),
) -> None:
    """Evaluate a decision policy on a genuine WAD map inside the live ViZDoom engine.

    Requires a real DOOM.WAD (--wad) and the vizdoom package. Fails loudly
    when either is missing; never substitutes a synthetic simulator.
    """
    from anydecision.games.ultimate_doom import RealDoomEvaluator
    engine = DecisionEngine(model=model)
    RealDoomEvaluator.run_map_evaluation(
        engine=engine,
        map_code=map_code,
        skill_level=skill,
        wad_path=wad_path,
        render_console=True,
        num_episodes=episodes,
        seed=seed,
        policy=policy,
        observation_mode=observation_mode,
        output_path=output,
    )


@app.command("vizdoom")
def vizdoom_cmd(
    scenario: str = typer.Option("defend_the_center", "--scenario", "-s", help="ViZDoom scenario: defend_the_center, defend_the_line, deadly_corridor, basic, etc."),
    episodes: int = typer.Option(3, "--episodes", "-e", help="Number of benchmark episodes"),
    skill: int = typer.Option(4, "--skill", "-k", help="Doom skill level: 1 to 5 (default 4 = Ultra-Violence)"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Max steps per episode (default: 1200 if rendered, 100 headless)"),
    wad: Optional[str] = typer.Option(None, "--wad", "-w", help="Optional path to custom DOOM.WAD binary"),
    model: str = typer.Option("mock", "--model", help="Decision model backend"),
    render: bool = typer.Option(False, "--render", help="Render graphical game window"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Deterministic seed (recorded in artifact)"),
    policy: str = typer.Option("anydecision", "--policy", help="Policy: anydecision, random, scripted, learned"),
    observation_mode: str = typer.Option("HYBRID", "--observation-mode", help="Observation mode: STATE, VISION, HYBRID"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Machine-readable JSON artifact path"),
    estimator: Optional[str] = typer.Option(None, "--estimator", help="State-estimator .npz (learned policy; default: bundled artifact)"),
) -> None:
    """Run a typed decision policy inside the live ViZDoom engine (ViZDoom integration)."""
    from anydecision.games.vizdoom_env import ViZDoomDecisionRunner
    engine = DecisionEngine(model=model)
    ViZDoomDecisionRunner.run_simulation(
        engine=engine,
        scenario=scenario,
        num_episodes=episodes,
        skill=skill,
        max_steps_per_episode=max_steps,
        wad_path=wad,
        window_visible=render,
        render_console=True,
        seed=seed,
        policy=policy,
        observation_mode=observation_mode,
        output_path=output,
        estimator_path=estimator,
    )



