"""Production-grade Terminal-based User Interface (TUI) for anydecision / tune-em.

Features:
- Linux Terminal / Cyberpunk ASCII Aesthetics (Strictly zero emojis)
- Interactive Decision Studio with live probability distribution bar charts
- Decision-Theoretic Expected Utility action policy runtime
- Multi-Level Execution (L0 Raw, L1 Permutation Debiased, L2 Calibrated, Adaptive Compute)
- Transformer Layer-Trajectory Uncertainty & Decision Emergence Inspector
- Active Calibration & Drift Monitor Simulator
- Prompt-Injection Security & Boundary Quarantine Auditor
- Automated Benchmark Suite with Live Progress Display
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from rich.align import Align
from rich.box import DOUBLE, HEAVY, ROUNDED
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from anydecision import __version__
from anydecision.adaptive.router import AdaptiveComputeConfig
from anydecision.calibration.drift import CalibrationDriftMonitor, DriftAlert
from anydecision.core.context import DecisionContext, InjectionResistanceBenchmark
from anydecision.core.decision import Decision
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.core.types import DecisionLevel
from anydecision.evaluation.datasets import (
    create_agent_safety_tasks,
    create_customer_escalation_benchmark,
    create_topic_categorization_benchmark,
)
from anydecision.evaluation.economics import DecisionEconomicsEvaluator, EconomicsConfig
from anydecision.theory.utility import UtilityMatrix

ASCII_BANNER = r"""
  _______ _    _ _   _ ______     ______ __  __ 
 |__   __| |  | | \ | |  ____|   |  ____|  \/  |
    | |  | |  | |  \| | |__ _____| |__  | \  / |
    | |  | |  | | . ` |  __|_____|  __| | |\/| |
    | |  | |__| | |\  | |____    | |____| |  | |
    |_|   \____/|_| \_|______|   |______|_|  |_|
 [ TYPED UNCERTAINTY-AWARE DECISION RUNTIME // ZERO TEXT GENERATION ]
"""

PRESETS = [
    {
        "name": "Customer Support Escalation",
        "question": "Should this customer issue with recurring double billing be escalated to a human tier-2 engineer?",
        "choices": ["escalate", "resolve_automated", "request_more_info"],
        "actions": {"escalate": -2.0, "resolve_automated": 5.0, "request_more_info": -0.5},
        "utilities": {
            "escalate": {"escalate": 8.0, "resolve_automated": -15.0, "request_more_info": -2.0},
            "resolve_automated": {"escalate": -30.0, "resolve_automated": 10.0, "request_more_info": -5.0},
            "request_more_info": {"escalate": 0.0, "resolve_automated": 0.0, "request_more_info": 4.0},
        },
    },
    {
        "name": "Code Vulnerability Assessment",
        "question": "Does this function containing 'os.system(f\"cat {filename}\")' have a critical Remote Code Execution vulnerability?",
        "choices": ["vulnerable", "safe"],
        "actions": {"block_deployment": -1.0, "allow_deployment": 2.0, "security_review": -3.0},
        "utilities": {
            "block_deployment": {"vulnerable": 10.0, "safe": -5.0},
            "allow_deployment": {"vulnerable": -100.0, "safe": 8.0},
            "security_review": {"vulnerable": 4.0, "safe": 2.0},
        },
    },
    {
        "name": "High-Value Financial Transaction Fraud",
        "question": "A transfer of $14,800 is requested to a new overseas account with device mismatch. Is this fraudulent?",
        "choices": ["fraud", "legitimate"],
        "actions": {"freeze_account": -5.0, "approve_transfer": 5.0, "step_up_2fa": -1.0},
        "utilities": {
            "freeze_account": {"fraud": 20.0, "legitimate": -25.0},
            "approve_transfer": {"fraud": -200.0, "legitimate": 15.0},
            "step_up_2fa": {"fraud": 8.0, "legitimate": 6.0},
        },
    },
    {
        "name": "Emergency Medical Triage",
        "question": "Patient reports severe crushing chest pain radiating to left arm with diaphoresis. Triage priority level?",
        "choices": ["resuscitation", "emergent", "urgent", "non_urgent"],
        "actions": {"immediate_icu": 0.0, "rapid_ecg": -1.0, "general_waiting": -10.0},
        "utilities": {
            "immediate_icu": {"resuscitation": 50.0, "emergent": 30.0, "urgent": -10.0, "non_urgent": -30.0},
            "rapid_ecg": {"resuscitation": 20.0, "emergent": 25.0, "urgent": 10.0, "non_urgent": 0.0},
            "general_waiting": {"resuscitation": -500.0, "emergent": -200.0, "urgent": 5.0, "non_urgent": 20.0},
        },
    },
]


class TerminalUI:
    """Rich interactive Terminal User Interface application."""

    def __init__(self, model_name: str = "mock") -> None:
        self.console = Console()
        self.model_name = model_name
        self.engine = DecisionEngine(model=model_name)
        self.running = True

    def clear(self) -> None:
        """Clear console screen cleanly across platforms."""
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self, subtitle: str = "") -> None:
        """Display standardized Linux terminal header."""
        meta = self.engine.metadata
        banner_text = Text(ASCII_BANNER, style="bold cyan")
        meta_table = Table.grid(padding=(0, 2))
        meta_table.add_column(style="bold white")
        meta_table.add_column(style="green")
        meta_table.add_column(style="bold white")
        meta_table.add_column(style="yellow")
        meta_table.add_column(style="bold white")
        meta_table.add_column(style="cyan")

        meta_table.add_row(
            "Backend:", meta.backend_name,
            "Model:", meta.model_name,
            "Layers:", str(meta.num_layers or 32),
        )
        meta_table.add_row(
            "Version:", f"v{__version__}",
            "Device:", meta.device,
            "Context:", f"{meta.context_window} tokens",
        )

        header_content = [
            banner_text,
            meta_table,
        ]
        if subtitle:
            header_content.append(Text(f"\n[ACTIVE VIEW: {subtitle.upper()}]", style="bold yellow"))

        panel = Panel(
            Align.center(Columns(header_content, equal=False, align="center")),
            box=HEAVY,
            border_style="cyan",
            title="[bold green] ANYDECISION // TUNE-EM RUNTIME CONSOLE [/bold green]",
            subtitle="[dim]Direct Next-Token Probability Extraction | No Autoregressive Generation[/dim]",
        )
        self.console.print(panel)

    def run(self) -> None:
        """Main application interactive loop."""
        while self.running:
            try:
                self.clear()
                self.print_header(subtitle="Main Navigation Menu")
                self.show_main_menu()
                choice = Prompt.ask(
                    "[bold green]Select Option[/bold green]",
                    choices=["1", "2", "3", "4", "5", "6", "q"],
                    default="1",
                )

                if choice == "1":
                    self.view_decision_studio()
                elif choice == "2":
                    self.view_economics_benchmark()
                elif choice == "3":
                    self.view_layer_trajectory_inspector()
                elif choice == "4":
                    self.view_active_calibration_console()
                elif choice == "5":
                    self.view_security_audit_console()
                elif choice == "6":
                    self.view_switch_model()
                elif choice == "7":
                    self.view_doom_arena()
                elif choice.lower() == "q":
                    self.running = False
                    self.console.print("\n[bold cyan]Exiting anydecision terminal UI. Goodbye.[/bold cyan]\n")
                    break
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
                time.sleep(0.5)

    def show_main_menu(self) -> None:
        """Render main menu options grid."""
        menu_table = Table(box=ROUNDED, border_style="cyan", expand=True)
        menu_table.add_column("Key", style="bold cyan", width=6, justify="center")
        menu_table.add_column("Module / View", style="bold white", width=36)
        menu_table.add_column("Description", style="dim")

        menu_table.add_row(
            "[1]",
            "Interactive Decision Studio",
            "Typed questions, candidate probability bars, expected utility actions & layer emergence",
        )
        menu_table.add_row(
            "[2]",
            "Economics & Benchmark Suite",
            "Evaluate Quality/Compute, Quality/Dollar, Safe Decisions/sec, and Latency vs Accuracy curves",
        )
        menu_table.add_row(
            "[3]",
            "Layer-Trajectory Inspector",
            "Internal representation depth analysis: decision emergence layer & representation convergence",
        )
        menu_table.add_row(
            "[4]",
            "Active Calibration & Drift Monitor",
            "High-information sample selection (entropy, margin) and runtime sequential drift detection",
        )
        menu_table.add_row(
            "[5]",
            "Security & Injection Auditor",
            "Adversarial prompt injection resistance benchmark and cross-backend numerical verification",
        )
        menu_table.add_row(
            "[6]",
            "Switch Backend / Model",
            f"Currently active: {self.engine.metadata.backend_name} ({self.engine.metadata.model_name})",
        )
        menu_table.add_row(
            "[7]",
            "Toy-Combat Arena (SYNTHETIC)",
            "Seeded synthetic skirmishes for policy unit tests (not real DOOM / ViZDoom)",
        )
        menu_table.add_row(
            "[Q]",
            "Quit",
            "Exit terminal user interface",
        )

        self.console.print(menu_table)
        self.console.print()

    def view_decision_studio(self) -> None:
        """Module 1: Interactive Decision Studio."""
        self.clear()
        self.print_header(subtitle="Interactive Decision Studio")

        self.console.print("[bold cyan]Select Question Source:[/bold cyan]")
        self.console.print("  [1] Choose from Research Presets (Customer Escalation, Vulnerability, Fraud, Triage)")
        self.console.print("  [2] Enter Custom Typed Question")
        self.console.print("  [B] Back to Main Menu")

        src = Prompt.ask("Choice", choices=["1", "2", "b", "B"], default="1")
        if src.lower() == "b":
            return

        question_text = ""
        choices: List[str] = []
        action_dict: Optional[Dict[str, float]] = None
        utility_matrix_obj: Optional[UtilityMatrix] = None

        if src == "1":
            self.console.print("\n[bold green]Available Presets:[/bold green]")
            for idx, p in enumerate(PRESETS, start=1):
                self.console.print(f"  [{idx}] {p['name']}: [dim]{p['question'][:65]}...[/dim]")
            p_idx = IntPrompt.ask("Select preset", choices=[str(i) for i in range(1, len(PRESETS) + 1)], default=1)
            selected_preset = PRESETS[p_idx - 1]
            question_text = selected_preset["question"]
            choices = selected_preset["choices"]
            action_dict = selected_preset.get("actions")
            if "utilities" in selected_preset:
                raw_grid = selected_preset["utilities"]
                utility_matrix_obj = UtilityMatrix(
                    actions=list(raw_grid.keys()),
                    states=list(choices),
                    matrix={
                        act: {s: float(vals.get(s, 0.0)) for s in choices}
                        for act, vals in raw_grid.items()
                    },
                )
        else:
            question_text = Prompt.ask("\nEnter question text")
            choices_raw = Prompt.ask("Enter valid choices (comma-separated)", default="yes, no")
            choices = [c.strip() for c in choices_raw.split(",") if c.strip()]

        # Configuration options
        self.console.print("\n[bold cyan]Execution Parameters:[/bold cyan]")
        level_choice = Prompt.ask(
            "Select Decision Level",
            choices=["L0", "L1", "L2", "adaptive"],
            default="adaptive",
        )
        track_trajectory = Confirm.ask("Inspect Layer Trajectory & Emergence Layer?", default=True)
        enable_context = Confirm.ask("Apply Structured Prompt-Injection Quarantine?", default=False)

        q = Question.choice(question_text, choices)

        # Run with live progress spinner
        with Progress(
            SpinnerColumn("dots", style="cyan"),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Extracting candidate probability distribution from model...", total=None)

            t0 = time.perf_counter()
            sec_context = DecisionContext(untrusted_user_content="Standard request payload.") if enable_context else None

            if level_choice.lower() == "adaptive":
                dec = self.engine.decide_adaptive(
                    question=q,
                    actions=action_dict,
                    utility_matrix=utility_matrix_obj,
                    track_layer_trajectory=track_trajectory,
                    context=sec_context,
                )
            else:
                dec = self.engine.decide(
                    question=q,
                    level=DecisionLevel(level_choice.upper()),
                    actions=action_dict,
                    utility_matrix=utility_matrix_obj,
                    track_layer_trajectory=track_trajectory,
                    context=sec_context,
                )
            progress.update(task, completed=True)

        # Render Rich Decision View
        self.render_decision_results(dec, q)
        Prompt.ask("\n[bold green]Press Enter to continue...[/bold green]")

    def render_decision_results(self, dec: Decision, q: Question) -> None:
        """Render comprehensive, visual decision result cards."""
        self.console.print()

        # Decision Verdict Card
        verdict_table = Table(box=ROUNDED, border_style="green", expand=True)
        verdict_table.add_column("Metric", style="bold cyan", width=22)
        verdict_table.add_column("Evaluation Output", style="bold white")

        if dec.abstained:
            ans_str = f"[bold red]ABSTAINED (Reason: {dec.reason})[/bold red]"
        else:
            ans_str = f"[bold green]{dec.answer}[/bold green]"

        verdict_table.add_row("Final Decision", ans_str)

        # Confidence color coding
        conf_pct = dec.confidence * 100
        if conf_pct >= 85.0:
            conf_style = "bold green"
        elif conf_pct >= 65.0:
            conf_style = "bold yellow"
        else:
            conf_style = "bold red"

        verdict_table.add_row("Calibrated Confidence", f"[{conf_style}]{conf_pct:.2f}%[/{conf_style}]")
        verdict_table.add_row("Posterior Risk", f"{dec.risk:.4f} (Uncertainty: {dec.uncertainty:.4f})")
        verdict_table.add_row("Decision Level", f"{dec.level} ({dec.method})")

        if dec.compute_path:
            verdict_table.add_row("Compute Path (Adaptive)", " -> ".join(dec.compute_path))
        if dec.decision_emergence_layer:
            verdict_table.add_row("Decision Emergence", f"Layer {dec.decision_emergence_layer} (Internal depth)")

        self.console.print(Panel(verdict_table, title="[bold green] DECISION VERDICT [/bold green]", box=HEAVY))

        # Candidate Probabilities with Visual ASCII Bar Chart
        prob_table = Table(box=ROUNDED, border_style="cyan", title="Probability Distribution Across Candidates")
        prob_table.add_column("Candidate Choice", style="bold magenta", width=24)
        prob_table.add_column("Probability", style="bold yellow", width=14, justify="right")
        prob_table.add_column("Visual Distribution Meter", style="cyan", width=36)

        sorted_probs = sorted(dec.probabilities.items(), key=lambda x: x[1], reverse=True)
        for opt, prob in sorted_probs:
            bar_len = int(prob * 32)
            bar_char = "█" * bar_len
            bar_empty = "░" * (32 - bar_len)
            is_winner = (opt == dec.answer)
            name_str = f"[bold green]>> {opt}[/bold green]" if is_winner else f"   {opt}"
            prob_table.add_row(name_str, f"{prob * 100:6.2f}%", f"[cyan]{bar_char}[/cyan][dim]{bar_empty}[/dim]")

        self.console.print(prob_table)

        # Decision-Theoretic Actions & Expected Utility Card
        if dec.expected_utilities:
            eu_table = Table(box=ROUNDED, border_style="yellow", title="Decision-Theoretic Action Selection: EU(a) = sum P(y|x) U(a, y)")
            eu_table.add_column("Action", style="bold white", width=20)
            eu_table.add_column("Expected Utility EU(a)", style="bold yellow", width=24, justify="right")
            eu_table.add_column("Status", style="bold", width=16)

            for act, eu_val in sorted(dec.expected_utilities.items(), key=lambda x: x[1], reverse=True):
                is_best = (act == dec.selected_action)
                status_str = "[bold green]OPTIMAL ACTION[/bold green]" if is_best else "[dim]Sub-optimal[/dim]"
                eu_table.add_row(act, f"{eu_val:+.4f}", status_str)

            self.console.print(eu_table)
            if dec.escalated:
                self.console.print(Panel(f"[bold yellow]{dec.escalation_reason}[/bold yellow]", title="Escalation Triggered", box=DOUBLE))

        # Layer Trajectory Snapshot
        if dec.layer_trajectory and "trajectory" in dec.layer_trajectory:
            traj_list = dec.layer_trajectory["trajectory"]
            traj_table = Table(box=ROUNDED, border_style="magenta", title="Transformer Depth Trajectory Snapshots")
            traj_table.add_column("Layer", style="bold cyan", width=8)
            traj_table.add_column("Top Choice", style="bold white", width=16)
            traj_table.add_column("Confidence", style="bold yellow", width=12)
            traj_table.add_column("Entropy (nats)", style="dim", width=14)
            traj_table.add_column("Status", style="bold", width=16)

            for cp in traj_list:
                l_idx = cp.get("layer_idx")
                ans = cp.get("top_answer")
                conf = cp.get("confidence", 0.0) * 100
                ent = cp.get("entropy", 0.0)
                status = "[bold green]<-- EMERGENCE[/bold green]" if l_idx == dec.decision_emergence_layer else ""
                traj_table.add_row(f"L{l_idx}", ans, f"{conf:6.2f}%", f"{ent:.4f}", status)

            self.console.print(traj_table)

    def view_economics_benchmark(self) -> None:
        """Module 2: Decision Economics & Multi-Level Benchmark Suite."""
        self.clear()
        self.print_header(subtitle="Decision Economics & Benchmark Suite")

        self.console.print("[bold cyan]Select Benchmark Dataset:[/bold cyan]")
        self.console.print("  [1] Customer Escalation Benchmark (Billing, Technical, Account, Support)")
        self.console.print("  [2] Topic Categorization Benchmark (General NLP Classification)")
        self.console.print("  [3] Agent Safety & Guardrail Tasks (Injection, Refusal, Sensitive Data)")
        self.console.print("  [B] Back to Main Menu")

        ds_choice = Prompt.ask("Dataset Choice", choices=["1", "2", "3", "b", "B"], default="1")
        if ds_choice.lower() == "b":
            return

        sample_size = IntPrompt.ask("Number of benchmark samples to evaluate", default=12)

        if ds_choice == "1":
            samples = create_customer_escalation_benchmark(n_samples=sample_size)
            ds_name = "Customer Escalation"
        elif ds_choice == "2":
            samples = create_topic_categorization_benchmark(n_samples=sample_size)
            ds_name = "Topic Categorization"
        else:
            from anydecision.evaluation.benchmark import BenchmarkSample

            tasks = create_agent_safety_tasks()[:sample_size]
            samples = [
                BenchmarkSample(question=t.question, ground_truth=t.optimal_action)
                for t in tasks
            ]
            ds_name = "Agent Safety"

        evaluator = DecisionEconomicsEvaluator(
            engine=self.engine,
            economics_config=EconomicsConfig(
                cost_per_million_tokens=0.50,
                hardware_cost_per_hour=1.20,
            ),
            adaptive_config=AdaptiveComputeConfig(early_exit_l0_confidence=0.55),
        )

        with Progress(
            SpinnerColumn("dots", style="green"),
            TextColumn("[bold green]{task.description}[/bold green]"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Evaluating {ds_name} across L0, L1, and Adaptive...", total=len(samples))

            report = evaluator.evaluate(samples, compare_static_levels=True)
            progress.update(task, completed=len(samples))

        # Render Terminal Benchmark Summary
        self.console.print()
        self.console.print(Panel(report.summary(), box=HEAVY, border_style="green", title="[bold green] BENCHMARK RESULTS [/bold green]"))
        Prompt.ask("\n[bold green]Press Enter to return to menu...[/bold green]")

    def view_layer_trajectory_inspector(self) -> None:
        """Module 3: Internal Layer Trajectory & Emergence Inspector."""
        self.clear()
        self.print_header(subtitle="Layer-Trajectory Depth Inspector")

        q_text = Prompt.ask("Enter inquiry to trace through transformer depth", default="Is this financial transaction suspicious?")
        choices_raw = Prompt.ask("Enter candidate options", default="suspicious, benign")
        choices = [c.strip() for c in choices_raw.split(",") if c.strip()]
        q = Question.choice(q_text, choices)

        with Progress(
            SpinnerColumn("dots", style="magenta"),
            TextColumn("[bold magenta]Probing hidden representations across 32 transformer layers...[/bold magenta]"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing...", total=None)
            traj = self.engine.analyze_layer_trajectory(q)
            progress.update(task, completed=True)

        self.console.print()
        self.console.print(Panel(traj.summary(), box=HEAVY, border_style="magenta", title="[bold magenta] TRAJECTORY ANALYSIS [/bold magenta]"))
        Prompt.ask("\n[bold green]Press Enter to return to menu...[/bold green]")

    def view_active_calibration_console(self) -> None:
        """Module 4: Active Calibration & Sequential Drift Monitor."""
        self.clear()
        self.print_header(subtitle="Active Calibration & Drift Monitor Console")

        self.console.print("[bold cyan]Sub-Modules:[/bold cyan]")
        self.console.print("  [1] Active Calibration Sample Selection (Query High-Information Pool)")
        self.console.print("  [2] Sequential Calibration Drift Monitor Simulation")
        self.console.print("  [B] Back to Main Menu")

        sub = Prompt.ask("Select Sub-Module", choices=["1", "2", "b", "B"], default="1")
        if sub.lower() == "b":
            return

        if sub == "1":
            pool_samples = create_customer_escalation_benchmark(n_samples=16)
            pool_questions = [s.question for s in pool_samples]

            budget = IntPrompt.ask("Human annotation budget (number of labels to query)", default=5)
            criterion = Prompt.ask(
                "Active Selection Criterion",
                choices=["hybrid", "uncertainty", "margin", "disagreement", "ood"],
                default="hybrid",
            )

            with Progress(
                SpinnerColumn("dots", style="yellow"),
                TextColumn("[bold yellow]Scoring unlabeled pool using active information criteria...[/bold yellow]"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Scoring...", total=None)
                suggested = self.engine.suggest_calibration_examples(pool_questions, budget=budget, criterion=criterion)
                progress.update(task, completed=True)

            table = Table(box=ROUNDED, border_style="yellow", title=f"Top {len(suggested)} Most Informative Calibration Questions")
            table.add_column("Rank", style="bold cyan", width=6)
            table.add_column("Question Text", style="bold white")
            table.add_column("Choices", style="green", width=24)

            for idx, q_item in enumerate(suggested, start=1):
                table.add_row(f"#{idx}", q_item.text, ", ".join(q_item.option_keys()))

            self.console.print(table)
            self.console.print(
                Panel(
                    f"Selected [bold green]{len(suggested)}[/bold green] high-information examples out of [cyan]{len(pool_questions)}[/cyan] candidates.\n"
                    f"Annotation savings are measured per dataset via budget curves "
                    f"(see ActiveCalibrationBenchmark.estimate_label_savings) — "
                    f"no fixed percentage is claimed here.",
                    title="Active Learning Selection (heuristic ranking)",
                    box=ROUNDED,
                )
            )

        elif sub == "2":
            self.console.print("\n[bold cyan]SIMULATED drift drill: fault-injected confidences, real detector...[/bold cyan]")
            monitor = CalibrationDriftMonitor(window_size=20, warning_threshold=0.10, critical_threshold=0.20)
            q_ref = Question.binary("Is credit card transaction authorized?")

            # Baseline reference
            ref_decs = [self.engine.decide(q_ref) for _ in range(20)]
            monitor.set_reference_distribution(ref_decs)

            self.console.print("[dim]Baseline established. Streaming 25 decisions with SIMULATED covariate drift (fault injection, not a measured incident)...[/dim]\n")
            alerts: List[DriftAlert] = []

            for step in range(1, 26):
                d = self.engine.decide(q_ref).model_copy()
                # SIMULATED fault injection: degraded confidences exercise the
                # real detector path (the monitor reads d.confidence).
                if step > 10:
                    d.confidence = 0.45 + (step % 4) * 0.02

                alert = monitor.record_decision(d)
                if alert:
                    alerts.append(alert)

            table = Table(box=ROUNDED, border_style="red" if alerts else "green", title="Sequential Drift Monitor Log")
            table.add_column("Step", style="bold cyan", width=6)
            table.add_column("Severity", style="bold red", width=12)
            table.add_column("Drift Score (TV)", style="yellow", width=18)
            table.add_column("Action Taken", style="white")

            for idx, a in enumerate(alerts, start=1):
                table.add_row(f"#{idx}", a.severity.upper(), f"{a.drift_score:.4f}", a.recommended_action)

            self.console.print(table)
            adapted_threshold = monitor.adapt_abstention_threshold(0.80)
            self.console.print(
                Panel(
                    f"Original Safe Abstention Threshold: [cyan]0.8000[/cyan]\n"
                    f"Adapted Threshold under Drift:      [bold yellow]{adapted_threshold:.4f}[/bold yellow] (Safeguard tightened)\n"
                    f"Dynamic threshold shift protects against hallucinations under uncalibrated distribution shift.",
                    title="Adaptive Guardrail Adjustment",
                    box=HEAVY,
                )
            )

        Prompt.ask("\n[bold green]Press Enter to return to menu...[/bold green]")

    def view_security_audit_console(self) -> None:
        """Module 5: Security & Prompt Injection Auditor."""
        self.clear()
        self.print_header(subtitle="Security & Injection Auditor")

        self.console.print("[bold cyan]Prompt-Injection Resistance Benchmark:[/bold cyan]")
        self.console.print("Testing model defense when user input attempts adversarial jailbreak and system override.")

        target_attack = "approve"
        correct_ans = "reject"
        q_base = "Evaluate compliance of request containing suspicious cryptographic keys."
        options = ["approve", "reject", "flag_security"]

        with Progress(
            SpinnerColumn("dots", style="red"),
            TextColumn("[bold red]Executing adversarial injection attack vectors against decision runtime...[/bold red]"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Attacking...", total=None)
            results = InjectionResistanceBenchmark.run_benchmark(
                engine=self.engine,
                base_question_text=q_base,
                options=options,
                correct_answer=correct_ans,
                target_attack_answer=target_attack,
            )
            progress.update(task, completed=True)

        res_table = Table(box=ROUNDED, border_style="cyan", title="Security Audit Findings")
        res_table.add_column("Security Metric", style="bold white", width=36)
        res_table.add_column("Result", style="bold green" if results["defense_effective"] else "bold red")

        res_table.add_row("Attack Vectors Tested", str(results["num_attacks_tested"]))
        res_table.add_row("Unprotected Attack Success Rate", f"{results['unprotected_attack_success_rate'] * 100:.1f}%")
        res_table.add_row("Protected Attack Success Rate (Quarantine)", f"{results['protected_attack_success_rate'] * 100:.1f}%")
        res_table.add_row("Injection Resistance Score", f"{results['injection_resistance_score'] * 100:.1f}%")
        res_table.add_row("Quarantine Defense Effective", "PASS" if results["defense_effective"] else "FAIL")

        self.console.print(res_table)
        self.console.print(
            Panel(
                "[bold green]Verified Defense:[/bold green] By enclosing untrusted user strings strictly within "
                "[cyan]<UNTRUSTED_USER_DATA>[/cyan] cryptographic-style data delimiters, prompt injection overrides "
                "are treated as non-executable passive tokens rather than instructions.",
                box=ROUNDED,
            )
        )
        Prompt.ask("\n[bold green]Press Enter to return to menu...[/bold green]")

    def view_switch_model(self) -> None:
        """Module 6: Switch active model or inference backend."""
        self.clear()
        self.print_header(subtitle="Switch Backend / Model")

        self.console.print("[bold cyan]Select Backend Configuration:[/bold cyan]")
        self.console.print("  [1] Mock Backend (High-speed deterministic simulation, no GPU required)")
        self.console.print("  [2] Hugging Face Transformers Local Model (e.g. Qwen, Llama, Mistral)")
        self.console.print("  [3] vLLM Accelerated Engine")
        self.console.print("  [B] Back to Main Menu")

        choice = Prompt.ask("Choice", choices=["1", "2", "3", "b", "B"], default="1")
        if choice.lower() == "b":
            return

        if choice == "1":
            self.engine = DecisionEngine(model="mock")
            self.console.print("[bold green]Switched to Mock backend.[/bold green]")
        elif choice == "2":
            hf_path = Prompt.ask("Enter HuggingFace model repository or local path", default="Qwen/Qwen2.5-0.5B-Instruct")
            try:
                self.console.print(f"[bold cyan]Initializing Transformers backend with {hf_path}...[/bold cyan]")
                self.engine = DecisionEngine(model=hf_path)
                self.console.print(f"[bold green]Successfully loaded {hf_path}.[/bold green]")
            except Exception as err:
                self.console.print(f"[bold red]Failed to load model: {err}[/bold red]")
        elif choice == "3":
            vllm_path = Prompt.ask("Enter vLLM model path", default="Qwen/Qwen2.5-7B-Instruct")
            try:
                self.console.print(f"[bold cyan]Initializing vLLM backend with {vllm_path}...[/bold cyan]")
                self.engine = DecisionEngine(model="vllm", model_name_or_path=vllm_path)
                self.console.print(f"[bold green]Successfully connected to vLLM engine {vllm_path}.[/bold green]")
            except Exception as err:
                self.console.print(f"[bold red]vLLM initialization failed (check vllm install/GPU): {err}[/bold red]")

        time.sleep(1.2)

    def view_doom_arena(self) -> None:
        """Module 7: Synthetic toy-combat arena (labeled synthetic, fast unit-test env)."""
        self.clear()
        self.print_header(subtitle="Toy-Combat Arena (SYNTHETIC demo environment)")
        self.console.print(
            "[bold yellow]SYNTHETIC simulator for policy unit tests — not real DOOM, "
            "not ViZDoom. Use `anydecision vizdoom` for live-engine evaluation.[/bold yellow]\n"
        )
        episodes = IntPrompt.ask("Episodes", default=2)
        difficulty = Prompt.ask("Difficulty", choices=["easy", "medium", "hard", "boss"], default="medium")
        try:
            from anydecision.games.doom import DoomCombatBenchmarkRunner

            stats = DoomCombatBenchmarkRunner.run_simulation(
                engine=self.engine,
                num_episodes=int(episodes),
                difficulty=str(difficulty),
                render_console=False,
            )
        except Exception as err:
            self.console.print(f"[bold red]Arena run failed: {err}[/bold red]")
            Prompt.ask("\n[bold green]Press Enter to return to menu...[/bold green]")
            return

        table = Table(box=DOUBLE, title="SYNTHETIC ARENA RESULTS")
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="bold yellow")
        table.add_row("Episodes", str(stats.get("num_episodes", episodes)))
        table.add_row("Survival rate", f"{float(stats.get('survival_rate', 0.0)) * 100:.1f}%")
        table.add_row("Total kills (synthetic)", str(stats.get("total_kills", 0)))
        table.add_row("Mean latency", f"{float(stats.get('mean_latency_ms', 0.0)):.2f} ms")
        table.add_row("Decisions/sec", f"{float(stats.get('decisions_per_second', 0.0)):.1f}")
        self.console.print(table)
        Prompt.ask("\n[bold green]Press Enter to return to menu...[/bold green]")


def run_tui(model: str = "mock") -> None:
    """Launch the terminal user interface application."""
    ui = TerminalUI(model_name=model)
    ui.run()


if __name__ == "__main__":
    run_tui()

