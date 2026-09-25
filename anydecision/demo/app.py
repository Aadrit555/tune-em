"""Interactive research demo visualizing L0 -> L1 -> L2 decision pipeline."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import gradio as gr

from anydecision.bias.templates import DEFAULT_TEMPLATES
from anydecision.calibration.temperature import TemperatureScaling
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.core.types import DecisionLevel


def create_demo_interface() -> gr.Blocks:
    """Build Gradio interface for visual probability readout and selective prediction."""
    engine = DecisionEngine(model="mock")
    # Pre-fit a default temperature calibrator so L2 works out-of-the-box
    default_calibrator = TemperatureScaling(temperature=1.45)
    default_calibrator.fitted = True
    engine.calibrator = default_calibrator

    with gr.Blocks(title="anydecision: Typed LLM Decision Runtime") as demo:
        gr.Markdown(
            """
            # 🎯 anydecision: Direct Probability-Based Typed LLM Decision Engine
            Extract decisions directly from **model output distributions** with zero generation,
            option-permutation debiasing (L1), statistical calibration (L2), and selective prediction/abstention.
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                question_input = gr.Textbox(
                    label="Question",
                    value="Should this high-priority customer support escalation be approved?",
                    lines=2,
                )
                choices_input = gr.Textbox(
                    label="Candidate Choices (comma-separated)",
                    value="yes, no, escalate_to_tier3",
                    lines=1,
                )
                context_input = gr.Textbox(
                    label="Optional Context",
                    value="Customer has an Enterprise SLA and is experiencing downtime on EU-West cluster.",
                    lines=2,
                )

                with gr.Accordion("Pipeline & Debiasing Settings", open=True):
                    level_radio = gr.Radio(
                        choices=["L0 (Raw Logits)", "L1 (Zero-Label Debiasing)", "L2 (Calibrated)"],
                        value="L1 (Zero-Label Debiasing)",
                        label="Decision Level",
                    )
                    template_dropdown = gr.Dropdown(
                        choices=DEFAULT_TEMPLATES.list_templates(),
                        value="structured",
                        label="Primary Prompt Template",
                    )
                    num_perms_slider = gr.Slider(
                        minimum=1,
                        maximum=6,
                        step=1,
                        value=3,
                        label="Number of Option Permutations (L1)",
                    )

                with gr.Accordion("Selective Prediction / Abstention Policy", open=True):
                    allow_abstain_cb = gr.Checkbox(value=True, label="Enable Selective Abstention")
                    min_conf_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.50,
                        label="Minimum Confidence Threshold",
                    )
                    target_error_slider = gr.Slider(
                        minimum=0.01,
                        maximum=0.50,
                        step=0.01,
                        value=0.15,
                        label="Target Maximum Posterior Risk",
                    )

                submit_btn = gr.Button("Evaluate Decision", variant="primary")

            with gr.Column(scale=5):
                gr.Markdown("### 📊 Decision & Uncertainty Output")
                decision_badge = gr.Markdown("### Click 'Evaluate Decision' to run.")

                probabilities_plot = gr.Label(
                    num_top_classes=5,
                    label="Probability Distribution over Candidates",
                )

                with gr.Row():
                    confidence_box = gr.Number(label="Confidence", precision=4)
                    uncertainty_box = gr.Number(label="Uncertainty", precision=4)
                    risk_box = gr.Number(label="Posterior Risk", precision=4)

                with gr.Accordion("Observable Research Diagnostics", open=True):
                    diagnostics_json = gr.JSON(label="Diagnostics & Bias Metrics")

                with gr.Accordion("Execution Trace", open=False):
                    trace_json = gr.JSON(label="Step-by-Step Decision Trace")

        def run_inference(
            q_text: str,
            choices_str: str,
            ctx_text: str,
            level_str: str,
            tmpl_name: str,
            num_perms: int,
            allow_abstain: bool,
            min_conf: float,
            target_err: float,
        ) -> tuple[str, Dict[str, float], float, float, float, Dict[str, Any], Dict[str, Any]]:
            choices = [c.strip() for c in choices_str.split(",") if c.strip()]
            q = Question.choice(
                text=q_text,
                choices=choices,
                context=ctx_text if ctx_text.strip() else None,
            )

            # Map level
            if "L0" in level_str:
                lvl = DecisionLevel.L0
            elif "L1" in level_str:
                lvl = DecisionLevel.L1
            else:
                lvl = DecisionLevel.L2

            res = engine.decide(
                q,
                level=lvl,
                min_confidence=min_conf,
                target_error=target_err,
                allow_abstain=allow_abstain,
                templates=[tmpl_name],
                num_permutations=int(num_perms),
                trace=True,
            )

            if res.abstained:
                badge = f"## ⚠️ ABSTAINED\n**Reason:** `{res.reason}` | **Risk:** `{res.risk:.3f}`"
            else:
                badge = f"## ✅ Decision: `{res.answer}`\n**Confidence:** `{res.confidence * 100:.2f}%` | **Level:** `{res.level}`"

            diag_dict = res.diagnostics.model_dump() if res.diagnostics else {}
            trace_dict = res.trace.model_dump() if res.trace else {}

            return (
                badge,
                res.probabilities,
                res.confidence,
                res.uncertainty,
                res.risk,
                diag_dict,
                trace_dict,
            )

        submit_btn.click(
            fn=run_inference,
            inputs=[
                question_input,
                choices_input,
                context_input,
                level_radio,
                template_dropdown,
                num_perms_slider,
                allow_abstain_cb,
                min_conf_slider,
                target_error_slider,
            ],
            outputs=[
                decision_badge,
                probabilities_plot,
                confidence_box,
                uncertainty_box,
                risk_box,
                diagnostics_json,
                trace_json,
            ],
        )

    return demo


def launch_demo(server_port: int = 7860, share: bool = False) -> None:
    demo = create_demo_interface()
    demo.launch(server_port=server_port, share=share)


if __name__ == "__main__":
    launch_demo()
