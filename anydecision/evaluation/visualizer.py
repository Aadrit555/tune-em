"""Research visualization generator for calibration, reliability, and risk-coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anydecision.calibration.report import CalibrationReport


class ResearchVisualizer:
    """Generates publication-quality research figures for calibration and selective prediction."""

    @staticmethod
    def plot_reliability_diagram(
        report: CalibrationReport,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Reliability Diagram (Calibration Curve)",
    ) -> Any:
        """Plot empirical accuracy vs confidence bins with ECE gap."""
        bins = report.reliability_diagram
        if not bins:
            return None

        fig, ax = plt.subplots(figsize=(6, 6))

        # Identity diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

        confidences = [b["confidence"] for b in bins]
        accuracies = [b["accuracy"] for b in bins]
        counts = [b["count"] for b in bins]
        total_counts = sum(counts)

        # Bar plot
        bin_centers = [(b["range"][0] + b["range"][1]) / 2.0 for b in bins]
        width = 1.0 / len(bins)

        ax.bar(
            bin_centers,
            accuracies,
            width=width * 0.85,
            alpha=0.65,
            color="#2563eb",
            edgecolor="#1d4ed8",
            label="Empirical Accuracy",
        )

        ax.plot(
            bin_centers,
            confidences,
            marker="o",
            color="#dc2626",
            linewidth=2,
            label="Mean Confidence",
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Confidence", fontsize=11)
        ax.set_ylabel("Empirical Accuracy", fontsize=11)
        ax.set_title(f"{title}\nECE = {report.ece:.4f} | Brier = {report.brier_score:.4f}", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=200)
            plt.close(fig)
            return p
        return fig

    @staticmethod
    def plot_risk_coverage(
        report: CalibrationReport,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Risk vs. Coverage (Selective Prediction)",
    ) -> Any:
        """Plot selective risk vs coverage trade-off curve."""
        curve = report.risk_coverage_curve
        if not curve:
            return None

        fig, ax = plt.subplots(figsize=(6, 4.5))

        coverages = [pt["coverage"] * 100 for pt in curve]
        risks = [pt["risk"] * 100 for pt in curve]

        ax.plot(coverages, risks, marker="s", color="#059669", linewidth=2.2, label="Selective Risk Curve")
        ax.axhline(5.0, color="#dc2626", linestyle="--", label="Target Error (5%)")

        ax.set_xlabel("Coverage (%)", fontsize=11)
        ax.set_ylabel("Risk / Error Rate (%)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=200)
            plt.close(fig)
            return p
        return fig

