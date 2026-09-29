"""Unit tests for the Terminal-based User Interface (TUI)."""

from __future__ import annotations

from typer.testing import CliRunner

from anydecision.cli.main import app
from anydecision.core.question import Question
from anydecision.tui.app import PRESETS, TerminalUI


def test_tui_initialization_and_header():
    """Test TerminalUI initialization and header rendering."""
    ui = TerminalUI(model_name="mock")
    assert ui.engine is not None
    assert ui.engine.metadata.backend_name == "mock"

    # Capture console output
    with ui.console.capture() as capture:
        ui.print_header(subtitle="Test Subtitle")
    output = capture.get()

    assert "TYPED UNCERTAINTY-AWARE DECISION RUNTIME" in output
    assert "TEST SUBTITLE" in output
    assert "Backend:" in output


def test_tui_render_decision_results():
    """Test rendering rich visual decision cards in terminal."""
    ui = TerminalUI(model_name="mock")
    q = Question.choice("Should access be granted?", ["grant", "deny"])
    dec = ui.engine.decide(q, track_layer_trajectory=True)

    with ui.console.capture() as capture:
        ui.render_decision_results(dec, q)
    output = capture.get()

    assert "DECISION VERDICT" in output
    assert "Probability Distribution" in output
    assert "grant" in output or "deny" in output


def test_tui_preset_utility_matrix_constructs():
    """Preset 'utilities' grids must build a valid UtilityMatrix over question states."""
    from anydecision.theory.utility import UtilityMatrix

    ui = TerminalUI(model_name="mock")
    preset = PRESETS[0]
    raw_grid = preset["utilities"]
    matrix = UtilityMatrix(
        actions=list(raw_grid.keys()),
        states=list(preset["choices"]),
        matrix={a: {s: float(v.get(s, 0.0)) for s in preset["choices"]} for a, v in raw_grid.items()},
    )
    assert set(matrix.states) == set(preset["choices"])
    best, _, _, _ = matrix.select_optimal_action({s: 1.0 / len(preset["choices"]) for s in preset["choices"]})
    assert best in matrix.actions
    assert ui.engine is not None


def test_tui_doom_arena_exists_and_runs_synthetic():
    """Menu option 7 must resolve to a working synthetic arena view."""
    import io
    from unittest.mock import patch

    ui = TerminalUI(model_name="mock")
    assert hasattr(ui, "view_doom_arena")
    with patch("sys.stdin", io.StringIO("1\neasy\n\n")):
        with ui.console.capture() as capture:
            ui.view_doom_arena()
    output = capture.get()
    assert "SYNTHETIC" in output


def test_tui_presets_validity():
    """Verify all predefined research presets are structurally valid."""
    assert len(PRESETS) >= 4
    for p in PRESETS:
        assert "name" in p
        assert "question" in p
        assert len(p["choices"]) >= 2
        q = Question.choice(p["question"], p["choices"])
        assert len(q.options) >= 2


def test_cli_tui_help():
    """Verify anydecision tui --help command works properly."""
    runner = CliRunner()
    result = runner.invoke(app, ["tui", "--help"])
    assert result.exit_code == 0
    assert "Launch the interactive Linux terminal-based UI" in result.output




