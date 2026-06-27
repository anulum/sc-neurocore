# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""Module-specific tests for the quantum cognition terminal dashboard."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from os import terminal_size
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from sc_neurocore.quantum_cognition import dashboard
from sc_neurocore.quantum_cognition.dashboard import TerminalDashboard

if TYPE_CHECKING:
    from sc_neurocore.quantum_cognition.gotm_brain import GOTMBrain


@dataclass
class FakeNeuron:
    """Minimal neuron telemetry record consumed by the dashboard renderer."""

    atp_level: float


class FakePool:
    """Minimal spin-pool telemetry record consumed by the dashboard renderer."""

    def __init__(self, entanglement_map: list[float]) -> None:
        """Store a deterministic entanglement map for dashboard rendering."""
        self.entanglement_map = np.asarray(entanglement_map, dtype=np.float64)


class FakeBrain:
    """Small dashboard-compatible brain fixture exercising the public draw path."""

    def __init__(self, history: list[dict[str, Any]] | None = None) -> None:
        """Initialise deterministic dashboard telemetry."""
        self.pool = FakePool([0.0, 0.0, 0.0, 0.0])
        self.neurons = [
            FakeNeuron(0.2),
            FakeNeuron(0.5),
            FakeNeuron(0.8),
            FakeNeuron(1.0),
            FakeNeuron(0.1),
        ]
        self._history = [] if history is None else history

    def get_learning_state(self) -> dict[str, Any]:
        """Return the state dictionary read by the dashboard header."""
        return {
            "n_neurons": len(self.neurons),
            "total_steps": len(self._history),
            "total_spikes": sum(int(item.get("n_spikes", 0)) for item in self._history),
            "bridge_backend": "emulated",
            "has_llm": False,
            "total_metabolic_failures": 2,
            "avg_entanglement": 0.0,
            "avg_atp": 0.52,
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Return spike and directive history entries for raster rendering."""
        return self._history


def test_dashboard_draw_covers_terminal_fallback_and_history_bands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Draw uses fallback terminal size and renders all spike-raster bands."""

    def fail_terminal_size(_fallback: tuple[int, int]) -> terminal_size:
        """Raise the terminal-size error handled by the renderer."""
        raise ValueError("terminal size unavailable")

    history = [
        {"n_spikes": 0, "directive": "FOCUS"},
        {"n_spikes": 2, "directive": "EXPLORE"},
        {"n_spikes": 5, "directive": "STABILIZE"},
        {"n_spikes": 9, "directive": "UNKNOWN"},
    ]
    monkeypatch.setattr(shutil, "get_terminal_size", fail_terminal_size)

    TerminalDashboard(max_raster_steps=4, clear_screen=True).draw(
        cast("GOTMBrain", FakeBrain(history))
    )

    output = capsys.readouterr().out
    assert "GOTM Quantum Cognition Brain" in output
    assert "Spike Raster" in output
    assert "Directive History" in output
    assert dashboard._CLEAR_SCREEN in output
    assert dashboard._RED in output
    assert dashboard._YELLOW in output
    assert dashboard._GREEN in output
    assert dashboard._DIM in output


def test_dashboard_draw_reports_hidden_neuron_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Draw reports hidden neuron counts when the terminal is narrow."""
    monkeypatch.setattr(shutil, "get_terminal_size", lambda _fallback: terminal_size((9, 24)))

    TerminalDashboard(clear_screen=False).draw(cast("GOTMBrain", FakeBrain()))

    output = capsys.readouterr().out
    assert "(+2 more)" in output
    assert dashboard._CLEAR_SCREEN not in output


def test_bar_handles_non_positive_scale() -> None:
    """Bar renderer returns an empty-scale bar for non-positive maxima."""
    assert dashboard._bar(1.0, 0.0, width=4) == "░░░░"


def test_dashboard_repr_reports_raster_window() -> None:
    """The dashboard repr includes the configured raster window."""
    assert repr(TerminalDashboard(max_raster_steps=7)) == "TerminalDashboard(max_raster=7)"
