# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC bitstream scope transport, SCC fallback and rendering edges

"""Contracts for SC scope transport fallbacks, the Python SCC and renderer edges."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.debug import sc_scope as scope_module
from sc_neurocore.debug.sc_scope import (
    BitstreamSample,
    LayerErrorBudget,
    LiveAnalyzer,
    ScopeRenderer,
    ScopeSession,
    TransportBackend,
    TransportConfig,
    TransportType,
    _compute_scc_python,
    compute_scc,
)


def _sample(layer_id: int) -> BitstreamSample:
    """A small single-word bitstream sample for a given layer."""
    return BitstreamSample(
        timestamp_ns=0,
        layer_id=layer_id,
        neuron_id=0,
        words=np.array([0x0F0F0F0F], dtype=np.uint32),
        sample_index=0,
    )


def test_hardware_transport_connects_and_reads_none() -> None:
    """A non-simulated transport connects optimistically but yields no bitstream data."""
    backend = TransportBackend(TransportConfig(TransportType.JTAG))

    assert backend.connect() is True
    assert backend.is_connected is True
    assert backend.read_bitstream(8) is None


def test_disconnected_transport_reads_none() -> None:
    """A transport that has not connected returns no data."""
    backend = TransportBackend(TransportConfig(TransportType.SIMULATED))

    assert backend.read_bitstream(8) is None


@pytest.fixture
def python_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the pure-Python SCC by disabling the Rust extension dispatch."""
    monkeypatch.setattr(scope_module, "_HAS_RUST_SCC", False)


def test_python_scc_zero_for_uncorrelated_streams(python_scc: None) -> None:
    """Two empty streams have a zero numerator and report no correlation."""
    zeros = np.zeros(2, dtype=np.uint32)

    assert compute_scc(zeros, zeros) == 0.0


def test_python_scc_positive_for_identical_streams(python_scc: None) -> None:
    """Identical half-dense streams are maximally positively correlated."""
    stream = np.array([0x0000FFFF, 0xFFFF0000], dtype=np.uint32)

    assert compute_scc(stream, stream) == pytest.approx(1.0)


def test_python_scc_negative_for_disjoint_streams(python_scc: None) -> None:
    """Streams with disjoint active bits are maximally negatively correlated."""
    a = np.array([0x0000FFFF, 0x00000000], dtype=np.uint32)
    b = np.array([0xFFFF0000, 0x00000000], dtype=np.uint32)

    assert compute_scc(a, b) == pytest.approx(-1.0)


def test_compute_scc_handles_odd_length_arrays() -> None:
    """The dispatcher accepts odd-length word arrays (padded for the u64 kernel)."""
    a = np.array([0x0000FFFF], dtype=np.uint32)
    b = np.array([0x0000FFFF], dtype=np.uint32)

    result = compute_scc(a, b)

    assert -1.0 <= result <= 1.0
    assert result == pytest.approx(_compute_scc_python(a, b))


def test_live_analyzer_creates_window_for_unknown_layer() -> None:
    """Ingesting a sample for an unregistered layer creates its analysis window."""
    analyzer = LiveAnalyzer(num_layers=1)

    analyzer.ingest(_sample(layer_id=5))

    assert 5 in analyzer.windows
    assert analyzer.total_samples == 1


def test_layer_error_budget_reports_defaults_when_empty() -> None:
    """An error budget with no history reports zero error and a perfect pass rate."""
    budget = LayerErrorBudget(layer_id=0, expected_density=0.5)

    assert budget.current_error == 0.0
    assert budget.mean_error == 0.0
    assert budget.max_error == 0.0
    assert budget.pass_rate == 1.0


def test_layer_error_budget_tracks_history() -> None:
    """After recording densities the budget reports current, mean and max error."""
    budget = LayerErrorBudget(layer_id=0, expected_density=0.5, tolerance=0.05)

    assert budget.check(0.52) is True
    assert budget.check(0.70) is False

    assert budget.current_error == pytest.approx(0.20)
    assert budget.mean_error == pytest.approx(0.11)
    assert budget.max_error == pytest.approx(0.20)
    assert budget.pass_rate == pytest.approx(0.5)


def _session() -> ScopeSession:
    backend = TransportBackend(TransportConfig(TransportType.SIMULATED))
    return ScopeSession(transport=backend, analyzer=LiveAnalyzer(num_layers=2))


def test_scope_session_start_fails_when_transport_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport that refuses to connect makes the session fail to start."""
    session = _session()
    monkeypatch.setattr(session.transport, "connect", lambda: False)

    assert session.start() is False
    assert session.is_running is False


def test_capture_one_returns_none_when_transport_yields_no_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A running session returns no sample when the transport produces no words."""
    session = _session()
    assert session.start() is True
    monkeypatch.setattr(session.transport, "read_bitstream", lambda *_a, **_k: None)

    assert session.capture_one(layer_id=0) is None


def test_render_layer_summary_reports_no_data_for_empty_stats() -> None:
    """An empty stats mapping renders an explicit no-data row."""
    assert ScopeRenderer.render_layer_summary(3, {}) == "  L3: (no data)"


def test_render_session_includes_error_budget_section() -> None:
    """A session carrying an error budget renders the error-budget section."""
    session = _session()
    session.add_error_budget(layer_id=0, expected_density=0.4)

    rendered = ScopeRenderer.render_session(session)

    assert "Error Budgets" in rendered
    assert "L0:" in rendered


def test_render_density_bar_fills_proportionally() -> None:
    """The density bar fills a proportional number of cells and prints the value."""
    bar: str = ScopeRenderer.render_density_bar(0.5, width=10)

    assert bar.count("█") == 5
    assert "0.500" in bar


def test_session_status_reports_zero_elapsed_before_start() -> None:
    """Status before any start reports a zero elapsed time."""
    session: ScopeSession = _session()

    status: dict[str, Any] = session.status()

    assert status["running"] is False
    assert status["elapsed_s"] == 0
