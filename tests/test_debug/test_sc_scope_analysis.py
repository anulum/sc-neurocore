# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC scope analysis edge tests

"""Contracts for live analyzer windows and layer error budgets."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.debug import sc_scope as scope_module
from sc_neurocore.debug.sc_scope import (
    LayerErrorBudget,
    LiveAnalyzer,
    _compute_scc_python,
    compute_scc,
)
from tests.test_debug.sc_scope_edges_support import _sample


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
