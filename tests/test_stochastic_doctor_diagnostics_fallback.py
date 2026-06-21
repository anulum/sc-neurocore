# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for stochastic-doctor diagnostics pure-Python fallbacks

"""Contracts for the stochastic-doctor SCC and the pure-Python diagnostic fallbacks."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.stochastic_doctor import diagnostics as diag
from sc_neurocore.stochastic_doctor.diagnostics import StochasticDoctor, compute_scc


@pytest.fixture
def force_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the Rust backend so the pure-Python diagnostic paths execute."""
    monkeypatch.setattr(diag, "_HAS_PYO3", False)
    monkeypatch.setattr(diag, "_sdc_rust", None)


def test_scc_python_distinguishes_correlation_sign() -> None:
    """_scc_python yields +1 for identical and -1 for anti-correlated bitstreams."""
    a = np.array([1, 1, 0, 0], dtype=np.uint8)

    assert diag._scc_python(a, a) == pytest.approx(1.0)
    assert diag._scc_python(a, 1 - a) == pytest.approx(-1.0)


def test_scc_python_returns_zero_for_independent_streams() -> None:
    """A vanishing numerator short-circuits to zero correlation."""
    a = np.array([1, 0, 1, 0], dtype=np.uint8)
    b = np.array([1, 1, 0, 0], dtype=np.uint8)

    assert diag._scc_python(a, b) == pytest.approx(0.0)


def test_compute_scc_uses_python_fallback(force_python: None) -> None:
    """compute_scc falls back to the pure-Python SCC when Rust is unavailable."""
    a = np.array([1, 0, 1, 0], dtype=np.uint8)

    assert compute_scc(a, a) == pytest.approx(1.0)


def test_estimate_precision_python_fallback(force_python: None) -> None:
    """estimate_precision computes probability and variance, including the empty case."""
    doctor = StochasticDoctor()

    p, variance = doctor.estimate_precision(np.array([1, 1, 0, 0], dtype=np.uint8))
    assert p == pytest.approx(0.5)
    assert variance == pytest.approx(0.5 * 0.5 / 4)

    assert doctor.estimate_precision(np.array([], dtype=np.uint8)) == (0.0, 0.0)


def test_compute_histogram_python_fallback(force_python: None) -> None:
    """compute_histogram bins per-word popcounts in pure Python."""
    doctor = StochasticDoctor()

    hist = doctor.compute_histogram(np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8), word_size=4)

    assert int(hist.sum()) == 2
    assert int(hist[3]) == 1
    assert int(hist[0]) == 1
