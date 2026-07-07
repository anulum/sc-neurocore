# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic cross-correlation diagnostic tests

"""Contract and Monte-Carlo tests for :mod:`sc_neurocore.core.sc_correlation`.

The SCC estimator is checked against constructed independent/comonotone/
countermonotone streams and, crucially, against the WC-C3 bias formula: the
SCC measured from real bitstreams must predict the observed ``AND`` bias.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.core import (
    CorrelationDiagnostic,
    correlation_diagnostic,
    estimate_scc,
    observed_and_bias,
)
from sc_neurocore.core.sc_error_bounds import multiply_correlation_bias
from sc_neurocore.encoding.encoders import rate_encode


def _shared_source_streams(
    p_a: float, p_b: float, n: int, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Two comonotone streams from one uniform source (SCC == +1)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return (u < p_a).astype(np.uint8), (u < p_b).astype(np.uint8)


# --- estimate_scc ------------------------------------------------------------


def test_scc_independent_streams_near_zero() -> None:
    a = rate_encode(np.full(1, 0.5), T=4000, seed=1)[:, 0].astype(np.uint8)
    b = rate_encode(np.full(1, 0.5), T=4000, seed=2)[:, 0].astype(np.uint8)
    assert abs(estimate_scc(a, b)) < 0.1


def test_scc_comonotone_is_plus_one() -> None:
    a, b = _shared_source_streams(0.6, 0.4, 5000, seed=3)
    assert estimate_scc(a, b) == pytest.approx(1.0, abs=1e-9)


def test_scc_countermonotone_is_minus_one() -> None:
    rng = np.random.default_rng(4)
    u = rng.random(5000)
    a = (u < 0.6).astype(np.uint8)
    b = (u >= 1.0 - 0.4).astype(np.uint8)  # inverted source -> anti-correlated
    assert estimate_scc(a, b) == pytest.approx(-1.0, abs=1e-9)


def test_scc_degenerate_stream_returns_zero() -> None:
    ones = np.ones(100, dtype=np.uint8)
    mixed = np.tile([1, 0], 50).astype(np.uint8)
    assert estimate_scc(ones, mixed) == 0.0


# --- observed bias + WC-C3 consistency --------------------------------------


def test_observed_bias_zero_for_independent() -> None:
    a = rate_encode(np.full(1, 0.6), T=4000, seed=10)[:, 0].astype(np.uint8)
    b = rate_encode(np.full(1, 0.5), T=4000, seed=11)[:, 0].astype(np.uint8)
    assert abs(observed_and_bias(a, b)) < 0.02


def test_scc_predicts_observed_and_bias_comonotone() -> None:
    # The SCC measured from the streams must reproduce the observed AND bias
    # through the WC-C3 multiply_correlation_bias formula.
    p_a, p_b = 0.6, 0.4
    a, b = _shared_source_streams(p_a, p_b, 5000, seed=7)
    scc = estimate_scc(a, b)
    predicted = multiply_correlation_bias(float(np.mean(a)), float(np.mean(b)), scc)
    assert predicted == pytest.approx(observed_and_bias(a, b), abs=1e-9)
    # Comonotone bias equals min(p_a, p_b) - p_a * p_b.
    assert observed_and_bias(a, b) == pytest.approx(min(p_a, p_b) - p_a * p_b, abs=0.02)


# --- correlation_diagnostic --------------------------------------------------


def test_diagnostic_flags_correlated_pair() -> None:
    a, b = _shared_source_streams(0.6, 0.4, 5000, seed=8)
    result = correlation_diagnostic(a, b, bias_threshold=0.01)
    assert isinstance(result, CorrelationDiagnostic)
    assert result.flagged is True
    assert result.scc == pytest.approx(1.0, abs=1e-9)
    assert result.predicted_and_bias == pytest.approx(result.observed_and_bias, abs=1e-9)


def test_diagnostic_passes_independent_pair() -> None:
    a = rate_encode(np.full(1, 0.5), T=5000, seed=20)[:, 0].astype(np.uint8)
    b = rate_encode(np.full(1, 0.5), T=5000, seed=21)[:, 0].astype(np.uint8)
    result = correlation_diagnostic(a, b, bias_threshold=0.05)
    assert result.flagged is False


# --- validation (fail-closed) ------------------------------------------------


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_scc([1, 0, 1], [1, 0])


def test_empty_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_scc([], [])


def test_non_binary_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_scc([0, 2, 1], [0, 1, 1])


def test_non_1d_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_scc([[1, 0], [0, 1]], [[1, 0], [0, 1]])


def test_observed_bias_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        observed_and_bias([1, 0, 1], [1, 0])


def test_diagnostic_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        correlation_diagnostic([1, 0, 1], [1, 0])


def test_diagnostic_negative_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        correlation_diagnostic([1, 0, 1], [0, 1, 1], bias_threshold=-0.1)
