# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sensitivity analysis tests

"""Tests for adaptive precision sensitivity estimation contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.compiler.sensitivity_analysis import analyze_sensitivity


def test_analyze_sensitivity_returns_reproducible_layer_scores() -> None:
    """Sensitivity analysis returns stable non-negative scores per matrix layer."""

    weights = [
        np.array([[0.2, 0.4], [0.3, 0.5]], dtype=np.float64),
        np.array([[0.1, 0.6, 0.2]], dtype=np.float64),
    ]

    first = analyze_sensitivity(weights, lengths=[16, 32], n_trials=3, seed=7)
    second = analyze_sensitivity(weights, lengths=[16, 32], n_trials=3, seed=7)

    assert first == second
    assert len(first) == 2
    assert all(score >= 0.0 for score in first)


def test_analyze_sensitivity_handles_vector_weights() -> None:
    """Vector weights are treated as a single-output dense layer."""

    scores = analyze_sensitivity(
        [np.array([0.1, 0.4, 0.7], dtype=np.float64)],
        lengths=[8, 16],
        n_trials=2,
        seed=11,
    )

    assert len(scores) == 1
    assert scores[0] >= 0.0


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ([np.array([], dtype=np.float64)], "must not be empty"),
        ([np.array([np.nan], dtype=np.float64)], "must be finite"),
        ([np.zeros((1, 1, 1), dtype=np.float64)], "one-dimensional or two-dimensional"),
    ],
)
def test_analyze_sensitivity_rejects_invalid_weights(
    weights: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]],
    match: str,
) -> None:
    """Sensitivity analysis fails closed on malformed weight arrays."""

    with pytest.raises(ValueError, match=match):
        analyze_sensitivity(weights, lengths=[8, 16], n_trials=1)


@pytest.mark.parametrize(
    ("lengths", "match"),
    [
        ([], "must not be empty"),
        ([0], "positive integers"),
        ([True], "positive integers"),
    ],
)
def test_analyze_sensitivity_rejects_invalid_lengths(
    lengths: list[int],
    match: str,
) -> None:
    """Sensitivity analysis rejects unusable bitstream length ladders."""

    with pytest.raises(ValueError, match=match):
        analyze_sensitivity([np.array([0.25], dtype=np.float64)], lengths=lengths, n_trials=1)


def test_analyze_sensitivity_rejects_non_positive_trial_count() -> None:
    """Sensitivity analysis requires at least one stochastic trial."""

    with pytest.raises(ValueError, match="n_trials"):
        analyze_sensitivity([np.array([0.25], dtype=np.float64)], lengths=[8], n_trials=0)
