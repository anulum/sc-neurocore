# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused tests for delayed-recall alternative memory routes

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.experimental import (
    AlternativePathConfig,
    AlternativePathMode,
    make_delayed_recall_shared_state_route,
)


def _compact_cues() -> npt.NDArray[np.float64]:
    return np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_delayed_recall_route_accepts_compact_binary_matrix() -> None:
    route = make_delayed_recall_shared_state_route()

    result = route.run(
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
        2,
        cues=_compact_cues(),
        seed_count=1,
        shared_state_dim=2,
    )

    assert result.returned_path == "shadow-baseline"
    assert result.baseline_value is not None
    assert result.candidate_value is not None
    assert result.baseline_value["cue_count"] == 2
    assert result.candidate_value["shared_state_dim"] == 2


def test_delayed_recall_route_rejects_negative_delay() -> None:
    route = make_delayed_recall_shared_state_route()

    with pytest.raises(ValueError, match="delay_steps must be non-negative"):
        route.run(AlternativePathConfig(), -1, cues=_compact_cues(), seed_count=1)


def test_delayed_recall_route_rejects_non_positive_seed_count() -> None:
    route = make_delayed_recall_shared_state_route()

    with pytest.raises(ValueError, match="seed_count must be positive"):
        route.run(AlternativePathConfig(), 1, cues=_compact_cues(), seed_count=0)


def test_delayed_recall_route_rejects_one_dimensional_cues() -> None:
    route = make_delayed_recall_shared_state_route()

    with pytest.raises(ValueError, match="cues must be a finite 2D matrix"):
        route.run(
            AlternativePathConfig(),
            1,
            cues=np.array([1.0, 0.0, 1.0], dtype=np.float64),
            seed_count=1,
        )


def test_delayed_recall_route_rejects_empty_cues() -> None:
    route = make_delayed_recall_shared_state_route()

    with pytest.raises(ValueError, match="cues must contain at least one"):
        route.run(
            AlternativePathConfig(),
            1,
            cues=np.empty((0, 3), dtype=np.float64),
            seed_count=1,
        )


def test_delayed_recall_route_rejects_nonfinite_cues() -> None:
    route = make_delayed_recall_shared_state_route()

    with pytest.raises(ValueError, match="cues must be finite"):
        route.run(
            AlternativePathConfig(),
            1,
            cues=np.array([[1.0, np.nan, 0.0]], dtype=np.float64),
            seed_count=1,
        )


def test_delayed_recall_route_rejects_nonbinary_cues() -> None:
    route = make_delayed_recall_shared_state_route()

    with pytest.raises(ValueError, match="cues must contain only binary"):
        route.run(
            AlternativePathConfig(),
            1,
            cues=np.array([[1.0, 0.25, 0.0]], dtype=np.float64),
            seed_count=1,
        )


def test_delayed_recall_candidate_rejects_invalid_shared_state_dim() -> None:
    route = make_delayed_recall_shared_state_route()

    with pytest.raises(ValueError, match="shared_state_dim must be positive"):
        route.run(
            AlternativePathConfig(
                enabled=True,
                mode=AlternativePathMode.CANDIDATE,
                fail_open=False,
            ),
            1,
            cues=_compact_cues(),
            seed_count=1,
            shared_state_dim=0,
        )


def test_delayed_recall_comparator_reports_delay_mismatch() -> None:
    route = make_delayed_recall_shared_state_route()
    baseline = route.baseline(2, cues=_compact_cues(), seed_count=1)
    candidate = route.candidate(3, cues=_compact_cues(), seed_count=1, shared_state_dim=2)

    stats = route.comparator(baseline, candidate, AlternativePathConfig())

    assert not stats.matched
    assert stats.comparable_leaf_count == 0
    assert "delay mismatch" in stats.detail


def test_delayed_recall_comparator_enforces_long_delay_gain() -> None:
    route = make_delayed_recall_shared_state_route()

    stats = route.comparator(
        {"delay_steps": 16, "mean_accuracy": 0.7},
        {"delay_steps": 16, "mean_accuracy": 0.75},
        AlternativePathConfig(),
    )

    assert not stats.matched
    assert "failed to improve recall enough" in stats.detail
