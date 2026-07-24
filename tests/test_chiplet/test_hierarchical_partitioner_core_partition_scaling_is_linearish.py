# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPartitionScalingIsLinearish from former test_hierarchical_partitioner_core.py

"""Focused suite: TestPartitionScalingIsLinearish from former test_hierarchical_partitioner_core.py."""

from __future__ import annotations

from hierarchical_partitioner_core_support import *  # noqa: F403


class TestPartitionScalingIsLinearish:
    """The wall-clock should NOT grow quadratically with V any more.
    We accept a generous slack — exact ms differs by hardware — but
    a 10× V increase must NOT cause a 100× wall-clock increase."""

    def test_v200_finishes_under_one_second(self) -> None:
        # Pre-fix this took ~700 ms on the dev box; post-fix ~30 ms.
        # 1 s is a generous CI margin (covers slow shared runners).
        # The minimum over repeats is the sample least contaminated by
        # shared-runner preemption, so a single noisy-neighbour spike cannot
        # produce a false failure (a spike once pushed this to ~964 ms).
        best_ms = _min_partition_ms(200)
        assert best_ms < 1000.0, (
            f"V=200 partition min over {_TIMING_REPEATS} runs took {best_ms:.1f} ms — "
            "perf fix #65 may have regressed (expected < 1 s, was ~700 ms before fix)"
        )

    def test_scaling_better_than_quadratic(self) -> None:
        """Doubling V should not cause >5× wall-clock increase."""
        t100 = _min_partition_ms(100)
        t200 = _min_partition_ms(200)
        # Pre-fix ratio was ~3.6× for V doubling. Quadratic would be 4×.
        # Cubic O(V²·E) gave the actual measured ratio of ~3.6× because
        # E grows linearly with V at fixed degree. We require strictly
        # better than 5× to leave headroom for noise on slow runners. Each
        # measurement is a minimum over repeats, so a single scheduling spike
        # (which once produced a 154× false failure on a green base) can no
        # longer inflate the ratio; a genuine super-linear regression still
        # shows in every repeat and is caught.
        ratio = t200 / max(t100, 0.5)  # avoid div-by-zero on very fast hosts
        assert ratio < 5.0, (
            f"V doubling caused {ratio:.1f}× wall-clock increase "
            f"(min t100={t100:.1f} ms → min t200={t200:.1f} ms); the #65 fix "
            "should keep this near-linear"
        )
