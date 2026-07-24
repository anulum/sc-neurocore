# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGoParity from former test_spike_stats_sorting_quality.py

"""Focused suite: TestGoParity from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403


@pytest.mark.skipif(not _GO_AVAILABLE, reason="Go sorting-quality library not built")
class TestGoParity:
    def test_parity(self) -> None:
        for nc, nn, d in [(20, 30, 3), (15, 25, 2), (10, 20, 1)]:
            cluster, noise = _cluster_noise(nc, nn, d)
            for fn in (isolation_distance, l_ratio):
                py = fn(cluster, noise, backend="python")
                go = fn(cluster, noise, backend="go")
                npt.assert_allclose(go, py, atol=1e-7)

    def test_ensure_go_is_cached(self) -> None:
        assert _SQ._ensure_go_sq() is True
        assert _SQ._ensure_go_sq() is True
