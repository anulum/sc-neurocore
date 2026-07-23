# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustParity from former test_spike_stats_sorting_quality.py

"""Focused suite: TestRustParity from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403

@pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust engine not built")
class TestRustParity:
    def test_parity_across_sizes(self) -> None:
        for nc, nn, d in [(20, 30, 3), (15, 25, 2), (10, 20, 1), (40, 60, 5)]:
            cluster, noise = _cluster_noise(nc, nn, d)
            for fn in (isolation_distance, l_ratio):
                py = fn(cluster, noise, backend="python")
                ru = fn(cluster, noise, backend="rust")
                npt.assert_allclose(ru, py, atol=1e-7)

    def test_auto_selects_rust(self) -> None:
        cluster, noise = _cluster_noise(20, 30, 3)
        npt.assert_array_equal(
            isolation_distance(cluster, noise, backend="auto"),
            isolation_distance(cluster, noise, backend="rust"),
        )
