# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestJuliaParity from former test_spike_stats_sorting_quality.py

"""Focused suite: TestJuliaParity from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

import numpy.testing as npt
from sc_neurocore.analysis.spike_stats.sorting_quality import (
    isolation_distance,
    l_ratio,
)
from tests.spike_stats_sorting_quality_support import (
    _cluster_noise,
    _SQ,
)
from tests.julia_requirement import require_julia

require_julia()


class TestJuliaParity:
    """The Julia sorting-quality backend against the Python reference."""

    def test_parity(self) -> None:
        """Isolation distance and the L-ratio on Julia match the Python reference within 1e-7 for two shapes."""
        for nc, nn, d in [(20, 30, 3), (10, 20, 1)]:
            cluster, noise = _cluster_noise(nc, nn, d)
            for fn in (isolation_distance, l_ratio):
                py = fn(cluster, noise, backend="python")
                ju = fn(cluster, noise, backend="julia")
                npt.assert_allclose(ju, py, atol=1e-7)

    def test_ensure_julia_is_cached(self) -> None:
        """The Julia backend probe returns True on two successive calls."""
        assert _SQ._ensure_julia_sq() is True
        assert _SQ._ensure_julia_sq() is True
