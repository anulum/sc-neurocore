# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestJuliaParity from former test_spike_stats_dimensionality.py

"""Focused suite: TestJuliaParity from former test_spike_stats_dimensionality.py."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import (
    _DIM,
    _parity,
)
from tests.julia_requirement import require_julia

require_julia()


class TestJuliaParity:
    """The Julia dimensionality backend against the Python reference."""

    def test_parity(self) -> None:
        """PCA, demixed PCA and factor analysis on Julia match the Python reference within 1e-6."""
        _parity("julia")

    def test_ensure_cached(self) -> None:
        """The Julia backend probe returns True on two successive calls."""
        assert _DIM._ensure_julia_dim() is True
        assert _DIM._ensure_julia_dim() is True
