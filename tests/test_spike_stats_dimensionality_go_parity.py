# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGoParity from former test_spike_stats_dimensionality.py

"""Focused suite: TestGoParity from former test_spike_stats_dimensionality.py."""

from __future__ import annotations

import pytest
from tests.spike_stats_dimensionality_support import (
    _DIM,
    _GO_AVAILABLE,
    _parity,
)


@pytest.mark.skipif(not _GO_AVAILABLE, reason="Go dimensionality library not built")
class TestGoParity:
    """The Go dimensionality backend against the Python reference."""

    def test_parity(self) -> None:
        """PCA, demixed PCA and factor analysis on Go match the Python reference within 1e-6."""
        _parity("go")

    def test_ensure_cached(self) -> None:
        """The Go backend probe returns True on two successive calls."""
        assert _DIM._ensure_go_dim() is True
        assert _DIM._ensure_go_dim() is True
