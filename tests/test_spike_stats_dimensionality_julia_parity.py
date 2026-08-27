# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestJuliaParity from former test_spike_stats_dimensionality.py

"""Focused suite: TestJuliaParity from former test_spike_stats_dimensionality.py."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403
from tests.julia_requirement import require_julia

require_julia()


class TestJuliaParity:
    def test_parity(self) -> None:
        _parity("julia")

    def test_ensure_cached(self) -> None:
        assert _DIM._ensure_julia_dim() is True
        assert _DIM._ensure_julia_dim() is True
