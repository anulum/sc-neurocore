# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWaveguidePairPhysics from former test_crosstalk.py

"""Focused suite: TestWaveguidePairPhysics from former test_crosstalk.py."""

from __future__ import annotations

from crosstalk_support import *  # noqa: F403

class TestWaveguidePairPhysics:
    def test_isolation_grows_with_gap(self):
        narrow = WaveguidePair(gap_nm=100.0, coupling_length_um=10.0)
        wide = WaveguidePair(gap_nm=400.0, coupling_length_um=10.0)
        assert wide.isolation_db > narrow.isolation_db

    def test_coupling_ratio_in_unit_interval(self):
        p = WaveguidePair(gap_nm=200.0, coupling_length_um=10.0)
        assert 0.0 <= p.coupling_ratio <= 1.0

    def test_zero_coupling_length_is_perfect_isolation(self):
        p = WaveguidePair(gap_nm=200.0, coupling_length_um=0.0)
        assert p.coupling_ratio == pytest.approx(0.0, abs=1e-30)
        assert p.isolation_db >= 300.0 - 1.0  # saturates to ceiling

    def test_coupling_coefficient_positive(self):
        p = WaveguidePair(gap_nm=200.0, coupling_length_um=10.0)
        assert p.coupling_coefficient > 0.0

    def test_larger_index_contrast_tighter_mode(self):
        # High contrast (Si/SiO2) should give shorter L_decay and hence
        # smaller Δn_eff at the same gap compared with a low-contrast stack.
        hi = WaveguidePair(gap_nm=300.0, core_index=3.48, cladding_index=1.45)
        lo = WaveguidePair(gap_nm=300.0, core_index=1.55, cladding_index=1.44)
        assert hi.effective_index_diff < lo.effective_index_diff
