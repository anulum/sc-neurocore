# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAnalyzeBank from former test_crosstalk.py

"""Focused suite: TestAnalyzeBank from former test_crosstalk.py."""

from __future__ import annotations

from crosstalk_support import *  # noqa: F403


class TestAnalyzeBank:
    @pytest.fixture
    def model(self) -> CrosstalkModel:
        return CrosstalkModel()

    def test_requires_at_least_one_waveguide(self, model):
        with pytest.raises(ValueError):
            model.analyze_bank(waveguides=0, gap_nm=200.0, coupling_length_um=10.0)

    def test_pair_counts_match_bank_size(self, model):
        r = model.analyze_bank(waveguides=5, gap_nm=200.0, coupling_length_um=10.0)
        assert r["num_near_pairs"] == 4  # N-1 adjacent
        assert r["num_far_pairs"] == 3  # N-2 next-nearest

    def test_single_waveguide_has_no_pairs(self, model):
        r = model.analyze_bank(waveguides=1, gap_nm=200.0, coupling_length_um=10.0)
        assert r["num_near_pairs"] == 0
        assert r["num_far_pairs"] == 0
        assert math.isinf(r["worst_isolation_db"])

    def test_isolation_grows_with_gap(self, model):
        narrow = model.analyze_bank(waveguides=4, gap_nm=100.0, coupling_length_um=10.0)
        wide = model.analyze_bank(waveguides=4, gap_nm=400.0, coupling_length_um=10.0)
        assert wide["worst_isolation_db"] > narrow["worst_isolation_db"]

    def test_crosstalk_safe_flag_at_20_db_threshold(self, model):
        r = model.analyze_bank(waveguides=8, gap_nm=600.0, coupling_length_um=10.0)
        assert (r["worst_isolation_db"] > 20.0) == r["crosstalk_safe"]

    def test_near_pair_couples_more_than_far_pair(self, model):
        r = model.analyze_bank(waveguides=4, gap_nm=200.0, coupling_length_um=10.0)
        assert r["adjacent_coupling_ratio"] >= r["next_nearest_coupling_ratio"]

    def test_python_fallback_bank_preserves_physical_accounting(self, monkeypatch):
        import sc_neurocore.optics.photonic_emitter as mod

        monkeypatch.setattr(mod, "_HAS_RUST_PH", False)
        r = mod.CrosstalkModel().analyze_bank(
            waveguides=6,
            gap_nm=250.0,
            coupling_length_um=12.0,
            wavelength_nm=1550.0,
            core_index=3.48,
            cladding_index=1.45,
        )

        near = mod.WaveguidePair(gap_nm=250.0, coupling_length_um=12.0)
        far = mod.WaveguidePair(gap_nm=500.0, coupling_length_um=12.0)
        expected_mean = (5 * near.coupling_ratio + 4 * far.coupling_ratio) / 9

        assert r["backend"] == "python"
        assert r["num_near_pairs"] == 5
        assert r["num_far_pairs"] == 4
        assert r["worst_isolation_db"] == pytest.approx(min(near.isolation_db, far.isolation_db))
        assert r["mean_coupling_ratio"] == pytest.approx(expected_mean)
        assert r["max_coupling_ratio"] == pytest.approx(
            max(near.coupling_ratio, far.coupling_ratio)
        )
