# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAnalyzePairs from former test_crosstalk.py

"""Focused suite: TestAnalyzePairs from former test_crosstalk.py."""

from __future__ import annotations

from crosstalk_support import *  # noqa: F403

class TestAnalyzePairs:
    @pytest.fixture
    def model(self) -> CrosstalkModel:
        return CrosstalkModel()

    def test_length_mismatch_rejected(self, model):
        with pytest.raises(ValueError):
            model.analyze_pairs(
                pair_indices=[(0, 1), (1, 2)],
                gaps_nm=[200.0],  # mismatched
                coupling_lengths_um=[10.0, 10.0],
            )

    def test_returns_one_entry_per_input_pair(self, model):
        r = model.analyze_pairs(
            pair_indices=[(0, 1), (1, 2), (0, 2)],
            gaps_nm=[200.0, 400.0, 800.0],
            coupling_lengths_um=[10.0, 10.0, 10.0],
        )
        assert r["num_pairs"] == 3
        assert len(r["isolation_db"]) == 3

    def test_isolation_ordering_by_gap(self, model):
        r = model.analyze_pairs(
            pair_indices=[(0, 1), (1, 2), (0, 2)],
            gaps_nm=[200.0, 400.0, 800.0],
            coupling_lengths_um=[10.0, 10.0, 10.0],
        )
        iso = r["isolation_db"]
        assert iso[0] <= iso[1] <= iso[2]

    def test_empty_input_returns_zero_pairs(self, model):
        r = model.analyze_pairs(pair_indices=[], gaps_nm=[], coupling_lengths_um=[])
        assert r["num_pairs"] == 0

    def test_python_fallback_pairs_preserves_per_pair_geometry(self, monkeypatch):
        import sc_neurocore.optics.photonic_emitter as mod

        monkeypatch.setattr(mod, "_HAS_RUST_PH", False)
        pair_indices = [(0, 1), (1, 3), (2, 4)]
        gaps = [180.0, 260.0, 520.0]
        lengths = [8.0, 12.0, 16.0]

        r = mod.CrosstalkModel().analyze_pairs(pair_indices, gaps, lengths)

        expected = [
            mod.WaveguidePair(gap_nm=gap, coupling_length_um=length)
            for gap, length in zip(gaps, lengths)
        ]
        assert r["backend"] == "python"
        assert r["pair_a"] == [0, 1, 2]
        assert r["pair_b"] == [1, 3, 4]
        assert r["gap_nm"] == gaps
        assert r["coupling_length_um"] == lengths
        assert r["num_pairs"] == len(pair_indices)
        np.testing.assert_allclose(
            r["coupling_coefficient_per_um"],
            [pair.coupling_coefficient for pair in expected],
        )
        np.testing.assert_allclose(r["coupling_ratio"], [pair.coupling_ratio for pair in expected])
        np.testing.assert_allclose(r["isolation_db"], [pair.isolation_db for pair in expected])
