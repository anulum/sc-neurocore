# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for photonic crosstalk analysis

"""Multi-angle tests for ``sc_neurocore.optics.photonic_emitter.CrosstalkModel``.

Covers:

- Coupled-mode transfer-matrix unitarity and energy conservation.
- :class:`WaveguidePair` physical invariants (larger gap ⇒ better isolation,
  longer coupler ⇒ worse isolation up to the first half-period).
- :meth:`CrosstalkModel.analyze_bank` uniform-bank analysis, near+far pair
  accounting, ``crosstalk_safe`` threshold.
- :meth:`CrosstalkModel.analyze_pairs` arbitrary-geometry O(N²) path.
- Rust-vs-Python backend parity (requires ``_HAS_RUST_PH``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.optics.photonic_emitter import (
    CrosstalkModel,
    WaveguidePair,
)


# ---------------------------------------------------------------------------
# WaveguidePair physical invariants.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Transfer-matrix unitarity (energy conservation).
# ---------------------------------------------------------------------------


class TestTransferMatrixUnitarity:
    @pytest.fixture
    def model(self) -> CrosstalkModel:
        return CrosstalkModel()

    def test_transfer_matrix_is_unitary(self, model):
        pair = WaveguidePair(gap_nm=150.0, coupling_length_um=25.0)
        t = model.transfer_matrix(pair)
        # T · T† == I for any real κL
        identity = t @ t.conj().T
        assert np.allclose(identity, np.eye(2), atol=1e-12)

    def test_power_conservation_on_single_port_excitation(self, model):
        pair = WaveguidePair(gap_nm=200.0, coupling_length_um=7.0)
        p_a, p_b = model.compute_crosstalk(pair, (1.0, 0.0))
        assert p_a + p_b == pytest.approx(1.0, rel=1e-12)

    def test_power_conservation_on_two_port_excitation(self, model):
        # ``compute_crosstalk`` takes input field amplitudes (the FFI name
        # ``input_power`` is historical); output power sums must match
        # |a|² + |b|² of the input amplitude tuple under unitary evolution.
        pair = WaveguidePair(gap_nm=200.0, coupling_length_um=7.0)
        amp_a, amp_b = 0.6, 0.4
        p_a, p_b = model.compute_crosstalk(pair, (amp_a, amp_b))
        assert p_a + p_b == pytest.approx(amp_a**2 + amp_b**2, rel=1e-12)


# ---------------------------------------------------------------------------
# analyze_bank — uniform-bank path.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# analyze_pairs — per-pair O(N²) path.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Rust-vs-Python backend parity.
# ---------------------------------------------------------------------------


class TestBackendParity:
    """The Rust and Python code paths solve the same coupled-mode equations
    — their outputs must agree to within floating-point tolerance. Any drift
    is a regression in the Rust FFI layer.
    """

    def test_analyze_bank_rust_matches_python(self):
        import sc_neurocore.optics.photonic_emitter as mod

        if not mod._HAS_RUST_PH:
            pytest.skip("Rust photonic bindings not built")
        model_rust = CrosstalkModel()
        r_rust = model_rust.analyze_bank(waveguides=6, gap_nm=250.0, coupling_length_um=12.0)
        assert r_rust["backend"] == "rust"

        # Force Python path via monkey-patched flag.
        import sc_neurocore.optics.photonic_emitter as mod

        orig = mod._HAS_RUST_PH
        try:
            mod._HAS_RUST_PH = False
            r_py = CrosstalkModel().analyze_bank(
                waveguides=6, gap_nm=250.0, coupling_length_um=12.0
            )
        finally:
            mod._HAS_RUST_PH = orig
        assert r_py["backend"] == "python"

        for key in (
            "adjacent_coupling_ratio",
            "adjacent_isolation_db",
            "next_nearest_coupling_ratio",
            "next_nearest_isolation_db",
            "worst_isolation_db",
            "mean_coupling_ratio",
        ):
            assert r_rust[key] == pytest.approx(r_py[key], rel=1e-9, abs=1e-12), (
                f"{key}: rust={r_rust[key]} python={r_py[key]}"
            )

    def test_analyze_pairs_rust_matches_python(self):
        import sc_neurocore.optics.photonic_emitter as mod

        if not mod._HAS_RUST_PH:
            pytest.skip("Rust photonic bindings not built")
        pair_indices = [(0, 1), (1, 2), (0, 2), (3, 4)]
        gaps = [200.0, 300.0, 500.0, 150.0]
        lengths = [10.0, 15.0, 20.0, 5.0]

        r_rust = CrosstalkModel().analyze_pairs(pair_indices, gaps, lengths)
        assert r_rust["backend"] == "rust"

        import sc_neurocore.optics.photonic_emitter as mod

        orig = mod._HAS_RUST_PH
        try:
            mod._HAS_RUST_PH = False
            r_py = CrosstalkModel().analyze_pairs(pair_indices, gaps, lengths)
        finally:
            mod._HAS_RUST_PH = orig

        assert np.allclose(r_rust["isolation_db"], r_py["isolation_db"], rtol=1e-9, atol=1e-12)
        assert np.allclose(r_rust["coupling_ratio"], r_py["coupling_ratio"], rtol=1e-9, atol=1e-12)
