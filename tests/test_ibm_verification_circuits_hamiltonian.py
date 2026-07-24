# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHamiltonian from former test_ibm_verification_circuits.py

"""Focused suite: TestHamiltonian from former test_ibm_verification_circuits.py."""

from __future__ import annotations

from tests.ibm_verification_circuits_support import *  # noqa: F403


class TestHamiltonian:
    def test_hermitian(self):
        H = posner_hamiltonian(1.0, HF1, HF2)
        assert np.allclose(H, H.conj().T)

    def test_256x256(self):
        assert posner_hamiltonian(1.0, HF1, HF2).shape == (256, 256)

    def test_nuclear_dipolar_present(self):
        """Nuclear dipolar terms must affect H when coupling is non-negligible."""
        # Use exaggerated coupling to verify the code path works
        H_with = posner_hamiltonian(1.0, HF1, HF2, d_nuc=0.03, d_nuc_cross=0.01)
        H_without = posner_hamiltonian(1.0, HF1, HF2, d_nuc=0.0, d_nuc_cross=0.0)
        assert not np.allclose(H_with, H_without), "Dipolar coupling must affect H"

    def test_cross_site_dipolar_present(self):
        """Cross-site nuclear dipolar (9 pairs) must change H at exaggerated coupling."""
        H_no_cross = posner_hamiltonian(1.0, HF1, HF2, d_nuc=0.03, d_nuc_cross=0.0)
        H_with_cross = posner_hamiltonian(1.0, HF1, HF2, d_nuc=0.03, d_nuc_cross=0.01)
        assert not np.allclose(H_no_cross, H_with_cross), "Cross-site dipolar missing"

    def test_cross_site_weaker_than_intra(self):
        """At equal distance, cross-site constant < intra-site (longer distance)."""
        assert DEFAULT_NUC_DIPOLAR_CROSS < DEFAULT_NUC_DIPOLAR

    def test_dipolar_physically_correct_magnitude(self):
        """Dipolar coupling must be ~10⁻⁸ in dimensionless units (not 0.03)."""
        assert DEFAULT_NUC_DIPOLAR < 1e-6, f"Nuclear dipolar too large: {DEFAULT_NUC_DIPOLAR}"
        assert DEFAULT_NUC_DIPOLAR > 1e-10, f"Nuclear dipolar too small: {DEFAULT_NUC_DIPOLAR}"
        # All per-pair couplings also in correct range
        for _, _, d in _DIPOLAR_PAIRS:
            assert 1e-10 < d < 1e-6, f"Dipolar coupling out of range: {d}"

    def test_15_dipolar_pairs_total(self):
        assert len(_INTRA_PAIRS) == 6
        assert len(_CROSS_PAIRS) == 9

    def test_anisotropic_differs(self):
        iso = [{"Axx": 0.5, "Ayy": 0.5, "Azz": 0.5}] * 3
        H_iso = posner_hamiltonian(1.0, iso, iso)
        H_aniso = posner_hamiltonian(1.0, HF1, HF2)
        assert not np.allclose(H_iso, H_aniso)

    def test_off_diagonal_hf(self):
        """Off-diagonal HF (Axy, Axz, Ayz) must change H vs diagonal-only."""
        diag = [{"Axx": 0.5, "Ayy": 0.5, "Azz": 0.5, "Axy": 0, "Axz": 0, "Ayz": 0}] * 3
        full = [{"Axx": 0.5, "Ayy": 0.5, "Azz": 0.5, "Axy": 0.05, "Axz": 0.03, "Ayz": 0.02}] * 3
        H_diag = posner_hamiltonian(1.0, diag, diag)
        H_full = posner_hamiltonian(1.0, full, full)
        assert not np.allclose(H_diag, H_full), "Off-diagonal HF must affect H"
        assert np.allclose(H_full, H_full.conj().T), "Must remain Hermitian"

    def test_per_pair_dipolar_distances(self):
        """Per-pair table has 15 entries with distinct couplings."""
        assert len(_DIPOLAR_PAIRS) == 15
        couplings = [d for _, _, d in _DIPOLAR_PAIRS]
        # Couplings are ~10⁻⁸, need high-precision rounding
        assert len(set(round(c, 12) for c in couplings)) >= 3, (
            f"Must have >=3 distinct values, got {set(round(c, 12) for c in couplings)}"
        )

    def test_external_dipolar_pairs_require_all_31p_pairs(self):
        pairs = [
            {
                "qubit_i": i,
                "qubit_j": j,
                "Axx": -1e-8 * (i + j),
                "Ayy": -1.1e-8 * (i + j),
                "Azz": 2.1e-8 * (i + j),
                "Axy": 0.1e-8,
                "Axz": 0.2e-8,
                "Ayz": 0.3e-8,
            }
            for i in range(2, 8)
            for j in range(i + 1, 8)
        ]
        parsed = _parse_dipolar_pairs(pairs)
        assert len(parsed) == 15
        assert parsed[0][0:2] == (2, 3)
        assert parsed[0][2]["Azz"] == 10.5e-8

    def test_external_dipolar_pairs_fail_closed_when_incomplete(self):
        pairs = [
            [i, j, -1e-8, -1e-8, 2e-8, 0.0, 0.0, 0.0] for i in range(2, 8) for j in range(i + 1, 8)
        ]
        with pytest.raises(ValueError, match="15 unique"):
            _parse_dipolar_pairs(pairs[:-1])

    def test_external_dipolar_pairs_reject_scalar_magnitude_only(self):
        pairs = [[i, j, 1e-8] for i in range(2, 8) for j in range(i + 1, 8)]
        with pytest.raises(ValueError, match="full tensor"):
            _parse_dipolar_pairs(pairs)
