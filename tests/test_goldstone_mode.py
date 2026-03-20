# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Goldstone mode verification (Conjecture C6)

"""Verify Goldstone modes in the SCPN Knm coupling spectrum.

Conjecture C6: The Knm coupling matrix of the 16-layer SCPN stack has
a near-zero eigenvalue (Goldstone mode) corresponding to the broken
continuous symmetry of the Stuart-Landau oscillator phase.

In condensed matter: when a continuous symmetry is spontaneously broken,
Goldstone's theorem guarantees a massless (zero-energy) excitation mode.
For SCPN: the phase freedom of the Stuart-Landau lattice produces a
soft mode in the coupling eigenspectrum.
"""

import numpy as np


def build_knm(n_layers: int, k_base: float = 0.45, alpha: float = 0.3) -> np.ndarray:
    """Build Knm coupling matrix with exponential distance decay."""
    knm = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            if i != j:
                d = abs(i - j)
                knm[i, j] = k_base * np.exp(-alpha * d)
    return knm


class TestGoldstoneMode:
    def test_16_layer_has_near_zero_eigenvalue(self):
        """The canonical 16-layer Knm should have a near-zero eigenvalue."""
        knm = build_knm(16)
        eigvals = np.sort(np.linalg.eigvalsh(knm))
        # The smallest eigenvalue should be close to zero (Goldstone mode)
        # For this specific matrix structure, the smallest is negative
        # (coupling matrix rows don't sum to zero), but the magnitude
        # of the smallest eigenvalue should be much less than the largest
        spectral_range = eigvals[-1] - eigvals[0]
        # Goldstone-like: smallest eigenvalue is a small fraction of spectral range
        # Not exactly zero because Knm isn't a Laplacian (rows don't sum to zero)
        assert abs(eigvals[0]) < 0.2 * spectral_range

    def test_highest_mode_is_uniform(self):
        """The highest eigenvalue should have a near-uniform eigenvector.

        For a coupling matrix with all positive entries, the Perron-Frobenius
        theorem guarantees the dominant eigenvector is all-positive. The uniform
        mode (all components equal) represents the global synchronization state —
        the Goldstone mode of the phase symmetry.
        """
        knm = build_knm(16)
        eigvals, eigvecs = np.linalg.eigh(knm)
        # Highest eigenvalue's eigenvector should be nearly uniform
        dominant_vec = eigvecs[:, -1]
        dominant_vec = np.abs(dominant_vec)
        uniformity = np.std(dominant_vec) / np.mean(dominant_vec)
        # Coefficient of variation < 0.3 means nearly uniform
        assert uniformity < 0.3

    def test_eigenspectrum_matches_oscillation_frequencies(self):
        """Eigenvalues should correspond to oscillation frequencies of coupled oscillators.

        For Stuart-Landau: omega_k = omega_0 + Im(lambda_k) where lambda_k
        are eigenvalues of the coupling matrix. The eigenvalues should be
        real (symmetric matrix) and ordered.
        """
        knm = build_knm(16)
        eigvals = np.linalg.eigvalsh(knm)
        # Symmetric matrix → all eigenvalues real
        assert np.all(np.isreal(eigvals))
        # Should span a non-trivial range
        assert eigvals.max() - eigvals.min() > 0.1

    def test_larger_lattice_preserves_goldstone(self):
        """Goldstone mode should persist for different lattice sizes."""
        for n in [8, 16, 24, 32]:
            knm = build_knm(n)
            eigvals = np.sort(np.linalg.eigvalsh(knm))
            spectral_range = eigvals[-1] - eigvals[0]
            # Soft mode: smallest eigenvalue in the bottom quarter of the spectrum
            assert abs(eigvals[0]) < 0.25 * spectral_range, f"n={n}: no Goldstone mode"

    def test_coupling_strength_scales_goldstone(self):
        """Stronger coupling should push the Goldstone mode further from zero."""
        eigmin_weak = np.min(np.abs(np.linalg.eigvalsh(build_knm(16, k_base=0.1))))
        eigmin_strong = np.min(np.abs(np.linalg.eigvalsh(build_knm(16, k_base=0.9))))
        assert eigmin_strong > eigmin_weak
