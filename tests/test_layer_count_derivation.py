# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Test for Conjecture C3/C18: 16 layers from free energy

"""Test that L=16 uniquely maximizes the eigenvalue gap of the Knm coupling matrix.

Conjecture C3 (Round 3): The 16-layer structure is derivable from free energy
minimization on a Stuart-Landau lattice. The steady-state amplitude profile
has exactly N = 16 stable nodes for the canonical Paper 27 parameters.

Conjecture C18 (Round 6): L=16 uniquely maximizes the minimum eigenvalue gap
(no near-degeneracies).
"""

import numpy as np


def build_knm(n_layers: int, k_base: float = 0.45, alpha: float = 0.3) -> np.ndarray:
    """Build a Knm coupling matrix with exponential distance decay."""
    knm = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            if i != j:
                d = abs(i - j)
                knm[i, j] = k_base * np.exp(-alpha * d)
    return knm


def min_eigenvalue_gap(knm: np.ndarray) -> float:
    """Compute the minimum gap between consecutive nonzero eigenvalues."""
    eigvals = np.sort(np.linalg.eigvalsh(knm))
    # Skip the zero/near-zero eigenvalues
    nonzero = eigvals[np.abs(eigvals) > 1e-10]
    if len(nonzero) < 2:
        return 0.0
    gaps = np.diff(nonzero)
    return float(np.min(np.abs(gaps)))


class TestLayerCountDerivation:
    def test_16_maximizes_min_gap(self):
        """Sweep L from 4 to 32 and verify L=16 has the largest min eigenvalue gap."""
        results = {}
        for L in range(4, 33):
            knm = build_knm(L)
            results[L] = min_eigenvalue_gap(knm)

        best_L = max(results, key=results.get)
        # The conjecture: L=16 (or near it) maximizes the gap
        assert 14 <= best_L <= 18, f"Expected L near 16, got {best_L}"

    def test_canonical_16_has_nondegenerate_spectrum(self):
        """The canonical 16-layer Knm has no near-degenerate eigenvalue pairs."""
        knm = build_knm(16)
        eigvals = np.sort(np.linalg.eigvalsh(knm))
        gaps = np.abs(np.diff(eigvals))
        # No gap smaller than 1% of the spectral range
        spectral_range = eigvals[-1] - eigvals[0]
        assert np.all(gaps > 0.01 * spectral_range)

    def test_small_and_large_L_have_smaller_gaps(self):
        """L=8 and L=24 should have smaller min gaps than L=16."""
        gap_8 = min_eigenvalue_gap(build_knm(8))
        gap_16 = min_eigenvalue_gap(build_knm(16))
        gap_24 = min_eigenvalue_gap(build_knm(24))
        assert gap_16 > gap_8
        assert gap_16 > gap_24
