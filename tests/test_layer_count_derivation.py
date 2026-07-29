# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Test for Conjecture C3/C18: 16 layers from free energy

"""Test the exponential-decay Knm surrogate and its research boundary.

Conjecture C3 (Round 3): The 16-layer structure is derivable from free energy
minimization on a Stuart-Landau lattice.

Conjecture C18 (Round 6): L=16 uniquely maximizes the minimum eigenvalue gap.

The matrix in this module is only an exponential-distance surrogate; it is not
the conjectured Stuart-Landau coupling model. Its eigengap decreases with layer
count and therefore cannot provide evidence for C3/C18. The tests below lock
that negative-control boundary until a sourced full model is implemented.
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
    nonzero = eigvals[np.abs(eigvals) > 1e-10]
    if len(nonzero) < 2:
        return 0.0
    gaps = np.diff(nonzero)
    return float(np.min(np.abs(gaps)))


class TestExponentialDecayBoundary:
    def test_minimum_gap_decreases_across_the_sweep(self) -> None:
        """Prove the surrogate selects the smallest candidate, not 16 layers."""

        results = {}
        for layer_count in range(4, 33):
            knm = build_knm(layer_count)
            results[layer_count] = min_eigenvalue_gap(knm)

        assert max(results, key=results.get) == 4
        assert all(results[count] > results[count + 1] for count in range(4, 32))

    def test_16_layer_surrogate_has_near_degenerate_pairs(self) -> None:
        """Record why the surrogate cannot support the nondegeneracy claim."""

        knm = build_knm(16)
        eigvals = np.sort(np.linalg.eigvalsh(knm))
        gaps = np.abs(np.diff(eigvals))
        spectral_range = eigvals[-1] - eigvals[0]
        assert np.any(gaps <= 0.01 * spectral_range)

    def test_selected_gap_order_is_opposite_the_conjecture(self) -> None:
        """Keep the 8/16/24 negative control explicit and quantitative."""

        gap_8 = min_eigenvalue_gap(build_knm(8))
        gap_16 = min_eigenvalue_gap(build_knm(16))
        gap_24 = min_eigenvalue_gap(build_knm(24))
        assert gap_8 > gap_16 > gap_24

    def test_knm_is_symmetric(self) -> None:
        """Knm should be symmetric for real eigenvalues."""
        knm = build_knm(16)
        np.testing.assert_allclose(knm, knm.T)

    def test_eigenvalues_span_nonzero_range(self) -> None:
        """The eigenspectrum should have a non-trivial range."""
        knm = build_knm(16)
        eigvals = np.linalg.eigvalsh(knm)
        assert eigvals.max() - eigvals.min() > 0.1
