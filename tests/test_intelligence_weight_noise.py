# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight-noise contracts

"""Contracts for compiler weight-noise injection."""

from __future__ import annotations

import pytest


class TestWeightNoise:
    """Analog device-variation weight-noise injection."""

    def test_gaussian_noise_is_seeded_and_shape_preserving(self) -> None:
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        weights = [[0.25, -0.5], [1.0, 0.0]]
        first = inject_weight_noise(weights, noise_model="gaussian", sigma=0.1, seed=7)
        second = inject_weight_noise(weights, noise_model="gaussian", sigma=0.1, seed=7)

        assert first == second
        assert len(first) == len(weights)
        assert all(len(row) == len(src) for row, src in zip(first, weights, strict=True))

    def test_uniform_noise_is_bounded_by_sigma_scale(self) -> None:
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        weights = [[-2.0, 0.0, 2.0]]
        noisy = inject_weight_noise(weights, noise_model="uniform", sigma=0.25, seed=11)

        for original, perturbed in zip(weights[0], noisy[0], strict=True):
            assert abs(perturbed - original) <= 0.5

    def test_zero_matrix_has_absolute_noise_scale_fallback(self) -> None:
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        noisy = inject_weight_noise([[0.0, 0.0]], noise_model="uniform", sigma=0.1, seed=3)

        assert all(-0.1 <= value <= 0.1 for value in noisy[0])

    def test_rejects_unknown_noise_model(self) -> None:
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        with pytest.raises(ValueError, match="Unsupported weight noise model"):
            inject_weight_noise([[1.0]], noise_model="triangular")
