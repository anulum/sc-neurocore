# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWeightNoise from former test_platforms.py

"""Focused suite: TestWeightNoise from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


class TestWeightNoise:
    """Device-variation noise injection for analog robustness."""

    def test_gaussian_noise_changes_weights(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, -1.0], [0.5, 0.0]]
        noisy = inject_weight_noise(w, seed=42)
        # At least one value should differ
        differs = any(w[i][j] != noisy[i][j] for i in range(len(w)) for j in range(len(w[0])))
        assert differs

    def test_noise_is_reproducible(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, 0.5], [-0.3, 0.8]]
        n1 = inject_weight_noise(w, seed=123)
        n2 = inject_weight_noise(w, seed=123)
        assert n1 == n2

    def test_different_seeds_differ(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, 0.5]]
        n1 = inject_weight_noise(w, seed=1)
        n2 = inject_weight_noise(w, seed=2)
        assert n1 != n2

    def test_uniform_noise(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, -1.0]]
        noisy = inject_weight_noise(w, noise_model="uniform", seed=42)
        assert len(noisy) == 1
        assert len(noisy[0]) == 2

    def test_lognormal_noise(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, 0.5]]
        noisy = inject_weight_noise(w, noise_model="lognormal", seed=42)
        assert len(noisy[0]) == 2

    def test_noise_profile_creation(self):
        from sc_neurocore.compiler.intelligence import create_noise_profile

        p = create_noise_profile(
            sigma=0.03,
            target="rain_neuromorphic",
        )
        assert p.noise_model == "gaussian"
        assert p.sigma == 0.03
        assert p.target_platform == "rain_neuromorphic"

    def test_zero_sigma_no_noise(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, -0.5, 0.3]]
        noisy = inject_weight_noise(w, sigma=0.0, seed=42)
        assert noisy == w
