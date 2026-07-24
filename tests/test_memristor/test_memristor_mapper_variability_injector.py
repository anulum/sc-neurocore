# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVariabilityInjector from former test_memristor_mapper.py

"""Focused suite: TestVariabilityInjector from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403


class TestVariabilityInjector:
    def test_quantize_weights(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.array([[0.0, 0.5, 1.0]])
        levels = inj.quantize_weights(w)
        assert levels[0, 0] == 0
        assert levels[0, 2] == m.num_levels - 1

    def test_inject_d2d_changes_values(self) -> None:
        m = ConductanceModel(MemristorTechnology.RERAM_HFOX)
        inj = VariabilityInjector(m, seed=42)
        levels = np.array([[8, 8, 8]])
        g = inj.inject_d2d(levels)
        assert not np.all(g == g[0, 0])

    def test_inject_rw_adds_noise(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        g = np.full((4, 4), 50e-6)
        noisy = inj.inject_rw(g)
        assert not np.allclose(g, noisy)

    def test_inject_full_pipeline(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        levels, cond = inj.inject_full(w)
        assert levels.shape == (4, 4)
        assert cond.shape == (4, 4)
        assert np.all(levels >= 0)
        assert np.all(levels < m.num_levels)

    def test_compute_error_positive(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        levels, cond = inj.inject_full(w)
        err = inj.compute_error(w, cond)
        assert err["mae"] >= 0
        assert err["mean_rel_err"] >= 0

    def test_deterministic_with_same_seed(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        w = np.random.default_rng(0).random((3, 3))
        inj1 = VariabilityInjector(m, seed=42)
        _, g1 = inj1.inject_full(w)
        inj2 = VariabilityInjector(m, seed=42)
        _, g2 = inj2.inject_full(w)
        np.testing.assert_array_equal(g1, g2)
