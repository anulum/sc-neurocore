# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConductanceModel from former test_memristor_mapper.py

"""Focused suite: TestConductanceModel from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403


class TestConductanceModel:
    def test_defaults_from_technology(self) -> None:
        m = ConductanceModel(MemristorTechnology.RERAM_HFOX)
        assert m.g_on == 100e-6
        assert m.g_off == 1e-6
        assert m.sigma_g == 0.05

    def test_dynamic_range(self) -> None:
        m = ConductanceModel(MemristorTechnology.RERAM_HFOX)
        assert m.dynamic_range == pytest.approx(100.0)

    def test_level_step_positive(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        assert m.level_step > 0

    def test_target_conductance_bounds(self) -> None:
        m = ConductanceModel(MemristorTechnology.PCM)
        assert m.target_conductance(0) == m.g_off
        assert m.target_conductance(m.num_levels - 1) == pytest.approx(m.g_on)

    def test_target_conductance_clamps(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        assert m.target_conductance(-1) == m.g_off
        assert m.target_conductance(9999) == pytest.approx(m.g_on)

    def test_sample_d2d_different_each_call(self) -> None:
        m = ConductanceModel(MemristorTechnology.RERAM_HFOX)
        rng = np.random.default_rng(42)
        s1 = m.sample_d2d(8, rng)
        s2 = m.sample_d2d(8, rng)
        assert s1 != s2

    def test_sample_rw_adds_noise(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        rng = np.random.default_rng(42)
        vals = [m.sample_rw(50e-6, rng) for _ in range(100)]
        assert np.std(vals) > 0

    def test_all_technologies_load(self) -> None:
        for tech in MemristorTechnology:
            m = ConductanceModel(tech)
            assert m.g_on > m.g_off
            assert m.num_levels >= 1

    def test_mythic_high_levels(self) -> None:
        m = ConductanceModel(MemristorTechnology.MYTHIC_AMP)
        assert m.num_levels == 256

    def test_2d_material_higher_endurance_model(self) -> None:
        m = ConductanceModel(MemristorTechnology.RERAM_2D)
        assert m.sigma_g == 0.08
