# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyMetrics from former test_sources_and_profiling.py

"""Focused suite: TestEnergyMetrics from former test_sources_and_profiling.py."""

from __future__ import annotations

from tests.sources_and_profiling_support import *  # noqa: F403


class TestEnergyMetrics:
    def test_defaults(self):
        em = EnergyMetrics()
        assert em.total_ops_and == 0
        assert em.total_ops_xor == 0
        assert em.total_bits_mem == 0

    def test_estimate_energy_zero(self):
        em = EnergyMetrics()
        assert em.estimate_energy() == 0.0

    def test_estimate_energy_with_ops(self):
        em = EnergyMetrics()
        em.total_ops_and = 1_000_000
        energy = em.estimate_energy()
        expected = 1_000_000 * 0.1e-15
        assert energy == pytest.approx(expected)

    def test_estimate_energy_combined(self):
        em = EnergyMetrics()
        em.total_ops_and = 100
        em.total_ops_xor = 200
        em.total_bits_mem = 300
        energy = em.estimate_energy()
        expected = 100 * 0.1e-15 + 200 * 0.15e-15 + 300 * 5.0e-15
        assert energy == pytest.approx(expected)

    def test_reset(self):
        em = EnergyMetrics()
        em.total_ops_and = 999
        em.total_ops_xor = 888
        em.total_bits_mem = 777
        em.reset()
        assert em.total_ops_and == 0
        assert em.total_ops_xor == 0
        assert em.total_bits_mem == 0

    def test_co2_emission(self):
        em = EnergyMetrics()
        em.total_ops_and = 1_000_000_000  # 1 billion AND ops
        co2 = em.co2_emission_g()
        assert co2 > 0
        assert isinstance(co2, float)

    def test_co2_custom_intensity(self):
        em = EnergyMetrics()
        em.total_ops_and = 1_000_000
        co2_default = em.co2_emission_g(carbon_intensity_g_per_kwh=475)
        co2_green = em.co2_emission_g(carbon_intensity_g_per_kwh=50)
        assert co2_green < co2_default
