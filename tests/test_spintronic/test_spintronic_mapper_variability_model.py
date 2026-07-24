# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVariabilityModel from former test_spintronic_mapper.py

"""Focused suite: TestVariabilityModel from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403


class TestVariabilityModel:
    def test_apply(self):
        rng = np.random.default_rng(42)
        base = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        var = VariabilityModel()
        varied = var.apply(base, rng)
        assert varied.width_nm != base.width_nm or varied.length_nm != base.length_nm

    def test_apply_clamps(self):
        rng = np.random.default_rng(42)
        base = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        var = VariabilityModel(width_sigma_pct=500)
        varied = var.apply(base, rng)
        assert varied.width_nm >= 10.0
        assert varied.material.damping_alpha >= 0.001

    def test_zero_variability(self):
        rng = np.random.default_rng(42)
        var = VariabilityModel(
            width_sigma_pct=0,
            length_sigma_pct=0,
            ku_sigma_pct=0,
            dmi_sigma_pct=0,
            damping_sigma_pct=0,
            ms_sigma_pct=0,
        )
        base = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        varied = var.apply(base, rng)
        assert abs(varied.width_nm - base.width_nm) < 1e-6
