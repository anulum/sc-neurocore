# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMaterialParams from former test_spintronic_mapper.py

"""Focused suite: TestMaterialParams from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403


class TestMaterialParams:
    def test_cofeb_mgo(self):
        m = MaterialParams.cofeb_mgo()
        assert m.saturation_magnetisation_a_m > 0
        assert m.damping_alpha > 0

    def test_pt_co(self):
        m = MaterialParams.pt_co_multilayer()
        assert m.dmi_strength_j_m2 > 0  # skyrmion host requires DMI

    def test_w_cofeb(self):
        m = MaterialParams.w_cofeb()
        assert m.saturation_magnetisation_a_m > 0
