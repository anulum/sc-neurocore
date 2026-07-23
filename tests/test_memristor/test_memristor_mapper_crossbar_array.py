# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossbarArray from former test_memristor_mapper.py

"""Focused suite: TestCrossbarArray from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403

class TestCrossbarArray:
    def test_num_devices_standard(self) -> None:
        xbar = CrossbarArray(64, 64)
        assert xbar.num_devices == 4096

    def test_num_devices_differential(self) -> None:
        xbar = CrossbarArray(32, 32, CrossbarTopology.DIFFERENTIAL)
        assert xbar.num_devices == 2048

    def test_conductance_model(self) -> None:
        xbar = CrossbarArray(16, 16, technology=MemristorTechnology.PCM)
        m = xbar.conductance_model
        assert m.technology == MemristorTechnology.PCM
