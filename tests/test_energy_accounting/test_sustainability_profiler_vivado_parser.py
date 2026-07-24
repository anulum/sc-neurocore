# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVivadoParser from former test_sustainability_profiler.py

"""Focused suite: TestVivadoParser from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403


class TestVivadoParser:
    def test_from_vivado_dict(self):
        d = {
            "LUT": 50000,
            "FF": 30000,
            "BRAM_KB": 256,
            "DSP": 20,
            "Toggle_Rate": 0.2,
            "Clock_MHz": 150,
            "Voltage_V": 0.9,
            "Static_Power_mW": 80,
        }
        r = FPGAResourceReport.from_vivado_dict(d)
        assert r.luts == 50000
        assert r.clock_mhz == 150
        assert r.toggle_rate == 0.2

    def test_from_vivado_dict_defaults(self):
        r = FPGAResourceReport.from_vivado_dict({})
        assert r.luts == 0
        assert r.clock_mhz == 100.0
