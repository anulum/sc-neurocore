# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalFullFlow from former test_e2e_pipeline.py

"""Focused suite: TestThermalFullFlow from former test_e2e_pipeline.py."""

from __future__ import annotations

from tests.e2e.e2e_pipeline_support import *  # noqa: F403

@pytest.mark.e2e
class TestThermalFullFlow:
    """Power estimation → thermal analysis → derated constraints."""

    def test_power_to_thermal_to_constraints(self):
        """Power → thermal → XDC: derated frequency propagates."""
        from sc_neurocore.compiler.static_analysis import estimate_power
        from sc_neurocore.compiler.intelligence import (
            thermal_analysis,
            generate_thermal_constraints,
        )

        verilog = (
            "reg signed [15:0] v_reg;\n"
            "wire signed [31:0] _mul0 = a * b;\n"
            "wire signed [31:0] _mul1 = c * d;\n"
            "wire signed [15:0] _t0 = a + b - c + d;\n"
        )
        power = estimate_power(verilog, freq_mhz=500.0, process_nm=16)
        therm = thermal_analysis(
            power.total_mw,
            500.0,
            process_nm=16,
            mul_count=2,
        )
        xdc = generate_thermal_constraints("sc_lif_thermal", therm)

        assert str(therm.derated_freq_mhz) in xdc
        assert "create_clock" in xdc

    def test_high_power_triggers_warning(self):
        """Very high power → thermal unsafe → warning in XDC."""
        from sc_neurocore.compiler.intelligence import (
            thermal_analysis,
            generate_thermal_constraints,
        )

        therm = thermal_analysis(50000.0, 500.0)
        assert not therm.thermal_safe

        xdc = generate_thermal_constraints("sc_hot", therm)
        assert "WARNING" in xdc
