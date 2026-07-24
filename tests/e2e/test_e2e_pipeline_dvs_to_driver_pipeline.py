# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDVSToDriverPipeline from former test_e2e_pipeline.py

"""Focused suite: TestDVSToDriverPipeline from former test_e2e_pipeline.py."""

from __future__ import annotations

from tests.e2e.e2e_pipeline_support import *  # noqa: F403


@pytest.mark.e2e
class TestDVSToDriverPipeline:
    """DVS event camera → AER bridge → RISC-V driver chain."""

    def test_dvs_aer_bridge_valid(self):
        """DVS bridge Verilog is structurally valid."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        bridge = generate_dvs_aer_bridge(
            module_name="sc_dvs_bridge",
            addr_width=16,
        )
        assert "module sc_dvs_bridge" in bridge
        assert "endmodule" in bridge

    def test_dvs_bridge_plus_riscv_driver(self):
        """DVS bridge + RISC-V driver: both produce valid artefacts."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        bridge = generate_dvs_aer_bridge()
        driver = generate_riscv_driver(
            "sc_dvs_neuron",
            params={"v": 16, "I_t": 16},
            data_width=16,
            rtos="baremetal",
        )
        assert "endmodule" in bridge
        assert "uint" in driver or "void" in driver
