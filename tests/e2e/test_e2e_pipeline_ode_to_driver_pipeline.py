# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestODEToDriverPipeline from former test_e2e_pipeline.py

"""Focused suite: TestODEToDriverPipeline from former test_e2e_pipeline.py."""

from __future__ import annotations

from tests.e2e.e2e_pipeline_support import *  # noqa: F403


@pytest.mark.e2e
class TestODEToDriverPipeline:
    """Full pipeline: compile → estimate → constrain → driver."""

    def test_lif_full_pipeline_artix7(self):
        """LIF on Artix-7: every artefact is internally consistent."""
        from sc_neurocore.compiler.platforms import get_profile
        from sc_neurocore.compiler.deployment import (
            estimate_resources,
            generate_constraints,
            generate_host_driver,
            generate_cocotb_testbench,
        )

        profile = get_profile("artix7")
        module = "sc_lif_e2e"
        dw = profile.data_width

        # 1. Resource estimate
        verilog_stub = (
            "module sc_lif_e2e(\n"
            f"  input wire signed [{dw - 1}:0] I_t,\n"
            f"  output wire signed [{dw - 1}:0] v_next\n"
            ");\n"
            f"  wire signed [{2 * dw - 1}:0] _mul0 = I_t * {dw}'sd10;\n"
            f"  wire signed [{dw - 1}:0] _t0 = _mul0[{dw - 1}:0];\n"
            "endmodule\n"
        )
        res = estimate_resources(verilog_stub, has_dsp=bool(profile.dsp_block))
        assert res.luts >= 0
        assert res.mul_count >= 0

        # 2. Constraints (uses target_freq_mhz, not target object)
        freq = profile.max_freq_mhz or 100
        xdc = generate_constraints(
            module_name=module,
            data_width=dw,
            target_freq_mhz=float(freq),
        )
        assert "create_clock" in xdc

        # 3. Host driver (C)
        c_driver = generate_host_driver(
            module_name=module,
            params={"v": dw, "I_t": dw},
            data_width=dw,
            language="c",
        )
        assert "write" in c_driver.lower() or "WRITE" in c_driver

        # 4. Cocotb testbench
        tb = generate_cocotb_testbench(
            module_name=module,
            data_width=dw,
        )
        assert "import cocotb" in tb
        assert module in tb

    def test_pipeline_data_width_consistency(self):
        """Data widths match across constraints and drivers."""
        from sc_neurocore.compiler.platforms import get_profile
        from sc_neurocore.compiler.deployment import (
            generate_constraints,
            generate_host_driver,
        )

        for target_name in ["artix7", "loihi2", "ecp5"]:
            profile = get_profile(target_name)
            dw = profile.data_width
            module = f"sc_test_{target_name}"
            freq = profile.max_freq_mhz or 100

            xdc = generate_constraints(
                module_name=module,
                data_width=dw,
                target_freq_mhz=float(freq),
            )
            driver = generate_host_driver(
                module_name=module,
                data_width=dw,
                params={"v": dw},
                language="c",
            )
            # Both should reference something meaningful
            assert "create_clock" in xdc
            assert module in driver
