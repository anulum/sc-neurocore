# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNetworkPipeline from former test_e2e_pipeline.py

"""Focused suite: TestNetworkPipeline from former test_e2e_pipeline.py."""

from __future__ import annotations

from tests.e2e.e2e_pipeline_support import *  # noqa: F403


@pytest.mark.e2e
class TestNetworkPipeline:
    """BRAM array → weight ROM → constraints → testbench."""

    def test_bram_array_is_synthesisable(self):
        """BRAM array Verilog is structurally valid."""
        from sc_neurocore.compiler.intelligence import (
            storage_recommendation,
            generate_bram_array,
        )

        rec = storage_recommendation(512, 16)
        assert rec.strategy == "bram"

        v = generate_bram_array(neuron_count=512, data_width=16)
        assert "module sc_neuron_array" in v
        assert "endmodule" in v
        assert "state_bram" in v
        assert "ram_style" in v
        assert "spike_out" in v
        assert "tick_done" in v

    def test_weight_rom_matches_dimensions(self):
        """Weight ROM entries match weight matrix dimensions."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        weights = [[i * 10 + j for j in range(4)] for i in range(8)]
        mif = generate_weight_rom(weights, output_format="mif")
        coe = generate_weight_rom(weights, output_format="coe")

        assert "DEPTH=32" in mif  # 8×4 = 32

    def test_bram_array_plus_constraints(self):
        """BRAM array → constraints: valid artefacts from same data width."""
        from sc_neurocore.compiler.intelligence import generate_bram_array
        from sc_neurocore.compiler.platforms import get_profile
        from sc_neurocore.compiler.deployment import generate_constraints

        profile = get_profile("artix7")
        dw = profile.data_width

        v = generate_bram_array(
            module_name="sc_net_512",
            neuron_count=512,
            data_width=dw,
        )
        xdc = generate_constraints(
            module_name="sc_net_512",
            data_width=dw,
            target_freq_mhz=float(profile.max_freq_mhz or 100),
        )
        assert str(dw - 1) in v
        assert "create_clock" in xdc
