# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical ASIC flow tests

"""Exercise block synthesis and top-level hard-macro integration."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    BlockConfig,
    DesignParams,
    HierarchicalFlow,
    PDKConfig,
    PDKType,
)


class TestHierarchicalFlow:
    def test_add_blocks(self) -> None:
        hf = HierarchicalFlow(top_design=DesignParams())
        hf.add_block(BlockConfig("neuron_array", DesignParams(top_module="neuron_array")))
        hf.add_block(BlockConfig("router", DesignParams(top_module="aer_router")))
        assert hf.block_names() == ["neuron_array", "router"]

    def test_block_scripts(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        hf = HierarchicalFlow(top_design=DesignParams())
        hf.add_block(BlockConfig("core", DesignParams(top_module="sc_core")))
        scripts = hf.generate_block_scripts(pdk)
        assert "core" in scripts
        assert "synth" in scripts["core"].lower()

    def test_top_integration(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        hf = HierarchicalFlow(top_design=DesignParams(top_module="chip_top"))
        hf.add_block(BlockConfig("mem", DesignParams(), is_hard_macro=True, abstract_lef="mem.lef"))
        tcl = hf.generate_top_integration(pdk)
        assert "mem.lef" in tcl
        assert "chip_top" in tcl

    def test_top_integration_omits_soft_block_lef(self) -> None:
        """Logical blocks do not inject hard-macro LEFs into top integration."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        flow = HierarchicalFlow(top_design=DesignParams(top_module="chip_top"))
        flow.add_block(BlockConfig("logic", DesignParams(), abstract_lef="logic.lef"))

        script = flow.generate_top_integration(pdk)

        assert "logic.lef" not in script
        assert "link_design chip_top" in script
