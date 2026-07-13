# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical multi-block ASIC deck generation

"""Compose block-level synthesis decks into hierarchical ASIC integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from sc_neurocore.asic_flow.design import DesignParams
from sc_neurocore.asic_flow.flow import ASICFlowGenerator
from sc_neurocore.asic_flow.pdk import PDKConfig


@dataclass
class BlockConfig:
    """One block in a hierarchical ASIC flow."""

    name: str
    design: DesignParams
    is_hard_macro: bool = False
    abstract_lef: str = ""


@dataclass
class HierarchicalFlow:
    """Multi-block ASIC flow with per-block synthesis + top integration."""

    top_design: DesignParams
    blocks: List[BlockConfig] = field(default_factory=list)

    def add_block(self, block: BlockConfig) -> None:
        """Append one logical or hard-macro block to the flow."""
        self.blocks.append(block)

    def block_names(self) -> List[str]:
        """Return block names in deterministic insertion order."""
        return [b.name for b in self.blocks]

    def generate_block_scripts(self, pdk: PDKConfig) -> Dict[str, str]:
        """Generate one Yosys synthesis script per configured block."""
        gen = ASICFlowGenerator()
        result = {}
        for block in self.blocks:
            output = gen.generate(pdk, block.design)
            result[block.name] = output.synth_tcl
        return result

    def generate_top_integration(self, pdk: PDKConfig) -> str:
        """Render top-level netlist linkage and hard-macro LEF reads."""
        lines = [f"# Hierarchical integration for {self.top_design.top_module}"]
        for block in self.blocks:
            if block.is_hard_macro and block.abstract_lef:
                lines.append(f"read_lef {block.abstract_lef}  ;# macro: {block.name}")
        lines.append(f"read_verilog synth_{self.top_design.top_module}.v")
        lines.append(f"link_design {self.top_design.top_module}")
        return "\n".join(lines) + "\n"
