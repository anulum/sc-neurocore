# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MLIR / CIRCT Emitter for Stochastic Computing pipelines

"""
MLIR / CIRCT Emitter for Stochastic Computing pipelines.

This module provides the frontend to lower sc-neurocore's Python-based
Stochastic IR into MLIR hardware dialects (HW, Seq, Comb) via CIRCT.
This allows us to leverage LLVM's optimization passes directly for
FPGA bitstream synthesis, skipping string-based Verilog generation.
"""

from dataclasses import dataclass
from typing import List, Any


@dataclass
class MLIRNode:
    op_type: str
    inputs: List[str]
    output: str
    attributes: dict[str, Any]


class MLIREmitter:
    """
    Translates sc-neurocore objects into MLIR text formatted for CIRCT.
    """

    def __init__(self, module_name: str = "sc_neurocore_top"):
        self.module_name = module_name
        self.nodes: List[MLIRNode] = []
        self._wire_counter = 0

    def get_wire(self) -> str:
        self._wire_counter += 1
        return f"%w{self._wire_counter}"

    def emit_and(self, lhs: str, rhs: str) -> str:
        """Emits a comb.and operation for stochastic multiplication."""
        out = self.get_wire()
        self.nodes.append(MLIRNode("comb.and", [lhs, rhs], out, {}))
        return out

    def emit_lfsr(self, width: int, seed: int) -> str:
        """Emits an LFSR instantiation."""
        out = self.get_wire()
        self.nodes.append(
            MLIRNode(
                "hw.instance",
                [],
                out,
                {
                    "sym_name": "lfsr",
                    "module": "sc_lfsr",
                    "parameters": {"WIDTH": width, "SEED": seed},
                },
            )
        )
        return out

    def emit_xor(self, lhs: str, rhs: str) -> str:
        """Emits a comb.xor operation."""
        out = self.get_wire()
        self.nodes.append(MLIRNode("comb.xor", [lhs, rhs], out, {}))
        return out

    def emit_mux(self, cond: str, true_val: str, false_val: str) -> str:
        """Emits a comb.mux operation (used for SC scaled addition)."""
        out = self.get_wire()
        self.nodes.append(MLIRNode("comb.mux", [cond, true_val, false_val], out, {}))
        return out

    def generate(self) -> str:
        """Generates the final MLIR string for the module."""
        lines = []
        # Modern CIRCT / MLIR HW dialect syntax
        lines.append(f"hw.module @{self.module_name}(in %clk: i1, in %rst: i1, out out: i1) {{")

        for node in self.nodes:
            ins = ", ".join(node.inputs)
            if node.op_type == "comb.and":
                lines.append(f"  {node.output} = comb.and {ins} : i1")
            elif node.op_type == "comb.xor":
                lines.append(f"  {node.output} = comb.xor {ins} : i1")
            elif node.op_type == "comb.mux":
                c, t, f = node.inputs
                lines.append(f"  {node.output} = comb.mux {c}, {t}, {f} : i1")
            elif node.op_type == "hw.instance":
                lines.append(
                    f"  {node.output} = hw.instance \"{node.attributes['sym_name']}\" @{node.attributes['module']}() -> (i1)"
                )

        # Final output assignment (taking the last node's output as an example)
        last_wire = self.nodes[-1].output if self.nodes else "0"
        lines.append(f"  hw.output {last_wire} : i1")
        lines.append("}")
        return "\n".join(lines)
