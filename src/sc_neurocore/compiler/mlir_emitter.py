# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

from ..hdl_gen._ident import sanitize_ident


@dataclass
class MLIRNode:
    """Operation record emitted into the dependency-free MLIR text builder."""

    op_type: str
    inputs: List[str]
    output: str
    attributes: dict[str, Any]


@dataclass(frozen=True)
class MLIRBundle:
    """Generated MLIR file and evidence manifest."""

    output_dir: str
    mlir_path: str
    manifest_path: str
    module_name: str
    node_count: int
    op_counts: dict[str, int]
    firtool_path: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable manifest representation."""
        return asdict(self)


class MLIREmitter:
    """Translate sc-neurocore objects into MLIR text formatted for CIRCT."""

    def __init__(self, module_name: str = "sc_neurocore_top"):
        self.module_name = module_name
        self.nodes: List[MLIRNode] = []
        self._wire_counter = 0

    def get_wire(self) -> str:
        """Allocate the next SSA wire name for emitted MLIR operations."""
        self._wire_counter += 1
        return f"%w{self._wire_counter}"

    def _sanitize_ssa_name(self, name: str, context: str) -> str:
        ident = name[1:] if name.startswith("%") else name
        return f"%{sanitize_ident(ident, context=context)}"

    def emit_and(self, lhs: str, rhs: str) -> str:
        """Emit a comb.and operation for stochastic multiplication."""
        out = self.get_wire()
        self.nodes.append(MLIRNode("comb.and", [lhs, rhs], out, {}))
        return out

    def emit_lfsr(self, width: int, seed: int) -> str:
        """Emit an LFSR instance placeholder for CIRCT lowering."""
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
        """Emit a comb.xor operation."""
        out = self.get_wire()
        self.nodes.append(MLIRNode("comb.xor", [lhs, rhs], out, {}))
        return out

    def emit_mux(self, cond: str, true_val: str, false_val: str) -> str:
        """Emit a comb.mux operation for SC scaled addition."""
        out = self.get_wire()
        self.nodes.append(MLIRNode("comb.mux", [cond, true_val, false_val], out, {}))
        return out

    def generate(self) -> str:
        """Generate the final MLIR string for the module."""
        lines = []
        safe_module_name = sanitize_ident(self.module_name, context="module name")
        # Modern CIRCT / MLIR HW dialect syntax
        lines.append(f"hw.module @{safe_module_name}(in %clk: i1, in %rst: i1, out out: i1) {{")

        for node in self.nodes:
            ins = ", ".join(
                self._sanitize_ssa_name(inp, context="signal name") for inp in node.inputs
            )
            safe_output = self._sanitize_ssa_name(node.output, context="signal name")
            if node.op_type == "comb.and":
                lines.append(f"  {safe_output} = comb.and {ins} : i1")
            elif node.op_type == "comb.xor":
                lines.append(f"  {safe_output} = comb.xor {ins} : i1")
            elif node.op_type == "comb.mux":
                c, t, f = node.inputs
                lines.append(
                    f"  {safe_output} = comb.mux "
                    f"{self._sanitize_ssa_name(c, context='signal name')}, "
                    f"{self._sanitize_ssa_name(t, context='signal name')}, "
                    f"{self._sanitize_ssa_name(f, context='signal name')} : i1"
                )
            elif node.op_type == "hw.instance":
                sym_name = sanitize_ident(node.attributes["sym_name"], context="signal name")
                module_name = sanitize_ident(node.attributes["module"], context="module name")
                lines.append(f'  {safe_output} = hw.instance "{sym_name}" @{module_name}() -> (i1)')

        # Final output assignment (taking the last node's output as an example)
        last_wire = (
            self._sanitize_ssa_name(self.nodes[-1].output, context="signal name")
            if self.nodes
            else "0"
        )
        lines.append(f"  hw.output {last_wire} : i1")
        lines.append("}")
        return "\n".join(lines)

    def write_bundle(
        self,
        output_dir: str | Path,
        *,
        firtool: str = "firtool",
        run_circt: bool = False,
    ) -> MLIRBundle:
        """Write MLIR plus a manifest describing CIRCT lowering readiness.

        The helper is intentionally evidence-first: by default it records
        whether ``firtool`` is available, but does not run it. Set
        ``run_circt=True`` only after wiring a controlled external-tool runner.
        """
        return generate_mlir_bundle(self, output_dir, firtool=firtool, run_circt=run_circt)


def generate_mlir_bundle(
    emitter: MLIREmitter,
    output_dir: str | Path,
    *,
    firtool: str = "firtool",
    run_circt: bool = False,
) -> MLIRBundle:
    """Write a CIRCT-ready MLIR file and reproducibility manifest."""
    if run_circt:
        raise NotImplementedError(
            "CIRCT execution is not launched by generate_mlir_bundle yet; "
            "use the manifest firtool_path to run external lowering explicitly."
        )

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    safe_module_name = sanitize_ident(emitter.module_name, context="module name")
    mlir_path = out / f"{safe_module_name}.mlir"
    manifest_path = out / "mlir_bundle_manifest.json"

    mlir_text = emitter.generate()
    mlir_path.write_text(mlir_text + "\n", encoding="utf-8")

    op_counts = dict(sorted(Counter(node.op_type for node in emitter.nodes).items()))
    firtool_path = shutil.which(firtool)
    manifest = {
        "schema": "sc-neurocore.mlir_bundle_manifest.v1",
        "module_name": safe_module_name,
        "mlir_path": str(mlir_path),
        "node_count": len(emitter.nodes),
        "op_counts": op_counts,
        "dialects": ["hw", "comb", "seq"],
        "circt": {
            "firtool": firtool,
            "firtool_path": firtool_path,
            "available": firtool_path is not None,
            "executed": False,
        },
        "claim_status": {
            "mlir_emitted": True,
            "circt_lowering_executed": False,
            "verilog_generated_from_mlir": False,
            "reason": (
                "Bundle contains CIRCT-ready MLIR text and tool availability "
                "metadata; downstream Verilog/EDA claims require an attached "
                "firtool/OpenROAD execution record."
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return MLIRBundle(
        output_dir=str(out),
        mlir_path=str(mlir_path),
        manifest_path=str(manifest_path),
        module_name=safe_module_name,
        node_count=len(emitter.nodes),
        op_counts=op_counts,
        firtool_path=firtool_path,
    )
