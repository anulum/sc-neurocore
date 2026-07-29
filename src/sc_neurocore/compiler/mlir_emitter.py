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
import os
import shutil

# External CIRCT tools are invoked with fixed argv lists and shell=False.
import subprocess  # nosec B404
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

from ..exceptions import SCCompilerError
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
    """Generated MLIR file, optional lowered Verilog, and evidence manifest."""

    output_dir: str
    mlir_path: str
    manifest_path: str
    module_name: str
    node_count: int
    op_counts: dict[str, int]
    circt_opt_path: str | None
    verilog_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable manifest representation."""
        return asdict(self)


class MLIREmitter:
    """Translate sc-neurocore objects into MLIR text formatted for CIRCT."""

    def __init__(self, module_name: str = "sc_neurocore_top"):
        self.module_name = module_name
        self.nodes: List[MLIRNode] = []
        self._wire_counter = 0
        self._instance_counter = 0

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
        """Emit a clocked, parametric ``sc_lfsr`` instance for CIRCT lowering.

        Each call produces a uniquely named ``hw.instance`` of the parametric
        ``sc_lfsr`` extern (``WIDTH``/``SEED`` as ``i32`` module parameters,
        ``clk``/``rst`` wired from the enclosing module). Uniqueness of the
        instance symbol is required — CIRCT rejects duplicate instance names in
        one module.
        """
        out = self.get_wire()
        self._instance_counter += 1
        self.nodes.append(
            MLIRNode(
                "hw.instance",
                [],
                out,
                {
                    "sym_name": f"lfsr{self._instance_counter}",
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
        """Generate CIRCT-consumable ``hw``/``comb`` dialect MLIR for the module.

        The emitted text is valid, ``circt-opt``-verifiable MLIR: every
        instantiated sub-module is declared as a parametric ``hw.module.extern``
        before the top module, ``comb`` operations carry explicit ``i1`` types,
        and each ``hw.instance`` names its result port and binds ``clk``/``rst``
        plus its ``i32`` parameters. ``circt-opt --export-verilog`` lowers the
        result directly to Verilog (see ``tests/test_mlir_circt_roundtrip.py``).
        """
        lines: list[str] = []
        safe_module_name = sanitize_ident(self.module_name, context="module name")

        # Declare each instantiated sub-module as a parametric extern exactly
        # once, so the emitted module is self-contained and CIRCT can resolve
        # every ``hw.instance`` reference. Emitting instances of an undeclared
        # symbol makes the MLIR unverifiable.
        extern_signatures: dict[str, str] = {}
        for node in self.nodes:
            if node.op_type != "hw.instance":
                continue
            module = sanitize_ident(node.attributes["module"], context="module name")
            if module in extern_signatures:
                continue
            param_decl = ", ".join(f"{name}: i32" for name in node.attributes["parameters"])
            generic = f"<{param_decl}>" if param_decl else ""
            extern_signatures[module] = (
                f"hw.module.extern @{module}{generic}(in %clk: i1, in %rst: i1, out out: i1)"
            )
        lines.extend(extern_signatures.values())

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
                param_bind = ", ".join(
                    f"{name}: i32 = {value}"
                    for name, value in node.attributes["parameters"].items()
                )
                generic = f"<{param_bind}>" if param_bind else ""
                lines.append(
                    f'  {safe_output} = hw.instance "{sym_name}" @{module_name}{generic}'
                    "(clk: %clk: i1, rst: %rst: i1) -> (out: i1)"
                )

        if self.nodes:
            last_wire = self._sanitize_ssa_name(self.nodes[-1].output, context="signal name")
            lines.append(f"  hw.output {last_wire} : i1")
        else:
            # An empty pipeline still has to drive its single output port with a
            # typed constant — ``hw.output`` cannot reference a bare literal.
            lines.append("  %c0_i1 = hw.constant false")
            lines.append("  hw.output %c0_i1 : i1")
        lines.append("}")
        return "\n".join(lines)

    def write_bundle(
        self,
        output_dir: str | Path,
        *,
        circt_opt: str = "circt-opt",
        run_circt: bool = False,
    ) -> MLIRBundle:
        """Write MLIR plus a manifest describing CIRCT lowering readiness.

        The helper is evidence-first: by default (``run_circt=False``) it records
        whether ``circt-opt`` is available but does not run it. Set
        ``run_circt=True`` to verify and lower the module through ``circt-opt``,
        emitting Verilog and recording genuine execution evidence; it fails
        closed if the tool is missing or rejects the MLIR.
        """
        return generate_mlir_bundle(self, output_dir, circt_opt=circt_opt, run_circt=run_circt)


def _lower_with_circt(circt_opt: str, mlir_path: Path, verilog_path: Path) -> None:
    """Verify then lower MLIR to Verilog with ``circt-opt``, failing closed.

    Runs ``circt-opt --verify-diagnostics`` followed by ``circt-opt
    --export-verilog`` (the exported Verilog arrives on stdout; the ``-o`` sink
    is discarded). Raises :class:`SCCompilerError` if either step fails, so a
    bundle never records a lowering that did not actually succeed.
    """
    # circt_opt comes from shutil.which and every argument is literal or a local path.
    verify = subprocess.run(  # nosec B603
        [circt_opt, "--verify-diagnostics", str(mlir_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.returncode != 0:
        raise SCCompilerError(
            f"circt-opt --verify-diagnostics rejected the emitted MLIR:\n{verify.stderr}"
        )
    # The same resolved binary receives only literal flags and the local MLIR path.
    export = subprocess.run(  # nosec B603
        [circt_opt, "--export-verilog", str(mlir_path), "-o", os.devnull],
        capture_output=True,
        text=True,
        check=False,
    )
    if export.returncode != 0:
        raise SCCompilerError(f"circt-opt --export-verilog failed:\n{export.stderr}")
    verilog_path.write_text(export.stdout, encoding="utf-8")


def generate_mlir_bundle(
    emitter: MLIREmitter,
    output_dir: str | Path,
    *,
    circt_opt: str = "circt-opt",
    run_circt: bool = False,
) -> MLIRBundle:
    """Write a CIRCT-ready MLIR file, a manifest, and optionally lowered Verilog.

    With ``run_circt=False`` (default) the bundle is evidence-first: it records
    whether ``circt-opt`` is available but does not execute it. With
    ``run_circt=True`` it verifies the module and lowers it to Verilog through
    ``circt-opt``, writing ``<module>.v`` and recording genuine execution
    evidence in the manifest; it raises :class:`SCCompilerError` if ``circt-opt``
    is missing or rejects the MLIR, rather than claiming an un-run lowering.
    """
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    safe_module_name = sanitize_ident(emitter.module_name, context="module name")
    mlir_path = out / f"{safe_module_name}.mlir"
    manifest_path = out / "mlir_bundle_manifest.json"

    mlir_text = emitter.generate()
    mlir_path.write_text(mlir_text + "\n", encoding="utf-8")

    op_counts = dict(sorted(Counter(node.op_type for node in emitter.nodes).items()))
    circt_opt_path = shutil.which(circt_opt)

    verilog_path: Path | None = None
    if run_circt:
        if circt_opt_path is None:
            raise SCCompilerError(
                f"run_circt=True requires {circt_opt!r} on PATH; refusing to "
                "record a CIRCT lowering that was never executed."
            )
        verilog_path = out / f"{safe_module_name}.v"
        _lower_with_circt(circt_opt_path, mlir_path, verilog_path)

    executed = verilog_path is not None
    manifest = {
        "schema": "sc-neurocore.mlir_bundle_manifest.v1",
        "module_name": safe_module_name,
        "mlir_path": str(mlir_path),
        "node_count": len(emitter.nodes),
        "op_counts": op_counts,
        "dialects": ["hw", "comb", "seq"],
        "circt": {
            "tool": circt_opt,
            "tool_path": circt_opt_path,
            "available": circt_opt_path is not None,
            "executed": executed,
        },
        "claim_status": {
            "mlir_emitted": True,
            "circt_lowering_executed": executed,
            "verilog_generated_from_mlir": executed,
            "reason": (
                f"circt-opt verified the MLIR and exported Verilog to {verilog_path}."
                if executed
                else "Bundle contains CIRCT-ready MLIR text and tool-availability "
                "metadata; set run_circt=True to verify and lower it with circt-opt."
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
        circt_opt_path=circt_opt_path,
        verilog_path=str(verilog_path) if verilog_path is not None else None,
    )
