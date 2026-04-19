# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — TVM Relay Lowering

"""SC-IR to TVM Relay IR lowering for heterogeneous accelerator targets.

Generates Relay IR text representation from SC-NeuroCore IR graphs.
Supports target-specific schedule annotations for FPGA (Xilinx/Intel),
GPU (CUDA), and CPU backends. No TVM runtime dependency required —
emits structural IR text that can be fed to the TVM compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


class TargetDevice(Enum):
    CPU = "llvm"
    CUDA = "cuda"
    FPGA_XILINX = "vitis_ai"
    FPGA_INTEL = "aocl"


@dataclass
class TargetSchedule:
    device: TargetDevice
    opt_level: int = 3
    relay_passes: List[str] = field(
        default_factory=lambda: [
            "FoldConstant",
            "FuseOps",
            "AlterOpLayout",
        ]
    )
    sc_specific: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_fpga(cls, vendor: str = "xilinx") -> TargetSchedule:
        dev = TargetDevice.FPGA_XILINX if vendor == "xilinx" else TargetDevice.FPGA_INTEL
        return cls(
            device=dev,
            opt_level=2,
            relay_passes=["FoldConstant", "FuseOps"],
            sc_specific={
                "bitstream_packing": True,
                "lfsr_sharing": True,
                "popcount_tree": "adder_tree",
            },
        )

    @classmethod
    def for_gpu(cls) -> TargetSchedule:
        return cls(
            device=TargetDevice.CUDA,
            opt_level=3,
            relay_passes=["FoldConstant", "FuseOps", "AlterOpLayout", "CombineParallelBatchMatmul"],
            sc_specific={
                "warp_level_popcount": True,
                "shared_lfsr_bank": 32,
            },
        )

    @classmethod
    def for_cpu(cls) -> TargetSchedule:
        return cls(
            device=TargetDevice.CPU,
            opt_level=3,
        )


# Relay IR type strings
RELAY_TYPE_MAP = {
    "SC_AND": "bool",
    "SC_MUX": "bool",
    "SC_POPCOUNT": "int32",
    "LIF_MEMBRANE": "bool",
}


@dataclass
class RelayFunction:
    name: str
    params: List[Tuple[str, str]]  # (name, type_annotation)
    body_lines: List[str] = field(default_factory=list)
    ret_var: str = ""
    ret_type: str = "bool"

    def to_relay_text(self) -> str:
        sig_parts = [f"%{p[0]}: Tensor[{p[1]}]" for p in self.params]
        sig = ", ".join(sig_parts)
        lines = [f"def @{self.name}({sig}) -> Tensor[{self.ret_type}] {{"]
        for line in self.body_lines:
            lines.append(f"  {line}")
        lines.append(f"  {self.ret_var}")
        lines.append("}")
        return "\n".join(lines)


class TVMLowering:
    """Lowers SC-NeuroCore IR to TVM Relay IR text representation."""

    def __init__(self, schedule: TargetSchedule | None = None):
        self.schedule = schedule or TargetSchedule.for_cpu()

    def _shape_str(self, shape: Tuple[int, ...], dtype: str = "bool") -> str:
        dims = ", ".join(str(d) for d in shape)
        return f"({dims}), dtype={dtype}"

    def _lower_node(self, node: Any, shapes: Dict[str, Tuple[int, ...]]) -> Tuple[str, str]:
        """Returns (relay_line, output_type_str)."""
        in_refs = [f"%{inp}" for inp in node.inputs]

        if node.type == "SC_AND":
            out_shape = shapes.get(node.inputs[0], (1,))
            shapes[node.output] = out_shape
            shape_s = self._shape_str(out_shape, "bool")
            line = f"let %{node.output} = nn.bitwise_and({in_refs[0]}, {in_refs[1]}) /* Tensor[{shape_s}] */;"
            return line, "bool"

        if node.type == "SC_MUX":
            out_shape = shapes.get(node.inputs[0], (1,))
            shapes[node.output] = out_shape
            shape_s = self._shape_str(out_shape, "bool")
            line = (
                f"let %{node.output} = where({in_refs[0]}, {in_refs[1]}, {in_refs[2]}) "
                f"/* Tensor[{shape_s}] */;"
            )
            return line, "bool"

        if node.type == "SC_POPCOUNT":
            in_shape = shapes.get(node.inputs[0], (1,))
            out_shape = in_shape[:-1] + (1,) if len(in_shape) > 1 else (1,)
            shapes[node.output] = out_shape
            shape_s = self._shape_str(out_shape, "int32")
            line = (
                f'let %{node.output} = sum(cast({in_refs[0]}, dtype="int32"), axis=-1, keepdims=True) '
                f"/* Tensor[{shape_s}] */;"
            )
            return line, "int32"

        if node.type == "LIF_MEMBRANE":
            th = getattr(node, "threshold", 1.0)
            lk = getattr(node, "leak", 0.9)
            out_shape = shapes.get(node.inputs[0], (1,))
            shapes[node.output] = out_shape
            shape_s = self._shape_str(out_shape, "bool")
            line = (
                f"let %{node.output} = @scpn.lif({in_refs[0]}, "
                f"threshold={th}, leak={lk}) /* Tensor[{shape_s}] */;"
            )
            return line, "bool"

        shapes[node.output] = (1,)
        return f"let %{node.output} = {in_refs[0]}; /* passthrough */", "bool"

    def lower(
        self,
        ir_graph: Any,
        input_shapes: Dict[str, Tuple[int, ...]],
        func_name: str = "sc_forward",
    ) -> str:
        """Lower SC-IR graph to Relay IR text."""
        from sc_neurocore.export.compiler_export import CompilerExporter

        exporter = CompilerExporter()
        sorted_nodes = exporter._topological_sort(ir_graph.nodes)

        shapes = dict(input_shapes)
        params = [(name, self._shape_str(shape, "bool")) for name, shape in input_shapes.items()]

        func = RelayFunction(name=func_name, params=params)

        last_out = ""
        last_type = "bool"
        for node in sorted_nodes:
            line, dtype = self._lower_node(node, shapes)
            func.body_lines.append(line)
            last_out = f"%{node.output}"
            last_type = dtype

        func.ret_var = last_out
        func.ret_type = self._shape_str(
            shapes.get(sorted_nodes[-1].output, (1,)) if sorted_nodes else (1,),
            last_type,
        )

        # Add schedule preamble
        header_lines = [
            f"// Target: {self.schedule.device.value}",
            f"// Opt Level: {self.schedule.opt_level}",
            f"// Passes: {', '.join(self.schedule.relay_passes)}",
        ]
        if self.schedule.sc_specific:
            for k, v in self.schedule.sc_specific.items():
                header_lines.append(f"// SC Config: {k} = {v}")
        header_lines.append("")

        return "\n".join(header_lines) + func.to_relay_text()

    def emit_build_script(self, relay_text: str) -> str:
        """Generate a TVM build script stub for the lowered IR."""
        return (
            "import tvm\n"
            "from tvm import relay\n\n"
            f"target = tvm.target.Target('{self.schedule.device.value}')\n"
            f"opt_level = {self.schedule.opt_level}\n\n"
            "# Parse the relay module\n"
            "mod = relay.fromtext(relay_ir)\n\n"
            "# Build\n"
            "with tvm.transform.PassContext(opt_level=opt_level):\n"
            "    lib = relay.build(mod, target=target)\n"
        )
