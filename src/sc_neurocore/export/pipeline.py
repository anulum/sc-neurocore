# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-End Export Pipeline

"""One-command pipeline: Neuron ODE → ONNX → TVM Relay → MLIR/SSA → SystemVerilog.

Chains Model Zoo neuron plugins through the full compiler stack to produce
deployable hardware from a neuron model specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sc_neurocore.model_zoo.model_zoo import (
    PluginRegistry,
    VerilogGenerator,
)
from sc_neurocore.export.onnx_export import ONNXExporter
from sc_neurocore.export.tvm_lowering import TVMLowering, TargetSchedule
from sc_neurocore.export.compiler_export import CompilerExporter


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage."""

    stage: str
    success: bool
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Full pipeline result."""

    stages: List[PipelineStageResult] = field(default_factory=list)
    verilog: str = ""
    onnx_json: str = ""
    relay_text: str = ""
    mlir_text: str = ""

    @property
    def success(self) -> bool:
        return all(s.success for s in self.stages)

    def summary(self) -> str:
        lines = ["Export Pipeline Result"]
        for s in self.stages:
            status = "✓" if s.success else "✗"
            lines.append(f"  [{status}] {s.stage}: {len(s.output)} chars")
        return "\n".join(lines)


class ExportPipeline:
    """End-to-end: NeuronPlugin → ONNX → TVM → MLIR → Verilog.

    Usage::

        pipeline = ExportPipeline()
        result = pipeline.run("LIF", n_neurons=64, bitstream_length=256)
        print(result.verilog)
    """

    def __init__(
        self,
        registry: Optional[PluginRegistry] = None,
        target: Optional[TargetSchedule] = None,
    ):
        self.registry = registry or PluginRegistry()
        self.target = target

    def run(
        self,
        neuron_name: str,
        n_neurons: int = 64,
        bitstream_length: int = 256,
        module_name: str = "sc_exported_network",
    ) -> PipelineResult:
        """Execute the full export pipeline for a neuron model."""
        result = PipelineResult()

        # Stage 1: Model Zoo → IR graph
        stage1 = self._stage_model_zoo(neuron_name, n_neurons, bitstream_length)
        result.stages.append(stage1)
        if not stage1.success:
            return result

        ir_graph = stage1.metadata.get("ir_graph")

        # Stage 2: IR → Verilog (direct path)
        stage2 = self._stage_verilog(neuron_name, n_neurons, bitstream_length, module_name)
        result.stages.append(stage2)
        result.verilog = stage2.output

        # Stage 3: IR → ONNX
        stage3 = self._stage_onnx(ir_graph, n_neurons, bitstream_length)
        result.stages.append(stage3)
        result.onnx_json = stage3.output

        # Stage 4: IR → TVM Relay
        stage4 = self._stage_tvm(ir_graph, n_neurons, bitstream_length)
        result.stages.append(stage4)
        result.relay_text = stage4.output

        # Stage 5: IR → MLIR/SSA
        stage5 = self._stage_mlir(ir_graph, n_neurons, bitstream_length)
        result.stages.append(stage5)
        result.mlir_text = stage5.output

        return result

    def _stage_model_zoo(
        self,
        neuron_name: str,
        n_neurons: int,
        bitstream_length: int,
    ) -> PipelineStageResult:
        """Stage 1: Look up neuron plugin and validate."""
        try:
            plugin = self.registry.get(neuron_name)
            if plugin is None:
                return PipelineStageResult(
                    stage="model_zoo",
                    success=False,
                    output=f"Neuron '{neuron_name}' not found in registry",
                )
            meta = plugin.meta()
            state = plugin.default_state()
            params = plugin.default_params()

            # Build a simple IR-like description
            ir_graph = _build_ir_graph(neuron_name, n_neurons, bitstream_length, meta)

            return PipelineStageResult(
                stage="model_zoo",
                success=True,
                output=f"Loaded {meta.name} ({meta.ode_order}-order ODE, {n_neurons} neurons)",
                metadata={"ir_graph": ir_graph, "plugin": plugin},
            )
        except Exception as e:
            return PipelineStageResult(
                stage="model_zoo",
                success=False,
                output=str(e),
            )

    def _stage_verilog(
        self,
        neuron_name: str,
        n_neurons: int,
        bitstream_length: int,
        module_name: str,
    ) -> PipelineStageResult:
        """Stage 2: Generate SystemVerilog."""
        try:
            gen = VerilogGenerator()
            verilog = gen.emit(
                neuron_type=neuron_name,
                n_neurons=n_neurons,
                bitstream_length=bitstream_length,
                module_name=module_name,
            )
            return PipelineStageResult(
                stage="verilog",
                success=True,
                output=verilog,
            )
        except Exception as e:
            return PipelineStageResult(
                stage="verilog",
                success=False,
                output=str(e),
            )

    def _stage_onnx(
        self,
        ir_graph: Any,
        n_neurons: int,
        bitstream_length: int,
    ) -> PipelineStageResult:
        """Stage 3: Export to ONNX JSON."""
        try:
            exporter = ONNXExporter()
            onnx_json = exporter.export(ir_graph)
            return PipelineStageResult(
                stage="onnx",
                success=True,
                output=onnx_json,
            )
        except Exception as e:
            return PipelineStageResult(
                stage="onnx",
                success=False,
                output=str(e),
            )

    def _stage_tvm(
        self,
        ir_graph: Any,
        n_neurons: int,
        bitstream_length: int,
    ) -> PipelineStageResult:
        """Stage 4: Lower to TVM Relay."""
        try:
            lowering = TVMLowering(schedule=self.target)
            shapes = {
                "input": (n_neurons, bitstream_length),
            }
            relay_text = lowering.lower(ir_graph, shapes)
            return PipelineStageResult(
                stage="tvm_relay",
                success=True,
                output=relay_text,
            )
        except Exception as e:
            return PipelineStageResult(
                stage="tvm_relay",
                success=False,
                output=str(e),
            )

    def _stage_mlir(
        self,
        ir_graph: Any,
        n_neurons: int,
        bitstream_length: int,
    ) -> PipelineStageResult:
        """Stage 5: Export to MLIR/SSA."""
        try:
            exporter = CompilerExporter(target="mlir")
            shapes = {"input": (n_neurons, bitstream_length)}
            mlir_text = exporter.export_to_mlir(ir_graph, shapes)
            return PipelineStageResult(
                stage="mlir",
                success=True,
                output=mlir_text,
            )
        except Exception as e:
            return PipelineStageResult(
                stage="mlir",
                success=False,
                output=str(e),
            )


def _build_ir_graph(
    neuron_name: str,
    n_neurons: int,
    bitstream_length: int,
    meta: Any,
) -> Any:
    """Build a lightweight IR graph from neuron plugin metadata."""

    class IRNode:
        def __init__(self, name, node_type, inputs=None, attrs=None):
            self.name = name
            self.type = node_type
            self.inputs = inputs or []
            self.attrs = attrs or {}

    class IRGraph:
        def __init__(self):
            self.nodes = []
            self.name = f"{neuron_name}_network"

    g = IRGraph()
    g.nodes.append(
        IRNode(
            "input",
            "sc_input",
            attrs={"shape": (n_neurons, bitstream_length)},
        )
    )
    g.nodes.append(
        IRNode(
            f"{neuron_name}_layer",
            "sc_neuron",
            inputs=["input"],
            attrs={
                "neuron_type": neuron_name,
                "n_neurons": n_neurons,
                "bitstream_length": bitstream_length,
            },
        )
    )
    g.nodes.append(
        IRNode(
            "output",
            "sc_output",
            inputs=[f"{neuron_name}_layer"],
            attrs={"shape": (n_neurons,)},
        )
    )
    return g
