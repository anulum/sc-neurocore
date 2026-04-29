# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Export pipeline contract tests

"""Contract tests for the end-to-end export pipeline orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sc_neurocore.export import pipeline as pipeline_module


class _Plugin:
    def meta(self) -> SimpleNamespace:
        return SimpleNamespace(name="lif_contract", state_variables=["v", "i"])

    def default_state(self) -> dict[str, float]:
        return {"v": 0.0, "i": 0.0}

    def default_params(self) -> dict[str, float]:
        return {"threshold": 1.0, "leak": 0.95}


class _Registry:
    def __init__(self, plugin: _Plugin | None = None) -> None:
        self.plugin = plugin
        self.lookups: list[str] = []

    def get(self, name: str) -> _Plugin | None:
        self.lookups.append(name)
        return self.plugin


def test_pipeline_runs_all_stages_with_consistent_ir(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid plugin is threaded through Verilog, ONNX, TVM, and MLIR stages."""
    seen: dict[str, object] = {}

    class VerilogGenerator:
        def generate(self, plugin: _Plugin) -> str:
            seen["verilog_plugin"] = plugin
            return "module lif_contract_network; endmodule"

    class ONNXExporter:
        def export(self, graph: object, shapes: dict[str, tuple[int, ...]]) -> str:
            seen["onnx_nodes"] = [node.name for node in graph.nodes]
            seen["onnx_shapes"] = shapes
            return '{"graph": "lif_contract"}'

    class TVMLowering:
        def __init__(self, schedule: object | None = None) -> None:
            seen["tvm_schedule"] = schedule

        def lower(self, graph: object, shapes: dict[str, tuple[int, ...]]) -> str:
            seen["tvm_layer_attrs"] = graph.nodes[1].attrs
            seen["tvm_shapes"] = shapes
            return "relay lif_contract"

    class CompilerExporter:
        def __init__(self, target: str) -> None:
            seen["compiler_target"] = target

        def export_to_mlir(self, graph: object, shapes: dict[str, tuple[int, ...]]) -> str:
            seen["mlir_output_node"] = graph.nodes[-1].inputs
            seen["mlir_shapes"] = shapes
            return "module { scpn.lif }"

    monkeypatch.setattr(pipeline_module, "VerilogGenerator", VerilogGenerator)
    monkeypatch.setattr(pipeline_module, "ONNXExporter", ONNXExporter)
    monkeypatch.setattr(pipeline_module, "TVMLowering", TVMLowering)
    monkeypatch.setattr(pipeline_module, "CompilerExporter", CompilerExporter)

    registry = _Registry(_Plugin())
    result = pipeline_module.ExportPipeline(registry=registry, target="fpga-small").run(
        "lif_contract",
        n_neurons=3,
        bitstream_length=8,
        module_name="contract_top",
    )

    assert result.success is True
    assert [stage.stage for stage in result.stages] == [
        "model_zoo",
        "verilog",
        "onnx",
        "tvm_relay",
        "mlir",
    ]
    assert registry.lookups == ["lif_contract", "lif_contract"]
    assert result.verilog.startswith("module lif_contract")
    assert result.onnx_json == '{"graph": "lif_contract"}'
    assert result.relay_text == "relay lif_contract"
    assert result.mlir_text == "module { scpn.lif }"

    graph = result.stages[0].metadata["ir_graph"]
    assert graph.name == "lif_contract_network"
    assert [(node.name, node.type, node.inputs) for node in graph.nodes] == [
        ("input", "sc_input", []),
        ("lif_contract_layer", "sc_neuron", ["input"]),
        ("output", "sc_output", ["lif_contract_layer"]),
    ]
    assert graph.nodes[0].attrs["shape"] == (3, 8)
    assert graph.nodes[1].attrs == {
        "neuron_type": "lif_contract",
        "n_neurons": 3,
        "bitstream_length": 8,
    }
    assert graph.nodes[2].attrs["shape"] == (3,)
    assert result.stages[1].metadata == {
        "n_neurons": 3,
        "bitstream_length": 8,
        "module_name": "contract_top",
    }
    assert seen == {
        "verilog_plugin": registry.plugin,
        "onnx_nodes": ["input", "lif_contract_layer", "output"],
        "onnx_shapes": {"input": (3, 8)},
        "tvm_schedule": "fpga-small",
        "tvm_layer_attrs": {
            "neuron_type": "lif_contract",
            "n_neurons": 3,
            "bitstream_length": 8,
        },
        "tvm_shapes": {"input": (3, 8)},
        "compiler_target": "mlir",
        "mlir_output_node": ["lif_contract_layer"],
        "mlir_shapes": {"input": (3, 8)},
    }


def test_pipeline_missing_plugin_returns_single_failed_stage() -> None:
    result = pipeline_module.ExportPipeline(registry=_Registry(None)).run("missing_neuron")

    assert result.success is False
    assert len(result.stages) == 1
    assert result.stages[0].stage == "model_zoo"
    assert result.stages[0].success is False
    assert "missing_neuron" in result.stages[0].output


def test_pipeline_reports_model_zoo_plugin_exceptions() -> None:
    class BrokenPlugin(_Plugin):
        def meta(self) -> SimpleNamespace:
            raise RuntimeError("metadata unavailable")

    result = pipeline_module.ExportPipeline(registry=_Registry(BrokenPlugin())).run("broken")

    assert result.success is False
    assert len(result.stages) == 1
    assert result.stages[0].stage == "model_zoo"
    assert result.stages[0].output == "metadata unavailable"


def test_pipeline_stage_verilog_reports_missing_plugin() -> None:
    pipeline = pipeline_module.ExportPipeline(registry=_Registry(None))

    stage = pipeline._stage_verilog("missing", 4, 16, "top")

    assert stage.stage == "verilog"
    assert stage.success is False
    assert "missing" in stage.output


def test_pipeline_converts_stage_exceptions_to_failed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenVerilogGenerator:
        def generate(self, plugin: _Plugin) -> str:
            raise RuntimeError("verilog backend rejected the plugin")

    class ONNXExporter:
        def export(self, graph: object, shapes: dict[str, tuple[int, ...]]) -> str:
            return "onnx still produced"

    class TVMLowering:
        def __init__(self, schedule: object | None = None) -> None:
            pass

        def lower(self, graph: object, shapes: dict[str, tuple[int, ...]]) -> str:
            return "relay still produced"

    class CompilerExporter:
        def __init__(self, target: str) -> None:
            pass

        def export_to_mlir(self, graph: object, shapes: dict[str, tuple[int, ...]]) -> str:
            return "mlir still produced"

    monkeypatch.setattr(pipeline_module, "VerilogGenerator", BrokenVerilogGenerator)
    monkeypatch.setattr(pipeline_module, "ONNXExporter", ONNXExporter)
    monkeypatch.setattr(pipeline_module, "TVMLowering", TVMLowering)
    monkeypatch.setattr(pipeline_module, "CompilerExporter", CompilerExporter)

    result = pipeline_module.ExportPipeline(registry=_Registry(_Plugin())).run("lif_contract")

    assert result.success is False
    assert result.stages[1].stage == "verilog"
    assert result.stages[1].success is False
    assert result.stages[1].output == "verilog backend rejected the plugin"
    assert result.stages[2].success is True
    assert result.stages[3].success is True
    assert result.stages[4].success is True


def test_pipeline_exporter_stage_failures_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenONNXExporter:
        def export(self, graph: object, shapes: dict[str, tuple[int, ...]]) -> str:
            raise RuntimeError("onnx shape mismatch")

    class BrokenTVMLowering:
        def __init__(self, schedule: object | None = None) -> None:
            pass

        def lower(self, graph: object, shapes: dict[str, tuple[int, ...]]) -> str:
            raise RuntimeError("relay lowering failed")

    class BrokenCompilerExporter:
        def __init__(self, target: str) -> None:
            pass

        def export_to_mlir(self, graph: object, shapes: dict[str, tuple[int, ...]]) -> str:
            raise RuntimeError("mlir export failed")

    monkeypatch.setattr(pipeline_module, "ONNXExporter", BrokenONNXExporter)
    monkeypatch.setattr(pipeline_module, "TVMLowering", BrokenTVMLowering)
    monkeypatch.setattr(pipeline_module, "CompilerExporter", BrokenCompilerExporter)

    pipeline = pipeline_module.ExportPipeline(registry=_Registry(_Plugin()))
    graph = pipeline._stage_model_zoo("lif_contract", 2, 4).metadata["ir_graph"]

    onnx = pipeline._stage_onnx(graph, 2, 4)
    tvm = pipeline._stage_tvm(graph, 2, 4)
    mlir = pipeline._stage_mlir(graph, 2, 4)

    assert (onnx.stage, onnx.success, onnx.output) == ("onnx", False, "onnx shape mismatch")
    assert (tvm.stage, tvm.success, tvm.output) == ("tvm_relay", False, "relay lowering failed")
    assert (mlir.stage, mlir.success, mlir.output) == ("mlir", False, "mlir export failed")


def test_pipeline_summary_reports_success_and_failure_marks() -> None:
    result = pipeline_module.PipelineResult(
        stages=[
            pipeline_module.PipelineStageResult("ok", True, "abc"),
            pipeline_module.PipelineStageResult("bad", False, "error"),
        ]
    )

    summary = result.summary()

    assert "Export Pipeline Result" in summary
    assert "[✓] ok: 3 chars" in summary
    assert "[✗] bad: 5 chars" in summary
