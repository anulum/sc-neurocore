# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Identifier sanitisation regressions for HDL-facing paths

from __future__ import annotations

import json

import pytest

from sc_neurocore.compiler.equation_compiler import equation_to_fpga
from sc_neurocore.compiler.mlir_emitter import MLIREmitter, MLIRNode
from sc_neurocore.export.compiler_export import CompilerExporter
from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator
from sc_neurocore.studio.project import load_project


class MockNode:
    def __init__(self, t: str, i: str, ins: list[str], out: str, **kwargs):
        self.type = t
        self.id = i
        self.inputs = ins
        self.output = out
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockGraph:
    def __init__(self, nodes: list[MockNode]):
        self.nodes = nodes


@pytest.mark.parametrize("name", ["hidden_layer_0", "_my_signal", "mzi42", "a" * 64, "ASSIGN"])
def test_sanitize_ident_accepts_valid_names(name: str):
    assert sanitize_ident(name) == name


@pytest.mark.parametrize(
    "name",
    ['hidden"; // injected', "a" * 65, "9starts_with_digit", "with-dash", "module", "always"],
)
def test_sanitize_ident_rejects_invalid_names(name: str):
    with pytest.raises(ValueError, match="Invalid identifier"):
        sanitize_ident(name)


def test_verilog_generator_rejects_malicious_layer_name():
    gen = VerilogGenerator()
    with pytest.raises(ValueError, match="layer name"):
        gen.add_layer("Dense", 'hidden"; // injected', {"n_neurons": 4})


def test_verilog_generator_rejects_malicious_module_name():
    with pytest.raises(ValueError, match="module name"):
        VerilogGenerator(module_name="test); malicious_module(")


def test_equation_to_fpga_rejects_malicious_module_name():
    with pytest.raises(ValueError, match="module name"):
        equation_to_fpga("dv/dt = I", init={"v": 0.0}, module_name="test); @trojan(")


def test_equation_to_fpga_rejects_invalid_parameter_name():
    with pytest.raises(ValueError, match="parameter name"):
        equation_to_fpga("dv/dt = I", params={"bad-name": 1.0}, init={"v": 0.0})


def test_equation_to_fpga_rejects_reserved_state_variable_name():
    with pytest.raises(ValueError, match="state variable"):
        equation_to_fpga("dmodule/dt = I", init={"module": 0.0})


def test_mlir_emitter_rejects_invalid_module_name():
    emitter = MLIREmitter("test); @trojan(")
    emitter.emit_lfsr(16, 0xACE1)
    with pytest.raises(ValueError, match="module name"):
        emitter.generate()


def test_mlir_emitter_rejects_invalid_instance_module_name():
    emitter = MLIREmitter("safe_top")
    emitter.nodes.append(
        MLIRNode(
            "hw.instance",
            [],
            "%w1",
            {"sym_name": "lfsr", "module": "bad module"},
        )
    )
    with pytest.raises(ValueError, match="module name"):
        emitter.generate()


def test_compiler_export_rejects_invalid_input_name():
    exporter = CompilerExporter()
    nodes = [MockNode("SC_AND", "m1", ["input; // bad", "input_b"], "mac_1")]
    inputs = {"input; // bad": (128,), "input_b": (128,)}
    with pytest.raises(ValueError, match="input name"):
        exporter.export_to_mlir(MockGraph(nodes), inputs)


def test_load_project_rejects_malicious_layer_name(tmp_path, monkeypatch):
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
    payload = {
        "name": "malicious",
        "saved_at": 0.0,
        "version": "0.3.0",
        "state": {
            "layers": [{"type": "Dense", "name": 'hidden"; // injected', "params": {}}],
        },
    }
    (tmp_path / "malicious.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="layer name"):
        load_project("malicious")
