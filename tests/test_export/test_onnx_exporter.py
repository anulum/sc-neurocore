# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCOnnxExporter JSON and sidecar outputs

"""Tests for SCOnnxExporter JSON and sidecar outputs."""

import json
import os
import time
import builtins

import numpy as np
import pytest

from sc_neurocore.export.onnx_exporter import SCOnnxExporter
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


class DummyLayer:
    """Simple layer without Dense/Vectorized name for op_type testing."""

    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
        self.n_neurons = 3
        self.length = 8


class BareLayer:
    """Layer exposing only n_inputs for default-attribute branch coverage."""

    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs


def _make_layers():
    np.random.seed(0)
    layer1 = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=8)
    layer2 = VectorizedSCLayer(n_inputs=2, n_neurons=1, length=8)
    return [layer1, layer2]


def test_onnx_export_writes_json(tmp_path):
    """Export should create JSON file."""
    layers = _make_layers()
    path = tmp_path / "model.json"
    SCOnnxExporter.export(layers, str(path))
    assert path.exists()


def test_onnx_export_node_count(tmp_path):
    """Node count should match number of layers."""
    layers = _make_layers()
    path = tmp_path / "model.json"
    SCOnnxExporter.export(layers, str(path))
    data = json.loads(path.read_text())
    assert len(data["nodes"]) == 2


def test_onnx_export_input_shape(tmp_path):
    """Input shape should use first layer n_inputs."""
    layers = _make_layers()
    path = tmp_path / "model.json"
    SCOnnxExporter.export(layers, str(path))
    data = json.loads(path.read_text())
    assert data["inputs"][0]["shape"][1] == layers[0].n_inputs


def test_onnx_export_output_name(tmp_path):
    """Output name should be last node output."""
    layers = _make_layers()
    path = tmp_path / "model.json"
    SCOnnxExporter.export(layers, str(path))
    data = json.loads(path.read_text())
    assert data["outputs"][0]["name"] == "output_1"


def test_onnx_export_attributes_present(tmp_path):
    """Nodes should include n_neurons and length attributes."""
    layers = _make_layers()
    path = tmp_path / "model.json"
    SCOnnxExporter.export(layers, str(path))
    data = json.loads(path.read_text())
    attrs = data["nodes"][0]["attributes"]
    assert "n_neurons" in attrs
    assert "length" in attrs


def test_onnx_export_writes_weight_sidecar(tmp_path):
    """Weights should be saved as .npy files when present."""
    layers = _make_layers()
    path = tmp_path / "model.json"
    SCOnnxExporter.export(layers, str(path))
    sidecar = tmp_path / "model.json_layer_0_weights.npy"
    assert sidecar.exists()


def test_onnx_export_op_type_dense(tmp_path):
    """Vectorized or Dense layers should use SC_Dense op_type."""
    layers = _make_layers()
    path = tmp_path / "model.json"
    SCOnnxExporter.export(layers, str(path))
    data = json.loads(path.read_text())
    assert data["nodes"][0]["op_type"] == "SC_Dense"


def test_onnx_export_custom_op_type(tmp_path):
    """Non-dense layers should use SC_Custom op_type."""
    dummy = DummyLayer(n_inputs=2)
    path = tmp_path / "model.json"
    SCOnnxExporter.export([dummy], str(path))
    data = json.loads(path.read_text())
    assert data["nodes"][0]["op_type"] == "SC_Custom"


def test_onnx_export_uses_default_attributes_when_layer_fields_missing(tmp_path):
    bare = BareLayer(n_inputs=2)
    path = tmp_path / "model.json"
    SCOnnxExporter.export([bare], str(path))
    data = json.loads(path.read_text())
    attrs = data["nodes"][0]["attributes"]
    assert attrs["n_neurons"] == -1
    assert attrs["length"] == 256
    assert "has_weights" not in attrs


def test_onnx_export_json_schema_fields(tmp_path):
    """Export should include expected top-level fields."""
    layers = _make_layers()
    path = tmp_path / "model.json"
    SCOnnxExporter.export(layers, str(path))
    data = json.loads(path.read_text())
    assert {"producer_name", "producer_version", "nodes", "inputs", "outputs"} <= set(data.keys())


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_onnx_export_perf_small(tmp_path):
    """Benchmark exporting a small model."""
    layers = _make_layers()
    path = tmp_path / "model.json"
    start = time.perf_counter()
    SCOnnxExporter.export(layers, str(path))
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0


# ── Protobuf export tests ───────────────────────────────────────

onnx = pytest.importorskip("onnx")


def test_protobuf_export_creates_file(tmp_path):
    layers = _make_layers()
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(layers, str(path))
    assert path.exists()
    assert path.stat().st_size > 0


def test_protobuf_roundtrip_loads(tmp_path):
    layers = _make_layers()
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(layers, str(path))
    model = onnx.load(str(path))
    assert model.producer_name == "sc-neurocore"


def test_protobuf_node_count(tmp_path):
    layers = _make_layers()
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(layers, str(path))
    model = onnx.load(str(path))
    assert len(model.graph.node) == 2


def test_protobuf_custom_domain(tmp_path):
    layers = _make_layers()
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(layers, str(path))
    model = onnx.load(str(path))
    assert model.graph.node[0].domain == "sc_neurocore"


def test_protobuf_op_type_dense(tmp_path):
    layers = _make_layers()
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(layers, str(path))
    model = onnx.load(str(path))
    assert model.graph.node[0].op_type == "SC_Dense"


def test_protobuf_op_type_custom(tmp_path):
    dummy = DummyLayer(n_inputs=2)
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export([dummy], str(path))
    model = onnx.load(str(path))
    assert model.graph.node[0].op_type == "SC_Custom"


def test_protobuf_uses_default_attributes_when_layer_fields_missing(tmp_path):
    bare = BareLayer(n_inputs=2)
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export([bare], str(path))
    model = onnx.load(str(path))
    attrs = {a.name: getattr(a, "i", None) for a in model.graph.node[0].attribute}
    assert attrs["n_neurons"] == -1
    assert attrs["length"] == 256


def test_protobuf_embeds_weights(tmp_path):
    layers = _make_layers()
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(layers, str(path))
    model = onnx.load(str(path))
    init_names = [t.name for t in model.graph.initializer]
    assert "Layer_0_weights" in init_names


def test_protobuf_input_output_names(tmp_path):
    layers = _make_layers()
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(layers, str(path))
    model = onnx.load(str(path))
    assert model.graph.input[0].name == "input_0"
    assert model.graph.output[0].name == "output_1"


def test_protobuf_export_raises_dependency_error_when_onnx_missing(monkeypatch, tmp_path):
    layers = _make_layers()
    out = tmp_path / "missing.onnx"
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("onnx not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(Exception, match="requires onnx"):
        SCOnnxExporter._export_protobuf(layers, str(out))


def test_json_export_surfaces_oserror(monkeypatch, tmp_path):
    layers = _make_layers()
    out = tmp_path / "denied.json"

    def _failing_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(builtins, "open", _failing_open)
    with pytest.raises(OSError, match="disk full"):
        SCOnnxExporter._export_json(layers, str(out))


class DenseWeightedLayer:
    """Dense-named layer exposing weights so the weight-emitting branches run."""

    def __init__(self, n_inputs: int, n_neurons: int):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.length = 8
        self.weights = np.ones((n_inputs, n_neurons), dtype=np.float32)


def test_json_export_emits_weight_sidecar_for_dense_layer(tmp_path):
    """A Dense layer with weights is classified SC_Dense and gets a weight sidecar."""
    out = tmp_path / "dense.json"
    SCOnnxExporter.export([DenseWeightedLayer(3, 2)], str(out))

    graph = json.loads(out.read_text())
    node = graph["nodes"][0]

    assert node["op_type"] == "SC_Dense"
    assert node["attributes"]["has_weights"] is True
    sidecar = node["attributes"]["weights_file"]
    assert os.path.exists(sidecar)
    np.testing.assert_array_equal(np.load(sidecar), np.ones((3, 2), dtype=np.float32))


def test_protobuf_export_embeds_dense_weights_initializer(tmp_path):
    """A Dense layer with weights is exported as an ONNX initializer in the protobuf graph."""
    onnx = pytest.importorskip("onnx")
    out = tmp_path / "dense.onnx"

    SCOnnxExporter.export([DenseWeightedLayer(3, 2)], str(out))

    model = onnx.load(str(out))
    node = model.graph.node[0]
    assert node.op_type == "SC_Dense"
    assert "Layer_0_weights" in node.input
    initializer_names = {init.name for init in model.graph.initializer}
    assert "Layer_0_weights" in initializer_names
