"""Tests for SCOnnxExporter JSON and sidecar outputs."""

import json
import os
import time

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
