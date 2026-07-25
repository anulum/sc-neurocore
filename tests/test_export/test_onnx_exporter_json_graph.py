# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCOnnxExporter JSON graph contracts

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sc_neurocore.export.onnx_exporter import SCOnnxExporter
from tests.test_export.onnx_exporter_support import (
    BareLayer,
    DenseWeightedLayer,
    DummyLayer,
    make_layers,
)


def test_onnx_export_writes_json(tmp_path: Path) -> None:
    """Export should create a JSON file."""

    path = tmp_path / "model.json"
    SCOnnxExporter.export(make_layers(), str(path))
    assert path.exists()


def test_onnx_export_node_count(tmp_path: Path) -> None:
    """Node count should match the number of layers."""

    path = tmp_path / "model.json"
    SCOnnxExporter.export(make_layers(), str(path))
    data = json.loads(path.read_text())
    assert len(data["nodes"]) == 2


def test_onnx_export_input_shape(tmp_path: Path) -> None:
    """Input shape should use the first layer's input count."""

    layers = make_layers()
    path = tmp_path / "model.json"
    SCOnnxExporter.export(layers, str(path))
    data = json.loads(path.read_text())
    assert data["inputs"][0]["shape"][1] == layers[0].n_inputs


def test_onnx_export_output_name(tmp_path: Path) -> None:
    """Output name should be the last node output."""

    path = tmp_path / "model.json"
    SCOnnxExporter.export(make_layers(), str(path))
    data = json.loads(path.read_text())
    assert data["outputs"][0]["name"] == "output_1"


def test_onnx_export_attributes_present(tmp_path: Path) -> None:
    """Nodes should include neuron-count and stream-length attributes."""

    path = tmp_path / "model.json"
    SCOnnxExporter.export(make_layers(), str(path))
    attrs = json.loads(path.read_text())["nodes"][0]["attributes"]
    assert "n_neurons" in attrs
    assert "length" in attrs


def test_onnx_export_writes_weight_sidecar(tmp_path: Path) -> None:
    """Weights should be saved as NumPy sidecars when present."""

    path = tmp_path / "model.json"
    SCOnnxExporter.export(make_layers(), str(path))
    assert (tmp_path / "model.json_layer_0_weights.npy").exists()


def test_onnx_export_op_type_dense(tmp_path: Path) -> None:
    """Vectorized or dense layers should use the SC_Dense op type."""

    path = tmp_path / "model.json"
    SCOnnxExporter.export(make_layers(), str(path))
    data = json.loads(path.read_text())
    assert data["nodes"][0]["op_type"] == "SC_Dense"


def test_onnx_export_custom_op_type(tmp_path: Path) -> None:
    """Non-dense layers should use the SC_Custom op type."""

    path = tmp_path / "model.json"
    SCOnnxExporter.export([DummyLayer(n_inputs=2)], str(path))
    data = json.loads(path.read_text())
    assert data["nodes"][0]["op_type"] == "SC_Custom"


def test_onnx_export_uses_default_attributes_when_layer_fields_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.json"
    SCOnnxExporter.export([BareLayer(n_inputs=2)], str(path))
    attrs = json.loads(path.read_text())["nodes"][0]["attributes"]
    assert attrs["n_neurons"] == -1
    assert attrs["length"] == 256
    assert "has_weights" not in attrs


def test_onnx_export_json_schema_fields(tmp_path: Path) -> None:
    """Export should include the expected top-level fields."""

    path = tmp_path / "model.json"
    SCOnnxExporter.export(make_layers(), str(path))
    data = json.loads(path.read_text())
    assert {"producer_name", "producer_version", "nodes", "inputs", "outputs"} <= set(data)


def test_json_export_surfaces_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "denied.json"

    def _failing_open(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(builtins, "open", _failing_open)
    with pytest.raises(OSError, match="disk full"):
        SCOnnxExporter._export_json(make_layers(), str(out))


def test_json_export_emits_weight_sidecar_for_dense_layer(tmp_path: Path) -> None:
    """A weighted dense layer is classified and gets a weight sidecar."""

    out = tmp_path / "dense.json"
    SCOnnxExporter.export([DenseWeightedLayer(3, 2)], str(out))

    node = json.loads(out.read_text())["nodes"][0]
    assert node["op_type"] == "SC_Dense"
    assert node["attributes"]["has_weights"] is True
    sidecar = node["attributes"]["weights_file"]
    assert os.path.exists(sidecar)
    np.testing.assert_array_equal(np.load(sidecar), np.ones((3, 2), dtype=np.float32))
