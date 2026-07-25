# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCOnnxExporter protobuf graph contracts

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.export.onnx_exporter import SCOnnxExporter
from tests.test_export.onnx_exporter_support import (
    BareLayer,
    DenseWeightedLayer,
    DummyLayer,
    make_layers,
)

onnx = pytest.importorskip("onnx")


def test_protobuf_export_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(make_layers(), str(path))
    assert path.exists()
    assert path.stat().st_size > 0


def test_protobuf_roundtrip_loads(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(make_layers(), str(path))
    model = onnx.load(str(path))
    assert model.producer_name == "sc-neurocore"


def test_protobuf_node_count(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(make_layers(), str(path))
    assert len(onnx.load(str(path)).graph.node) == 2


def test_protobuf_custom_domain(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(make_layers(), str(path))
    assert onnx.load(str(path)).graph.node[0].domain == "sc_neurocore"


def test_protobuf_op_type_dense(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(make_layers(), str(path))
    assert onnx.load(str(path)).graph.node[0].op_type == "SC_Dense"


def test_protobuf_op_type_custom(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export([DummyLayer(n_inputs=2)], str(path))
    assert onnx.load(str(path)).graph.node[0].op_type == "SC_Custom"


def test_protobuf_uses_default_attributes_when_layer_fields_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export([BareLayer(n_inputs=2)], str(path))
    model = onnx.load(str(path))
    attrs = {
        attribute.name: getattr(attribute, "i", None) for attribute in model.graph.node[0].attribute
    }
    assert attrs["n_neurons"] == -1
    assert attrs["length"] == 256


def test_protobuf_embeds_weights(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(make_layers(), str(path))
    initializer_names = [tensor.name for tensor in onnx.load(str(path)).graph.initializer]
    assert "Layer_0_weights" in initializer_names


def test_protobuf_input_output_names(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    SCOnnxExporter.export(make_layers(), str(path))
    model = onnx.load(str(path))
    assert model.graph.input[0].name == "input_0"
    assert model.graph.output[0].name == "output_1"


def test_protobuf_export_raises_dependency_error_when_onnx_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "missing.onnx"
    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnx":
            raise ImportError("onnx not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(Exception, match="requires onnx"):
        SCOnnxExporter._export_protobuf(make_layers(), str(out))


def test_protobuf_export_embeds_dense_weights_initializer(tmp_path: Path) -> None:
    """Weighted dense layers become ONNX initializers."""

    out = tmp_path / "dense.onnx"
    SCOnnxExporter.export([DenseWeightedLayer(3, 2)], str(out))

    model = onnx.load(str(out))
    node = model.graph.node[0]
    assert node.op_type == "SC_Dense"
    assert "Layer_0_weights" in node.input
    assert "Layer_0_weights" in {item.name for item in model.graph.initializer}
