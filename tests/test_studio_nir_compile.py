# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio server-side NIR -> FPGA compilation

"""Server-side NIR ingest: a .nir graph lowered to FPGA Verilog via the Studio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nir")
pytest.importorskip("fastapi")

import nir
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.nir_compile import compile_nir_file_bytes, compile_nir_graph


def _two_layer_graph() -> nir.NIRGraph:
    rng = np.random.default_rng(0)
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "fc1": nir.Affine(weight=rng.normal(size=(3, 2)), bias=np.zeros(3)),
            "lif1": nir.LIF(
                tau=np.full(3, 10.0), r=np.ones(3), v_leak=np.zeros(3), v_threshold=np.ones(3)
            ),
            "output": nir.Output(output_type={"output": np.array([3])}),
        },
        edges=[("input", "fc1"), ("fc1", "lif1"), ("lif1", "output")],
    )


def _nir_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "model.nir"
    nir.write(str(path), _two_layer_graph())
    return path.read_bytes()


def test_compile_nir_graph_returns_verilog_artefacts() -> None:
    artefacts = compile_nir_graph(_two_layer_graph(), module_name="snn_demo")

    assert artefacts["module_name"] == "snn_demo"
    assert artefacts["total_neurons"] == 3
    assert "lif" in artefacts["neuron_modules"]
    assert "module snn_demo" in artefacts["top_module"]
    assert "module" in artefacts["weight_rom"]
    assert artefacts["source_modules"]
    assert artefacts["q_format"]


def test_compile_nir_file_bytes_reads_hdf5_and_compiles(tmp_path: Path) -> None:
    artefacts = compile_nir_file_bytes(_nir_bytes(tmp_path), module_name="from_file")

    assert artefacts["module_name"] == "from_file"
    assert artefacts["total_neurons"] == 3
    assert "module from_file" in artefacts["top_module"]


def test_compile_nir_file_bytes_rejects_empty_upload() -> None:
    with pytest.raises(ValueError, match="Empty NIR upload"):
        compile_nir_file_bytes(b"")


def test_api_nir_compile_endpoint_lowers_uploaded_graph(tmp_path: Path) -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")
    data = _nir_bytes(tmp_path)

    response = client.post(
        "/api/nir/compile",
        params={"module_name": "api_snn"},
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["module_name"] == "api_snn"
    assert body["total_neurons"] == 3
    assert "module api_snn" in body["top_module"]
    assert "lif" in body["neuron_modules"]


def test_api_nir_compile_endpoint_rejects_empty_upload() -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")

    response = client.post(
        "/api/nir/compile",
        content=b"",
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 422
