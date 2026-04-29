# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for browser deployment scaffold

"""Tests for deterministic browser deployment artefact generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.cli import main
from sc_neurocore.edge import WebDeploymentConfig, build_web_deployment


def test_build_web_deployment_writes_expected_artefacts(tmp_path: Path) -> None:
    model = tmp_path / "demo.nir"
    model.write_bytes(b"nir-demo")
    output = tmp_path / "web"

    manifest = build_web_deployment(
        model,
        output,
        WebDeploymentConfig(dt=0.5, bitstream_length=128, enable_wasm_threads=True),
    )

    assert manifest.target == "web"
    assert manifest.model_format == "nir"
    assert manifest.bitstream_length == 128
    assert manifest.capabilities["webgpu"] is True
    assert manifest.capabilities["wasm_threads"] is True
    assert (output / "index.html").is_file()
    assert (output / "runtime" / "sc_neurocore_web.js").is_file()
    assert (output / "runtime" / "sc_neurocore_webgpu.wgsl").is_file()
    assert (output / "model" / "demo.nir").read_bytes() == b"nir-demo"


def test_build_web_deployment_manifest_is_deterministic_json(tmp_path: Path) -> None:
    model = tmp_path / "weights.pt"
    model.write_bytes(b"torch-demo")
    output = tmp_path / "web"

    build_web_deployment(model, output, WebDeploymentConfig(dt=1.0, bitstream_length=256))

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest) == [
        "artefacts",
        "bitstream_length",
        "capabilities",
        "dt",
        "model_format",
        "model_name",
        "runtime_contract",
        "schema_version",
        "target",
    ]
    assert manifest["artefacts"]["model"] == "model/weights.pt"
    assert manifest["runtime_contract"]["wasm_module"] is None


def test_build_web_deployment_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model file not found"):
        build_web_deployment(tmp_path / "missing.nir", tmp_path / "web")


def test_build_web_deployment_rejects_unsupported_model_suffix(tmp_path: Path) -> None:
    model = tmp_path / "weights.pkl"
    model.write_bytes(b"unsafe")

    with pytest.raises(ValueError, match="unsupported web model format"):
        build_web_deployment(model, tmp_path / "web")


def test_web_deployment_config_validates_positive_values() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        WebDeploymentConfig(dt=0)
    with pytest.raises(ValueError, match="bitstream_length must be positive"):
        WebDeploymentConfig(bitstream_length=0)


def test_cli_deploy_web_generates_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "demo.json"
    model.write_text('{"layers": []}\n', encoding="utf-8")
    output = tmp_path / "web"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sc-neurocore",
            "deploy",
            str(model),
            "--target",
            "web",
            "--output",
            str(output),
            "--dt",
            "0.25",
            "--T",
            "64",
        ],
    )

    assert main() == 0
    captured = capsys.readouterr()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "[1/1] Browser deployment scaffold generated" in captured.out
    assert manifest["dt"] == 0.25
    assert manifest["bitstream_length"] == 64
