# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tests for stochastic backpropagation export manifest

"""Tests for exporting stochastic backpropagation evidence into SC-NIR metadata."""

from __future__ import annotations

import json

import pytest

from sc_neurocore.benchmarks.stochastic_backprop import build_stochastic_backprop_benchmark
from sc_neurocore.ir.scnir_schema import SCNIR_SCHEMA_VERSION, validate_scnir_dict
from sc_neurocore.training.sc_estimators import DifferentiableSCConfig
from sc_neurocore.training.stochastic_backprop_export import (
    STOCHASTIC_BACKPROP_EXPORT_SCHEMA_VERSION,
    build_stochastic_backprop_export_manifest,
    write_stochastic_backprop_handoff_bundle,
    write_stochastic_backprop_export_manifest,
)


def _config() -> DifferentiableSCConfig:
    return DifferentiableSCConfig(
        bitstream_length=256,
        encoding="bipolar",
        generator="sobol",
        estimator="pathwise_relaxation",
        input_seed=101,
        weight_seed=211,
        correlation=0.0,
    )


def test_export_manifest_contains_valid_scnir_stream_metadata() -> None:
    benchmark = build_stochastic_backprop_benchmark(bitstream_length=256, steps=8)

    manifest = build_stochastic_backprop_export_manifest(benchmark, _config())
    scnir_document = manifest["scnir_document"]

    assert manifest["schema_version"] == STOCHASTIC_BACKPROP_EXPORT_SCHEMA_VERSION
    assert manifest["scnir_schema_version"] == SCNIR_SCHEMA_VERSION
    assert manifest["training"]["estimator"] == "pathwise_relaxation"
    assert manifest["evidence"]["benchmark_schema_version"] == benchmark["schema_version"]
    assert (
        manifest["evidence"]["sampled_product_mae"]
        == benchmark["stream_evidence"]["sampled_product_mae"]
    )
    validate_scnir_dict(scnir_document)

    stream_by_id = {stream["stream_id"]: stream for stream in scnir_document["streams"]}
    assert set(stream_by_id) == {"fmoat1.input", "fmoat1.weight", "fmoat1.product"}
    assert stream_by_id["fmoat1.input"]["source"]["kind"] == "sobol"
    assert stream_by_id["fmoat1.input"]["source"]["seed"] == 101
    assert stream_by_id["fmoat1.weight"]["signal_kind"] == "weight"
    assert stream_by_id["fmoat1.weight"]["source"]["seed"] == 211
    assert stream_by_id["fmoat1.product"]["source"]["kind"] == "replay"
    assert (
        stream_by_id["fmoat1.product"]["correlation_constraints"][0]["peer_stream_id"]
        == "fmoat1.input"
    )


def test_export_manifest_rejects_config_benchmark_mismatch() -> None:
    benchmark = build_stochastic_backprop_benchmark(bitstream_length=128, steps=8)

    with pytest.raises(ValueError, match="bitstream_length"):
        build_stochastic_backprop_export_manifest(benchmark, _config())


def test_write_export_manifest_writes_canonical_json(tmp_path) -> None:
    benchmark = build_stochastic_backprop_benchmark(bitstream_length=256, steps=8)
    output = tmp_path / "stochastic_backprop_export.json"

    path = write_stochastic_backprop_export_manifest(output, benchmark, _config())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == output
    assert payload == build_stochastic_backprop_export_manifest(benchmark, _config())
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_write_handoff_bundle_materialises_auditable_scnir_hdl_directory(tmp_path) -> None:
    from sc_neurocore.ir.scnir_handoff_audit import audit_scnir_hdl_handoff

    benchmark = build_stochastic_backprop_benchmark(bitstream_length=256, steps=8)
    manifest = build_stochastic_backprop_export_manifest(benchmark, _config())
    output_dir = tmp_path / "handoff"

    report = write_stochastic_backprop_handoff_bundle(manifest, output_dir)

    assert report.module_name == "stochastic_backprop_handoff"
    assert report.stream_count == 3
    assert report.source_module_count == 3
    assert report.signal_kinds == {"spike": 2, "weight": 1}
    assert "scnir_src_002_fmoat1_product.v" in report.artefacts
    assert audit_scnir_hdl_handoff(output_dir).as_dict() == report.as_dict()
