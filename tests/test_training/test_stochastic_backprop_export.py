# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for stochastic backpropagation export manifest

"""Tests for exporting stochastic backpropagation evidence into SC-NIR metadata."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from sc_neurocore.benchmarks.stochastic_backprop import build_stochastic_backprop_benchmark
from sc_neurocore.ir.scnir_schema import SCNIR_SCHEMA_VERSION, validate_scnir_dict
from sc_neurocore.training.sc_estimators import DifferentiableSCConfig
from sc_neurocore.training.stochastic_backprop_export import (
    STOCHASTIC_BACKPROP_EVIDENCE_BOUNDARY,
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


def _config_from_benchmark(benchmark: dict) -> DifferentiableSCConfig:
    sc_config = benchmark["sc_config"]
    return DifferentiableSCConfig(
        bitstream_length=int(sc_config["bitstream_length"]),
        encoding=sc_config["encoding"],
        generator=sc_config["generator"],
        estimator=sc_config["estimator"],
        input_seed=int(sc_config["input_seed"]),
        weight_seed=int(sc_config["weight_seed"]),
        correlation=float(sc_config["correlation"]),
    )


def test_export_manifest_contains_valid_scnir_stream_metadata() -> None:
    benchmark = build_stochastic_backprop_benchmark(bitstream_length=256, steps=8)

    manifest = build_stochastic_backprop_export_manifest(benchmark, _config_from_benchmark(benchmark))
    scnir_document = manifest["scnir_document"]

    assert manifest["schema_version"] == STOCHASTIC_BACKPROP_EXPORT_SCHEMA_VERSION
    assert manifest["evidence_boundary"] == STOCHASTIC_BACKPROP_EVIDENCE_BOUNDARY
    assert manifest["scnir_schema_version"] == SCNIR_SCHEMA_VERSION
    assert manifest["training"]["estimator"] == "pathwise_relaxation"
    assert manifest["training"]["joint_design"]["enabled"] is True
    assert manifest["training"]["joint_design"]["selected_bitstream_length"] == 256
    assert manifest["evidence"]["benchmark_schema_version"] == benchmark["schema_version"]
    assert manifest["evidence"]["evidence_boundary"] == STOCHASTIC_BACKPROP_EVIDENCE_BOUNDARY
    assert manifest["evidence"]["hardware_measurement_claimed"] is False
    assert (
        manifest["evidence"]["sampled_product_mae"]
        == benchmark["stream_evidence"]["sampled_product_mae"]
    )
    assert manifest["evidence"]["estimator_variance"]["sample_count"] == benchmark[
        "estimator_variance"
    ]["sample_count"]
    assert (
        manifest["evidence"]["estimator_variance"]["estimators"]["score_function"]["variance"]
        > 0.0
    )
    validate_scnir_dict(scnir_document)

    stream_by_id = {stream["stream_id"]: stream for stream in scnir_document["streams"]}
    assert set(stream_by_id) == {
        "stochastic_backprop.input",
        "stochastic_backprop.weight",
        "stochastic_backprop.product",
    }
    assert stream_by_id["stochastic_backprop.input"]["source"]["kind"] == "sobol"
    assert stream_by_id["stochastic_backprop.input"]["source"]["seed"] == 101
    assert stream_by_id["stochastic_backprop.weight"]["signal_kind"] == "weight"
    assert stream_by_id["stochastic_backprop.weight"]["source"]["seed"] == 211
    assert stream_by_id["stochastic_backprop.product"]["source"]["kind"] == "replay"
    assert (
        stream_by_id["stochastic_backprop.product"]["correlation_constraints"][0]["peer_stream_id"]
        == "stochastic_backprop.input"
    )


def test_export_manifest_rejects_config_benchmark_mismatch() -> None:
    benchmark = build_stochastic_backprop_benchmark(bitstream_length=128, steps=8)

    with pytest.raises(ValueError, match="bitstream_length"):
        build_stochastic_backprop_export_manifest(benchmark, _config())


def test_export_manifest_rejects_unbounded_benchmark_evidence() -> None:
    benchmark = build_stochastic_backprop_benchmark(bitstream_length=256, steps=8)
    benchmark["evidence_boundary"] = "physical_hardware_measurement"

    with pytest.raises(ValueError, match="evidence_boundary"):
        build_stochastic_backprop_export_manifest(
            benchmark,
            _config_from_benchmark(benchmark),
        )


def test_write_export_manifest_writes_canonical_json(tmp_path) -> None:
    benchmark = build_stochastic_backprop_benchmark(bitstream_length=256, steps=8)
    output = tmp_path / "stochastic_backprop_export.json"
    config = _config_from_benchmark(benchmark)

    path = write_stochastic_backprop_export_manifest(output, benchmark, config)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == output
    assert payload == build_stochastic_backprop_export_manifest(benchmark, config)
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_write_handoff_bundle_materialises_auditable_scnir_hdl_directory(tmp_path) -> None:
    from sc_neurocore.ir.scnir_handoff_audit import audit_scnir_hdl_handoff

    benchmark = build_stochastic_backprop_benchmark(bitstream_length=256, steps=8)
    manifest = build_stochastic_backprop_export_manifest(benchmark, _config_from_benchmark(benchmark))
    output_dir = tmp_path / "handoff"

    report = write_stochastic_backprop_handoff_bundle(manifest, output_dir)

    assert report.module_name == "stochastic_backprop_handoff"
    assert report.stream_count == 3
    assert report.source_module_count == 3
    assert report.signal_kinds == {"spike": 2, "weight": 1}
    assert "scnir_src_002_stochastic_backprop_product.v" in report.artefacts
    assert audit_scnir_hdl_handoff(output_dir).as_dict() == report.as_dict()

    parity = json.loads(
        (output_dir / "stochastic_backprop_trained_design_parity.json").read_text(
            encoding="utf-8"
        )
    )
    assert parity["SPDX-License-Identifier"] == "AGPL-3.0-or-later"
    assert parity["hardware_measurement_claimed"] is False
    assert parity["selected_bitstream_length"] == manifest["training"]["bitstream_length"]
    assert parity["selected_encoding"] == manifest["training"]["encoding"]
    assert parity["expected_bitstream_length_q16"] == round(
        manifest["training"]["joint_design"]["expected_bitstream_length"] * 65536.0
    )

    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("iverilog/vvp are required for executable HDL parity")

    testbench = tmp_path / "trained_design_tb.v"
    testbench.write_text(
        "\n".join(
            [
                "module trained_design_tb;",
                "  wire [31:0] selected_bitstream_length;",
                "  wire [31:0] expected_bitstream_length_q16;",
                "  wire signed [31:0] correlation_q16;",
                "  wire encoding_bipolar;",
                "  stochastic_backprop_trained_design dut (",
                "    .selected_bitstream_length(selected_bitstream_length),",
                "    .expected_bitstream_length_q16(expected_bitstream_length_q16),",
                "    .correlation_q16(correlation_q16),",
                "    .encoding_bipolar(encoding_bipolar)",
                "  );",
                "  initial begin",
                f"    if (selected_bitstream_length !== 32'd{parity['selected_bitstream_length']}) $fatal(1);",
                f"    if (expected_bitstream_length_q16 !== 32'd{parity['expected_bitstream_length_q16']}) $fatal(2);",
                f"    if (correlation_q16 !== 32'sd{parity['correlation_q16']}) $fatal(3);",
                f"    if (encoding_bipolar !== 1'b{1 if parity['selected_encoding'] == 'bipolar' else 0}) $fatal(4);",
                '    $display("trained-design-parity-pass");',
                "  end",
                "endmodule",
                "",
            ]
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "trained_design_tb"
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-o",
            str(executable),
            str(output_dir / "stochastic_backprop_trained_design.v"),
            str(testbench),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    simulation = subprocess.run(
        [vvp, str(executable)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "trained-design-parity-pass" in simulation.stdout
