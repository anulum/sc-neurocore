# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for stochastic backpropagation benchmark tool

"""Tests for the stochastic backpropagation benchmark CLI."""

from __future__ import annotations

import json

from tools import stochastic_backprop_benchmark


def test_stochastic_backprop_benchmark_cli_writes_report(tmp_path, capsys) -> None:
    output = tmp_path / "stochastic_backprop.json"
    export_manifest = tmp_path / "stochastic_backprop_export.json"
    estimator_regression_manifest = tmp_path / "stochastic_backprop_estimator_regression.json"
    handoff_dir = tmp_path / "handoff"

    exit_code = stochastic_backprop_benchmark.main(
        [
            "--output",
            str(output),
            "--export-manifest",
            str(export_manifest),
            "--estimator-regression-manifest",
            str(estimator_regression_manifest),
            "--handoff-dir",
            str(handoff_dir),
            "--bitstream-length",
            "128",
            "--steps",
            "16",
            "--learning-rate",
            "0.4",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads(export_manifest.read_text(encoding="utf-8"))
    estimator_manifest = json.loads(estimator_regression_manifest.read_text(encoding="utf-8"))
    audit_report = json.loads((handoff_dir / "stochastic_backprop_handoff_audit.json").read_text())
    assert exit_code == 0
    assert str(output) in captured.out
    assert str(export_manifest) in captured.out
    assert str(estimator_regression_manifest) in captured.out
    assert str(handoff_dir) in captured.out
    assert payload["sc_config"]["bitstream_length"] == 128
    assert payload["training"]["steps"] == 16
    assert manifest["training"]["bitstream_length"] == 128
    assert manifest["scnir_document"]["streams"][1]["signal_kind"] == "weight"
    assert audit_report["status"] == "valid"
    assert audit_report["stream_count"] == 3
    joint_design = payload["joint_design"]
    assert joint_design["enabled"] is True
    assert (
        joint_design["final"]["selected_bitstream_length"]
        == payload["sc_config"]["bitstream_length"]
    )
    assert joint_design["final"]["selected_encoding"] == payload["sc_config"]["encoding"]
    assert joint_design["final"]["correlation"] == payload["sc_config"]["correlation"]
    assert (
        joint_design["final"]["expected_bitstream_length"]
        > joint_design["initial"]["expected_bitstream_length"]
    )
    assert (
        joint_design["final"]["length_probabilities"]
        != joint_design["initial"]["length_probabilities"]
    )
    assert manifest["training"]["joint_design"]["enabled"] is True
    assert manifest["training"]["joint_design"]["selected_bitstream_length"] == 128
    assert estimator_manifest["status"] == "pass"
    assert estimator_manifest["bitstream_lengths"] == [64, 128, 256]


def test_stochastic_backprop_benchmark_cli_reports_invalid_input(tmp_path, capsys) -> None:
    output = tmp_path / "invalid.json"

    exit_code = stochastic_backprop_benchmark.main(
        [
            "--output",
            str(output),
            "--bitstream-length",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "stochastic backpropagation benchmark invalid" in captured.err
    assert not output.exists()
