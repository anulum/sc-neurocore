# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tests for stochastic backpropagation benchmark tool

"""Tests for the stochastic backpropagation benchmark CLI."""

from __future__ import annotations

import json

from tools import stochastic_backprop_benchmark


def test_stochastic_backprop_benchmark_cli_writes_report(tmp_path, capsys) -> None:
    output = tmp_path / "stochastic_backprop.json"
    export_manifest = tmp_path / "stochastic_backprop_export.json"
    handoff_dir = tmp_path / "handoff"

    exit_code = stochastic_backprop_benchmark.main(
        [
            "--output",
            str(output),
            "--export-manifest",
            str(export_manifest),
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
    audit_report = json.loads((handoff_dir / "stochastic_backprop_handoff_audit.json").read_text())
    assert exit_code == 0
    assert str(output) in captured.out
    assert str(export_manifest) in captured.out
    assert str(handoff_dir) in captured.out
    assert payload["sc_config"]["bitstream_length"] == 128
    assert payload["training"]["steps"] == 16
    assert manifest["training"]["bitstream_length"] == 128
    assert manifest["scnir_document"]["streams"][1]["signal_kind"] == "weight"
    assert audit_report["status"] == "valid"
    assert audit_report["stream_count"] == 3


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
