# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MLPerf-SC runner tool

"""CLI tests for the lightweight MLPerf-SC fixture runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sc_neurocore.benchmarks import validate_mlperf_sc_result


def test_mlperf_sc_run_tool_writes_valid_fixture_result(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/mlperf_sc_run.py",
            "--output",
            str(tmp_path),
            "--seed",
            "5",
            "--bitstream-length",
            "32",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    assert "mlperf_sc_result.json" in result.stdout
    result_path = tmp_path / "mlperf_sc_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_mlperf_sc_result(payload, artifact_root=tmp_path)
    assert payload["execution"]["bitstream_length"] == 32


def test_mlperf_sc_run_tool_writes_external_reference_fixture(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/mlperf_sc_run.py",
            "--output",
            str(tmp_path),
            "--model",
            "fixture_external_majority",
            "--seed",
            "5",
            "--bitstream-length",
            "32",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    payload = json.loads((tmp_path / "mlperf_sc_result.json").read_text(encoding="utf-8"))
    validate_mlperf_sc_result(payload, artifact_root=tmp_path)
    assert payload["run"]["producer"] == "external-reference-fixture"
    assert payload["execution"]["sc_mode"] == "deterministic_replay"


def test_mlperf_sc_run_tool_rejects_invalid_bitstream_length(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/mlperf_sc_run.py",
            "--output",
            str(tmp_path),
            "--bitstream-length",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 1
    assert "bitstream_length" in result.stderr
    assert not (tmp_path / "mlperf_sc_result.json").exists()
