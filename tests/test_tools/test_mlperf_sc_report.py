# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MLPerf-SC report tool

"""CLI tests for MLPerf-SC result aggregation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sc_neurocore.benchmarks import run_mlperf_sc_fixture


def test_mlperf_sc_report_tool_aggregates_valid_results(tmp_path: Path) -> None:
    first = run_mlperf_sc_fixture(output_dir=tmp_path / "a", seed=3, bitstream_length=32)
    second = run_mlperf_sc_fixture(output_dir=tmp_path / "b", seed=4, bitstream_length=64)
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/mlperf_sc_report.py",
            "--output",
            str(report_path),
            str(second),
            str(first),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    assert str(report_path) in result.stdout
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["result_count"] == 2
    assert [row["run_id"] for row in payload["results"]] == [
        "synthetic_sc_xor-fixture_sc_linear-seed3-l32",
        "synthetic_sc_xor-fixture_sc_linear-seed4-l64",
    ]


def test_mlperf_sc_report_tool_rejects_invalid_result(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/mlperf_sc_report.py",
            "--output",
            str(report_path),
            str(invalid),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 1
    assert "MLPerf-SC aggregation invalid" in result.stderr
    assert not report_path.exists()
