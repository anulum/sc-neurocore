# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for side-channel benchmark tool

"""CLI tests for analytic side-channel benchmark report generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_side_channel_benchmark_tool_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "side_channel_benchmark.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/side_channel_benchmark.py",
            "--output",
            str(output),
            "--probabilities",
            "0.25,0.5",
            "--labels",
            "0,1",
            "--bitstream-length",
            "16",
            "--seed",
            "3",
            "--dummy-streams-per-record",
            "1",
            "--max-dummy-overhead-ratio",
            "1.0",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    assert str(output) in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evidence_boundary"] == "analytic_simulation_only"
    assert payload["report"]["protected"]["dummy_stream_overhead_ratio"] == 1.0
    assert payload["report"]["max_class_mean_gap_reduction"] > 0.0


def test_side_channel_benchmark_tool_rejects_mismatched_labels(tmp_path: Path) -> None:
    output = tmp_path / "side_channel_benchmark.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/side_channel_benchmark.py",
            "--output",
            str(output),
            "--probabilities",
            "0.25,0.5",
            "--labels",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 1
    assert "side-channel benchmark invalid" in result.stderr
    assert not output.exists()
