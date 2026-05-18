# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for side-channel HDL hook tool

"""CLI tests for protected side-channel encoding HDL hook generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_side_channel_hdl_emit_tool_writes_verilog_and_manifest(tmp_path: Path) -> None:
    verilog_path = tmp_path / "rtl" / "sc_side_channel_hook.v"
    manifest_path = tmp_path / "rtl" / "sc_side_channel_hook_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/side_channel_hdl_emit.py",
            "--verilog-output",
            str(verilog_path),
            "--manifest-output",
            str(manifest_path),
            "--module-name",
            "sc_side_channel_hook",
            "--probability",
            "0.25",
            "--bitstream-length",
            "8",
            "--seed",
            "5",
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
    assert str(verilog_path) in result.stdout
    assert str(manifest_path) in result.stdout
    assert "module sc_side_channel_hook" in verilog_path.read_text(encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "sc-neurocore.side-channel-hdl-hook.v0.1"
    assert manifest["module_name"] == "sc_side_channel_hook"
    assert manifest["verilog_path"] == str(verilog_path)
    assert manifest["evidence_boundary"] == "analytic_simulation_only"
    assert manifest["dummy_streams"] == 1


def test_side_channel_hdl_emit_tool_rejects_invalid_probability(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/side_channel_hdl_emit.py",
            "--verilog-output",
            str(tmp_path / "out.v"),
            "--manifest-output",
            str(tmp_path / "out.json"),
            "--probability",
            "1.5",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 1
    assert "side-channel HDL hook invalid" in result.stderr
