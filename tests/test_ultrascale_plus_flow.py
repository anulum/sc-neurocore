# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vivado-gated UltraScale+ batch-flow contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ZU3EG = REPO_ROOT / "hdl" / "targets" / "ultrascale_plus" / "build_zu3eg.tcl"
BUILD_ZU9EG = REPO_ROOT / "hdl" / "targets" / "ultrascale_plus" / "build_zu9eg.tcl"
ZU3EG_XDC = REPO_ROOT / "hdl" / "targets" / "ultrascale_plus" / "zu3eg.xdc"
ZU9EG_XDC = REPO_ROOT / "hdl" / "targets" / "ultrascale_plus" / "zu9eg.xdc"


def test_ultrascale_plus_flow_files_avoid_unverified_pin_locations() -> None:
    for path in [ZU3EG_XDC, ZU9EG_XDC]:
        text = path.read_text(encoding="utf-8")
        assert "create_clock" in text
        assert "set_property PACKAGE_PIN" not in text
        assert "set_property LOC" not in text
    for path in [BUILD_ZU3EG, BUILD_ZU9EG]:
        text = path.read_text(encoding="utf-8")
        assert "source $PROJECT_TCL" in text


def test_ultrascale_plus_vivado_ci_builds_zu3eg_bitstream(tmp_path: Path) -> None:
    if os.environ.get("MIF_VIVADO_CI") != "1":
        pytest.skip("NEU-C.1 Vivado-CI gated: set MIF_VIVADO_CI=1 on a Vivado 2024.2 runner")
    vivado = shutil.which("vivado")
    if vivado is None:
        pytest.skip("NEU-C.1 Vivado-CI gated: Vivado executable is not installed on this host")

    source = tmp_path / "top.sv"
    source.write_text(
        "module top(input wire clk, input wire rst_n, output wire out); assign out = clk & rst_n; endmodule\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "top": "top",
                "sku": "zu3eg",
                "clock_mhz": 250,
                "sources": [str(source)],
                "xdc": [str(ZU3EG_XDC)],
                "output_dir": str(tmp_path / "out"),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated_tcl = tmp_path / "generated.tcl"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "gen_vivado_project.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(generated_tcl),
        ],
        check=True,
    )
    subprocess.run(
        [vivado, "-mode", "batch", "-source", str(BUILD_ZU3EG), "-tclargs", str(generated_tcl)],
        cwd=tmp_path,
        check=True,
    )
