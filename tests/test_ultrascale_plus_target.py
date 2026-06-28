# SPDX-License-Identifier: AGPL-3.0-or-later
"""UltraScale+ target compiler and Vivado-project generator contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from gen_vivado_project import generate_tcl, load_manifest


def test_ultrascale_plus_rust_target_contracts(cargo_lib_test) -> None:
    completed = cargo_lib_test("ultrascale_plus")
    assert completed.returncode == 0


def test_gen_vivado_project_emits_zu3eg_tcl_without_unverified_dsp_claims(tmp_path: Path) -> None:
    source = tmp_path / "top.sv"
    source.write_text(
        "module top(input wire clk, input wire rst_n, output wire out); assign out = clk & rst_n; endmodule\n",
        encoding="utf-8",
    )
    xdc = tmp_path / "zu3eg.xdc"
    xdc.write_text("create_clock -period 4.000 [get_ports clk]\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "top": "top",
                "sku": "zu3eg",
                "clock_mhz": 250,
                "sources": ["top.sv"],
                "xdc": ["zu3eg.xdc"],
                "output_dir": "out/zu3eg",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_manifest(manifest)
    tcl = generate_tcl(loaded)

    assert "set PART xczu3eg-sbva484-1-e" in tcl
    assert "set CLOCK_PERIOD_NS 4.000000" in tcl
    assert f"read_verilog -sv {source}" in tcl
    assert f"read_xdc {xdc}" in tcl
    assert "report_timing_summary" in tcl
    assert "write_bitstream" in tcl
    assert "DSP" + "58" not in tcl


def test_gen_vivado_project_rejects_unknown_sku(tmp_path: Path) -> None:
    source = tmp_path / "top.sv"
    source.write_text("module top; endmodule\n", encoding="utf-8")
    xdc = tmp_path / "base.xdc"
    xdc.write_text("create_clock -period 4.000 [get_ports clk]\n", encoding="utf-8")
    manifest = tmp_path / "bad.json"
    manifest.write_text(
        json.dumps(
            {
                "top": "top",
                "sku": "vck190",
                "clock_mhz": 250,
                "sources": ["top.sv"],
                "xdc": ["base.xdc"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_manifest(manifest)
    except ValueError as exc:
        assert "sku must be one of" in str(exc)
    else:
        raise AssertionError("unknown UltraScale+ SKU must be rejected")
