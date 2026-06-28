# SPDX-License-Identifier: AGPL-3.0-or-later
"""DCLS RTL emission and synthesis-elaboration contract."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
HDL_FILES = [
    REPO_ROOT / "hdl" / "sc_dcls_axonal_delay.v",
    REPO_ROOT / "hdl" / "sc_dcls_tent_kernel.v",
    REPO_ROOT / "hdl" / "sc_dcls_layer_core.v",
]


def test_dcls_rust_reference_and_systemverilog_emitter_contracts(cargo_lib_test) -> None:
    cargo_lib_test("dcls")


def test_dcls_layer_core_elaborates_with_yosys() -> None:
    yosys = shutil.which("yosys")
    assert yosys is not None, "yosys must be installed for DCLS RTL elaboration evidence"
    script = "; ".join(
        [
            "read_verilog -sv " + " ".join(str(path) for path in HDL_FILES),
            "hierarchy -check -top sc_dcls_layer_core",
            "proc",
            "opt",
            "stat",
        ]
    )
    completed = subprocess.run(
        [yosys, "-p", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "sc_dcls_layer_core" in completed.stdout
    assert "Number of cells:" in completed.stdout
