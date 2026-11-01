# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""DCLS RTL emission and synthesis-elaboration contract."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
HDL_FILES = [
    REPO_ROOT / "hdl" / "sc_dcls_axonal_delay.v",
    REPO_ROOT / "hdl" / "sc_dcls_tent_kernel.v",
    REPO_ROOT / "hdl" / "sc_dcls_layer_core.v",
]


def test_dcls_rust_reference_and_systemverilog_emitter_contracts(
    cargo_lib_test: Callable[[str], subprocess.CompletedProcess[str]],
) -> None:
    """Execute the real Rust reference and maintained SystemVerilog emitter contracts."""
    cargo_lib_test("dcls")


def test_dcls_layer_core_elaborates_with_yosys(tmp_path: Path) -> None:
    """Elaborate the complete layer and inspect actual structured native statistics."""
    yosys = shutil.which("yosys")
    assert yosys is not None, "yosys must be installed for DCLS RTL elaboration evidence"
    statistics = tmp_path / "statistics.json"
    script = "; ".join(
        [
            "read_verilog -sv " + " ".join(str(path) for path in HDL_FILES),
            "hierarchy -check -top sc_dcls_layer_core",
            "proc",
            "opt",
            f"tee -o {statistics.name} stat -json -top sc_dcls_layer_core",
        ]
    )
    completed = subprocess.run(
        [yosys, "-p", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "sc_dcls_layer_core" in completed.stdout
    module = json.loads(statistics.read_text())["modules"]["\\sc_dcls_layer_core"]
    assert module["num_cells"] > 0
    assert module["num_wires"] > 0
    assert module["num_processes"] == 0
