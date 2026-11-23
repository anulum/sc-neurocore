# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (hdl_compile) from former test_kuramoto_rtl.py

"""Compilation of the emitted Kuramoto RTL."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import (
    Path,
)
from sc_neurocore.hdl_gen import (
    KuramotoEmitter,
)
from sc_neurocore.hdl_gen.verilog_generator import (
    VerilogGenerator,
)


def test_verilog_generator_can_emit_kuramoto_phase() -> None:
    """The Verilog generator emits a Kuramoto phase module with its oscillators."""
    gen = VerilogGenerator(module_name="kuramoto_wrap")
    code = gen.emit_kuramoto_phase(
        n_oscillators=2,
        omegas=[0.95, 1.05],
        initial_phases=[0.0, 0.1],
        coupling=0.05,
    )
    assert "module kuramoto_wrap" in code
    assert "localparam integer N_OSC = 2;" in code
    assert "wire signed [DATA_WIDTH-1:0] phase_velocity_0" in code


def test_kuramoto_emitter_smoke_compiles_with_iverilog(tmp_path: Path) -> None:
    """A four-oscillator Kuramoto module compiles with Icarus Verilog."""
    iverilog = shutil.which("iverilog")
    if iverilog is None:
        raise AssertionError("iverilog must be available for HDL smoke tests")

    emitter = KuramotoEmitter(
        module_name="kuramoto_compile",
        n_oscillators=4,
        omegas=[0.8, 1.0, 1.1, 1.3],
        initial_phases=[0.0, 0.3, 0.6, 0.9],
        coupling=0.12,
        dt=5e-3,
    )
    rtl_path = tmp_path / "kuramoto_compile.v"
    rtl_path.write_text(emitter.generate())

    result = subprocess.run(
        [iverilog, "-g2012", "-t", "null", str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_kuramoto_emitter_large_lut_compiles_without_index_truncation(tmp_path: Path) -> None:
    """A 512-entry sine table compiles without truncating its index."""
    iverilog = shutil.which("iverilog")
    if iverilog is None:
        raise AssertionError("iverilog must be available for HDL smoke tests")

    emitter = KuramotoEmitter(
        module_name="kuramoto_large_lut",
        n_oscillators=2,
        lut_size=512,
    )
    rtl_path = tmp_path / "kuramoto_large_lut.v"
    rtl_path.write_text(emitter.generate())

    result = subprocess.run(
        [iverilog, "-g2012", "-t", "null", str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Numeric constant truncated" not in result.stderr
