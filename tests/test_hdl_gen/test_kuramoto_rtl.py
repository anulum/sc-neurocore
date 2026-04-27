# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for research-stage Kuramoto HDL emission

from __future__ import annotations

import shutil
import subprocess

from sc_neurocore.hdl_gen import KuramotoEmitter
from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator


def test_kuramoto_emitter_has_expected_ports_and_helpers() -> None:
    emitter = KuramotoEmitter(
        module_name="kuramoto_top",
        n_oscillators=3,
        omegas=[0.9, 1.0, 1.1],
        initial_phases=[0.0, 0.2, 0.4],
    )
    code = emitter.generate()
    assert "module kuramoto_top" in code
    assert "input wire step_en" in code
    assert "output reg update_done" in code
    assert "function automatic signed [DATA_WIDTH-1:0] sin_lut;" in code
    assert "wire signed [DATA_WIDTH-1:0] phase_diff_0_1" in code
    assert "assign phase_bus[71:48] = phase_reg_2;" in code


def test_kuramoto_emitter_rejects_configuration_mismatch() -> None:
    try:
        KuramotoEmitter(
            n_oscillators=3,
            omegas=[1.0, 1.1],
            initial_phases=[0.0, 0.1, 0.2],
        )
    except ValueError as exc:
        assert "omegas length must equal n_oscillators" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for omega length mismatch")


def test_verilog_generator_can_emit_kuramoto_phase() -> None:
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


def test_kuramoto_emitter_smoke_compiles_with_iverilog(tmp_path) -> None:
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
