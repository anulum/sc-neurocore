# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end co-simulation of the KuramotoStep IR lowering

"""End-to-end co-simulation of the ``sc.kuramoto_step`` IR lowering.

The IR graph is built through the Rust engine's Python bindings, emitted to
SystemVerilog with :meth:`ScGraph.emit_sv`, instantiated against the hand-written
``hdl/sc_kuramoto_step.v`` core, and simulated with Icarus Verilog. The observed
next-phase vector is compared bit-for-bit against a fixed-point oracle that mirrors
the emitter quantisation and the RTL datapath exactly, and against the ideal float
Kuramoto Euler step within the sine-LUT resolution.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import pytest

pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)

from sc_neurocore_engine.ir import ScGraphBuilder

# Fixed-point contract baked into hdl/sc_kuramoto_step.v: signed Q8.16, 64-entry LUT.
DATA_WIDTH = 24
FRACTION = 16
LUT_SIZE = 64
SCALE = 1 << FRACTION
PHASE_MODULUS = round(2.0 * math.pi * SCALE)  # 411775
HALF_PHASE_MODULUS = round(math.pi * SCALE)  # 205887

REPO_ROOT = Path(__file__).resolve().parent.parent
KURAMOTO_HDL = REPO_ROOT / "hdl" / "sc_kuramoto_step.v"


def _to_signed(value: int, bits: int) -> int:
    """Reinterpret the low ``bits`` of ``value`` as a two's-complement integer."""
    value &= (1 << bits) - 1
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _quantise(value: float) -> int:
    """Quantise a real value into signed Q8.16 (matches emit_sv's kuramoto_fixed)."""
    return round(value * SCALE)


def _wrap_phase(phase: int) -> int:
    phase = _to_signed(phase, DATA_WIDTH)
    if phase >= PHASE_MODULUS:
        phase -= PHASE_MODULUS
    elif phase < 0:
        phase += PHASE_MODULUS
    return _to_signed(phase, DATA_WIDTH)


def _wrap_delta(delta: int) -> int:
    delta = _to_signed(delta, DATA_WIDTH)
    if delta > HALF_PHASE_MODULUS:
        delta -= PHASE_MODULUS
    elif delta < -HALF_PHASE_MODULUS:
        delta += PHASE_MODULUS
    return _to_signed(delta, DATA_WIDTH)


def _sin_lut(phase: int) -> int:
    wrapped = _wrap_phase(phase)
    index = ((wrapped * LUT_SIZE) // PHASE_MODULUS) & (LUT_SIZE - 1)
    return round(math.sin(2.0 * math.pi * index / LUT_SIZE) * SCALE)


def fixed_point_step(
    phases: list[float], omega: list[float], coupling: list[float], dt: float
) -> list[int]:
    """Golden fixed-point next-phase vector mirroring the RTL bit-for-bit."""
    n = len(phases)
    acc_width = 2 * DATA_WIDTH + 8
    phase_fixed = [_quantise(theta % (2.0 * math.pi)) for theta in phases]
    omega_fixed = [_quantise(w) for w in omega]
    coupling_fixed = [_quantise(k) for k in coupling]
    dt_fixed = _quantise(dt)

    out: list[int] = []
    for row in range(n):
        phase_n = phase_fixed[row]
        acc = 0
        for col in range(n):
            diff = _wrap_delta(phase_fixed[col] - phase_n)
            acc += coupling_fixed[row * n + col] * _sin_lut(diff)
        acc = _to_signed(acc, acc_width)
        coupling_term = _to_signed(acc >> FRACTION, DATA_WIDTH)
        velocity = _to_signed(omega_fixed[row] + coupling_term, DATA_WIDTH)
        delta_mult = _to_signed(velocity * dt_fixed, 2 * DATA_WIDTH)
        phase_delta = _to_signed(delta_mult >> FRACTION, DATA_WIDTH)
        out.append(_wrap_phase(phase_n + phase_delta))
    return out


def _float_step(
    phases: list[float], omega: list[float], coupling: list[float], dt: float
) -> list[float]:
    """Ideal noiseless Kuramoto Euler step: theta += dt*(omega + sum K sin(diff))."""
    n = len(phases)
    out: list[float] = []
    for row in range(n):
        coupling_sum = sum(
            coupling[row * n + col] * math.sin(phases[col] - phases[row]) for col in range(n)
        )
        velocity = omega[row] + coupling_sum
        out.append((phases[row] + dt * velocity) % (2.0 * math.pi))
    return out


def _build_emitted_sv(
    name: str, phases: list[float], omega: list[float], coupling: list[float], dt: float
) -> str:
    """Construct the IR graph via the engine bindings and emit SystemVerilog."""
    n = len(phases)
    builder = ScGraphBuilder(name)
    phase_id = builder.constant_f64_vec(list(phases), f"vec<fixed<24,16>,{n}>")
    omega_id = builder.constant_f64_vec(list(omega), f"vec<fixed<24,16>,{n}>")
    coupling_id = builder.constant_f64_vec(list(coupling), f"vec<fixed<24,16>,{n * n}>")
    next_id = builder.kuramoto_step(phase_id, omega_id, coupling_id, dt)
    builder.output("phases_next", next_id)
    graph = builder.build()
    assert graph.verify() is None
    return graph.emit_sv()


def _run_cosim(
    name: str,
    phases: list[float],
    omega: list[float],
    coupling: list[float],
    dt: float,
    tmp_path: Path,
) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for HDL simulation tests")

    n = len(phases)
    expected = fixed_point_step(phases, omega, coupling, dt)
    emitted = _build_emitted_sv(name, phases, omega, coupling, dt)
    assert "sc_kuramoto_step" in emitted
    assert "no synthesizable RTL implementation yet" not in emitted

    checks = "\n".join(
        f"        if ($signed(phases_next[{(i + 1) * DATA_WIDTH - 1}:{i * DATA_WIDTH}])"
        f" !== 24'sd{expected[i]})\n"
        f'            $fatal(1, "osc {i}: got %0d want {expected[i]}",'
        f" $signed(phases_next[{(i + 1) * DATA_WIDTH - 1}:{i * DATA_WIDTH}]));"
        for i in range(n)
    )
    testbench = f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    wire signed [{n * DATA_WIDTH - 1}:0] phases_next;

    {name} dut (
        .clk(clk),
        .rst_n(rst_n),
        .phases_next(phases_next)
    );

    initial begin
        #1;
{checks}
        $display("PASS {name}");
        $finish(0);
    end
endmodule
"""
    top_path = tmp_path / f"{name}.v"
    sim_path = tmp_path / f"{name}.out"
    top_path.write_text(emitted + testbench)

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(top_path), str(KURAMOTO_HDL)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    sim_result = subprocess.run([vvp, str(sim_path)], capture_output=True, text=True, check=False)
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr
    assert "PASS" in sim_result.stdout, sim_result.stdout

    # The fixed-point step must track the ideal float Kuramoto step within LUT resolution.
    float_next = _float_step([theta % (2.0 * math.pi) for theta in phases], omega, coupling, dt)
    lut_resolution = 2.0 * math.pi / LUT_SIZE
    for got_fixed, want_float in zip(expected, float_next):
        got_rad = got_fixed / SCALE
        error = ((got_rad - want_float + math.pi) % (2.0 * math.pi)) - math.pi
        assert abs(error) <= dt * lut_resolution + 1.0 / SCALE, (got_rad, want_float)


def test_single_oscillator_matches_fixed_point(tmp_path: Path) -> None:
    _run_cosim("kuramoto_one", [0.7], [1.0], [0.0], 0.01, tmp_path)


def test_two_oscillator_symmetric_coupling(tmp_path: Path) -> None:
    _run_cosim(
        "kuramoto_two",
        [0.0, 1.5],
        [0.3, -0.2],
        [0.0, 0.5, 0.5, 0.0],
        0.01,
        tmp_path,
    )


def test_three_oscillator_asymmetric_coupling(tmp_path: Path) -> None:
    _run_cosim(
        "kuramoto_three",
        [0.0, 1.5, 3.0],
        [0.3, -0.2, 0.1],
        [0.0, 0.4, -0.2, 0.4, 0.0, 0.3, -0.2, 0.3, 0.0],
        0.02,
        tmp_path,
    )


def test_emitted_sv_instantiates_core_with_baked_parameters(tmp_path: Path) -> None:
    emitted = _build_emitted_sv(
        "kuramoto_params", [0.0, 1.5], [0.3, -0.2], [0.0, 0.5, 0.5, 0.0], 0.01
    )
    assert ".N_OSC(2)" in emitted
    assert ".DT_FIXED(24'sd655)" in emitted
    assert ".PHASE_MODULUS(24'sd411775)" in emitted
    assert ".HALF_PHASE_MODULUS(24'sd205887)" in emitted
