# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Bit-for-bit co-simulation proof for the bit-true neuron kernel.

The claim that :func:`generate_bittrue_kernel_from_neuron` produces code
*identical to the Verilog RTL* is verified here rather than asserted: the same
neuron is compiled to Verilog (:func:`compile_to_verilog`) and to the bit-true C
kernel, both are driven with the same constant current from the same reset
state, and their per-cycle state traces are required to be equal integer-for-
integer. A companion check compiles the Rust kernel and requires it to match the
C kernel, so the Rust path inherits the same guarantee transitively.

The tests skip when the toolchain (Icarus Verilog / gcc / rustc) is absent, so
they never gate a coverage run; the pure-Python structural coverage of the
generators lives in ``tests/test_c_fixed_emitter.py`` and
``tests/test_bit_true_kernel.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    generate_bittrue_kernel_from_neuron,
)
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import from_equations

HAS_COSIM = shutil.which("iverilog") is not None and shutil.which("gcc") is not None
HAS_RUST = shutil.which("rustc") is not None


def _q_input(current: float, dw: int, frac: int) -> int:
    q = Q88(data_width=dw, fraction=frac)
    raw = q.encode(current)
    if raw >= (1 << (dw - 1)):
        raw -= 1 << dw
    return raw


def _testbench(neuron, module: str, current: float, n_steps: int, dw: int, frac: int) -> str:
    q = Q88(data_width=dw, fraction=frac)
    i_val = q.encode_signed_literal(current)
    svs = list(neuron.equations)
    ports = [
        "    .clk(clk),",
        "    .rst_n(rst_n),",
        f"    .I_t({i_val}),",
        "    .spike_out(spike_out),",
    ]
    wires = []
    for v in svs:
        ports.append(f"    .{v}_out({v}_out),")
        wires.append(f"wire signed [{dw - 1}:0] {v}_out;")
    ports[-1] = ports[-1].rstrip(",")
    fmt = " ".join("%0d" for _ in svs)
    args = ", ".join(f"$signed({v}_out)" for v in svs)
    # rst_n is deasserted at t=23 — strictly between clock edges — so the first
    # sampled posedge is a genuine step-1 (no reset/step race, no off-by-one).
    return "\n".join(
        [
            "`timescale 1ns/1ps",
            f"module tb_{module};",
            "reg clk; reg rst_n; wire spike_out;",
            *wires,
            f"{module} uut (",
            *ports,
            ");",
            "initial clk=0; always #5 clk=~clk;",
            "integer k;",
            "initial begin",
            "  rst_n=0;",
            "  #23; rst_n=1;",
            f"  for(k=0;k<{n_steps};k=k+1) begin",
            "    @(posedge clk); #1;",
            f'    $display("{fmt}", {args});',
            "  end",
            "  $finish;",
            "end",
            "endmodule",
        ]
    )


def _c_main(neuron, module: str, i_q: int, n_steps: int, dw: int) -> str:
    svs = list(neuron.equations)
    fmt = " ".join("%d" for _ in svs)
    args = ", ".join(f"(int)st.{v}_out" for v in svs)
    return (
        "#include <stdio.h>\n"
        f"int main(void) {{ {module}_state_t st; {module}_reset(&st); int{dw}_t I = {i_q};\n"
        f'  for (int k = 0; k < {n_steps}; k++) {{ {module}_step(&st, I); printf("{fmt}\\n", {args}); }}\n'
        "  return 0; }\n"
    )


def _rust_main(neuron, module: str, i_q: int, n_steps: int, dw: int) -> str:
    struct = f"{module.capitalize()}State"
    svs = list(neuron.equations)
    fields = ", ".join([f"{v}: 0, {v}_out: 0" for v in svs] + ["spike_out: 0"])
    outs = ", ".join(f"st.{v}_out" for v in svs)
    fmt = " ".join("{}" for _ in svs)
    return (
        "fn main() {\n"
        f"    let mut st = {struct} {{ {fields} }};\n"
        "    st.reset();\n"
        f"    let i: i{dw} = {i_q};\n"
        f'    for _ in 0..{n_steps} {{ st.step(i); println!("{fmt}", {outs}); }}\n'
        "}\n"
    )


def _rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.strip().splitlines():
        toks = line.split()
        if toks and all(t.lstrip("-").isdigit() for t in toks):
            rows.append(toks)
    return rows


def _verilog_trace(neuron, module, current, n_steps, dw, frac, tmp: Path) -> list[list[str]]:
    (tmp / f"{module}.v").write_text(compile_to_verilog(neuron, module, dw, frac))
    (tmp / "tb.v").write_text(_testbench(neuron, module, current, n_steps, dw, frac))
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(tmp / "sim"), str(tmp / f"{module}.v"), str(tmp / "tb.v")],
        check=True,
        capture_output=True,
        timeout=60,
    )
    out = subprocess.run(["vvp", str(tmp / "sim")], capture_output=True, text=True, timeout=60)
    return _rows(out.stdout)


def _c_trace(neuron, module, i_q, n_steps, dw, frac, tmp: Path) -> list[list[str]]:
    kernel = generate_bittrue_kernel_from_neuron(neuron, module, data_width=dw, fraction=frac)
    (tmp / f"{module}.c").write_text(kernel + "\n" + _c_main(neuron, module, i_q, n_steps, dw))
    subprocess.run(
        ["gcc", "-O2", "-o", str(tmp / "cker"), str(tmp / f"{module}.c")],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return _rows(
        subprocess.run([str(tmp / "cker")], capture_output=True, text=True, timeout=60).stdout
    )


def _rust_trace(neuron, module, i_q, n_steps, dw, frac, tmp: Path) -> list[list[str]]:
    kernel = generate_bittrue_kernel_from_neuron(
        neuron, module, data_width=dw, fraction=frac, language="rust"
    )
    src = "#![allow(warnings)]\n" + kernel + "\n" + _rust_main(neuron, module, i_q, n_steps, dw)
    (tmp / f"{module}.rs").write_text(src)
    subprocess.run(
        ["rustc", "--edition", "2021", "-O", "-o", str(tmp / "rker"), str(tmp / f"{module}.rs")],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return _rows(
        subprocess.run([str(tmp / "rker")], capture_output=True, text=True, timeout=60).stdout
    )


def _lif(dt=1.0):
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
        dt=dt,
    )


def _izhikevich():
    return from_equations(
        "dv/dt = 0.04*v**2 + 5*v + 140 - u + I",
        "du/dt = a*(b*v - u)",
        threshold="v > 30",
        reset="v = c; u = u + d",
        params=dict(a=0.02, b=0.2, c=-65, d=8),
        init=dict(v=-70, u=-14),
        dt=1.0,
    )


def _tanh_cell():
    return from_equations("dv/dt = 0.2*(tanh(v) - v) + I", init=dict(v=0.0), dt=0.5)


def _fhn():
    return from_equations(
        "dv/dt = v - v**3 - w + I",
        "dw/dt = 0.08*(v + 0.7 - 0.8*w)",
        init=dict(v=0.1, w=0.0),
        dt=0.1,
    )


def _fire():
    # A ramp integrate-and-fire that reliably crosses threshold and resets in
    # fixed point, so the threshold / reset / spike sequencing is co-simulated.
    return from_equations("dv/dt = I", threshold="v > 5", reset="v = 0", init=dict(v=0.0), dt=1.0)


_CASES = [
    ("lif_q88", _lif, 10.0, 60, 16, 8),
    ("lif_q1616", _lif, 10.0, 60, 32, 16),
    ("izh_q1616", _izhikevich, 10.0, 120, 32, 16),
    ("tanh_q88", _tanh_cell, 1.0, 50, 16, 8),
    ("tanh_q1616", _tanh_cell, 1.0, 50, 32, 16),
    ("fhn_q1616", _fhn, 0.5, 60, 32, 16),
    ("fire_q88", _fire, 1.0, 30, 16, 8),
    ("fire_q1616", _fire, 1.0, 30, 32, 16),
]


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestBitTrueCosim:
    @pytest.mark.parametrize(
        "name,factory,current,steps,dw,frac", _CASES, ids=[c[0] for c in _CASES]
    )
    def test_c_kernel_matches_rtl(self, name, factory, current, steps, dw, frac):
        neuron = factory()
        module = f"sc_{name}"
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            vtrace = _verilog_trace(neuron, module, current, steps, dw, frac, tmp)
            ctrace = _c_trace(neuron, module, _q_input(current, dw, frac), steps, dw, frac, tmp)
        assert vtrace, "Verilog produced no trace"
        assert len(vtrace) == len(ctrace) == steps
        assert vtrace == ctrace, f"{name}: bit-true kernel diverged from RTL"

    def test_fire_case_actually_resets(self):
        # Guard against a trivially-matching (monotone) trace: the fire case must
        # exercise the threshold / reset path — its output must drop at least once.
        neuron = _fire()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            vtrace = _verilog_trace(neuron, "sc_fire_rst", 1.0, 30, 16, 8, tmp)
        v_series = [int(row[0]) for row in vtrace]
        drops = [b < a for a, b in zip(v_series, v_series[1:])]
        assert any(drops), "expected at least one reset (falling edge) in the trace"


@pytest.mark.skipif(
    not (HAS_COSIM and HAS_RUST), reason="Icarus Verilog / gcc / rustc not all available"
)
class TestRustMatchesC:
    @pytest.mark.parametrize(
        "name,factory,current,steps,dw,frac", _CASES, ids=[c[0] for c in _CASES]
    )
    def test_rust_kernel_matches_c(self, name, factory, current, steps, dw, frac):
        neuron = factory()
        module = f"sc_{name}"
        i_q = _q_input(current, dw, frac)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctrace = _c_trace(neuron, module, i_q, steps, dw, frac, tmp)
            rtrace = _rust_trace(neuron, module, i_q, steps, dw, frac, tmp)
        assert ctrace == rtrace, f"{name}: Rust kernel diverged from C kernel"
