# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — equation compiler: reproducers of the 2026-10-19 review findings

"""The reproducers of the RIGOR-FOUNDRY reviews 260, 262 and 263, each with a control.

Each test builds the public reproducer the review reported and checks the
behaviour the finding asked for: an inactive graded or reset expression does
not trap the C and Rust kernels; a state named ``spike_value`` or ``rng_step``
is refused or compiled instead of emitting duplicate members; the folded PE of a
crossing neuron carries its threshold history; ``xi`` in any expression is
refused before invalid source is returned; warming a neuron up does not change
the reset state its compiled forms start from; ``expit`` runs in the model;
the hardware contract describes the lowering and the draws actually emitted;
and the table measurement does not overflow int64. Cases needing a toolchain
skip without it.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.compiler import lut_accuracy
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    generate_bittrue_kernel,
    generate_bittrue_kernel_from_neuron,
)
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_refusals import EquationDivisionFailure, EquationRefusal
from tests.bit_true_cosim_support import HAS_COSIM, HAS_RUST, _rows


def _run_c(source: str, main: str) -> subprocess.CompletedProcess[str]:
    """Compile a C kernel with undefined-behaviour checks and run it."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "m.c").write_text(source + "\n" + main)
        subprocess.run(
            [
                "gcc",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
                "-o",
                str(tmp / "k"),
                str(tmp / "m.c"),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return subprocess.run([str(tmp / "k")], capture_output=True, text=True, timeout=60)


def _run_rust(source: str, main: str) -> subprocess.CompletedProcess[str]:
    """Compile a Rust kernel with overflow checks and run it."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "m.rs").write_text(
            "#![allow(non_snake_case, dead_code, unused)]\n" + source + "\n" + main
        )
        subprocess.run(
            ["rustc", "-C", "overflow-checks=on", "-o", str(tmp / "r"), str(tmp / "m.rs")],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return subprocess.run([str(tmp / "r")], capture_output=True, text=True, timeout=60)


_C_MAIN = (
    "#include <stdio.h>\n"
    "int main(void) { m_state_t st; m_reset(&st);\n"
    "  for (int k = 0; k < 2; k++) { int spike = m_step(&st, 0);\n"
    '    printf("%d %d %d\\n", spike, (int)st.spike_value_out, (int)st.v_out); }\n'
    "  return 0; }\n"
)
_RUST_MAIN = (
    "fn main() {\n"
    "    let mut st = MState { v: 0, v_out: 0, spike_out: 0, spike_value_out: 0 };\n"
    "    st.reset();\n"
    "    for _ in 0..2 { let spike = st.step(0);\n"
    '        println!("{} {} {}", spike, st.spike_value_out, st.v_out); }\n'
    "}\n"
)


def _reciprocal(initial: float) -> EquationNeuron:
    """Return v = v with value and reset 1/v: undefined only while v is 0, i.e. off the spike."""
    return EquationNeuron(
        equations={"v": "v"},
        state={"v": initial},
        dt=1.0,
        method="map",
        threshold="v > 0.0",
        reset={"v": "1.0 / v"},
        spike_value="1.0 / v",
    )


@pytest.mark.skipif(not HAS_COSIM, reason="gcc not available")
def test_inactive_reciprocal_does_not_trap_the_c_kernel() -> None:
    """NEURO-C35-GRADE-EAGER-01: no spike, so neither 1/v is evaluated; v = 1 is the control."""
    idle = _run_c(generate_bittrue_kernel_from_neuron(_reciprocal(0.0), "m"), _C_MAIN)
    assert idle.returncode == 0, idle.stderr
    assert _rows(idle.stdout) == [["0", "0", "0"]] * 2
    active = _run_c(generate_bittrue_kernel_from_neuron(_reciprocal(1.0), "m"), _C_MAIN)
    assert _rows(active.stdout) == [["1", "256", "256"]] * 2


@pytest.mark.skipif(not HAS_RUST, reason="rustc not available")
def test_inactive_reciprocal_does_not_trap_the_rust_kernel() -> None:
    """The Rust kernel, with overflow checks, gives the C kernel's rows."""
    source = generate_bittrue_kernel_from_neuron(_reciprocal(0.0), "m", language="rust")
    idle = _run_rust(source, _RUST_MAIN)
    assert idle.returncode == 0, idle.stderr
    assert _rows(idle.stdout) == [["0", "0", "0"]] * 2
    source = generate_bittrue_kernel_from_neuron(_reciprocal(1.0), "m", language="rust")
    assert _rows(_run_rust(source, _RUST_MAIN).stdout) == [["1", "256", "256"]] * 2


def test_a_state_named_spike_value_is_refused_with_a_graded_spike() -> None:
    """NEURO-C35-GRADE-NAME-01: the state's output would be the graded output."""
    with pytest.raises(EquationRefusal, match="collides with the graded output"):
        EquationNeuron(
            equations={"spike_value": "spike_value"},
            state={"spike_value": 1.0},
            threshold="spike_value > 0.0",
            method="map",
            dt=1.0,
            spike_value="spike_value",
        )
    # Without a graded spike the name is free.
    EquationNeuron(
        equations={"spike_value": "spike_value"},
        state={"spike_value": 1.0},
        threshold="spike_value > 0.0",
        method="map",
        dt=1.0,
    )


@pytest.mark.skipif(not (HAS_COSIM and HAS_RUST), reason="gcc / rustc not available")
def test_a_state_named_rng_step_without_draws_compiles() -> None:
    """NEURO-8AE-NAME-01: no draw, so no counter field; both kernels build and run."""
    neuron = EquationNeuron(
        equations={"rng_step": "rng_step + 1.0"}, state={"rng_step": 0.0}, method="map", dt=1.0
    )
    assert not neuron.uses_random_draws
    c_main = (
        "#include <stdio.h>\n"
        "int main(void) { m_state_t st; m_reset(&st); m_step(&st, 0);\n"
        '  printf("%d\\n", (int)st.rng_step_out); return 0; }\n'
    )
    assert _run_c(generate_bittrue_kernel_from_neuron(neuron, "m"), c_main).stdout.strip() == "256"
    rust_main = (
        "fn main() { let mut st = MState { rng_step: 0, rng_step_out: 0, spike_out: 0 };\n"
        '    st.reset(); st.step(0); println!("{}", st.rng_step_out); }\n'
    )
    source = generate_bittrue_kernel_from_neuron(neuron, "m", language="rust")
    assert _run_rust(source, rust_main).stdout.strip() == "256"


def test_a_state_named_rng_step_with_draws_is_refused_by_the_kernel() -> None:
    """With draws the kernel's counter field is rng_step; the clash is named."""
    neuron = EquationNeuron(
        equations={"rng_step": "rand()"}, state={"rng_step": 0.0}, method="map", dt=1.0
    )
    with pytest.raises(ValueError, match="collides with the step counter"):
        generate_bittrue_kernel_from_neuron(neuron, "m")


def test_the_simple_kernel_refuses_draws() -> None:
    """NEURO-8AE-SIMPLE-01: the {var: derivative} kernel has no counter, seed or stream."""
    for expression in ("rand(0)", "randn(0)", "poisson(2.0, 0)"):
        with pytest.raises(ValueError, match="no step counter"):
            generate_bittrue_kernel("m", {"v": expression})
    assert "m_step" in generate_bittrue_kernel("m", {"v": "I - v"})


def _crossing_oscillator(initial: float) -> EquationNeuron:
    """Return v alternating between 0 and 1; crossing 0.5 on every rise, graded 2·v; no reset."""
    return EquationNeuron(
        equations={"v": "1.0 - v"},
        state={"v": initial},
        dt=1.0,
        method="map",
        threshold="v > 0.5",
        detection="crossing",
        spike_value="2.0 * v",
    )


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog not available")
@pytest.mark.parametrize("initial", [0.0, 1.0])
def test_folded_pe_carries_the_crossing_history(initial: float) -> None:
    """NEURO-C35-FOLDED-EDGE-01: the PE driven with its own thr_cond_out equals the model.

    From v = 1 (already above threshold) the model waits for a re-crossing; the PE,
    given ``initial_threshold_active()`` after reset, does the same.
    """
    neuron = _crossing_oscillator(initial)
    pe = compile_to_datapath(neuron, "pe", 16, 8)
    assert "input wire thr_prev_in," in pe and "output wire thr_cond_out," in pe
    v0 = int(initial * 256)
    thr0 = 1 if neuron.initial_threshold_active() else 0
    tb = "\n".join(
        [
            "`timescale 1ns/1ps",
            "module tb;",
            "reg clk; reg rst_n; reg signed [15:0] v; reg thr;",
            "wire sp; wire tc; wire signed [15:0] sv; wire signed [15:0] vn;",
            "pe uut (.I_t(16'sd0), .thr_prev_in(thr), .v_reg(v), .spike_out(sp),",
            "    .thr_cond_out(tc), .spike_value_out(sv), .v_next_out(vn));",
            "initial clk = 0; always #5 clk = ~clk;",
            "always @(posedge clk or negedge rst_n)",
            f"    if (!rst_n) begin v <= 16'sd{v0}; thr <= 1'b{thr0}; end",
            "    else begin v <= vn; thr <= tc; end",
            "integer k;",
            "initial begin",
            "  rst_n = 0; #3; rst_n = 1;",
            "  for (k = 0; k < 6; k = k + 1) begin",
            "    #1;",
            '    $display("%0d %0d %0d", sp, sv, vn);',
            "    @(posedge clk);",
            "  end",
            "  $finish;",
            "end",
            "endmodule",
        ]
    )
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "pe.v").write_text(pe)
        (tmp / "tb.v").write_text(tb)
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(tmp / "s"), str(tmp / "pe.v"), str(tmp / "tb.v")],
            check=True,
            capture_output=True,
            timeout=120,
        )
        rows = _rows(
            subprocess.run(
                ["vvp", str(tmp / "s")], capture_output=True, text=True, timeout=120
            ).stdout
        )
    model = []
    for _ in range(6):
        spike = neuron.step(0.0)
        model.append(
            [str(spike), str(int(neuron.last_spike_value * 256)), str(int(neuron.state["v"] * 256))]
        )
    assert rows == model
    assert any(row[0] == "1" for row in model)


def test_level_pe_has_no_history_ports() -> None:
    """A level or resetting neuron keeps the PE's port list unchanged."""
    neuron = EquationNeuron(
        equations={"v": "v + 0.25"},
        state={"v": 0.0},
        dt=1.0,
        method="map",
        threshold="v > 1.0",
        reset={"v": "0.0"},
    )
    assert "thr_prev_in" not in compile_to_datapath(neuron, "pe")


@pytest.mark.parametrize(
    ("equation", "reset", "spike_value"),
    [
        ("-v", "0.0", "xi"),
        ("-v + 0.1*xi", "0.0", None),
        ("-v", "xi", None),
    ],
)
def test_xi_is_refused_by_every_target(equation: str, reset: str, spike_value: str | None) -> None:
    """NEURO-C35-GRADE-XI-01: xi anywhere, including the graded value, is refused."""
    neuron = EquationNeuron(
        equations={"v": equation},
        state={"v": 0.0},
        dt=0.1,
        threshold="v > 1.0",
        reset={"v": reset},
        spike_value=spike_value,
        noise_rng=np.random.default_rng(3),
    )
    assert neuron.uses_diffusion_noise
    forms: tuple[Callable[[], str], ...] = (
        lambda: compile_to_verilog(neuron, "m"),
        lambda: compile_to_datapath(neuron, "pe"),
        lambda: generate_bittrue_kernel_from_neuron(neuron, "m"),
        lambda: generate_bittrue_kernel_from_neuron(neuron, "m", language="rust"),
    )
    for compile_form in forms:
        with pytest.raises(ValueError, match="xi"):
            compile_form()


def test_warming_up_does_not_change_the_compiled_reset_state() -> None:
    """NEURO-8AE-EDGE-01: the initial threshold is evaluated with the draws of step 0."""
    for seed in (0, 1, 2, 3):

        def build(seed: int = seed) -> EquationNeuron:
            return EquationNeuron(
                equations={"v": "v"},
                state={"v": 0.5},
                dt=1.0,
                method="map",
                threshold="v > rand()",
                detection="crossing",
                rng_seed=seed,
            )

        fresh = build()
        warm = build()
        for _ in range(3):
            warm.step(0.0)
        assert warm.initial_threshold_active() == fresh.initial_threshold_active()
        assert compile_to_verilog(warm, "m") == compile_to_verilog(fresh, "m")
        assert generate_bittrue_kernel_from_neuron(
            warm, "m"
        ) == generate_bittrue_kernel_from_neuron(fresh, "m")


def test_expit_runs_in_the_model_like_sigmoid() -> None:
    """NEURO-2DCC-NAMESPACE-01: the alias the compiler accepts is bound in the model."""
    values = []
    for name in ("expit", "sigmoid"):
        neuron = EquationNeuron(equations={"v": f"{name}(I)"}, method="map", dt=1.0)
        neuron.step(0.5)
        values.append(neuron.state["v"])
    assert values[0] == values[1]


def test_contract_describes_the_lowering_and_the_draws() -> None:
    """NEURO-2DCC-CONTRACT-01 and NEURO-8AE-CONTRACT-01: the arithmetic matches what is emitted."""
    q88 = QFormat.from_string("Q8.8")
    sine = EquationNeuron(equations={"v": "sin(I)"}, method="map", dt=1.0)
    v2 = hardware_numeric_contract(sine, q88, function_lowering="v2").arithmetic
    v1 = hardware_numeric_contract(sine, q88).arithmetic
    assert v2 is not None and v1 is not None
    assert str(v2["transcendentals"]).startswith("v2 function units")
    assert str(v1["transcendentals"]).startswith("look-up tables")
    assert str(v1["randomness"]).startswith("none")
    drawing = EquationNeuron(
        equations={"v": "rand()"}, method="map", dt=1.0, rng_seed=123, random_stream=7
    )
    arithmetic = hardware_numeric_contract(drawing, q88).arithmetic
    assert arithmetic is not None
    assert "seed 123" in str(arithmetic["randomness"])
    assert "stream 7" in str(arithmetic["randomness"])


def test_contract_index_width_follows_the_grids() -> None:
    """At Q1.15 the [-16, 16) offset needs more than data_width + 1 bits."""
    contract = hardware_numeric_contract(
        EquationNeuron(equations={"v": "exp(I)"}, method="map", dt=1.0),
        QFormat.from_string("Q1.15"),
    ).arithmetic
    assert contract is not None
    assert "17 bits" not in str(contract["transcendentals"])


def test_table_measurement_does_not_overflow_int64() -> None:
    """NEURO-2DCC-WIDTH-01: a format whose shifted index exceeds 63 bits takes the exact path."""
    assert not lut_accuracy.fits_int64("cbrt", 58, 57)
    with pytest.raises(ValueError, match="63"):
        lut_accuracy.lut_read("cbrt", np.array([0], dtype=np.int64), 58, 57)
    codes = [-(1 << 57), -1, 0, 1, (1 << 57) - 1]
    exact = [lut_accuracy.lut_read_int("cbrt", c, 58, 57) for c in codes]
    assert len(exact) == len(codes)
    # Where the int64 path is safe it equals the exact one.
    for function in lut_accuracy.table_functions():
        assert lut_accuracy.fits_int64(function, 16, 8)
        sample = np.array([-32768, -1, 0, 1, 32767], dtype=np.int64)
        vector = lut_accuracy.lut_read(function, sample, 16, 8).tolist()
        assert vector == [lut_accuracy.lut_read_int(function, int(c), 16, 8) for c in sample]


def test_failed_stochastic_step_restores_the_graded_value() -> None:
    """NEURO-C35-GRADE-ROLLBACK-01: state, LFSR and last_spike_value all roll back."""
    neuron = EquationNeuron(
        equations={"v": "I"},
        state={"v": 2.0},
        dt=1.0,
        method="map",
        detection="poisson",
        probability_expression="1.0",
        reset={"v": "1.0 / (v - v)"},
        spike_value="v",
    )
    before = neuron.last_spike_value
    with pytest.raises(EquationDivisionFailure, match="divides by zero"):
        neuron.step(1.0)
    assert neuron.last_spike_value == before
    assert neuron.state["v"] == 2.0
