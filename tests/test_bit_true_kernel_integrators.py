# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the bit-true C and Rust kernels of every integration method

"""The C and Rust kernels equal the registered RTL for RK4, exp_euler and Gauss–Seidel.

The RTL (Icarus Verilog), the C kernel (gcc with undefined-behaviour checks) and
the Rust kernel (rustc with overflow checks) run the same stimulus; their traces
of spike and every state word must be identical. RK4 cases cover a two-variable
model with a reset, a table function inside the stages, negative stage products,
saturation, wrap overflow, nearest product rounding (of the stage scalings too)
and Q16.16; exp_euler a gating variable through exprel; Gauss–Seidel a coupled
pair whose second variable reads the first's committed value. Euler with nearest
rounding is included: before 2026-10-19 its dt scaling rounded in the kernel but
not in the RTL. Cases needing a toolchain skip without it.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    generate_bittrue_kernel_from_neuron,
    kernel_arithmetic_contract,
)
from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import HAS_COSIM, HAS_RUST, _c_main, _rows, _rust_main, _testbench


def _fhn() -> EquationNeuron:
    """Return a FitzHugh–Nagumo neuron with a reset, integrated by RK4."""
    return EquationNeuron(
        equations={"v": "v - v*v*v/3.0 - w + I", "w": "0.08*(v + 0.7 - 0.8*w)"},
        state={"v": -1.0, "w": -0.5},
        dt=0.25,
        method="rk4",
        threshold="v > 1.5",
        reset={"v": "-1.2", "w": "w + 0.3"},
    )


def _tabled() -> EquationNeuron:
    """Return a leaky unit whose derivative reads exp() in every stage."""
    return EquationNeuron(
        equations={"v": "-v + 0.5*exp(-v) + I"},
        state={"v": -2.0},
        dt=0.5,
        method="rk4",
        threshold="v > 1.0",
        reset={"v": "-2.0"},
    )


def _traces(
    neuron: EquationNeuron,
    current: float,
    steps: int,
    dw: int,
    fr: int,
    overflow: str = "saturate",
    rounding: str = "truncate",
) -> tuple[list[list[str]], list[list[str]]]:
    """Return the RTL and C kernel rows for ``steps`` steps at a constant input."""
    i_q = round(current * (1 << fr))
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "m.v").write_text(
            compile_to_verilog(neuron, "m", dw, fr, overflow=overflow, rounding=rounding)
        )
        (tmp / "tb.v").write_text(_testbench(neuron, "m", current, steps, dw, fr))
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(tmp / "s"), str(tmp / "m.v"), str(tmp / "tb.v")],
            check=True,
            capture_output=True,
            timeout=120,
        )
        rtl = _rows(
            subprocess.run(
                ["vvp", str(tmp / "s")], capture_output=True, text=True, timeout=120
            ).stdout
        )
        kernel = generate_bittrue_kernel_from_neuron(
            neuron, "m", data_width=dw, fraction=fr, overflow=overflow, rounding=rounding
        )
        (tmp / "m.c").write_text(kernel + "\n" + _c_main(neuron, "m", i_q, steps, dw))
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
        c = _rows(
            subprocess.run([str(tmp / "k")], capture_output=True, text=True, timeout=60).stdout
        )
    return rtl, c


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestRtlEqualsC:
    """RTL = C kernel for RK4 neurons."""

    def test_fitzhugh_nagumo_with_reset(self) -> None:
        """Two coupled variables, cubic term, reset: 200 steps identical, with spikes."""
        rtl, c = _traces(_fhn(), 0.6, 200, 16, 8)
        assert rtl == c
        assert sum(int(row[0]) for row in rtl) > 0

    def test_table_function_in_every_stage(self) -> None:
        """exp() is evaluated at four stage states; the tables are shared correctly."""
        rtl, c = _traces(_tabled(), 1.5, 120, 16, 8)
        assert rtl == c
        assert sum(int(row[0]) for row in rtl) > 0

    def test_q16_16_and_negative_stage_products(self) -> None:
        """At Q16.16 a decaying state gives negative stage products; floor matches >>>."""
        neuron = EquationNeuron(equations={"v": "-0.37*v"}, state={"v": 3.0}, dt=0.3, method="rk4")
        rtl, c = _traces(neuron, 0.0, 60, 32, 16)
        assert rtl == c
        assert any(int(row[1]) > 0 for row in rtl)

    def test_nearest_rounding_applies_to_the_stage_scalings(self) -> None:
        """Nearest rounding narrows the stage and dt/6 products as every product."""
        rtl, c = _traces(_fhn(), 0.6, 150, 16, 8, rounding="nearest")
        assert rtl == c

    def test_saturation_and_wrap(self) -> None:
        """A runaway state saturates, or wraps under the wrap policy, identically."""
        neuron = EquationNeuron(
            equations={"v": "2.0*v + 30.0"}, state={"v": 1.0}, dt=0.5, method="rk4"
        )
        for overflow in ("saturate", "wrap"):
            rtl, c = _traces(neuron, 0.0, 40, 16, 8, overflow=overflow)
            assert rtl == c


@pytest.mark.skipif(not (HAS_COSIM and HAS_RUST), reason="gcc / rustc not available")
@pytest.mark.parametrize("method", ["rk4", "exp_euler", "gauss_seidel"])
def test_rust_kernel_equals_c(method: str) -> None:
    """The Rust kernel of each method, with overflow checks, prints the C kernel's trace."""
    neuron = EquationNeuron(
        equations={"v": "v - v*v*v/3.0 - w + I", "w": "0.08*(v + 0.7 - 0.8*w)"},
        state={"v": -1.0, "w": -0.5},
        dt=0.25,
        method=method,
        threshold="v > 1.5",
        reset={"v": "-1.2", "w": "w + 0.3"},
    )
    i_q = round(0.6 * 256)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = generate_bittrue_kernel_from_neuron(neuron, "m", language="rust")
        (tmp / "m.rs").write_text(
            "#![allow(non_snake_case, dead_code, unused)]\n"
            + source
            + "\n"
            + _rust_main(neuron, "m", i_q, 200, 16)
        )
        subprocess.run(
            ["rustc", "-C", "overflow-checks=on", "-o", str(tmp / "r"), str(tmp / "m.rs")],
            check=True,
            capture_output=True,
            timeout=300,
        )
        rust = _rows(
            subprocess.run([str(tmp / "r")], capture_output=True, text=True, timeout=60).stdout
        )
    _rtl, c = _traces(neuron, 0.6, 200, 16, 8)
    assert rust == c


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestOtherMethods:
    """exp_euler, Gauss–Seidel and Euler with nearest rounding: RTL = C kernel."""

    def test_exp_euler_gating_variable(self) -> None:
        """A gate relaxing through exprel(A·dt) and a driven membrane, with spikes."""
        neuron = EquationNeuron(
            equations={
                "v": "-0.2*v + 2.0*m + I",
                "m": "(1.0/(1.0 + exp(-v)) - m)/3.0",
            },
            state={"v": -1.0, "m": 0.1},
            dt=0.5,
            method="exp_euler",
            threshold="v > 1.5",
            reset={"v": "-1.0"},
        )
        rtl, c = _traces(neuron, 0.4, 150, 16, 8)
        assert rtl == c
        assert sum(int(row[0]) for row in rtl) > 0

    def test_gauss_seidel_reads_the_committed_earlier_variable(self) -> None:
        """The derivative of w reads v at its committed next value; the trace differs from Euler."""

        def coupled(method: str) -> EquationNeuron:
            return EquationNeuron(
                equations={"v": "-0.5*v + I - w", "w": "0.4*v - 0.2*w"},
                state={"v": 0.0, "w": 0.0},
                dt=0.25,
                method=method,
                threshold="v > 1.0",
                reset={"v": "0.0"},
            )

        rtl, c = _traces(coupled("gauss_seidel"), 2.0, 120, 16, 8)
        assert rtl == c
        euler_rtl, _ = _traces(coupled("euler"), 2.0, 120, 16, 8)
        assert rtl != euler_rtl

    def test_euler_nearest_rounding(self) -> None:
        """The dt scaling rounds to nearest in the RTL as it does in the kernel."""
        neuron = EquationNeuron(
            equations={"v": "-0.37*v + I", "w": "0.3*v - 0.11*w"},
            state={"v": 3.0, "w": -1.0},
            dt=0.3,
            threshold="v > 2.5",
            reset={"v": "-1.0"},
        )
        rtl, c = _traces(neuron, 0.9, 200, 16, 8, rounding="nearest")
        assert rtl == c


def _macro_rows(
    neuron: EquationNeuron, current: float, macro_steps: int, extra: str = ""
) -> tuple[list[list[str]], list[list[str]]]:
    """Return the RTL rows at macro boundaries and the C kernel rows per macro step.

    ``extra`` names an additional output (``spike_value_out``) printed after the spike.
    """
    substeps = neuron.substeps
    tb = _testbench(neuron, "m", current, macro_steps * substeps, 16, 8)
    main = _c_main(neuron, "m", round(current * 256), macro_steps, 16)
    if extra:
        tb = tb.replace(
            "    .spike_out(spike_out),", f"    .spike_out(spike_out),\n    .{extra}(ex),", 1
        )
        tb = tb.replace(
            "reg clk; reg rst_n; wire spike_out;",
            "reg clk; reg rst_n; wire spike_out; wire signed [15:0] ex;",
            1,
        )
        tb = tb.replace("$unsigned(spike_out),", "$unsigned(spike_out), $signed(ex),", 1)
        tb = tb.replace('$display("%0d ', '$display("%0d %0d ', 1)
        main = main.replace('"%d ', '"%d %d ', 1).replace("spike, ", f"spike, (int)st.{extra}, ", 1)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "m.v").write_text(compile_to_verilog(neuron, "m", 16, 8))
        (tmp / "tb.v").write_text(tb)
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(tmp / "s"), str(tmp / "m.v"), str(tmp / "tb.v")],
            check=True,
            capture_output=True,
            timeout=120,
        )
        rtl = _rows(
            subprocess.run(
                ["vvp", str(tmp / "s")], capture_output=True, text=True, timeout=120
            ).stdout
        )
        (tmp / "m.c").write_text(generate_bittrue_kernel_from_neuron(neuron, "m") + "\n" + main)
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
        c = _rows(
            subprocess.run([str(tmp / "k")], capture_output=True, text=True, timeout=60).stdout
        )
    return rtl[substeps - 1 :: substeps], c


def _oscillator() -> EquationNeuron:
    """Return a FitzHugh–Nagumo oscillator without reset, crossing detection, four sub-steps."""
    return EquationNeuron(
        equations={"v": "v - v*v*v/3.0 - w + I", "w": "0.08*(v + 0.7 - 0.8*w)"},
        state={"v": -1.0, "w": -0.5},
        dt=0.25,
        threshold="v > 1.0",
        detection="crossing",
        substeps=4,
    )


def _flip(substeps: int) -> EquationNeuron:
    """Return the map ``x -> 1 - x`` that fires when ``x`` falls below its value before the step."""
    return EquationNeuron(
        equations={"x": "1.0 - x"},
        state={"x": 0.0},
        dt=1.0,
        method="map",
        threshold="x < x_prev",
        detection="crossing",
        substeps=substeps,
    )


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestSubsteps:
    """Macro steps of sub-stepped neurons: the kernel equals the RTL at every boundary."""

    def test_oscillator_spikes_once_per_crossing(self) -> None:
        """Four sub-steps per macro step; spike and state equal at 80 boundaries."""
        rtl, c = _macro_rows(_oscillator(), 0.5, 80)
        assert rtl == c
        assert sum(int(row[0]) for row in rtl) > 0

    def test_graded_value_and_draws_inside_the_macro_step(self) -> None:
        """A graded value and a randn() drawn once per macro step stay equal."""
        neuron = EquationNeuron(
            equations={"v": "v - v*v*v/3.0 - w + I + 0.2*randn()", "w": "0.08*(v + 0.7 - 0.8*w)"},
            state={"v": -1.0, "w": -0.5},
            dt=0.25,
            threshold="v > 1.0",
            detection="crossing",
            substeps=3,
            spike_value="2.0 * v",
            rng_seed=21,
        )
        rtl, c = _macro_rows(neuron, 0.5, 60, extra="spike_value_out")
        assert rtl == c
        assert any(row[1] != "0" for row in rtl)

    @pytest.mark.parametrize("substeps", [2, 1])
    def test_prev_alias_is_the_macro_step_entry_state(self, substeps: int) -> None:
        """``x < x_prev`` compares a macro step's exit with its entry in RTL, C and Rust.

        RIGOR review 275: with two sub-steps EquationNeuron.step compares the macro step's
        exit with its entry; the RTL and kernels compared with the penultimate sub-step.
        ``x -> 1 - x`` returns to its entry value after two sub-steps, so Python never spikes;
        the RTL compared with the penultimate sub-step and spiked once. One sub-step is the
        control, where the alias is the state register itself.
        """
        neuron = _flip(substeps)
        golden = _flip(substeps)
        python = []
        for _ in range(8):
            spike = golden.step(0.0)
            python.append([str(int(spike)), str(round(golden.state["x"] * 256))])
        rtl, c = _macro_rows(neuron, 0.0, 8)
        assert rtl == c == python
        if HAS_RUST:
            with tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                (tmp / "m.rs").write_text(
                    "#![allow(non_snake_case, dead_code, unused)]\n"
                    + generate_bittrue_kernel_from_neuron(neuron, "m", language="rust")
                    + "\n"
                    + _rust_main(neuron, "m", 0, 8, 16)
                )
                subprocess.run(
                    ["rustc", "-C", "overflow-checks=on", "-o", str(tmp / "r"), str(tmp / "m.rs")],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
                rust = _rows(
                    subprocess.run(
                        [str(tmp / "r")], capture_output=True, text=True, timeout=60
                    ).stdout
                )
            assert rust == python

    def test_the_folded_pe_refuses_substeps(self) -> None:
        """The folded PE computes one integration step per visit, so it refuses sub-steps."""
        with pytest.raises(NotImplementedError, match="substeps > 1"):
            compile_to_datapath(_flip(2), "pe")
        assert "module pe" in compile_to_datapath(_flip(1), "pe")

    def test_resetting_substeps_are_refused(self) -> None:
        """The RTL lowers sub-steps only for non-resetting crossing models; so does the kernel."""
        neuron = EquationNeuron(
            equations={"v": "1.0"},
            state={"v": 0.0},
            dt=0.1,
            threshold="v > 0.5",
            reset={"v": "0.0"},
            substeps=2,
        )
        with pytest.raises(ValueError, match="substeps > 1 only"):
            generate_bittrue_kernel_from_neuron(neuron, "m")


def test_contract_describes_the_rk4_update() -> None:
    """The arithmetic contract states the stage arithmetic the kernel emits."""
    update = str(kernel_arithmetic_contract(method="rk4")["state_update"])
    assert "k_1 + 2 k_2 + 2 k_3 + k_4" in update
    assert "under the rounding mode" in update
