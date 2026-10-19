# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — rand()/randn() through the compiler: RTL, bit-true kernel and Python model

"""Whole-neuron co-simulation of compiled ``rand()`` and ``randn()``.

Each neuron is compiled to RTL (Icarus Verilog) and to the bit-true C kernel
(gcc with undefined-behaviour checks); the traces must be identical. Against
the Python model, ``rand()`` must equal ``floor(model·2**fraction)`` at every
step and ``randn()`` must be within half an LSB plus the transform error. A
reset expression with a table function checks the kernel's reset statements,
which were not emitted before 2026-10-19. Cases needing the toolchain skip
without Icarus or gcc.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler import random_units as ru
from sc_neurocore.compiler.c_expr_emitter import emit_c_expr
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_random import RandomDraws
from tests.bit_true_cosim_support import HAS_COSIM, HAS_RUST, _c_main, _rows, _rust_main, _testbench
from tests.test_folded_datapath import _dual_testbench


def _traces(
    neuron: EquationNeuron, dw: int, fr: int, steps: int
) -> tuple[list[list[str]], list[list[str]]]:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "m.v").write_text(compile_to_verilog(neuron, "m", dw, fr))
        (tmp / "tb.v").write_text(_testbench(neuron, "m", 0.0, steps, dw, fr))
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
        kernel = generate_bittrue_kernel_from_neuron(neuron, "m", data_width=dw, fraction=fr)
        (tmp / "m.c").write_text(kernel + "\n" + _c_main(neuron, "m", 0, steps, dw))
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
class TestCosim:
    """RTL = kernel, and both follow the Python model."""

    def test_rand_is_the_floor_of_the_model(self) -> None:
        """Map neuron v = rand(): every RTL code is floor(model·2**16)."""
        neuron = EquationNeuron(
            equations={"v": "rand()"},
            state={"v": 0.0},
            dt=1.0,
            method="map",
            rng_seed=12345,
            random_stream=7,
        )
        rtl, c = _traces(neuron, 32, 16, 20)
        model = []
        for _ in range(20):
            neuron.step(0.0)
            model.append(neuron.state["v"])
        assert rtl == c
        assert [int(row[1]) for row in rtl] == [int(v * 65536) for v in model]

    def test_randn_is_within_half_an_lsb_plus_the_transform_error(self) -> None:
        """Map neuron v = randn(): every RTL code is within 0.5 LSB + 2**-24 of the model."""
        neuron = EquationNeuron(
            equations={"v": "randn()"}, state={"v": 0.0}, dt=1.0, method="map", rng_seed=2026
        )
        rtl, c = _traces(neuron, 32, 16, 20)
        model = []
        for _ in range(20):
            neuron.step(0.0)
            model.append(neuron.state["v"])
        assert rtl == c
        bound = 0.5 + ru.NORMAL_TRANSFORM_ERROR * 65536
        assert max(abs(int(row[1]) / 65536 - v) * 65536 for row, v in zip(rtl, model)) <= bound

    def test_poisson_equals_the_model_exactly(self) -> None:
        """Map neuron v = poisson(3.5): every RTL code is the model's count times 2**8."""
        neuron = EquationNeuron(
            equations={"v": "poisson(3.5)"}, state={"v": 0.0}, dt=1.0, method="map", rng_seed=11
        )
        rtl, c = _traces(neuron, 16, 8, 40)
        model = []
        for _ in range(40):
            neuron.step(0.0)
            model.append(neuron.state["v"])
        assert rtl == c
        assert [int(row[1]) for row in rtl] == [int(v) * 256 for v in model]

    def test_noisy_euler_neuron_with_random_reset(self) -> None:
        """Noise in the drift and a random reset: RTL and kernel spike and move identically."""
        neuron = EquationNeuron(
            equations={"v": "-v + 2.0*randn() + 0.5*poisson(2.0)"},
            state={"v": 0.0},
            dt=0.1,
            threshold="v > 0.2",
            reset={"v": "rand() - 0.5"},
            rng_seed=99,
        )
        rtl, c = _traces(neuron, 16, 8, 80)
        assert rtl == c
        assert sum(int(row[0]) for row in rtl) > 0

    def test_reset_with_a_table_function_compiles_in_the_kernel(self) -> None:
        """A reset using exp(): the kernel emits the reset's table statements and matches RTL."""
        neuron = EquationNeuron(
            equations={"v": "-v + I + 1.0"},
            state={"v": 0.0},
            dt=0.1,
            threshold="v > 0.3",
            reset={"v": "exp(v) - 1.5"},
        )
        rtl, c = _traces(neuron, 16, 8, 40)
        assert rtl == c and sum(int(row[0]) for row in rtl) > 0


@pytest.mark.skipif(
    not (HAS_COSIM and HAS_RUST), reason="Icarus Verilog / gcc / rustc not available"
)
def test_rust_kernel_draws_what_the_c_kernel_draws() -> None:
    """The Rust kernel, with overflow checks, prints the C kernel's trace for all three draws."""
    neuron = EquationNeuron(
        equations={"v": "-v + 2.0*randn() + 0.5*poisson(2.0)"},
        state={"v": 0.0},
        dt=0.1,
        threshold="v > 0.2",
        reset={"v": "rand() - 0.5"},
        rng_seed=(1 << 64) - 3,
        random_stream=9,
    )
    main = _rust_main(neuron, "m", 0, 80, 32).replace(
        "spike_out: 0", "spike_out: 0, rng_step: 0", 1
    )
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = generate_bittrue_kernel_from_neuron(
            neuron, "m", data_width=32, fraction=16, language="rust"
        )
        (tmp / "m.rs").write_text(
            "#![allow(non_snake_case, dead_code, unused)]\n" + source + "\n" + main
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
    rtl, c = _traces(neuron, 32, 16, 80)
    assert rust == c == rtl
    assert sum(int(row[0]) for row in rust) > 0


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
def test_folded_pe_draws_what_the_module_draws() -> None:
    """The PE, given the module's step count and stream, matches it state and spike for spike."""
    neuron = EquationNeuron(
        equations={"v": "-v + I + randn() + 0.5*poisson(2.0)"},
        state={"v": 0.0},
        dt=0.1,
        threshold="v > 0.6",
        reset={"v": "rand() - 0.5"},
        rng_seed=4242,
        random_stream=3,
    )
    tb = _dual_testbench(neuron, "inst", "pe_mod", 0.0, 200)
    for old, new in (
        ("reg pe_spike;", "reg pe_spike; reg [63:0] pe_step;"),
        ("pe_mod pe (", "pe_mod pe (\n    .rng_step(pe_step),\n    .P_RNG_STREAM(32'd3),"),
        ("        pe_spike <= 1'b0;", "        pe_spike <= 1'b0; pe_step <= 64'd0;"),
        (
            "        pe_spike <= pe_spike_comb;",
            "        pe_spike <= pe_spike_comb; pe_step <= pe_step + 64'd1;",
        ),
        (
            "        if (inst_spike !== pe_spike) mismatches = mismatches + 1;",
            "        if (inst_spike !== pe_spike || v_out !== pe_v) mismatches = mismatches + 1;",
        ),
    ):
        assert tb.count(old) == 1, old
        tb = tb.replace(old, new)
    pe = compile_to_datapath(neuron, "pe_mod", 16, 8)
    assert "input wire [31:0] P_RNG_STREAM," in pe
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "inst.v").write_text(compile_to_verilog(neuron, "inst", 16, 8))
        (tmp / "pe.v").write_text(pe)
        (tmp / "tb.v").write_text(tb)
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(tmp / "s")]
            + [str(tmp / f) for f in ("inst.v", "pe.v", "tb.v")],
            check=True,
            capture_output=True,
            timeout=120,
        )
        out = subprocess.run(
            ["vvp", str(tmp / "s")], capture_output=True, text=True, timeout=120
        ).stdout
    match = re.search(r"MISMATCHES=(\d+) SPIKES=(\d+)", out)
    assert match, out
    assert match.group(1) == "0" and int(match.group(2)) > 0, out


def _rtl(neuron: EquationNeuron, source: str, clocks: int, dw: int, fr: int) -> list[list[str]]:
    """Return the rows of a compiled module ``m`` run for ``clocks`` clocks."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "m.v").write_text(source)
        (tmp / "tb.v").write_text(_testbench(neuron, "m", 0.0, clocks, dw, fr))
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(tmp / "s"), str(tmp / "m.v"), str(tmp / "tb.v")],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return _rows(
            subprocess.run(
                ["vvp", str(tmp / "s")], capture_output=True, text=True, timeout=120
            ).stdout
        )


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog not available")
class TestCounterPerLogicalStep:
    """rng_step counts logical steps in pipelined and multi-substep modules."""

    def test_pipelined_module_draws_what_the_unpipelined_module_draws(self) -> None:
        """Each valid cycle of the pipelined module equals one step of the plain module."""
        neuron = EquationNeuron(
            equations={"v": "-v + 2.0*randn() + 0.5*poisson(2.0) + rand()*v"},
            state={"v": 0.0},
            dt=0.1,
            threshold="v > 0.2",
            reset={"v": "rand() - 0.5"},
            rng_seed=77,
            random_stream=5,
        )
        steps = 60
        pipelined = compile_to_verilog(neuron, "m", 16, 8, pipeline_stages=1)
        match = re.search(r"assign latency = \d+'d(\d+);", pipelined)
        assert match, "pipelined module has no latency"
        latency = int(match.group(1))
        assert latency > 0
        plain = _rtl(neuron, compile_to_verilog(neuron, "m", 16, 8), steps, 16, 8)
        piped = _rtl(neuron, pipelined, steps * (latency + 1), 16, 8)
        assert piped[latency :: latency + 1] == plain
        assert sum(int(row[0]) for row in plain) > 0

    def test_every_substep_of_a_macro_step_draws_the_same_number(self) -> None:
        """dv/dt = rand(), dt = 1/4, four sub-steps: each macro step adds 4·(code >> 2).

        The code is ``floor(u·2**16)`` of the model's draw at the macro step's counter, so
        the RTL state equals the sum computed from the model's draws exactly; a counter
        advancing per sub-step would add four different codes.
        """
        neuron = EquationNeuron(
            equations={"v": "rand()"},
            state={"v": 0.0},
            dt=0.25,
            threshold="v > 1000.0",
            detection="crossing",
            substeps=4,
            rng_seed=8,
            random_stream=2,
        )
        macro_steps = 12
        rows = _rtl(neuron, compile_to_verilog(neuron, "m", 32, 16), macro_steps * 4, 32, 16)
        draws = RandomDraws(8, 2)
        expected = []
        total = 0
        for step in range(macro_steps):
            draws.step = step
            total += 4 * (int(draws.rand(0) * 65536) >> 2)
            expected.append(str(total))
        assert [row[1] for row in rows[3::4]] == expected


class TestRefusals:
    """Where draws are not yet lowered, the compiler says so."""

    def test_variable_poisson_mean_is_refused(self) -> None:
        """The unit's thresholds are built from a constant mean."""
        neuron = EquationNeuron(
            equations={"v": "poisson(v)"}, state={"v": 1.0}, method="map", dt=1.0
        )
        with pytest.raises(ValueError, match="constant mean"):
            compile_to_verilog(neuron, "m")

    def test_hls_refuses_draws(self) -> None:
        """The HLS C++ lowering has no counter-based generator."""
        with pytest.raises(ValueError, match="no C/C\\+\\+ \\(HLS\\) lowering"):
            emit_c_expr("rand(0)", set())

    def test_contract_does_not_encode_the_site_number(self) -> None:
        """The call-site literal is not reported as an encoded value."""
        neuron = EquationNeuron(equations={"v": "rand() + rand()"}, method="map", dt=1.0)
        contract = hardware_numeric_contract(neuron, QFormat.from_string("Q8.8"))
        assert not [q for q in contract.quantities if q.kind == "literal"]
