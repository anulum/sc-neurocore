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
step and ``randn()`` must be within 0.5 LSB. A reset expression with a table
function checks the kernel's reset statements, which were not emitted before
2026-10-19. Cases needing the toolchain skip without Icarus or gcc.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.c_expr_emitter import emit_c_expr
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import HAS_COSIM, _c_main, _rows, _testbench
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

    def test_randn_is_within_half_an_lsb(self) -> None:
        """Map neuron v = randn(): every RTL code is within 0.5 LSB of the model."""
        neuron = EquationNeuron(
            equations={"v": "randn()"}, state={"v": 0.0}, dt=1.0, method="map", rng_seed=2026
        )
        rtl, c = _traces(neuron, 32, 16, 20)
        model = []
        for _ in range(20):
            neuron.step(0.0)
            model.append(neuron.state["v"])
        assert rtl == c
        assert max(abs(int(row[1]) / 65536 - v) * 65536 for row, v in zip(rtl, model)) <= 0.5

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


class TestRefusals:
    """Where draws are not yet lowered, the compiler says so."""

    def test_pipelined_module_refuses_draws(self) -> None:
        """Pipelining would decouple the step counter from the step."""
        neuron = EquationNeuron(
            equations={"v": "rand() * v"}, state={"v": 1.0}, method="map", dt=1.0
        )
        with pytest.raises(NotImplementedError, match="one step per clock"):
            compile_to_verilog(neuron, "m", pipeline_stages=1)

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
