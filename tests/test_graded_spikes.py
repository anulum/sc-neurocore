# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — graded (multi-bit) spikes: spike_value through the model and the compiler

"""Graded spikes: ``EquationNeuron(spike_value=...)`` and its compiled ``spike_value_out``.

The value is evaluated where the reset rules are (the integrated state before
reset) and is 0 without a spike. The registered RTL (Icarus), the bit-true C
kernel (gcc with undefined-behaviour checks) and the Rust kernel (rustc with
overflow checks) must give identical traces; a model on a dyadic grid must
equal them exactly; the folded processing element must equal the module.
Cases needing a toolchain skip without it.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_refusals import EquationRefusal
from tests.bit_true_cosim_support import HAS_COSIM, HAS_RUST, _c_main, _rows, _rust_main, _testbench
from tests.test_folded_datapath import _dual_testbench


def _ramp() -> EquationNeuron:
    """Return a map neuron on a dyadic grid that climbs by 0.25 and spikes 2·v above 1."""
    return EquationNeuron(
        equations={"v": "v + 0.25"},
        state={"v": 0.0},
        dt=1.0,
        method="map",
        threshold="v > 1.0",
        reset={"v": "0.0"},
        spike_value="2.0 * v",
    )


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _rtl_and_c(neuron: EquationNeuron, steps: int) -> tuple[list[list[str]], list[list[str]]]:
    tb = _testbench(neuron, "m", 0.0, steps, 16, 8)
    tb = _replace_once(
        tb, "    .spike_out(spike_out),", "    .spike_out(spike_out),\n    .spike_value_out(sv),"
    )
    tb = _replace_once(
        tb,
        "reg clk; reg rst_n; wire spike_out;",
        "reg clk; reg rst_n; wire spike_out; wire signed [15:0] sv;",
    )
    tb = _replace_once(tb, "$unsigned(spike_out),", "$unsigned(spike_out), $signed(sv),")
    tb = tb.replace('$display("%0d ', '$display("%0d %0d ', 1)
    main = _c_main(neuron, "m", 0, steps, 16)
    main = _replace_once(main, '"%d ', '"%d %d ')
    main = _replace_once(main, "spike, ", "spike, (int)st.spike_value_out, ")
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
        kernel = generate_bittrue_kernel_from_neuron(neuron, "m", data_width=16, fraction=8)
        (tmp / "m.c").write_text(kernel + "\n" + main)
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
    """RTL = C kernel, and both equal the model on a dyadic grid."""

    def test_ramp_equals_the_model_exactly(self) -> None:
        """Every fifth step spikes with value 2.5 (code 640); other steps carry 0."""
        neuron = _ramp()
        rtl, c = _rtl_and_c(neuron, 20)
        model = []
        for _ in range(20):
            spike = neuron.step(0.0)
            model.append(
                [
                    str(spike),
                    str(int(neuron.last_spike_value * 256)),
                    str(int(neuron.state["v"] * 256)),
                ]
            )
        assert rtl == c == model
        assert [row[1] for row in rtl if row[0] == "1"] == ["640"] * 4

    def test_graded_relu_with_noise(self) -> None:
        """A Lava-style graded unit: |v| above 0.5 sends v and resets; RTL = C with draws and abs()."""
        neuron = EquationNeuron(
            equations={"v": "v + 0.3*randn()"},
            state={"v": 0.0},
            dt=1.0,
            method="map",
            threshold="abs(v) > 0.5",
            reset={"v": "0.0"},
            spike_value="v",
            rng_seed=31,
        )
        rtl, c = _rtl_and_c(neuron, 120)
        assert rtl == c
        spikes = [row for row in rtl if row[0] == "1"]
        assert spikes and all(abs(int(row[1])) > 128 for row in spikes)
        assert all(row[1] == "0" for row in rtl if row[0] == "0")
        assert {int(row[1]) > 0 for row in spikes} == {True, False}


@pytest.mark.skipif(not (HAS_COSIM and HAS_RUST), reason="gcc / rustc not available")
def test_rust_kernel_equals_c() -> None:
    """The Rust kernel, with overflow checks, prints the C kernel's trace."""
    neuron = _ramp()
    main = _rust_main(neuron, "m", 0, 20, 16)
    main = _replace_once(main, "spike_out: 0", "spike_out: 0, spike_value_out: 0")
    main = _replace_once(main, '"{} ', '"{} {} ')
    main = _replace_once(main, "spike, ", "spike, st.spike_value_out, ")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = generate_bittrue_kernel_from_neuron(
            neuron, "m", data_width=16, fraction=8, language="rust"
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
    _rtl, c = _rtl_and_c(_ramp(), 20)
    assert rust == c


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog not available")
def test_folded_pe_sends_the_module_value() -> None:
    """The folded PE's combinational spike_value_out equals the module's, cycle for cycle."""
    neuron = _ramp()
    tb = _dual_testbench(neuron, "inst", "pe_mod", 0.0, 60)
    tb = _replace_once(
        tb,
        "wire inst_spike;",
        "wire inst_spike; wire signed [15:0] inst_sv; wire signed [15:0] pe_sv_comb;",
    )
    tb = _replace_once(tb, "reg pe_spike;", "reg pe_spike; reg signed [15:0] pe_sv;")
    tb = _replace_once(
        tb,
        "    .spike_out(inst_spike),",
        "    .spike_out(inst_spike),\n    .spike_value_out(inst_sv),",
    )
    tb = _replace_once(
        tb,
        "    .spike_out(pe_spike_comb)",
        "    .spike_out(pe_spike_comb),\n    .spike_value_out(pe_sv_comb)",
    )
    tb = _replace_once(
        tb, "        pe_spike <= 1'b0;", "        pe_spike <= 1'b0; pe_sv <= 16'sd0;"
    )
    tb = _replace_once(
        tb,
        "        pe_spike <= pe_spike_comb;",
        "        pe_spike <= pe_spike_comb; pe_sv <= pe_sv_comb;",
    )
    tb = _replace_once(
        tb,
        "        if (inst_spike !== pe_spike) mismatches = mismatches + 1;",
        "        if (inst_spike !== pe_spike || inst_sv !== pe_sv) mismatches = mismatches + 1;",
    )
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "inst.v").write_text(compile_to_verilog(neuron, "inst", 16, 8))
        (tmp / "pe.v").write_text(compile_to_datapath(neuron, "pe_mod", 16, 8))
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


class TestModelAndRefusals:
    """The model's value, the reset of the value, and where a graded spike is refused."""

    def test_value_is_zero_without_a_spike_and_after_reset(self) -> None:
        """last_spike_value is 2.5 on the spiking step and 0 on every other step and after reset."""
        neuron = _ramp()
        values = []
        for _ in range(5):
            neuron.step(0.0)
            values.append(neuron.last_spike_value)
        assert values == [0.0, 0.0, 0.0, 0.0, 2.5]
        neuron.reset()
        assert neuron.last_spike_value == 0.0

    def test_needs_a_spike_mechanism(self) -> None:
        """Without a threshold or stochastic mechanism there is no spike to grade."""
        with pytest.raises(EquationRefusal, match="spike_value needs"):
            EquationNeuron(equations={"v": "v"}, method="map", dt=1.0, spike_value="v")

    def test_substeps_are_refused(self) -> None:
        """The macro-step datapath has no graded output yet."""
        neuron = EquationNeuron(
            equations={"v": "1.0"},
            state={"v": 0.0},
            dt=0.1,
            threshold="v > 0.5",
            detection="crossing",
            substeps=2,
            spike_value="v",
        )
        with pytest.raises(NotImplementedError, match="graded spike"):
            compile_to_verilog(neuron, "m")

    def test_contract_reports_the_value_literal(self) -> None:
        """A literal of the value expression is encoded and therefore reported."""
        contract = hardware_numeric_contract(_ramp(), QFormat.from_string("Q8.8"))
        assert 2.0 in {q.value for q in contract.quantities if q.kind == "literal"}
