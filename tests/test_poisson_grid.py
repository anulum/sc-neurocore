# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — poisson_grid(lam): a compiled Poisson draw with a variable mean

"""``poisson_grid(lam)``: a variable mean floored to 1/16 on ``[0, 16]``, inverted exactly.

The reference rows are :func:`~sc_neurocore.compiler.random_units.poisson_thresholds`
of every grid mean; the integer program equals the reference for every code
(negative, between grid points and above 16); the model, the RTL (Icarus), the
C kernel (gcc with undefined-behaviour checks) and the Rust kernel (rustc with
overflow checks) give identical counts; the counts follow the Poisson moments of
the grid mean. Cases needing a toolchain skip without it.
"""

from __future__ import annotations

import statistics
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler import random_units as ru
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.random_unit_programs import program
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import HAS_COSIM, HAS_RUST, _c_main, _rows, _rust_main, _testbench


def test_rows_are_the_exact_thresholds_of_each_grid_mean() -> None:
    """Row i is poisson_thresholds(i/16), padded with 2**32 to the common width."""
    table = ru.poisson_grid_table()
    assert len(table) == ru.GRID_ROWS == 257
    assert len({len(row) for row in table}) == 1
    for index in (0, 1, 17, 56, 128, 256):
        exact = ru.poisson_thresholds(index / 16)
        assert table[index][: len(exact)] == exact
        assert all(value == 1 << 32 for value in table[index][len(exact) :])


@pytest.mark.parametrize(
    ("lam", "row"),
    [
        (-1.0, 0),
        (float("nan"), 0),
        (0.0, 0),
        (0.0624, 0),
        (0.0625, 1),
        (3.53, 56),
        (16.0, 256),
        (99.0, 256),
    ],
)
def test_the_model_floors_and_clamps_the_mean(lam: float, row: int) -> None:
    """The grid row is floor(16·lam), clamped to [0, 256]; NaN is row 0."""
    assert ru.poisson_grid_index(lam) == row


@pytest.mark.parametrize(("data_width", "fraction"), [(16, 8), (32, 16), (16, 3), (32, 4)])
def test_program_equals_the_reference_for_every_kind_of_code(
    data_width: int, fraction: int
) -> None:
    """Negative, between-grid, grid and above-range mean codes: program = reference."""
    unit = program("poisson_grid", 4, data_width, fraction)
    top = (1 << (data_width - 1)) - 1
    codes = [-(top + 1), -7, 0, 1, (3 << fraction) + 1, 16 << fraction, top]
    for step in range(40):
        word = ru.draw(4, step, 2, 99)[0]
        for code in codes:
            row = ru.poisson_grid_index_code(code, fraction)
            expected = fu.saturate(ru.poisson_grid_count(word, row) << fraction, data_width)
            assert unit.evaluate(code, {"step": step, "seed": 99, "stream": 2}) == expected


def test_counts_follow_the_grid_mean() -> None:
    """20 000 draws at mean 3.53 (grid 3.5): sample mean and variance within 0.1 of 3.5."""
    row = ru.poisson_grid_index(3.53)
    counts = [ru.poisson_grid_count(ru.draw(0, s, 0, 5)[0], row) for s in range(20000)]
    assert abs(statistics.mean(counts) - 3.5) < 0.1
    assert abs(statistics.variance(counts) - 3.5) < 0.1


def _neuron() -> EquationNeuron:
    """Return a map neuron whose state is a count drawn at the mean held in its input."""
    return EquationNeuron(
        equations={"v": "poisson_grid(I)"}, state={"v": 0.0}, dt=1.0, method="map", rng_seed=17
    )


def _rtl_and_c(
    neuron: EquationNeuron, current: float, steps: int
) -> tuple[list[list[str]], list[list[str]]]:
    """Return the RTL and C kernel rows at a constant input, Q8.8."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "m.v").write_text(compile_to_verilog(neuron, "m", 16, 8))
        (tmp / "tb.v").write_text(_testbench(neuron, "m", current, steps, 16, 8))
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
        kernel = generate_bittrue_kernel_from_neuron(neuron, "m")
        (tmp / "m.c").write_text(
            kernel + "\n" + _c_main(neuron, "m", round(current * 256), steps, 16)
        )
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
@pytest.mark.parametrize("current", [3.5, 3.53, 0.0, 20.0])
def test_rtl_kernel_and_model_give_the_same_counts(current: float) -> None:
    """A grid mean, one between grid points, zero and one above 16: all equal the model."""
    neuron = _neuron()
    rtl, c = _rtl_and_c(neuron, current, 40)
    model = []
    for _ in range(40):
        neuron.step(current)
        model.append(int(neuron.state["v"]) * 256)
    assert rtl == c
    assert [int(row[1]) for row in rtl] == model


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
def test_mean_from_the_state_changes_per_step() -> None:
    """The mean is the previous count plus one: a variable mean, RTL = C kernel."""
    neuron = EquationNeuron(
        equations={"v": "poisson_grid(v * 0.5 + 1.0)"}, state={"v": 0.0}, dt=1.0, method="map"
    )
    rtl, c = _rtl_and_c(neuron, 0.0, 60)
    assert rtl == c
    assert len({row[1] for row in rtl}) > 2


@pytest.mark.skipif(not (HAS_COSIM and HAS_RUST), reason="gcc / rustc not available")
def test_rust_kernel_equals_c() -> None:
    """The Rust kernel, with overflow checks, prints the C kernel's counts."""
    neuron = _neuron()
    main = _rust_main(neuron, "m", round(3.53 * 256), 40, 16).replace(
        "spike_out: 0", "spike_out: 0, rng_step: 0", 1
    )
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = generate_bittrue_kernel_from_neuron(neuron, "m", language="rust")
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
    _rtl, c = _rtl_and_c(_neuron(), 3.53, 40)
    assert rust == c


def test_constant_poisson_still_refuses_a_variable_mean_and_names_poisson_grid() -> None:
    """poisson() stays exact for its constant mean; the refusal points to poisson_grid()."""
    neuron = EquationNeuron(equations={"v": "poisson(v)"}, state={"v": 1.0}, method="map", dt=1.0)
    with pytest.raises(ValueError, match="poisson_grid"):
        compile_to_verilog(neuron, "m")


def test_contract_reports_the_literals_of_the_mean() -> None:
    """The mean expression is encoded as a word, so its literal is reported; the site is not."""
    neuron = EquationNeuron(equations={"v": "poisson_grid(2.5 * I)"}, method="map", dt=1.0)
    literals = {
        q.value
        for q in hardware_numeric_contract(neuron, QFormat.from_string("Q8.8")).quantities
        if q.kind == "literal"
    }
    assert literals == {2.5}
