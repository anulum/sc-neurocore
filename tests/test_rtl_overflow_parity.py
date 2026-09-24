# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated RTL and bit-true kernel agree when values leave the word

"""The RTL saturates, divides, looks up and compares as the bit-true kernel does.

Each neuron is compiled by the production equation compiler, simulated by
Icarus Verilog and mirrored by the generated C kernel under a schedule that
drives the input to both ends of the word and resets mid-run. Each case also
checks that the behaviour it targets actually occurs in the trace, so an
agreement cannot come from a stimulus that never reaches it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.studio.model_cosim import (
    PhaseSimulation,
    _c_main,
    StimulusPhase,
    simulate_phases,
    stress_schedule,
)

HAS_TOOLS = all(shutil.which(tool) for tool in ("iverilog", "vvp", "gcc"))
pytestmark = pytest.mark.skipif(not HAS_TOOLS, reason="Icarus Verilog and gcc are required")


def _rows(text: str) -> list[list[int]]:
    """Return the numeric trace rows; Icarus also prints its ``$finish`` notice."""
    return [
        [int(token) for token in line.split()]
        for line in text.strip().splitlines()
        if line.split() and all(token.lstrip("-").isdigit() for token in line.split())
    ]


def _run(
    neuron: EquationNeuron,
    *,
    data_width: int = 16,
    fraction: int = 8,
    overflow: str = "saturate",
    phases: tuple[StimulusPhase, ...] | None = None,
) -> list[list[int]]:
    """Simulate RTL and kernel; assert equal traces and return the RTL one."""
    schedule = phases or (StimulusPhase(1 << fraction, 8), *stress_schedule(0, data_width))
    rtl = compile_to_verilog(
        neuron, "sc_case", data_width=data_width, fraction=fraction, overflow=overflow
    )
    simulation: PhaseSimulation = simulate_phases(
        neuron,
        rtl,
        "sc_case",
        schedule,
        data_width=data_width,
        fraction=fraction,
        overflow=overflow,
    )
    rtl_rows, reference_rows = _rows(simulation.rtl_output), _rows(simulation.reference_output)
    assert len(rtl_rows) == sum(phase.steps for phase in schedule)
    assert rtl_rows == reference_rows
    return rtl_rows


def test_a_map_sum_outside_the_word_saturates_instead_of_wrapping() -> None:
    """``x + x + I`` leaves the word; the commit must clamp it, not wrap it."""
    trace = _run(EquationNeuron(equations={"x": "x + x + I"}, method="map", dt=1.0))
    states = [row[1] for row in trace]
    assert 32767 in states and -32768 in states


def test_a_reset_value_outside_the_word_saturates() -> None:
    neuron = EquationNeuron(
        equations={"v": "v + I"},
        threshold="v >= 100",
        reset={"v": "v + 100"},
        method="map",
        dt=1.0,
    )
    trace = _run(neuron)
    assert any(row[0] == 1 and row[1] == 32767 for row in trace)


def test_a_reset_value_wraps_under_the_wrap_policy() -> None:
    neuron = EquationNeuron(
        equations={"v": "v + I"},
        threshold="v >= 100",
        reset={"v": "v + 100"},
        method="map",
        dt=1.0,
    )
    trace = _run(neuron, overflow="wrap")
    assert any(row[0] == 1 and row[1] < 0 for row in trace)


def test_division_keeps_operands_outside_the_word() -> None:
    neuron = EquationNeuron(
        equations={"x": "(x + I + 100) / (x - I + k)", "y": "(-y - 1) / k + I * 0.01"},
        parameters={"k": 50.0},
        state={"x": 1.0, "y": 1.0},
        method="map",
        dt=1.0,
    )
    _run(neuron)


def test_a_table_argument_below_its_grid_clamps_to_the_first_entry() -> None:
    """``exp(x - 100)`` is below the table's [-16, 16) grid for every state."""
    trace = _run(EquationNeuron(equations={"x": "exp(x - 100) + I"}, method="map", dt=1.0))
    # With the input at zero the state settles on exp's first entry, exp(-16) -> 0.
    assert trace[-1][1] == 0


def test_a_table_whose_step_is_finer_than_the_word_multiplies_the_offset() -> None:
    """At two fraction bits the 1/8 table step needs a left shift of the offset."""
    _run(
        EquationNeuron(equations={"x": "exp(x * 0.25) - x + I"}, method="map", dt=1.0),
        fraction=2,
    )


def test_comparisons_of_sums_and_truth_values_as_numbers() -> None:
    """A comparison reads unwrapped sums; used as a number it is 1.0, not one LSB."""
    neuron = EquationNeuron(
        equations={
            "x": "w * (x + x > 1.0) - 0.25 + I * 0.01",
            "y": "(1.0 if (x > 0.5 and y < 2.0) or x <= -3.0 else -1.0) * 0.5",
        },
        parameters={"w": -2.0},
        state={"x": 1.0, "y": 0.0},
        method="map",
        dt=1.0,
    )
    trace = _run(neuron)
    assert {row[2] for row in trace} == {128, -128}


def test_floor_division_by_a_power_of_two() -> None:
    _run(EquationNeuron(equations={"x": "x // 4 + I"}, method="map", dt=1.0))


def test_crossing_detection_starts_from_the_initial_state() -> None:
    """The initial state is above threshold, so the first step does not fire."""
    neuron = EquationNeuron(
        equations={"x": "x + I"},
        threshold="x >= 0.5",
        detection="crossing",
        state={"x": 1.0},
        method="map",
        dt=1.0,
    )
    trace = _run(neuron)
    assert trace[0][0] == 0
    assert any(row[0] == 1 for row in trace)


def test_products_beyond_64_bits_wrap_like_the_rtl() -> None:
    """At Q16.16 a product of two unwrapped sums leaves 64 bits in the kernel."""
    neuron = EquationNeuron(
        equations={"x": "(x + x + x + x) * (x + x + x + x) + I"},
        state={"x": 30000.0},
        method="map",
        dt=1.0,
    )
    _run(neuron, data_width=32, fraction=16)


def test_euler_with_saturating_increments() -> None:
    neuron = EquationNeuron(
        equations={"v": "(-v + I) / tau + v * v * 0.5"},
        parameters={"tau": 4.0},
        threshold="v >= 60",
        reset={"v": "-60.0"},
        dt=0.5,
    )
    _run(neuron)


def test_a_comparison_reads_a_sum_that_leaves_the_word() -> None:
    """``x + x > y`` for x above 64 compares 128 or more, not its wrap below zero."""
    neuron = EquationNeuron(
        equations={"x": "x + I", "y": "y", "z": "1.0 if x + x > y else 0.0"},
        state={"x": 0.0, "y": 100.0, "z": 0.0},
        method="map",
        dt=1.0,
    )
    schedule = (StimulusPhase(256, 8), *stress_schedule(0, 16))
    trace = _run(neuron, phases=schedule)
    # z reads the pre-step x, except on the first step after a reset.
    starts, step = set(), 0
    for phase in schedule:
        if phase.reset_before:
            starts.add(step)
        step += phase.steps
    wide = [
        trace[k][3] for k in range(1, len(trace)) if k not in starts and trace[k - 1][1] >= 64 * 256
    ]
    assert wide and set(wide) == {256}


@pytest.mark.parametrize(
    ("equations", "parameters", "state", "data_width", "fraction"),
    [
        ({"y": "(-y - 1) / k + I * 0.01"}, {"k": 50.0}, {"y": 1.0}, 16, 8),
        (
            {"x": "(x + x + x + x) * (x + x + x + x) + I"},
            {},
            {"x": 30000.0},
            32,
            16,
        ),
        ({"x": "exp(x * 0.25) - x + I"}, {}, {"x": 0.0}, 16, 2),
    ],
    ids=["negative-numerator-divide", "product-beyond-64-bits", "table-offset-left-shift"],
)
def test_the_kernel_has_no_undefined_behaviour(
    tmp_path: Path,
    equations: dict[str, str],
    parameters: dict[str, float],
    state: dict[str, float],
    data_width: int,
    fraction: int,
) -> None:
    """Undefined C need not show in a trace at -O2, so the sanitiser checks it."""
    neuron = EquationNeuron(
        equations=equations, parameters=parameters, state=state, method="map", dt=1.0
    )
    schedule = (StimulusPhase(1 << fraction, 8), *stress_schedule(0, data_width))
    source = tmp_path / "kernel.c"
    source.write_text(
        generate_bittrue_kernel_from_neuron(
            neuron, "sc_case", data_width=data_width, fraction=fraction
        )
        + "\n"
        + _c_main(neuron, "sc_case", schedule, data_width)
    )
    binary = tmp_path / "kernel"
    subprocess.run(
        [
            "gcc",
            "-O0",
            "-std=c11",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-o",
            str(binary),
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert (run.returncode, run.stderr) == (0, "")


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc is required")
@pytest.mark.parametrize(
    ("neuron", "overflow", "data_width", "fraction"),
    [
        (
            EquationNeuron(
                equations={
                    "x": "x // 4 + w * (x + x > 1.0) + (1.0 if (x > 0.5 and x < 2.0) else -0.5) + I",
                    "y": "y + 1.0 if x else y - 1.0",
                },
                parameters={"w": -2.0},
                threshold="x >= 0.5",
                detection="crossing",
                state={"x": 1.0, "y": 0.0},
                method="map",
                dt=1.0,
            ),
            "saturate",
            16,
            8,
        ),
        (
            EquationNeuron(
                equations={"v": "v + I"},
                threshold="v >= 100",
                reset={"v": "v + 100"},
                method="map",
                dt=1.0,
            ),
            "wrap",
            16,
            8,
        ),
        (
            EquationNeuron(
                equations={"x": "(x + x + x + x) * (x + x + x + x) + I"},
                state={"x": 30000.0},
                method="map",
                dt=1.0,
            ),
            "saturate",
            32,
            16,
        ),
    ],
    ids=["crossing-select-floor", "reset-wrap", "product-beyond-64-bits"],
)
def test_the_rust_kernel_matches_the_c_kernel(
    tmp_path: Path, neuron: EquationNeuron, overflow: str, data_width: int, fraction: int
) -> None:
    from tests.bit_true_cosim_support import _c_trace, _rust_trace

    current = 1 << fraction
    c = _c_trace(neuron, "sc_case", current, 40, data_width, fraction, tmp_path, overflow=overflow)
    rust = _rust_trace(
        neuron, "sc_case", current, 40, data_width, fraction, tmp_path, overflow=overflow
    )
    assert len(c) == 40 and c == rust


def test_clamps_and_extrema_compare_unwrapped_sums() -> None:
    neuron = EquationNeuron(
        equations={
            "x": "x + I",
            "y": "clip(x + x, -100.0, 100.0) * 0.5",
            "z": "max(x + x, -x - x) * 0.25 + min(x + x, 50.0) * 0.25 + abs(x + x) * 0.25",
        },
        state={"x": 0.0, "y": 0.0, "z": 0.0},
        method="map",
        dt=1.0,
    )
    trace = _run(neuron)
    assert any(row[2] == 50 * 256 for row in trace)
