# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA kinetic refusal cases shared by the Python and engine tests

"""Runs the DNA kinetic simulators refuse, with the one message both give.

The Python simulator (:class:`sc_neurocore.bridges.dna_simulation.KineticSimulator`) and the engine's
``py_dna_simulate_kinetics`` refuse the same runs before they start, with the same words. Each case is
``(gates, duration_s, dt, message)``; a gate is ``(kind, input names, output name)`` with the kind's
upper-case name. Every case gives the input ``A`` at 200 nM.
"""

from __future__ import annotations

from collections.abc import Sequence

from sc_neurocore.bridges.dna_mapper import DNACircuitDesign, DNAGate, GateType

Gate = tuple[str, tuple[str, ...], str]
Case = tuple[tuple[Gate, ...], float, float, str]

_SAMPLES = (
    "the run would hold more than 33554432 samples over its traces; "
    "use a larger dt or a shorter duration_s"
)


def _count(index: int, kind: str, count: int, needed: str) -> str:
    """Return the refusal of a gate whose input name count does not fit its kind.

    Parameters
    ----------
    index : int
        The gate's position.
    kind : str
        The kind's upper-case name.
    count : int
        The gate's input name count.
    needed : str
        What the kind needs: ``"exactly 1"``, ``"exactly 3"`` or ``"at least 1"``.
    """
    return f"gate {index} ({kind}) has an input name count of {count}; its kind needs {needed}"


SHARED_CASES: dict[str, Case] = {
    "NAND has no rate law": (
        (("NAND", ("A", "B"), "Y"),),
        10.0,
        1.0,
        "gate 0 (NAND) has no rate law, so it cannot be simulated",
    ),
    "XOR has no rate law": (
        (("XOR", ("A", "B"), "Y"),),
        10.0,
        1.0,
        "gate 0 (XOR) has no rate law, so it cannot be simulated",
    ),
    "NOT without an input": ((("NOT", (), "Y"),), 10.0, 1.0, _count(0, "NOT", 0, "exactly 1")),
    "NOT with two inputs": (
        (("NOT", ("A", "B"), "Y"),),
        10.0,
        1.0,
        _count(0, "NOT", 2, "exactly 1"),
    ),
    "THRESHOLD without an input": (
        (("THRESHOLD", (), "Y"),),
        10.0,
        1.0,
        _count(0, "THRESHOLD", 0, "exactly 1"),
    ),
    "AMPLIFIER without an input": (
        (("AMPLIFIER", (), "Y"),),
        10.0,
        1.0,
        _count(0, "AMPLIFIER", 0, "exactly 1"),
    ),
    "BUFFER without an input": (
        (("BUFFER", (), "Y"),),
        10.0,
        1.0,
        _count(0, "BUFFER", 0, "exactly 1"),
    ),
    "MUX with two inputs": (
        (("MUX", ("A", "B"), "Y"),),
        10.0,
        1.0,
        _count(0, "MUX", 2, "exactly 3"),
    ),
    "MUX with four inputs": (
        (("MUX", ("A", "B", "C", "D"), "Y"),),
        10.0,
        1.0,
        _count(0, "MUX", 4, "exactly 3"),
    ),
    "AND without inputs": ((("AND", (), "Y"),), 10.0, 1.0, _count(0, "AND", 0, "at least 1")),
    "OR without inputs": ((("OR", (), "Y"),), 10.0, 1.0, _count(0, "OR", 0, "at least 1")),
    "the first refused gate is named": (
        (("AND", ("A",), "X"), ("NOT", (), "Y"), ("NAND", ("A", "B"), "Z")),
        10.0,
        1.0,
        _count(1, "NOT", 0, "exactly 1"),
    ),
    "two gates write one output": (
        (("AND", ("A",), "C"), ("OR", ("A",), "C")),
        10.0,
        1.0,
        "gate 1's output 'C' is already a trace of the result",
    ),
    "a gate writes time": (
        (("BUFFER", ("A",), "time"),),
        10.0,
        1.0,
        "gate 0's output 'time' is already a trace of the result",
    ),
    "dt 0": ((("BUFFER", ("A",), "Y"),), 10.0, 0.0, "dt must be finite and above 0"),
    "dt negative": ((("BUFFER", ("A",), "Y"),), 10.0, -1.0, "dt must be finite and above 0"),
    "dt NaN": ((("BUFFER", ("A",), "Y"),), 10.0, float("nan"), "dt must be finite and above 0"),
    "dt infinite": (
        (("BUFFER", ("A",), "Y"),),
        10.0,
        float("inf"),
        "dt must be finite and above 0",
    ),
    "the step is checked before the gates": (
        (("NAND", ("A", "B"), "Y"),),
        10.0,
        0.0,
        "dt must be finite and above 0",
    ),
    "duration NaN": (
        (("BUFFER", ("A",), "Y"),),
        float("nan"),
        1.0,
        "duration_s must be finite and at least dt",
    ),
    "duration infinite": (
        (("BUFFER", ("A",), "Y"),),
        float("inf"),
        1.0,
        "duration_s must be finite and at least dt",
    ),
    "duration below one step": (
        (("BUFFER", ("A",), "Y"),),
        0.5,
        1.0,
        "duration_s must be finite and at least dt",
    ),
    "duration negative": (
        (("BUFFER", ("A",), "Y"),),
        -1.0,
        1.0,
        "duration_s must be finite and at least dt",
    ),
    "a tiny dt": ((("BUFFER", ("A",), "Y"),), 10.0, 1e-300, _SAMPLES),
    "a ratio that overflows": ((("BUFFER", ("A",), "Y"),), 1e10, 5e-324, _SAMPLES),
}

_CONCENTRATION = "input '{name}' must have a finite concentration of at least 0"

#: Runs refused for an input concentration, ``(gates, inputs, message)``, with a duration of 10 and a step of 1. The
#: inputs are checked by name in sorted order, after the step and the duration and before the gates.
CONCENTRATION_CASES: dict[str, tuple[tuple[Gate, ...], dict[str, float], str]] = {
    "a negative concentration": (
        (("BUFFER", ("A",), "Y"),),
        {"A": -50.0},
        _CONCENTRATION.format(name="A"),
    ),
    "a NaN concentration": (
        (("OR", ("A",), "Y"),),
        {"A": float("nan")},
        _CONCENTRATION.format(name="A"),
    ),
    "an infinite concentration": (
        (("AND", ("A",), "Y"),),
        {"A": float("inf")},
        _CONCENTRATION.format(name="A"),
    ),
    "the first refused input by name": (
        (("BUFFER", ("A",), "Y"),),
        {"B": -1.0, "A": float("nan")},
        _CONCENTRATION.format(name="A"),
    ),
    "inputs are checked before the gates": (
        (("NAND", ("A", "B"), "Y"),),
        {"A": -1.0},
        _CONCENTRATION.format(name="A"),
    ),
}

#: A circuit with every kind that has a rate law, and its inputs, for the accepted-run tests.
ACCEPTED_GATES: tuple[Gate, ...] = (
    ("AND", ("A", "B"), "and"),
    ("OR", ("A", "B"), "or"),
    ("NOT", ("A",), "not"),
    ("THRESHOLD", ("A",), "threshold"),
    ("MUX", ("S", "A", "B"), "mux"),
    ("AMPLIFIER", ("B",), "amplifier"),
    ("BUFFER", ("A",), "buffer"),
)
ACCEPTED_INPUTS: dict[str, float] = {"A": 150.0, "B": 50.0, "S": 100.0}


def design(gates: Sequence[Gate]) -> DNACircuitDesign:
    """Return a Python circuit of ``gates``, each with threshold 0.5 and leak 1e-7, as the engine runs get.

    Parameters
    ----------
    gates : sequence of Gate
        The gates, in order.
    """
    return DNACircuitDesign(
        name="kinetics",
        gates=[
            DNAGate(
                gate_id=index,
                gate_type=GateType[kind],
                input_names=list(inputs),
                output_name=output,
                threshold=0.5,
                leak_rate=1e-7,
            )
            for index, (kind, inputs, output) in enumerate(gates)
        ],
    )


def engine_arguments(
    gates: Sequence[Gate], duration_s: float, dt: float, inputs: dict[str, float]
) -> dict[str, object]:
    """Return the engine's ``py_dna_simulate_kinetics`` arguments for the same run as :func:`design`.

    Parameters
    ----------
    gates : sequence of Gate
        The gates, in order.
    duration_s, dt : float
        The run's duration and step.
    inputs : dict[str, float]
        The input concentrations (nM).
    """
    return {
        "gate_types": [kind for kind, _, _ in gates],
        "gate_inputs": [list(names) for _, names, _ in gates],
        "gate_outputs": [output for _, _, output in gates],
        "gate_thresholds": [0.5] * len(gates),
        "gate_leaks": [1e-7] * len(gates),
        "input_names": list(inputs),
        "input_concs": list(inputs.values()),
        "duration_s": duration_s,
        "dt": dt,
    }


__all__ = [
    "ACCEPTED_GATES",
    "ACCEPTED_INPUTS",
    "CONCENTRATION_CASES",
    "SHARED_CASES",
    "Case",
    "Gate",
    "design",
    "engine_arguments",
]
