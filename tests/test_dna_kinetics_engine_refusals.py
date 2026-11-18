# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — engine DNA kinetic simulation refusal tests

"""The engine's DNA kinetic simulation raises ``ValueError`` for what it cannot simulate; it does not end the interpreter.

The engine's release build aborted on a panic until it was set to unwind, and ``py_dna_simulate_kinetics`` panicked on a gate without its input
names, so the process ended with no exception. It also mapped an unknown gate type to ``AND``, cut unequal lists to
the shortest, kept the last of two concentrations given for one name, ran ``NAND`` and ``XOR`` on the leak alone, and
asked for an endless allocation for a step of 0. Each refusal runs in a child interpreter, so a regression shows as a
dead child rather than a dead test run. The refusals the Python simulator shares carry its words, and an accepted run
with the same integrator gives its traces exactly.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from sc_neurocore.bridges.dna_mapper import KineticSimulator
from tests.dna_kinetics_refusal_cases import (
    ACCEPTED_GATES,
    ACCEPTED_INPUTS,
    CONCENTRATION_CASES,
    SHARED_CASES,
    design,
    engine_arguments,
)
from tests.engine_requirement import require_engine

ENGINE = require_engine()

_PROGRAM = (
    "import json, sys\n"
    "from sc_neurocore_engine.sc_neurocore_engine import py_dna_simulate_kinetics\n"
    "try:\n"
    "    py_dna_simulate_kinetics(**json.loads(sys.argv[1]))\n"
    "except ValueError as error:\n"
    "    print('ValueError:', error)\n"
    "else:\n"
    "    print('accepted')\n"
)

_TWO_GATES = (("AND", ("A",), "X"), ("OR", ("A",), "Y"))

_ENGINE_ONLY: dict[str, tuple[dict[str, object], str]] = {
    "an unknown gate type": (
        engine_arguments((("FOO", ("A",), "Y"),), 10.0, 1.0, {"A": 200.0}),
        "gate 0 has the unknown type 'FOO'",
    ),
    "a gate list shorter than the others": (
        {**engine_arguments(_TWO_GATES, 10.0, 1.0, {"A": 200.0}), "gate_inputs": [["A"]]},
        "gate_types, gate_inputs, gate_outputs, gate_thresholds and gate_leaks must have one entry per gate; "
        "got 2, 1, 2, 2, 2",
    ),
    "fewer concentrations than names": (
        {
            **engine_arguments(_TWO_GATES, 10.0, 1.0, {"A": 200.0, "B": 100.0}),
            "input_concs": [200.0],
        },
        "input_names and input_concs must have the same length; got 2 and 1",
    ),
    "a name given twice": (
        {
            **engine_arguments(_TWO_GATES, 10.0, 1.0, {"A": 200.0}),
            "input_names": ["A", "A"],
            "input_concs": [1.0, 200.0],
        },
        "input name 'A' is given twice",
    ),
}


def _child_line(arguments: dict[str, object]) -> str:
    """Call the engine in a child interpreter and return the last line it printed.

    Parameters
    ----------
    arguments : dict
        The keyword arguments of ``py_dna_simulate_kinetics``.
    """
    child = subprocess.run(
        [sys.executable, "-c", _PROGRAM, json.dumps(arguments)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert child.returncode == 0, child.stderr[-2000:]
    return child.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("case", sorted(SHARED_CASES))
def test_a_shared_refusal_raises_the_python_simulators_words_in_a_live_interpreter(
    case: str,
) -> None:
    """The engine refuses each run the Python simulator refuses, with the same message.

    Parameters
    ----------
    case : str
        The refused run.
    """
    gates, duration_s, dt, message = SHARED_CASES[case]
    assert _child_line(engine_arguments(gates, duration_s, dt, {"A": 200.0})) == (
        f"ValueError: {message}"
    )


@pytest.mark.parametrize("case", sorted(CONCENTRATION_CASES))
def test_a_refused_concentration_raises_the_python_simulators_words_in_a_live_interpreter(
    case: str,
) -> None:
    """The engine refuses a NaN, infinite or negative input with the Python simulator's message.

    Parameters
    ----------
    case : str
        The refused run.
    """
    gates, inputs, message = CONCENTRATION_CASES[case]
    assert _child_line(engine_arguments(gates, 10.0, 1.0, inputs)) == f"ValueError: {message}"


@pytest.mark.parametrize("case", sorted(_ENGINE_ONLY))
def test_a_malformed_call_is_refused_in_a_live_interpreter(case: str) -> None:
    """The lists the engine takes in place of a circuit must agree, and a type must be a kind.

    Parameters
    ----------
    case : str
        The malformed call.
    """
    arguments, message = _ENGINE_ONLY[case]
    assert _child_line(arguments) == f"ValueError: {message}"


def test_gate_types_are_read_without_regard_to_case() -> None:
    """A lower-case type is the same kind, as before."""
    arguments = engine_arguments((("buffer", ("A",), "Y"),), 10.0, 1.0, {"A": 200.0})
    assert _child_line(arguments) == "accepted"


@pytest.mark.parametrize("use_rk4", [False, True])
def test_an_accepted_run_gives_the_python_simulators_traces_exactly(use_rk4: bool) -> None:
    """Every kind with a rate law gives the same samples, and the same time axis, in both simulators.

    Parameters
    ----------
    use_rk4 : bool
        RK4 when true, forward Euler when false.
    """
    engine = ENGINE.py_dna_simulate_kinetics(
        **engine_arguments(ACCEPTED_GATES, 10.0, 0.5, ACCEPTED_INPUTS), use_rk4=use_rk4
    )
    python = KineticSimulator(integrator="rk4" if use_rk4 else "euler").simulate(
        design(ACCEPTED_GATES), ACCEPTED_INPUTS, duration_s=10.0, dt=0.5
    )
    assert sorted(engine) == sorted(python)
    for key, trace in python.items():
        np.testing.assert_array_equal(engine[key], trace, err_msg=key)
