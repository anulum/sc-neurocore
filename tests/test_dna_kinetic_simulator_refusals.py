# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python DNA kinetic simulator refusal tests

"""The Python kinetic simulator refuses what it cannot simulate before it runs, as the engine does.

Before, a single-input gate without its input name raised ``IndexError``, ``AND`` and ``OR`` without inputs raised
from ``min`` or ``max``, a step of 0 divided by zero, a step of NaN failed in ``int``, a tiny step asked NumPy for an
enormous array, ``NAND``, ``XOR`` and ``CATALYTIC`` ran on the leak alone, and a gate that wrote ``"time"`` or another
gate's output replaced that trace. The time axis was also spread over ``duration_s`` instead of stepping by ``dt``,
so it named later times than the samples were taken at. The refusals and their words are the engine's
(:mod:`tests.dna_kinetics_refusal_cases`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.bridges.dna_mapper import DNACircuitDesign, DNAGate, GateType, KineticSimulator
from sc_neurocore.bridges.dna_simulation import MAX_KINETIC_SAMPLES, check_kinetics
from tests.dna_kinetics_refusal_cases import (
    ACCEPTED_GATES,
    ACCEPTED_INPUTS,
    CONCENTRATION_CASES,
    SHARED_CASES,
    design,
)


def _never(*_arguments: Any) -> float:
    """Stand in for the rate law, which a refused run must never reach."""
    raise AssertionError("a refused run evaluated a gate")


@pytest.mark.parametrize("case", sorted(SHARED_CASES))
def test_a_refused_run_raises_the_engines_words_before_any_gate_is_evaluated(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulation and the check alone both raise ``ValueError`` with the shared message.

    Parameters
    ----------
    case : str
        The refused run.
    """
    gates, duration_s, dt, message = SHARED_CASES[case]
    circuit = design(gates)
    monkeypatch.setattr(KineticSimulator, "_compute_k_eff", _never)
    with pytest.raises(ValueError) as simulated:
        KineticSimulator().simulate(circuit, {"A": 200.0}, duration_s=duration_s, dt=dt)
    assert str(simulated.value) == message
    with pytest.raises(ValueError) as checked:
        check_kinetics(circuit, {"A": 200.0}, duration_s, dt)
    assert str(checked.value) == message


@pytest.mark.parametrize("case", sorted(CONCENTRATION_CASES))
def test_a_concentration_that_is_not_finite_or_is_negative_is_refused_before_any_gate(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NaN, infinite or negative input is refused with the engine's words, the first by name.

    Parameters
    ----------
    case : str
        The refused run.
    """
    gates, inputs, message = CONCENTRATION_CASES[case]
    circuit = design(gates)
    monkeypatch.setattr(KineticSimulator, "_compute_k_eff", _never)
    with pytest.raises(ValueError) as simulated:
        KineticSimulator().simulate(circuit, inputs, duration_s=10.0, dt=1.0)
    assert str(simulated.value) == message
    with pytest.raises(ValueError) as checked:
        check_kinetics(circuit, inputs, 10.0, 1.0)
    assert str(checked.value) == message


def test_zero_and_the_largest_finite_concentration_are_accepted() -> None:
    """The bound is inclusive at 0 and open only at infinity."""
    circuit = design((("BUFFER", ("A",), "Y"),))
    for value in (0.0, 1.7976931348623157e308):
        assert check_kinetics(circuit, {"A": value}, 10.0, 1.0) == 10


def test_a_catalytic_gate_has_no_rate_law_and_is_refused() -> None:
    """The kind only Python has is refused like ``NAND`` and ``XOR``, not run on its leak."""
    gate = DNAGate(gate_id=0, gate_type=GateType.CATALYTIC, input_names=["A"], output_name="Y")
    with pytest.raises(
        ValueError, match=r"^gate 0 \(CATALYTIC\) has no rate law, so it cannot be simulated$"
    ):
        KineticSimulator().simulate(
            DNACircuitDesign(gates=[gate]), {"A": 200.0}, duration_s=20.0, dt=1.0
        )


def test_a_run_at_the_sample_limit_is_accepted_and_one_above_it_is_not() -> None:
    """The limit bounds every trace together, the time axis included."""
    circuit = design((("BUFFER", ("A",), "Y"),))
    steps = MAX_KINETIC_SAMPLES // 2
    assert MAX_KINETIC_SAMPLES == 1 << 25
    assert check_kinetics(circuit, {}, float(steps), 1.0) == steps
    with pytest.raises(ValueError, match="more than 33554432 samples"):
        check_kinetics(circuit, {}, float(steps + 1), 1.0)


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_an_accepted_run_steps_its_time_axis_by_dt(integrator: str) -> None:
    """Sample ``t`` is taken ``t * dt`` after the start, as the engine's time axis says.

    Parameters
    ----------
    integrator : str
        The simulator's integrator.
    """
    result = KineticSimulator(integrator=integrator).simulate(
        design(ACCEPTED_GATES), ACCEPTED_INPUTS, duration_s=10.0, dt=0.5
    )
    np.testing.assert_array_equal(result["time"], np.arange(20) * 0.5)
    assert sorted(result) == sorted(["time", *(output for _, _, output in ACCEPTED_GATES)])
    for _, _, output in ACCEPTED_GATES:
        trace = result[output]
        assert trace.shape == (20,)
        assert trace[0] == 0.0
        assert trace[-1] > 0.0
        assert np.all((trace >= 0.0) & (trace <= 200.0))
