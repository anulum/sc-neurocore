# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC resetting-MAT mutable state and reset contracts

"""Exercise complete public resetting-MAT admission, reset and recovery."""

from __future__ import annotations

from dataclasses import fields
import math
import struct

import pytest

from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron

FIELDS = tuple(field.name for field in fields(SCResettingMATNeuron))
CONFIGURATION_FIELDS = FIELDS[3:]
INVALID_CONFIGURATIONS = (
    ("v_reset", -201.0),
    ("v_reset", 101.0),
    ("tau_m", 0.0),
    ("tau_m", -1.0),
    ("tau_1", 0.0),
    ("tau_1", -1.0),
    ("tau_2", 0.0),
    ("tau_2", -1.0),
    ("h1", -1.0),
    ("h1", 1.0e9 + 1.0),
    ("h2", -1.0),
    ("h2", 1.0e9 + 1.0),
    ("resistance", 0.0),
    ("resistance", -1.0),
    ("dt", 0.0),
    ("dt", -1.0),
)


def _bits(cell: SCResettingMATNeuron) -> bytes:
    """Capture every public state and configuration field without normalizing NaNs."""
    return struct.pack("<" + "d" * len(FIELDS), *(float(getattr(cell, name)) for name in FIELDS))


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_constructor_refuses_nonfinite_complete_state(field: str, value: float) -> None:
    """Every declared field must be finite before a neuron is admitted."""
    with pytest.raises(ValueError, match="state and parameters must be finite"):
        SCResettingMATNeuron(**{field: value})


@pytest.mark.parametrize(("field", "value"), INVALID_CONFIGURATIONS)
def test_constructor_refuses_invalid_finite_configuration(field: str, value: float) -> None:
    """Finite parameter values retain the established physical domain."""
    with pytest.raises(ValueError):
        SCResettingMATNeuron(**{field: value})


@pytest.mark.parametrize("rest", [-1.0e308, -500.0, -201.0, 101.0, 500.0, 1.0e308])
def test_admitted_resting_voltage_reset_refusal_preserves_state(rest: float) -> None:
    """An admitted configuration can refuse its resting candidate without mutation."""
    cell = SCResettingMATNeuron(v=-65.0, theta1=2.0, theta2=3.0, v_rest=rest)
    before = _bits(cell)
    with pytest.raises(ValueError, match="voltage is outside the safety envelope"):
        cell.reset()
    assert _bits(cell) == before
    cell.v_rest = -70.0
    cell.reset()
    assert (cell.v, cell.theta1, cell.theta2) == (-70.0, 0.0, 0.0)
    assert cell.step(0.0) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        *INVALID_CONFIGURATIONS,
        *[(name, bad) for name in CONFIGURATION_FIELDS for bad in (math.nan, math.inf, -math.inf)],
    ],
)
def test_mutated_configuration_reset_is_atomic_and_recovers(field: str, value: float) -> None:
    """Reset refuses every invalid configuration while preserving all field bits."""
    cell = SCResettingMATNeuron(v=-65.0, theta1=2.0, theta2=3.0)
    setattr(cell, field, value)
    before = _bits(cell)
    with pytest.raises(ValueError):
        cell.reset()
    assert _bits(cell) == before
    setattr(cell, field, getattr(SCResettingMATNeuron(), field))
    cell.reset()
    assert (cell.v, cell.theta1, cell.theta2) == (-70.0, 0.0, 0.0)
    assert cell.step(0.0) == 0


@pytest.mark.parametrize("field", FIELDS[:3])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.0e308, 1.0e308])
def test_valid_reset_recovers_corrupted_dynamic_state(field: str, value: float) -> None:
    """A valid complete resting candidate can repair invalid dynamic fields."""
    cell = SCResettingMATNeuron(v_rest=-65.0, tau_m=12.0, h1=4.0)
    setattr(cell, field, value)
    cell.reset()
    assert (cell.v, cell.theta1, cell.theta2) == (-65.0, 0.0, 0.0)
    assert (cell.tau_m, cell.h1) == (12.0, 4.0)
    assert cell.step(0.0) == 0


@pytest.mark.parametrize("rest", [-200.0, -70.0, -65.0, 100.0])
def test_reset_envelope_boundaries_preserve_configuration(rest: float) -> None:
    """Both voltage bounds reset successfully and retain the numerical profile."""
    cell = SCResettingMATNeuron(v_rest=rest, tau_1=20.0, tau_2=250.0, dt=0.25)
    before = tuple(getattr(cell, name) for name in CONFIGURATION_FIELDS)
    cell.reset()
    assert (cell.v, cell.theta1, cell.theta2) == (rest, 0.0, 0.0)
    assert tuple(getattr(cell, name) for name in CONFIGURATION_FIELDS) == before


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_mutated_nonfinite_step_preserves_complete_state(field: str, value: float) -> None:
    """An invalid complete state refuses before any dynamic or parameter write."""
    cell = SCResettingMATNeuron()
    setattr(cell, field, value)
    before = _bits(cell)
    with pytest.raises(ValueError):
        cell.step(50.0)
    assert _bits(cell) == before
    setattr(cell, field, getattr(SCResettingMATNeuron(), field))
    assert cell.step(0.0) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", -201.0),
        ("v", 101.0),
        ("theta1", -1.0),
        ("theta1", 1.0e9 + 1.0),
        ("theta2", -1.0),
        ("theta2", 1.0e9 + 1.0),
    ],
)
def test_finite_dynamic_envelope_refusal_preserves_state_and_recovers(
    field: str, value: float
) -> None:
    """Constructor and edited state retain the finite dynamic safety envelope."""
    with pytest.raises(ValueError):
        SCResettingMATNeuron(**{field: value})
    cell = SCResettingMATNeuron()
    setattr(cell, field, value)
    before = _bits(cell)
    with pytest.raises(ValueError):
        cell.step(0.0)
    assert _bits(cell) == before
    cell.reset()
    assert cell.step(0.0) == 0


@pytest.mark.parametrize("current", [math.nan, math.inf, -math.inf, 1.0e308])
def test_invalid_current_or_finite_candidate_overflow_is_atomic(current: float) -> None:
    """Invalid currents and finite RK4 overflow preserve the next valid transition."""
    cell = SCResettingMATNeuron()
    before = _bits(cell)
    with pytest.raises(ValueError):
        cell.step(current)
    assert _bits(cell) == before
    assert cell.step(0.0) == SCResettingMATNeuron().step(0.0)


@pytest.mark.parametrize(("theta", "increment"), [("theta1", "h1"), ("theta2", "h2")])
def test_post_event_threshold_overflow_preserves_complete_state(theta: str, increment: str) -> None:
    """An excessive post-event threshold commits no voltage or adaptation value."""
    cell = SCResettingMATNeuron(v_threshold_base=-1.0e9, **{theta: 1.0, increment: 1.0e9})
    before = _bits(cell)
    with pytest.raises(ValueError, match="post-spike adaptation left the safety envelope"):
        cell.step(0.0)
    assert _bits(cell) == before
    setattr(cell, increment, 0.0)
    assert cell.step(0.0) == 1
