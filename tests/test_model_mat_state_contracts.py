# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source MAT complete mutable state and reset contracts

"""Exercise all public MAT fields, reset transactions and recovery paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
import math
import struct

import pytest

from sc_neurocore.neurons.models.mat import MATNeuron

FIELDS = tuple(field.name for field in fields(MATNeuron))
CONFIGURATION_FIELDS = FIELDS[4:]
INVALID_CONFIGURATIONS = [
    *[("omega", value) for value in (-1e9 - 1.0, 1e9 + 1.0)],
    *[(field, value) for field in ("alpha_1", "alpha_2") for value in (-1.0, 1e9 + 1.0)],
    *[
        (field, value)
        for field in ("tau_m", "tau_1", "tau_2", "resistance", "dt")
        for value in (0.0, -1.0)
    ],
    ("refractory_period", -1.0),
]
INVALID_DYNAMICS = [
    *[("v", value) for value in (-201.0, 201.0)],
    *[(field, value) for field in ("theta1", "theta2") for value in (-1.0, 1e9 + 1.0)],
    *[("refractory_remaining", value) for value in (-1.0, 3.0)],
]


def _bits(cell: MATNeuron) -> bytes:
    """Retain every binary64 field bit, including refused nonfinite values."""
    return struct.pack("<" + "d" * len(FIELDS), *(float(getattr(cell, name)) for name in FIELDS))


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_constructor_refuses_each_nonfinite_field(field: str, value: float) -> None:
    """Complete constructor admission cannot omit any declared field."""
    with pytest.raises(ValueError, match="state and parameters must be finite"):
        MATNeuron(**{field: value})


@pytest.mark.parametrize(("field", "value"), [*INVALID_CONFIGURATIONS, *INVALID_DYNAMICS])
def test_finite_constructor_and_mutated_step_refuse_atomically(field: str, value: float) -> None:
    """Both complete admission and an edited live field retain their envelopes."""
    with pytest.raises(ValueError):
        MATNeuron(**{field: value})
    cell = MATNeuron()
    setattr(cell, field, value)
    before = _bits(cell)
    with pytest.raises(ValueError):
        cell.step(0.0)
    assert _bits(cell) == before
    setattr(cell, field, getattr(MATNeuron(), field))
    assert cell.step(0.0) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        *INVALID_CONFIGURATIONS,
        *[
            (field, value)
            for field in CONFIGURATION_FIELDS
            for value in (math.nan, math.inf, -math.inf)
        ],
    ],
)
def test_complete_reset_refusal_preserves_every_field_and_recovers(
    field: str, value: float
) -> None:
    """Invalid retained configuration cannot commit even one zero-rest field."""
    cell = MATNeuron(v=20.0, theta1=2.0, theta2=3.0, refractory_remaining=0.5)
    setattr(cell, field, value)
    before = _bits(cell)
    with pytest.raises(ValueError):
        cell.reset()
    assert _bits(cell) == before
    setattr(cell, field, getattr(MATNeuron(), field))
    cell.reset()
    assert (cell.v, cell.theta1, cell.theta2, cell.refractory_remaining) == (0.0, 0.0, 0.0, 0.0)
    assert cell.step(0.0) == 0


@pytest.mark.parametrize("field", FIELDS[:4])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1e308, 1e308])
def test_valid_reset_recovers_corrupted_dynamics_and_preserves_profile(
    field: str, value: float
) -> None:
    """Zero-rest recovery replaces bad dynamics while retaining complete configuration."""
    cell = MATNeuron(tau_m=8.0, alpha_1=7.0, resistance=40.0, refractory_period=0.0)
    configuration = tuple(getattr(cell, name) for name in CONFIGURATION_FIELDS)
    setattr(cell, field, value)
    cell.reset()
    assert (cell.v, cell.theta1, cell.theta2, cell.refractory_remaining) == (0.0, 0.0, 0.0, 0.0)
    assert tuple(getattr(cell, name) for name in CONFIGURATION_FIELDS) == configuration
    assert cell.step(0.0) == 0


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_mutated_step_retains_state_and_valid_retry(field: str, value: float) -> None:
    """A refused live-state transition preserves every bit before recovery."""
    cell = MATNeuron()
    setattr(cell, field, value)
    before = _bits(cell)
    with pytest.raises(ValueError):
        cell.step(0.7)
    assert _bits(cell) == before
    setattr(cell, field, getattr(MATNeuron(), field))
    assert cell.step(0.7) == MATNeuron().step(0.7)


@pytest.mark.parametrize("current", [math.nan, math.inf, -math.inf, 1e308])
def test_nonfinite_current_and_finite_voltage_overflow_are_atomic(current: float) -> None:
    """Current refusal or candidate overflow leaves a subsequent valid step intact."""
    cell = MATNeuron()
    before = _bits(cell)
    with pytest.raises((ValueError, FloatingPointError)):
        cell.step(current)
    assert _bits(cell) == before
    assert cell.step(0.0) == 0


@pytest.mark.parametrize("theta", ["theta1", "theta2"])
def test_post_event_threshold_overflow_is_atomic(theta: str) -> None:
    """An excessive event-history increment commits none of the four dynamics."""
    tau = "tau_1" if theta == "theta1" else "tau_2"
    increment = "alpha_1" if theta == "theta1" else "alpha_2"
    cell = MATNeuron(omega=-1e9, **{theta: 1e9, tau: 1e308})
    before = _bits(cell)
    with pytest.raises(ValueError, match="post-spike threshold left"):
        cell.step(0.0)
    assert _bits(cell) == before
    setattr(cell, increment, 0.0)
    assert cell.step(0.0) == 1


@pytest.mark.parametrize(
    ("factory", "profile"),
    [
        (MATNeuron.regular_spiking, (19.0, 37.0, 2.0)),
        (MATNeuron.intrinsically_bursting, (26.0, 1.7, 2.0)),
        (MATNeuron.fast_spiking, (11.0, 10.0, 0.002)),
    ],
)
def test_named_factory_profile_reset_and_non_resetting_event(
    factory: Callable[..., MATNeuron], profile: tuple[float, float, float]
) -> None:
    """Public paper-profile factories retain their configured threshold and recurrence."""
    cell = factory(v=40.0, refractory_period=0.0, resistance=40.0)
    assert (cell.omega, cell.alpha_1, cell.alpha_2) == profile
    assert cell.threshold == profile[0]
    assert cell.step(0.7) == 1
    assert cell.v > 39.0
    assert cell.threshold == profile[0] + profile[1] + profile[2]
    cell.reset()
    assert cell.threshold == profile[0]
    assert (cell.resistance, cell.refractory_period) == (40.0, 0.0)


@pytest.mark.parametrize(
    ("factory", "profile"),
    [
        (MATNeuron.regular_spiking, (19.0, 37.0, 2.0)),
        (MATNeuron.intrinsically_bursting, (26.0, 1.7, 2.0)),
        (MATNeuron.fast_spiking, (11.0, 10.0, 0.002)),
    ],
)
@pytest.mark.parametrize(
    ("field", "value"),
    list(zip(FIELDS, (4.0, 2.0, 3.0, 0.5, 15.0, 8.0, 12.0, 250.0, 7.0, 3.0, 40.0, 1.0, 0.05))),
)
def test_named_factory_accepts_each_override_and_preserves_other_fields(
    factory: Callable[..., MATNeuron], profile: tuple[float, float, float], field: str, value: float
) -> None:
    """Each declared field overrides a profile default through public construction."""
    expected = MATNeuron(omega=profile[0], alpha_1=profile[1], alpha_2=profile[2])
    configuration = {name: float(getattr(expected, name)) for name in FIELDS}
    configuration[field] = value
    overrides = {field: value}
    before = overrides.copy()
    cell = factory(**overrides)
    assert _bits(cell) == _bits(MATNeuron(**configuration))
    assert overrides == before
    cell.reset()
    assert tuple(getattr(cell, name) for name in CONFIGURATION_FIELDS) == tuple(
        configuration[name] for name in CONFIGURATION_FIELDS
    )


@pytest.mark.parametrize(
    "factory", [MATNeuron.regular_spiking, MATNeuron.intrinsically_bursting, MATNeuron.fast_spiking]
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        *INVALID_CONFIGURATIONS,
        *INVALID_DYNAMICS,
        *[(field, value) for field in FIELDS for value in (math.nan, math.inf, -math.inf)],
    ],
)
def test_named_factory_refuses_invalid_override_through_complete_admission(
    factory: Callable[..., MATNeuron], field: str, value: float
) -> None:
    """Profile defaults cannot bypass normal constructor domain and finite refusal."""
    with pytest.raises(ValueError):
        factory(**{field: value})


@pytest.mark.parametrize(
    "factory", [MATNeuron.regular_spiking, MATNeuron.intrinsically_bursting, MATNeuron.fast_spiking]
)
def test_named_factory_accepts_complete_configuration_and_unknown_field_refuses(
    factory: Callable[..., MATNeuron],
) -> None:
    """Complete overrides preserve caller values; unsupported fields still fail."""
    configuration = dict(
        zip(FIELDS, (4.0, 2.0, 3.0, 0.5, 15.0, 8.0, 12.0, 250.0, 7.0, 3.0, 40.0, 1.0, 0.05))
    )
    cell = factory(**configuration)
    source = MATNeuron(**configuration)
    assert _bits(cell) == _bits(source)
    for current in (0.7, 0.0, 0.5, 0.2) * 64:
        assert cell.step(current) == source.step(current)
        assert _bits(cell) == _bits(source)
    with pytest.raises(TypeError, match="unexpected keyword"):
        factory(**{"unknown_mat_field": 1.0})
