# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Motor unit behavioural tests

"""Behavioural and fail-loud refusal tests for the alpha motor neuron and muscle unit."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import pytest
from sc_neurocore.neurons.models.motor_unit import MotorUnit


def _relax(previous: float, steady: float, tau: float, dt: float) -> float:
    return steady + (previous - steady) * math.exp(-dt / tau)


def _reference_step(unit: MotorUnit, drive: float) -> MotorUnit:
    force = unit.force * math.exp(-unit.dt / unit.tau_twitch)
    input_drive = unit.gain * max(0.0, drive) - unit.adapt
    v_target = unit.v_rest + input_drive
    v_candidate = _relax(unit.v, v_target, unit.tau_m, unit.dt)
    adapt_target = unit.a_adapt * (v_candidate - unit.v_rest)
    adapt = _relax(unit.adapt, adapt_target, unit.tau_adapt, unit.dt)
    if v_candidate >= unit.v_threshold:
        v_candidate = unit.v_reset
        force = min(1.0, force + unit.twitch_amp)
    return replace(unit, v=v_candidate, adapt=adapt, force=force)


def test_motor_unit_exact_lif_adaptation_and_force_decay_step() -> None:
    """Single step matches the analytical LIF, adaptation, and twitch-force relaxation."""
    unit = MotorUnit()
    expected = _reference_step(MotorUnit(), 20.0)

    spike = unit.step(20.0)

    assert spike == 0
    assert math.isclose(unit.v, expected.v, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(unit.adapt, expected.adapt, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(unit.force, expected.force, rel_tol=0.0, abs_tol=1e-12)

    spiking = MotorUnit(v=-50.5)
    spiking_expected = _reference_step(MotorUnit(v=-50.5), 50.0)
    assert spiking.step(50.0) == 1
    assert math.isclose(spiking.v, spiking_expected.v, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(spiking.force, spiking_expected.force, rel_tol=0.0, abs_tol=1e-12)


def test_motor_unit_invalid_drive_raises_and_preserves_state() -> None:
    """Non-finite drive raises ``ValueError`` and leaves state unchanged."""
    unit = MotorUnit()
    for _ in range(20):
        unit.step(20.0)
    before = asdict(unit)

    with pytest.raises(ValueError, match="drive"):
        unit.step(math.nan)
    assert asdict(unit) == before
    with pytest.raises(ValueError, match="drive"):
        unit.step(math.inf)
    assert asdict(unit) == before


def test_motor_unit_excess_drive_raises_and_preserves_state() -> None:
    """Candidate voltage outside ``[-150, 100]`` raises ``ValueError`` without mutating state."""
    unit = MotorUnit()
    before = asdict(unit)

    with pytest.raises(ValueError, match="voltage|range|finite"):
        unit.step(1.0e8)

    assert asdict(unit) == before


def test_motor_unit_spike_adds_twitch_and_force_stays_bounded() -> None:
    """Spikes add twitch force capped at 1 and decay when drive returns to zero."""
    unit = MotorUnit.fast()
    spikes = sum(unit.step(50.0) for _ in range(1000))

    assert spikes > 0
    assert 0.0 <= unit.force <= 1.0
    force_after_drive = unit.force

    for _ in range(200):
        unit.step(0.0)

    assert 0.0 <= unit.force <= force_after_drive
