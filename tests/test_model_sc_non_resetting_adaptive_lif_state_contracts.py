# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mutable adaptive LIF state and reset contracts

"""Exercise the public source model after configuration edits and refused transitions."""

from __future__ import annotations

from dataclasses import replace
import math
import struct

import pytest

from sc_neurocore.neurons.models.sc_non_resetting_adaptive_lif import (
    SCNonResettingAdaptiveLIFNeuron,
)
from tests.test_sc_non_resetting_adaptive_lif_engine_binding_configuration import (
    FIELDS,
    INVALID_CONFIGURATIONS,
)


@pytest.mark.parametrize("overrides", INVALID_CONFIGURATIONS)
def test_source_constructor_refuses_invalid_configuration(overrides: dict[str, float]) -> None:
    """Finite invalid domains refuse before the model can advance."""
    with pytest.raises(ValueError):
        SCNonResettingAdaptiveLIFNeuron(**overrides)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_source_configuration_refuses(field: str, value: float) -> None:
    """Every state and configuration field rejects nonfinite construction."""
    with pytest.raises(ValueError, match=f"^runtime {field} must be finite$"):
        SCNonResettingAdaptiveLIFNeuron(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        *INVALID_CONFIGURATIONS,
        *[{field: value} for field in FIELDS[2:] for value in (math.nan, math.inf, -math.inf)],
    ],
)
def test_mutated_configuration_reset_refusal_is_atomic(overrides: dict[str, float]) -> None:
    """Reset refuses invalid configuration without overwriting either dynamic value."""
    cell = SCNonResettingAdaptiveLIFNeuron()
    cell.step(20.0)
    before = (cell.v, cell.theta)
    for field, value in overrides.items():
        setattr(cell, field, value)
    with pytest.raises(ValueError):
        cell.reset()
    assert (cell.v, cell.theta) == before
    defaults = SCNonResettingAdaptiveLIFNeuron()
    for field in overrides:
        setattr(cell, field, getattr(defaults, field))
    cell.reset()
    assert (cell.v, cell.theta) == (-65.0, -50.0)
    assert cell.step(20.0) == 0


@pytest.mark.parametrize("rest", [-1e308, -201.0, 101.0, 1e308])
@pytest.mark.parametrize("threshold", [-1e308, -50.0, 1e308])
def test_valid_reset_recovers_invalid_dynamic_state(rest: float, threshold: float) -> None:
    """All finite resting states recover invalid dynamics without a new voltage bound."""
    cell = SCNonResettingAdaptiveLIFNeuron(
        v_rest=rest, theta_rest=threshold, r_m=0.0, delta_theta=0.0
    )
    cell.v, cell.theta = math.nan, math.inf
    cell.reset()
    assert (cell.v, cell.theta) == (rest, threshold)
    cell.step(0.0)
    assert math.isfinite(cell.v) and math.isfinite(cell.theta)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_mutated_nonfinite_field_refuses_step_without_mutation(field: str, value: float) -> None:
    """Refusing an edited field preserves both dynamic bit patterns and permits recovery."""
    cell = SCNonResettingAdaptiveLIFNeuron()
    setattr(cell, field, value)
    before = struct.pack("<dd", cell.v, cell.theta)
    with pytest.raises(ValueError, match=f"^runtime {field} must be finite$"):
        cell.step(20.0)
    assert struct.pack("<dd", cell.v, cell.theta) == before
    setattr(cell, field, getattr(SCNonResettingAdaptiveLIFNeuron(), field))
    assert cell.step(20.0) == 0


@pytest.mark.parametrize("overrides", INVALID_CONFIGURATIONS)
def test_mutated_finite_domain_refuses_step_and_recovers(overrides: dict[str, float]) -> None:
    """Every invalid finite parameter edit refuses before a partial transition."""
    cell = SCNonResettingAdaptiveLIFNeuron()
    for field, value in overrides.items():
        setattr(cell, field, value)
    before = (cell.v, cell.theta)
    with pytest.raises(ValueError):
        cell.step(20.0)
    assert (cell.v, cell.theta) == before
    for field in overrides:
        setattr(cell, field, getattr(SCNonResettingAdaptiveLIFNeuron(), field))
    assert cell.step(20.0) == 0


@pytest.mark.parametrize("current", [math.nan, math.inf, -math.inf])
def test_nonfinite_current_refuses_atomically(current: float) -> None:
    """Invalid current refuses before either dynamic value changes."""
    cell = SCNonResettingAdaptiveLIFNeuron()
    control = replace(cell)
    before = (cell.v, cell.theta)
    with pytest.raises(ValueError, match="^current must be finite$"):
        cell.step(current)
    assert (cell.v, cell.theta) == before
    assert cell.step(20.0) == control.step(20.0)
    assert (cell.v, cell.theta) == (control.v, control.theta)


@pytest.mark.parametrize(
    ("overrides", "current", "message"),
    [
        ({"r_m": 1e308}, 20.0, "membrane exact relaxation update must remain finite"),
        (
            {"v": 1.5e308, "theta": 1e308, "theta_rest": 1e308, "delta_theta": 1e308},
            0.0,
            "threshold exact relaxation update must remain finite",
        ),
    ],
)
def test_finite_overflow_refuses_atomically(
    overrides: dict[str, float], current: float, message: str
) -> None:
    """Steady-state and event-increment overflow preserve state and a valid reset retry."""
    cell = SCNonResettingAdaptiveLIFNeuron(**overrides)
    before = (cell.v, cell.theta)
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == message and (cell.v, cell.theta) == before
    cell.reset()
    assert cell.step(0.0) == 0
