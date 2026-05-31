# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LugaroCell behavioural tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.lugaro_cell import LugaroCell


def _snapshot(cell: LugaroCell) -> tuple[float, float]:
    return cell.v, cell.adapt


def _exact_relax(value: float, target: float, tau: float, dt: float) -> float:
    return target + (value - target) * math.exp(-dt / tau)


def test_default_step_preserves_bounds_and_adaptation() -> None:
    cell = LugaroCell()

    for _ in range(200):
        spike = cell.step(0.0)

    assert spike in (0, 1)
    assert -100.0 <= cell.v <= 60.0
    assert math.isfinite(cell.adapt)
    assert cell.adapt >= 0.0


def test_serotonin_reduces_current_needed_for_firing() -> None:
    no_serotonin = LugaroCell()
    with_serotonin = LugaroCell.with_serotonin(1.0)
    spikes_without = 0
    spikes_with = 0

    for _ in range(2000):
        spikes_without += no_serotonin.step(3.0)
        spikes_with += with_serotonin.step(3.0)

    assert spikes_with >= spikes_without


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau_m": 0.0},
        {"tau_adapt": 0.0},
        {"a_adapt": -1.0},
        {"gain": -1.0},
        {"serotonin": -0.1},
        {"serotonin": 1.1},
        {"dt": 0.0},
        {"v_threshold": -70.0},
        {"v": -100.1},
        {"v": 60.1},
        {"v": math.inf},
        {"adapt": math.nan},
    ],
)
def test_invalid_configuration_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LugaroCell(**kwargs)


def test_nonfinite_current_does_not_mutate_state() -> None:
    cell = LugaroCell()
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        cell.step(math.nan)

    assert _snapshot(cell) == before


def test_closed_form_membrane_and_adaptation_relaxation() -> None:
    cell = LugaroCell(v=-56.0, adapt=0.2, gain=0.0)

    v_inf = cell.v_rest - cell.adapt
    expected_v = _exact_relax(cell.v, v_inf, cell.tau_m, cell.dt)
    adapt_inf = max(0.0, cell.a_adapt * max(0.0, expected_v - cell.v_rest))
    expected_adapt = max(0.0, _exact_relax(cell.adapt, adapt_inf, cell.tau_adapt, cell.dt))

    assert cell.step(0.0) == 0
    assert cell.v == pytest.approx(expected_v)
    assert cell.adapt == pytest.approx(expected_adapt)


def test_corrupted_runtime_state_does_not_mutate_state() -> None:
    cell = LugaroCell()
    cell.adapt = math.nan
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        cell.step(5.0)

    assert _snapshot(cell) == before
