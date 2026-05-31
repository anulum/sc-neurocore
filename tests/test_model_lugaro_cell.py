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


def test_corrupted_runtime_state_does_not_mutate_state() -> None:
    cell = LugaroCell()
    cell.adapt = math.nan
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        cell.step(5.0)

    assert _snapshot(cell) == before
