# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — StellateCell behavioural tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.stellate_cell import StellateCell


def _snapshot(cell: StellateCell) -> tuple[float, float, float, float]:
    return cell.v, cell.h, cell.n, cell.p


def test_default_step_preserves_wb_kv3_bounds() -> None:
    cell = StellateCell()

    for _ in range(200):
        spike = cell.step(0.0)

    assert spike in (0, 1)
    assert -100.0 <= cell.v <= 60.0
    assert all(0.0 <= gate <= 1.0 for gate in (cell.h, cell.n, cell.p))


def test_kv3_gate_activates_more_during_depolarisation_than_rest() -> None:
    resting = StellateCell()
    depolarised = StellateCell()

    for _ in range(100):
        resting.step(0.0)
        depolarised.step(8.0)

    assert depolarised.p > resting.p


@pytest.mark.parametrize(
    "kwargs",
    [
        {"h": -0.1},
        {"n": 1.1},
        {"p": -0.1},
        {"g_na": -1.0},
        {"g_k": -1.0},
        {"g_kv3": -1.0},
        {"g_l": -1.0},
        {"c_m": 0.0},
        {"phi": 0.0},
        {"dt": 0.0},
        {"_sub_steps": 0},
        {"gain": -1.0},
        {"v": math.inf},
    ],
)
def test_invalid_configuration_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StellateCell(**kwargs)


def test_nonfinite_current_does_not_mutate_state() -> None:
    cell = StellateCell()
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        cell.step(math.nan)

    assert _snapshot(cell) == before


def test_corrupted_runtime_state_does_not_mutate_state() -> None:
    cell = StellateCell()
    cell.h = -0.1
    before = _snapshot(cell)

    with pytest.raises(ValueError):
        cell.step(8.0)

    assert _snapshot(cell) == before
