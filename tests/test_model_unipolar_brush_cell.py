# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Unipolar brush cell model tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.unipolar_brush_cell import UnipolarBrushCell


def _relax(previous: float, steady_state: float, dt: float, tau: float) -> float:
    return previous + (steady_state - previous) * (-math.expm1(-dt / tau))


def test_unipolar_brush_cell_uses_closed_form_persistent_and_membrane_relaxation() -> None:
    cell = UnipolarBrushCell()

    spike = cell.step(1.0)

    input_drive = cell.gain * 1.0
    expected_persistent = _relax(
        0.0,
        cell.persistent_gain * input_drive,
        cell.dt,
        cell.tau_persistent,
    )
    expected_v = _relax(
        cell.v_rest,
        cell.v_rest + input_drive + expected_persistent,
        cell.dt,
        cell.tau_m,
    )
    assert spike == 0
    assert math.isclose(cell.persistent, expected_persistent, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cell.v, expected_v, rel_tol=0.0, abs_tol=1e-12)


def test_unipolar_brush_cell_negative_drive_decays_persistent_exponentially() -> None:
    cell = UnipolarBrushCell(persistent=4.0)

    spike = cell.step(-100.0)

    expected_persistent = _relax(4.0, 0.0, cell.dt, cell.tau_persistent)
    assert spike == 0
    assert math.isclose(cell.persistent, expected_persistent, rel_tol=0.0, abs_tol=1e-12)
    assert cell.persistent > 0.0


def test_unipolar_brush_cell_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="tau_m"):
        UnipolarBrushCell(tau_m=0.0)
    with pytest.raises(ValueError, match="v_reset"):
        UnipolarBrushCell(v_reset=-40.0, v_threshold=-50.0)


def test_unipolar_brush_cell_preserves_state_on_invalid_runtime_current() -> None:
    cell = UnipolarBrushCell(v=-63.0, persistent=2.0)

    with pytest.raises(ValueError, match="current"):
        cell.step(math.nan)

    assert cell.v == -63.0
    assert cell.persistent == 2.0
