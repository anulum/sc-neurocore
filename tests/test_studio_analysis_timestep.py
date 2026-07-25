# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio analysis timestep contracts

from __future__ import annotations

import math

import pytest

from sc_neurocore.studio.platform.analysis_limits import (
    STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS,
    AnalysisBudgetError,
    resolve_request_timestep,
    simulation_step_count,
)


def test_simulation_step_count_rounds_up() -> None:
    assert simulation_step_count(200.0, 0.1) == 2000
    assert simulation_step_count(10.0, 3.0) == 4  # ceil(3.33)


@pytest.mark.parametrize(
    "duration,dt",
    [
        (0.0, 0.1),
        (-1.0, 0.1),
        (100.0, 0.0),
        (100.0, -0.1),
        (math.inf, 0.1),
        (100.0, math.inf),
        (math.nan, 0.1),
        (100.0, math.nan),
    ],
)
def test_simulation_step_count_rejects_invalid_timestep(duration: float, dt: float) -> None:
    with pytest.raises(AnalysisBudgetError) as excinfo:
        simulation_step_count(duration, dt)

    assert excinfo.value.limit == "timestep"


def test_resolve_request_timestep_defaults_only_for_none() -> None:
    assert resolve_request_timestep(None) == STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS
    assert resolve_request_timestep(0.05) == 0.05
    # A supplied non-positive dt is returned unchanged so the gate can reject it.
    assert resolve_request_timestep(0.0) == 0.0
    assert resolve_request_timestep(-0.2) == -0.2
