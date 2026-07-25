# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive-threshold accelerator dispatch test support

"""Shared parameters and result builders for adaptive-threshold dispatch tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.accel import adaptive_threshold_if as backends

_PARAMETERS = (-63.5, -52.5, -68.0, -67.0, -49.0, 4.5, 8.0, 42.0, 0.05)
_NORMALISE_KWARGS = {
    "v_reset": -67.0,
    "theta_rest": -49.0,
    "delta_theta": 4.5,
    "tau_theta": 42.0,
    "dt": 0.05,
}


def _baseline(steps: int = 2) -> dict[str, Any]:
    drive = np.linspace(20.0, 21.0, steps, dtype=np.float64)
    return dict(backends.simulate_python(*_PARAMETERS, drive))


def _spiking_baseline() -> dict[str, Any]:
    return dict(
        backends.simulate_python(
            -50.5,
            -51.0,
            -65.0,
            -65.0,
            -50.0,
            5.0,
            10.0,
            50.0,
            0.1,
            [0.0],
        )
    )


def _normalise(result: dict[str, Any], *, n_steps: int, initial: tuple[float, float]) -> Any:
    return backends.normalise_result(
        result,
        n_steps=n_steps,
        initial=initial,
        **_NORMALISE_KWARGS,
    )
