# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared resonate-and-fire accelerator fixtures

"""Shared parameters and Python reference results for accelerator contracts."""

from typing import Any

import numpy as np

from sc_neurocore.accel import resonate_and_fire as backends

_PARAMETERS = (0.13, -0.27, -0.8, 7.5, 0.9, 0.006)


def _baseline(steps: int = 2) -> dict[str, Any]:
    drive = np.linspace(3.0, 4.0, steps, dtype=np.float64)
    return dict(backends.simulate_python(*_PARAMETERS, drive))


def _spiking_baseline() -> dict[str, Any]:
    return dict(backends.simulate_python(0.0, 0.99, 0.0, 1.0, 1.0, 0.1, [10.0]))
