# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha accelerator dispatch fixtures

from __future__ import annotations

import numpy as np

from sc_neurocore.accel import alpha as backends
from sc_neurocore.neurons.models.alpha import AlphaResult

PARAMETERS = (0.15, 0.08, 0.05, 0.04, 0.03, -0.5, 1.2, 16.0, 4.0, 9.0, 0.5)


def baseline(steps: int = 2) -> dict[str, object]:
    """Return the maintained Python baseline result."""

    exc = np.linspace(2.0, 2.5, steps, dtype=np.float64)
    inh = np.linspace(0.5, 0.75, steps, dtype=np.float64)
    return dict(backends.simulate_python(*PARAMETERS, exc, inh))


def normalise(
    result: dict[str, object],
    *,
    n_steps: int,
    initial: tuple[float, float, float, float, float],
) -> AlphaResult:
    """Validate and normalize one accelerator result."""

    return backends.normalise_result(result, n_steps=n_steps, initial=initial, v_rest=-0.5)
