# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic IF co-simulation reference

"""Independent analytic reference for the zero-current Quadratic IF flow."""

from __future__ import annotations

import math


def _quadratic_if_zero_current_features(*, dt: float, steps: int) -> dict[str, float]:
    values = [-1.0 / (1.0 + step * dt) for step in range(1, steps + 1)]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }
