# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — exponential-relaxation co-simulation reference

"""Closed-form exponential-relaxation feature contracts."""

from __future__ import annotations

import math


def _closed_form_features(
    *,
    initial: float,
    steady: float,
    tau: float,
    dt: float,
    steps: int,
) -> dict[str, float]:
    values = [
        steady + (initial - steady) * math.exp(-(step * dt) / tau) for step in range(1, steps + 1)
    ]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }
