# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta neuron co-simulation reference

"""Independent analytic reference for the constant-current Theta neuron flow."""

from __future__ import annotations

import math


def _theta_constant_current_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return continuous theta-neuron phase features for constant positive current."""
    if current <= 0.0:
        msg = "theta analytic helper requires positive current"
        raise ValueError(msg)
    root_current = math.sqrt(current)
    values = [
        2.0 * math.atan(root_current * math.tan(root_current * step * dt))
        for step in range(1, steps + 1)
    ]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.theta": values[-1],
        "min.theta": min(values),
        "max.theta": max(values),
        "mean.theta": math.fsum(values) / len(values),
    }
