# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_resonate_and_fire.py

from __future__ import annotations

"""Verify the Izhikevich (2001) complex resonator and maintained event rule."""

import math


from typing import cast


import numpy as np


import pytest


from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron


def _exact_flow(
    x: float,
    y: float,
    current: float,
    b: float,
    omega: float,
    dt: float,
) -> tuple[float, float]:
    """Independent closed-form constant-real-input flow."""
    denominator = b * b + omega * omega
    x_ss = -b * current / denominator
    y_ss = omega * current / denominator
    decay = math.exp(b * dt)
    angle = omega * dt
    dx = x - x_ss
    dy = y - y_ss
    return (
        x_ss + decay * (dx * math.cos(angle) - dy * math.sin(angle)),
        y_ss + decay * (dx * math.sin(angle) + dy * math.cos(angle)),
    )


__all__ = ["math", "cast", "np", "pytest", "ResonateAndFireNeuron", "_exact_flow"]
