# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight noise models

"""Weight noise and device variation models for analog targets.

Simulates manufacturing variations, read noise, and retention loss
in analog compute-in-memory and memristive crossbar targets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightNoiseProfile:
    """Device-variation noise model for analog/memristive targets.

    Attributes
    ----------
    noise_model : str
        ``"gaussian"``, ``"uniform"``, or ``"lognormal"``.
    sigma : float
        Standard deviation of noise (fraction of weight range).
    cycle_drift : float
        Weight drift per program/erase cycle (fraction).
    retention_loss_per_day : float
        Daily retention loss (fraction).
    target_platform : str
        Target platform.
    """

    noise_model: str
    sigma: float
    cycle_drift: float
    retention_loss_per_day: float
    target_platform: str


def inject_weight_noise(
    weights: list[list[float | int]],
    *,
    noise_model: str = "gaussian",
    sigma: float = 0.05,
    seed: int | None = None,
) -> list[list[float]]:
    """Inject device-variation noise into a weight matrix.

    Simulates manufacturing variations and read noise in analog
    compute-in-memory (Mythic, IBM PCM) and memristive crossbar
    (Rain AI) targets. Enables robustness validation before tapeout.

    Parameters
    ----------
    weights : list[list[float | int]]
        Original weight matrix.
    noise_model : str
        ``"gaussian"``, ``"uniform"``, or ``"lognormal"``.
    sigma : float
        Noise magnitude (fraction of weight range).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[list[float]]
        Weight matrix with injected noise.
    """
    rng = random.Random(seed)

    flat = [abs(w) for row in weights for w in row]
    max_abs = max(flat) if flat else 1.0
    if max_abs == 0:
        max_abs = 1.0
    noise_scale = sigma * max_abs

    noisy = []
    for row in weights:
        noisy_row = []
        for w in row:
            if noise_model == "gaussian":
                noise = rng.gauss(0, noise_scale)
            elif noise_model == "uniform":
                noise = rng.uniform(-noise_scale, noise_scale)
            elif noise_model == "lognormal":
                sign = 1 if w >= 0 else -1
                log_noise = rng.gauss(0, sigma)
                noise = sign * abs(w) * (math.exp(log_noise) - 1.0)
            else:
                raise ValueError(f"Unsupported weight noise model: {noise_model!r}")
            noisy_row.append(round(w + noise, 8))
        noisy.append(noisy_row)

    return noisy


def create_noise_profile(
    *,
    noise_model: str = "gaussian",
    sigma: float = 0.05,
    cycle_drift: float = 0.001,
    retention_loss_per_day: float = 0.0005,
    target: str = "analog_ai",
) -> WeightNoiseProfile:
    """Create a device-variation noise profile for analog targets.

    Parameters
    ----------
    noise_model : str
        Noise distribution type.
    sigma : float
        Read noise standard deviation.
    cycle_drift : float
        Weight drift per program/erase cycle.
    retention_loss_per_day : float
        Daily state retention loss.
    target : str
        Target platform.

    Returns
    -------
    WeightNoiseProfile
        Complete noise characterisation.
    """
    return WeightNoiseProfile(
        noise_model=noise_model,
        sigma=sigma,
        cycle_drift=cycle_drift,
        retention_loss_per_day=retention_loss_per_day,
        target_platform=target,
    )
