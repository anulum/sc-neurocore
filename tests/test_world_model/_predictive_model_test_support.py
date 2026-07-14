# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive-model test data and simulation support

"""Deterministic LGSSM fixtures shared by responsibility-specific tests."""

from __future__ import annotations

import numpy as np

from sc_neurocore.world_model._lgssm_types import FloatArray
from sc_neurocore.world_model.predictive_model import LinearGaussianSSM


def scalar_random_walk() -> LinearGaussianSSM:
    """Return a scalar random walk with a noisy direct observation."""
    return LinearGaussianSSM(
        A=np.array([[1.0]]),
        B=np.zeros((1, 0)),
        C=np.array([[1.0]]),
        D=np.zeros((1, 0)),
        Q=np.array([[0.1]]),
        R=np.array([[1.0]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )


def controlled_scalar_model() -> LinearGaussianSSM:
    """Return a stable scalar model with non-zero ``B`` and ``D`` terms."""
    return LinearGaussianSSM(
        A=np.array([[0.72]]),
        B=np.array([[1.4]]),
        C=np.array([[1.1]]),
        D=np.array([[0.8]]),
        Q=np.array([[0.04]]),
        R=np.array([[0.06]]),
        mu_0=np.array([0.2]),
        Sigma_0=np.array([[0.3]]),
    )


def simulate_model(
    model: LinearGaussianSSM,
    *,
    time_steps: int,
    seed: int,
    controls: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Sample latent states and observations from a supplied model."""
    rng = np.random.default_rng(seed)
    if controls is None:
        controls = np.zeros((time_steps, model.control_dim), dtype=np.float64)
    if controls.shape != (time_steps, model.control_dim):
        raise ValueError("test controls do not match the model")

    states = np.zeros((time_steps, model.state_dim), dtype=np.float64)
    observations = np.zeros((time_steps, model.obs_dim), dtype=np.float64)
    state = rng.multivariate_normal(model.mu_0, model.Sigma_0)
    for time_index in range(time_steps):
        control = controls[time_index]
        states[time_index] = state
        observations[time_index] = (
            model.C @ state
            + model.D @ control
            + rng.multivariate_normal(np.zeros(model.obs_dim), model.R)
        )
        state = (
            model.A @ state
            + model.B @ control
            + rng.multivariate_normal(np.zeros(model.state_dim), model.Q)
        )
    return states, observations
