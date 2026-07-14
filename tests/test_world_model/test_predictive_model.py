# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public predictive-model integration contracts

"""Exercise the public LGSSM facade as one end-to-end production surface."""

from __future__ import annotations

import numpy as np

from _predictive_model_test_support import controlled_scalar_model, simulate_model
from sc_neurocore.world_model import _lgssm_backends as backends
from sc_neurocore.world_model._lgssm_backends import ExplicitBackendName
from sc_neurocore.world_model._lgssm_types import FloatArray
from sc_neurocore.world_model.predictive_model import (
    EMLearner,
    KalmanFilter,
    LinearGaussianSSM,
    PredictiveWorldModel,
    RTSSmoother,
)


NATIVE_BACKENDS: tuple[ExplicitBackendName, ...] = (
    "mojo",
    "rust",
    "julia",
    "go",
)


def _controlled_observations() -> tuple[FloatArray, FloatArray]:
    model = controlled_scalar_model()
    controls = np.random.default_rng(928).normal(size=(80, 1))
    _, observations = simulate_model(
        model,
        time_steps=80,
        seed=929,
        controls=controls,
    )
    return observations, controls


def test_public_facade_runs_filter_smoother_em_and_planner_pipeline() -> None:
    """All public responsibility owners interoperate through facade imports."""
    model = controlled_scalar_model()
    observations, controls = _controlled_observations()

    filtered = KalmanFilter(model).filter(
        observations,
        controls,
        backend="python",
    )
    smoothed = RTSSmoother(model).smooth(filtered)
    learner = EMLearner(max_iter=4, tol=0.0)
    learned = learner.fit(
        observations,
        model,
        controls,
        backend="python",
    )
    planner_model = PredictiveWorldModel(state_dim=2, action_dim=1, seed=930)
    forecast = planner_model.forecast_with_cov(
        np.zeros(2),
        np.eye(2),
        [np.zeros(1), np.ones(1)],
    )

    assert filtered.means.shape == (80, 1)
    assert smoothed.cross_covariances.shape == (79, 1, 1)
    assert isinstance(learned, LinearGaussianSSM)
    assert np.all(np.diff(learner.log_likelihood_history) >= -1e-7)
    assert [(mean.shape, covariance.shape) for mean, covariance in forecast] == [
        ((2,), (2, 2)),
        ((2,), (2, 2)),
    ]


def test_auto_dispatch_matches_declared_fastest_available_backend() -> None:
    """Default dispatch and the declared fastest available backend agree."""
    model = controlled_scalar_model()
    observations, controls = _controlled_observations()
    resolved = backends.resolve_backend("auto")

    automatic = KalmanFilter(model).filter(observations, controls)
    explicit = KalmanFilter(model).filter(
        observations,
        controls,
        backend=resolved,
    )

    assert resolved == next(
        backend for backend in backends.AUTO_BACKEND_ORDER if backends.probe_backend(backend)[0]
    )
    np.testing.assert_array_equal(automatic.means, explicit.means)
    np.testing.assert_array_equal(automatic.covariances, explicit.covariances)
    assert automatic.log_likelihood == explicit.log_likelihood


def test_all_maintained_native_filters_match_controlled_python_result() -> None:
    """The installed five-language forward-filter surface is value-consistent."""
    unavailable = {
        backend: reason
        for backend in NATIVE_BACKENDS
        for available, reason in [backends.probe_backend(backend)]
        if not available
    }
    assert unavailable == {}, unavailable

    model = controlled_scalar_model()
    observations, controls = _controlled_observations()
    reference = KalmanFilter(model).filter(
        observations,
        controls,
        backend="python",
    )
    for backend in NATIVE_BACKENDS:
        candidate = KalmanFilter(model).filter(
            observations,
            controls,
            backend=backend,
        )
        np.testing.assert_allclose(candidate.means, reference.means, atol=1e-9)
        np.testing.assert_allclose(
            candidate.covariances,
            reference.covariances,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            candidate.pred_means,
            reference.pred_means,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            candidate.pred_covariances,
            reference.pred_covariances,
            atol=1e-9,
        )
        assert abs(candidate.log_likelihood - reference.log_likelihood) <= 1e-7
