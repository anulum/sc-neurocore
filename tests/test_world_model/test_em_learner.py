# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled LGSSM expectation-maximisation tests

"""Likelihood, control-term, convergence, and held-out tests for LGSSM EM."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.world_model.predictive_model import (
    EMLearner,
    FilterResult,
    KalmanFilter,
    LinearGaussianSSM,
)
from _predictive_model_test_support import (
    controlled_scalar_model,
    scalar_random_walk,
    simulate_model,
)


def _scalar_initial_model() -> LinearGaussianSSM:
    return LinearGaussianSSM(
        A=np.array([[0.5]]),
        B=np.zeros((1, 0)),
        C=np.array([[2.0]]),
        D=np.zeros((1, 0)),
        Q=np.array([[1.0]]),
        R=np.array([[5.0]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )


@pytest.mark.parametrize(
    ("max_iter", "tol", "message"),
    [
        (0, 1e-4, "max_iter must be a positive integer"),
        (True, 1e-4, "max_iter must be a positive integer"),
        (10, -1.0, "tol must be finite and non-negative"),
        (10, np.inf, "tol must be finite and non-negative"),
    ],
)
def test_learner_rejects_invalid_configuration(
    max_iter: int,
    tol: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EMLearner(max_iter=max_iter, tol=tol)


def test_em_requires_two_finite_observations_and_matching_controls() -> None:
    learner = EMLearner(max_iter=2)
    model = controlled_scalar_model()
    with pytest.raises(ValueError, match="at least two"):
        learner.fit(np.zeros((1, 1)), model, np.zeros((1, 1)))
    with pytest.raises(ValueError, match="finite"):
        learner.fit(np.array([[0.0], [np.nan]]), model, np.zeros((2, 1)))
    with pytest.raises(ValueError, match="controls must have shape"):
        learner.fit(np.zeros((2, 1)), model, np.zeros((3, 1)))


def test_scalar_em_likelihood_is_monotone_non_decreasing() -> None:
    _, observations = simulate_model(
        scalar_random_walk(),
        time_steps=200,
        seed=42,
    )
    learner = EMLearner(max_iter=10, tol=1e-9)
    learner.fit(observations, _scalar_initial_model())

    assert len(learner.log_likelihood_history) > 1
    assert np.all(np.diff(learner.log_likelihood_history) >= -1e-7)


def test_em_improves_held_out_observation_likelihood() -> None:
    true_model = LinearGaussianSSM(
        A=np.array([[0.95]]),
        B=np.zeros((1, 0)),
        C=np.array([[2.0]]),
        D=np.zeros((1, 0)),
        Q=np.array([[0.05]]),
        R=np.array([[0.1]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    _, training = simulate_model(true_model, time_steps=350, seed=7)
    _, held_out = simulate_model(true_model, time_steps=180, seed=99)
    initial = LinearGaussianSSM(
        A=np.array([[0.95]]),
        B=np.zeros((1, 0)),
        C=np.array([[0.5]]),
        D=np.zeros((1, 0)),
        Q=np.array([[0.05]]),
        R=np.array([[0.1]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    initial_likelihood = (
        KalmanFilter(initial)
        .filter(
            held_out,
            backend="python",
        )
        .log_likelihood
    )
    learned = EMLearner(max_iter=30, tol=1e-6).fit(training, initial)
    learned_likelihood = (
        KalmanFilter(learned)
        .filter(
            held_out,
            backend="python",
        )
        .log_likelihood
    )

    assert learned_likelihood > initial_likelihood


def test_controlled_em_subtracts_fixed_b_and_d_contributions() -> None:
    true_model = controlled_scalar_model()
    rng = np.random.default_rng(32018)
    controls = rng.normal(size=(300, 1))
    _, observations = simulate_model(
        true_model,
        time_steps=300,
        seed=32019,
        controls=controls,
    )
    initial = LinearGaussianSSM(
        A=np.array([[0.45]]),
        B=true_model.B,
        C=np.array([[0.7]]),
        D=true_model.D,
        Q=np.array([[0.2]]),
        R=np.array([[0.2]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    learner = EMLearner(max_iter=15, tol=0.0)
    learned = learner.fit(observations, initial, controls)

    assert np.all(np.diff(learner.log_likelihood_history) >= -1e-7)
    np.testing.assert_array_equal(learned.B, initial.B)
    np.testing.assert_array_equal(learned.D, initial.D)
    assert learned.C[0, 0] > 0.8
    assert learned.A[0, 0] == pytest.approx(true_model.A[0, 0], abs=0.12)


def test_multivariate_em_preserves_positive_covariance_contracts() -> None:
    true_model = LinearGaussianSSM(
        A=np.array([[0.82, 0.18], [-0.11, 0.73]]),
        B=np.zeros((2, 0)),
        C=np.array([[1.0, 0.35], [-0.25, 0.9]]),
        D=np.zeros((2, 0)),
        Q=np.array([[0.07, 0.015], [0.015, 0.05]]),
        R=np.array([[0.09, 0.01], [0.01, 0.08]]),
        mu_0=np.array([0.1, -0.2]),
        Sigma_0=np.array([[0.5, 0.08], [0.08, 0.4]]),
    )
    _, observations = simulate_model(true_model, time_steps=240, seed=1802)
    initial = LinearGaussianSSM(
        A=np.array([[0.55, 0.05], [0.02, 0.5]]),
        B=np.zeros((2, 0)),
        C=np.eye(2),
        D=np.zeros((2, 0)),
        Q=np.eye(2) * 0.2,
        R=np.eye(2) * 0.2,
        mu_0=np.zeros(2),
        Sigma_0=np.eye(2),
    )
    learner = EMLearner(max_iter=12, tol=0.0)
    learned = learner.fit(observations, initial)

    assert np.all(np.diff(learner.log_likelihood_history) >= -1e-7)
    assert float(np.min(np.linalg.eigvalsh(learned.Q))) >= -1e-12
    assert float(np.min(np.linalg.eigvalsh(learned.R))) > 0.0
    assert float(np.min(np.linalg.eigvalsh(learned.Sigma_0))) > 0.0


def test_em_honours_convergence_tolerance() -> None:
    _, observations = simulate_model(
        scalar_random_walk(),
        time_steps=80,
        seed=123,
    )
    learner = EMLearner(max_iter=25, tol=1e6)
    learned = learner.fit(observations, _scalar_initial_model())

    assert isinstance(learned, LinearGaussianSSM)
    assert len(learner.log_likelihood_history) == 2


def test_em_rejects_unknown_filter_backend() -> None:
    _, observations = simulate_model(scalar_random_walk(), time_steps=3, seed=4)
    with pytest.raises(ValueError, match="backend must be"):
        EMLearner(max_iter=1).fit(
            observations,
            _scalar_initial_model(),
            backend="cuda",
        )


def test_em_rejects_decreasing_final_candidate_with_one_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_filter = KalmanFilter.filter
    call_count = 0

    def decreasing_filter(
        filter_instance: KalmanFilter,
        observations: npt.ArrayLike,
        controls: npt.ArrayLike | None = None,
        backend: str = "auto",
    ) -> FilterResult:
        nonlocal call_count
        result = original_filter(
            filter_instance,
            observations,
            controls,
            backend,
        )
        result.log_likelihood = -1.0 - call_count
        call_count += 1
        return result

    monkeypatch.setattr(KalmanFilter, "filter", decreasing_filter)
    _, observations = simulate_model(scalar_random_walk(), time_steps=8, seed=19)
    with pytest.raises(RuntimeError, match="decreased beyond float64 round-off"):
        EMLearner(max_iter=1, tol=0.0).fit(observations, _scalar_initial_model())
    assert call_count == 2
