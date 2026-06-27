# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available

"""Production-surface tests for spike-domain few-shot learners."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from sc_neurocore.few_shot import HebbianFewShot, SpikePrototypeNet


FloatArray = NDArray[np.float64]


def _vec(values: list[float]) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def test_hebbian_memory_scores_queries_and_exports_defensive_weights() -> None:
    """HebbianFewShot stores real support vectors and exposes copy-safe weights."""
    learner = HebbianFewShot(n_features=4, n_classes=3, lr_hebbian=0.5)

    learner.store(_vec([1.0, 0.0, 1.0, 0.0]), label=0)
    learner.store(_vec([0.0, 1.0, 0.0, 1.0]), label=1)

    scores = learner.query_scores(_vec([0.8, 0.1, 0.9, 0.0]))
    assert learner.query(_vec([0.8, 0.1, 0.9, 0.0])) == 0
    assert scores.shape == (3,)
    assert scores[0] > scores[1]
    assert scores[2] == 0.0

    weights = learner.export_weights()
    weights[0, 0] = 999.0
    assert learner.memory[0, 0] == 0.5


def test_hebbian_few_shot_episode_resets_old_memory() -> None:
    """Episode calls reset stale support rows before storing new support sets."""
    learner = HebbianFewShot(n_features=4, n_classes=2)
    learner.store(_vec([1.0, 0.0, 0.0, 0.0]), 0)

    predictions = learner.few_shot_episode(
        support_x=[_vec([0.0, 0.0, 1.0, 0.0])],
        support_y=[1],
        query_x=[_vec([0.0, 0.1, 0.9, 0.0])],
    )

    assert predictions == [1]
    assert learner.export_weights()[0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_temporal_spike_trains_are_averaged_before_storage_and_query() -> None:
    """Temporal support and query tensors use their spike-rate mean vector."""
    learner = HebbianFewShot(n_features=4, n_classes=2)
    spike_train = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    learner.store(spike_train, label=0)

    assert learner.query(spike_train) == 0
    assert np.allclose(learner.export_weights()[0], _vec([2.0 / 30.0, 0.0, 0.0, 0.0]))


def test_hebbian_zero_query_scores_as_zero_similarity() -> None:
    """Zero-norm query vectors produce bounded zero similarity scores."""
    learner = HebbianFewShot(n_features=3, n_classes=1)
    learner.store(_vec([1.0, 0.0, 0.0]), label=0)

    assert learner.query_scores(_vec([0.0, 0.0, 0.0])).tolist() == [0.0]


def test_hebbian_validation_fails_closed_before_state_mutation() -> None:
    """Constructor, label, shape, finite, and support-list guards reject bad inputs."""
    with pytest.raises(ValueError, match="n_features"):
        HebbianFewShot(n_features=0, n_classes=2)
    with pytest.raises(ValueError, match="n_classes"):
        HebbianFewShot(n_features=2, n_classes=0)
    with pytest.raises(ValueError, match="lr_hebbian"):
        HebbianFewShot(n_features=2, n_classes=2, lr_hebbian=float("nan"))

    learner = HebbianFewShot(n_features=2, n_classes=2)
    before = learner.export_weights()

    with pytest.raises(ValueError, match="label"):
        learner.store(_vec([1.0, 0.0]), label=3)
    with pytest.raises(ValueError, match="features"):
        learner.store(_vec([1.0, 0.0, 0.0]), label=0)
    with pytest.raises(ValueError, match="finite"):
        learner.store(_vec([1.0, float("inf")]), label=0)
    with pytest.raises(ValueError, match="shape"):
        learner.store(np.zeros((1, 1, 2), dtype=np.float64), label=0)
    with pytest.raises(ValueError, match="support example"):
        learner.query(_vec([1.0, 0.0]))
    with pytest.raises(ValueError, match="same length"):
        learner.few_shot_episode([_vec([1.0, 0.0])], [], [_vec([1.0, 0.0])])

    assert np.array_equal(learner.export_weights(), before)


def test_spike_prototype_net_classifies_with_cosine_euclidean_and_hamming() -> None:
    """Prototype classifier supports every documented metric on real vectors."""
    support_x = [_vec([1.0, 0.0, 0.0, 0.0]), _vec([0.0, 0.0, 1.0, 0.0])]
    support_y = [0, 1]
    query_x = [_vec([0.9, 0.1, 0.0, 0.0])]

    assert SpikePrototypeNet(n_features=4, metric="cosine").classify(
        support_x, support_y, query_x
    ) == [0]
    assert SpikePrototypeNet(n_features=4, metric="euclidean").classify(
        support_x, support_y, query_x
    ) == [0]
    assert SpikePrototypeNet(n_features=4, metric="hamming").classify(
        support_x, support_y, query_x
    ) == [0]


def test_spike_prototype_net_averages_multi_support_and_exports_copies() -> None:
    """Multi-shot class prototypes are mean vectors and export defensively."""
    net = SpikePrototypeNet(n_features=3)

    predictions = net.classify(
        support_x=[_vec([1.0, 0.0, 0.0]), _vec([0.5, 0.5, 0.0]), _vec([0.0, 0.0, 1.0])],
        support_y=[0, 0, 1],
        query_x=[_vec([0.7, 0.2, 0.0])],
    )

    exported = net.export_prototypes()
    exported[0][0] = 999.0
    assert predictions == [0]
    assert np.allclose(net.prototypes[0], _vec([0.75, 0.25, 0.0]))


def test_spike_prototype_net_validation_rejects_invalid_contracts() -> None:
    """Prototype learner rejects unsupported metrics and malformed episodes."""
    with pytest.raises(ValueError, match="n_features"):
        SpikePrototypeNet(n_features=0)
    with pytest.raises(ValueError, match="metric"):
        SpikePrototypeNet(n_features=2, metric="chebyshev")  # type: ignore[arg-type]  # invalid metric exercises runtime guard

    net = SpikePrototypeNet(n_features=2)
    with pytest.raises(ValueError, match="at least one"):
        net.classify([], [], [_vec([1.0, 0.0])])
    with pytest.raises(ValueError, match="same length"):
        net.classify([_vec([1.0, 0.0])], [], [_vec([1.0, 0.0])])
    with pytest.raises(ValueError, match="integers"):
        net.classify(
            [_vec([1.0, 0.0])],
            [1.5],  # type: ignore[list-item]  # non-integer label exercises runtime guard
            [_vec([1.0, 0.0])],
        )
    with pytest.raises(ValueError, match="features"):
        net.classify([_vec([1.0])], [0], [_vec([1.0, 0.0])])
