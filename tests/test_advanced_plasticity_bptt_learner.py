# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBPTTLearner from former test_advanced_plasticity.py

"""Focused suite: TestBPTTLearner from former test_advanced_plasticity.py."""

from __future__ import annotations

from typing import Any
from sc_neurocore.network.population import Population
import numpy as np
import pytest
from sc_neurocore.network.network import Network
from sc_neurocore.network.projection import Projection
from sc_neurocore.learning.advanced import (
    BPTTLearner,
    TBPTTLearner,
)


class TestBPTTLearner:
    """Backpropagation through time on a two-population network."""

    def test_train_step_returns_loss(
        self, simple_net: tuple[Network, Population, Population, Projection]
    ) -> None:
        """A training step returns a non-negative float loss."""
        net, pop_a, pop_b, _ = simple_net
        n_steps = 10
        inputs = np.random.randn(n_steps, pop_a.n) * 5
        targets = np.zeros((n_steps, pop_a.n))

        def mse(pred: np.ndarray[Any, Any], tgt: np.ndarray[Any, Any]) -> float:
            return float(np.mean((pred - tgt) ** 2))

        learner = BPTTLearner(net, loss_fn=mse, lr=1e-3)
        loss = learner.train_step(inputs, targets)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_weights_change(
        self, simple_net: tuple[Network, Population, Population, Projection]
    ) -> None:
        """A training step changes the projection weights."""
        net, pop_a, _, proj = simple_net
        w_before = proj.data.copy()
        inputs = np.random.randn(10, pop_a.n) * 10
        targets = np.ones((10, pop_a.n))
        learner = BPTTLearner(net, loss_fn=lambda p, t: float(np.mean((p - t) ** 2)))
        learner.train_step(inputs, targets)
        assert not np.allclose(proj.data, w_before)

    @pytest.mark.parametrize(
        ("input_shape", "target_shape", "error"),
        [
            ((5,), (2, 5), "two-dimensional"),
            ((2, 5), (3, 5), "same number of timesteps"),
            ((2, 4), (2, 5), "input width"),
            ((2, 5), (2, 4), "target width"),
            ((0, 5), (0, 5), "at least one timestep"),
        ],
    )
    def test_refuses_invalid_training_batch(
        self,
        simple_net: tuple[Network, Population, Population, Projection],
        input_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        error: str,
    ) -> None:
        """Each malformed training batch is rejected."""
        net, _, _, _ = simple_net
        learner = BPTTLearner(net, loss_fn=lambda p, t: float(np.mean((p - t) ** 2)))
        with pytest.raises(ValueError, match=error):
            learner.train_step(np.zeros(input_shape), np.zeros(target_shape))

    def test_refuses_non_direct_or_delayed_network(
        self, simple_net: tuple[Network, Population, Population, Projection]
    ) -> None:
        """Networks that are not one direct, undelayed projection are refused."""
        _, source, target, _ = simple_net
        arrays = (np.zeros((1, source.n)), np.zeros((1, target.n)))

        def loss(p: np.ndarray[Any, Any], t: np.ndarray[Any, Any]) -> float:
            return float(np.mean((p - t) ** 2))

        with pytest.raises(NotImplementedError, match="exactly two populations"):
            BPTTLearner(Network(source), loss_fn=loss).train_step(*arrays)
        reverse = Projection(target, source, weight=0.3)
        with pytest.raises(NotImplementedError, match="direct input-to-output"):
            BPTTLearner(Network(source, target, reverse), loss_fn=loss).train_step(*arrays)
        delayed = Projection(source, target, weight=0.3, delay=1.0)
        with pytest.raises(NotImplementedError, match="delayed projections"):
            BPTTLearner(Network(source, target, delayed), loss_fn=loss).train_step(*arrays)

    def test_tbptt_refuses_nonpositive_window(
        self, simple_net: tuple[Network, Population, Population, Projection]
    ) -> None:
        """A truncation window that is not positive is rejected."""
        net, source, target, _ = simple_net
        learner = TBPTTLearner(net, loss_fn=lambda p, t: 0.0, k=0)
        with pytest.raises(ValueError, match="k must be positive"):
            learner.train_step(np.zeros((1, source.n)), np.zeros((1, target.n)))
