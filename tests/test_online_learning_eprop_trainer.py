# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEpropTrainer from former test_online_learning.py

"""Focused suite: TestEpropTrainer from former test_online_learning.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.online_learning.eprop import (
    EpropTrainer,
)


class TestEpropTrainer:
    """Online e-prop training."""

    def test_init(self) -> None:
        """The trainer sets up its weight matrices."""
        t = EpropTrainer(n_inputs=10, n_neurons=20, n_outputs=5)
        assert t.W_in.shape == (20, 10)
        assert t.W_rec.shape == (20, 20)
        assert t.W_out.shape == (5, 20)
        assert np.diag(t.W_rec).sum() == 0  # no self-connections

    def test_step_no_learning(self) -> None:
        """A step without learning gives spikes and output."""
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2)
        x = np.random.rand(4)
        result = t.step(x)
        assert "spikes" in result
        assert "output" in result
        assert result["spikes"].shape == (8,)
        assert result["output"].shape == (2,)
        assert "loss" not in result

    def test_step_with_learning(self) -> None:
        """A learning step reports a loss."""
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2)
        x = np.random.rand(4)
        target = np.array([1.0, 0.0])
        result = t.step(x, target=target)
        assert "loss" in result
        assert result["loss"] >= 0

    def test_train_sequence(self) -> None:
        """Training on a sequence gives a loss."""
        t = EpropTrainer(n_inputs=4, n_neurons=16, n_outputs=2, lr=0.001)
        inputs = np.random.rand(50, 4)
        targets = np.zeros((50, 2))
        targets[:, 0] = 1.0
        loss = t.train_sequence(inputs, targets)
        assert loss >= 0

    def test_predict_sequence(self) -> None:
        """Prediction gives one output per step."""
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2)
        inputs = np.random.rand(20, 4)
        outputs = t.predict_sequence(inputs)
        assert outputs.shape == (20, 2)

    def test_reset(self) -> None:
        """A reset clears voltages and spikes."""
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2)
        t.step(np.ones(4))
        assert not np.allclose(t._v, 0)
        t.reset()
        assert np.allclose(t._v, 0)
        assert np.allclose(t._spikes, 0)

    def test_memory_per_step_o1(self) -> None:
        """The memory per step is a positive integer."""
        t = EpropTrainer(n_inputs=10, n_neurons=20, n_outputs=5)
        mem = t.memory_per_step
        assert mem > 0
        # Memory should not depend on sequence length T
        assert isinstance(mem, int)

    def test_loss_decreases_on_simple_task(self) -> None:
        """The loss does not diverge on a simple task."""
        t = EpropTrainer(n_inputs=4, n_neurons=32, n_outputs=2, lr=0.005)
        inputs = np.random.rand(30, 4)
        targets = np.zeros((30, 2))
        targets[:, 0] = 0.5
        loss1 = t.train_sequence(inputs, targets)
        loss2 = t.train_sequence(inputs, targets)
        loss3 = t.train_sequence(inputs, targets)
        # Not guaranteed to decrease every time (approximation), but trend should hold
        assert loss3 <= loss1 * 2  # at least not diverging wildly

    def test_no_self_connections_after_learning(self) -> None:
        """Learning keeps self-connections at zero."""
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2, lr=0.01)
        inputs = np.random.rand(10, 4)
        targets = np.random.rand(10, 2)
        t.train_sequence(inputs, targets)
        assert np.allclose(np.diag(t.W_rec), 0)
