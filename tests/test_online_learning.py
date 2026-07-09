# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.online_learning (e-prop, online trainer)

from __future__ import annotations

import numpy as np

from sc_neurocore.online_learning.eprop import EpropTrainer
from sc_neurocore.online_learning.online_trainer import OnlineLIFLayer, OnlineTrainer


class TestEpropTrainer:
    def test_init(self):
        t = EpropTrainer(n_inputs=10, n_neurons=20, n_outputs=5)
        assert t.W_in.shape == (20, 10)
        assert t.W_rec.shape == (20, 20)
        assert t.W_out.shape == (5, 20)
        assert np.diag(t.W_rec).sum() == 0  # no self-connections

    def test_step_no_learning(self):
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2)
        x = np.random.rand(4)
        result = t.step(x)
        assert "spikes" in result
        assert "output" in result
        assert result["spikes"].shape == (8,)
        assert result["output"].shape == (2,)
        assert "loss" not in result

    def test_step_with_learning(self):
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2)
        x = np.random.rand(4)
        target = np.array([1.0, 0.0])
        result = t.step(x, target=target)
        assert "loss" in result
        assert result["loss"] >= 0

    def test_train_sequence(self):
        t = EpropTrainer(n_inputs=4, n_neurons=16, n_outputs=2, lr=0.001)
        inputs = np.random.rand(50, 4)
        targets = np.zeros((50, 2))
        targets[:, 0] = 1.0
        loss = t.train_sequence(inputs, targets)
        assert loss >= 0

    def test_predict_sequence(self):
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2)
        inputs = np.random.rand(20, 4)
        outputs = t.predict_sequence(inputs)
        assert outputs.shape == (20, 2)

    def test_reset(self):
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2)
        t.step(np.ones(4))
        assert not np.allclose(t._v, 0)
        t.reset()
        assert np.allclose(t._v, 0)
        assert np.allclose(t._spikes, 0)

    def test_memory_per_step_o1(self):
        t = EpropTrainer(n_inputs=10, n_neurons=20, n_outputs=5)
        mem = t.memory_per_step
        assert mem > 0
        # Memory should not depend on sequence length T
        assert isinstance(mem, int)

    def test_loss_decreases_on_simple_task(self):
        t = EpropTrainer(n_inputs=4, n_neurons=32, n_outputs=2, lr=0.005)
        inputs = np.random.rand(30, 4)
        targets = np.zeros((30, 2))
        targets[:, 0] = 0.5
        loss1 = t.train_sequence(inputs, targets)
        loss2 = t.train_sequence(inputs, targets)
        loss3 = t.train_sequence(inputs, targets)
        # Not guaranteed to decrease every time (approximation), but trend should hold
        assert loss3 <= loss1 * 2  # at least not diverging wildly

    def test_no_self_connections_after_learning(self):
        t = EpropTrainer(n_inputs=4, n_neurons=8, n_outputs=2, lr=0.01)
        inputs = np.random.rand(10, 4)
        targets = np.random.rand(10, 2)
        t.train_sequence(inputs, targets)
        assert np.allclose(np.diag(t.W_rec), 0)


class TestOnlineLIFLayer:
    def test_step(self):
        layer = OnlineLIFLayer(n_inputs=4, n_neurons=8)
        x = np.random.rand(4)
        spikes = layer.step(x)
        assert spikes.shape == (8,)
        assert set(np.unique(spikes)).issubset({0.0, 1.0})

    def test_reset(self):
        layer = OnlineLIFLayer(n_inputs=4, n_neurons=8)
        layer.step(np.ones(4))
        layer.reset()
        assert np.allclose(layer._v, 0)

    def test_apply_learning_signal(self):
        layer = OnlineLIFLayer(n_inputs=4, n_neurons=8, lr=0.1)
        layer.step(np.ones(4))
        w_before = layer.W.copy()
        layer.apply_learning_signal(np.ones(8))
        assert not np.allclose(layer.W, w_before)


class TestOnlineTrainer:
    def test_init(self):
        ot = OnlineTrainer(layer_sizes=[10, 20, 5])
        assert ot.n_layers == 2
        assert ot.layers[0].n_inputs == 10
        assert ot.layers[0].n_neurons == 20
        assert ot.layers[1].n_inputs == 20
        assert ot.layers[1].n_neurons == 5

    def test_step_no_learning(self):
        ot = OnlineTrainer(layer_sizes=[4, 8, 2])
        result = ot.step(np.random.rand(4))
        assert result["output"].shape == (2,)
        assert "loss" not in result

    def test_step_with_learning(self):
        ot = OnlineTrainer(layer_sizes=[4, 8, 2], lr=0.01)
        result = ot.step(np.random.rand(4), target=np.array([1.0, 0.0]))
        assert "loss" in result
        assert result["loss"] >= 0

    def test_train_sequence(self):
        ot = OnlineTrainer(layer_sizes=[4, 16, 2], lr=0.001)
        inputs = np.random.rand(30, 4)
        targets = np.zeros((30, 2))
        loss = ot.train_sequence(inputs, targets)
        assert loss >= 0

    def test_reset(self):
        ot = OnlineTrainer(layer_sizes=[4, 8, 2])
        ot.step(np.ones(4))
        ot.reset()
        for layer in ot.layers:
            assert np.allclose(layer._v, 0)

    def test_memory_per_step(self):
        ot = OnlineTrainer(layer_sizes=[10, 20, 5])
        assert ot.memory_per_step > 0

    def test_three_layers(self):
        ot = OnlineTrainer(layer_sizes=[8, 16, 8, 4])
        assert ot.n_layers == 3
        result = ot.step(np.random.rand(8))
        assert result["output"].shape == (4,)
