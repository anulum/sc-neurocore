# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikingNet from former test_torch_training.py

"""Focused suite: TestSpikingNet from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestSpikingNet:
    def test_forward(self):
        net = SpikingNet(n_input=784, n_hidden=128, n_output=10, n_layers=2)
        x = torch.randn(25, 8, 784)  # (T=25, batch=8, features=784)
        spike_counts, mem_acc = net(x)
        assert spike_counts.shape == (8, 10)
        assert mem_acc.shape == (8, 10)

    def test_gradient_flow(self):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5)
        x = torch.randn(10, 4, 16)
        spike_counts, _ = net(x)
        loss = spike_counts.sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None

    def test_to_sc_weights(self):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5)
        sc = net.to_sc_weights()
        assert len(sc) == 3  # 2 hidden + 1 output
        for layer in sc:
            w = layer["weight"]
            assert w.min() >= 0.0
            assert w.max() <= 1.0

    def test_learnable_dynamics(self):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5, learn_beta=True, learn_threshold=True)
        info = model_info(net)
        assert len(info["learnable_dynamics"]) > 0

    @pytest.mark.parametrize(
        "surrogate", [fast_sigmoid, superspike, atan_surrogate, sigmoid_surrogate]
    )
    def test_different_surrogates(self, surrogate):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5, surrogate_fn=surrogate)
        x = torch.randn(5, 2, 16)
        spike_counts, _ = net(x)
        loss = spike_counts.sum()
        loss.backward()
