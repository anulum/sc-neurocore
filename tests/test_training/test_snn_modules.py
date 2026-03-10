# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for differentiable SNN modules."""

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.training.snn_modules import (
    LIFCell,
    RecurrentLIFCell,
    SpikingNet,
)
from sc_neurocore.training.surrogate import superspike


class TestLIFCell:
    def test_above_threshold_spikes(self):
        lif = LIFCell(beta=0.0, threshold=0.5)
        v = torch.zeros(4)
        spike, _ = lif(torch.ones(4), v)
        assert spike.sum().item() == 4

    def test_below_threshold_silent(self):
        lif = LIFCell(beta=0.0, threshold=2.0)
        v = torch.zeros(4)
        spike, _ = lif(torch.ones(4), v)
        assert spike.sum().item() == 0

    def test_membrane_reset_after_spike(self):
        lif = LIFCell(beta=0.0, threshold=1.0)
        spike, v = lif(torch.tensor([2.0]), torch.zeros(1))
        assert spike.item() == 1.0
        assert v.item() == pytest.approx(1.0)

    def test_leak_decay(self):
        lif = LIFCell(beta=0.5, threshold=10.0)
        _, v1 = lif(torch.tensor([0.0]), torch.tensor([4.0]))
        assert v1.item() == pytest.approx(2.0)

    def test_gradient_flows(self):
        lif = LIFCell()
        current = torch.randn(8, requires_grad=True)
        v = torch.zeros(8)
        spike, _ = lif(current, v)
        spike.sum().backward()
        assert current.grad is not None

    def test_alternate_surrogate(self):
        lif = LIFCell(surrogate_fn=superspike)
        spike, _ = lif(torch.tensor([2.0]), torch.zeros(1))
        assert spike.item() == 1.0


class TestRecurrentLIFCell:
    def test_recurrent_connection(self):
        cell = RecurrentLIFCell(n_neurons=4)
        v = torch.zeros(1, 4)
        spike_prev = torch.ones(1, 4)
        current = torch.zeros(1, 4)
        spike, v_next = cell(current, v, spike_prev)
        assert spike.shape == (1, 4)

    def test_gradient_flows(self):
        cell = RecurrentLIFCell(n_neurons=4)
        current = torch.randn(1, 4, requires_grad=True)
        v = torch.zeros(1, 4)
        spike_prev = torch.zeros(1, 4)
        spike, _ = cell(current, v, spike_prev)
        spike.sum().backward()
        assert current.grad is not None


class TestSpikingNet:
    def test_forward_shape(self):
        net = SpikingNet(n_input=10, n_hidden=32, n_output=3, n_layers=2)
        x = torch.randn(25, 8, 10)
        spk, mem = net(x)
        assert spk.shape == (8, 3)
        assert mem.shape == (8, 3)

    def test_gradient_flows_to_all_params(self):
        net = SpikingNet(n_input=10, n_hidden=16, n_output=5)
        x = torch.randn(10, 4, 10)
        spk, _ = net(x)
        spk.sum().backward()
        for name, p in net.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_spike_counts_nonnegative(self):
        net = SpikingNet(n_input=5, n_hidden=8, n_output=3)
        x = torch.randn(20, 4, 5)
        spk, _ = net(x)
        assert (spk >= 0).all()

    def test_to_sc_weights(self):
        net = SpikingNet(n_input=5, n_hidden=8, n_output=3, n_layers=1)
        weights = net.to_sc_weights()
        assert len(weights) == 2
        for w in weights:
            assert w.min() >= 0.0
            assert w.max() <= 1.0

    def test_single_layer(self):
        net = SpikingNet(n_input=5, n_hidden=8, n_output=3, n_layers=0)
        x = torch.randn(10, 2, 5)
        spk, mem = net(x)
        assert spk.shape == (2, 3)
