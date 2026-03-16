# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for differentiable SNN modules

"""Tests for differentiable SNN modules."""

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.training.snn_modules import (
    ALIFCell,
    ConvSpikingNet,
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


class TestLearnableBeta:
    def test_beta_is_parameter(self):
        lif = LIFCell(beta=0.9, learn_beta=True)
        param_names = [n for n, _ in lif.named_parameters()]
        assert "_beta_logit" in param_names

    def test_beta_in_valid_range(self):
        lif = LIFCell(beta=0.9, learn_beta=True)
        assert 0 < lif.beta.item() < 1

    def test_beta_gradient_flows(self):
        lif = LIFCell(beta=0.9, learn_beta=True)
        current = torch.ones(4) * 2.0
        v = torch.ones(4) * 0.5
        spike, v_next = lif(current, v)
        (spike.sum() + v_next.sum()).backward()
        assert lif._beta_logit.grad is not None
        assert lif._beta_logit.grad.abs().item() > 0

    def test_beta_round_trips(self):
        lif = LIFCell(beta=0.85, learn_beta=True)
        assert lif.beta.item() == pytest.approx(0.85, abs=1e-5)


class TestLearnableThreshold:
    def test_threshold_is_parameter(self):
        lif = LIFCell(threshold=1.0, learn_threshold=True)
        param_names = [n for n, _ in lif.named_parameters()]
        assert "_threshold_log" in param_names

    def test_threshold_gradient_flows(self):
        lif = LIFCell(threshold=1.0, learn_threshold=True)
        current = torch.ones(4) * 1.5
        v = torch.zeros(4)
        spike, _ = lif(current, v)
        spike.sum().backward()
        assert lif._threshold_log.grad is not None

    def test_threshold_round_trips(self):
        lif = LIFCell(threshold=2.5, learn_threshold=True)
        assert lif.threshold.item() == pytest.approx(2.5, abs=1e-5)


class TestALIFCell:
    def test_adaptation_increases_threshold(self):
        alif = ALIFCell(beta=0.0, threshold=1.0, rho=0.9, beta_adapt=1.0)
        v = torch.zeros(4)
        a = torch.zeros(4)
        current = torch.ones(4) * 2.0
        spike, v_next, a_next = alif(current, v, a)
        # After first spike, a should increase
        assert (a_next > 0).all()

    def test_adaptation_decays_without_spikes(self):
        alif = ALIFCell(beta=0.0, threshold=100.0, rho=0.5)
        v = torch.zeros(4)
        a = torch.ones(4)
        current = torch.zeros(4)
        _, _, a_next = alif(current, v, a)
        # rho=0.5, no spikes: a decays to 0.5
        assert a_next.mean().item() == pytest.approx(0.5, abs=0.01)

    def test_gradient_flows(self):
        alif = ALIFCell()
        current = torch.randn(8, requires_grad=True)
        v = torch.zeros(8)
        a = torch.zeros(8)
        spike, _, _ = alif(current, v, a)
        spike.sum().backward()
        assert current.grad is not None

    def test_output_shapes(self):
        alif = ALIFCell()
        current = torch.randn(2, 8)
        v = torch.zeros(2, 8)
        a = torch.zeros(2, 8)
        spike, v_next, a_next = alif(current, v, a)
        assert spike.shape == (2, 8)
        assert v_next.shape == (2, 8)
        assert a_next.shape == (2, 8)


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

    def test_learnable_params_gradient(self):
        net = SpikingNet(
            n_input=10,
            n_hidden=16,
            n_output=5,
            learn_beta=True,
            learn_threshold=True,
        )
        x = torch.randn(5, 2, 10)
        spk, _ = net(x)
        spk.sum().backward()
        beta_params = [p for n, p in net.named_parameters() if "beta_logit" in n]
        thresh_params = [p for n, p in net.named_parameters() if "threshold_log" in n]
        assert len(beta_params) > 0
        assert len(thresh_params) > 0
        for p in beta_params + thresh_params:
            assert p.grad is not None


class TestConvSpikingNet:
    def test_forward_shape(self):
        net = ConvSpikingNet(n_output=10)
        x = torch.randn(10, 4, 1, 28, 28)
        spk, mem = net(x)
        assert spk.shape == (4, 10)
        assert mem.shape == (4, 10)

    def test_gradient_flows(self):
        net = ConvSpikingNet(n_output=5)
        x = torch.randn(5, 2, 1, 28, 28, requires_grad=True)
        spk, _ = net(x)
        spk.sum().backward()
        assert x.grad is not None

    def test_to_sc_weights(self):
        net = ConvSpikingNet(n_output=3)
        weights = net.to_sc_weights()
        assert len(weights) == 4  # conv1, conv2, fc1, fc2
        for w in weights:
            assert w.min() >= 0.0
            assert w.max() <= 1.0

    def test_learnable_params(self):
        net = ConvSpikingNet(n_output=5, learn_beta=True, learn_threshold=True)
        x = torch.randn(3, 2, 1, 28, 28)
        spk, _ = net(x)
        spk.sum().backward()
        beta_params = [p for n, p in net.named_parameters() if "beta_logit" in n]
        assert len(beta_params) == 4  # 4 LIF layers

    def test_spike_counts_nonnegative(self):
        net = ConvSpikingNet(n_output=10)
        x = torch.randn(5, 4, 1, 28, 28)
        spk, _ = net(x)
        assert (spk >= 0).all()
