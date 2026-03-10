# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for new training features: neuron models, surrogates, encoding, regularization."""

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.training.encoding import delta_encode, latency_encode, rate_encode
from sc_neurocore.training.losses import spike_l1_loss, spike_l2_loss
from sc_neurocore.training.loops import auto_device
from sc_neurocore.training.snn_modules import IFCell, SynapticCell
from sc_neurocore.training.surrogate import sigmoid_surrogate, straight_through, triangular


# ---------------------------------------------------------------------------
# New surrogate gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [sigmoid_surrogate, straight_through, triangular])
class TestNewSurrogates:
    def test_forward_is_heaviside(self, fn):
        x = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0])
        out = fn(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert torch.equal(out, expected)

    def test_backward_nonzero(self, fn):
        x = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
        fn(x).sum().backward()
        assert x.grad is not None
        assert (x.grad.abs() > 0).all()

    def test_batch_shape(self, fn):
        x = torch.randn(16, 64, requires_grad=True)
        out = fn(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad.shape == x.shape


def test_sigmoid_slope_effect():
    x1 = torch.tensor([0.01], requires_grad=True)
    sigmoid_surrogate(x1, slope=20.0).backward()
    x2 = torch.tensor([0.01], requires_grad=True)
    sigmoid_surrogate(x2, slope=2.0).backward()
    assert x1.grad.abs().item() > x2.grad.abs().item()


def test_triangular_width_effect():
    x1 = torch.tensor([0.8], requires_grad=True)
    triangular(x1, width=0.5).backward()
    x2 = torch.tensor([0.8], requires_grad=True)
    triangular(x2, width=2.0).backward()
    # x=0.8 outside width=0.5 (zero grad) but inside width=2.0 (nonzero grad)
    assert x2.grad.abs().item() > x1.grad.abs().item()


# ---------------------------------------------------------------------------
# New neuron cells
# ---------------------------------------------------------------------------


class TestIFCell:
    def test_no_leak(self):
        cell = IFCell(threshold=10.0)
        v = torch.tensor([5.0])
        spike, v_next = cell(torch.tensor([0.0]), v)
        assert v_next.item() == pytest.approx(5.0)
        assert spike.item() == 0.0

    def test_spike_and_reset(self):
        cell = IFCell(threshold=1.0)
        spike, v = cell(torch.tensor([2.0]), torch.zeros(1))
        assert spike.item() == 1.0
        assert v.item() == pytest.approx(1.0)

    def test_gradient_flows(self):
        cell = IFCell()
        x = torch.randn(8, requires_grad=True)
        spike, _ = cell(x, torch.zeros(8))
        spike.sum().backward()
        assert x.grad is not None


class TestSynapticCell:
    def test_dual_exponential(self):
        cell = SynapticCell(alpha=0.5, beta=0.5, threshold=100.0)
        i_syn = torch.zeros(4)
        v = torch.zeros(4)
        current = torch.ones(4)
        spike, i_syn_next, v_next = cell(current, i_syn, v)
        # i_syn = 0.5*0 + 1 = 1, v = 0.5*0 + 1 = 1
        assert i_syn_next.mean().item() == pytest.approx(1.0, abs=0.01)
        assert v_next.mean().item() == pytest.approx(1.0, abs=0.01)

    def test_spike_and_reset(self):
        cell = SynapticCell(alpha=0.0, beta=0.0, threshold=0.5)
        spike, _, _ = cell(torch.ones(4), torch.zeros(4), torch.zeros(4))
        assert spike.sum().item() == 4.0

    def test_learnable_beta(self):
        cell = SynapticCell(beta=0.8, learn_beta=True)
        assert any("beta_logit" in n for n, _ in cell.named_parameters())
        assert 0 < cell.beta.item() < 1

    def test_gradient_flows(self):
        cell = SynapticCell()
        x = torch.randn(8, requires_grad=True)
        spike, _, _ = cell(x, torch.zeros(8), torch.zeros(8))
        spike.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Spike encoding
# ---------------------------------------------------------------------------


class TestRateEncode:
    def test_shape(self):
        x = torch.rand(16, 10)
        spikes = rate_encode(x, n_timesteps=25)
        assert spikes.shape == (25, 16, 10)

    def test_values_binary(self):
        spikes = rate_encode(torch.rand(8), n_timesteps=100)
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

    def test_high_prob_more_spikes(self):
        high = rate_encode(torch.tensor([0.9]), n_timesteps=1000)
        low = rate_encode(torch.tensor([0.1]), n_timesteps=1000)
        assert high.sum() > low.sum()

    def test_clamps_input(self):
        spikes = rate_encode(torch.tensor([1.5, -0.5]), n_timesteps=10)
        assert spikes.shape == (10, 2)


class TestLatencyEncode:
    def test_shape(self):
        x = torch.rand(8, 4)
        spikes = latency_encode(x, n_timesteps=20)
        assert spikes.shape == (20, 8, 4)

    def test_one_spike_per_input(self):
        x = torch.tensor([0.5])
        spikes = latency_encode(x, n_timesteps=20, tau=5.0)
        assert spikes.sum().item() == 1.0

    def test_high_value_spikes_early(self):
        spikes_high = latency_encode(torch.tensor([0.95]), n_timesteps=20, tau=5.0)
        spikes_low = latency_encode(torch.tensor([0.1]), n_timesteps=20, tau=5.0)
        t_high = spikes_high.squeeze().argmax().item()
        t_low = spikes_low.squeeze().argmax().item()
        assert t_high <= t_low


class TestDeltaEncode:
    def test_shape(self):
        x = torch.randn(10, 4)
        spikes = delta_encode(x, threshold=0.1)
        assert spikes.shape == (10, 4)

    def test_constant_signal_no_spikes(self):
        x = torch.ones(10, 4) * 5.0
        spikes = delta_encode(x, threshold=0.1)
        assert spikes.sum().item() == 0.0

    def test_step_change_spikes(self):
        x = torch.zeros(10, 1)
        x[5:] = 1.0
        spikes = delta_encode(x, threshold=0.5)
        assert spikes[5, 0].item() == 1.0


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------


class TestRegularization:
    def test_l1_nonnegative(self):
        spk = torch.randn(8, 10).abs()
        loss = spike_l1_loss(spk, n_timesteps=25)
        assert loss.item() >= 0

    def test_l2_nonnegative(self):
        spk = torch.randn(8, 10).abs()
        loss = spike_l2_loss(spk, n_timesteps=25)
        assert loss.item() >= 0

    def test_l1_zero_for_silent(self):
        spk = torch.zeros(8, 10)
        assert spike_l1_loss(spk, n_timesteps=25).item() == 0.0

    def test_l2_zero_for_silent(self):
        spk = torch.zeros(8, 10)
        assert spike_l2_loss(spk, n_timesteps=25).item() == 0.0

    def test_l2_higher_for_active(self):
        active = torch.ones(8, 10) * 20.0
        sparse = torch.ones(8, 10) * 2.0
        assert spike_l2_loss(active, 25) > spike_l2_loss(sparse, 25)


# ---------------------------------------------------------------------------
# GPU auto-detection
# ---------------------------------------------------------------------------


def test_auto_device_returns_device():
    dev = auto_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda", "mps")
