# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore — Tests for PyTorch surrogate gradient training stack

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import sc_neurocore.training.loops as training_loops
from sc_neurocore.training import (
    HAS_TORCH,
    # Cells
    IFCell,
    LIFCell,
    ALIFCell,
    SynapticCell,
    RecurrentLIFCell,
    ExpIFCell,
    AdExCell,
    LapicqueCell,
    AlphaCell,
    SecondOrderLIFCell,
    # Networks
    SpikingNet,
    ConvSpikingNet,
    # Surrogate
    fast_sigmoid,
    superspike,
    atan_surrogate,
    sigmoid_surrogate,
    straight_through,
    triangular,
    # Encoding
    rate_encode,
    latency_encode,
    delta_encode,
    # Losses
    spike_count_loss,
    membrane_loss,
    spike_rate_loss,
    spike_l1_loss,
    spike_l2_loss,
    # Training
    auto_device,
    train_epoch,
    evaluate,
    # Utilities
    SpikeMonitor,
    model_info,
    population_decode,
    reset_states,
    # Delay
    DelayLinear,
)


class TestHasTorch:
    def test_torch_available(self):
        assert HAS_TORCH is True


# ── Surrogate gradients ──────────────────────────────────────────────


class TestSurrogates:
    @pytest.mark.parametrize(
        "fn", [fast_sigmoid, superspike, atan_surrogate, sigmoid_surrogate, triangular]
    )
    def test_forward_is_heaviside(self, fn):
        x = torch.tensor([-1.0, -0.1, 0.1, 1.0])
        out = fn(x)
        assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_straight_through_forward(self):
        x = torch.tensor([-1.0, 0.5])
        out = straight_through(x)
        assert torch.equal(out, torch.tensor([0.0, 1.0]))

    @pytest.mark.parametrize(
        "fn", [fast_sigmoid, superspike, atan_surrogate, sigmoid_surrogate, triangular]
    )
    def test_backward_nonzero(self, fn):
        x = torch.tensor([0.0], requires_grad=True)
        out = fn(x)
        out.backward()
        assert x.grad is not None
        assert x.grad.item() > 0

    def test_straight_through_backward(self):
        x = torch.tensor([0.5], requires_grad=True)
        out = straight_through(x)
        out.backward()
        assert x.grad.item() == 1.0

    @pytest.mark.parametrize(
        "fn", [fast_sigmoid, superspike, atan_surrogate, sigmoid_surrogate, triangular]
    )
    def test_gradient_peak_at_threshold(self, fn):
        """Surrogate gradient should peak near x=0 (threshold crossing)."""
        near = torch.tensor([0.01], requires_grad=True)
        far = torch.tensor([5.0], requires_grad=True)
        fn(near).backward()
        fn(far).backward()
        assert near.grad.abs().item() > far.grad.abs().item()

    def test_batch_surrogates(self):
        x = torch.randn(32, 128, requires_grad=True)
        out = atan_surrogate(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad.shape == x.shape


# ── Neuron cells ──────────────────────────────────────────────────────


class TestLIFCell:
    def test_forward_shape(self):
        cell = LIFCell(beta=0.9)
        current = torch.randn(8, 64)
        v = torch.zeros(8, 64)
        spike, v_next = cell(current, v)
        assert spike.shape == (8, 64)
        assert v_next.shape == (8, 64)

    def test_spikes_binary(self):
        cell = LIFCell(beta=0.9, threshold=0.5)
        current = torch.randn(32, 16)
        v = torch.zeros(32, 16)
        spike, _ = cell(current, v)
        assert set(spike.unique().tolist()).issubset({0.0, 1.0})

    def test_membrane_reset(self):
        cell = LIFCell(beta=0.9, threshold=1.0)
        current = torch.tensor([[2.0]])  # above threshold
        v = torch.zeros(1, 1)
        spike, v_next = cell(current, v)
        assert spike.item() == 1.0
        assert v_next.item() < 2.0  # reset subtracts threshold

    def test_learnable_beta(self):
        cell = LIFCell(beta=0.9, learn_beta=True)
        assert any("beta_logit" in n for n, _ in cell.named_parameters())

    def test_learnable_threshold(self):
        cell = LIFCell(threshold=1.0, learn_threshold=True)
        assert any("threshold_log" in n for n, _ in cell.named_parameters())

    def test_gradient_flows(self):
        cell = LIFCell(beta=0.9)
        w = torch.randn(16, 32, requires_grad=True)
        x = torch.randn(4, 32)
        current = x @ w.T
        v = torch.zeros(4, 16)
        spike, _ = cell(current, v)
        loss = spike.sum()
        loss.backward()
        assert w.grad is not None


class TestIFCell:
    def test_no_leak(self):
        cell = IFCell(threshold=10.0)
        v = torch.tensor([[5.0]])
        current = torch.tensor([[1.0]])
        _, v_next = cell(current, v)
        assert v_next.item() == pytest.approx(6.0, abs=0.01)


class TestSynapticCell:
    def test_forward_three_outputs(self):
        cell = SynapticCell(alpha=0.9, beta=0.8)
        current = torch.randn(4, 16)
        i_syn = torch.zeros(4, 16)
        v = torch.zeros(4, 16)
        spike, i_syn_next, v_next = cell(current, i_syn, v)
        assert spike.shape == (4, 16)
        assert i_syn_next.shape == (4, 16)


class TestALIFCell:
    def test_adaptive_threshold(self):
        cell = ALIFCell(beta=0.9, threshold=1.0, beta_adapt=1.8)
        v = torch.tensor([[2.0]])
        a = torch.tensor([[0.0]])
        current = torch.tensor([[0.0]])
        spike, _, a_next = cell(current, v, a)
        assert spike.item() == 1.0
        assert a_next.item() > 0  # adaptation increased


class TestExpIFCell:
    def test_exponential_upstroke(self):
        cell = ExpIFCell(beta=0.9, delta_t=0.5, v_rh=0.8, threshold=2.0)
        current = torch.randn(4, 8)
        v = torch.zeros(4, 8)
        spike, v_next = cell(current, v)
        assert spike.shape == (4, 8)


class TestAdExCell:
    def test_adaptation_current(self):
        cell = AdExCell(beta=0.9, a=0.01, b=0.1)
        current = torch.tensor([[3.0]])
        v = torch.zeros(1, 1)
        w = torch.zeros(1, 1)
        spike, v_next, w_next = cell(current, v, w)
        if spike.item() == 1.0:
            assert w_next.item() > 0  # b * spike added


class TestLapicqueCell:
    def test_decay_and_gain(self):
        cell = LapicqueCell(tau=20.0, r=1.0, dt=1.0, threshold=5.0)
        current = torch.tensor([[1.0]])
        v = torch.zeros(1, 1)
        _, v_next = cell(current, v)
        assert 0 < v_next.item() < 1.0  # small current, big threshold


class TestAlphaCell:
    def test_excitatory_inhibitory(self):
        cell = AlphaCell()
        exc = torch.tensor([[1.0]])
        inh = torch.tensor([[0.5]])
        i_exc = torch.zeros(1, 1)
        i_inh = torch.zeros(1, 1)
        v = torch.zeros(1, 1)
        spike, ie, ii, v_next = cell(exc, inh, i_exc, i_inh, v)
        assert ie.item() > ii.item()  # more excitation


class TestSecondOrderLIFCell:
    def test_inertial_dynamics(self):
        cell = SecondOrderLIFCell(alpha=0.95, beta=0.9)
        current = torch.randn(4, 8)
        a = torch.zeros(4, 8)
        v = torch.zeros(4, 8)
        spike, a_next, v_next = cell(current, a, v)
        assert a_next.shape == (4, 8)


class TestRecurrentLIFCell:
    def test_recurrent_connection(self):
        cell = RecurrentLIFCell(n_neurons=16, beta=0.9)
        current = torch.randn(4, 16)
        v = torch.zeros(4, 16)
        spike_prev = torch.zeros(4, 16)
        spike, v_next = cell(current, v, spike_prev)
        assert spike.shape == (4, 16)


# ── Networks ──────────────────────────────────────────────────────────


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


class TestConvSpikingNet:
    def test_forward(self):
        net = ConvSpikingNet(n_output=10)
        x = torch.randn(10, 4, 1, 28, 28)  # (T=10, batch=4, C=1, H=28, W=28)
        spike_counts, mem_acc = net(x)
        assert spike_counts.shape == (4, 10)

    def test_to_sc_weights(self):
        net = ConvSpikingNet(n_output=10)
        sc = net.to_sc_weights()
        assert len(sc) == 4  # conv1, conv2, fc1, fc2


# ── Encoding ──────────────────────────────────────────────────────────


class TestEncoding:
    def test_rate_encode_shape(self):
        x = torch.rand(8, 784)
        spikes = rate_encode(x, n_timesteps=25)
        assert spikes.shape == (25, 8, 784)

    def test_rate_encode_binary(self):
        x = torch.rand(4, 16)
        spikes = rate_encode(x, n_timesteps=10)
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

    def test_rate_encode_rate_proportional(self):
        """Higher input values should produce more spikes on average."""
        torch.manual_seed(42)
        low = torch.tensor([0.1])
        high = torch.tensor([0.9])
        low_spikes = rate_encode(low, n_timesteps=1000).sum()
        high_spikes = rate_encode(high, n_timesteps=1000).sum()
        assert high_spikes > low_spikes

    def test_latency_encode_shape(self):
        x = torch.rand(8, 16)
        spikes = latency_encode(x, n_timesteps=20)
        assert spikes.shape == (20, 8, 16)

    def test_latency_encode_one_spike(self):
        """Each input neuron should spike exactly once."""
        x = torch.rand(4, 8)
        spikes = latency_encode(x, n_timesteps=20)
        assert (spikes.sum(dim=0) == 1.0).all()

    def test_latency_strong_input_spikes_earlier(self):
        x = torch.tensor([0.9, 0.1])
        spikes = latency_encode(x, n_timesteps=20, tau=5.0)
        first_spike_0 = spikes[:, 0].argmax().item()
        first_spike_1 = spikes[:, 1].argmax().item()
        assert first_spike_0 < first_spike_1

    def test_delta_encode(self):
        x = torch.tensor([[0.0], [0.0], [1.0], [1.0], [0.0]])  # step up, then down
        spikes = delta_encode(x, threshold=0.5)
        assert spikes[2, 0].item() == 1.0  # step up
        assert spikes[4, 0].item() == 1.0  # step down


# ── Losses ────────────────────────────────────────────────────────────


class TestLosses:
    def test_spike_count_loss(self):
        counts = torch.randn(8, 10, requires_grad=True)
        targets = torch.randint(0, 10, (8,))
        loss = spike_count_loss(counts, targets)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_membrane_loss(self):
        mem = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        loss = membrane_loss(mem, targets)
        assert loss.item() > 0

    def test_spike_rate_loss(self):
        counts = torch.rand(8, 10) * 20
        targets = torch.randint(0, 10, (8,))
        loss = spike_rate_loss(counts, targets, n_timesteps=25)
        assert loss.item() >= 0

    def test_spike_l1_loss(self):
        counts = torch.rand(8, 10) * 10
        loss = spike_l1_loss(counts, n_timesteps=25)
        assert loss.item() >= 0

    def test_spike_l2_loss(self):
        counts = torch.rand(8, 10) * 10
        loss = spike_l2_loss(counts, n_timesteps=25)
        assert loss.item() >= 0


# ── Training loops ────────────────────────────────────────────────────


class TestTrainingLoops:
    @pytest.fixture
    def tiny_loader(self):
        from torch.utils.data import DataLoader, TensorDataset

        x = torch.rand(32, 16)
        y = torch.randint(0, 5, (32,))
        return DataLoader(TensorDataset(x, y), batch_size=8)

    @pytest.fixture
    def tiny_model(self):
        return SpikingNet(n_input=16, n_hidden=32, n_output=5, n_layers=1)

    def test_auto_device(self):
        dev = auto_device()
        assert isinstance(dev, torch.device)

    def test_auto_device_falls_back_when_cuda_probe_fails(self, monkeypatch):
        monkeypatch.setattr(training_loops.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(training_loops, "_device_usable", lambda device: device.type != "cuda")
        if hasattr(training_loops.torch.backends, "mps"):
            monkeypatch.setattr(
                training_loops.torch.backends.mps, "is_available", lambda: False, raising=False
            )

        assert training_loops.auto_device().type == "cpu"

    def test_auto_device_uses_cuda_when_probe_passes(self, monkeypatch):
        monkeypatch.setattr(training_loops.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(training_loops.torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(training_loops, "_cuda_device_supported", lambda index: True)
        monkeypatch.setattr(training_loops, "_device_usable", lambda device: True)

        assert training_loops.auto_device().type == "cuda"

    def test_train_epoch(self, tiny_model, tiny_loader):
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        loss, acc = train_epoch(tiny_model, tiny_loader, optimizer, n_timesteps=5)
        assert isinstance(loss, float)
        assert 0 <= acc <= 1

    def test_evaluate(self, tiny_model, tiny_loader):
        loss, acc = evaluate(tiny_model, tiny_loader, n_timesteps=5)
        assert isinstance(loss, float)
        assert 0 <= acc <= 1

    def test_train_reduces_loss(self, tiny_model, tiny_loader):
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)
        loss_0, _ = train_epoch(tiny_model, tiny_loader, optimizer, n_timesteps=5)
        for _ in range(5):
            train_epoch(tiny_model, tiny_loader, optimizer, n_timesteps=5)
        loss_5, _ = evaluate(tiny_model, tiny_loader, n_timesteps=5)
        assert loss_5 < loss_0

    def test_grad_clipping(self, tiny_model, tiny_loader):
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        loss, acc = train_epoch(
            tiny_model, tiny_loader, optimizer, n_timesteps=5, max_grad_norm=1.0
        )
        assert isinstance(loss, float)


# ── Utilities ─────────────────────────────────────────────────────────


class TestUtilities:
    def test_model_info(self):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5)
        info = model_info(net)
        assert info["total_params"] > 0
        assert info["spiking_cells"] > 0
        assert "LIFCell" in info["cell_types"]

    def test_population_decode(self):
        counts = torch.tensor([[0.0, 1.0, 5.0, 0.0]])  # peak at index 2
        decoded = population_decode(counts)
        # weights = [0, 1/6, 5/6, 0], preferred = [0,1,2,3]
        # decoded = 1/6*1 + 5/6*2 = 11/6 ≈ 1.833
        assert decoded.item() == pytest.approx(11 / 6, abs=0.01)

    def test_population_decode_with_preferred(self):
        counts = torch.tensor([[0.0, 0.0, 1.0]])
        preferred = torch.tensor([0.0, 45.0, 90.0])
        decoded = population_decode(counts, preferred)
        assert decoded.item() == pytest.approx(90.0)

    def test_spike_monitor(self):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5, n_layers=1)
        mon = SpikeMonitor(net)
        x = torch.randn(5, 2, 16)
        net(x)
        assert len(mon.layer_names) > 0
        for name in mon.layer_names:
            data = mon.get(name)
            assert data is not None
        mon.reset()
        for name in mon.layer_names:
            assert mon.get(name) is None
        mon.remove()

    def test_reset_states(self):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5)
        mon = SpikeMonitor(net)
        reset_states([mon])
        reset_states(None)  # should not raise


# ── DelayLinear ───────────────────────────────────────────────────────


class TestDelayLinear:
    def test_forward_shape(self):
        dl = DelayLinear(in_features=16, out_features=8, max_delay=4)
        x = torch.randn(4, 16)
        out = dl.step(x)
        assert out.shape == (4, 8)

    def test_unbatched(self):
        dl = DelayLinear(in_features=8, out_features=4, max_delay=3)
        x = torch.randn(8)
        out = dl.step(x)
        assert out.shape == (4,)

    def test_delays_int(self):
        dl = DelayLinear(in_features=8, out_features=4, max_delay=8)
        d = dl.delays_int
        assert d.shape == (4, 8)
        assert (d >= 0).all()
        assert (d <= 8).all()

    def test_nir_export(self):
        dl = DelayLinear(in_features=4, out_features=2, max_delay=3)
        arr = dl.to_nir_delay_array()
        assert arr.shape == (8,)  # 2 * 4

    def test_delay_gradient(self):
        dl = DelayLinear(in_features=4, out_features=2, max_delay=4, learn_delay=True)
        dl.reset()
        x = torch.randn(4, requires_grad=True)
        dl.step(x)
        out = dl.step(x)
        out.sum().backward()
        assert dl.delay.grad is not None

    def test_reset(self):
        dl = DelayLinear(in_features=4, out_features=2, max_delay=3)
        dl.step(torch.ones(4))
        dl.reset()
        assert dl._t == 0
        assert dl._history.abs().sum().item() == 0.0

    def test_multi_timestep_sequence(self):
        dl = DelayLinear(in_features=4, out_features=2, max_delay=3, learn_delay=False)
        dl.reset()
        outputs = []
        for t in range(10):
            x = torch.randn(4)
            outputs.append(dl.step(x))
        assert len(outputs) == 10


# ── End-to-end gradient flow ─────────────────────────────────────────


class TestEndToEnd:
    def test_full_pipeline_gradient(self):
        """Complete pipeline: encode → network → loss → backward."""
        x = torch.rand(8, 16)
        labels = torch.randint(0, 5, (8,))
        spikes = rate_encode(x, n_timesteps=10)

        net = SpikingNet(n_input=16, n_hidden=32, n_output=5)
        spike_counts, mem = net(spikes)
        loss = spike_count_loss(spike_counts, labels)
        loss.backward()

        for p in net.parameters():
            assert p.grad is not None

    def test_all_surrogates_train(self):
        """Verify all surrogate functions produce valid training gradients."""
        for fn in [
            fast_sigmoid,
            superspike,
            atan_surrogate,
            sigmoid_surrogate,
            straight_through,
            triangular,
        ]:
            net = SpikingNet(n_input=8, n_hidden=16, n_output=3, surrogate_fn=fn)
            x = torch.randn(5, 2, 8)
            spike_counts, _ = net(x)
            loss = spike_counts.sum()
            loss.backward()
            grads = [p.grad for p in net.parameters() if p.grad is not None]
            assert len(grads) > 0, f"No gradients with {fn.__name__}"

    def test_sc_export_after_training(self):
        """Train briefly, then export to SC weights."""
        from torch.utils.data import DataLoader, TensorDataset

        net = SpikingNet(n_input=8, n_hidden=16, n_output=3)
        loader = DataLoader(
            TensorDataset(torch.rand(16, 8), torch.randint(0, 3, (16,))), batch_size=8
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        train_epoch(net, loader, opt, n_timesteps=5)
        sc = net.to_sc_weights()
        for layer in sc:
            assert (layer["weight"] >= 0).all()
            assert (layer["weight"] <= 1).all()
