# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for training utilities

"""Tests for SpikeMonitor, population_decode, reset_states."""

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.training.snn_modules import SpikingNet
from sc_neurocore.training.utils import SpikeMonitor, population_decode, reset_states


class TestSpikeMonitor:
    """Tests for recording and hook lifecycle behaviour."""

    def test_records_spikes(self) -> None:
        """Forward hooks collect spike tensors for every spiking cell."""
        net = SpikingNet(n_input=10, n_hidden=16, n_output=3, n_layers=1)
        monitor = SpikeMonitor(net)
        x = torch.randn(5, 4, 10)
        net(x)
        assert len(monitor.layer_names) > 0
        for name in monitor.layer_names:
            rec = monitor.get(name)
            assert rec is not None
            assert rec.shape[0] == 5  # T timesteps
        monitor.remove()

    def test_reset_clears(self) -> None:
        """Reset clears recorded tensors but keeps layer names available."""
        net = SpikingNet(n_input=5, n_hidden=8, n_output=2, n_layers=1)
        monitor = SpikeMonitor(net)
        net(torch.randn(3, 2, 5))
        monitor.reset()
        for name in monitor.layer_names:
            assert monitor.get(name) is None
        monitor.remove()

    def test_remove_hooks(self) -> None:
        """Remove drops all registered hooks."""
        net = SpikingNet(n_input=5, n_hidden=8, n_output=2, n_layers=1)
        monitor = SpikeMonitor(net)
        monitor.remove()
        assert len(monitor._hooks) == 0


class TestPopulationDecode:
    """Tests for population-vector decoding contracts."""

    def test_argmax_equivalent(self) -> None:
        """With one-hot spike counts, should recover the index."""
        counts = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
        decoded = population_decode(counts)
        assert decoded.item() == pytest.approx(2.0)

    def test_weighted_average(self) -> None:
        """Uniform two-neuron activity decodes to the midpoint index."""
        counts = torch.tensor([[1.0, 1.0]])
        decoded = population_decode(counts)
        assert decoded.item() == pytest.approx(0.5)  # mean of 0 and 1

    def test_custom_preferred_values(self) -> None:
        """Custom one-dimensional preferred values override neuron indices."""
        counts = torch.tensor([[1.0, 0.0, 0.0]])
        preferred = torch.tensor([10.0, 20.0, 30.0])
        decoded = population_decode(counts, preferred)
        assert decoded.item() == pytest.approx(10.0)

    def test_batch(self) -> None:
        """Batch decoding returns one value per sample."""
        counts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        decoded = population_decode(counts)
        assert decoded.shape == (2,)
        assert decoded[0].item() == pytest.approx(0.0)
        assert decoded[1].item() == pytest.approx(1.0)

    def test_multidim_preferred(self) -> None:
        """Multi-dimensional preferred values decode to vector outputs."""
        counts = torch.tensor([[1.0, 1.0]])
        preferred = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
        decoded = population_decode(counts, preferred)
        assert decoded.shape == (1, 2)
        assert decoded[0, 0].item() == pytest.approx(1.0)
        assert decoded[0, 1].item() == pytest.approx(2.0)


class TestResetStates:
    """Tests for the reset_states convenience helper."""

    def test_clears_monitor_logs(self) -> None:
        """reset_states clears every supplied monitor log."""
        net = SpikingNet(n_input=5, n_hidden=8, n_output=2, n_layers=1)
        monitor = SpikeMonitor(net)
        net(torch.randn(3, 2, 5))
        assert any(len(v) > 0 for v in monitor._records.values())
        reset_states([monitor])
        assert all(len(v) == 0 for v in monitor._records.values())
        monitor.remove()

    def test_reset_states_none(self) -> None:
        """Passing None is a no-op."""
        reset_states(None)  # should not raise
