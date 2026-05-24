# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for spike-training loss functions

"""Behavioural contracts for spike-training regularisation losses."""

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.training.losses import spike_l1_loss, spike_l2_loss


class TestSpikeL1Loss:
    def test_loss_is_nonnegative(self):
        spikes = torch.randn(8, 10).abs()

        loss = spike_l1_loss(spikes, n_timesteps=25)

        assert loss.item() >= 0

    def test_silent_spikes_have_zero_penalty(self):
        spikes = torch.zeros(8, 10)

        assert spike_l1_loss(spikes, n_timesteps=25).item() == 0.0


class TestSpikeL2Loss:
    def test_loss_is_nonnegative(self):
        spikes = torch.randn(8, 10).abs()

        loss = spike_l2_loss(spikes, n_timesteps=25)

        assert loss.item() >= 0

    def test_silent_spikes_have_zero_penalty(self):
        spikes = torch.zeros(8, 10)

        assert spike_l2_loss(spikes, n_timesteps=25).item() == 0.0

    def test_penalty_increases_with_activity_magnitude(self):
        active = torch.ones(8, 10) * 20.0
        sparse = torch.ones(8, 10) * 2.0

        assert spike_l2_loss(active, 25) > spike_l2_loss(sparse, 25)
