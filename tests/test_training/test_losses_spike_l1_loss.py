# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeL1Loss from former test_losses.py

"""Focused suite: TestSpikeL1Loss from former test_losses.py."""

from __future__ import annotations

from tests.test_training.losses_support import *  # noqa: F403

class TestSpikeL1Loss:
    def test_loss_is_nonnegative(self):
        spikes = torch.randn(8, 10).abs()

        loss = spike_l1_loss(spikes, n_timesteps=25)

        assert loss.item() >= 0

    def test_silent_spikes_have_zero_penalty(self):
        spikes = torch.zeros(8, 10)

        assert spike_l1_loss(spikes, n_timesteps=25).item() == 0.0
