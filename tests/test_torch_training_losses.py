# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLosses from former test_torch_training.py

"""Focused suite: TestLosses from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


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
