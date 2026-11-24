# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLosses from former test_torch_training.py

"""Focused suite: TestLosses from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import (
    membrane_loss,
    spike_count_loss,
    spike_l1_loss,
    spike_l2_loss,
    spike_rate_loss,
    torch,
)


class TestLosses:
    """The spike-count, membrane, rate and regularisation losses."""

    def test_spike_count_loss(self) -> None:
        """The spike-count loss is positive and differentiable."""
        counts = torch.randn(8, 10, requires_grad=True)
        targets = torch.randint(0, 10, (8,))
        loss = spike_count_loss(counts, targets)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_membrane_loss(self) -> None:
        """The membrane loss is positive."""
        mem = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        loss = membrane_loss(mem, targets)
        assert loss.item() > 0

    def test_spike_rate_loss(self) -> None:
        """The spike-rate loss is non-negative."""
        counts = torch.rand(8, 10) * 20
        targets = torch.randint(0, 10, (8,))
        loss = spike_rate_loss(counts, targets, n_timesteps=25)
        assert loss.item() >= 0

    def test_spike_l1_loss(self) -> None:
        """The L1 spike regulariser is non-negative."""
        counts = torch.rand(8, 10) * 10
        loss = spike_l1_loss(counts, n_timesteps=25)
        assert loss.item() >= 0

    def test_spike_l2_loss(self) -> None:
        """The L2 spike regulariser is non-negative."""
        counts = torch.rand(8, 10) * 10
        loss = spike_l2_loss(counts, n_timesteps=25)
        assert loss.item() >= 0
