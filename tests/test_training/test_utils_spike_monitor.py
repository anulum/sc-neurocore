# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeMonitor from former test_utils.py

"""Focused suite: TestSpikeMonitor from former test_utils.py."""

from __future__ import annotations

from tests.test_training.utils_support import *  # noqa: F403


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
