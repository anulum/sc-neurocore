# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResetStates from former test_utils.py

"""Focused suite: TestResetStates from former test_utils.py."""

from __future__ import annotations

from tests.test_training.utils_support import *  # noqa: F403

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
