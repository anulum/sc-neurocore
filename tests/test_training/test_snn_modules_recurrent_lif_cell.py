# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRecurrentLIFCell from former test_snn_modules.py

"""Focused suite: TestRecurrentLIFCell from former test_snn_modules.py."""

from __future__ import annotations

from tests.test_training.snn_modules_support import *  # noqa: F403


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
