# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRecurrentLIFCell from former test_torch_training.py

"""Focused suite: TestRecurrentLIFCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestRecurrentLIFCell:
    def test_recurrent_connection(self):
        cell = RecurrentLIFCell(n_neurons=16, beta=0.9)
        current = torch.randn(4, 16)
        v = torch.zeros(4, 16)
        spike_prev = torch.zeros(4, 16)
        spike, v_next = cell(current, v, spike_prev)
        assert spike.shape == (4, 16)
