# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSynapticCell from former test_torch_training.py

"""Focused suite: TestSynapticCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestSynapticCell:
    def test_forward_three_outputs(self):
        cell = SynapticCell(alpha=0.9, beta=0.8)
        current = torch.randn(4, 16)
        i_syn = torch.zeros(4, 16)
        v = torch.zeros(4, 16)
        spike, i_syn_next, v_next = cell(current, i_syn, v)
        assert spike.shape == (4, 16)
        assert i_syn_next.shape == (4, 16)
