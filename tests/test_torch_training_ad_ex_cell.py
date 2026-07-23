# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExCell from former test_torch_training.py

"""Focused suite: TestAdExCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403

class TestAdExCell:
    def test_adaptation_current(self):
        cell = AdExCell(beta=0.9, a=0.01, b=0.1)
        current = torch.tensor([[3.0]])
        v = torch.zeros(1, 1)
        w = torch.zeros(1, 1)
        spike, v_next, w_next = cell(current, v, w)
        if spike.item() == 1.0:
            assert w_next.item() > 0  # b * spike added
