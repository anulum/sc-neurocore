# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestALIFCell from former test_torch_training.py

"""Focused suite: TestALIFCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403

class TestALIFCell:
    def test_adaptive_threshold(self):
        cell = ALIFCell(beta=0.9, threshold=1.0, beta_adapt=1.8)
        v = torch.tensor([[2.0]])
        a = torch.tensor([[0.0]])
        current = torch.tensor([[0.0]])
        spike, _, a_next = cell(current, v, a)
        assert spike.item() == 1.0
        assert a_next.item() > 0  # adaptation increased
