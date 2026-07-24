# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFCell from former test_torch_training.py

"""Focused suite: TestExpIFCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestExpIFCell:
    def test_exponential_upstroke(self):
        cell = ExpIFCell(beta=0.9, delta_t=0.5, v_rh=0.8, threshold=2.0)
        current = torch.randn(4, 8)
        v = torch.zeros(4, 8)
        spike, v_next = cell(current, v)
        assert spike.shape == (4, 8)
