# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSecondOrderLIFCell from former test_torch_training.py

"""Focused suite: TestSecondOrderLIFCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestSecondOrderLIFCell:
    def test_inertial_dynamics(self):
        cell = SecondOrderLIFCell(alpha=0.95, beta=0.9)
        current = torch.randn(4, 8)
        a = torch.zeros(4, 8)
        v = torch.zeros(4, 8)
        spike, a_next, v_next = cell(current, a, v)
        assert a_next.shape == (4, 8)
