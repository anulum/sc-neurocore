# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLapicqueCell from former test_torch_training.py

"""Focused suite: TestLapicqueCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403

class TestLapicqueCell:
    def test_decay_and_gain(self):
        cell = LapicqueCell(tau=20.0, r=1.0, dt=1.0, threshold=5.0)
        current = torch.tensor([[1.0]])
        v = torch.zeros(1, 1)
        _, v_next = cell(current, v)
        assert 0 < v_next.item() < 1.0  # small current, big threshold
