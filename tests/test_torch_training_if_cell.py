# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIFCell from former test_torch_training.py

"""Focused suite: TestIFCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestIFCell:
    def test_no_leak(self):
        cell = IFCell(threshold=10.0)
        v = torch.tensor([[5.0]])
        current = torch.tensor([[1.0]])
        _, v_next = cell(current, v)
        assert v_next.item() == pytest.approx(6.0, abs=0.01)
