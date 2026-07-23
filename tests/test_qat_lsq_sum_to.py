# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSumTo from former test_qat_lsq.py

"""Focused suite: TestSumTo from former test_qat_lsq.py."""

from __future__ import annotations

from tests.qat_lsq_support import *  # noqa: F403

class TestSumTo:
    def test_reduces_to_scalar(self) -> None:
        g = torch.ones(4, 5)
        out = _sum_to(g, ())
        assert out.shape == torch.Size([])
        assert out.item() == 20.0

    def test_reduces_to_channel_vector(self) -> None:
        g = torch.ones(3, 4)
        out = _sum_to(g, (3, 1))
        assert out.shape == torch.Size([3, 1])
        assert torch.allclose(out, torch.full((3, 1), 4.0))
