# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFakeQuantize from former test_qat_observers.py

"""Focused suite: TestFakeQuantize from former test_qat_observers.py."""

from __future__ import annotations

from tests.qat_observers_support import *  # noqa: F403


class TestFakeQuantize:
    def test_broadcasts_scalar(self) -> None:
        x = torch.tensor([0.0, 1.0, 2.0])
        out = fake_quantize(x, torch.tensor(1.0), torch.tensor(0.0), n_bits=8, unsigned=False)
        assert torch.allclose(out, x)

    def test_clamps_to_grid(self) -> None:
        x = torch.tensor([1000.0])
        out = fake_quantize(x, torch.tensor(1.0), torch.tensor(0.0), n_bits=4, unsigned=False)
        # 4-bit signed max code = 7.
        assert out.item() == 7.0
