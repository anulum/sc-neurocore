# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantisation from former test_extract_shd_weights.py

"""Focused suite: TestQuantisation from former test_extract_shd_weights.py."""

from __future__ import annotations

from extract_shd_weights_support import *  # noqa: F403

class TestQuantisation:
    def test_zero_tensor(self) -> None:
        w = torch.zeros(10, 5)
        w_q, scale = quantise_per_tensor_symmetric(w)
        assert (w_q == 0).all()
        assert scale == 0.0

    def test_symmetric_range(self) -> None:
        w = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        w_q, scale = quantise_per_tensor_symmetric(w)
        # max abs = 1.0 → scale = 1/127
        assert scale == pytest.approx(1.0 / 127, rel=1e-9)
        # ±1.0 should map to ±127
        assert int(w_q[0, 0]) == -127
        assert int(w_q[0, 4]) == 127
        assert int(w_q[0, 2]) == 0

    def test_int8_dtype(self) -> None:
        w = torch.randn(10, 10) * 0.5
        w_q, _ = quantise_per_tensor_symmetric(w)
        assert w_q.dtype == torch.int8
        assert int(w_q.min()) >= -128
        assert int(w_q.max()) <= 127

    def test_dequant_max_error_bounded(self) -> None:
        w = torch.randn(50, 50) * 2.5
        w_q, scale = quantise_per_tensor_symmetric(w)
        w_dequant = w_q.float() * scale
        max_err = float((w - w_dequant).abs().max().item())
        # Per-tensor symmetric int8 has max error ≈ scale/2 = abs_max/254
        assert max_err <= scale / 2 + 1e-6

    def test_clipping_outliers(self) -> None:
        # Single huge outlier shouldn't make all other weights zero
        w = torch.tensor([[100.0, 0.5, -0.5, 0.0, 1.0]])
        w_q, scale = quantise_per_tensor_symmetric(w)
        # 100 / (100/127) = 127
        assert int(w_q[0, 0]) == 127
        # 0.5 / (100/127) = 0.635 → rounds to 1
        assert int(w_q[0, 1]) == 1
