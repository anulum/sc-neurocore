# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTEQuantize from former test_qat_torch.py

"""Focused suite: TestSTEQuantize from former test_qat_torch.py."""

from __future__ import annotations

from tests.qat_torch_support import *  # noqa: F403

class TestSTEQuantize:
    def test_output_is_quantized(self):
        x = torch.randn(10)
        x_q = ste_quantize(x, n_bits=2)
        # 2-bit symmetric: values in {-1, 0, 1} * scale
        unique = x_q.unique()
        assert len(unique) <= 2**2

    def test_gradient_flows(self):
        x = torch.randn(10, requires_grad=True)
        x_q = ste_quantize(x, n_bits=4)
        loss = x_q.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad == 1.0).all()

    def test_8bit_range(self):
        x = torch.linspace(-2, 2, 100)
        x_q = ste_quantize(x, n_bits=8)
        assert x_q.min() >= x.min()
        assert x_q.max() <= x.max()

    def test_identity_for_zero(self):
        x = torch.zeros(5)
        x_q = ste_quantize(x, n_bits=8)
        assert (x_q == 0).all()
