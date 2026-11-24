# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTEQuantize from former test_qat_torch.py

"""Focused suite: TestSTEQuantize from former test_qat_torch.py."""

from __future__ import annotations

from typing import Any, cast
from tests.qat_torch_support import (
    ste_quantize,
    torch,
)


class TestSTEQuantize:
    """Torch straight-through quantisation and its gradient."""

    def test_output_is_quantized(self) -> None:
        """Output is quantized."""
        x = torch.randn(10)
        x_q = ste_quantize(x, n_bits=2)
        # 2-bit symmetric: values in {-1, 0, 1} * scale
        unique = cast(Any, x_q).unique()
        assert len(unique) <= 2**2

    def test_gradient_flows(self) -> None:
        """The straight-through gradient is one everywhere."""
        x = torch.randn(10, requires_grad=True)
        x_q = ste_quantize(x, n_bits=4)
        loss = x_q.sum()
        cast(Any, loss).backward()
        assert x.grad is not None
        assert (x.grad == 1.0).all()

    def test_8bit_range(self) -> None:
        """Quantising to 8 bits stays inside the input range."""
        x = torch.linspace(-2, 2, 100)
        x_q = ste_quantize(x, n_bits=8)
        assert x_q.min() >= x.min()
        assert x_q.max() <= x.max()

    def test_identity_for_zero(self) -> None:
        """A zero input quantises to zero."""
        x = torch.zeros(5)
        x_q = ste_quantize(x, n_bits=8)
        assert (x_q == 0).all()
