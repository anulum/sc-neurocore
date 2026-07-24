# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantizedLinear from former test_qat_torch.py

"""Focused suite: TestQuantizedLinear from former test_qat_torch.py."""

from __future__ import annotations

from tests.qat_torch_support import *  # noqa: F403


class TestQuantizedLinear:
    def test_forward_shape(self):
        layer = QuantizedLinear(784, 128, n_bits=4)
        x = torch.randn(32, 784)
        out = layer(x)
        assert out.shape == (32, 128)

    def test_gradient_flows_through_qat(self):
        layer = QuantizedLinear(10, 5, n_bits=4)
        x = torch.randn(4, 10)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.linear.weight.grad is not None

    def test_export_quantized(self):
        layer = QuantizedLinear(10, 5, n_bits=8)
        result = layer.export_quantized()
        assert "weight_int" in result
        assert result["weight_int"].dtype == torch.int8
        assert result["n_bits"] == 8
        assert "scale" in result

    def test_different_bits(self):
        for bits in [2, 4, 8, 16]:
            layer = QuantizedLinear(10, 5, n_bits=bits)
            x = torch.randn(4, 10)
            out = layer(x)
            assert out.shape == (4, 5)
