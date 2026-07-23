# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLSQLinear from former test_qat_lsq.py

"""Focused suite: TestLSQLinear from former test_qat_lsq.py."""

from __future__ import annotations

from tests.qat_lsq_support import *  # noqa: F403

class TestLSQLinear:
    def test_forward_shape(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4)
        out = layer(torch.randn(3, 8))
        assert out.shape == (3, 5)

    def test_default_is_per_channel(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4)
        assert layer.weight_quant.per_channel
        assert layer.weight_quant.step.shape == torch.Size([5])

    def test_scalar_step_when_not_per_channel(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4, per_channel=False)
        assert layer.weight_quant.step.shape == torch.Size([])

    def test_gradients_reach_weights_and_step(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4)
        out = layer(torch.randn(3, 8))
        out.sum().backward()
        assert layer.linear.weight.grad is not None
        assert layer.weight_quant.step.grad is not None

    def test_export_quantized_shapes(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4, per_channel=True)
        layer(torch.randn(3, 8))  # initialise step
        export = layer.export_quantized()
        assert export["weight_int"].shape == (5, 8)
        assert export["weight_int"].dtype == torch.int32
        assert export["step"].shape == torch.Size([5])
        assert export["n_bits"] == 4
        assert export["per_channel"] is True
        assert export["weight_int"].abs().max() <= 8

    def test_export_without_bias(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4, bias=False)
        layer(torch.randn(3, 8))
        export = layer.export_quantized()
        assert "bias" not in export

    def test_export_with_bias(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4, bias=True)
        layer(torch.randn(3, 8))
        export = layer.export_quantized()
        assert "bias" in export
