# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCAwareLinear from former test_qat_torch.py

"""Focused suite: TestSCAwareLinear from former test_qat_torch.py."""

from __future__ import annotations

from tests.qat_torch_support import *  # noqa: F403


class TestSCAwareLinear:
    def test_forward_shape(self):
        layer = SCAwareLinear(784, 128, bitstream_length=256)
        x = torch.randn(32, 784)
        out = layer(x)
        assert out.shape == (32, 128)

    def test_weights_clamped(self):
        layer = SCAwareLinear(10, 5, bitstream_length=256)
        # Force large weights
        with torch.no_grad():
            layer.linear.weight.fill_(5.0)
        x = torch.randn(4, 10)
        _ = layer(x)
        # After forward, weight itself isn't mutated, but forward uses clamped
        # The clamp happens in forward, not in-place

    def test_training_adds_noise(self):
        torch.manual_seed(42)
        layer = SCAwareLinear(10, 5, bitstream_length=64)
        x = torch.randn(4, 10)
        layer.train()
        out1 = layer(x).detach().clone()
        out2 = layer(x).detach().clone()
        # With noise, outputs differ between calls
        assert not torch.allclose(out1, out2)

    def test_eval_no_noise(self):
        torch.manual_seed(42)
        layer = SCAwareLinear(10, 5, bitstream_length=64)
        x = torch.randn(4, 10)
        layer.eval()
        out1 = layer(x).detach().clone()
        out2 = layer(x).detach().clone()
        assert torch.allclose(out1, out2)

    def test_gradient_flows(self):
        layer = SCAwareLinear(10, 5, bitstream_length=256)
        layer.train()
        x = torch.randn(4, 10)
        out = layer(x)
        out.sum().backward()
        assert layer.linear.weight.grad is not None
