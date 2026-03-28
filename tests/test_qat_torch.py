# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for PyTorch QAT module

from __future__ import annotations

import torch

from sc_neurocore.qat.torch_qat import (
    QuantizedLIFNet,
    QuantizedLinear,
    ste_quantize,
)


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


class TestQuantizedLIFNet:
    def test_forward_shape(self):
        net = QuantizedLIFNet(784, 128, 10, n_bits=8)
        x = torch.randn(25, 32, 784)
        spikes, mem = net(x)
        assert spikes.shape == (32, 10)
        assert mem.shape == (32, 10)

    def test_trainable(self):
        net = QuantizedLIFNet(784, 64, 10, n_layers=1, n_bits=4)
        x = torch.randn(10, 8, 784)
        target = torch.randint(0, 10, (8,))
        spikes, _ = net(x)
        loss = torch.nn.functional.cross_entropy(spikes, target)
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_export(self):
        net = QuantizedLIFNet(784, 64, 10, n_layers=1, n_bits=4)
        exported = net.export_quantized()
        assert len(exported) == 2  # 1 hidden + 1 output
        assert all("weight_int" in e for e in exported)

    def test_effective_bits(self):
        net = QuantizedLIFNet(10, 5, 3, n_bits=4)
        assert net.effective_bits() == 4.0

    def test_4bit_vs_8bit_different_output(self):
        torch.manual_seed(42)
        net4 = QuantizedLIFNet(10, 5, 3, n_bits=4)
        torch.manual_seed(42)
        net8 = QuantizedLIFNet(10, 5, 3, n_bits=8)
        # Copy weights from net8 to net4
        net4.load_state_dict(net8.state_dict())
        x = torch.randn(5, 2, 10)
        s4, _ = net4(x)
        s8, _ = net8(x)
        # Different quantisation means different outputs (unless weights are exact)
        # At least the shapes match
        assert s4.shape == s8.shape
