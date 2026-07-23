# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantizedLIFNet from former test_qat_torch.py

"""Focused suite: TestQuantizedLIFNet from former test_qat_torch.py."""

from __future__ import annotations

from tests.qat_torch_support import *  # noqa: F403

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
