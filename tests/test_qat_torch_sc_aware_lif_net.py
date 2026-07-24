# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCAwareLIFNet from former test_qat_torch.py

"""Focused suite: TestSCAwareLIFNet from former test_qat_torch.py."""

from __future__ import annotations

from tests.qat_torch_support import *  # noqa: F403


class TestSCAwareLIFNet:
    def test_forward_shape(self):
        net = SCAwareLIFNet(784, 128, 10, bitstream_length=256)
        x = torch.randn(25, 32, 784)
        spikes, mem = net(x)
        assert spikes.shape == (32, 10)

    def test_trainable(self):
        net = SCAwareLIFNet(784, 64, 10, n_layers=1, bitstream_length=128)
        x = torch.randn(10, 8, 784)
        target = torch.randint(0, 10, (8,))
        spikes, _ = net(x)
        loss = torch.nn.functional.cross_entropy(spikes, target)
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_export_bipolar(self):
        net = SCAwareLIFNet(10, 5, 3, n_layers=1, bitstream_length=256)
        exported = net.export_bipolar_weights()
        assert len(exported) == 2
        for e in exported:
            assert e["weight"].min() >= -1.0
            assert e["weight"].max() <= 1.0
