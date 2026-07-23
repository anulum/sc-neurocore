# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCNASNetwork from former test_darts_sc_nas.py

"""Focused suite: TestSCNASNetwork from former test_darts_sc_nas.py."""

from __future__ import annotations

from darts_sc_nas_support import *  # noqa: F403

class TestSCNASNetwork(unittest.TestCase):
    def test_forward_shape(self) -> None:
        net = SCNASNetwork()
        net.eval()
        x = torch.rand(2, 1, 28, 28)
        out = net(x)
        self.assertEqual(out.shape, (2, 10))

    def test_hardware_penalty_returns_two_tensors(self) -> None:
        net = SCNASNetwork()
        luts, power = net.hardware_penalty()
        self.assertIsInstance(luts, torch.Tensor)
        self.assertIsInstance(power, torch.Tensor)

    def test_gradients_flow(self) -> None:
        net = SCNASNetwork()
        net.train()
        x = torch.rand(2, 1, 28, 28)
        out = net(x)
        target = torch.randint(0, 10, (2,))
        loss = torch.nn.functional.cross_entropy(out, target)
        _backward(loss)
        grad = net.layer1.alphas.grad
        self.assertIsNotNone(grad)
        assert grad is not None
        self.assertGreater(grad.norm().item(), 0)

    def test_hardware_penalty_differentiable(self) -> None:
        net = SCNASNetwork()
        luts, power = net.hardware_penalty()
        total = luts + power
        _backward(total)
        self.assertIsNotNone(net.layer1.alphas.grad)
