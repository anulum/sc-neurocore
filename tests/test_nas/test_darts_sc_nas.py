# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NAS Tests

from __future__ import annotations

from typing import Protocol
import unittest

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; DARTS tests require it")

from sc_neurocore.nas.darts_sc_nas import (  # noqa: E402
    BitstreamCandidate,
    SCMixedOp,
    SCNASNetwork,
)


class _Backwardable(Protocol):
    def backward(self) -> None: ...


def _backward(value: _Backwardable) -> None:
    value.backward()


class TestBitstreamCandidate(unittest.TestCase):
    def test_eval_passthrough(self) -> None:
        op = BitstreamCandidate(256, 100.0, 1.0)
        op.eval()
        x = torch.tensor([0.3, 0.5, 0.7])
        out = op(x)
        self.assertTrue(torch.equal(x, out))

    def test_train_adds_noise(self) -> None:
        torch.manual_seed(42)
        op = BitstreamCandidate(64, 100.0, 1.0)
        op.train()
        x = torch.full((100,), 0.5)
        out = op(x)
        self.assertFalse(torch.equal(x, out))

    def test_output_clamped_to_unit(self) -> None:
        torch.manual_seed(0)
        op = BitstreamCandidate(64, 100.0, 1.0)
        op.train()
        x = torch.full((1000,), 0.5)
        out = op(x)
        self.assertTrue((out >= 0.0).all())
        self.assertTrue((out <= 1.0).all())

    def test_longer_bitstream_less_noise(self) -> None:
        torch.manual_seed(42)
        op_short = BitstreamCandidate(64, 100.0, 1.0)
        op_short.train()
        x = torch.full((1000,), 0.5)
        out_short = op_short(x)
        var_short = out_short.var().item()

        torch.manual_seed(42)
        op_long = BitstreamCandidate(4096, 100.0, 1.0)
        op_long.train()
        out_long = op_long(x)
        var_long = out_long.var().item()

        self.assertGreater(var_short, var_long)

    def test_cost_attributes(self) -> None:
        op = BitstreamCandidate(256, 42.0, 3.14)
        self.assertEqual(op.length, 256)
        self.assertAlmostEqual(op.lut_cost, 42.0)
        self.assertAlmostEqual(op.power_cost, 3.14)


class TestSCMixedOp(unittest.TestCase):
    def test_forward_shape(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        op.eval()
        x = torch.rand(2, 1, 8, 8)
        out = op(x)
        self.assertEqual(out.shape, (2, 16, 8, 8))

    def test_alphas_are_learnable(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        self.assertEqual(op.alphas.shape[0], 7)
        self.assertTrue(op.alphas.requires_grad)

    def test_expected_resource_cost_positive(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        luts, power = op.expected_resource_cost()
        self.assertGreater(luts.item(), 0)
        self.assertGreater(power.item(), 0)

    def test_extract_optimal_returns_valid_length(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        config = op.extract_optimal_config()
        self.assertIn(config, [64, 128, 256, 512, 1024, 2048, 4096])

    def test_seven_candidate_ops(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        self.assertEqual(len(op.ops), 7)
        self.assertEqual(op.num_ops, 7)

    def test_strided_conv_halves_spatial(self) -> None:
        op = SCMixedOp(16, 32, 3, 2, 1)
        op.eval()
        x = torch.rand(1, 16, 16, 16)
        out = op(x)
        self.assertEqual(out.shape, (1, 32, 8, 8))


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


if __name__ == "__main__":
    unittest.main()
