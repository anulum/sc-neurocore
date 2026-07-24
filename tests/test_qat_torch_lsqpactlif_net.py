# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLSQPACTLIFNet from former test_qat_torch.py

"""Focused suite: TestLSQPACTLIFNet from former test_qat_torch.py."""

from __future__ import annotations

from tests.qat_torch_support import *  # noqa: F403


class TestLSQPACTLIFNet:
    def test_forward_shape(self):
        net = LSQPACTLIFNet(784, 128, 10, weight_bits=4, act_bits=4)
        x = torch.randn(25, 32, 784)
        spikes, mem = net(x)
        assert spikes.shape == (32, 10)
        assert mem.shape == (32, 10)

    def test_per_channel_weight_steps(self):
        net = LSQPACTLIFNet(20, 16, 5, n_layers=1, weight_bits=4, per_channel=True)
        net(torch.randn(5, 4, 20))  # initialise LSQ steps
        # First hidden layer has 16 output neurons -> 16 per-channel steps.
        assert net.linears[0].weight_quant.step.shape == torch.Size([16])

    def test_trainable_end_to_end(self):
        net = LSQPACTLIFNet(20, 16, 3, n_layers=1, weight_bits=4, act_bits=4)
        x = torch.randn(6, 8, 20)
        target = torch.randint(0, 3, (8,))
        spikes, _ = net(x)
        loss = torch.nn.functional.cross_entropy(spikes, target)
        loss.backward()
        # Both the LSQ steps and the PACT alpha must receive gradients.
        assert net.linears[0].weight_quant.step.grad is not None
        assert net.input_quant.alpha.grad is not None

    def test_export_quantized(self):
        net = LSQPACTLIFNet(20, 16, 3, n_layers=1, weight_bits=4, act_bits=4)
        net(torch.randn(5, 4, 20))
        export = net.export_quantized()
        assert len(export["layers"]) == 2
        assert export["act_bits"] == 4
        assert "input_scale" in export
        assert export["layers"][0]["weight_int"].abs().max() <= 8
