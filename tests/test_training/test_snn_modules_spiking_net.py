# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikingNet from former test_snn_modules.py

"""Focused suite: TestSpikingNet from former test_snn_modules.py."""

from __future__ import annotations

from tests.test_training.snn_modules_support import *  # noqa: F403


class TestSpikingNet:
    def test_forward_shape(self):
        net = SpikingNet(n_input=10, n_hidden=32, n_output=3, n_layers=2)
        x = torch.randn(25, 8, 10)
        spk, mem = net(x)
        assert spk.shape == (8, 3)
        assert mem.shape == (8, 3)

    def test_gradient_flows_to_all_params(self):
        net = SpikingNet(n_input=10, n_hidden=16, n_output=5)
        x = torch.randn(10, 4, 10)
        spk, _ = net(x)
        spk.sum().backward()
        for name, p in net.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_spike_counts_nonnegative(self):
        net = SpikingNet(n_input=5, n_hidden=8, n_output=3)
        x = torch.randn(20, 4, 5)
        spk, _ = net(x)
        assert (spk >= 0).all()

    def test_to_sc_weights(self):
        net = SpikingNet(n_input=5, n_hidden=8, n_output=3, n_layers=1)
        weights = net.to_sc_weights()
        assert len(weights) == 2
        for entry in weights:
            assert "weight" in entry
            assert entry["weight"].min() >= 0.0
            assert entry["weight"].max() <= 1.0

    def test_to_sc_weights_bipolar_preserves_sign(self):
        net = SpikingNet(n_input=2, n_hidden=2, n_output=1, n_layers=0)
        with torch.no_grad():
            net.linears[0].weight.copy_(torch.tensor([[-2.0, 1.0]]))

        weights = net.to_sc_weights(encoding="bipolar", include_bias=False)

        assert weights[0]["encoding"] == "bipolar"
        torch.testing.assert_close(weights[0]["weight"], torch.tensor([[-1.0, 0.5]]))

    def test_to_sc_weights_bipolar_scales_bias_with_weight_normalisation(self):
        net = SpikingNet(n_input=2, n_hidden=2, n_output=1, n_layers=0)
        with torch.no_grad():
            net.linears[0].weight.copy_(torch.tensor([[-4.0, 2.0]]))
            net.linears[0].bias.copy_(torch.tensor([2.0]))

        weights = net.to_sc_weights(encoding="bipolar")

        torch.testing.assert_close(weights[0]["weight"], torch.tensor([[-1.0, 0.5]]))
        torch.testing.assert_close(weights[0]["bias"], torch.tensor([0.5]))
        assert weights[0]["weight_scale"].item() == pytest.approx(4.0)

    def test_to_sc_weights_binomial_noise_is_deterministic(self):
        net = SpikingNet(n_input=5, n_hidden=8, n_output=3, n_layers=1)
        model = SCWeightNoiseModel(mode="binomial", bitstream_length=32, seed=17)
        first = net.to_sc_weights(noise_model=model)
        second = net.to_sc_weights(noise_model=model)
        assert first[0]["noise_model"]["mode"] == "binomial"
        assert first[0]["noise_model"]["bitstream_length"] == 32
        assert torch.equal(first[0]["weight"], second[0]["weight"])
        assert first[0]["weight"].min() >= 0.0
        assert first[0]["weight"].max() <= 1.0
