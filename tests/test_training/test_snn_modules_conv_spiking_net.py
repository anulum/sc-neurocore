# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConvSpikingNet from former test_snn_modules.py

"""Focused suite: TestConvSpikingNet from former test_snn_modules.py."""

from __future__ import annotations

from tests.test_training.snn_modules_support import *  # noqa: F403


class TestConvSpikingNet:
    def test_forward_shape(self):
        net = ConvSpikingNet(n_output=10)
        x = torch.randn(10, 4, 1, 28, 28)
        spk, mem = net(x)
        assert spk.shape == (4, 10)
        assert mem.shape == (4, 10)

    def test_gradient_flows(self):
        net = ConvSpikingNet(n_output=5)
        x = torch.randn(5, 2, 1, 28, 28, requires_grad=True)
        spk, _ = net(x)
        spk.sum().backward()
        assert x.grad is not None

    def test_to_sc_weights(self):
        net = ConvSpikingNet(n_output=3)
        weights = net.to_sc_weights()
        assert len(weights) == 4  # conv1, conv2, fc1, fc2
        for entry in weights:
            assert "weight" in entry
            assert entry["weight"].min() >= 0.0
            assert entry["weight"].max() <= 1.0

    def test_to_sc_weights_bipolar_range(self):
        net = ConvSpikingNet(n_output=3)
        weights = net.to_sc_weights(encoding="bipolar", include_bias=False)
        assert len(weights) == 4
        for entry in weights:
            assert entry["encoding"] == "bipolar"
            assert entry["weight"].min() >= -1.0
            assert entry["weight"].max() <= 1.0
            assert bool((entry["weight"] < 0.0).any())

    def test_to_sc_weights_bipolar_scales_conv_and_linear_biases(self):
        net = ConvSpikingNet(n_output=3)
        with torch.no_grad():
            net.conv1.weight.fill_(2.0)
            net.conv1.bias.fill_(1.0)
            net.fc2.weight.fill_(4.0)
            net.fc2.bias.fill_(2.0)

        weights = net.to_sc_weights(encoding="bipolar")

        torch.testing.assert_close(weights[0]["bias"], torch.full_like(weights[0]["bias"], 0.5))
        torch.testing.assert_close(weights[3]["bias"], torch.full_like(weights[3]["bias"], 0.5))
        assert weights[0]["weight_scale"].item() == pytest.approx(2.0)
        assert weights[3]["weight_scale"].item() == pytest.approx(4.0)

    def test_to_sc_weights_noise_metadata(self):
        net = ConvSpikingNet(n_output=3)
        weights = net.to_sc_weights(
            noise_model=SCWeightNoiseModel(mode="binomial", bitstream_length=16, seed=5)
        )
        assert len(weights) == 4
        assert all(entry["noise_model"]["mode"] == "binomial" for entry in weights)

    def test_learnable_params(self):
        net = ConvSpikingNet(n_output=5, learn_beta=True, learn_threshold=True)
        x = torch.randn(3, 2, 1, 28, 28)
        spk, _ = net(x)
        spk.sum().backward()
        beta_params = [p for n, p in net.named_parameters() if "beta_logit" in n]
        assert len(beta_params) == 4  # 4 LIF layers

    def test_spike_counts_nonnegative(self):
        net = ConvSpikingNet(n_output=10)
        x = torch.randn(5, 4, 1, 28, 28)
        spk, _ = net(x)
        assert (spk >= 0).all()
