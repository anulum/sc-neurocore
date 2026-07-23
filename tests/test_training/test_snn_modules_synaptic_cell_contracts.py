# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSynapticCellContracts from former test_snn_modules.py

"""Focused suite: TestSynapticCellContracts from former test_snn_modules.py."""

from __future__ import annotations

from tests.test_training.snn_modules_support import *  # noqa: F403

class TestSynapticCellContracts:
    def test_dual_exponential_state_update(self):
        cell = SynapticCell(alpha=0.5, beta=0.5, threshold=100.0)

        spike, synaptic_current, voltage = cell(
            torch.ones(4),
            torch.zeros(4),
            torch.zeros(4),
        )

        assert spike.sum().item() == 0.0
        assert synaptic_current.mean().item() == pytest.approx(1.0, abs=0.01)
        assert voltage.mean().item() == pytest.approx(1.0, abs=0.01)

    def test_spike_path_crosses_threshold_for_all_units(self):
        cell = SynapticCell(alpha=0.0, beta=0.0, threshold=0.5)

        spike, _, _ = cell(torch.ones(4), torch.zeros(4), torch.zeros(4))

        assert spike.sum().item() == 4.0

    def test_learnable_beta_is_bounded_parameter(self):
        cell = SynapticCell(beta=0.8, learn_beta=True)

        assert any("beta_logit" in name for name, _ in cell.named_parameters())
        assert 0 < cell.beta.item() < 1

    def test_surrogate_allows_synaptic_input_gradient_flow(self):
        cell = SynapticCell()
        current = torch.randn(8, requires_grad=True)

        spike, _, _ = cell(current, torch.zeros(8), torch.zeros(8))
        spike.sum().backward()

        assert current.grad is not None

    def test_to_sc_weights_gaussian_noise_uses_sigma(self):
        net = SpikingNet(n_input=5, n_hidden=8, n_output=3, n_layers=1)
        base = net.to_sc_weights(noise_model="none")
        noisy = net.to_sc_weights(noise_model={"mode": "gaussian", "sigma": 0.05, "seed": 11})
        assert noisy[0]["noise_model"]["sigma"] == 0.05
        assert not torch.equal(base[0]["weight"], noisy[0]["weight"])
        assert noisy[0]["weight"].min() >= 0.0
        assert noisy[0]["weight"].max() <= 1.0

    def test_to_sc_weights_rejects_invalid_noise_model(self):
        net = SpikingNet(n_input=5, n_hidden=8, n_output=3, n_layers=1)
        with pytest.raises(ValueError, match="unsupported"):
            net.to_sc_weights(noise_model="shot")

    def test_single_layer(self):
        net = SpikingNet(n_input=5, n_hidden=8, n_output=3, n_layers=0)
        x = torch.randn(10, 2, 5)
        spk, mem = net(x)
        assert spk.shape == (2, 3)

    def test_learnable_params_gradient(self):
        net = SpikingNet(
            n_input=10,
            n_hidden=16,
            n_output=5,
            learn_beta=True,
            learn_threshold=True,
        )
        x = torch.randn(5, 2, 10)
        spk, _ = net(x)
        spk.sum().backward()
        beta_params = [p for n, p in net.named_parameters() if "beta_logit" in n]
        thresh_params = [p for n, p in net.named_parameters() if "threshold_log" in n]
        assert len(beta_params) > 0
        assert len(thresh_params) > 0
        for p in beta_params + thresh_params:
            assert p.grad is not None
