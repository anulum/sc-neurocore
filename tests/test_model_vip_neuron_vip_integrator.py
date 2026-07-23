# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVIPIntegrator from former test_model_vip_neuron.py

"""Focused suite: TestVIPIntegrator from former test_model_vip_neuron.py."""

from __future__ import annotations

from tests.model_vip_neuron_support import *  # noqa: F403

class TestVIPIntegrator:
    def test_default_integrator_is_rk4(self):
        assert VIPNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="Unsupported integrator"):
            VIPNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self):
        rk4 = VIPNeuron()
        euler = VIPNeuron(integrator="baseline_euler")
        assert _spikes(rk4, 1.0, 40000) > 0
        assert _spikes(euler, 1.0, 40000) > 0
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self):
        rk4 = VIPNeuron(dt=1e-5)
        euler = VIPNeuron(dt=1e-5, integrator="baseline_euler")
        for _ in range(200):
            rk4.step(1.0)
            euler.step(1.0)
        assert abs(rk4.v - euler.v) < 1e-2
