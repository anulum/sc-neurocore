# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilIntegrator from former test_model_pospischil.py

"""Focused suite: TestPospischilIntegrator from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403

class TestPospischilIntegrator:
    def test_default_integrator_is_rk4(self):
        assert PospischilNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="Unsupported integrator"):
            PospischilNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self):
        rk4 = PospischilNeuron()
        euler = PospischilNeuron(integrator="baseline_euler")
        rk4_spikes = sum(rk4.step(7.0) for _ in range(40000))
        euler_spikes = sum(euler.step(7.0) for _ in range(40000))
        assert rk4_spikes > 0 and euler_spikes > 0
        # The two integrators advance the same RHS but produce distinct
        # trajectories; their final membrane potentials differ.
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self):
        rk4 = PospischilNeuron(dt=1e-4)
        euler = PospischilNeuron(dt=1e-4, integrator="baseline_euler")
        for _ in range(200):
            rk4.step(5.0)
            euler.step(5.0)
        # As dt -> 0 the schemes converge; the membrane potentials stay close.
        assert abs(rk4.v - euler.v) < 1e-2
