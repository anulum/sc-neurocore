# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGolombFSIntegrator from former test_model_golomb_fs.py

"""Focused suite: TestGolombFSIntegrator from former test_model_golomb_fs.py."""

from __future__ import annotations

from tests.model_golomb_fs_support import *  # noqa: F403

class TestGolombFSIntegrator:
    def test_default_integrator_is_rk4(self):
        assert GolombFSNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="Unsupported integrator"):
            GolombFSNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self):
        rk4 = GolombFSNeuron()
        euler = GolombFSNeuron(integrator="baseline_euler")
        rk4_spikes = sum(rk4.step(5.0) for _ in range(40000))
        euler_spikes = sum(euler.step(5.0) for _ in range(40000))
        assert rk4_spikes > 0 and euler_spikes > 0
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self):
        rk4 = GolombFSNeuron(dt=1e-5)
        euler = GolombFSNeuron(dt=1e-5, integrator="baseline_euler")
        for _ in range(200):
            rk4.step(5.0)
            euler.step(5.0)
        assert abs(rk4.v - euler.v) < 1e-2
