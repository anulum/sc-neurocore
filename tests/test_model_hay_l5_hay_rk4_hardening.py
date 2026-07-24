# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHayRK4Hardening from former test_model_hay_l5.py

"""Focused suite: TestHayRK4Hardening from former test_model_hay_l5.py."""

from __future__ import annotations

from tests.model_hay_l5_support import *  # noqa: F403


class TestHayRK4Hardening:
    def test_default_integrator_is_rk4(self) -> None:
        n = HayL5PyramidalNeuron()
        assert n.integrator == "rk4"

    def test_unknown_integrator_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported integrator"):
            HayL5PyramidalNeuron(integrator="bad")  # type: ignore[arg-type]

    def test_rk4_and_euler_regression_paths_diverge(self) -> None:
        rk4 = HayL5PyramidalNeuron()
        euler = HayL5PyramidalNeuron(integrator="baseline_euler")
        rk4_spikes = sum(rk4.step(10.0) for _ in range(20_000))
        euler_spikes = sum(euler.step(10.0) for _ in range(20_000))
        assert rk4_spikes == 1
        assert euler_spikes == 10
        assert abs(rk4.v_s - euler.v_s) < 2e-8

    def test_cross_backend_somatic_anchor(self) -> None:
        n = HayL5PyramidalNeuron()
        spikes = sum(n.step(10.0) for _ in range(20_000))
        assert spikes == 1
        assert n.ca_a >= 0.0

    def test_cross_backend_dual_input_anchor(self) -> None:
        n = HayL5PyramidalNeuron()
        spikes = sum(n.step(5.0, 5.0) for _ in range(20_000))
        assert spikes == 4
        assert n.ca_a >= 0.0

    def test_invalid_input_preserves_state(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(10):
            n.step(10.0)
        old_state = (n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a)
        with pytest.raises(ValueError, match="current_soma must be finite"):
            n.step(float("nan"))
        assert (n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a) == old_state

    def test_corrupt_state_preserves_state(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(10):
            n.step(10.0)
        old_state = (n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a)
        n.ca_a = float("nan")
        with pytest.raises(ValueError, match="ca_a must be finite"):
            n.step(10.0)
        assert (n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a) == old_state[:-1]

    def test_runtime_configuration_rejects_invalid_dt(self) -> None:
        n = HayL5PyramidalNeuron()
        n.dt = 0.0
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(10.0)

    @pytest.mark.parametrize("current", [0.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(3000):
            n.step(current)
        assert np.isfinite(n.v_s)
