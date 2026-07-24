# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeSchutterRK4Hardening from former test_model_de_schutter_purkinje.py

"""Focused suite: TestDeSchutterRK4Hardening from former test_model_de_schutter_purkinje.py."""

from __future__ import annotations

from tests.model_de_schutter_purkinje_support import *  # noqa: F403


class TestDeSchutterRK4Hardening:
    def test_default_integrator_is_rk4(self) -> None:
        assert DeSchutterPurkinjeNeuron().integrator == "rk4"

    def test_unknown_integrator_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported integrator"):
            DeSchutterPurkinjeNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_rk4_and_baseline_euler_paths_diverge(self) -> None:
        rk4 = DeSchutterPurkinjeNeuron()
        euler = DeSchutterPurkinjeNeuron(integrator="baseline_euler")
        for _ in range(2_000):
            rk4.step(200.0)
            euler.step(200.0)
        assert abs(rk4.v - euler.v) > 1.0e-8

    def test_cross_backend_spike_anchor(self) -> None:
        n = DeSchutterPurkinjeNeuron()
        spikes = sum(n.step(500.0) for _ in range(20_000))
        assert spikes == 1

    def test_non_finite_current_rejected_without_mutation(self) -> None:
        n = DeSchutterPurkinjeNeuron()
        for _ in range(10):
            n.step(200.0)
        old = (n.v, n.h_na, n.n_k, n.m_cap, n.h_cap, n.q_kca, n.ca)
        with pytest.raises(ValueError, match="current"):
            n.step(float("nan"))
        assert (n.v, n.h_na, n.n_k, n.m_cap, n.h_cap, n.q_kca, n.ca) == old

    def test_non_finite_runtime_state_rejected_before_mutation(self) -> None:
        n = DeSchutterPurkinjeNeuron()
        n.ca = float("nan")
        with pytest.raises(ValueError, match="ca"):
            n.step(200.0)
        assert np.isnan(n.ca)

    def test_calcium_stays_non_negative_under_rk4(self) -> None:
        n = DeSchutterPurkinjeNeuron()
        for _ in range(20_000):
            n.step(500.0)
            assert n.ca >= 0.0
