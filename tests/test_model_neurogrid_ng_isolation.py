# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNGIsolation from former test_model_neurogrid.py

"""Focused suite: TestNGIsolation from former test_model_neurogrid.py."""

from __future__ import annotations

from tests.model_neurogrid_support import *  # noqa: F403


class TestNGIsolation:
    def test_defaults(self) -> None:
        n = NeuroGridNeuron()
        assert n.v_s == -65.0 and n.v_d == -65.0
        assert n.tau_s == 20.0 and n.tau_d == 50.0
        assert n.g_c == 0.5 and n.delta_t == 2.0
        assert n.v_threshold == -50.0 and n.v_peak == 20.0
        assert n.dt == 0.1

    def test_two_compartments(self) -> None:
        n = NeuroGridNeuron()
        assert hasattr(n, "v_s") and hasattr(n, "v_d")

    def test_step_returns_binary(self) -> None:
        assert NeuroGridNeuron().step(0.0) in (0, 1)

    def test_both_compartments_evolve(self) -> None:
        n = NeuroGridNeuron()
        vs0, vd0 = n.v_s, n.v_d
        for _ in range(500):
            n.step(50.0)
        assert n.v_s != vs0 or n.v_d != vd0

    def test_state_finite_long_run(self) -> None:
        n = NeuroGridNeuron()
        for _ in range(100_000):
            n.step(100.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)

    def test_reset_restores_defaults(self) -> None:
        n = NeuroGridNeuron()
        for _ in range(5000):
            n.step(100.0)
        n.reset()
        assert n.v_s == -65.0 and n.v_d == -65.0

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = NeuroGridNeuron()
            trace = [(n.step(100.0), n.v_s, n.v_d) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
