# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCableIsolation from former test_model_rall_cable.py

"""Focused suite: TestRallCableIsolation from former test_model_rall_cable.py."""

from __future__ import annotations

from tests.model_rall_cable_support import *  # noqa: F403

class TestRallCableIsolation:
    def test_construction_defaults(self) -> None:
        n = RallCableNeuron()
        assert n.n_comp == 5
        assert n.tau_m == 20.0
        assert n.v_rest == -65.0
        assert n.g_ratio == 0.5
        assert n.v.shape == (5,)
        np.testing.assert_allclose(n.v, -65.0)

    def test_step_returns_binary(self) -> None:
        assert RallCableNeuron().step(0.0) in (0, 1)

    def test_compartments_evolve(self) -> None:
        """All compartments should change from rest under current."""
        n = RallCableNeuron()
        for _ in range(1000):
            n.step(100.0)
        # Distal end (current injection) should depolarise most
        assert n.v[-1] > n.v_rest

    def test_state_finite_long_run(self) -> None:
        n = RallCableNeuron()
        for _ in range(50000):
            n.step(100.0)
        assert np.all(np.isfinite(n.v))

    def test_reset(self) -> None:
        n = RallCableNeuron()
        for _ in range(500):
            n.step(100.0)
        n.reset()
        np.testing.assert_allclose(n.v, n.v_rest)
