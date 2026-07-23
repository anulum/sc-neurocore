# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTMIsolation from former test_model_tsodyks_markram.py

"""Focused suite: TestTMIsolation from former test_model_tsodyks_markram.py."""

from __future__ import annotations

from tests.model_tsodyks_markram_support import *  # noqa: F403

class TestTMIsolation:
    def test_defaults(self):
        n = TsodyksMarkramNeuron()
        assert n.v == -65.0 and n.x == 1.0 and n.u == 0.2
        assert n.tau_d == 200.0 and n.tau_f == 600.0

    def test_step_returns_binary(self):
        assert TsodyksMarkramNeuron().step(0.0) in (0, 1)

    def test_dual_input_signature(self):
        """step(current, presynaptic_spike) — two inputs."""
        n = TsodyksMarkramNeuron()
        s = n.step(10.0, presynaptic_spike=True)
        assert s in (0, 1)

    def test_state_finite(self):
        n = TsodyksMarkramNeuron()
        for _ in range(50000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.x) and np.isfinite(n.u)

    def test_reset(self):
        n = TsodyksMarkramNeuron()
        for _ in range(100):
            n.step(20.0, presynaptic_spike=True)
        n.reset()
        assert n.v == n.v_rest and n.x == 1.0 and n.u == n.u_se
