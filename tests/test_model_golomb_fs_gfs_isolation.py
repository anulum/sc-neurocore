# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGFSIsolation from former test_model_golomb_fs.py

"""Focused suite: TestGFSIsolation from former test_model_golomb_fs.py."""

from __future__ import annotations

from tests.model_golomb_fs_support import *  # noqa: F403

class TestGFSIsolation:
    def test_defaults(self):
        n = GolombFSNeuron()
        assert n.v == -65.0 and n.h == 0.9 and n.n == 0.1 and n.p == 0.0
        assert n.g_kv3 == 150.0  # Kv3 — signature channel
        assert n.dt == 0.01 and n.v_threshold == -20.0

    def test_step_returns_binary(self):
        assert GolombFSNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = GolombFSNeuron()
        for _ in range(5000):
            n.step(5.0)
        for attr in ["v", "h", "n", "p"]:
            assert np.isfinite(getattr(n, attr))

    def test_reset_restores_defaults(self):
        n = GolombFSNeuron()
        for _ in range(1000):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0 and n.h == 0.9 and n.n == 0.1 and n.p == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = GolombFSNeuron()
            trace = [(n.step(5.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
