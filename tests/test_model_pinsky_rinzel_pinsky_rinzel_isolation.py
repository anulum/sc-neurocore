# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelIsolation from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelIsolation from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

from tests.model_pinsky_rinzel_support import *  # noqa: F403

class TestPinskyRinzelIsolation:
    def test_construction_defaults(self):
        n = PinskyRinzelNeuron()
        assert n.v_s == -60.0
        assert n.v_d == -60.0
        assert (n.h, n.n, n.s, n.c, n.q, n.ca) == (0.999, 0.001, 0.009, 0.007, 0.01, 0.2)
        assert n.cm == 3.0
        assert n.gc == 2.1
        assert n.p == 0.5
        assert n.dt == 0.02

    def test_step_returns_binary(self):
        assert PinskyRinzelNeuron().step(0.0) in (0, 1)

    def test_dual_input_signature(self):
        assert PinskyRinzelNeuron().step(5.0, 3.0) in (0, 1)

    def test_eight_state_variables_evolve(self):
        n = PinskyRinzelNeuron()
        initial = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca)
        for _ in range(2000):
            n.step(20.0)
        final = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca)
        diffs = [abs(f - i) for f, i in zip(final, initial)]
        assert all(d > 1e-10 for d in diffs), f"Some variables did not evolve: {diffs}"

    def test_state_finite_long_run(self):
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(30.0)
        for var in (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca):
            assert np.isfinite(var)

    def test_reset_restores_initial(self):
        n = PinskyRinzelNeuron()
        for _ in range(1000):
            n.step(30.0)
        n.reset()
        assert (n.v_s, n.v_d) == (-60.0, -60.0)
        assert (n.h, n.n, n.s, n.c, n.q, n.ca) == (0.999, 0.001, 0.009, 0.007, 0.01, 0.2)
