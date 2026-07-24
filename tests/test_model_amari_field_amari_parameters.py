# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmariParameters from former test_model_amari_field.py

"""Focused suite: TestAmariParameters from former test_model_amari_field.py."""

from __future__ import annotations

from tests.model_amari_field_support import *  # noqa: F403


class TestAmariParameters:
    def test_custom_n(self):
        n = AmariNeuralField(n=128)
        assert n.u.shape == (128,) and n._w.shape == (128,)

    def test_tau_controls_speed(self):
        """Larger tau → slower dynamics."""
        n_fast = AmariNeuralField(tau=1.0, a_exc=0.5, b_inh=0.5)
        n_slow = AmariNeuralField(tau=100.0, a_exc=0.5, b_inh=0.5)
        I = np.ones(64) * 0.5
        n_fast.step(I)
        n_slow.step(I)
        assert np.max(np.abs(n_fast.u)) > np.max(np.abs(n_slow.u))

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AmariNeuralField()
            I = np.ones(64) * 0.3
            trace = [n.step(I) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]
