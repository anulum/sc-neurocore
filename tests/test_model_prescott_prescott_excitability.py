# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrescottExcitability from former test_model_prescott.py

"""Focused suite: TestPrescottExcitability from former test_model_prescott.py."""

from __future__ import annotations

from tests.model_prescott_support import *  # noqa: F403

class TestPrescottExcitability:
    def test_beta_w_modulates_firing(self):
        """Higher beta_w (more positive) recruits more slow K feedback."""
        n_low = PrescottNeuron(beta_w=-30.0)  # Type I-like
        n_high = PrescottNeuron(beta_w=-10.0)  # Type II/III-like
        s_low = len(_run(n_low, current=50.0, steps=100000))
        s_high = len(_run(n_high, current=50.0, steps=100000))
        assert s_low >= s_high, f"beta_w=-30: {s_low} spikes, beta_w=-10: {s_high}"

    def test_high_beta_w_suppresses_firing(self):
        """At beta_w=0, slow K is highly activated and suppresses firing."""
        n = PrescottNeuron(beta_w=0.0)
        spikes = _run(n, current=50.0, steps=100000)
        assert len(spikes) <= 5, f"beta_w=0: {len(spikes)} spikes — expected suppression"

    def test_w_dynamics_timescale(self):
        """w evolves on tau_w timescale. Larger tau_w → slower adaptation."""
        n_fast = PrescottNeuron(tau_w=50.0)
        n_slow = PrescottNeuron(tau_w=200.0)
        for _ in range(5000):
            n_fast.step(50.0)
            n_slow.step(50.0)
        # Both should have evolved, but rates differ
        # (hard to assert direction — just verify w moved)
        assert n_fast.w != 0.0
        assert n_slow.w != 0.0
