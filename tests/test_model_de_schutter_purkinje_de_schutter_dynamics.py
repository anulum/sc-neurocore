# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeSchutterDynamics from former test_model_de_schutter_purkinje.py

"""Focused suite: TestDeSchutterDynamics from former test_model_de_schutter_purkinje.py."""

from __future__ import annotations

from tests.model_de_schutter_purkinje_support import *  # noqa: F403

class TestDeSchutterDynamics:
    def test_converges_to_fixed_point(self) -> None:
        """V converges to stable FP at I=0."""
        n = DeSchutterPurkinjeNeuron()
        for _ in range(20000):
            n.step(0.0)
        v1 = n.v
        for _ in range(10000):
            n.step(0.0)
        assert abs(n.v - v1) < 0.1

    def test_v_shifts_with_current(self) -> None:
        n0 = DeSchutterPurkinjeNeuron()
        n100 = DeSchutterPurkinjeNeuron()
        for _ in range(20000):
            n0.step(0.0)
            n100.step(100.0)
        assert n100.v > n0.v

    def test_high_current_transient_spike(self) -> None:
        """I=500+ can produce 1 transient spike."""
        n = DeSchutterPurkinjeNeuron()
        spikes = _run(n, current=500.0, steps=20000)
        assert len(spikes) >= 1

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = DeSchutterPurkinjeNeuron()
            trace = [(n.step(10.0), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]
