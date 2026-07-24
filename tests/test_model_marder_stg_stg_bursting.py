# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGBursting from former test_model_marder_stg.py

"""Focused suite: TestSTGBursting from former test_model_marder_stg.py."""

from __future__ import annotations

from tests.model_marder_stg_support import *  # noqa: F403


class TestSTGBursting:
    def test_fires_at_zero_current(self):
        assert len(_run(MarderSTGNeuron(), current=0.0, steps=50_000)) >= 10

    def test_bursting_pattern(self):
        spikes = _run(MarderSTGNeuron(), current=0.0, steps=100_000)
        isis = np.diff(spikes)
        assert isis.max() > 3 * np.median(isis), "expected bursts separated by quiescent gaps"

    def test_calcium_accumulates_during_spiking(self):
        n = MarderSTGNeuron()
        for _ in range(50_000):
            n.step(0.0)
        assert n.ca > 1.0

    def test_calcium_non_negative(self):
        n = MarderSTGNeuron()
        for _ in range(100_000):
            n.step(2.0)
            assert n.ca >= 0.0

    def test_voltage_bounded(self):
        n = MarderSTGNeuron()
        vs = [n.v for _ in range(50_000) if (n.step(0.0) or True)]
        assert min(vs) > -100.0 and max(vs) < 80.0
