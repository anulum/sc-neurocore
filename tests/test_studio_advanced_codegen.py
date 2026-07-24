# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio advanced codegen

"""Focused suite: TestCodegenAdvanced from former test_studio_advanced.py."""

from __future__ import annotations

from tests.studio_advanced_support import *  # noqa: F403


class TestCodegenAdvanced:
    def test_ode_script(self):
        script = generate_ode_script(
            equations=["dv/dt = -(v - E_L) / tau_m + I / C"],
            threshold="v > -50",
            reset="v = -65",
            params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            init={"v": -65.0},
            duration=100,
            current=30,
            dt=0.1,
        )
        assert "from_equations" in script
        assert "E_L" in script

    def test_oneliner(self):
        line = generate_oneliner("COBALIFNeuron", {"c_m": 200}, 10)
        assert "COBALIFNeuron" in line
        assert "step" in line

    def test_classifier_adapting(self):
        isis_adapting = list(range(50, 150, 10))
        spikes = []
        t = 100
        for isi in isis_adapting:
            spikes.append(t)
            t += isi
        r = classify_firing_pattern(spikes, 2000, 0.1)
        assert r["pattern"] in ("adapting", "irregular", "tonic")

    def test_classifier_bursting(self):
        spikes = []
        for burst_start in range(0, 1000, 200):
            for i in range(5):
                spikes.append(burst_start + i * 5)
        r = classify_firing_pattern(spikes, 1200, 0.1)
        assert r["pattern"] in ("bursting", "irregular", "chaotic")
