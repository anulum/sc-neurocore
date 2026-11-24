# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio advanced codegen

"""Focused suite: equation exports and the firing-pattern classifier."""

from __future__ import annotations

from sc_neurocore.studio.experiment_spec import resolve_experiment
from sc_neurocore.studio.replay_pack import pinned_request

from tests.studio_advanced_support import (
    classify_firing_pattern,
    generate_experiment_script,
    generate_oneliner,
)

EQUATIONS = {
    "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
    "threshold": "v > -50",
    "reset": "v = -65",
    "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
    "init": {"v": -65.0},
    "duration": 100.0,
    "current": 30.0,
    "dt": 0.1,
}


class TestCodegenAdvanced:
    """Code generation from equation experiments and firing-pattern classification."""

    def test_the_equation_export_carries_the_equations_and_the_initial_state(self) -> None:
        """The exported script carries the equation, the parameters, the initial state and the experiment hash."""
        spec = resolve_experiment(EQUATIONS)
        script = generate_experiment_script(spec, pinned_request(EQUATIONS, spec))
        assert "'dv/dt = -(v - E_L) / tau_m + I / C'" in script
        assert "'E_L': -65.0" in script
        assert "'init': {'v': -65.0}" in script
        assert spec.experiment_sha256 in script

    def test_the_oneliner_runs_the_same_experiment(self) -> None:
        """The one-line export runs the same resolved experiment on a single line."""
        spec = resolve_experiment(EQUATIONS)
        line = generate_oneliner(spec, pinned_request(EQUATIONS, spec))
        assert "run_experiment(resolve_experiment(" in line
        assert "dv/dt = -(v - E_L) / tau_m + I / C" in line
        assert line.count("\n") == 0

    def test_classifier_adapting(self) -> None:
        """An adapting trace is classed as adapting, irregular or tonic."""
        isis_adapting = list(range(50, 150, 10))
        spikes = []
        t = 100
        for isi in isis_adapting:
            spikes.append(t)
            t += isi
        r = classify_firing_pattern(spikes, 2000, 0.1)
        assert r["pattern"] in ("adapting", "irregular", "tonic")

    def test_classifier_bursting(self) -> None:
        """A bursting trace is classed as bursting, irregular or chaotic."""
        spikes = []
        for burst_start in range(0, 1000, 200):
            for i in range(5):
                spikes.append(burst_start + i * 5)
        r = classify_firing_pattern(spikes, 1200, 0.1)
        assert r["pattern"] in ("bursting", "irregular", "chaotic")
