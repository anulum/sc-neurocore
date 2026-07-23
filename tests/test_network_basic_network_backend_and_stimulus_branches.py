# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNetworkBackendAndStimulusBranches from former test_network_basic.py

"""Focused suite: TestNetworkBackendAndStimulusBranches from former test_network_basic.py."""

from __future__ import annotations

from tests.network_basic_support import *  # noqa: F403

class TestNetworkBackendAndStimulusBranches:
    """Backend dispatch and stimulus-application edge branches.

    The Rust (`_run_rust`) and MPI (`_run_mpi`) execution bodies are exercised
    by the engine-injecting and mpi4py tests elsewhere in the suite; here we
    cover the pure-Python progress reporting, the stimulus-routing edge cases,
    and the Rust-absent dispatch guards (forced via the module engine handle).
    """

    def test_rust_backend_raises_when_engine_unavailable(self, monkeypatch):
        import sc_neurocore.network.network as network_mod

        monkeypatch.setattr(network_mod, "_RUST_ENGINE", False)
        net = Network(Population("LapicqueNeuron", 2))
        with pytest.raises(RuntimeError, match="Rust engine not available"):
            net.run(0.005, backend="rust")

    def test_auto_backend_uses_python_when_rust_unavailable(self, monkeypatch):
        import sc_neurocore.network.network as network_mod

        monkeypatch.setattr(network_mod, "_RUST_ENGINE", False)
        net = Network(Population("LapicqueNeuron", 2))
        assert net._can_use_rust() is False
        net.run(0.005, backend="auto")  # falls through to the Python backend

    def test_python_backend_progress_reporting(self, capsys):
        net = Network(Population("LapicqueNeuron", 2))
        net.run(0.02, dt=0.001, backend="python", progress=True)
        out = capsys.readouterr().out
        assert "%" in out
        assert "step" in out

    def test_stimulus_without_target_and_no_population_is_skipped(self):
        net = Network()
        net.add(TimedArray([1.0, 2.0, 3.0]))  # target None, no populations
        net.run(0.003, dt=0.001, backend="python")

    def test_stimulus_targeting_foreign_population_is_skipped(self):
        net = Network(Population("LapicqueNeuron", 2))
        foreign = Population("LapicqueNeuron", 2)
        stim = TimedArray([1.0, 2.0, 3.0])
        stim.target = foreign  # not registered with this network
        net.add(stim)
        net.run(0.003, dt=0.001, backend="python")

    def test_timed_array_stimulus_applied_to_target(self):
        pop = Population("LapicqueNeuron", 2)
        net = Network(pop)
        stim = TimedArray([5.0, 5.0, 5.0])
        stim.target = pop
        net.add(stim)
        net.run(0.003, dt=0.001, backend="python")
