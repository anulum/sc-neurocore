# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rust_mpi_workspace) from former test_network_execution_paths.py

from __future__ import annotations

from tests.network_execution_paths_support import *  # noqa: F403


def test_workspace_release_profile_uses_abort_panic() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = _load_toml(project_root / "Cargo.toml")
    profile = manifest.get("profile")
    assert isinstance(profile, dict)
    release = profile.get("release")
    assert isinstance(release, dict)
    assert release["panic"] == "abort"

    engine_manifest = _load_toml(project_root / "engine" / "Cargo.toml")
    assert "profile" not in engine_manifest


def test_run_rust_decodes_voltages_and_spike_events(monkeypatch: pytest.MonkeyPatch) -> None:
    pop = Population(_MODEL, 3)
    monitor = SpikeMonitor(pop)
    net = Network(pop, monitor)
    _install_fake_rust_engine(monkeypatch)
    net.run(0.005, dt=0.001, backend="rust")
    # The crafted spike event (neuron 1 at step 2) is decoded into the monitor.
    assert 1 in monitor._neuron_ids
    assert 2 in monitor._timesteps


def test_run_mpi_invokes_runner_for_a_clean_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeMPIRunner:
        def __init__(self, net: object) -> None:
            calls["constructed"] = True

        def run(self, n_steps: int, dt: float) -> None:
            calls["n_steps"] = n_steps

    import sc_neurocore.network.mpi_runner as mpi_runner_module

    monkeypatch.setattr(mpi_runner_module, "MPIRunner", _FakeMPIRunner)
    pop = Population(_MODEL, 2)
    Network(pop).run(0.005, dt=0.001, backend="mpi")
    assert calls["constructed"] is True
    assert calls["n_steps"] == 5
