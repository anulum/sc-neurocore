# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network orchestrator execution-path coverage

"""Execution-path coverage for the Network simulation orchestrator: the Rust
backend body, the MPI dispatch guards, the engine-detection helpers, and the
pure-Python FIM / plasticity / torch-bridge paths."""

from __future__ import annotations

import builtins
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

import sc_neurocore.network.network as network_module
from sc_neurocore.network.monitor import RateMonitor, SpikeMonitor, StateMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.stimulus import StepCurrent

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_MODEL = "AdExNeuron"


def test_engine_detection_helpers_after_cache_reset() -> None:
    # Reset the cached engine so the loader runs under the tracer.
    network_module._RUST_ENGINE = None
    engine = network_module._get_rust_engine()
    assert engine is not False
    # A bare model name present directly in the supported set is matched.
    assert network_module._rust_supports_model("AdEx") is True
    # A "...Neuron"-suffixed name matches via the suffix-stripped lookup.
    assert network_module._rust_supports_model("AdExNeuron") is True
    assert network_module._rust_supports_model("DefinitelyNotARealModelNeuron") is False


def test_engine_loader_falls_back_to_top_level_network_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sc_neurocore_engine.network":
            raise ImportError("bridge helper unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert network_module._load_network_runner_class().__name__ == "NetworkRunner"


def test_engine_loader_reports_missing_network_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"sc_neurocore_engine.network", "sc_neurocore_engine"}:
            raise ImportError("engine unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="NetworkRunner is unavailable"):
        network_module._load_network_runner_class()


def test_get_rust_engine_caches_false_when_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_loader() -> Any:
        raise ImportError("engine unavailable")

    monkeypatch.setattr(network_module, "_RUST_ENGINE", None)
    monkeypatch.setattr(network_module, "_load_network_runner_class", fake_loader)

    assert network_module._get_rust_engine() is False


def test_rust_supports_model_returns_false_without_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(network_module, "_RUST_ENGINE", False)
    assert network_module._rust_supports_model("AdEx") is False


def test_can_use_rust_false_for_unsupported_model(monkeypatch: pytest.MonkeyPatch) -> None:
    pop = Population(_MODEL, 2)
    net = Network(pop)
    monkeypatch.setattr(network_module, "_rust_supports_model", lambda _name: False)
    assert net._can_use_rust() is False


def test_rust_backend_runs_populations_and_projection() -> None:
    source = Population(_MODEL, 3)
    target = Population(_MODEL, 3)
    projection = Projection(source, target, weight=0.5, topology="all_to_all")
    monitor = SpikeMonitor(target)
    net = Network(source, target, projection, monitor)

    assert net._can_use_rust() is True
    net.run(0.01, dt=0.001, backend="rust")


def test_can_use_rust_is_disabled_by_stimuli_and_plasticity() -> None:
    pop = Population(_MODEL, 2)
    assert Network(pop, StepCurrent(onset=0, offset=5, amplitude=1.0))._can_use_rust() is False

    source = Population(_MODEL, 2)
    target = Population(_MODEL, 2)
    plastic = Projection(source, target, weight=0.5, topology="all_to_all", plasticity="stdp")
    assert Network(source, target, plastic)._can_use_rust() is False


def test_mpi_backend_guards_each_unsupported_feature() -> None:
    pop = Population(_MODEL, 2)
    with pytest.raises(NotImplementedError, match="spike_gating"):
        Network(pop).run(0.01, dt=0.001, backend="mpi", spike_gating=True)

    with pytest.raises(NotImplementedError, match="fim_lambda"):
        Network(pop, fim_lambda=1.0).run(0.01, dt=0.001, backend="mpi")

    with pytest.raises(NotImplementedError, match="embedded stimuli"):
        Network(pop, StepCurrent(onset=0, offset=5, amplitude=1.0)).run(
            0.01, dt=0.001, backend="mpi"
        )

    with pytest.raises(NotImplementedError, match="state monitors"):
        Network(pop, StateMonitor(pop, ["v"])).run(0.01, dt=0.001, backend="mpi")

    source = Population(_MODEL, 2)
    target = Population(_MODEL, 2)
    plastic = Projection(source, target, weight=0.5, topology="all_to_all", plasticity="stdp")
    with pytest.raises(NotImplementedError, match="synaptic plasticity"):
        Network(source, target, plastic).run(0.01, dt=0.001, backend="mpi")


def test_python_backend_runs_with_fim_feedback() -> None:
    source = Population(_MODEL, 3)
    target = Population(_MODEL, 3)
    projection = Projection(source, target, weight=0.5, topology="all_to_all")
    net = Network(source, target, projection, fim_lambda=2.0)
    # fim_lambda > 0 drives the _apply_fim call site in the step loop.
    net.run(0.003, dt=0.001, backend="python")


def test_apply_fim_adjusts_projection_weights_for_non_uniform_spikes() -> None:
    source = Population(_MODEL, 3)
    target = Population(_MODEL, 3)
    projection = Projection(source, target, weight=0.5, topology="all_to_all")
    net = Network(source, target, projection, fim_lambda=4.0)
    before = projection.data.copy()
    # A non-uniform source spike vector gives non-zero deviations, so the inner
    # weight-correction loop runs for the spiking neuron.
    net._apply_fim({id(source): np.array([1, 0, 0], dtype=np.int8)})
    assert not np.array_equal(before, projection.data)


def test_python_backend_runs_with_plastic_projection() -> None:
    source = Population(_MODEL, 3)
    target = Population(_MODEL, 3)
    plastic = Projection(source, target, weight=0.5, topology="all_to_all", plasticity="stdp")
    net = Network(source, target, plastic)
    # The plastic projection drives _update_plasticity inside the step loop.
    net.run(0.003, dt=0.001, backend="python")


def test_to_torch_rejects_stimuli_and_builds_bridge() -> None:
    pop = Population(_MODEL, 2)
    with pytest.raises(NotImplementedError, match="does not support embedded stimuli"):
        Network(pop, StepCurrent(onset=0, offset=5, amplitude=1.0)).to_torch()

    # The torch bridge supports LapicqueNeuron cells; the default surrogate is
    # resolved when none is supplied.
    source = Population("LapicqueNeuron", 2)
    target = Population("LapicqueNeuron", 2)
    projection = Projection(source, target, weight=0.5, topology="all_to_all")
    bridge = Network(source, target, projection).to_torch()
    assert bridge is not None


def test_apply_stimuli_defaults_to_first_population_when_target_unset() -> None:
    pop = Population(_MODEL, 2)
    stimulus = StepCurrent(onset=0, offset=5, amplitude=1.0)
    assert stimulus.target is None  # untargeted -> first population fallback
    net = Network(pop, stimulus)
    # The python loop routes the untargeted stimulus to populations[0].
    net.run(0.003, dt=0.001, backend="python")


class _FakeNetworkRunner:
    """Stand-in Rust runner returning crafted voltages and packed spike events,
    so the Rust result-decode body runs deterministically without the engine."""

    instances: list[_FakeNetworkRunner] = []

    def __init__(self) -> None:
        self.added_models: list[str] = []
        type(self).instances.append(self)

    @staticmethod
    def supported_models() -> set[str]:
        return {_MODEL}

    def add_population(self, model_name: str, n: int) -> int:
        self.added_models.append(model_name)
        return 0

    def add_projection(self, *args: object) -> None:
        return None

    def run(self, n_steps: int) -> dict[str, object]:
        # voltages[0] length must equal the population size to sync back; the
        # packed spike encodes neuron 1 firing at timestep 2.
        return {"voltages": [[0.1, 0.2, 0.3]], "spike_data": [[(1 << 32) | 2]]}


def _install_fake_rust_engine(monkeypatch: pytest.MonkeyPatch) -> type[_FakeNetworkRunner]:
    """Install a deterministic fake Rust runner for Python-side dispatch tests."""
    _FakeNetworkRunner.instances.clear()
    monkeypatch.setattr(network_module, "_get_rust_engine", lambda: _FakeNetworkRunner)
    return _FakeNetworkRunner


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML manifest through the Python-version appropriate parser."""
    with path.open("rb") as manifest_file:
        return tomllib.load(manifest_file)


def test_auto_backend_uses_python_for_state_monitors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)
    monitor = StateMonitor(pop, ["v"])

    Network(pop, monitor).run(0.003, dt=0.001, backend="auto")

    assert fake_runner.instances == []
    assert monitor.traces["v"].shape == (3, 2)
    assert monitor.t.tolist() == [0, 1, 2]


def test_forced_rust_rejects_state_monitors_until_step_traces_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)
    monitor = StateMonitor(pop, ["v"])

    with pytest.raises(NotImplementedError, match="StateMonitor"):
        Network(pop, monitor).run(0.003, dt=0.001, backend="rust")


def test_forced_rust_rejects_rate_monitors_until_step_traces_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)
    monitor = RateMonitor(pop)

    with pytest.raises(NotImplementedError, match="RateMonitor"):
        Network(pop, monitor).run(0.003, dt=0.001, backend="rust")


def test_auto_backend_uses_python_for_spike_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)

    Network(pop).run(0.003, dt=0.001, backend="auto", spike_gating=True)

    assert fake_runner.instances == []


def test_forced_rust_rejects_spike_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)

    with pytest.raises(NotImplementedError, match="spike_gating"):
        Network(pop).run(0.003, dt=0.001, backend="rust", spike_gating=True)


def test_auto_backend_uses_python_for_fim_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)

    Network(pop, fim_lambda=1.0).run(0.003, dt=0.001, backend="auto")

    assert fake_runner.instances == []


def test_forced_rust_rejects_fim_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)

    with pytest.raises(NotImplementedError, match="fim_lambda"):
        Network(pop, fim_lambda=1.0).run(0.003, dt=0.001, backend="rust")


def test_auto_rust_dispatch_uses_model_identity_not_population_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2, label="exc")

    Network(pop).run(0.003, dt=0.001, backend="auto")

    assert len(fake_runner.instances) == 1
    assert fake_runner.instances[0].added_models == [_MODEL]


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
