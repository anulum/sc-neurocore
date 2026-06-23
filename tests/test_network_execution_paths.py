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

import numpy as np
import pytest

import sc_neurocore.network.network as network_module
from sc_neurocore.network.monitor import SpikeMonitor, StateMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.stimulus import StepCurrent

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

    def add_population(self, model_name: str, n: int) -> int:
        return 0

    def add_projection(self, *args: object) -> None:
        return None

    def run(self, n_steps: int) -> dict[str, object]:
        # voltages[0] length must equal the population size to sync back; the
        # packed spike encodes neuron 1 firing at timestep 2.
        return {"voltages": [[0.1, 0.2, 0.3]], "spike_data": [[(1 << 32) | 2]]}


def test_run_rust_decodes_voltages_and_spike_events(monkeypatch: pytest.MonkeyPatch) -> None:
    pop = Population(_MODEL, 3)
    monitor = SpikeMonitor(pop)
    net = Network(pop, monitor)
    monkeypatch.setattr(network_module, "_get_rust_engine", lambda: _FakeNetworkRunner)
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
