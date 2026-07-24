# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (backends_and_features) from former test_network_execution_paths.py

from __future__ import annotations

from tests.network_execution_paths_support import *  # noqa: F403


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
