# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - The native network bridge may only run the network it was given

"""The native network dispatch may only take a network it can reproduce.

The Rust runner receives a model name and a neuron count. It therefore builds
default neurons, and a population carrying constructor parameters, independently
derived seeds, or state from an earlier run cannot be reproduced there. Before
this gate, ``backend='auto'`` dispatched such a network anyway: the run returned
results for the defaults while every Python object went on reporting the
caller's parameters.

Every case here fails on that former behaviour. The central one is the
equivalence test — a parameterised network run under ``'auto'`` must land on the
same state as the same network run under ``'python'`` — because that is the
guarantee a caller actually depends on, and it is the one that was broken.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.network import (
    Network,
    PoissonInput,
    Population,
    Projection,
    SpikeMonitor,
    StepCurrent,
    TimedArray,
)
from sc_neurocore.network.rust_dispatch import (
    BRIDGE_CONSTRUCTOR_ARGUMENTS,
    network_divergences,
    population_divergence,
    projection_divergences,
)
from sc_neurocore.neurons import models as model_registry

# A model whose whole state is v: the runner returns voltages only, so a model
# with any other state (AdEx's w) is refused (see TestStateAndIdentity).
MODEL = "SCLapicqueLIFNeuron"
STOCHASTIC_MODEL = "PoissonNeuron"
VECTOR_MODEL = "GLMNeuron"
OVERRIDES = {"v_threshold": 1.5, "v_rest": 0.1}


def _network(params: dict[str, Any] | None) -> tuple[Network, Population, Population]:
    """Return a two-population network wired source to target."""
    source = Population(MODEL, 12, params=params, label="src")
    target = Population(MODEL, 12, params=params, label="tgt")
    projection = Projection(source, target, weight=50.0, probability=0.5, seed=7)
    network = Network(source, target, projection, SpikeMonitor(target), seed=3)
    return network, source, target


def _final_voltages(population: Population) -> np.ndarray[Any, Any]:
    """Return every neuron's membrane voltage after a run."""
    return np.array([neuron.v for neuron in population.neurons], dtype=np.float64)


def _stub_population(
    monkeypatch: pytest.MonkeyPatch, model: type[Any], count: int = 1
) -> Population:
    """Return a population of *model* that the registry resolves under ``MODEL``.

    Some attribute kinds — a string, a flag, an object whose comparison does
    not answer — no catalogue model exposes on the path this check walks. The
    registry entry the bridge would resolve is replaced by *model*, and the
    population's neurons with instances of it, so both sides of every
    comparison are that kind rather than a number.
    """
    population = Population(MODEL, count)
    monkeypatch.setattr(model_registry, MODEL, model, raising=False)
    population.neurons = [model() for _ in range(count)]
    return population


class TestPopulationDivergence:
    """Whether one population is what the bridge builds from its name."""

    def test_a_default_population_is_what_the_bridge_builds(self) -> None:
        """A population built with no parameters matches the bridge's own neurons."""
        assert population_divergence(Population(MODEL, 4)) == ""

    def test_constructor_parameters_are_named(self) -> None:
        """A parameterised population reports the parameters the bridge cannot receive."""
        reason = population_divergence(Population(MODEL, 4, params=dict(OVERRIDES)))
        assert "v_rest" in reason and "v_threshold" in reason
        assert "cannot receive" in reason

    def test_derived_per_neuron_seeds_are_a_divergence(self) -> None:
        """A stochastic population's own streams are not the bridge's single default."""
        reason = population_divergence(Population(STOCHASTIC_MODEL, 4))
        assert "seed" in reason

    def test_a_neuron_changed_in_place_is_a_divergence(self) -> None:
        """The check reads the neurons, so a parameter set after construction counts."""
        population = Population(MODEL, 4)
        assert population_divergence(population) == ""
        population.neurons[2].v_threshold = 3.0
        assert "v_threshold" in population_divergence(population)

    def test_an_empty_population_is_dispatchable(self) -> None:
        """Zero neurons are zero neurons on either side."""
        assert population_divergence(Population(MODEL, 0)) == ""

    def test_a_model_outside_the_registry_cannot_be_built_from_its_name(self) -> None:
        """The bridge resolves a name; a class that no name resolves to is refused."""

        class LocalOnlyNeuron:
            """A model the public registry does not export."""

            def __init__(self) -> None:
                self.v = 0.0

            def step(self, current: float) -> int:
                """Consume one input sample and never spike."""
                self.v += current
                return 0

        reason = population_divergence(Population(LocalOnlyNeuron, 2))
        assert "cannot build from its name alone" in reason

    def test_a_registry_entry_that_is_not_callable_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name that resolves to something uncallable answers no question."""
        population = Population(MODEL, 2)
        monkeypatch.setattr(model_registry, MODEL, "not a model", raising=False)
        assert "cannot build from its name alone" in population_divergence(population)

    def test_a_model_that_needs_arguments_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A model the bridge cannot default-construct is not one it can stand in for."""
        population = Population(MODEL, 2)

        class NeedsArguments:
            """A model that refuses to be built with no arguments."""

            def __init__(self, required: float) -> None:
                self.required = required

        monkeypatch.setattr(model_registry, MODEL, NeedsArguments, raising=False)
        assert "cannot build from its name alone" in population_divergence(population)

    def test_a_different_constructor_behind_the_same_name_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name that now builds another type cannot describe the neurons held."""
        population = Population(MODEL, 2)

        class OtherModel:
            """A model of a different type carrying the same attribute."""

            def __init__(self) -> None:
                self.v = 0.0

        monkeypatch.setattr(model_registry, MODEL, OtherModel, raising=False)
        assert "different constructor" in population_divergence(population)

    def test_many_differing_attributes_are_summarised(self) -> None:
        """A long divergence names the first few attributes and counts the rest."""
        population = Population(MODEL, 2)
        for index, name in enumerate(("v", "v_rest", "v_threshold", "tau", "dt")):
            setattr(population.neurons[0], name, float(index) + 0.5)
        reason = population_divergence(population)
        assert "and 1 more" in reason

    def test_an_attribute_only_one_side_carries_is_a_divergence(self) -> None:
        """An attribute the bridge's neuron does not have is still a difference."""
        population = Population(MODEL, 2)
        population.neurons[0].extra_register = 1.0
        assert "extra_register" in population_divergence(population)

    def test_a_comparison_that_raises_counts_as_a_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A value whose equality raises is treated as different, not as an error."""

        class Unequal:
            """A value whose comparison raises rather than answering."""

            def __eq__(self, other: object) -> bool:
                raise RuntimeError("comparison refused")

            __hash__ = None  # type: ignore[assignment] # deliberately unhashable

        class RefusingNeuron:
            """A model holding a value that refuses to be compared."""

            def __init__(self) -> None:
                self.register = Unequal()

        population = _stub_population(monkeypatch, RefusingNeuron)
        assert "register" in population_divergence(population)

    def test_a_nan_parameter_is_a_divergence(self) -> None:
        """A non-finite value never equals itself, so it is never the default."""
        population = Population(MODEL, 2)
        population.neurons[0].v = math.nan
        assert "v" in population_divergence(population)

    def test_a_number_that_passed_through_an_array_still_matches(self) -> None:
        """A parameter is its value, not the scalar type it arrived in."""
        population = Population(MODEL, 2)
        population.neurons[0].v_rest = np.float64(population.neurons[0].v_rest)
        assert population_divergence(population) == ""

    def test_a_number_replaced_by_a_non_number_is_a_divergence(self) -> None:
        """A parameter that stopped being a number is not the default value."""
        population = Population(MODEL, 2)
        population.neurons[0].v_rest = "-65.0"
        assert "v_rest" in population_divergence(population)

    def test_equal_arrays_match_and_a_changed_element_does_not(self) -> None:
        """A model carrying arrays is compared element by element, not by identity.

        GLM holds four arrays and an RNG object. A default population diverges
        on the RNG alone, which is exactly the evidence that the four arrays —
        distinct objects with equal contents — compared as equal. Changing one
        element of one of them adds that array to the reason.
        """
        population = Population(VECTOR_MODEL, 2)
        assert "carries _rng," in population_divergence(population)

        kernel = np.array(population.neurons[0].k, dtype=np.float64)
        kernel[0] += 1.0
        population.neurons[0].k = kernel
        assert "_rng, k," in population_divergence(population)

    def test_an_array_against_a_scalar_is_a_divergence(self) -> None:
        """One side holding an array and the other a scalar is not a match."""
        population = Population(MODEL, 2)
        population.neurons[0].v = np.zeros(3, dtype=np.float64)
        assert "v" in population_divergence(population)

    def test_a_comparison_that_answers_with_a_non_bool_is_a_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only a plain True settles a comparison; anything else is a difference."""

        class Undecided:
            """A value whose comparison answers with something other than a bool."""

            def __eq__(self, other: object) -> Any:
                return "maybe"

            __hash__ = None  # type: ignore[assignment] # deliberately unhashable

        class UndecidedNeuron:
            """A model holding a value whose comparison never answers True."""

            def __init__(self) -> None:
                self.register = Undecided()

        population = _stub_population(monkeypatch, UndecidedNeuron)
        assert "register" in population_divergence(population)

    def test_an_object_that_compares_equal_is_not_a_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An object that answers a plain True settles as a match."""

        class Settled:
            """A value that compares equal to any other of its kind."""

            def __eq__(self, other: object) -> bool:
                return isinstance(other, Settled)

            __hash__ = None  # type: ignore[assignment] # deliberately unhashable

        class SettledNeuron:
            """A model holding a value that compares equal across instances."""

            def __init__(self) -> None:
                self.register = Settled()

        population = _stub_population(monkeypatch, SettledNeuron)
        assert population_divergence(population) == ""

    def test_flags_and_names_are_compared_by_kind_and_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A string or a boolean is not a number, and is compared as itself."""

        class FlaggedNeuron:
            """A model whose configuration is a profile name and a flag."""

            def __init__(self) -> None:
                self.profile = "source"
                self.excited = False

        population = _stub_population(monkeypatch, FlaggedNeuron)
        assert population_divergence(population) == ""

        population.neurons[0].excited = True
        assert "excited" in population_divergence(population)

        population.neurons[0].excited = False
        population.neurons[0].profile = "extended"
        assert "profile" in population_divergence(population)

    def test_a_flag_replaced_by_a_number_is_a_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A boolean is not the number one, however it compares."""

        class FlagNeuron:
            """A model whose only register is a flag."""

            def __init__(self) -> None:
                self.excited = False

        population = _stub_population(monkeypatch, FlagNeuron)
        population.neurons[0].excited = 0
        assert "excited" in population_divergence(population)

    def test_a_model_without_an_instance_dictionary_is_compared_by_its_slots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A slotted model carries no ``__dict__``, and must not read as carrying nothing.

        Reading only ``vars`` would make every slotted population compare empty
        against empty and dispatch — a wrong "yes", the one direction this
        check may never take.
        """

        class SlottedNeuron:
            """A model whose attributes live in slots rather than a dictionary."""

            __slots__ = ("v", "v_rest")

            def __init__(self) -> None:
                self.v = 0.0
                self.v_rest = -65.0

        population = _stub_population(monkeypatch, SlottedNeuron)
        assert population_divergence(population) == ""

        population.neurons[0].v = 1.0
        assert "v" in population_divergence(population)

    def test_an_unassigned_slot_is_absent_on_both_sides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared slot that was never assigned is absent, exactly as for a key."""

        class PartiallyAssignedNeuron:
            """A model that declares a slot it does not always fill."""

            __slots__ = ("v", "optional_register")

            def __init__(self) -> None:
                self.v = 0.0

        population = _stub_population(monkeypatch, PartiallyAssignedNeuron)
        assert population_divergence(population) == ""

        population.neurons[0].optional_register = 2.0
        assert "optional_register" in population_divergence(population)

    def test_a_model_whose_type_changed_is_still_refused_before_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The constructor check runs first, so a mismatched type never reaches equality."""

        class SlottedNeuron:
            """A model whose attributes live in slots rather than a dictionary."""

            __slots__ = ("v",)

            def __init__(self) -> None:
                self.v = 0.0

        population = Population(MODEL, 1)
        monkeypatch.setattr(model_registry, MODEL, SlottedNeuron, raising=False)
        assert "different constructor" in population_divergence(population)


class TestNetworkDivergences:
    """One reason per population the bridge cannot build."""

    def test_a_default_network_reports_nothing(self) -> None:
        """Every population matching the bridge leaves the list empty."""
        _, source, target = _network(None)
        assert network_divergences([source, target]) == []

    def test_one_reason_per_diverging_population(self) -> None:
        """Each population that cannot be reproduced is named separately."""
        _, source, target = _network(dict(OVERRIDES))
        reasons = network_divergences([source, target])
        assert len(reasons) == 2
        assert "'src'" in reasons[0] and "'tgt'" in reasons[1]

    def test_the_bridge_arguments_are_the_documented_pair(self) -> None:
        """The constant states exactly what crosses the boundary."""
        assert BRIDGE_CONSTRUCTOR_ARGUMENTS == ("model_name", "n")


class TestNetworkDispatch:
    """Auto and forced dispatch follow the divergences."""

    def test_a_default_network_still_dispatches_to_rust(self) -> None:
        """The gate must not cost the dispatch it was added to protect."""
        network, _, _ = _network(None)
        assert network._can_use_rust() is True

    def test_a_parameterised_network_does_not_dispatch_to_rust(self) -> None:
        """A network the bridge cannot build is not eligible for it."""
        network, _, _ = _network(dict(OVERRIDES))
        assert network._can_use_rust() is False

    def test_auto_reproduces_the_python_run_for_a_parameterised_network(self) -> None:
        """The guarantee that was broken: 'auto' must not change the model."""
        expected_network, _, expected_target = _network(dict(OVERRIDES))
        expected_network.run(0.05, dt=0.001, backend="python")

        actual_network, _, actual_target = _network(dict(OVERRIDES))
        actual_network.run(0.05, dt=0.001, backend="auto")

        np.testing.assert_array_equal(
            _final_voltages(actual_target), _final_voltages(expected_target)
        )
        assert actual_target.neurons[0].v_threshold == OVERRIDES["v_threshold"]

    def test_auto_reproduces_the_python_run_for_neurons_changed_in_place(self) -> None:
        """State set on the neurons before a run is part of the model too."""
        results = []
        for backend in ("python", "auto"):
            network, source, target = _network(None)
            for neuron in source.neurons:
                neuron.v = neuron.v_threshold - 0.1
            network.run(0.05, dt=0.001, backend=backend)
            results.append(_final_voltages(target))
        np.testing.assert_array_equal(results[1], results[0])

    def test_a_second_run_does_not_discard_the_state_of_the_first(self) -> None:
        """After a run the neurons have moved, so the bridge can no longer stand in.

        A quadratic integrate-and-fire neuron drifts from its default voltage
        without input, so its first run (dispatched) leaves it off the default.
        """
        network = Network(Population("QuadraticIFNeuron", 3))
        assert network._can_use_rust() is True
        network.run(0.02, dt=0.001, backend="auto")
        assert network._can_use_rust() is False

    def test_forced_rust_raises_and_names_the_population(self) -> None:
        """A caller who forces the backend is told which population and which values."""
        network, _, _ = _network(dict(OVERRIDES))
        with pytest.raises(NotImplementedError) as excinfo:
            network.run(0.01, dt=0.001, backend="rust")
        message = str(excinfo.value)
        assert "'src'" in message
        assert "v_threshold" in message
        assert "backend='python'" in message


class TestStateAndIdentity:
    """RIGOR review 279: the runner returns voltages only, and one name, two models."""

    def test_a_model_with_state_besides_v_is_refused(self) -> None:
        """NEURO-RUST-DISPATCH-STATE-01: AdEx's w stayed 0 after a native run."""
        reason = population_divergence(Population("AdExNeuron", 3))
        assert "evolves w besides v" in reason
        reason = population_divergence(Population("RulkovMapNeuron", 3))
        assert "evolves" in reason and "x" in reason and "y" in reason

    def test_auto_runs_adex_in_python_and_keeps_w(self) -> None:
        """Auto falls back, so w after 1000 zero-input steps is the Python value."""
        expected = Population("AdExNeuron", 1)
        Network(expected).run(1.0, dt=0.001, backend="python")
        actual = Population("AdExNeuron", 1)
        Network(actual).run(1.0, dt=0.001, backend="auto")
        assert actual.neurons[0].w == expected.neurons[0].w != 0.0

    def test_forced_rust_refuses_adex(self) -> None:
        """A forced run is refused before the runner starts."""
        with pytest.raises(NotImplementedError, match="evolves w"):
            Network(Population("AdExNeuron", 2)).run(0.01, dt=0.001, backend="rust")

    def test_lapicque_names_another_model_in_the_engine(self) -> None:
        """NEURO-RUST-DISPATCH-PROFILE-01: the engine's LapicqueNeuron is the 1907 profile."""
        reason = population_divergence(Population("LapicqueNeuron", 2))
        assert "1907" in reason and "SCLapicqueLIFNeuron" in reason
        assert population_divergence(Population("SCLapicqueLIFNeuron", 2)) == ""


def _edge(**options: Any) -> tuple[Network, Population]:
    """Return a one-source, three-target network with weights 0.1, 0.5 and 1.0."""
    source = Population(MODEL, 1, label="src")
    target = Population(MODEL, 3, label="tgt")
    csr = (np.array([0, 3]), np.array([0, 1, 2]), np.array([0.1, 0.5, 1.0]))
    projection = Projection(source, target, 1.0, topology=csr, **options)
    return Network(source, target, projection), target


class TestWhatTheRunnerCannotCarry:
    """RIGOR reviews 267 and 268: stimuli and projection features the runner drops."""

    @pytest.mark.parametrize(
        "stimulus",
        [
            lambda: StepCurrent(10, 40, 2.0),
            lambda: TimedArray(np.full(50, 2.0)),
            lambda: PoissonInput(2, rate_hz=500.0, weight=2.0, seed=1),
        ],
    )
    def test_embedded_stimuli_are_refused(self, stimulus: Any) -> None:
        """NEURO-STIM-FORCED-RUST-01: forced Rust ran as if there were no input."""
        network = Network(Population(MODEL, 2), stimulus())
        assert network._can_use_rust() is False
        with pytest.raises(NotImplementedError, match="embedded stimuli"):
            network.run(0.01, dt=0.001, backend="rust")

    @pytest.mark.parametrize(
        ("options", "named"),
        [
            ({"delay": np.array([0.0, 3.0, 1.0])}, "per-synapse delays"),
            ({"weight_threshold": 0.5}, "weight_threshold"),
            ({"plasticity": "stdp"}, "plastic"),
        ],
    )
    def test_projection_features_the_runner_drops_are_refused(
        self, options: dict[str, Any], named: str
    ) -> None:
        """NEURO-PROJ-RUST-DELAY/THRESHOLD/FORCED-STDP-01: each was silently dropped."""
        network, _ = _edge(**options)
        assert any(named in reason for reason in projection_divergences(network.projections))
        assert network._can_use_rust() is False
        with pytest.raises(NotImplementedError, match=named):
            network.run(0.01, dt=0.001, backend="rust")

    def test_a_uniform_delay_still_dispatches(self) -> None:
        """The runner takes one uniform delay, so that projection is not refused."""
        network, _ = _edge(delay=2.0)
        assert projection_divergences(network.projections) == []
        assert network._can_use_rust() is True


def test_the_threshold_prunes_per_synapse_delays_in_python() -> None:
    """NEURO-PROJ-THRESHOLD-DELAY-01: weights [.1, .5, 1], threshold .5 gave [.1, .5, 1].

    With per-synapse delays the threshold was ignored; the scalar-delay control
    already gave [0, 0, 1]. Both paths now prune the same synapses.
    """
    _, target = _edge()
    source = Population(MODEL, 1)
    csr = (np.array([0, 3]), np.array([0, 1, 2]), np.array([0.1, 0.5, 1.0]))
    delayed = Projection(
        source, target, 1.0, topology=csr, delay=np.array([1.0, 1.0, 1.0]), weight_threshold=0.5
    )
    scalar = Projection(source, target, 1.0, topology=csr, delay=1.0, weight_threshold=0.5)
    spike = np.array([1], dtype=np.int8)
    quiet = np.array([0], dtype=np.int8)
    delayed_out = [delayed.propagate(spike), delayed.propagate(quiet)]
    scalar_out = [scalar.propagate(spike), scalar.propagate(quiet)]
    assert delayed_out[1].tolist() == scalar_out[1].tolist() == [0.0, 0.0, 1.0]
