# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - A gated run must be the ungated run

"""Spike gating must not change what a population computes.

Gating skipped any neuron whose input was zero and whose voltage sat within one
percent of rest. A neuron a little away from rest is exactly the one with
relaxation still to do, so the skip froze the leak, the adaptation current and
the refractory countdown the model would have advanced — and the run diverged
from the same run ungated. The docstring said skipped neurons still decayed;
they did not.

A neuron may now be skipped only when skipping it and stepping it are the same
thing: its input is exactly zero and its whole state matches a state this
model's zero-input map was *measured* to leave unchanged. Every case here fails
on the former behaviour, and the central one is the equivalence: gated and
ungated runs must agree exactly, on the catalogue, not on a chosen few.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from sc_neurocore.network import Population
from sc_neurocore.network.quiescence import (
    INCOMPARABLE,
    QUIESCENT_DRIVE,
    is_quiescent,
    quiescent_signature,
    state_signature,
)
from sc_neurocore.neurons.models import _CLASS_TO_MODULE

# Models that stand for the families the gate used to break: an exponential
# whose rest is not a fixed point of its own map, an adaptive threshold, and a
# source-profile integrator.
REPRESENTATIVE_MODELS = ("AdExNeuron", "AdaptiveThresholdIFNeuron", "LapicqueNeuron")

STEPS = 120
POPULATION = 4

# A drive that makes the adaptive-threshold model actually fire, so the cases
# that need spiking are not vacuously true.
SPIKING_DRIVE = 500.0

# The catalogue sweep: enough steps for a relaxation to show, on a fixed seed so
# a model drawing from the process-wide generator cannot differ for that reason.
CATALOGUE_STEPS = 20
CATALOGUE_SEED = 23


def _instance(class_name: str) -> Any:
    """Return a default-constructed catalogue model."""
    module = importlib.import_module(f"sc_neurocore.neurons.models.{_CLASS_TO_MODULE[class_name]}")
    model: Any = getattr(module, class_name)()
    return model


class CountingQuiescent:
    """A model that holds still under zero input and counts every step."""

    steps = 0

    def __init__(self) -> None:
        self.v = 0.0

    def step(self, current: float) -> int:
        """Count the call, then move only when driven."""
        type(self).steps += 1
        self.v += current
        return 0


class CountingRestless:
    """A model that drifts under zero input and counts every step."""

    steps = 0

    def __init__(self) -> None:
        self.v = 0.0

    def step(self, current: float) -> int:
        """Count the call, then drift regardless of the drive."""
        type(self).steps += 1
        self.v += current + 1.0
        return 0


def _park_near_rest(population: Population) -> None:
    """Move every neuron just inside the band the old gate skipped.

    Models whose voltage is not a plain number are left where they are; the
    point is to place a neuron where the old heuristic would have frozen it.
    """
    for neuron in population.neurons:
        if not isinstance(getattr(neuron, "v", None), float):
            continue
        rest = float(getattr(neuron, "v_rest", neuron.v))
        threshold = float(getattr(neuron, "v_threshold", 1.0))
        neuron.v = rest + 0.005 * abs(threshold - rest)


def _run(class_name: str, *, gated: bool, park: bool) -> list[tuple[str, str]]:
    """Return the exact state of the first neuron after a zero-input run."""
    population = Population(class_name, POPULATION)
    if park:
        _park_near_rest(population)
    drive = np.zeros(POPULATION, dtype=np.float64)
    for _ in range(STEPS):
        population.step_all(drive, spike_gating=gated)
    return list(state_signature(population.neurons[0]))


class TestQuiescenceProbe:
    def test_a_model_that_moves_under_zero_input_has_no_quiescent_state(self) -> None:
        """AdEx's exponential term is non-zero at rest, so rest is not a fixed point."""
        assert quiescent_signature(_instance("AdExNeuron")) is None

    def test_a_model_that_holds_still_reports_its_state(self) -> None:
        """A model the zero-input map returns unchanged gives a signature."""
        signature = quiescent_signature(_instance("LapicqueNeuron"))
        assert signature is not None
        assert dict(signature)["v"].startswith("number:")

    def test_a_neuron_away_from_rest_is_not_quiescent(self) -> None:
        """The test is exact: being nearly at rest is not being at rest."""
        neuron = _instance("LapicqueNeuron")
        signature = quiescent_signature(neuron)
        assert is_quiescent(neuron, signature)
        neuron.v += 1e-12
        assert not is_quiescent(neuron, signature)

    def test_no_signature_means_never_skippable(self) -> None:
        """A model with no measured fixed point can never match one."""
        assert is_quiescent(_instance("AdExNeuron"), None) is False

    def test_a_number_is_its_value_not_its_scalar_type(self) -> None:
        """A voltage that became a NumPy float is at the same state as before."""
        neuron = _instance("LapicqueNeuron")
        signature = quiescent_signature(neuron)
        neuron.v = np.float64(neuron.v)
        assert is_quiescent(neuron, signature)

    def test_a_model_that_refuses_a_zero_drive_has_no_quiescent_state(self) -> None:
        """A refusal is not quiescence."""

        class Refuses:
            """A model that rejects a zero input."""

            def __init__(self) -> None:
                self.v = 0.0

            def step(self, current: float) -> int:
                """Refuse every input."""
                raise ValueError("no")

        assert quiescent_signature(Refuses()) is None

    def test_a_model_that_spikes_at_rest_has_no_quiescent_state(self) -> None:
        """A pacemaker emits without input, so its rest is not quiescent."""

        class Pacemaker:
            """A model that fires on every step regardless of input."""

            def __init__(self) -> None:
                self.v = 0.0

            def step(self, current: float) -> int:
                """Emit unconditionally without moving any state."""
                return 1

        assert quiescent_signature(Pacemaker()) is None

    def test_an_object_that_cannot_be_stepped_has_no_quiescent_state(self) -> None:
        """Something a population cannot drive holds nothing still."""

        class NotANeuron:
            """An object with no step method."""

        assert quiescent_signature(NotANeuron()) is None

    def test_an_incomparable_attribute_blocks_skipping(self) -> None:
        """A value that cannot be compared by content never matches a signature."""

        class Opaque:
            """A value with no content-based comparison."""

        class HoldsOpaque:
            """A model carrying a value the signature cannot compare."""

            def __init__(self) -> None:
                self.v = 0.0
                self.register = Opaque()

            def step(self, current: float) -> int:
                """Hold every attribute still."""
                return 0

        neuron = HoldsOpaque()
        # Whether this model holds still cannot be established, because part of
        # its state cannot be compared by content. An unestablished fixed point
        # is not one, so it reports none and is never skipped.
        assert quiescent_signature(neuron) is None
        assert state_signature(neuron)[1][1] == INCOMPARABLE

    def test_array_and_sequence_state_compares_by_content(self) -> None:
        """A vector at the same values is at the same state."""

        class HoldsVectors:
            """A model whose state is an array and a list."""

            def __init__(self) -> None:
                self.v = np.zeros(3)
                self.history = [0.0, 0.0]

            def step(self, current: float) -> int:
                """Hold every attribute still."""
                return 0

        neuron = HoldsVectors()
        signature = quiescent_signature(neuron)
        neuron.v = np.zeros(3)
        neuron.history = [0.0, 0.0]
        assert is_quiescent(neuron, signature)
        neuron.history = [0.0, 1.0]
        assert not is_quiescent(neuron, signature)

    def test_a_model_holding_a_generator_is_never_quiescent(self) -> None:
        """A generator's identity is stable while its stream moves; refuse it."""

        class HoldsGenerator:
            """A model carrying its own random generator."""

            def __init__(self) -> None:
                self.v = 0.0
                self._rng = np.random.default_rng(3)

            def step(self, current: float) -> int:
                """Hold the voltage still while drawing from the generator."""
                self._rng.random()
                return 0

        assert quiescent_signature(HoldsGenerator()) is None

    def test_a_model_drawing_from_the_global_generator_is_never_quiescent(self) -> None:
        """A draw with no record in the instance still moves the run."""

        class DrawsGlobally:
            """A model whose noise comes from the process-wide generator."""

            def __init__(self) -> None:
                self.v = 0.0

            def step(self, current: float) -> int:
                """Draw from the global generator without recording anything."""
                np.random.randn()
                return 0

        assert quiescent_signature(DrawsGlobally()) is None

    def test_a_model_that_touches_nothing_global_is_still_quiescent(self) -> None:
        """The global-stream check must not refuse a model that draws nothing."""
        assert quiescent_signature(CountingQuiescent()) is not None

    def test_the_probe_leaves_the_neuron_untouched(self) -> None:
        """Measuring quiescence must not advance the model being measured."""
        neuron = _instance("AdExNeuron")
        before = state_signature(neuron)
        quiescent_signature(neuron)
        assert state_signature(neuron) == before

    def test_the_drive_the_fixed_point_holds_for_is_zero(self) -> None:
        """A neuron is only ever a skip candidate at exactly this input."""
        assert QUIESCENT_DRIVE == 0.0


class TestGatedRunsMatchUngatedRuns:
    @pytest.mark.parametrize("class_name", REPRESENTATIVE_MODELS)
    def test_a_neuron_parked_near_rest_relaxes_identically(self, class_name: str) -> None:
        """The defect: the old gate froze exactly these neurons."""
        assert _run(class_name, gated=True, park=True) == _run(class_name, gated=False, park=True)

    @pytest.mark.parametrize("class_name", REPRESENTATIVE_MODELS)
    def test_a_population_at_rest_runs_identically(self, class_name: str) -> None:
        """Skipping a genuine fixed point changes nothing, which is the point."""
        assert _run(class_name, gated=True, park=False) == _run(class_name, gated=False, park=False)

    def test_a_driven_population_runs_identically(self) -> None:
        """A neuron with input is never a skip candidate."""
        results = []
        for gated in (False, True):
            population = Population("AdaptiveThresholdIFNeuron", POPULATION)
            drive = np.full(POPULATION, SPIKING_DRIVE, dtype=np.float64)
            spikes = 0
            for _ in range(STEPS):
                spikes += int(population.step_all(drive, spike_gating=gated).sum())
            results.append((spikes, list(state_signature(population.neurons[0]))))
        assert results[0][0] > 0
        assert results[0] == results[1]

    def test_a_population_that_has_spiked_relaxes_identically(self) -> None:
        """After a spike a neuron is mid-recovery, and must not be frozen."""
        results = []
        for gated in (False, True):
            population = Population("AdaptiveThresholdIFNeuron", POPULATION)
            driven = np.full(POPULATION, SPIKING_DRIVE, dtype=np.float64)
            for _ in range(STEPS):
                population.step_all(driven, spike_gating=gated)
            for _ in range(STEPS):
                population.step_all(np.zeros(POPULATION), spike_gating=gated)
            results.append(list(state_signature(population.neurons[0])))
        assert results[0] == results[1]


class TestGatingStillSkips:
    def test_a_quiescent_population_is_skipped_and_a_driven_one_is_not(self) -> None:
        """Gating must still buy something exactly where it is sound."""
        population = Population(CountingQuiescent, POPULATION)
        assert population.quiescent_signature() is not None
        # The probe steps a copy, and the counter lives on the class, so it
        # counts the measurement too. Zero it once the measurement is done.
        CountingQuiescent.steps = 0

        population.step_all(np.zeros(POPULATION), spike_gating=True)
        assert CountingQuiescent.steps == 0

        population.step_all(np.full(POPULATION, 5.0), spike_gating=True)
        assert CountingQuiescent.steps == POPULATION

        # Having moved, the neurons no longer match the quiescent state, so a
        # zero drive no longer skips them.
        population.step_all(np.zeros(POPULATION), spike_gating=True)
        assert CountingQuiescent.steps == 2 * POPULATION

    def test_a_model_without_a_fixed_point_is_never_skipped(self) -> None:
        """A model that drifts under zero input must be stepped every time."""
        population = Population(CountingRestless, POPULATION)
        assert population.quiescent_signature() is None
        CountingRestless.steps = 0

        population.step_all(np.zeros(POPULATION), spike_gating=True)
        assert CountingRestless.steps == POPULATION

    def test_a_catalogue_model_without_a_fixed_point_is_never_skipped(self) -> None:
        """AdEx is the real instance of that family: its rest is not a fixed point."""
        assert Population("AdExNeuron", POPULATION).quiescent_signature() is None

    def test_the_probe_runs_once_per_population(self) -> None:
        """The measurement is cached; a run does not re-probe on every step."""
        population = Population("LapicqueNeuron", POPULATION)
        first = population.quiescent_signature()
        assert population.quiescent_signature() is first

    def test_an_empty_population_has_no_quiescent_state(self) -> None:
        """There is no neuron to measure, and nothing to skip."""
        assert Population("LapicqueNeuron", 0).quiescent_signature() is None


class TestTheWholeCatalogue:
    def test_every_model_runs_identically_gated_and_ungated(self) -> None:
        """The guarantee, on the catalogue rather than on a chosen few.

        Each model is parked just inside the band the old gate skipped and run
        with no input, which is exactly where gating used to freeze it. The
        process-wide generator is reseeded per run so a model that draws from
        it cannot make two runs differ for a reason that is not gating.
        """
        diverged: list[str] = []
        undrivable = 0
        for class_name in sorted(_CLASS_TO_MODULE):
            runs: list[list[tuple[str, str]]] = []
            try:
                for gated in (False, True):
                    np.random.seed(CATALOGUE_SEED)
                    population = Population(class_name, 2)
                    _park_near_rest(population)
                    np.random.seed(CATALOGUE_SEED)
                    for _ in range(CATALOGUE_STEPS):
                        population.step_all(np.zeros(2), spike_gating=gated)
                    runs.append(list(state_signature(population.neurons[0])))
            except Exception:  # noqa: BLE001 - a model this drive cannot run is not the subject
                undrivable += 1
                continue
            if runs[0] != runs[1]:
                diverged.append(class_name)
        assert diverged == []
        assert undrivable < len(_CLASS_TO_MODULE) // 4

    def test_gating_still_reaches_a_large_part_of_the_catalogue(self) -> None:
        """Exactness must not have cost the optimisation everywhere."""
        gateable = 0
        for class_name in sorted(_CLASS_TO_MODULE):
            try:
                population = Population(class_name, 1)
            except Exception:  # noqa: BLE001 - a model a population cannot hold is not the subject
                continue
            if population.quiescent_signature() is not None:
                gateable += 1
        assert gateable >= 50
