# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopulation from former test_network_basic.py

"""Focused suite: TestPopulation from former test_network_basic.py."""

from __future__ import annotations

from tests.network_basic_support import *  # noqa: F403


class TestPopulation:
    def test_create_by_string(self):
        pop = Population("LapicqueNeuron", 5)
        assert pop.n == 5
        assert pop.label == "LapicqueNeuron"

    def test_create_by_class(self):
        from sc_neurocore.neurons.models import AdExNeuron

        pop = Population(AdExNeuron, 3, label="exc")
        assert pop.n == 3
        assert pop.label == "exc"

    def test_unknown_model_string(self):
        with pytest.raises(ValueError, match="Unknown model"):
            Population("NonexistentNeuron", 2)

    def test_step_all_returns_spike_vector(self):
        pop = Population("LapicqueNeuron", 4)
        currents = np.array([10.0, 0.0, 10.0, 0.0])
        spikes = pop.step_all(currents)
        assert spikes.shape == (4,)
        assert spikes.dtype == np.int8

    def test_reset_all(self):
        pop = Population("LapicqueNeuron", 3)
        pop.step_all(np.array([100.0, 100.0, 100.0]))
        pop.reset_all()
        assert np.allclose(pop.voltages, 0.0)

    def test_get_states(self):
        pop = Population("LapicqueNeuron", 3)
        states = pop.get_states()
        assert "v" in states
        assert states["v"].shape == (3,)

    def test_get_states_uses_neuron_get_state_when_available(self):
        # Neuron models exposing get_state() drive the state keys directly,
        # rather than the dataclass-field fallback used for plain dataclasses.
        from sc_neurocore.neurons.models import Izhikevich2007Neuron

        pop = Population(Izhikevich2007Neuron, 3)
        states = pop.get_states()
        assert "v" in states and "u" in states
        assert states["v"].shape == (3,)

    def test_params_override(self):
        pop = Population("LapicqueNeuron", 2, params={"v_threshold": 0.5})
        assert pop.neurons[0].v_threshold == 0.5

    def test_spike_gating_skips_resting_silent_neuron_and_steps_active_neuron(self):
        class GatedNeuron:
            def __init__(self):
                self.v = 0.0
                self.v_rest = 0.0
                self.v_threshold = 1.0
                self.step_calls = 0

            def step(self, current):
                self.step_calls += 1
                self.v += current
                return self.v >= self.v_threshold

        pop = Population(GatedNeuron, n=2, label="gated")

        spikes = pop.step_all(np.array([0.0, 1.25]), spike_gating=True)

        assert pop.neurons[0].step_calls == 0
        assert pop.neurons[0].v == 0.0
        assert pop.voltages[0] == 0.0
        assert pop.neurons[1].step_calls == 1
        assert pop.neurons[1].v == 1.25
        assert pop.voltages[1] == 1.25
        np.testing.assert_array_equal(spikes, np.array([0, 1], dtype=np.int8))

    def test_get_states_falls_back_to_voltage_for_minimal_neuron(self):
        class MinimalVoltageNeuron:
            def __init__(self):
                self.v = -0.25

        pop = Population(MinimalVoltageNeuron, n=3, label="minimal")
        for neuron, voltage in zip(pop.neurons, [-0.5, 0.0, 0.75], strict=True):
            neuron.v = voltage

        states = pop.get_states()

        assert set(states) == {"v"}
        np.testing.assert_allclose(states["v"], np.array([-0.5, 0.0, 0.75]))

    def test_reset_all_prefers_reset_and_updates_voltage_cache(self):
        class ResetNeuron:
            def __init__(self):
                self.v = 1.0
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1
                self.v = -0.125

        pop = Population(ResetNeuron, n=2, label="reset")
        pop.neurons[0].v = 0.5
        pop.neurons[1].v = 0.75

        pop.reset_all()

        assert [neuron.reset_calls for neuron in pop.neurons] == [1, 1]
        np.testing.assert_allclose(pop.voltages, np.array([-0.125, -0.125]))

    def test_reset_all_uses_reset_state_when_reset_is_unavailable(self):
        class ResetStateNeuron:
            def __init__(self):
                self.v = 1.0
                self.reset_state_calls = 0

            def reset_state(self):
                self.reset_state_calls += 1
                self.v = -0.375

        pop = Population(ResetStateNeuron, n=2, label="reset_state")
        pop.neurons[0].v = 0.5
        pop.neurons[1].v = 0.75

        pop.reset_all()

        assert [neuron.reset_state_calls for neuron in pop.neurons] == [1, 1]
        np.testing.assert_allclose(pop.voltages, np.array([-0.375, -0.375]))

    def test_get_states_uses_dataclass_fields_without_timestep_parameter(self):
        @dataclass
        class DataclassStateNeuron:
            v: float = -0.5
            adaptation: float = 0.25
            dt: float = 0.001

        pop = Population(
            DataclassStateNeuron,
            n=2,
            params={"v": -0.4, "adaptation": 0.125, "dt": 0.002},
            label="dataclass",
        )
        pop.neurons[1].v = 0.6
        pop.neurons[1].adaptation = 0.5

        states = pop.get_states()

        assert set(states) == {"v", "adaptation"}
        np.testing.assert_allclose(states["v"], np.array([-0.4, 0.6]))
        np.testing.assert_allclose(states["adaptation"], np.array([0.125, 0.5]))

    def test_empty_population_exposes_empty_state_mapping(self):
        pop = Population("LapicqueNeuron", n=0, label="empty")

        assert pop.n == 0
        assert pop.get_states() == {}
