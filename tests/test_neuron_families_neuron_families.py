# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuronFamilies from former test_neuron_families.py

"""Focused suite: TestNeuronFamilies from former test_neuron_families.py."""

from __future__ import annotations

from tests.neuron_families_support import *  # noqa: F403

class TestNeuronFamilies:
    def test_step_returns_bool_or_int(self, model_and_current):
        name, neuron, I = model_and_current
        result = neuron.step(I=I)
        assert isinstance(result, (bool, int, np.bool_, np.integer)), (
            f"{name}: step() returned {type(result)}"
        )

    def test_state_stays_finite(self, model_and_current):
        name, neuron, I = model_and_current
        for _ in range(500):
            neuron.step(I=I)
        for var, val in neuron.state.items():
            assert np.isfinite(val), f"{name}: state {var} = {val} is not finite"

    def test_spike_occurs_under_drive(self, model_and_current):
        name, neuron, I = model_and_current
        found_spike = False
        for _ in range(5000):
            if neuron.step(I=I):
                found_spike = True
                break
        assert found_spike, f"{name}: no spike in 5000 steps at I={I}"

    def test_state_dict_keys_match_equations(self, model_and_current):
        name, neuron, I = model_and_current
        eq_vars = set(neuron.equations.keys())
        state_vars = set(neuron.state.keys())
        assert eq_vars == state_vars, f"{name}: equations have {eq_vars}, state has {state_vars}"

    def test_deterministic(self, model_and_current):
        """Same initial state + same input → same result."""
        name, factory_fn_and_I = model_and_current[0], model_and_current[1:]
        neuron1 = MODEL_FACTORIES[name][0]()
        neuron2 = MODEL_FACTORIES[name][0]()
        I = MODEL_FACTORIES[name][1]
        for _ in range(100):
            s1 = neuron1.step(I=I)
            s2 = neuron2.step(I=I)
            assert s1 == s2, f"{name}: non-deterministic at same input"
        for var in neuron1.state:
            np.testing.assert_allclose(
                neuron1.state[var],
                neuron2.state[var],
                rtol=1e-10,
                err_msg=f"{name}: divergent state in {var}",
            )
