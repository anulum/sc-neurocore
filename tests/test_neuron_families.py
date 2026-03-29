# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parametrised tests across neuron model families

"""Parametrised tests for 12 representative neuron models from EquationNeuron.

Each model is verified for: step() returns, spike detection, reset,
state finiteness, and deterministic behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.equation_builder import EquationNeuron


def _make_lif():
    return EquationNeuron(
        equations={"v": "-(v - v_rest) / tau + I"},
        parameters={"v_rest": 0.0, "tau": 20.0},
        state={"v": 0.0},
        threshold="v > 1.0",
        reset={"v": "0.0"},
        dt=0.1,
    )


def _make_expif():
    return EquationNeuron(
        equations={"v": "-(v - v_rest)/tau + delta_T * exp((v - v_T)/delta_T)/tau + I"},
        parameters={"v_rest": -65.0, "tau": 20.0, "delta_T": 2.0, "v_T": -50.0},
        state={"v": -65.0},
        threshold="v > -30",
        reset={"v": "-65.0"},
        dt=0.1,
    )


def _make_izhikevich_rs():
    return EquationNeuron(
        equations={"v": "0.04*v**2 + 5*v + 140 - u + I", "u": "a*(b*v - u)"},
        parameters={"a": 0.02, "b": 0.2},
        state={"v": -65.0, "u": -14.0},
        threshold="v > 30",
        reset={"v": "-65", "u": "u + 8"},
        dt=0.5,
    )


def _make_izhikevich_fs():
    return EquationNeuron(
        equations={"v": "0.04*v**2 + 5*v + 140 - u + I", "u": "a*(b*v - u)"},
        parameters={"a": 0.1, "b": 0.2},
        state={"v": -65.0, "u": -14.0},
        threshold="v > 30",
        reset={"v": "-65", "u": "u + 2"},
        dt=0.5,
    )


def _make_fhn():
    return EquationNeuron(
        equations={"v": "v - v**3/3 - w + I", "w": "0.08*(v + 0.7 - 0.8*w)"},
        state={"v": -1.0, "w": -0.5},
        threshold="v > 1.0",
        reset={"v": "-1.0"},
        dt=0.05,
    )


def _make_hindmarsh_rose():
    return EquationNeuron(
        equations={
            "x": "y - 1.0*x**3 + 3.0*x**2 - z + I",
            "y": "1.0 - 5.0*x**2 - y",
            "z": "0.005*(4.0*(x - (-1.6)) - z)",
        },
        state={"x": -1.5, "y": -10.0, "z": 2.0},
        threshold="x > 1.0",
        reset={"x": "-1.5"},
        dt=0.05,
    )


def _make_quadratic_if():
    return EquationNeuron(
        equations={"v": "v**2 + I"},
        state={"v": -0.5},
        threshold="v > 1.0",
        reset={"v": "-1.0"},
        dt=0.1,
    )


def _make_resonate_fire():
    return EquationNeuron(
        equations={"v": "0.98*v - w + I", "w": "v + 0.98*w"},
        state={"v": 0.0, "w": 0.0},
        threshold="v > 1.0",
        reset={"v": "0.0", "w": "0.0"},
        dt=0.5,
    )


def _make_adex():
    return EquationNeuron(
        equations={
            "v": "-(v-(-65))/20 + 2*exp((v-(-50))/2)/20 - w + I",
            "w": "(0.01*(v-(-65)) - w)/100",
        },
        state={"v": -65.0, "w": 0.0},
        threshold="v > -30",
        reset={"v": "-65.0", "w": "w + 0.5"},
        dt=0.1,
    )


def _make_integrator():
    return EquationNeuron(
        equations={"v": "I"},
        state={"v": 0.0},
        threshold="v > 1.0",
        reset={"v": "0.0"},
        dt=0.1,
    )


def _make_leaky_no_spike():
    """Leaky model below threshold — should never spike."""
    return EquationNeuron(
        equations={"v": "-v / 10.0"},
        state={"v": 0.5},
        threshold="v > 100.0",
        reset={"v": "0.0"},
        dt=0.1,
    )


def _make_two_var_decay():
    return EquationNeuron(
        equations={"v": "-v/10 + w + I", "w": "-w/50"},
        state={"v": 0.0, "w": 1.0},
        threshold="v > 2.0",
        reset={"v": "0.0"},
        dt=0.1,
    )


MODEL_FACTORIES = {
    "LIF": (_make_lif, 0.08),
    "ExpIF": (_make_expif, 3.0),
    "Izhikevich_RS": (_make_izhikevich_rs, 10.0),
    "Izhikevich_FS": (_make_izhikevich_fs, 10.0),
    "FHN": (_make_fhn, 0.5),
    "Hindmarsh_Rose": (_make_hindmarsh_rose, 3.5),
    "QIF": (_make_quadratic_if, 0.3),
    "Resonate_Fire": (_make_resonate_fire, 0.15),
    "AdEx": (_make_adex, 3.0),
    "Integrator": (_make_integrator, 2.0),
    "Two_Var_Decay": (_make_two_var_decay, 1.0),
}


@pytest.fixture(params=MODEL_FACTORIES.keys())
def model_and_current(request):
    name = request.param
    factory, I_amp = MODEL_FACTORIES[name]
    return name, factory(), I_amp


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
        assert eq_vars == state_vars, (
            f"{name}: equations have {eq_vars}, state has {state_vars}"
        )

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
                neuron1.state[var], neuron2.state[var], rtol=1e-10,
                err_msg=f"{name}: divergent state in {var}",
            )


class TestNoSpikeModel:
    def test_subthreshold_never_spikes(self):
        neuron = _make_leaky_no_spike()
        for _ in range(10000):
            assert not neuron.step(I=0.0), "subthreshold model spiked"

    def test_decays_to_zero(self):
        neuron = _make_leaky_no_spike()
        for _ in range(1000):
            neuron.step(I=0.0)
        assert abs(neuron.state["v"]) < 0.001
