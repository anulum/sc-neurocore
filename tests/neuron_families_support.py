# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_neuron_families.py

from __future__ import annotations

"""Parametrised tests for 12 representative neuron models from EquationNeuron.

Each model is verified for: step() returns, spike detection, reset,
state finiteness, and deterministic behaviour.
"""
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

__all__ = ['np', 'pytest', 'EquationNeuron', '_make_lif', '_make_expif', '_make_izhikevich_rs', '_make_izhikevich_fs', '_make_fhn', '_make_hindmarsh_rose', '_make_quadratic_if', '_make_resonate_fire', '_make_adex', '_make_integrator', '_make_leaky_no_spike', '_make_two_var_decay', 'MODEL_FACTORIES', 'model_and_current']
