# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cosim_emitters.py

from __future__ import annotations

"""Cross-model co-simulation infrastructure and integrator-emitter contracts."""
import pytest
from sc_neurocore.compiler.equation_compiler import generate_testbench
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _fitzhugh_nagumo_substep_neuron,
    _neuron_verilog_spike_count_q1616,
    _python_spike_count,
    _verilog_spike_count,
    _verilog_spike_count_generic,
    compile_to_verilog,
    simulate as _simulate,
    spike_count_method as _spike_count_method,
    verilog_spike_count_method as _verilog_spike_count_method,
)

_N_STEPS = 200
_INPUT_CURRENT = 50.0  # Higher than Python needs — overcomes Q8.8 precision loss
_COSIM_MODELS = [
    "lif",
    "lapicque",
    "quadratic_if",
    "izhikevich",
    "perfect_integrator",
]
_ALL_MODES = {
    "Q1.7": (8, 7),
    "Q8.8": (16, 8),
    "Q4.12": (16, 12),
    "Q1.15": (16, 15),
    "Q9.9": (18, 9),
    "Q12.12": (24, 12),
    "Q14.13": (27, 13),
    "Q20.12": (32, 12),
    "Q16.16": (32, 16),
    "Q8.24": (32, 24),
    "Q18.18": (36, 18),
}
_MV_RANGE_MODES = {
    name: spec for name, spec in _ALL_MODES.items() if -(1 << (spec[0] - 1)) / (1 << spec[1]) <= -65
}
_MV_ACCURACY_MODELS = ["lif", "lapicque"]
_RK4_EXACT_MODELS = [
    ("quadratic_if", 50.0, 300),
    ("theta", 50.0, 300),
    ("adex", 1000.0, 500),
]
_RK4_Q_FORMATS = [("Q16.16", 32, 16), ("Q12.12", 24, 12), ("Q18.18", 36, 18), ("Q20.12", 32, 12)]
_EXP_EULER_EXACT_MODELS = [
    ("lif", 50.0, 300),
    ("adex", 1000.0, 500),
    ("theta", 50.0, 300),
]
_EXP_EULER_Q_FORMATS = [
    ("Q16.16", 32, 16),
    ("Q12.12", 24, 12),
    ("Q18.18", 36, 18),
    ("Q20.12", 32, 12),
]


def _linear_oscillator(method: str) -> EquationNeuron:
    """Return a synthetic two-state oscillator for generic integrator tests."""
    return EquationNeuron(
        equations={"x": "-1.0*x - 10.0*y + I", "y": "10.0*x - 1.0*y"},
        state={"x": 0.0, "y": 0.0},
        threshold="y >= 1.0",
        reset={"x": "0.0", "y": "1.0"},
        dt=0.01,
        method=method,
        detection="crossing",
    )


def _linear_oscillator_spike_count(method: str, n_steps: int, current: float) -> int:
    """Return the Python spike count for the synthetic linear oscillator."""
    neuron = _linear_oscillator(method)
    return sum(neuron.step(I=current) for _ in range(n_steps))


def _linear_oscillator_verilog_spike_count(
    method: str,
    n_steps: int,
    current: float,
) -> int:
    """Return the Q16.16 RTL spike count for the synthetic linear oscillator."""
    neuron = _linear_oscillator(method)
    module_name = f"sc_linear_oscillator_{method}"
    verilog = compile_to_verilog(
        neuron,
        module_name=module_name,
        data_width=32,
        fraction=16,
    )
    testbench = generate_testbench(
        neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=32,
        fraction=16,
    )
    return _simulate(verilog, testbench, module_name)


def _zero_jacobian_neuron(method: str) -> EquationNeuron:
    """Return a true derivative-form perfect integrator for compiler tests."""
    return EquationNeuron(
        equations={"v": "I"},
        state={"v": 0.0},
        threshold="v >= 1.0",
        reset={"v": "0.0"},
        dt=0.1,
        method=method,
        detection="crossing",
    )


def _zero_jacobian_spike_count(method: str, n_steps: int, current: float) -> int:
    """Return the Python spike count for the zero-Jacobian fixture."""
    neuron = _zero_jacobian_neuron(method)
    return sum(neuron.step(I=current) for _ in range(n_steps))


def _zero_jacobian_verilog_spike_count(method: str, n_steps: int, current: float) -> int:
    """Return the Q16.16 RTL spike count for the zero-Jacobian fixture."""
    neuron = _zero_jacobian_neuron(method)
    module_name = f"sc_zero_jacobian_{method}"
    verilog = compile_to_verilog(
        neuron,
        module_name=module_name,
        data_width=32,
        fraction=16,
    )
    testbench = generate_testbench(
        neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=32,
        fraction=16,
    )
    return _simulate(verilog, testbench, module_name)


__all__ = [
    "pytest",
    "generate_testbench",
    "EquationNeuron",
    "HAS_IVERILOG",
    "_fitzhugh_nagumo_substep_neuron",
    "_neuron_verilog_spike_count_q1616",
    "_python_spike_count",
    "_verilog_spike_count",
    "_verilog_spike_count_generic",
    "compile_to_verilog",
    "_simulate",
    "_spike_count_method",
    "_verilog_spike_count_method",
    "_N_STEPS",
    "_INPUT_CURRENT",
    "_COSIM_MODELS",
    "_ALL_MODES",
    "_MV_RANGE_MODES",
    "_MV_ACCURACY_MODELS",
    "_RK4_EXACT_MODELS",
    "_RK4_Q_FORMATS",
    "_EXP_EULER_EXACT_MODELS",
    "_EXP_EULER_Q_FORMATS",
    "_linear_oscillator",
    "_linear_oscillator_spike_count",
    "_linear_oscillator_verilog_spike_count",
    "_zero_jacobian_spike_count",
    "_zero_jacobian_verilog_spike_count",
]
