# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former verilog_compiler_contracts

from __future__ import annotations

"""Focused contracts for the equation-to-Verilog compiler edge cases."""

from pathlib import Path


import shutil


import subprocess


from typing import cast


import pytest


from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog


from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations


from sc_neurocore.neurons.universal_dsl import UniversalNeuron


def _lif_without_threshold(dt: float = 0.01) -> EquationNeuron:
    """Build a one-state equation neuron without threshold or reset rules."""
    return from_equations(
        "dv/dt = -v/tau + I",
        params={"tau": 10.0},
        init={"v": 0.0},
        dt=dt,
    )


def _candidate_reset_neuron() -> EquationNeuron:
    """Build a two-state RK4 neuron whose adaptive reset reads the candidate state."""
    return EquationNeuron(
        equations={"v": "I", "a": "-a / tau"},
        parameters={"tau": 10.0, "kick": 2.0, "v_reset": 0.0},
        state={"v": 0.0, "a": 1.0},
        threshold="v >= 1.0",
        reset={"v": "v_reset", "a": "a + kick"},
        dt=1.0,
        method="rk4",
    )


def _escape_rate_neuron() -> EquationNeuron:
    """Load the canonical stochastic schema through the production DSL."""
    return UniversalNeuron.from_schema("escape_rate").to_equation_neuron()



__all__ = ['Path', 'shutil', 'subprocess', 'cast', 'pytest', 'compile_to_datapath', 'compile_to_verilog', 'EquationNeuron', 'from_equations', 'UniversalNeuron', '_lif_without_threshold', '_candidate_reset_neuron', '_escape_rate_neuron']
