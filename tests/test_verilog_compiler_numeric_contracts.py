# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog compiler numeric policy tests

"""Focused public contracts for fixed-point overflow policies."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations

Compiler = Callable[..., str]


def _signed_lif() -> EquationNeuron:
    """Build a signed neuron whose literals exercise the configured overflow path."""
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
        init={"v": -65.0},
    )


@pytest.mark.parametrize("compiler", [compile_to_verilog, compile_to_datapath])
def test_wrap_overflow_commits_raw_low_bits(compiler: Compiler) -> None:
    """Registered and folded RTL implement wrap as an exact low-word projection."""
    verilog = compiler(_signed_lif(), overflow="wrap")

    assert "v_next = v_raw[15:0];" in verilog
    assert "OVERFLOW TRAP" not in verilog


@pytest.mark.parametrize("compiler", [compile_to_verilog, compile_to_datapath])
def test_trap_overflow_emits_simulation_assertion(compiler: Compiler) -> None:
    """Both public RTL forms make signed overflow observable during simulation."""
    verilog = compiler(_signed_lif(), overflow="trap")

    assert "OVERFLOW TRAP: v_raw=%0d" in verilog
    assert "// synthesis translate_off" in verilog
    assert "// synthesis translate_on" in verilog
    assert "$fatal" in verilog
