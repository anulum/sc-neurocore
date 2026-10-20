# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — callable names and numeric equation bindings

"""Refuse compiler interpretations that disagree with actual numeric Python bindings."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.compiler.c_expr_emitter import emit_c_expr
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.c_fixed_emitter import emit_c_fixed_expr
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_refusals import EquationTypeFailure
from tests.test_function_lowering_atan2_cosim import _rust_trace
from tests.test_function_lowering_v2_cosim import _traces

_CALLS = [
    ("atan2", "I, 1"),
    ("sin", "I"),
    ("exp", "I"),
    ("abs", "I"),
    ("clip", "I, 0, 1"),
    ("max", "I, 1"),
]


@pytest.mark.parametrize("function,args", _CALLS)
@pytest.mark.parametrize("binding", ["parameter", "constant", "state"])
def test_numeric_bindings_cannot_silently_become_functions(
    function: str,
    args: str,
    binding: str,
) -> None:
    """Python's non-callable binding error has an explicit refusal in every lowerer."""
    expression = f"{function}({args})"
    equations = {"v": expression}
    state = {"v": 0.0}
    parameters: dict[str, float] = {}
    constants: dict[str, float] = {}
    if binding == "parameter":
        parameters[function] = 1.0
    elif binding == "constant":
        constants[function] = 1.0
    else:
        state[function] = 1.0
        equations[function] = "0"
    neuron = EquationNeuron(
        equations=equations,
        parameters=parameters,
        constants=constants,
        state=state,
        method="map",
        dt=1.0,
    )
    with pytest.raises(EquationTypeFailure):
        neuron.step(0.25)
    for lower in (compile_to_verilog, compile_to_datapath):
        with pytest.raises(ValueError, match="shadowed by a numeric binding"):
            lower(neuron, "m", function_lowering="v2")
    with pytest.raises(ValueError, match="shadowed by a numeric binding"):
        hardware_numeric_contract(neuron, QFormat.from_string("Q8.8"), function_lowering="v2")
    for lang in ("c", "rust"):
        with pytest.raises(ValueError, match="shadowed by a numeric binding"):
            generate_bittrue_kernel_from_neuron(neuron, "m", language=lang, function_lowering="v2")
    with pytest.raises(ValueError, match="shadowed by a numeric binding"):
        emit_c_fixed_expr(
            expression,
            {name: name for name in state},
            {name: 256 for name in parameters | constants},
            Q88(function_lowering="v2"),
        )
    with pytest.raises(ValueError, match="shadowed by a numeric binding"):
        emit_c_expr(expression, set(state), {name: f"p_{name}" for name in parameters | constants})


@pytest.mark.parametrize("function,args", _CALLS)
def test_numeric_function_names_remain_usable_as_numbers(function: str, args: str) -> None:
    """The guard applies to calls, preserving valid numeric names and unshadowed calls."""
    neuron = EquationNeuron(
        equations={"v": f"{function} + I"},
        parameters={function: 1.0},
        state={"v": 0.0},
        method="map",
        dt=1.0,
    )
    neuron.step(0.25)
    assert neuron.state["v"] == 1.25
    assert compile_to_verilog(neuron, "m", function_lowering="v2")
    for lang in ("c", "rust"):
        assert generate_bittrue_kernel_from_neuron(
            neuron, "m", language=lang, function_lowering="v2"
        )
    code, free = emit_c_expr(f"{function} + I", {"v"}, {function: "p_value"})
    assert "p_value" in code and not free
    unshadowed = EquationNeuron(
        equations={"v": f"{function}({args})"}, state={"v": 0.0}, method="map", dt=1.0
    )
    unshadowed.step(0.25)
    assert compile_to_verilog(unshadowed, "m", function_lowering="v2")


@pytest.mark.parametrize("intrinsic", ["exprel", "randn", "rand"])
def test_internal_calls_do_not_shadow_valid_numeric_names(intrinsic: str, tmp_path: Path) -> None:
    """Run injected integrator, diffusion and threshold calls on C/Rust and real RTL."""
    if intrinsic == "exprel":
        neuron = EquationNeuron(
            equations={"v": "-v + exprel + I"},
            parameters={"exprel": 0.25},
            state={"v": 0.0},
            dt=0.125,
            method="exp_euler",
        )
    elif intrinsic == "randn":
        neuron = EquationNeuron(
            equations={"v": "0.25*xi + randn"},
            parameters={"randn": 0.25},
            state={"v": 0.0},
            dt=0.25,
            rng_seed=2026,
        )
    else:
        neuron = EquationNeuron(
            equations={"v": "rand"},
            parameters={"rand": 0.25},
            state={"v": 0.0},
            dt=1.0,
            method="map",
            detection="poisson",
            probability_expression="rand",
            threshold_rng="philox",
            rng_seed=2026,
        )
    for _ in range(4):
        neuron.step(0.125)
    rtl, c = _traces(neuron, 0.125, 32, 16, 12)
    rust = _rust_trace(neuron, 0.125, 32, 16, 12, tmp_path, has_rng=intrinsic != "exprel")
    assert len(rtl) == 12 and rtl == c == rust
    assert compile_to_datapath(neuron, "m", data_width=32, fraction=16, function_lowering="v2")
    contract = hardware_numeric_contract(
        neuron, QFormat.from_string("Q16.16"), function_lowering="v2"
    )
    assert contract.arithmetic is not None and contract.mirror_refusal == ""


@pytest.mark.parametrize("location", ["threshold", "reset", "rate", "probability", "spike_value"])
def test_event_calls_respect_numeric_bindings(location: str) -> None:
    """Validate every authored event expression before intrinsic desugaring."""
    parameters = {"sin": 0.5}
    if location == "threshold":
        neuron = EquationNeuron({"v": "I"}, parameters=parameters, threshold="sin(v) > 0")
    elif location == "reset":
        neuron = EquationNeuron(
            {"v": "I"}, parameters=parameters, threshold="v > 0", reset={"v": "sin(v)"}
        )
    elif location == "rate":
        neuron = EquationNeuron(
            {"v": "I"}, parameters=parameters, detection="escape_rate", rate_expression="sin(v)"
        )
    elif location == "probability":
        neuron = EquationNeuron(
            {"v": "I"}, parameters=parameters, detection="poisson", probability_expression="sin(v)"
        )
    else:
        neuron = EquationNeuron(
            {"v": "I"}, parameters=parameters, threshold="v > 0", spike_value="sin(v)"
        )
    for lower in (compile_to_verilog, compile_to_datapath):
        with pytest.raises(ValueError, match="shadowed by a numeric binding"):
            lower(neuron, "m", function_lowering="v2")
    for language in ("c", "rust"):
        with pytest.raises(ValueError, match="shadowed by a numeric binding"):
            generate_bittrue_kernel_from_neuron(
                neuron, "m", language=language, function_lowering="v2"
            )
    with pytest.raises(ValueError, match="shadowed by a numeric binding"):
        hardware_numeric_contract(neuron, QFormat.from_string("Q8.8"), function_lowering="v2")
