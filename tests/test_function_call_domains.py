# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Individual function-call domain custody
"""Retain each authored and generated call through the public contract collector."""

import pytest

from sc_neurocore.compiler.function_call_domains import function_call_contracts
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron


def test_same_function_has_distinct_source_domains() -> None:
    """Two exponentials keep their distinct operands and measured sample counts."""
    model = EquationNeuron({"v": "exp(I)+exp(I/4)"}, method="map")
    calls = function_call_contracts(model, 8, 4)
    assert len(calls) == 2
    assert [call.arguments for call in calls] == [("I",), ("I / 4",)]
    assert [(call.ranges[0].minimum, call.ranges[0].maximum) for call in calls] == [
        (-128, 127),
        (-32, 31),
    ]
    assert len({call.column for call in calls}) == 2
    assert [call.accuracy.codes_evaluated if call.accuracy else None for call in calls] == [256, 64]


def test_calls_in_events_and_implicit_integrator_are_retained() -> None:
    """Jacobian, integration, threshold, reset and graded output calls have explicit origins."""
    model = EquationNeuron(
        {"v": "-v+exp(I)/4"},
        state={"v": 0.0},
        method="exp_euler",
        dt=0.125,
        threshold="exp(v)>2",
        reset={"v": "tanh(I)"},
        spike_value="sigmoid(v)",
    )
    calls = function_call_contracts(model, 8, 4)
    assert {call.location for call in calls} >= {
        "equation v",
        "threshold",
        "reset v",
        "spike value",
        "exponential Euler v",
    }
    implicit = [call for call in calls if call.implicit]
    assert len(implicit) == 1 and implicit[0].function == "exprel"
    assert implicit[0].ranges[0].minimum == implicit[0].ranges[0].maximum == -2


def test_power_domains_keep_literal_and_dynamic_exponents_separate() -> None:
    """A literal exponent is not encoded as a base-format operand."""
    model = EquationNeuron({"v": "I**1.5+(I/4)**(w/4)"}, state={"w": 0.5}, method="map")
    calls = function_call_contracts(model, 4, 2)
    assert [call.function for call in calls] == ["power", "power"]
    assert [call.exponent for call in calls] == [1.5, None]
    assert [len(call.ranges) for call in calls] == [1, 2]
    assert all(call.accuracy is not None for call in calls)
    assert [call.to_public_dict(2)["exponent"] for call in calls] == [
        {"kind": "literal", "value": 1.5},
        {"kind": "dynamic"},
    ]


def test_legacy_tables_also_retain_source_occurrences() -> None:
    """Per-call evidence does not turn the legacy default into a new lowering."""
    calls = function_call_contracts(
        EquationNeuron({"v": "exp(I)+exp(I/4)"}, method="map"), 8, 4, function_lowering="lut-v1"
    )
    assert len(calls) == 2 and all(call.accuracy is not None for call in calls)


def test_generated_exprel_does_not_duplicate_its_already_reported_jacobian() -> None:
    """Implicit-call metadata has the same exponential count as the generated datapath."""
    model = EquationNeuron({"v": "exp(v)+I"}, state={"v": 0.0}, method="exp_euler", dt=0.125)
    calls = function_call_contracts(model, 8, 4)
    assert [call.function for call in calls] == ["exp", "exp", "exprel"]
    rtl = compile_to_verilog(model, "calls", 8, 4, function_lowering="v2")
    assert rtl.count("// v2 function unit: exp\n") == 2
    assert rtl.count("// v2 function unit: exprel\n") == 1


def test_preserved_root_aliases_and_multiply_powers_have_distinct_roles() -> None:
    """Root units are reported, while a preserved integer multiply is not a function call."""
    calls = function_call_contracts(
        EquationNeuron({"v": "I**.5+I**(1/3)+I**2"}, method="map"), 8, 4
    )
    assert [call.function for call in calls] == ["sqrt", "cbrt"]


def test_model_valid_optional_output_has_an_explicit_hardware_call_refusal() -> None:
    """A model's NumPy out=None spelling does not acquire a false hardware accuracy claim."""
    model = EquationNeuron({"v": "exp(I,None)"}, method="map")
    model.step(0.0)
    assert model.state["v"] == 1.0
    (call,) = function_call_contracts(model, 8, 4)
    assert call.accuracy is None and call.refusal
    assert call.to_public_dict(4)["accuracy"] is None
    with pytest.raises(ValueError):
        compile_to_verilog(model, data_width=8, fraction=4, function_lowering="v2")


def test_unknown_lowering_is_refused_by_the_public_collector() -> None:
    """No call metadata silently substitutes another arithmetic engine."""
    with pytest.raises(ValueError, match="lowering"):
        function_call_contracts(EquationNeuron({"v": "exp(I)"}), 8, 4, function_lowering="unknown")
