# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public numeric-contract per-call integration
"""Exercise per-call domain reporting through the normal whole-neuron numeric API."""

from pathlib import Path

import pytest

from sc_neurocore.compiler.function_call_domains import function_call_contracts
from sc_neurocore.compiler.hardware_numeric_contract import (
    HardwareNumericContract,
    hardware_numeric_contract,
)
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron


def test_public_contract_keeps_inventory_and_call_evidence_distinct() -> None:
    """One shared unit inventory no longer erases two different argument ranges."""
    model = EquationNeuron({"v": "exp(I)+exp(I/4)"}, method="map")
    contract = hardware_numeric_contract(model, QFormat(4, 4), function_lowering="v2")
    assert len(contract.function_units) == 1
    assert len(contract.function_calls) == 2
    calls = contract.to_public_dict()["function_calls"]
    assert isinstance(calls, list)
    assert [call["argument_ranges"][0]["codes"] for call in calls] == [[-128, 127], [-32, 31]]
    assert [call["accuracy"]["codes_evaluated"] for call in calls] == [256, 64]


def test_documented_function_call_domain_example_executes() -> None:
    """The maintained guide computes its stated encoded intervals and measurement counts."""
    guide = Path(__file__).resolve().parents[1] / "docs/guides/function_call_domains.md"
    example = guide.read_text().split("```python\n", 1)[1].split("\n```", 1)[0]
    namespace: dict[str, object] = {}
    exec(compile(example, str(guide), "exec"), namespace)
    assert isinstance(namespace["contract"], HardwareNumericContract)


@pytest.mark.parametrize("detection", ["poisson", "escape_rate"])
def test_intrinsic_lowering_keeps_call_sites_and_the_callers_random_stream(
    detection: str,
) -> None:
    """Noise and Philox preparation agree across public APIs without consuming draws.

    Parameters
    ----------
    detection : str
        Probability or escape-rate threshold using a real Philox stream.
    """
    models = [
        EquationNeuron(
            {"v": "exp(I/4)+xi"},
            state={"v": 0.0},
            dt=0.25,
            detection=detection,
            probability_expression="sigmoid(v)" if detection == "poisson" else None,
            rate_expression="exp(v)/4" if detection == "escape_rate" else None,
            threshold_rng="philox",
            rng_seed=17,
            random_stream=9,
            reset={"v": "tanh(I)"},
        )
        for _ in range(2)
    ]
    model, control = models
    before = (
        dict(model.equations),
        dict(model.state),
        dict(model.initial_state),
        model.threshold_expr,
        model.rate_expression,
        model.probability_expression,
        model.detection,
    )
    direct = function_call_contracts(model, 8, 4)
    contract = hardware_numeric_contract(model, QFormat(4, 4), function_lowering="v2")
    assert contract.function_calls == direct
    assert all(call.accuracy is not None and not call.refusal for call in direct)
    rtl = compile_to_verilog(model, "intrinsic_calls", 8, 4, function_lowering="v2")
    for function in {call.function for call in direct}:
        assert rtl.count(f"// v2 function unit: {function}\n") == sum(
            call.function == function for call in direct
        )
    assert len(direct) == (3 if detection == "poisson" else 4)
    assert before == (
        model.equations,
        model.state,
        model.initial_state,
        model.threshold_expr,
        model.rate_expression,
        model.probability_expression,
        model.detection,
    )
    for drive in [0.0, 0.5, -0.5]:
        assert model.step(drive) == control.step(drive)
        assert model.state == control.state


def test_numeric_state_named_after_a_function_stays_a_state() -> None:
    """A bare function-like state name is not invented as an approximation call."""
    model = EquationNeuron(
        {"exp": "I/4", "v": "exp+sin(I)"},
        state={"exp": 0.0, "v": 0.0},
        method="map",
    )
    contract = hardware_numeric_contract(model, QFormat(4, 4), function_lowering="v2")
    assert [call.function for call in contract.function_calls] == ["sin"]
    assert contract.function_calls[0].arguments == ("I",)
    rtl = compile_to_verilog(model, "lexical_calls", 8, 4, function_lowering="v2")
    assert "exp_out" in rtl and "// v2 function unit: exp\n" not in rtl
