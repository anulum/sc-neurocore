# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Variants of a model with the sign of one term changed must fail a behaviour case

"""Bind the variants of a schema and what the behaviour cases say about each.

A variant is a model's schema with the sign of one term of one equation
changed. The cases of the model must tell every variant from the model: a
variant that holds every case names a term whose sign no case decides. The
schemas of these tests are small real ones, run by the Universal DSL as any
schema is, and the catalogue's own report is built once.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import re
import runpy
import sys

import pytest

from sc_neurocore.neurons.model_descriptor import BehaviourCase
from sc_neurocore.studio.candidate_catalogue import candidate_from_catalogue
from tools.model_behaviour_report import MAX_CASE_STEPS
from tools.model_behaviour_variants import (
    DEFAULT_OUTPUT,
    MODEL_STATES,
    REPORT_NOTES,
    REPORT_SCHEMA,
    build_report,
    check_problems,
    encode,
    equation_variants,
    evaluate_schema,
    main,
    render_summary,
    report_of,
)

ROOT = Path(__file__).resolve().parents[2]

#: Models of the catalogue that state behaviour cases. It moves only by a
#: deliberate change here, in the unit that adds or removes cases.
MODELS_WITH_CASES = 22


def _accumulator() -> dict[str, object]:
    """Return a schema that halves its level each step and adds a quarter of the current.

    With a current of 1 the level settles at 0.5 and never reaches the
    threshold of 1. With a current of 3 the level is 0.75, then 1.125: it
    fires on every second step.
    """
    return {
        "metadata": {"schema_version": 2, "name": "HalvingAccumulator"},
        "state": {"v": 0.0},
        "parameters": {"leak": 0.5, "gain": 0.25, "v_threshold": 1.0, "v_reset": 0.0, "dt": 1.0},
        "integration": {"dt": 1.0, "method": "map"},
        "dynamics": {"v": "v - leak * v + gain * I"},
        "threshold": {"condition": "v >= v_threshold", "detection": "level"},
        "reset": {"v": "v_reset"},
    }


def _two_equations() -> dict[str, object]:
    """Return the accumulator with a second state that no spike depends on."""
    schema = _accumulator()
    schema["state"] = {"v": 0.0, "trace": 0.0}
    schema["dynamics"] = {"v": "v - leak * v + gain * I", "trace": "trace + 0.0 * v"}
    return schema


def _spikes(name: str, current: float, bounds: tuple[float, float]) -> BehaviourCase:
    return BehaviourCase(
        name=name,
        kind="interval",
        steps=100,
        observable="spikes",
        inputs=(("current", current),),
        bounds=bounds,
    )


_SETTLES = BehaviourCase(
    name="a current of 1 settles the level at a half",
    kind="interval",
    steps=100,
    observable="state:v",
    inputs=(("current", 1.0),),
    bounds=(0.49, 0.51),
)
_FIRES = _spikes("a current of 3 fires on every second step", 3.0, (50, 50))
_SILENT = _spikes("without a current nothing fires", 0.0, (0, 0))


def test_every_term_of_a_sum_is_negated_in_turn() -> None:
    """Four terms give four variants; each differs from the equation in one sign."""
    variants = equation_variants("v - v * v * v / 3.0 - w + I")
    assert [variant["expression"] for variant in variants] == [
        "-v - v * v * v / 3.0 - w + I",
        "v + v * v * v / 3.0 - w + I",
        "v - v * v * v / 3.0 + w + I",
        "v - v * v * v / 3.0 - w - I",
    ]
    assert [variant["term"] for variant in variants] == [
        "+ v",
        "- v * v * v / 3.0",
        "- w",
        "+ I",
    ]
    assert all(variant["within"] == "v - v * v * v / 3.0 - w + I" for variant in variants)


def test_a_sum_inside_a_factor_is_varied_and_the_whole_side_is_negated_once() -> None:
    """A right-hand side that is one product gets its inner terms and its own sign."""
    variants = equation_variants("epsilon * (v + a - b * w)")
    assert [variant["expression"] for variant in variants] == [
        "epsilon * (-v + a - b * w)",
        "epsilon * (v - a - b * w)",
        "epsilon * (v + a + b * w)",
        "-(epsilon * (v + a - b * w))",
    ]
    assert [variant["within"] for variant in variants[:3]] == ["v + a - b * w"] * 3
    assert variants[3]["term"] == "the whole right-hand side"


def test_sums_inside_a_function_call_and_a_quotient_are_found() -> None:
    """Every sum of the adaptive exponential equation is varied: ten terms in five sums."""
    equation = "(-(v - v_rest) + delta_T * exp((v - v_rh) / delta_T)) / tau + (-w + I) / C"
    variants = equation_variants(equation)
    assert len(variants) == 10
    assert len({variant["expression"] for variant in variants}) == 10
    assert all(variant["expression"] != equation for variant in variants)
    assert [variant["within"] for variant in variants[4:8]] == [
        "v - v_rest",
        "v - v_rest",
        "v - v_rh",
        "v - v_rh",
    ]
    assert "exp((-v - v_rh) / delta_T)" in variants[6]["expression"]
    assert variants[8]["term"] == "+ -w"
    assert "(--w + I) / C" in variants[8]["expression"]


def test_a_term_that_is_zero_has_no_negation() -> None:
    """Negating zero changes nothing, so it is no variant."""
    assert equation_variants("0.0") == []
    assert [variant["expression"] for variant in equation_variants("0 + x")] == ["0 - x"]


def test_a_single_term_is_negated_as_a_whole() -> None:
    """An equation without any sum still has one sign."""
    (variant,) = equation_variants("v ** 2")
    assert variant["expression"] == "-v ** 2"
    assert variant["term"] == "the whole right-hand side"


def test_cases_that_tell_every_variant_from_the_schema() -> None:
    """Each changed sign breaks a case; the first broken case is named with what was observed."""
    result = evaluate_schema(_accumulator(), (_SETTLES, _FIRES))
    assert result["state"] == "discriminated"
    assert [entry["observed"] for entry in result["unchanged"]] == [0.5, 50]
    assert all(entry["held"] for entry in result["unchanged"])
    variants = result["variants"]
    assert [variant["term"] for variant in variants] == ["+ v", "- leak * v", "+ gain * I"]
    assert all(variant["outcome"] == "violated" for variant in variants)
    grows, falls = variants[1], variants[2]
    assert grows["expression"] == "v + leak * v + gain * I"
    assert (grows["case"], grows["observed"]) == (_SETTLES.name, 0.25)
    assert falls["expression"] == "v - leak * v - gain * I"
    assert (falls["case"], falls["observed"]) == (_SETTLES.name, -0.5)


def test_the_first_case_a_variant_breaks_is_the_one_named() -> None:
    """Cases run in the descriptor's order and a variant stops at its first broken case."""
    result = evaluate_schema(_accumulator(), (_SILENT, _FIRES, _SETTLES))
    falls = result["variants"][2]
    assert (falls["case"], falls["observed"]) == (_FIRES.name, 0)


def test_cases_at_rest_tell_no_variant_from_the_schema() -> None:
    """With no current the level stays at zero whatever the signs: no term is decided."""
    result = evaluate_schema(_accumulator(), (_SILENT,))
    assert result["state"] == "not_discriminated"
    assert [variant["outcome"] for variant in result["variants"]] == ["not_discriminated"] * 3
    assert all("case" not in variant for variant in result["variants"])


def test_one_term_no_case_decides_is_named() -> None:
    """A case on the spike count at a strong current does not see the sign of the first term."""
    result = evaluate_schema(
        _accumulator(), (_spikes("a current of 8 fires at every step", 8.0, (100, 100)),)
    )
    assert result["state"] == "not_discriminated"
    assert [variant["outcome"] for variant in result["variants"]] == [
        "not_discriminated",
        "not_discriminated",
        "violated",
    ]


def test_a_contrast_is_run_on_the_schema_at_both_drives() -> None:
    """The margin between two currents holds for the schema and breaks when the input's sign changes."""
    contrast = BehaviourCase(
        name="more current, more spikes",
        kind="contrast",
        steps=100,
        observable="spikes",
        inputs=(("current", 3.0),),
        reference_inputs=(("current", 1.0),),
        margin=40.0,
    )
    result = evaluate_schema(_accumulator(), (contrast,))
    assert result["unchanged"] == [
        {"name": contrast.name, "observed": 50, "reference_observed": 0, "held": True}
    ]
    falls = result["variants"][2]
    assert (falls["outcome"], falls["observed"], falls["reference_observed"]) == ("violated", 0, 0)


def test_the_schema_it_is_given_is_left_as_it_was() -> None:
    """Variants are copies: the caller's schema and each other's equations are not touched."""
    schema = _two_equations()
    before = copy.deepcopy(schema)
    result = evaluate_schema(schema, (_spikes("a current of 3 fires", 3.0, (50, 50)),))
    assert schema == before
    assert [variant["equation"] for variant in result["variants"]] == [
        "v",
        "v",
        "v",
        "trace",
        "trace",
    ]
    assert result["variants"][3]["expression"] == "-trace + 0.0 * v"
    # With the leak's sign changed the level is 0.75, then 1.875: it still fires on every second
    # step, so the spike count alone does not decide that sign. The trace decides no spike at all.
    assert [variant["outcome"] for variant in result["variants"]] == [
        "violated",
        "not_discriminated",
        "violated",
        "not_discriminated",
        "not_discriminated",
    ]
    assert (result["variants"][0]["observed"], result["variants"][2]["observed"]) == (33, 0)


def test_a_run_that_stops_is_reported_without_a_half_result() -> None:
    """The first drive of a contrast ran; the reference drive overflowed: nothing observed is kept."""
    schema: dict[str, object] = {
        "metadata": {"schema_version": 2, "name": "Multiplier"},
        "state": {"v": 1.0},
        "parameters": {"v_threshold": 2.0, "dt": 1.0},
        "integration": {"dt": 1.0, "method": "map"},
        "dynamics": {"v": "v * I"},
        "threshold": {"condition": "v < 0.0", "detection": "level"},
    }
    contrast = BehaviourCase(
        name="a unit current keeps the level",
        kind="contrast",
        steps=5,
        observable="state:v",
        inputs=(("current", 1.0),),
        reference_inputs=(("current", 1e200),),
        margin=0.5,
    )
    result = evaluate_schema(schema, (contrast,))
    assert result["state"] == "schema_differs"
    assert result["unchanged"] == [
        {"name": contrast.name, "stopped": "EquationStateFailure", "held": False}
    ]


def test_a_schema_that_does_not_hold_the_cases_is_reported_before_any_variant() -> None:
    """The unchanged schema is the first variant: when it fails, the schema and the class differ."""
    result = evaluate_schema(_accumulator(), (_spikes("fires rarely", 3.0, (10, 20)),))
    assert result["state"] == "schema_differs"
    assert result["unchanged"] == [{"name": "fires rarely", "observed": 50, "held": False}]
    assert result["variants"] == []


_NOT_DRIVABLE = {
    "a drive of two inputs": (
        BehaviourCase(
            name="both inputs",
            kind="interval",
            steps=100,
            observable="spikes",
            inputs=(("glutamate", 1.0), ("i_soma", 20.0)),
            bounds=(50, 70),
        ),
        "case 'both inputs' is not driven by the one input 'current'",
    ),
    "one input of another name": (
        BehaviourCase(
            name="another input",
            kind="contrast",
            steps=100,
            observable="spikes",
            inputs=(("drive", 3.0),),
            reference_inputs=(("drive", 1.0),),
            margin=1.0,
        ),
        "case 'another input' is not driven by the one input 'current'",
    ),
    "a state variable the schema does not hold": (
        BehaviourCase(
            name="calcium",
            kind="interval",
            steps=100,
            observable="state:ca",
            inputs=(("current", 1.0),),
            bounds=(0.0, 1.0),
        ),
        "case 'calcium' observes 'ca', which the schema does not hold",
    ),
    "more steps than a run may take": (
        BehaviourCase(
            name="long",
            kind="interval",
            steps=MAX_CASE_STEPS + 1,
            observable="spikes",
            inputs=(("current", 1.0),),
            bounds=(0, 0),
        ),
        f"case 'long' asks for more than {MAX_CASE_STEPS} steps",
    ),
}


@pytest.mark.parametrize("why", sorted(_NOT_DRIVABLE))
def test_cases_the_schema_cannot_be_driven_by_are_named(why: str) -> None:
    """A schema takes one current; a case of another shape has no variants, and the reason is given."""
    case, reason = _NOT_DRIVABLE[why]
    result = evaluate_schema(_accumulator(), (_SETTLES, case))
    assert result == {
        "state": "cases_not_drivable",
        "reason": reason,
        "unchanged": [],
        "variants": [],
    }


def test_a_schema_with_no_term_to_negate_has_no_variant() -> None:
    """An equation that is zero has no sign to change."""
    schema = _accumulator()
    schema["dynamics"] = {"v": "0.0"}
    result = evaluate_schema(schema, (_SILENT,))
    assert result["state"] == "no_variant"
    assert result["variants"] == []


def test_a_schema_the_engine_refuses_is_not_executable() -> None:
    """A descriptive record of the catalogue is refused by the engine itself, with its reason."""
    schema = json.loads(candidate_from_catalogue("BendaHerzNeuron")["candidate_text"])["model"]
    result = evaluate_schema(schema, (_SILENT,))
    assert result["state"] == "schema_not_executable"
    assert "descriptive record" in result["reason"]
    assert result["variants"] == []


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    """Build the report once from the shipped descriptors and schemas."""
    return build_report()


def _rows(report: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = report["models"]
    assert isinstance(rows, list)
    return {str(row["model"]): row for row in rows}


def test_the_committed_report_is_what_the_tool_computes(report: dict[str, object]) -> None:
    """A stale report cannot stay committed."""
    assert (ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8") == encode(report)
    assert report["schema_version"] == REPORT_SCHEMA
    assert report["notes"] == list(REPORT_NOTES)


def test_the_report_lists_the_models_that_state_cases(report: dict[str, object]) -> None:
    """Only a model with cases has a row; the others are counted."""
    rows = _rows(report)
    assert sorted(rows) == [
        "ChialvoMapNeuron",
        "CourageNekorkinMapNeuron",
        "DendriticNMDANeuron",
        "ErmentroutKopellMapNeuron",
        "ExpIFNeuron",
        "FitzHughNagumoNeuron",
        "FitzHughRinzelNeuron",
        "HindmarshRoseNeuron",
        "IbarzTanakaMapNeuron",
        "IntegerQIFNeuron",
        "MihalasNieburNeuron",
        "PernarowskiNeuron",
        "RulkovMapNeuron",
        "SCAdaptiveThresholdMapNeuron",
        "SCClippedLogisticBurstingMapNeuron",
        "SCFourStateGLIFNeuron",
        "SCInclusivePerfectIntegratorNeuron",
        "SCResettingWilsonHRNeuron",
        "SCSymmetricQuadraticIFNeuron",
        "SCUpwardCrossingRulkovMapNeuron",
        "SigmaDeltaNeuron",
        "WilsonHRNeuron",
    ]
    assert report["models_with_cases"] == MODELS_WITH_CASES
    assert report["models_without_cases"] == 185 - MODELS_WITH_CASES
    totals = report["totals"]
    assert isinstance(totals, dict)
    assert set(totals) == set(MODEL_STATES)
    assert totals == {
        "discriminated": 21,
        "not_discriminated": 0,
        "schema_differs": 0,
        "no_schema": 1,
        "schema_not_executable": 0,
        "cases_not_drivable": 0,
        "no_variant": 0,
    }


def test_every_variant_of_the_fitzhugh_nagumo_model_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Eight signs, eight variants, and the cases of the model's band tell each from the model."""
    row = _rows(report)["FitzHughNagumoNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 26, 28, 28, 1]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ v"),
        ("v", "- v * v * v / 3.0"),
        ("v", "- w"),
        ("v", "+ I"),
        ("w", "+ v"),
        ("w", "+ a"),
        ("w", "- b * w"),
        ("w", "the whole right-hand side"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)
    without_input = variants[3]
    assert without_input["expression"] == "v - v * v * v / 3.0 - w - I"
    assert without_input["case"] == "a current of 0.5 inside the oscillatory band fires 26 times"
    assert without_input["observed"] == 0
    runaway = variants[1]
    assert runaway["case"] == "without a current the model fires at most once"
    assert runaway["stopped"] == "EquationStateFailure"
    assert "observed" not in runaway


def test_every_variant_of_the_hindmarsh_rose_model_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Thirteen signs over three equations, and the enrolled counts tell each from the model.

    The schema also gives the 1 event in 2000 steps at 20.0 that tells the
    crossing's direction; no variant needs it.
    """
    row = _rows(report)["HindmarshRoseNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 0, 26, 40, 52, 1]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("x", "+ y"),
        ("x", "- x ** 3"),
        ("x", "+ b * x ** 2"),
        ("x", "- z"),
        ("x", "+ I"),
        ("y", "+ 1"),
        ("y", "- 5 * x ** 2"),
        ("y", "- y"),
        ("z", "+ s * (x - x_rest)"),
        ("z", "- z"),
        ("z", "+ x"),
        ("z", "- x_rest"),
        ("z", "the whole right-hand side"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)
    # Five signs of the fast subsystem send the state past the finite numbers.
    assert [index for index, variant in enumerate(variants) if "stopped" in variant] == [
        0,
        1,
        2,
        6,
        7,
    ]
    without_input = variants[4]
    assert without_input["case"] == "a current of 3.0 bursts with 26 upward crossings in 2000 steps"
    assert without_input["observed"] == 0


def test_every_variant_of_the_wilson_hr_model_breaks_a_case(report: dict[str, object]) -> None:
    """Every sign of the two polynomial equations is decided by the enrolled counts.

    The schema also gives the 1 event in 20 steps at 0.1 that tells the
    crossing's direction; no variant needs it.
    """
    row = _rows(report)["WilsonHRNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 44, 46, 49, 52, 1]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)"),
        ("v", "- 26.0 * r * (v + 0.92)"),
        ("v", "+ I"),
        ("v", "+ 17.81"),
        ("v", "+ 47.71 * v"),
        ("v", "+ 32.63 * v * v"),
        ("v", "+ v"),
        ("v", "- 0.55"),
        ("v", "+ v"),
        ("v", "+ 0.92"),
        ("v", "the whole right-hand side"),
        ("r", "+ -r"),
        ("r", "+ 1.35 * v"),
        ("r", "+ 1.03"),
        ("r", "the whole right-hand side"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)
    # Three signs leave the state finite and silence the model at a current of 0.08.
    silent = [index for index, variant in enumerate(variants) if "stopped" not in variant]
    assert silent == [2, 7, 8]
    assert {variants[index]["observed"] for index in silent} == {0}


def test_both_signs_of_the_symmetric_quadratic_if_model_are_decided(
    report: dict[str, object],
) -> None:
    """The square and the input each change the enrolled counts when their sign is flipped."""
    row = _rows(report)["SCSymmetricQuadraticIFNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 2, 3, 6, 11, 26, 100, 250]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ v ** 2"),
        ("v", "+ I"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)


def test_every_variant_of_the_resetting_wilson_model_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Fourteen signs of the resetting Wilson equations, each decided by the three enrolled counts."""
    row = _rows(report)["SCResettingWilsonHRNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 1, 4]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)"),
        ("v", "- 26.0 * r * (v + 0.92)"),
        ("v", "+ I"),
        ("v", "+ 17.81"),
        ("v", "+ 47.71 * v"),
        ("v", "+ 32.63 * v * v"),
        ("v", "+ v"),
        ("v", "- 0.55"),
        ("v", "+ v"),
        ("v", "+ 0.92"),
        ("r", "+ -r"),
        ("r", "+ 1.35 * v"),
        ("r", "+ 1.03"),
        ("r", "the whole right-hand side"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)


def test_every_variant_of_the_integer_qif_model_breaks_its_case(
    report: dict[str, object],
) -> None:
    """Eight signs of the integer QIF map, each decided by its one enrolled count."""
    row = _rows(report)["IntegerQIFNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [26]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ v"),
        ("v", "+ a * (v_rest - v) // 8 if v < branch_point else b * (v - v_threshold) // 8"),
        ("v", "+ I"),
        ("v", "+ v_rest"),
        ("v", "- v"),
        ("v", "+ v"),
        ("v", "- v_threshold"),
        ("v", "the whole right-hand side"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)


def test_every_variant_of_the_four_state_glif_model_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Fifteen signs of the four GLIF equations, each decided by the six enrolled counts."""
    row = _rows(report)["SCFourStateGLIFNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 0, 23, 54, 86, 95]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ -(v - v_rest)"),
        ("v", "+ resistance * I"),
        ("v", "+ i_asc1"),
        ("v", "+ i_asc2"),
        ("v", "+ v"),
        ("v", "- v_rest"),
        ("v", "the whole right-hand side"),
        ("theta", "+ theta_inf"),
        ("theta", "- theta"),
        ("theta", "+ a_theta * (v - v_rest)"),
        ("theta", "+ v"),
        ("theta", "- v_rest"),
        ("theta", "the whole right-hand side"),
        ("i_asc1", "the whole right-hand side"),
        ("i_asc2", "the whole right-hand side"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)


def test_every_variant_of_the_ibarz_tanaka_map_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Twenty-six signs of the piecewise map, decided by two enrolled and two measured counts."""
    row = _rows(report)["IbarzTanakaMapNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 3, 7, 18]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ -1.0"),
        ("v", "- alpha / 2.0"),
        ("v", "+ -(alpha * alpha) / 4.0"),
        ("v", "- alpha"),
        ("v", "+ I"),
        ("v", "+ u"),
        ("v", "+ alpha * v"),
        ("v", "+ (v + 1.0) * (v + 1.0)"),
        ("v", "+ I"),
        ("v", "+ u"),
        ("v", "+ v"),
        ("v", "+ 1.0"),
        ("v", "+ v"),
        ("v", "+ 1.0"),
        ("v", "+ 1.0"),
        ("v", "+ I"),
        ("v", "+ u"),
        ("v", "+ 1.0"),
        ("v", "+ I"),
        ("v", "+ u"),
        ("v", "the whole right-hand side"),
        ("u", "+ u"),
        ("u", "- mu * (v + 1.0 - sigma)"),
        ("u", "+ v"),
        ("u", "+ 1.0"),
        ("u", "- sigma"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)


def test_every_variant_of_the_rulkov_map_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Seventeen signs of the rational map, decided by three enrolled and three measured counts."""
    row = _rows(report)["RulkovMapNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 7, 10, 0, 34, 130]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("x", "+ alpha / (1.0 - x)"),
        ("x", "+ y"),
        ("x", "+ I"),
        ("x", "+ 1.0"),
        ("x", "- x"),
        ("x", "+ alpha"),
        ("x", "+ y"),
        ("x", "+ I"),
        ("x", "+ alpha"),
        ("x", "+ y"),
        ("x", "+ I"),
        ("x", "the whole right-hand side"),
        ("y", "+ y"),
        ("y", "- mu * (x + 1.0)"),
        ("y", "+ mu * sigma"),
        ("y", "+ x"),
        ("y", "+ 1.0"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)


def test_every_variant_of_the_courbage_nekorkin_map_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Nineteen signs of the piecewise map; the upper branch's two are decided by its state.

    The sign of x in the upper branch, and of m0 in the denominator of its
    breakpoint, change nothing while x stays below that breakpoint, and no count
    over the enrolled windows reaches the branch's return. After 41 steps at a
    current of 0.1 the model sits on the upper branch near 0.555; those two
    variants leave it, so the state case breaks them.
    """
    row = _rows(report)["CourageNekorkinMapNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 0, 1, 1, 3, 0.555102]
    variants = row["variants"]
    assert isinstance(variants, list)
    branch = (
        "-m0 * x if x <= a * m1 / (m0 + m1) else m1 * (x - a) if x < (m0 + a * m1) / (m0 + m1) "
        "else -m0 * (x - 1.0)"
    )
    assert [
        (variant["equation"], variant["term"], variant["within"][:12]) for variant in variants
    ] == [
        ("x", "+ x", "x + (-m0 * x"),
        ("x", f"+ {branch}", "x + (-m0 * x"),
        ("x", "- y", "x + (-m0 * x"),
        ("x", "- beta * (1.0 if x >= d else 0.0)", "x + (-m0 * x"),
        ("x", "+ I", "x + (-m0 * x"),
        ("x", "+ m0", "m0 + m1"),
        ("x", "+ m1", "m0 + m1"),
        ("x", "+ m0", "m0 + a * m1"),
        ("x", "+ a * m1", "m0 + a * m1"),
        ("x", "+ m0", "m0 + m1"),
        ("x", "+ m1", "m0 + m1"),
        ("x", "+ x", "x - a"),
        ("x", "- a", "x - a"),
        ("x", "+ x", "x - 1.0"),
        ("x", "- 1.0", "x - 1.0"),
        ("y", "+ y", "y + eps * (x"),
        ("y", "+ eps * (x - j)", "y + eps * (x"),
        ("y", "+ x", "x - j"),
        ("y", "- j", "x - j"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)
    upper = "after 41 steps at a current of 0.1 x is on the upper branch"
    assert [index for index, variant in enumerate(variants) if variant["case"] == upper] == [9, 13]


def test_every_variant_of_the_fitzhugh_rinzel_model_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Thirteen signs of three equations; the ultra-slow variable's four need the long count.

    Over 3000 steps y moves by about a hundredth, so the enrolled counts leave
    the sign of y in the fast equation and three signs of y's own equation
    undecided. Over 39600 steps at 0.4 each of those variants fires a different
    number of times than the model's 83. The schema also gives the 1 event in
    10000 steps at 2.0 that tells the crossing's direction; no variant needs it.
    """
    row = _rows(report)["FitzHughRinzelNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [7, 8, 8, 83, 1]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ v"),
        ("v", "- v * v * v / 3.0"),
        ("v", "- w"),
        ("v", "+ y"),
        ("v", "+ I"),
        ("w", "+ a"),
        ("w", "+ v"),
        ("w", "- b * w"),
        ("w", "the whole right-hand side"),
        ("y", "+ c"),
        ("y", "- v"),
        ("y", "- d * y"),
        ("y", "the whole right-hand side"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)
    long_count = "a current of 0.4 fires 83 times in 39600 steps"
    decided_late = [
        index for index, variant in enumerate(variants) if variant["case"] == long_count
    ]
    assert decided_late == [3, 10, 11, 12]


def test_every_variant_of_the_pernarowski_model_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Fourteen signs of three equations; nine leave the 5000-step count unchanged.

    The oscillator crosses 17 times in 5000 steps whatever the current, and so
    does each of nine variants. Over 16049 steps at -0.1 each of those nine
    crosses 52, 53 or 55 times, where the model crosses 54, and no crossing of a
    variant lies closer than 20 steps to the horizon. The schema also gives the 1
    event in 160 steps at 0.2 that tells the crossing's direction; no variant
    needs it.
    """
    row = _rows(report)["PernarowskiNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [17, 17, 17, 17, 54, 1]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ v"),
        ("v", "- v * v * v / 3.0"),
        ("v", "- w"),
        ("v", "- z"),
        ("v", "+ I"),
        ("w", "+ v"),
        ("w", "- gamma * w"),
        ("w", "+ alpha"),
        ("w", "the whole right-hand side"),
        ("z", "+ beta * (v + 0.7)"),
        ("z", "- z"),
        ("z", "+ v"),
        ("z", "+ 0.7"),
        ("z", "the whole right-hand side"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)
    long_count = "a current of -0.1 crosses the threshold 54 times in 16049 steps"
    decided_late = [
        index for index, variant in enumerate(variants) if variant["case"] == long_count
    ]
    assert decided_late == [3, 4, 6, 7, 9, 10, 11, 12, 13]


_MIHALAS_NIEBUR_JUMP_TERMS = [
    ("v", "+ i1"),
    ("v", "+ i2"),
    ("i1", "the whole right-hand side"),
    ("i2", "the whole right-hand side"),
]


def test_every_variant_of_the_mihalas_niebur_model_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Fourteen signs of four equations, all decided by the first burst under panel M's jumps.

    Both cases set the two current jumps. The first burst, 8 spikes in 400 steps,
    breaks every variant; the four that act through the internal currents give 1,
    16, 13 and 5 spikes.

    Parameters
    ----------
    report : dict[str, object]
        The report as the tool computes it.
    """
    row = _rows(report)["MihalasNieburNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [8, 14]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert len(variants) == 14
    assert all(variant["outcome"] == "violated" for variant in variants)
    first_burst = str(unchanged[0]["name"])
    assert {variant["case"] for variant in variants} == {first_burst}
    through_the_currents = {
        (variant["equation"], variant["term"]): variant["observed"]
        for variant in variants
        if (variant["equation"], variant["term"]) in _MIHALAS_NIEBUR_JUMP_TERMS
    }
    assert through_the_currents == {
        ("v", "+ i1"): 1,
        ("v", "+ i2"): 16,
        ("i1", "the whole right-hand side"): 13,
        ("i2", "the whole right-hand side"): 5,
    }


def test_without_the_jumps_no_mihalas_niebur_case_decides_the_current_terms() -> None:
    """The same windows on the model as built by default leave exactly the four current terms open.

    With both jumps at 0 the internal currents stay 0, so a variant that flips a term
    acting through them changes nothing: restated at their default counts (2 and 10),
    the two cases decide every other sign and none of those four. This is why the cases
    set the jumps.
    """
    from sc_neurocore.neurons.model_catalogue import load_descriptor

    descriptor = load_descriptor("MihalasNieburNeuron")
    assert descriptor is not None
    schema = json.loads(candidate_from_catalogue("MihalasNieburNeuron")["candidate_text"])["model"]
    by_default = tuple(
        dataclasses.replace(case, parameters=(), bounds=(count, count))
        for case, count in zip(descriptor.behaviour, (2, 10), strict=True)
    )
    result = evaluate_schema(schema, by_default)
    assert [entry["observed"] for entry in result["unchanged"]] == [2, 10]
    assert result["state"] == "not_discriminated"
    open_terms = [
        (variant["equation"], variant["term"])
        for variant in result["variants"]
        if variant["outcome"] == "not_discriminated"
    ]
    assert open_terms == _MIHALAS_NIEBUR_JUMP_TERMS


def test_every_variant_of_the_sc_inclusive_perfect_integrator_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Two signs of one step; with either flipped, no candidate reaches the threshold.

    The schema's step is v + I * dt / c_m. With the sign of v flipped, v swings
    between 0 and the step's increment, 0.5 at a current of 5, and never reaches
    1.0; with the sign of the increment flipped, v only falls. Both fire no
    event where the model fires 500 times, so the first case breaks both; the
    second, which keeps every candidate 0.1 from the threshold, would too.
    """
    row = _rows(report)["SCInclusivePerfectIntegratorNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [500, 500]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("v", "+ v"),
        ("v", "+ I * dt / c_m"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)
    boundary = "a current of 5 fires 500 times in 1000 steps on the inclusive boundary"
    assert [(variant["case"], variant["observed"]) for variant in variants] == [
        (boundary, 0),
        (boundary, 0),
    ]


def test_every_variant_of_the_ermentrout_kopell_map_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Eight signs of the phase map; the increment's own sign needs the slow count.

    With the whole increment flipped the phase turns backwards, and over the
    enrolled windows it still crosses as often as the model does. At a current
    of 0.02 over 1111 steps it does not: the model fires 5 times there.
    """
    row = _rows(report)["ErmentroutKopellMapNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 45, 64, 5]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [(variant["equation"], variant["term"]) for variant in variants] == [
        ("theta", "+ theta"),
        ("theta", "+ 0.1 * (1.0 - cos(theta) + (1.0 + cos(theta)) * gain * I)"),
        ("theta", "+ 1.0"),
        ("theta", "- cos(theta)"),
        ("theta", "+ (1.0 + cos(theta)) * gain * I"),
        ("theta", "+ 1.0"),
        ("theta", "+ cos(theta)"),
        ("theta", "the whole right-hand side"),
    ]
    assert all(variant["outcome"] == "violated" for variant in variants)
    slow_count = "a current of 0.02 fires 5 times in 1111 steps"
    decided_late = [
        index for index, variant in enumerate(variants) if variant["case"] == slow_count
    ]
    assert decided_late == [1]


def test_every_variant_of_the_upward_crossing_rulkov_map_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Seventeen signs of the two-branch map; two need the count past the windows.

    Over the 30-step windows the sign of y inside the upper branch's value and
    the sign of x in the slow equation change no event. Over 200 steps at 0.2
    each of them fires a different number of times than the map's 11.
    """
    row = _rows(report)["SCUpwardCrossingRulkovMapNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [0, 7, 10, 11, 11]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert len(variants) == 17
    assert [variant["equation"] for variant in variants] == ["x"] * 12 + ["y"] * 5
    assert all(variant["outcome"] == "violated" for variant in variants)
    measured = "a current of 0.2 fires 11 times in 200 steps"
    decided_late = [
        (variant["equation"], variant["term"], variant["within"])
        for variant in variants
        if variant["case"] == measured
    ]
    assert decided_late == [("x", "+ y", "alpha + y + I"), ("y", "+ x", "x + 1.0")]


def test_every_variant_of_the_clipped_logistic_bursting_map_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Ten signs of the clipped fast map and the slow drift; five need the chaotic window.

    The windows at 0.5, 1.0 and 2.0 fire within their first four steps and
    clip the fast map low from the sixth step at the latest, so the sign of y
    in the fast map and every sign of the slow equation leave their events
    unchanged. At 0.05 the map stays inside the clip for all 30 steps and fires
    7 times, and each of those five variants fires 9 or 2 times.
    """
    row = _rows(report)["SCClippedLogisticBurstingMapNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [2, 1, 1, 7]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert len(variants) == 10
    assert [variant["equation"] for variant in variants] == ["x"] * 6 + ["y"] * 4
    assert all(variant["outcome"] == "violated" for variant in variants)
    chaotic = "a current of 0.05 fires 7 times in 30 steps"
    decided_late = [
        (variant["equation"], variant["term"], variant["within"])
        for variant in variants
        if variant["case"] == chaotic
    ]
    assert decided_late == [
        ("x", "- y", "a * x * (1.0 - x) - y + I"),
        ("y", "+ y", "y + epsilon * (x - sigma)"),
        ("y", "+ epsilon * (x - sigma)", "y + epsilon * (x - sigma)"),
        ("y", "+ x", "x - sigma"),
        ("y", "- sigma", "x - sigma"),
    ]


def test_every_variant_of_the_adaptive_threshold_map_breaks_a_case(
    report: dict[str, object],
) -> None:
    """Nine signs of the clipped fast map and the adaptive threshold; the RTL window decides all.

    Over 32 steps at 0.4 the map fires 3 times. Every sign variant of either
    equation changes that count, so no variant needs the 38-step window, which is
    there to tell the crossing's direction.
    """
    row = _rows(report)["SCAdaptiveThresholdMapNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [3, 4]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [variant["equation"] for variant in variants] == ["x"] * 6 + ["theta"] * 3
    assert all(variant["outcome"] == "violated" for variant in variants)
    assert {variant["case"] for variant in variants} == {
        "a current of 0.4 fires 3 times in 32 steps"
    }


def test_every_variant_of_the_chialvo_map_breaks_a_case(report: dict[str, object]) -> None:
    """Eight signs of the two equations; the window at 0.01 decides all of them.

    Over 100 steps at 0.01 the map fires 3 times. Every sign variant of the fast
    variable's equation (five terms, two of them inside the exponential) and of the
    recovery variable's (three terms) changes that count, so no variant needs the
    window at 1.0, which is there to tell the crossing's direction.
    """
    row = _rows(report)["ChialvoMapNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [3, 1]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert [variant["equation"] for variant in variants] == ["x"] * 5 + ["y"] * 3
    assert all(variant["outcome"] == "violated" for variant in variants)
    assert {variant["case"] for variant in variants} == {
        "a current of 0.01 fires 3 times in 100 steps"
    }


def test_every_variant_of_the_exponential_if_breaks_a_case(report: dict[str, object]) -> None:
    """Eight signs of the membrane equation; the window at 20.0 decides all of them.

    Over 1,000 steps at 20.0 the model fires twice. Every sign variant of the leak,
    the exponential spike-initiation current, the input and the whole right-hand side
    changes that count, so no variant needs the window at 100.0, which is there for
    the reset voltage.
    """
    row = _rows(report)["ExpIFNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [2, 9]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert len(variants) == 8
    assert {variant["equation"] for variant in variants} == {"v"}
    assert all(variant["outcome"] == "violated" for variant in variants)
    assert {variant["case"] for variant in variants} == {
        "a current of 20.0 fires 2 times in 1000 steps"
    }


def test_every_variant_of_the_sigma_delta_breaks_its_case(report: dict[str, object]) -> None:
    """Eleven signs of the integrator, the reconstruction and the event; one window decides all.

    Over 128 steps at 2.0 the modulator fires 42 times. Every sign variant of its
    three equations, the integrator's input, the reconstruction's decay and increment,
    and the terms of the comparison that gates them, changes that count.
    """
    row = _rows(report)["SigmaDeltaNeuron"]
    assert row["state"] == "discriminated"
    unchanged = row["unchanged"]
    assert isinstance(unchanged, list)
    assert [entry["observed"] for entry in unchanged] == [42]
    variants = row["variants"]
    assert isinstance(variants, list)
    assert len(variants) == 11
    assert all(variant["outcome"] == "violated" for variant in variants)
    assert {variant["case"] for variant in variants} == {
        "a current of 2.0 fires 42 times in 128 steps"
    }


def test_a_model_without_a_bound_schema_is_said_to_have_none(report: dict[str, object]) -> None:
    """The dendritic NMDA model states cases and has no schema to vary."""
    row = _rows(report)["DendriticNMDANeuron"]
    assert row["state"] == "no_schema"
    assert row["variants"] == []


def test_the_summary_counts_the_states(report: dict[str, object]) -> None:
    """An operator sees how many models have their terms decided by their cases."""
    summary = render_summary(report)
    assert f"models that state cases: {MODELS_WITH_CASES} of 185" in summary
    assert re.search(r"^  discriminated\s+21$", summary, flags=re.M)
    assert re.search(r"^  no_schema\s+1$", summary, flags=re.M)
    assert re.search(r"^  not_discriminated\s+0$", summary, flags=re.M)
    assert "variants: 251, of them violated 251, not discriminated 0" in summary
    for note in REPORT_NOTES:
        assert note in summary


def test_the_check_fails_on_drift_and_on_a_term_no_case_decides() -> None:
    """A committed report that matches still fails when a variant holds every case."""
    decided = {"model": "A", **evaluate_schema(_accumulator(), (_SETTLES, _FIRES))}
    undecided = {"model": "B", **evaluate_schema(_accumulator(), (_SILENT,))}
    differs = {"model": "C", **evaluate_schema(_accumulator(), (_spikes("x", 3.0, (10, 20)),))}
    good = report_of([decided], catalogue_models=3)
    assert check_problems(good, encode(good)) == []
    assert check_problems(good, None) == [
        "the committed report is absent or has drifted; run --write"
    ]
    bad = report_of([decided, undecided, differs], catalogue_models=3)
    assert bad["models_without_cases"] == 0
    assert bad["variants"] == {"total": 6, "violated": 3, "not_discriminated": 3}
    assert check_problems(bad, encode(bad)) == [
        "cases do not decide every term: {'not_discriminated': 1, 'schema_differs': 1}"
    ]
    assert len(check_problems(bad, encode(good))) == 2


def test_write_then_check_and_a_stale_file_fails(
    tmp_path: Path, report: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """The command writes the report, passes its own check and refuses a changed file."""
    output = tmp_path / "variants.json"
    small = report_of(
        [{"model": "A", **evaluate_schema(_accumulator(), (_SETTLES, _FIRES))}], catalogue_models=1
    )
    assert main(["--write", "--output", str(output)], build=lambda: small) == 0
    assert output.read_text(encoding="utf-8") == encode(small)

    def built() -> dict[str, object]:
        return report

    assert main(["--write", "--output", str(output)], build=built) == 0
    assert output.read_text(encoding="utf-8") == encode(report)
    assert main(["--check", "--output", str(output)], build=built) == 0
    output.write_text(
        output.read_text(encoding="utf-8").replace("violated", "held"), encoding="utf-8"
    )
    assert main(["--check", "--output", str(output)], build=built) == 1
    assert main(["--check", "--output", str(tmp_path / "absent.json")], build=built) == 1
    assert "absent or has drifted" in capsys.readouterr().out


def test_no_argument_prints_the_summary(
    report: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """The bare command prints the summary and writes nothing."""
    assert main([], build=lambda: report) == 0
    assert f"models that state cases: {MODELS_WITH_CASES} of 185" in capsys.readouterr().out


def test_the_script_runs_as_a_program(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool runs from its own path and builds the report itself."""
    script = ROOT / "tools" / "model_behaviour_variants.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--check"])
    with pytest.raises(SystemExit) as finished:
        runpy.run_path(str(script), run_name="__main__")
    assert finished.value.code == 0


_FIRES_WITH_A_GAIN_OF_A_HALF = BehaviourCase(
    name="with a gain of a half, a current of 2 fires at every step",
    kind="interval",
    steps=100,
    observable="spikes",
    inputs=(("current", 2.0),),
    bounds=(100, 100),
    parameters=(("gain", 0.5),),
)


def test_a_case_s_parameters_are_set_on_the_schema_and_on_every_variant() -> None:
    """The unchanged schema and each variant run with the case's gain on top of the schema's.

    With the schema's own gain of a quarter, a current of 2 fires once in a hundred
    steps, and the variants that change the first two signs fire 20 and 50 times.
    With the case's gain of a half, all three fire at every step, so those two
    variants hold the case only because the gain was set on them too.
    """
    schema = _accumulator()
    before = copy.deepcopy(schema)
    result = evaluate_schema(schema, (_FIRES_WITH_A_GAIN_OF_A_HALF,))
    assert schema == before
    assert result["unchanged"] == [
        {"name": _FIRES_WITH_A_GAIN_OF_A_HALF.name, "observed": 100, "held": True}
    ]
    assert [variant["outcome"] for variant in result["variants"]] == [
        "not_discriminated",
        "not_discriminated",
        "violated",
    ]
    assert result["state"] == "not_discriminated"


def test_a_schema_without_a_parameter_a_case_sets_differs_from_the_model() -> None:
    """The case cannot run on that schema as stated, so the schema and the class differ."""
    case = dataclasses.replace(_FIRES_WITH_A_GAIN_OF_A_HALF, parameters=(("tau", 2.0),))
    result = evaluate_schema(_accumulator(), (case,))
    assert result == {
        "state": "schema_differs",
        "unchanged": [],
        "variants": [],
        "reason": f"case {case.name!r} sets the parameter 'tau', which the schema does not declare",
    }
