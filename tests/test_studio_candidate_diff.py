# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate diffs: mathematics before text

"""A rewritten equation is not a changed one, and a changed one says how.

The comparisons run through SymPy on expressions converted from their AST, so
the cases include rewrites only simplification recognises, constructs with no
symbolic reading, and differences too large to settle.
"""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.studio.candidate_diff import (
    DIFF_SCHEMA_VERSION,
    compare_expressions,
    diff_candidate,
    diff_models,
)
from sc_neurocore.neurons.equation_refusals import EquationRefusal
from tests.studio_candidate_support import adex_candidate


@pytest.mark.parametrize(
    ("parent", "candidate", "status"),
    [
        ("(v - a) / tau", "(v - a) / tau", "unchanged"),
        ("(v - a) / tau", "v / tau - a / tau", "equivalent"),
        ("0.5 * v", "v / 2", "equivalent"),
        ("exp(v) * exp(w)", "exp(v + w)", "equivalent"),
        ("1", "sin(v)**2 + cos(v)**2", "equivalent"),
        ("exprel(v)", "exprel(v) + 0", "equivalent"),
        ("-v + pi", "pi - v", "equivalent"),
        (
            "abs(v) + sqrt(w) + log(w) + tanh(v) + cosh(v) + sinh(v)",
            "sinh(v) + cosh(v) + tanh(v) + log(w) + sqrt(w) + abs(v)",
            "equivalent",
        ),
    ],
)
def test_rewrites_are_recognised(parent: str, candidate: str, status: str) -> None:
    """Recognise exact algebraic rewrites through the public comparison API.

    Parameters
    ----------
    parent : str
        Original admitted equation.
    candidate : str
        Identical or algebraically equivalent equation.
    status : str
        Expected distinction between unchanged text and equivalent mathematics.
    """
    assert compare_expressions(parent, candidate) == {"status": status}


def test_a_real_change_states_the_difference() -> None:
    """Report the simplified candidate-minus-parent equation for a real change."""
    assert compare_expressions("v / tau", "2 * v / tau") == {
        "status": "changed",
        "difference": "v/tau",
    }


def test_a_helper_with_its_own_numerics_is_equal_only_as_written() -> None:
    """Keep the numerical sigmoid helper distinct from an exponential rewrite."""
    result = compare_expressions("sigmoid(v)", "1 / (1 + exp(-v))")
    assert result["status"] == "changed"


@pytest.mark.parametrize(
    ("expression", "construct"),
    [
        ("v if v > 0 else 0", "IfExp"),
        ("v % 2", "BinOp"),
        ("True + v", "Constant"),
        ("v.real", "Attribute"),
    ],
)
def test_a_construct_without_a_symbolic_reading_is_not_guessed(
    expression: str, construct: str
) -> None:
    """Decline symbolic comparison of admitted constructs without an exact reading.

    Parameters
    ----------
    expression : str
        Admitted conditional, remainder, Boolean or attribute expression.
    construct : str
        AST node that prevents exact symbolic conversion.
    """
    assert compare_expressions("v", expression) == {
        "status": "not_comparable",
        "reason": f"uses {construct}, which has no symbolic reading",
    }


def test_module_qualified_call_is_refused_before_symbolic_conversion() -> None:
    """Propagate the equation gate's refusal of a module-qualified function call."""
    with pytest.raises(EquationRefusal) as refusal:
        compare_expressions("v", "math.exp(v)")
    assert str(refusal.value) == (
        "Only a function under a plain name can be called in an equation: 'math.exp(v)'"
    )


def test_a_difference_too_large_to_simplify_is_left_undecided() -> None:
    """Leave a large polynomial difference undecided within the comparison budget."""
    assert compare_expressions("v", "(v + w + x + y)**8") == {
        "status": "undecided",
        "reason": "the difference is too large to simplify here",
    }


def _model(**sections: Any) -> dict[str, Any]:
    """Build the minimal model document with explicit section overrides.

    Parameters
    ----------
    **sections : Any
        State, parameter, dynamics, threshold or integration sections to replace.

    Returns
    -------
    dict of str to Any
        Model document consumed by the public model comparison API.
    """
    return {"state": {"v": 0.0}, "parameters": {}, "dynamics": {"v": "-v"}, **sections}


def test_values_and_equations_are_listed_as_added_removed_changed_or_unchanged() -> None:
    """Classify state, parameter, dynamics and reset changes by their actual names."""
    parent = _model(
        state={"v": 0.0, "w": 0.0},
        parameters={"a": 1.0, "b": 2.0},
        dynamics={"v": "-v", "w": "-w"},
        reset={"v": "0"},
    )
    candidate = _model(
        state={"v": -1.0, "u": 0.0},
        parameters={"a": 1.0, "c": 3.0},
        dynamics={"v": "-v + a", "u": "-u"},
        reset={"v": "0", "u": "u + 1"},
    )
    diff = diff_models(parent, candidate)

    assert diff["state"] == [
        {"name": "u", "status": "added", "candidate": 0.0},
        {"name": "v", "status": "changed", "parent": 0.0, "candidate": -1.0},
        {"name": "w", "status": "removed", "parent": 0.0},
    ]
    assert [(row["name"], row["status"]) for row in diff["parameters"]] == [
        ("a", "unchanged"),
        ("b", "removed"),
        ("c", "added"),
    ]
    assert [(row["variable"], row["status"]) for row in diff["dynamics"]] == [
        ("u", "added"),
        ("v", "changed"),
        ("w", "removed"),
    ]
    assert [(row["variable"], row["status"]) for row in diff["reset"]] == [
        ("u", "added"),
        ("v", "unchanged"),
    ]


@pytest.mark.parametrize(
    ("parent", "candidate", "status"),
    [
        ({"condition": "v > 1"}, {"condition": "v > 1"}, "unchanged"),
        # A comparison has no symbolic reading, so a reworded condition is not guessed at.
        ({"condition": "v > 1"}, {"condition": "1 < v"}, "not_comparable"),
        ({"condition": "v > 1"}, {}, "removed"),
        ({}, {"condition": "v > 1"}, "added"),
        ({}, {}, "unchanged"),
    ],
)
def test_the_threshold_is_compared_whether_or_not_either_side_has_one(
    parent: dict[str, str], candidate: dict[str, str], status: str
) -> None:
    """Compare threshold presence and admitted conditions without guessing rewrites.

    Parameters
    ----------
    parent, candidate : dict of str to str
        Original and proposed threshold sections, possibly absent.
    status : str
        Expected presence, identity or non-comparability classification.
    """
    diff = diff_models(_model(threshold=parent), _model(threshold=candidate))
    assert diff["threshold"]["status"] == status


def test_integration_changes_name_their_fields() -> None:
    """Name changed integration fields while retaining the unchanged timestep."""
    diff = diff_models(
        _model(integration={"method": "euler", "dt": 0.1}),
        _model(integration={"method": "rk4", "dt": 0.1}),
    )
    assert diff["integration"]["status"] == "changed"
    assert diff["integration"]["changed_fields"] == ["method"]
    unchanged = diff_models(_model(integration={"dt": 0.1}), _model(integration={"dt": 0.1}))
    assert unchanged["integration"] == {
        "status": "unchanged",
        "changed_fields": [],
        "parent": {"dt": 0.1},
        "candidate": {"dt": 0.1},
    }


def test_a_candidate_is_diffed_against_its_parents_canonical_schema() -> None:
    """Compare the real AdEx candidate with its catalogue parent's canonical schema."""
    diff = diff_candidate(adex_candidate())
    assert diff["schema_version"] == DIFF_SCHEMA_VERSION
    assert diff["status"] == "compared"
    assert diff["parent"] == "AdExNeuron"
    assert diff["parent_schema"] == "adex"
    changed = [row["name"] for row in diff["parameters"] if row["status"] != "unchanged"]
    assert changed == ["tau_w"]
    assert {row["status"] for row in diff["dynamics"]} == {"unchanged"}


def test_without_a_parent_or_its_schema_there_is_nothing_to_diff() -> None:
    """Distinguish an orphan candidate from a catalogue parent without a DSL schema."""
    orphan = adex_candidate()
    orphan["parent"] = None
    assert diff_candidate(orphan) == {
        "schema_version": DIFF_SCHEMA_VERSION,
        "parent": None,
        "status": "no_parent",
    }
    unschemed = adex_candidate()
    unschemed["parent"] = "ATypeKNeuron"
    assert diff_candidate(unschemed)["status"] == "parent_has_no_schema"
