# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation safety capability-blocking tests

"""Blocked identifiers, attributes, dunder access, and the closed-grammar contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sc_neurocore.neurons.equation_safety import ExpressionSafetyValidator

SOURCE = Path(__file__).resolve().parents[1] / "src/sc_neurocore"


def test_blocked_name_is_rejected() -> None:
    """A bare blocked identifier is refused as a blocked function."""
    with pytest.raises(ValueError, match="Blocked function 'os'"):
        ExpressionSafetyValidator().validate("os")


def test_dunder_attribute_access_is_rejected() -> None:
    """Any double-underscore attribute access is refused outright."""
    with pytest.raises(ValueError, match="Dunder attribute access"):
        ExpressionSafetyValidator().validate("v.__class__")


def test_blocked_attribute_is_rejected() -> None:
    """A non-dunder attribute whose name is on the block list is refused."""
    with pytest.raises(ValueError, match="Blocked attribute 'subprocess'"):
        ExpressionSafetyValidator().validate("v.subprocess")


@pytest.mark.parametrize(
    "expression",
    [
        "v.dtype",
        "v.units",
        "v.magnitude + 1",
        "max.real.conjugate",
    ],
)
def test_only_the_listed_attributes_can_be_read(expression: str) -> None:
    """An attribute outside the list is refused, whatever object would carry it.

    Parameters
    ----------
    expression : str
        Expression reading an attribute that numeric scalars, the supplied
        functions or unit quantities carry.
    """
    with pytest.raises(ValueError, match="not part of the equation grammar"):
        ExpressionSafetyValidator().validate(expression)
    ExpressionSafetyValidator().validate("v.real + w.imag")


@pytest.mark.parametrize(
    "expression",
    [
        "v.real(w)",
        "v.tofile(w)",
        "v.dump(w)",
        "exp.at(v, 0)",
        "f(v)(w)",
        "table[0](v)",
        "(f if v else g)(w)",
        "[f][0](v)",
    ],
)
def test_only_a_function_under_a_plain_name_can_be_called(expression: str) -> None:
    """Calling the result of an attribute, an index, a choice or a call is refused.

    Parameters
    ----------
    expression : str
        Expression whose call target is not a plain name.
    """
    with pytest.raises(ValueError, match="Only a function under a plain name"):
        ExpressionSafetyValidator().validate(expression)
    ExpressionSafetyValidator().validate("f(g(v), w)")


@pytest.mark.parametrize(
    "module",
    [
        "neurons/equation_program.py",
        "neurons/equation_evaluation.py",
        "neurons/equation_builder.py",
        "neurons/equation_safety.py",
        "neurons/equation_units_runtime.py",
        "neurons/_units.py",
        "studio/nullclines.py",
    ],
)
def test_no_equation_module_executes_dynamic_code(module: str) -> None:
    """No module on the path of an authored expression calls the interpreter on text or code.

    Parameters
    ----------
    module : str
        Source file, relative to the package, that handles authored expressions.
    """
    tree = ast.parse((SOURCE / module).read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"eval", "exec", "compile", "__import__"}
    assert "nosec" not in (SOURCE / module).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "expression",
    ["round(v, out)", "round(v, out=out)", "heaviside(v, 0.5, out)", "heaviside(v, 0.5, out=out)"],
)
def test_discrete_functions_cannot_write_an_output_array(expression: str) -> None:
    """NumPy's optional output capability is unavailable to authored expressions."""
    with pytest.raises(ValueError, match="output argument|keyword"):
        ExpressionSafetyValidator().validate(expression)
