# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — call-argument rules and named constants shared by every lowering backend

"""``FUNCTION_ARITY``, ``check_call_arguments`` and ``NAMED_CONSTANTS`` of ``expr_lut_tables``.

The rules mirror what the Python model (``neurons.equation_namespace``) accepts:
a backend that lowered ``max(x)`` or ``clip(x, lo)`` to something else would
compute a value the model never computes.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from sc_neurocore.compiler import expr_lut_tables as tables
from sc_neurocore.neurons.equation_namespace import build_eval_namespace


@pytest.mark.parametrize("site", ["I", "True", "-1", "4294967296"])
def test_public_random_call_refuses_a_nonliteral_or_out_of_range_site(site: str) -> None:
    """The documented call-site API validates parsed RNG calls independently of the model."""
    with pytest.raises(ValueError, match="call-site number as an integer literal"):
        tables.random_site("rand", _call(f"rand({site})"))


def _call(text: str) -> ast.Call:
    node = ast.parse(text, mode="eval").body
    assert isinstance(node, ast.Call)
    return node


class TestArityTable:
    """The arity table covers exactly the supported vocabulary."""

    def test_covers_supported_functions(self) -> None:
        """Every supported function has an arity and nothing else does."""
        assert set(tables.FUNCTION_ARITY) == set(tables.SUPPORTED_FUNCTIONS)

    def test_python_model_agrees_on_the_boundaries(self) -> None:
        """The namespace's ``max``, ``min`` and ``clip`` refuse what the table refuses."""
        ns = build_eval_namespace()
        with pytest.raises(TypeError):
            ns["max"](0.5)
        with pytest.raises(TypeError):
            ns["clip"](0.5, 1.0)
        assert ns["max"](1.0, 2.0, 3.0) == 3.0
        assert ns["min"](1.0, 2.0, 3.0) == 1.0
        assert float(ns["clip"](np.float64(5.0), 0.0, 1.0)) == 1.0


class TestCheckCallArguments:
    """``check_call_arguments`` accepts the model's calls and refuses the rest."""

    @pytest.mark.parametrize(
        "text", ["exp(v)", "clip(v, 0.0, 1.0)", "max(a, b)", "max(a, b, c, d)", "min(a, b, c)"]
    )
    def test_accepts(self, text: str) -> None:
        """Calls the Python model evaluates pass."""
        node = _call(text)
        assert isinstance(node.func, ast.Name)
        tables.check_call_arguments(node.func.id, node)

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("max(v)", "at least 2"),
            ("min(v)", "at least 2"),
            ("clip(v)", "exactly 3"),
            ("clip(v, 1.0)", "exactly 3"),
            ("clip(v, 0.0, 1.0, 2.0)", "exactly 3"),
            ("exp(v, 2.0)", "exactly 1"),
            ("clip(v, a_min=0.0, a_max=1.0)", "positional arguments only"),
            ("exp(v, out=None)", "positional arguments only"),
        ],
    )
    def test_refuses(self, text: str, message: str) -> None:
        """A call whose arguments a backend would drop is refused with the reason."""
        node = _call(text)
        assert isinstance(node.func, ast.Name)
        with pytest.raises(ValueError, match=message):
            tables.check_call_arguments(node.func.id, node)


class TestNamedConstants:
    """Named constants are the values the Python model binds."""

    def test_pi_matches_the_namespace(self) -> None:
        """``pi`` has the namespace's value."""
        assert tables.NAMED_CONSTANTS["pi"] == build_eval_namespace()["pi"]

    def test_every_named_constant_is_bound_in_the_namespace(self) -> None:
        """No backend constant exists that the Python model would not evaluate."""
        ns = build_eval_namespace()
        for name, value in tables.NAMED_CONSTANTS.items():
            assert ns[name] == value
