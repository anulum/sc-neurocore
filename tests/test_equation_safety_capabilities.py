# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation safety capability-blocking tests

"""Blocked identifiers, attributes, dunder access, and eval-global contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_safety import EVAL_GLOBALS, ExpressionSafetyValidator


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


def test_eval_globals_expose_only_import_in_scoped_builtins() -> None:
    """``__builtins__`` exposes ``__import__`` and no other capability.

    ``__import__`` is intentional and unreachable from expressions (the AST
    allowlist blocks its name plus all dunder access); it must stay present because
    CPython's ``eval`` dereferences ``__builtins__['__import__']`` for some safe
    expressions. ``EVAL_GLOBALS`` is shared across eval sites, so the interpreter
    may inject a transient ``__warningregistry__`` (warning-filter state, not a
    builtin) into it when a warning fires during eval; that artefact carries no
    capability and is ignored, while any genuine leak would still surface as an
    extra key.
    """
    assert set(EVAL_GLOBALS) - {"__warningregistry__"} == {"__builtins__"}
    builtins_map = EVAL_GLOBALS["__builtins__"]
    assert builtins_map["__import__"] is __import__
    assert set(builtins_map) == {"__import__"}
