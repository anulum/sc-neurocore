# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the equation evaluation namespace

"""Unit tests for the numpy evaluation namespace of equation-defined neurons.

The namespace bindings are load-bearing for co-simulation bit-exactness (the
fixed-point emitter mirrors these exact functions), so these tests pin both the
identity of the standard bindings and the numerical behaviour of the three
custom helpers (``sigmoid``, ``sqrt``, ``exprel``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.equation_namespace import build_eval_namespace


def test_namespace_is_fresh_on_each_call() -> None:
    """Each build returns a distinct dict so neurons never alias a namespace."""
    first = build_eval_namespace()
    second = build_eval_namespace()
    assert first is not second
    assert first == second


def test_namespace_exposes_the_expected_symbols() -> None:
    """The namespace carries exactly the documented maths surface."""
    namespace = build_eval_namespace()
    assert set(namespace) == {
        "exp",
        "log",
        "sqrt",
        "abs",
        "sin",
        "cos",
        "tanh",
        "cosh",
        "sinh",
        "exprel",
        "sigmoid",
        "pi",
        "clip",
        "max",
        "min",
    }


def test_standard_bindings_are_the_numpy_originals() -> None:
    """The transcendental/arithmetic bindings must be the exact numpy callables.

    Co-simulation bit-exactness depends on ``tanh`` being :func:`numpy.tanh`
    and so on; a ``math`` substitute would diverge at the least-significant bits.
    """
    namespace = build_eval_namespace()
    assert namespace["exp"] is np.exp
    assert namespace["log"] is np.log
    assert namespace["sin"] is np.sin
    assert namespace["cos"] is np.cos
    assert namespace["tanh"] is np.tanh
    assert namespace["cosh"] is np.cosh
    assert namespace["sinh"] is np.sinh
    assert namespace["clip"] is np.clip
    assert namespace["abs"] is abs
    assert namespace["max"] is max
    assert namespace["min"] is min
    assert namespace["pi"] == math.pi


def test_sigmoid_is_logistic_and_saturates_without_overflow() -> None:
    """``sigmoid`` returns the logistic curve and clips extreme arguments."""
    sigmoid = build_eval_namespace()["sigmoid"]
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(1.0) == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    # Extreme arguments are clipped to [-500, 500] so ``exp`` never overflows.
    assert sigmoid(10_000.0) == pytest.approx(1.0)
    assert sigmoid(-10_000.0) == pytest.approx(0.0)


def test_sqrt_returns_root_and_rejects_negative_domain() -> None:
    """``sqrt`` computes the root but raises before a NumPy invalid-domain warning."""
    sqrt = build_eval_namespace()["sqrt"]
    assert sqrt(4.0) == pytest.approx(2.0)
    assert np.allclose(sqrt(np.array([0.0, 9.0])), np.array([0.0, 3.0]))
    with pytest.raises(ValueError, match="sqrt domain error"):
        sqrt(-1.0)
    with pytest.raises(ValueError, match="sqrt domain error"):
        sqrt(np.array([1.0, -4.0]))


def test_exprel_carries_the_removable_singularity() -> None:
    """``exprel`` is (exp(x)-1)/x with the exact limit ``exprel(0) = 1``."""
    exprel = build_eval_namespace()["exprel"]
    # Removable singularity at x = 0.
    assert float(exprel(0.0)) == pytest.approx(1.0)
    # Small-|x| branch uses the series limit 1 + x/2.
    assert float(exprel(1e-12)) == pytest.approx(1.0 + 1e-12 / 2.0)
    # General branch matches expm1(x)/x away from the singularity.
    assert float(exprel(2.0)) == pytest.approx(math.expm1(2.0) / 2.0)
    # Vectorises with the per-element branch selection.
    result = np.asarray(exprel(np.array([0.0, 2.0])))
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(math.expm1(2.0) / 2.0)
