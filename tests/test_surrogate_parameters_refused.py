# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — surrogate parameter refusal tests

"""A surrogate's steepness ``k`` must be finite and above 0, through every class that takes one.

Before, any ``k`` was taken: a fast sigmoid with ``k = 0`` gave an infinite gradient, ``k = -1`` a negative one and
``k = NaN`` a NaN. The extension's ``SurrogateLif`` and ``DifferentiableDenseLayer``, and their wrappers in
``sc_neurocore_engine.grad``, now raise ``ValueError`` for such a ``k`` before anything is built, and keep accepting the
family defaults and any finite positive ``k``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from tests.engine_requirement import require_engine

ENGINE = require_engine()

from sc_neurocore_engine import grad

_REASON = "^surrogate k must be finite and above 0$"
_REFUSED_K = {"zero": 0.0, "negative": -1.0, "NaN": math.nan, "infinite": math.inf}
_FAMILIES = ("fast_sigmoid", "superspike", "arctan")


def _build(kind: str, surrogate: str, k: float | None) -> Any:
    """Build one of the four classes that take a surrogate.

    Parameters
    ----------
    kind : str
        ``native lif``, ``wrapper lif``, ``native dense`` or ``wrapper dense``.
    surrogate : str
        The surrogate family.
    k : float or None
        The steepness, or ``None`` for the family default.
    """
    if kind == "native lif":
        return ENGINE.SurrogateLif(surrogate=surrogate, k=k)
    if kind == "wrapper lif":
        return grad.SurrogateLif(surrogate=surrogate, k=k)
    if kind == "native dense":
        return ENGINE.DifferentiableDenseLayer(4, 2, surrogate=surrogate, k=k)
    return grad.DifferentiableDenseLayer(4, 2, surrogate=surrogate, k=k)


_KINDS = ("native lif", "wrapper lif", "native dense", "wrapper dense")


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("value", sorted(_REFUSED_K))
def test_a_k_that_is_not_finite_and_above_zero_is_refused(
    kind: str, family: str, value: str
) -> None:
    """Each class refuses the steepness with ``ValueError`` and the reason.

    Parameters
    ----------
    kind : str
        The class.
    family : str
        The surrogate family.
    value : str
        The refused steepness.
    """
    with pytest.raises(ValueError, match=_REASON):
        _build(kind, family, _REFUSED_K[value])


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("k", [None, 1e-6, 5.0])
def test_the_family_default_and_a_finite_positive_k_are_accepted(
    kind: str, family: str, k: float | None
) -> None:
    """The defaults (25, 100 and 10) and any finite positive ``k`` still build.

    Parameters
    ----------
    kind : str
        The class.
    family : str
        The surrogate family.
    k : float or None
        The steepness, or ``None`` for the default.
    """
    assert _build(kind, family, k) is not None


@pytest.mark.parametrize("kind", ["native dense", "wrapper dense"])
def test_a_weight_gradient_of_the_wrong_shape_is_refused_and_the_weights_kept(kind: str) -> None:
    """The dense layer's update says which shape it expected, and changes nothing.

    Parameters
    ----------
    kind : str
        The class.
    """
    layer = _build(kind, "fast_sigmoid", None)
    weights = layer.get_weights() if kind == "native dense" else layer.weights.tolist()
    with pytest.raises(ValueError, match="^Expected 2 grad rows, got 1.$"):
        layer.update_weights([[0.1] * 4], 0.1)
    # The wrapper turns the gradient into one NumPy array first, so ragged rows fail there with NumPy's words.
    ragged = (
        "^Expected each grad row to have length 4.$" if kind == "native dense" else "inhomogeneous"
    )
    with pytest.raises(ValueError, match=ragged):
        layer.update_weights([[0.1] * 4, [0.1] * 3], 0.1)
    assert (layer.get_weights() if kind == "native dense" else layer.weights.tolist()) == weights
