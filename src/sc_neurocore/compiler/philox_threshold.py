# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — a Philox stochastic threshold lowered to a level threshold on rand()

"""Lower a ``threshold_rng="philox"`` stochastic threshold to ``rand(S) < p``.

The Python model (:class:`~sc_neurocore.neurons.equation_random.PhiloxThreshold`)
fires when the uniform draw of the reserved call site
:data:`~sc_neurocore.neurons.equation_random.STOCHASTIC_SITE` at the current
step is below the probability — ``p`` for ``detection="poisson"`` and
``1 - exp(-rate·dt)`` for ``detection="escape_rate"``. That is a level
threshold on an ordinary ``rand()`` draw, so :func:`lower_philox_threshold`
returns a copy of the neuron with exactly that threshold and no stochastic
detection: the registered and folded RTL and the C and Rust kernels then lower
it like any level threshold with a draw. In a format with ``f`` fraction bits
the comparison is ``floor(word0 / 2**(32 - f)) < P``, which is the model's
``word0 / 2**32 < p`` whenever ``p`` is the code ``P / 2**f`` exactly. The LFSR
generator (the default) is left to the existing stochastic-threshold lowering.
"""

from __future__ import annotations

import copy
from typing import TypeVar

from sc_neurocore.neurons.equation_random import STOCHASTIC_SITE

__all__ = ["lower_philox_threshold", "philox_trial_text"]

_N = TypeVar("_N")


def philox_trial_text(probability: str) -> str:
    """Return the level threshold of a Philox trial with ``probability``.

    Parameters
    ----------
    probability : str
        The probability expression of one step.

    Returns
    -------
    str
        ``rand(STOCHASTIC_SITE) < (probability)``.
    """
    return f"rand({STOCHASTIC_SITE}) < ({probability})"


def lower_philox_threshold(neuron: _N) -> _N:
    """Return the neuron with a Philox stochastic threshold spelled as a level threshold.

    Parameters
    ----------
    neuron : EquationNeuron
        The neuron to compile; returned unchanged unless it has a stochastic
        threshold with ``threshold_rng="philox"``.

    Returns
    -------
    EquationNeuron
        A shallow copy with ``threshold_expr = rand(STOCHASTIC_SITE) < (p)``,
        level detection and no stochastic-threshold runtime.

    Raises
    ------
    ValueError
        If the stochastic expression is missing.
    """
    attributes = vars(neuron)
    if not attributes.get("_stochastic_threshold_enabled", False):
        return neuron
    if attributes.get("threshold_rng", "lfsr16") != "philox":
        return neuron
    if attributes.get("_poisson_enabled", False):
        probability = attributes.get("probability_expression")
    else:
        rate = attributes.get("rate_expression")
        probability = None if not rate else f"1.0 - exp(-(({rate}) * {attributes['dt']!r}))"
    if not probability:
        raise ValueError("a Philox stochastic threshold needs its probability or rate expression")
    lowered = copy.copy(neuron)
    lowered_attributes = vars(lowered)
    lowered_attributes.update(
        threshold_expr=philox_trial_text(probability),
        detection="level",
        rate_expression=None,
        probability_expression=None,
        _stochastic_threshold_enabled=False,
        _poisson_enabled=False,
        _escape_rate_enabled=False,
        _edge_detection=False,
        _stochastic_rng=None,
    )
    return lowered
