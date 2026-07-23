# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — conductance-rate co-simulation references

"""NumPy-exact exponential and stable exprel contracts for conductance rates."""

from __future__ import annotations

import numpy as np


def _np_exp(x: float) -> float:
    """Return ``exp(x)`` through the same numpy implementation the schema runner uses.

    Parameters
    ----------
    x:
        Exponent argument.

    Returns
    -------
    float
        ``numpy.exp(x)`` as a Python float, bit-identical to the runner's rate terms.
    """
    return float(np.exp(x))


def _reference_exprel(x: float) -> float:
    """Return ``exprel(x) = (exp(x) - 1) / x`` with the removable-singularity limit.

    Mirrors ``EquationNeuron``'s vectorised ``exprel`` bit-for-bit: the ``|x| < 1e-9``
    branch returns the ``exprel(0) = 1`` limit as ``1 + x / 2``, and the regular
    branch uses ``numpy.expm1`` so conductance rate functions written as
    ``a / exprel(...)`` reproduce the runner exactly.

    Parameters
    ----------
    x:
        Rate-function argument.

    Returns
    -------
    float
        The exprel value matching the schema runner.
    """
    if abs(x) < 1e-9:
        return 1.0 + x / 2.0
    return float(np.expm1(x)) / x
