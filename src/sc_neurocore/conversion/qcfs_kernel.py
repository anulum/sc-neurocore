# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch-free QCFS quantisation and surrogate gradients

"""Evaluate QCFS and its straight-through derivatives in float64 without PyTorch.

Every operation follows the order in which ``QCFSActivation`` and PyTorch
autograd evaluate it, so a float64 activation and its per-element derivatives
agree bit for bit, including the sign of zero (Bu et al., 2022, Eq. 17).
"""

import math
import numbers

import numpy as np
import numpy.typing as npt

from .if_parameters import FloatArray

QCFS_STEP_LIMIT = 2**32 - 1
"""Largest step count every runtime accepts; the native ABIs carry it as uint32."""


def checked_qcfs_parameters(steps: int, theta: float) -> tuple[int, float]:
    """Admit the shared QCFS step domain and a finite positive threshold.

    Parameters
    ----------
    steps : int
        Simulation steps and quantisation intervals, ``1 <= steps <= 2**32 - 1``.
    theta : float
        Firing threshold and upper activation bound.

    Returns
    -------
    tuple of (int, float)
        The admitted step count and threshold.

    Raises
    ------
    ValueError
        Step count outside the shared domain or threshold not finite and positive.
    """
    if type(steps) is not int or not 1 <= steps <= QCFS_STEP_LIMIT:
        raise ValueError("QCFS T must be a positive integer no larger than 2**32 - 1")
    if isinstance(theta, bool) or not isinstance(theta, numbers.Real):
        raise ValueError("QCFS theta must be finite and positive")
    try:
        threshold = float(theta)
    except OverflowError as error:
        raise ValueError("QCFS theta must be finite and positive") from error
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("QCFS theta must be finite and positive")
    return steps, threshold


def qcfs_values(values: npt.ArrayLike, name: str) -> FloatArray:
    """Copy real activations into one contiguous float64 array.

    Parameters
    ----------
    values : array_like
        Real integer or floating activations of any shape.
    name : str
        Argument name used in the refusal message.

    Returns
    -------
    numpy.ndarray
        C-contiguous float64 copy with the input shape.

    Raises
    ------
    TypeError
        Boolean, complex, object or other non-real input.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"QCFS {name} must be real integer or floating values")
    return np.array(array, dtype=np.float64, order="C", copy=True)


def reference_forward(x: FloatArray, steps: int, theta: float) -> FloatArray:
    """Quantise admitted activations onto the shifted clipped rate lattice.

    Parameters
    ----------
    x : numpy.ndarray
        Contiguous float64 activations.
    steps, theta : int, float
        Admitted step count and threshold.

    Returns
    -------
    numpy.ndarray
        ``floor(clip(x * T / theta + 0.5, 0, T)) * theta / T``; NaN stays NaN.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        shifted = x * steps / theta + 0.5
        result: FloatArray = np.floor(np.clip(shifted, 0.0, steps)) * theta / steps
    return result


def reference_backward(
    x: FloatArray, upstream: FloatArray, steps: int, theta: float
) -> tuple[FloatArray, FloatArray]:
    """Return per-element input and threshold derivatives of the surrogate.

    Parameters
    ----------
    x, upstream : numpy.ndarray
        Contiguous float64 activations and equally shaped upstream gradients.
    steps, theta : int, float
        Admitted step count and threshold.

    Returns
    -------
    tuple of numpy.ndarray
        Input derivative (upstream on the open interior ``0 < s < T``, zero
        elsewhere) and each element's threshold contribution
        ``upstream * floor(c) / T`` plus, on the interior, ``-upstream * x / theta``
        in autograd's operation order: exactly the threshold gradient a
        one-element batch receives. Summing the second array reduces a shared
        threshold's gradient up to summation order.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        shifted = x * steps / theta + 0.5
        interior = (shifted > 0) & (shifted < steps)
        lattice = np.floor(np.clip(shifted, 0.0, steps))
        output_gradient = upstream / steps
        carried = np.where(interior, output_gradient * theta, 0.0)
        scaled_input = np.where(interior, x, 0.0) * steps
        input_gradient: FloatArray = np.where(interior, carried / theta * steps, 0.0)
        # Autograd reduces each threshold use to its shape from a +0.0
        # accumulator before adding the two uses, so a negative-zero
        # contribution becomes +0.0 exactly as in a one-element batch.
        threshold_gradient: FloatArray = (0.0 + output_gradient * lattice) + (
            0.0 + (-carried) * (scaled_input / theta / theta)
        )
    return input_gradient, threshold_gradient


__all__ = [
    "QCFS_STEP_LIMIT",
    "checked_qcfs_parameters",
    "qcfs_values",
    "reference_backward",
    "reference_forward",
]
