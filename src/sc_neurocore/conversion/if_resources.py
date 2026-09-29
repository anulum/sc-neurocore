# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense IF numeric buffer admission

"""Bound owned numeric buffers before coefficient copies or replay allocation."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

DEFAULT_WORKING_BYTES = 256 * 1024 * 1024


def admit_parameter_storage(
    weights: Sequence[npt.ArrayLike],
    biases: Sequence[npt.ArrayLike | None],
    max_working_bytes: int,
) -> int:
    """Admit coefficient snapshots and return their float64 element count.

    Parameters
    ----------
    weights, biases : sequence of array_like
        Coefficients whose metadata is inspected before owned float64 copies.
    max_working_bytes : int
        Positive addressable byte budget. Caller-owned storage and interpreter
        overhead are excluded; two complete coefficient copies are reserved.

    Returns
    -------
    int
        Total number of weight and bias elements.

    Raises
    ------
    ValueError
        If the budget is not a positive addressable integer.
    MemoryError
        If coefficient snapshots exceed the declared budget.
    """
    if type(max_working_bytes) is not int or not 0 < max_working_bytes <= np.iinfo(np.intp).max:
        raise ValueError("working byte budget must be a positive addressable integer")
    count = sum(np.asarray(weight).size for weight in weights)
    count += sum(np.asarray(bias).size for bias in biases if bias is not None)
    if 16 * count > max_working_bytes:
        raise MemoryError("coefficient snapshots exceed working byte budget")
    return count


def admit_replay_buffers(
    weights: Sequence[npt.NDArray[np.float64]],
    biases: Sequence[npt.NDArray[np.float64] | None],
    shape: tuple[int, int, int],
    *,
    trace: bool,
    linear: bool,
    max_working_bytes: int,
) -> int:
    """Admit a conservative portable bound on one replay's numeric buffers.

    Parameters
    ----------
    weights, biases : sequence of ndarray
        Checked connected dense coefficients.
    shape : tuple of int
        Time, batch and input-neuron dimensions.
    trace, linear : bool
        Trace retention and final linear-integrator mode.
    max_working_bytes : int
        Operator-selected positive addressable byte budget.

    Returns
    -------
    int
        Reserved numeric bytes, excluding caller buffers and runtime overhead.

    Raises
    ------
    MemoryError
        If the complete reservation exceeds the supplied budget.

    Notes
    -----
    Reserve eight bytes per element of ``2P + 2F + 2S + 2O + H + 5M + I``:
    coefficients P, full frames F, all states S, output O, requested state/event
    traces H, largest layer M and one input frame I. Doubled buffers and five
    largest-layer buffers cover validation, snapshots and arithmetic temporaries.
    Python integer arithmetic cannot wrap the reservation.
    """
    coefficients = admit_parameter_storage(weights, biases, max_working_bytes)
    steps, batch, inputs = shape
    widths = [int(weight.shape[0]) for weight in weights]
    states = batch * sum(widths)
    output = batch * widths[-1]
    largest = batch * max(widths)
    frame = batch * inputs
    traces = steps * batch * (sum(widths) + sum(widths[:-1] if linear else widths)) if trace else 0
    reservation = 8 * (
        2 * coefficients
        + 2 * steps * frame
        + 2 * states
        + 2 * output
        + traces
        + 5 * largest
        + frame
    )
    if reservation > max_working_bytes:
        raise MemoryError("replay buffers exceed working byte budget")
    return reservation
