# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense IF parameter snapshots and coupled shape validation

"""Freeze checked, owned float64 parameters for one dense IF replay."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .if_resources import DEFAULT_WORKING_BYTES, admit_parameter_storage

FloatArray = npt.NDArray[np.float64]
OutputMode = Literal["spikes", "linear"]


@dataclass(frozen=True)
class IFParameters:
    """Owned row-major coefficients and the declared final-layer response."""

    weights: tuple[FloatArray, ...]
    biases: tuple[FloatArray | None, ...]
    thresholds: tuple[float, ...]
    initial_membrane_fraction: float
    layer_membrane_fractions: tuple[float, ...]
    output_mode: OutputMode


def real_array(values: npt.ArrayLike, name: str) -> FloatArray:
    """Copy real numeric data to owned contiguous float64 storage.

    Parameters
    ----------
    values : array_like
        Boolean, integer or floating input data.
    name : str
        Parameter name included in refusal messages.

    Returns
    -------
    ndarray
        Finite, owned float64 values in C order.

    Raises
    ------
    ValueError
        If the data is nonreal, nonfinite or cannot be represented in float64.
    """
    original = np.asarray(values)
    if original.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.array(original, dtype=np.float64, copy=True, order="C")
    except FloatingPointError as error:
        raise ValueError(f"{name} cannot be represented in float64") from error
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def parameter_snapshot(
    weights: Sequence[npt.ArrayLike],
    biases: Sequence[npt.ArrayLike | None],
    thresholds: Sequence[float],
    initial_membrane_fraction: float,
    output_mode: OutputMode,
    *,
    max_working_bytes: int = DEFAULT_WORKING_BYTES,
    layer_membrane_fractions: Sequence[float] | None = None,
) -> IFParameters:
    """Validate coupled layer dimensions and freeze independent parameter copies.

    Parameters
    ----------
    weights : sequence of array_like
        Output-by-input weight matrices, connected in sequence.
    biases : sequence of array_like or None
        Per-step output currents, one per layer; None denotes absent bias.
    thresholds : sequence of float
        Positive finite IF thresholds, one per layer.
    initial_membrane_fraction : float
        Finite default membrane offset in threshold units.
    output_mode : {'spikes', 'linear'}
        Final IF spike count or integrated linear readout.

    max_working_bytes : int
        Numeric buffer budget checked before owned coefficient copies.

    layer_membrane_fractions : sequence of float, optional
        Per-layer preloads; None repeats the global membrane fraction.

    Returns
    -------
    IFParameters
        Validated owned coefficients for a complete replay.

    Raises
    ------
    ValueError
        If lengths, dimensions, parameter domains or output mode are invalid.
    """
    if not weights or len(weights) != len(biases) or len(weights) != len(thresholds):
        raise ValueError("weights, biases and thresholds must describe the same nonempty layers")
    initial_membrane_fraction = float(initial_membrane_fraction)
    if not np.isfinite(initial_membrane_fraction):
        raise ValueError("initial membrane fraction must be finite")
    if output_mode not in ("spikes", "linear"):
        raise ValueError("output mode must be spikes or linear")
    admit_parameter_storage(weights, biases, max_working_bytes)
    fractions = (
        (float(initial_membrane_fraction),) * len(weights)
        if layer_membrane_fractions is None
        else tuple(float(value) for value in layer_membrane_fractions)
    )
    if len(fractions) != len(weights) or not all(np.isfinite(value) for value in fractions):
        raise ValueError("layer membrane fractions must be finite and match every layer")
    copied_weights: list[FloatArray] = []
    copied_biases: list[FloatArray | None] = []
    copied_thresholds: list[float] = []
    previous_width: int | None = None
    for weight, bias, threshold in zip(weights, biases, thresholds):
        threshold = float(threshold)
        w = real_array(weight, "weight")
        if w.ndim != 2 or 0 in w.shape:
            raise ValueError("weight must be a nonempty output-by-input matrix")
        if previous_width is not None and w.shape[1] != previous_width:
            raise ValueError("adjacent layer widths must match")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        b = None if bias is None else real_array(bias, "bias")
        if b is not None and b.shape != (w.shape[0],):
            raise ValueError("bias must have one value per output neuron")
        w.setflags(write=False)
        if b is not None:
            b.setflags(write=False)
        copied_weights.append(w)
        copied_biases.append(b)
        copied_thresholds.append(float(threshold))
        previous_width = w.shape[0]
    return IFParameters(
        tuple(copied_weights),
        tuple(copied_biases),
        tuple(copied_thresholds),
        float(initial_membrane_fraction),
        tuple(float(value) for value in fractions),
        output_mode,
    )
