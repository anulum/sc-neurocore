# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mixed-precision Q8.8×Q16.16 dense MAC — bit-true reference + dispatch

"""Bit-true integer reference for the mixed-precision Q8.8 × Q16.16 dense MAC.

Weights are stored Q8.8 (``int16``), inputs and accumulators run Q16.16
(``int32`` codes). Each dense output contracts a weight row with an input vector
in an integer accumulator, divides by the Q8.8 weight scale (an arithmetic right
shift by the fractional width, which equals floor division for a power-of-two
scale) and saturates to the Q16.16 code range, raising explicit overflow and
underflow flags.

This is the integer branch of the mixed-precision pipeline (the per-tensor scale
folded so the accumulator divisor is exactly the weight scale — the production
Zynq/UltraScale+ contract). Because the whole path is exact integer arithmetic,
the Python floor and the Rust, Julia, Go and Mojo backends agree bit-for-bit;
the parity tolerance is exactly zero.

Accumulation contract
---------------------
The kernel accumulates in signed 64-bit integers. The caller must keep
``max|weight| * max|input| * n_inputs`` within ``int64`` range; the reference
checks this conservative bound up front and fails closed rather than wrapping.
The Q16.16 saturation range is ``[-2**31, 2**31 - 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

#: Fractional bit width of the Q8.8 weight format (the accumulator divisor shift).
WEIGHT_FRACTION = 8
#: Saturation bounds of the Q16.16 accumulator code range.
ACCUM_MIN = -(1 << 31)
ACCUM_MAX = (1 << 31) - 1
_I64_MAX = (1 << 63) - 1

#: Acceleration backends in fastest-measured-first dispatch order.
FASTEST_FIRST_BACKENDS = ("rust", "mojo", "julia", "go", "python")


@dataclass(frozen=True)
class MixedDenseBatchResult:
    """Per-element results of a batched mixed-precision dense contraction.

    Each array has shape ``(n_batch, n_outputs)``.

    Attributes
    ----------
    outputs_q1616 : numpy.ndarray
        Saturated Q16.16 accumulator codes, ``int32``.
    overflow : numpy.ndarray
        ``True`` where the accumulator left the Q16.16 range, ``bool_``.
    underflow : numpy.ndarray
        ``True`` where a non-zero contraction rounded to zero without
        overflowing, ``bool_``.
    """

    outputs_q1616: npt.NDArray[np.int32]
    overflow: npt.NDArray[np.bool_]
    underflow: npt.NDArray[np.bool_]


def _validate_and_shape(
    weights_q88: npt.ArrayLike,
    inputs_q1616: npt.ArrayLike,
    n_outputs: int,
    n_inputs: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Validate shapes and the accumulation bound; return (weights, inputs) int64."""
    if n_outputs <= 0 or n_inputs <= 0:
        raise ValueError(f"n_outputs and n_inputs must be positive, got {n_outputs}, {n_inputs}")
    weights = np.ascontiguousarray(weights_q88, dtype=np.int64).reshape(-1)
    inputs = np.ascontiguousarray(inputs_q1616, dtype=np.int64).reshape(-1)
    if weights.size != n_outputs * n_inputs:
        raise ValueError(
            "weights length must be n_outputs * n_inputs "
            f"({n_outputs * n_inputs}): got {weights.size}"
        )
    if inputs.size % n_inputs != 0:
        raise ValueError(f"inputs length {inputs.size} is not a multiple of n_inputs {n_inputs}")
    if weights.size and inputs.size:
        bound = int(np.abs(weights).max()) * int(np.abs(inputs).max()) * n_inputs
        if bound > _I64_MAX:
            raise ValueError(
                "mixed-dense accumulation can exceed int64; reduce n_inputs or value range "
                f"(conservative bound {bound} > {_I64_MAX})"
            )
    return weights, inputs


def mixed_dense_forward_batch_q88_q1616(
    weights_q88: npt.ArrayLike,
    inputs_q1616: npt.ArrayLike,
    n_outputs: int,
    n_inputs: int,
) -> MixedDenseBatchResult:
    """Pure-Python batched mixed-precision dense MAC — the bit-true floor reference.

    Parameters
    ----------
    weights_q88 : array_like
        Row-major ``n_outputs * n_inputs`` Q8.8 weights (``int16`` range).
    inputs_q1616 : array_like
        Row-major ``n_batch * n_inputs`` Q16.16 input codes (``int32`` range).
    n_outputs : int
        Number of dense output channels.
    n_inputs : int
        Number of dense input channels.

    Returns
    -------
    MixedDenseBatchResult
        Per-batch, per-output saturated Q16.16 codes with overflow/underflow flags.

    Raises
    ------
    ValueError
        If shapes are inconsistent or the accumulation can exceed ``int64``.
    """
    weights, inputs = _validate_and_shape(weights_q88, inputs_q1616, n_outputs, n_inputs)
    weight_matrix = weights.reshape(n_outputs, n_inputs)
    input_matrix = inputs.reshape(-1, n_inputs)

    raw = input_matrix @ weight_matrix.T
    scaled = raw >> WEIGHT_FRACTION
    overflow = (scaled < ACCUM_MIN) | (scaled > ACCUM_MAX)
    underflow = (raw != 0) & (scaled == 0) & ~overflow
    outputs = np.clip(scaled, ACCUM_MIN, ACCUM_MAX).astype(np.int32)
    return MixedDenseBatchResult(
        outputs_q1616=outputs,
        overflow=overflow.astype(np.bool_),
        underflow=underflow.astype(np.bool_),
    )


def _result_from_mapping(
    payload: Mapping[str, npt.ArrayLike], n_batch: int, n_outputs: int
) -> MixedDenseBatchResult:
    """Convert a backend dict payload into a typed result with ``(n_batch, n_outputs)`` arrays."""
    shape = (n_batch, n_outputs)
    return MixedDenseBatchResult(
        outputs_q1616=np.ascontiguousarray(payload["outputs_q1616"], dtype=np.int32).reshape(shape),
        overflow=np.ascontiguousarray(payload["overflow"], dtype=np.bool_).reshape(shape),
        underflow=np.ascontiguousarray(payload["underflow"], dtype=np.bool_).reshape(shape),
    )


def _backend_python(
    weights_q88: npt.ArrayLike, inputs_q1616: npt.ArrayLike, n_outputs: int, n_inputs: int
) -> MixedDenseBatchResult:
    return mixed_dense_forward_batch_q88_q1616(weights_q88, inputs_q1616, n_outputs, n_inputs)


def _n_batch(inputs_q1616: npt.ArrayLike, n_inputs: int) -> int:
    return int(np.asarray(inputs_q1616).size // n_inputs)


def _backend_rust(
    weights_q88: npt.ArrayLike, inputs_q1616: npt.ArrayLike, n_outputs: int, n_inputs: int
) -> MixedDenseBatchResult:
    from sc_neurocore_engine import py_mixed_dense_forward_batch_q88_q1616

    payload = py_mixed_dense_forward_batch_q88_q1616(
        np.ascontiguousarray(weights_q88, dtype=np.int16).reshape(-1),
        np.ascontiguousarray(inputs_q1616, dtype=np.int32).reshape(-1),
        int(n_outputs),
        int(n_inputs),
    )
    return _result_from_mapping(payload, _n_batch(inputs_q1616, n_inputs), n_outputs)


def _backend_julia(
    weights_q88: npt.ArrayLike, inputs_q1616: npt.ArrayLike, n_outputs: int, n_inputs: int
) -> MixedDenseBatchResult:
    from sc_neurocore.accel.julia.mixed_dense import mixed_dense_forward_batch as julia_batch

    payload = julia_batch(weights_q88, inputs_q1616, n_outputs, n_inputs)
    return _result_from_mapping(payload, _n_batch(inputs_q1616, n_inputs), n_outputs)


def _backend_go(
    weights_q88: npt.ArrayLike, inputs_q1616: npt.ArrayLike, n_outputs: int, n_inputs: int
) -> MixedDenseBatchResult:
    from sc_neurocore.accel.go.mixed_dense import mixed_dense_forward_batch as go_batch

    payload = go_batch(weights_q88, inputs_q1616, n_outputs, n_inputs)
    return _result_from_mapping(payload, _n_batch(inputs_q1616, n_inputs), n_outputs)


def _backend_mojo(
    weights_q88: npt.ArrayLike, inputs_q1616: npt.ArrayLike, n_outputs: int, n_inputs: int
) -> MixedDenseBatchResult:
    from sc_neurocore.accel.mojo.mixed_dense import mixed_dense_forward_batch as mojo_batch

    payload = mojo_batch(weights_q88, inputs_q1616, n_outputs, n_inputs)
    return _result_from_mapping(payload, _n_batch(inputs_q1616, n_inputs), n_outputs)


_BACKEND_DISPATCH: dict[
    str,
    Callable[[npt.ArrayLike, npt.ArrayLike, int, int], MixedDenseBatchResult],
] = {
    "python": _backend_python,
    "rust": _backend_rust,
    "julia": _backend_julia,
    "go": _backend_go,
    "mojo": _backend_mojo,
}


def available_backends() -> dict[str, bool]:
    """Probe which acceleration backends can run the mixed-dense kernel.

    Returns
    -------
    dict
        Mapping of backend name to availability, in fastest-first order. The
        ``python`` floor is always ``True``.
    """
    status: dict[str, bool] = {}
    probe_weights = np.ones(1, dtype=np.int16)
    probe_inputs = np.ones(1, dtype=np.int32)
    for name in FASTEST_FIRST_BACKENDS:
        if name == "python":
            status[name] = True
            continue
        try:
            _BACKEND_DISPATCH[name](probe_weights, probe_inputs, 1, 1)
            status[name] = True
        except (ImportError, OSError, RuntimeError, FileNotFoundError):
            status[name] = False
    return status


def mixed_dense_forward_batch(
    weights_q88: npt.ArrayLike,
    inputs_q1616: npt.ArrayLike,
    n_outputs: int,
    n_inputs: int,
    *,
    backend: str = "auto",
) -> MixedDenseBatchResult:
    """Run the batched mixed-precision dense MAC through the fastest available backend.

    Parameters
    ----------
    weights_q88, inputs_q1616, n_outputs, n_inputs
        See :func:`mixed_dense_forward_batch_q88_q1616`.
    backend : str, optional
        ``"auto"`` (default) selects the fastest available backend in
        :data:`FASTEST_FIRST_BACKENDS` order. A specific name forces that backend.

    Returns
    -------
    MixedDenseBatchResult
        Bit-identical to the Python floor for every backend.

    Raises
    ------
    ValueError
        If ``backend`` is not a known name.
    ImportError
        If an explicitly requested accelerator backend is unavailable.
    """
    if backend != "auto":
        if backend not in _BACKEND_DISPATCH:
            raise ValueError(
                f"unknown backend {backend!r}; choose from {('auto', *FASTEST_FIRST_BACKENDS)}"
            )
        return _BACKEND_DISPATCH[backend](weights_q88, inputs_q1616, n_outputs, n_inputs)
    for name in FASTEST_FIRST_BACKENDS:
        if name == "python":
            break
        try:
            return _BACKEND_DISPATCH[name](weights_q88, inputs_q1616, n_outputs, n_inputs)
        except (ImportError, OSError, RuntimeError, FileNotFoundError):
            continue
    return _backend_python(weights_q88, inputs_q1616, n_outputs, n_inputs)


__all__ = [
    "ACCUM_MAX",
    "ACCUM_MIN",
    "FASTEST_FIRST_BACKENDS",
    "WEIGHT_FRACTION",
    "MixedDenseBatchResult",
    "available_backends",
    "mixed_dense_forward_batch",
    "mixed_dense_forward_batch_q88_q1616",
]
