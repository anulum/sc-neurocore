# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DCLS-max Q8.8 tent kernel — bit-true reference + dispatch

"""Bit-true Q8.8 reference for the DCLS-max triangular (tent) weighting kernel.

Dilated convolution with learnable spacings (DCLS), ``max`` variant, after
Khalfaoui-Hassani, Pellegrini & Masquelier (2023), *Dilated convolution with
learnable spacings*, ICLR. Each delay tap is weighted by a triangular kernel
centred on the learnable spacing ``centre`` with half-width ``sigma``; the
synapse contracts the active spike taps through that kernel into a fixed-point
accumulator.

The whole computation is integer Q8.8 arithmetic with a Q16.16 accumulator and
saturating output, identical to the synthesisable RTL in ``hdl/sc_dcls_*.v`` and
the engine reference in ``engine/src/scpn/dcls.rs``. Because every operation is
exact integer arithmetic, all acceleration backends (Rust, Mojo, Julia, Go and
this Python floor) agree bit-for-bit; the parity tolerance is exactly zero.

Fixed-point contract
--------------------
* ``centre_q88``, ``sigma_q88``, weights and the tent gate are signed Q8.8
  (one unit = ``1 / 256``); ``256`` represents ``1.0``.
* The per-synapse accumulator runs in Q16.16 and saturates to ``int32``.
* The Q8.8 output is the arithmetic right shift of the accumulator by the
  fractional width, saturated to ``int16``.

The single-synapse :func:`dcls_max_forward_q88` mirrors the engine contract one
contraction at a time; :func:`dcls_max_forward_batch_q88` applies it across many
output channels with per-channel learnable ``centre``/``sigma``, which is the
shape the acceleration backends and the benchmark exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import FASTEST_FIRST_BACKENDS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

#: Fractional bit width of the Q8.8 DCLS contract.
DEFAULT_FRACTION = 8
#: Q8.8 representation of ``1.0`` (the tent peak gate).
Q88_ONE = 1 << DEFAULT_FRACTION
_I16_MAX = (1 << 15) - 1
_I16_MIN = -(1 << 15)
_I32_MAX = (1 << 31) - 1
_I32_MIN = -(1 << 31)
#: Accumulator magnitudes at which the Q8.8 output saturates.
I16_MAX_Q16_16 = _I16_MAX << DEFAULT_FRACTION
I16_MIN_Q16_16 = _I16_MIN << DEFAULT_FRACTION


@dataclass(frozen=True)
class DclsForwardResult:
    """Result of one DCLS-max tent contraction.

    Attributes
    ----------
    output_q88 : int
        Saturated Q8.8 synapse output.
    accumulator_q16_16 : int
        Saturated Q16.16 accumulator before the output shift.
    overflow : bool
        ``True`` when accumulator or output saturation occurred.
    active_tap_count : int
        Number of non-zero spike taps consumed by the contraction.
    max_gate_q88 : int
        Largest Q8.8 tent gate applied to an active spike tap.
    """

    output_q88: int
    accumulator_q16_16: int
    overflow: bool
    active_tap_count: int
    max_gate_q88: int


@dataclass(frozen=True)
class DclsBatchResult:
    """Per-channel results of a batched DCLS-max tent contraction.

    Each array is indexed by output channel and has length ``B`` (the number of
    ``centre``/``sigma`` pairs supplied).

    Attributes
    ----------
    outputs_q88 : numpy.ndarray
        Saturated Q8.8 outputs, ``int16``.
    accumulators_q16_16 : numpy.ndarray
        Saturated Q16.16 accumulators, ``int32``.
    overflow : numpy.ndarray
        Saturation flags, ``bool_``.
    active_tap_counts : numpy.ndarray
        Active spike-tap counts per channel, ``int64``.
    max_gates_q88 : numpy.ndarray
        Largest applied tent gate per channel, ``int16``.
    """

    outputs_q88: npt.NDArray[np.int16]
    accumulators_q16_16: npt.NDArray[np.int32]
    overflow: npt.NDArray[np.bool_]
    active_tap_counts: npt.NDArray[np.int64]
    max_gates_q88: npt.NDArray[np.int16]


def tent_gate_q88(tap_index: int, centre_q88: int, sigma_q88: int) -> int:
    """Return the Q8.8 triangular tent gate for a delay tap.

    The gate is ``max(0, 1 - |delay - centre| / sigma)`` evaluated in Q8.8 with
    truncating integer division, so it matches the synthesisable RTL exactly.

    Parameters
    ----------
    tap_index : int
        Zero-based delay tap; the delay is ``tap_index`` in Q8.8 whole units.
    centre_q88 : int
        Learnable tent centre in Q8.8.
    sigma_q88 : int
        Tent half-width in Q8.8; must be positive.

    Returns
    -------
    int
        Tent gate in ``[0, 256]`` Q8.8 (``256`` is ``1.0``).

    Raises
    ------
    ValueError
        If ``sigma_q88`` is not positive or ``tap_index`` is negative.
    """
    if sigma_q88 <= 0:
        raise ValueError(f"DCLS tent sigma must be positive, got {sigma_q88}")
    if tap_index < 0:
        raise ValueError(f"DCLS tap index must be non-negative, got {tap_index}")
    delay_q88 = tap_index << DEFAULT_FRACTION
    distance_q88 = abs(delay_q88 - centre_q88)
    if distance_q88 >= sigma_q88:
        return 0
    gate = ((sigma_q88 - distance_q88) << DEFAULT_FRACTION) // sigma_q88
    return min(Q88_ONE, max(0, gate))


def _saturate_contraction(accumulator: int) -> tuple[int, int, bool]:
    """Saturate a raw integer accumulator into the Q16.16/Q8.8 output contract."""
    accumulator_q16_16 = min(_I32_MAX, max(_I32_MIN, accumulator))
    accumulator_overflow = accumulator_q16_16 != accumulator
    if accumulator > I16_MAX_Q16_16:
        return _I16_MAX, accumulator_q16_16, True
    if accumulator < I16_MIN_Q16_16:
        return _I16_MIN, accumulator_q16_16, True
    output_q88 = accumulator >> DEFAULT_FRACTION
    return output_q88, accumulator_q16_16, accumulator_overflow


def dcls_max_forward_q88(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centre_q88: int,
    sigma_q88: int,
) -> DclsForwardResult:
    """Run one DCLS-max tent contraction in bit-true Q8.8 arithmetic.

    Parameters
    ----------
    spikes : array_like
        Per-tap spike flags; any non-zero entry marks an active tap.
    weights_q88 : array_like
        Per-tap synaptic weights in Q8.8, same length as ``spikes``.
    centre_q88 : int
        Learnable tent centre in Q8.8.
    sigma_q88 : int
        Tent half-width in Q8.8; must be positive.

    Returns
    -------
    DclsForwardResult
        Saturated Q8.8 output, Q16.16 accumulator and saturation diagnostics.

    Raises
    ------
    ValueError
        If the inputs are empty, length-mismatched or ``sigma_q88`` is not
        positive.
    """
    spike_arr = np.ascontiguousarray(spikes, dtype=np.int64).reshape(-1)
    weight_arr = np.ascontiguousarray(weights_q88, dtype=np.int64).reshape(-1)
    if spike_arr.size == 0:
        raise ValueError("DCLS forward pass requires at least one tap")
    if spike_arr.size != weight_arr.size:
        raise ValueError(
            f"DCLS spike/weight length mismatch: spikes={spike_arr.size}, weights={weight_arr.size}"
        )
    if sigma_q88 <= 0:
        raise ValueError(f"DCLS tent sigma must be positive, got {sigma_q88}")

    accumulator = 0
    active_tap_count = 0
    max_gate_q88 = 0
    for tap_index in range(spike_arr.size):
        if spike_arr[tap_index] == 0:
            continue
        active_tap_count += 1
        gate_q88 = tent_gate_q88(tap_index, int(centre_q88), int(sigma_q88))
        max_gate_q88 = max(max_gate_q88, gate_q88)
        accumulator += int(weight_arr[tap_index]) * gate_q88

    output_q88, accumulator_q16_16, overflow = _saturate_contraction(accumulator)
    return DclsForwardResult(
        output_q88=int(output_q88),
        accumulator_q16_16=int(accumulator_q16_16),
        overflow=bool(overflow),
        active_tap_count=active_tap_count,
        max_gate_q88=int(max_gate_q88),
    )


def _validate_batch(
    spikes: npt.NDArray[np.int64],
    weights: npt.NDArray[np.int64],
    centres: npt.NDArray[np.int64],
    sigmas: npt.NDArray[np.int64],
    n_taps: int,
) -> int:
    """Validate batch shapes and return the channel count ``B``."""
    if n_taps <= 0:
        raise ValueError(f"n_taps must be positive, got {n_taps}")
    if centres.size != sigmas.size:
        raise ValueError(f"centres/sigmas length mismatch: {centres.size} vs {sigmas.size}")
    n_channels = int(centres.size)
    if n_channels == 0:
        raise ValueError("DCLS batch requires at least one output channel")
    expected = n_channels * n_taps
    if spikes.size != expected or weights.size != expected:
        raise ValueError(
            "DCLS batch spike/weight length must be n_channels * n_taps "
            f"({expected}): spikes={spikes.size}, weights={weights.size}"
        )
    if np.any(sigmas <= 0):
        raise ValueError("every DCLS sigma must be positive")
    return n_channels


def dcls_max_forward_batch_q88(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centres_q88: npt.ArrayLike,
    sigmas_q88: npt.ArrayLike,
    n_taps: int,
) -> DclsBatchResult:
    """Pure-Python batched DCLS-max contraction — the bit-true floor reference.

    Each output channel ``b`` contracts its own ``n_taps``-long spike/weight row
    through a tent kernel with channel-specific learnable ``centre``/``sigma``.

    Parameters
    ----------
    spikes : array_like
        Flattened ``n_channels * n_taps`` spike flags (row-major per channel).
    weights_q88 : array_like
        Flattened ``n_channels * n_taps`` Q8.8 weights (row-major per channel).
    centres_q88 : array_like
        Per-channel learnable tent centres in Q8.8, length ``n_channels``.
    sigmas_q88 : array_like
        Per-channel tent half-widths in Q8.8, length ``n_channels``; positive.
    n_taps : int
        Number of delay taps per channel.

    Returns
    -------
    DclsBatchResult
        Per-channel saturated outputs, accumulators and diagnostics.

    Raises
    ------
    ValueError
        If shapes are inconsistent or any ``sigma`` is non-positive.
    """
    spike_arr = np.ascontiguousarray(spikes, dtype=np.int64).reshape(-1)
    weight_arr = np.ascontiguousarray(weights_q88, dtype=np.int64).reshape(-1)
    centre_arr = np.ascontiguousarray(centres_q88, dtype=np.int64).reshape(-1)
    sigma_arr = np.ascontiguousarray(sigmas_q88, dtype=np.int64).reshape(-1)
    n_channels = _validate_batch(spike_arr, weight_arr, centre_arr, sigma_arr, n_taps)

    outputs = np.empty(n_channels, dtype=np.int16)
    accumulators = np.empty(n_channels, dtype=np.int32)
    overflow = np.empty(n_channels, dtype=np.bool_)
    active_counts = np.empty(n_channels, dtype=np.int64)
    max_gates = np.empty(n_channels, dtype=np.int16)

    for channel in range(n_channels):
        base = channel * n_taps
        centre = int(centre_arr[channel])
        sigma = int(sigma_arr[channel])
        accumulator = 0
        active = 0
        max_gate = 0
        for tap_index in range(n_taps):
            if spike_arr[base + tap_index] == 0:
                continue
            active += 1
            gate = tent_gate_q88(tap_index, centre, sigma)
            if gate > max_gate:
                max_gate = gate
            accumulator += int(weight_arr[base + tap_index]) * gate
        output_q88, accumulator_q16_16, overflowed = _saturate_contraction(accumulator)
        outputs[channel] = output_q88
        accumulators[channel] = accumulator_q16_16
        overflow[channel] = overflowed
        active_counts[channel] = active
        max_gates[channel] = max_gate

    return DclsBatchResult(
        outputs_q88=outputs,
        accumulators_q16_16=accumulators,
        overflow=overflow,
        active_tap_counts=active_counts,
        max_gates_q88=max_gates,
    )


def _result_from_mapping(payload: Mapping[str, npt.ArrayLike]) -> DclsBatchResult:
    """Convert a backend dict payload into a typed :class:`DclsBatchResult`."""
    return DclsBatchResult(
        outputs_q88=np.ascontiguousarray(payload["outputs_q88"], dtype=np.int16),
        accumulators_q16_16=np.ascontiguousarray(payload["accumulators_q16_16"], dtype=np.int32),
        overflow=np.ascontiguousarray(payload["overflow"], dtype=np.bool_),
        active_tap_counts=np.ascontiguousarray(payload["active_tap_counts"], dtype=np.int64),
        max_gates_q88=np.ascontiguousarray(payload["max_gates_q88"], dtype=np.int16),
    )


def _backend_python(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centres_q88: npt.ArrayLike,
    sigmas_q88: npt.ArrayLike,
    n_taps: int,
) -> DclsBatchResult:
    return dcls_max_forward_batch_q88(spikes, weights_q88, centres_q88, sigmas_q88, n_taps)


def _backend_rust(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centres_q88: npt.ArrayLike,
    sigmas_q88: npt.ArrayLike,
    n_taps: int,
) -> DclsBatchResult:
    from sc_neurocore_engine import py_dcls_max_forward_batch_q88

    payload = py_dcls_max_forward_batch_q88(
        np.ascontiguousarray(spikes, dtype=np.uint8).reshape(-1),
        np.ascontiguousarray(weights_q88, dtype=np.int16).reshape(-1),
        np.ascontiguousarray(centres_q88, dtype=np.int16).reshape(-1),
        np.ascontiguousarray(sigmas_q88, dtype=np.int16).reshape(-1),
        int(n_taps),
    )
    return _result_from_mapping(payload)


def _backend_julia(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centres_q88: npt.ArrayLike,
    sigmas_q88: npt.ArrayLike,
    n_taps: int,
) -> DclsBatchResult:
    from sc_neurocore.accel.julia.scpn import dcls_max_forward_batch as julia_batch

    return _result_from_mapping(julia_batch(spikes, weights_q88, centres_q88, sigmas_q88, n_taps))


def _backend_go(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centres_q88: npt.ArrayLike,
    sigmas_q88: npt.ArrayLike,
    n_taps: int,
) -> DclsBatchResult:
    from sc_neurocore.accel.go.dcls_tent import dcls_max_forward_batch as go_batch

    return _result_from_mapping(go_batch(spikes, weights_q88, centres_q88, sigmas_q88, n_taps))


def _backend_mojo(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centres_q88: npt.ArrayLike,
    sigmas_q88: npt.ArrayLike,
    n_taps: int,
) -> DclsBatchResult:
    from sc_neurocore.accel.mojo.dcls_tent import dcls_max_forward_batch as mojo_batch

    return _result_from_mapping(mojo_batch(spikes, weights_q88, centres_q88, sigmas_q88, n_taps))


_BACKEND_DISPATCH: dict[
    str,
    Callable[
        [npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, int],
        DclsBatchResult,
    ],
] = {
    "python": _backend_python,
    "rust": _backend_rust,
    "julia": _backend_julia,
    "go": _backend_go,
    "mojo": _backend_mojo,
}


def available_backends() -> dict[str, bool]:
    """Probe which acceleration backends can run the DCLS batch kernel.

    Returns
    -------
    dict
        Mapping of backend name to availability, in fastest-first order. The
        ``python`` floor is always ``True``.
    """
    status: dict[str, bool] = {}
    for name in FASTEST_FIRST_BACKENDS:
        if name == "python":
            status[name] = True
            continue
        try:
            _BACKEND_DISPATCH[name](
                np.zeros(1, dtype=np.uint8),
                np.zeros(1, dtype=np.int16),
                np.zeros(1, dtype=np.int16),
                np.ones(1, dtype=np.int16),
                1,
            )
            status[name] = True
        except (ImportError, OSError, RuntimeError, FileNotFoundError):
            status[name] = False
    return status


def dcls_max_forward_batch(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centres_q88: npt.ArrayLike,
    sigmas_q88: npt.ArrayLike,
    n_taps: int,
    *,
    backend: str = "auto",
) -> DclsBatchResult:
    """Run the batched DCLS-max tent kernel through the fastest available backend.

    Parameters
    ----------
    spikes, weights_q88, centres_q88, sigmas_q88, n_taps
        See :func:`dcls_max_forward_batch_q88`.
    backend : str, optional
        ``"auto"`` (default) picks the fastest available backend in
        :data:`FASTEST_FIRST_BACKENDS` order. A specific name (``"rust"``,
        ``"mojo"``, ``"julia"``, ``"go"``, ``"python"``) forces that backend.

    Returns
    -------
    DclsBatchResult
        Identical to the pure-Python floor for every backend (bit-exact).

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
        return _BACKEND_DISPATCH[backend](spikes, weights_q88, centres_q88, sigmas_q88, n_taps)
    for name in FASTEST_FIRST_BACKENDS:
        if name == "python":
            break
        try:
            return _BACKEND_DISPATCH[name](spikes, weights_q88, centres_q88, sigmas_q88, n_taps)
        except (ImportError, OSError, RuntimeError, FileNotFoundError):
            continue
    return _backend_python(spikes, weights_q88, centres_q88, sigmas_q88, n_taps)


__all__ = [
    "DEFAULT_FRACTION",
    "FASTEST_FIRST_BACKENDS",
    "I16_MAX_Q16_16",
    "I16_MIN_Q16_16",
    "Q88_ONE",
    "DclsBatchResult",
    "DclsForwardResult",
    "available_backends",
    "dcls_max_forward_batch",
    "dcls_max_forward_batch_q88",
    "dcls_max_forward_q88",
    "tent_gate_q88",
]
