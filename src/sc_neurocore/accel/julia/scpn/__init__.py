# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia-backed SCPN kernels (juliacall dispatch)

"""Python entry points for the Julia SCPN kernels in this directory.

The DCLS-max tent kernel is exact integer Q8.8 arithmetic, so the Julia backend
returns results that are bit-identical to the Rust, Go, Mojo and Python
references. Julia boots lazily via ``juliacall`` on first call (cold JIT warm-up
is a few seconds; warm calls are sub-millisecond).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    from juliacall import Main as _jl

    _HAS_JULIA_SCPN = True
except ImportError:
    _jl = None
    _HAS_JULIA_SCPN = False


_KERNEL_DIR = Path(__file__).resolve().parent
_DCLS_LOADED = False


def _ensure_dcls_loaded() -> Any:
    """Include ``dcls.jl`` into Julia ``Main`` on first use; return the module."""
    global _DCLS_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _DCLS_LOADED:
        jl_path = _KERNEL_DIR / "dcls.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"dcls.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _DCLS_LOADED = True
    return _jl.DclsTentAccel


def dcls_max_forward_batch(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centres_q88: npt.ArrayLike,
    sigmas_q88: npt.ArrayLike,
    n_taps: int,
) -> dict[str, npt.NDArray[Any]]:
    """Julia-accelerated batched DCLS-max tent kernel.

    Parameters
    ----------
    spikes, weights_q88, centres_q88, sigmas_q88, n_taps
        See :func:`sc_neurocore.scpn.dcls_tent_kernel.dcls_max_forward_batch_q88`.

    Returns
    -------
    dict
        Mapping with ``outputs_q88``, ``accumulators_q16_16``, ``overflow``,
        ``active_tap_counts`` and ``max_gates_q88`` arrays — bit-identical to the
        Python floor.
    """
    mod = _ensure_dcls_loaded()
    spike_arr = np.ascontiguousarray(spikes, dtype=np.uint8).reshape(-1)
    weight_arr = np.ascontiguousarray(weights_q88, dtype=np.int16).reshape(-1)
    centre_arr = np.ascontiguousarray(centres_q88, dtype=np.int16).reshape(-1)
    sigma_arr = np.ascontiguousarray(sigmas_q88, dtype=np.int16).reshape(-1)
    n_channels = int(centre_arr.size)
    if sigma_arr.size != n_channels:
        raise ValueError(f"centres/sigmas length mismatch: {n_channels} vs {sigma_arr.size}")
    expected = n_channels * int(n_taps)
    if spike_arr.size != expected or weight_arr.size != expected:
        raise ValueError(
            "DCLS batch spike/weight length must be n_channels * n_taps "
            f"({expected}): spikes={spike_arr.size}, weights={weight_arr.size}"
        )

    outputs: npt.NDArray[np.int16] = np.empty(n_channels, dtype=np.int16)
    accumulators: npt.NDArray[np.int32] = np.empty(n_channels, dtype=np.int32)
    overflow: npt.NDArray[np.uint8] = np.empty(n_channels, dtype=np.uint8)
    active_counts: npt.NDArray[np.int64] = np.empty(n_channels, dtype=np.int64)
    max_gates: npt.NDArray[np.int16] = np.empty(n_channels, dtype=np.int16)

    mod.dcls_max_forward_batch_q88_b(
        spike_arr,
        weight_arr,
        centre_arr,
        sigma_arr,
        int(n_taps),
        outputs,
        accumulators,
        overflow,
        active_counts,
        max_gates,
    )
    return {
        "outputs_q88": outputs,
        "accumulators_q16_16": accumulators,
        "overflow": overflow.astype(np.bool_),
        "active_tap_counts": active_counts,
        "max_gates_q88": max_gates,
    }


__all__ = ["dcls_max_forward_batch"]
