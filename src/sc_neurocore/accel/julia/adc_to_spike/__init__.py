# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia-backed ADC-to-spike encoder (juliacall dispatch)

"""Python entry point for the Julia ADC-to-spike window encoder.

The integer per-window encode is exact, so the Julia backend is bit-identical to
the Rust, Go, Mojo and Python references. Julia boots lazily via ``juliacall``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    from juliacall import Main as _jl

    _HAS_JULIA_ADC_TO_SPIKE = True
except ImportError:
    _jl = None
    _HAS_JULIA_ADC_TO_SPIKE = False


_KERNEL_DIR = Path(__file__).resolve().parent
_LOADED = False


def _ensure_loaded() -> Any:
    """Include ``adc_to_spike.jl`` into Julia ``Main`` on first use; return the module."""
    global _LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _LOADED:
        jl_path = _KERNEL_DIR / "adc_to_spike.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"adc_to_spike.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _LOADED = True
    return _jl.AdcToSpikeAccel


def adc_to_spike_windows(
    samples: npt.ArrayLike,
    adc_width: int,
    q_int: int,
    q_frac: int,
    decimation: int,
    signed_input: int,
    threshold_q: int,
) -> dict[str, npt.NDArray[Any]]:
    """Julia-accelerated per-window ADC-to-spike encode.

    Parameters
    ----------
    samples : array_like
        Raw ADC samples.
    adc_width, q_int, q_frac, decimation, signed_input, threshold_q
        Fixed-point/decimation contract (``signed_input`` is ``0`` or ``1``).

    Returns
    -------
    dict
        ``window_values_q``, ``spike_counts`` and ``polarities`` arrays —
        bit-identical to the Python floor.
    """
    mod = _ensure_loaded()
    sample_arr = np.ascontiguousarray(samples, dtype=np.int64).reshape(-1)
    n_windows = int(sample_arr.size) // int(decimation)
    if n_windows == 0:
        raise ValueError(f"need at least decimation={decimation} samples, got {sample_arr.size}")

    window_values: npt.NDArray[np.int32] = np.empty(n_windows, dtype=np.int32)
    spike_counts: npt.NDArray[np.int32] = np.empty(n_windows, dtype=np.int32)
    polarities: npt.NDArray[np.uint8] = np.empty(n_windows, dtype=np.uint8)

    mod.adc_to_spike_windows_b(
        sample_arr,
        int(adc_width),
        int(q_int),
        int(q_frac),
        int(decimation),
        int(signed_input),
        int(threshold_q),
        window_values,
        spike_counts,
        polarities,
    )
    return {
        "window_values_q": window_values,
        "spike_counts": spike_counts,
        "polarities": polarities.astype(np.bool_),
    }


__all__ = ["adc_to_spike_windows"]
