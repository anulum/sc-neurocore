# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go-backed ADC-to-spike encoder (ctypes dispatch)

"""Python entry point for the Go-compiled ADC-to-spike window encoder.

Build::

    cd src/sc_neurocore/accel/go/adc_to_spike
    PATH=/usr/local/go/bin:$PATH go build -buildmode=c-shared \\
        -o libadc_to_spike.so adc_to_spike.go

The ``.so`` is platform-specific and gitignored; the ``.go`` source and the
generated ``.h`` header are tracked. ``_HAS_GO_ADC_TO_SPIKE`` is ``True`` iff the
shared library is present at import time.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_LIB_PATH = Path(__file__).resolve().parent / "libadc_to_spike.so"
_lib: ctypes.CDLL | None

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
    _lib.adc_to_spike_windows_c.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # n_windows, adc_width, q_int, q_frac
        ctypes.c_int,
        ctypes.c_int,  # decimation, signed_input
        ctypes.c_longlong,  # threshold_q
        ctypes.c_void_p,  # samples
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,  # window_values, spike_counts, polarities
    ]
    _lib.adc_to_spike_windows_c.restype = ctypes.c_int
    _HAS_GO_ADC_TO_SPIKE = True
except OSError:
    _lib = None
    _HAS_GO_ADC_TO_SPIKE = False


def adc_to_spike_windows(
    samples: npt.ArrayLike,
    adc_width: int,
    q_int: int,
    q_frac: int,
    decimation: int,
    signed_input: int,
    threshold_q: int,
) -> dict[str, npt.NDArray[Any]]:
    """Go-accelerated per-window ADC-to-spike encode.

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

    Raises
    ------
    ImportError
        If ``libadc_to_spike.so`` is not built.
    ValueError
        If fewer than ``decimation`` samples are given or the config is invalid.
    """
    if _lib is None:
        raise ImportError(
            f"libadc_to_spike.so not built. Run: cd {_LIB_PATH.parent} && "
            f"go build -buildmode=c-shared -o {_LIB_PATH.name} adc_to_spike.go"
        )
    sample_arr = np.ascontiguousarray(samples, dtype=np.int64).reshape(-1)
    n_windows = int(sample_arr.size) // int(decimation)
    if n_windows == 0:
        raise ValueError(f"need at least decimation={decimation} samples, got {sample_arr.size}")

    window_values: npt.NDArray[np.int32] = np.empty(n_windows, dtype=np.int32)
    spike_counts: npt.NDArray[np.int32] = np.empty(n_windows, dtype=np.int32)
    polarities: npt.NDArray[np.uint8] = np.empty(n_windows, dtype=np.uint8)

    rc = _lib.adc_to_spike_windows_c(
        ctypes.c_int(n_windows),
        ctypes.c_int(int(adc_width)),
        ctypes.c_int(int(q_int)),
        ctypes.c_int(int(q_frac)),
        ctypes.c_int(int(decimation)),
        ctypes.c_int(int(signed_input)),
        ctypes.c_longlong(int(threshold_q)),
        sample_arr.ctypes.data,
        window_values.ctypes.data,
        spike_counts.ctypes.data,
        polarities.ctypes.data,
    )
    if rc != 0:
        raise ValueError(f"adc_to_spike_windows_c returned non-zero: {rc}")

    return {
        "window_values_q": window_values,
        "spike_counts": spike_counts,
        "polarities": polarities.astype(np.bool_),
    }


__all__ = ["adc_to_spike_windows"]
