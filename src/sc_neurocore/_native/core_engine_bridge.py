# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python ctypes bridge for core_engine Rust C-FFI

"""C-FFI bridge to Rust ``libcore_engine.so`` for SC arithmetic hot paths.

Provides:
- sc_multiply, sc_mux, sc_popcount, sc_popcount64
- sc_and_packed, sc_mux_packed, sc_popcount_packed
- sc_scc_packed, sc_cordiv_packed
- lfsr_create, lfsr_step, lfsr_encode, lfsr_destroy
"""

from __future__ import annotations

import ctypes as _ct
import pathlib as _pl

_LIB_PATH = _pl.Path(__file__).parent / "libcore_engine.so"
_HAS_CORE_ENGINE = False
_lib = None

if _LIB_PATH.exists():
    try:
        _lib = _ct.CDLL(str(_LIB_PATH))

        # --- scalar ops ---
        _lib.sc_multiply.argtypes = [_ct.c_uint32, _ct.c_uint32]
        _lib.sc_multiply.restype = _ct.c_uint32

        _lib.sc_mux.argtypes = [_ct.c_uint32, _ct.c_uint32, _ct.c_uint32]
        _lib.sc_mux.restype = _ct.c_uint32

        _lib.sc_popcount.argtypes = [_ct.c_uint32]
        _lib.sc_popcount.restype = _ct.c_uint32

        _lib.sc_popcount64.argtypes = [_ct.c_uint64]
        _lib.sc_popcount64.restype = _ct.c_uint32

        _lib.sc_saturating_sub.argtypes = [_ct.c_uint32, _ct.c_uint32]
        _lib.sc_saturating_sub.restype = _ct.c_uint32

        # --- packed SIMD ops ---
        _lib.sc_and_packed.argtypes = [
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.c_size_t,
        ]
        _lib.sc_and_packed.restype = None

        _lib.sc_mux_packed.argtypes = [
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.c_size_t,
        ]
        _lib.sc_mux_packed.restype = None

        _lib.sc_popcount_packed.argtypes = [_ct.POINTER(_ct.c_uint64), _ct.c_size_t]
        _lib.sc_popcount_packed.restype = _ct.c_uint64

        _lib.sc_scc_packed.argtypes = [
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.c_size_t,
        ]
        _lib.sc_scc_packed.restype = _ct.c_double

        _lib.sc_cordiv_packed.argtypes = [
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.c_size_t,
        ]
        _lib.sc_cordiv_packed.restype = None

        # --- LFSR ---
        _lib.lfsr_create.argtypes = [_ct.c_uint16]
        _lib.lfsr_create.restype = _ct.c_void_p

        _lib.lfsr_step.argtypes = [_ct.c_void_p]
        _lib.lfsr_step.restype = _ct.c_uint16

        _lib.lfsr_encode.argtypes = [
            _ct.c_void_p,
            _ct.c_uint16,
            _ct.c_uint32,
        ]
        _lib.lfsr_encode.restype = _ct.POINTER(_ct.c_uint64)

        _lib.lfsr_destroy.argtypes = [_ct.c_void_p]
        _lib.lfsr_destroy.restype = None

        _lib.bitstream_free.argtypes = [_ct.POINTER(_ct.c_uint64), _ct.c_size_t]
        _lib.bitstream_free.restype = None

        _HAS_CORE_ENGINE = True
    except OSError:
        pass


def is_available() -> bool:
    """Return True if the Rust core engine is loaded."""
    return _HAS_CORE_ENGINE


def sc_multiply(a: int, b: int) -> int:
    """SC multiply: AND of two u32 bitstreams."""
    if not _HAS_CORE_ENGINE:
        return a & b
    return int(_lib.sc_multiply(_ct.c_uint32(a), _ct.c_uint32(b)))


def sc_mux(a: int, b: int, sel: int) -> int:
    """SC MUX: (sel & a) | (~sel & b)."""
    if not _HAS_CORE_ENGINE:
        return (sel & a) | (~sel & b) & 0xFFFFFFFF
    return int(_lib.sc_mux(_ct.c_uint32(a), _ct.c_uint32(b), _ct.c_uint32(sel)))


def sc_popcount(a: int) -> int:
    """Population count of a u32."""
    if not _HAS_CORE_ENGINE:
        return bin(a).count("1")
    return int(_lib.sc_popcount(_ct.c_uint32(a)))


def sc_popcount64(a: int) -> int:
    """Population count of a u64."""
    if not _HAS_CORE_ENGINE:
        return bin(a).count("1")
    return int(_lib.sc_popcount64(_ct.c_uint64(a)))


def sc_popcount_packed(data: list[int]) -> int:
    """Popcount of packed u64 array (from Python list)."""
    if not _HAS_CORE_ENGINE:
        return sum(bin(w).count("1") for w in data)
    n = len(data)
    arr = (_ct.c_uint64 * n)(*data)
    return int(_lib.sc_popcount_packed(arr, _ct.c_size_t(n)))


def sc_popcount_packed_np(data):
    """Popcount of packed u64 numpy array (zero-copy)."""
    import numpy as np

    data = np.ascontiguousarray(data, dtype=np.uint64)
    ptr = data.ctypes.data_as(_ct.POINTER(_ct.c_uint64))
    return int(_lib.sc_popcount_packed(ptr, _ct.c_size_t(data.size)))


def sc_scc_packed(a: list[int], b: list[int]) -> float:
    """Stochastic cross-correlation of packed u64 arrays (from Python list)."""
    n = min(len(a), len(b))
    if not _HAS_CORE_ENGINE or n == 0:
        return 0.0
    arr_a = (_ct.c_uint64 * n)(*a[:n])
    arr_b = (_ct.c_uint64 * n)(*b[:n])
    return float(_lib.sc_scc_packed(arr_a, arr_b, _ct.c_size_t(n)))


def sc_scc_packed_np(a, b) -> float:
    """Stochastic cross-correlation of packed u64 numpy arrays (zero-copy)."""
    import numpy as np

    a = np.ascontiguousarray(a, dtype=np.uint64)
    b = np.ascontiguousarray(b, dtype=np.uint64)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    ptr_a = a[:n].ctypes.data_as(_ct.POINTER(_ct.c_uint64))
    ptr_b = b[:n].ctypes.data_as(_ct.POINTER(_ct.c_uint64))
    return float(_lib.sc_scc_packed(ptr_a, ptr_b, _ct.c_size_t(n)))
