# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner backend runtime

"""Typed discovery and loading for maintained KL-refinement backends."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any


_rust_kl_refine: Any | None
try:
    from sc_neurocore_engine import py_kl_refine as _loaded_rust_kl_refine
except (ImportError, AttributeError):
    _rust_kl_refine = None
    _HAS_RUST_KL_REFINE = False
else:
    _rust_kl_refine = _loaded_rust_kl_refine
    _HAS_RUST_KL_REFINE = True

_julia_kl_refine: Any | None = None
_HAS_JULIA_KL_REFINE = False
_go_kl_refine_lib: Any | None = None
_HAS_GO_KL_REFINE = False
_mojo_kl_refine_lib: Any | None = None
_HAS_MOJO_KL_REFINE = False


def _accel_path(*parts: str) -> Path:
    """Resolve a path below the package's acceleration directory."""
    return Path(__file__).resolve().parents[1].joinpath("accel", *parts)


def _ensure_julia_kl_refine_loaded() -> bool:
    """Load the Julia KL-refinement module on first use."""
    global _HAS_JULIA_KL_REFINE, _julia_kl_refine
    if _julia_kl_refine is not None:
        return True
    try:
        from juliacall import Main as julia
    except ImportError:
        return False

    module_path = _accel_path("julia", "chiplet", "kl_refine.jl")
    if not module_path.is_file():
        return False
    try:
        julia.include(str(module_path))
        _julia_kl_refine = julia.KLRefineAccel.kl_refine
    except Exception:
        return False
    _HAS_JULIA_KL_REFINE = True
    return True


def _ensure_go_kl_refine_loaded() -> bool:
    """Load and type the Go C-shared KL-refinement library on first use."""
    global _HAS_GO_KL_REFINE, _go_kl_refine_lib
    if _go_kl_refine_lib is not None:
        return True

    library_path = _accel_path("go", "partition", "libpartition.so")
    if not library_path.is_file():
        return False
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError:
        return False
    function = getattr(library, "kl_refine_c", None)
    if function is None:
        return False
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
    ]
    function.restype = ctypes.c_uint64
    _go_kl_refine_lib = library
    _HAS_GO_KL_REFINE = True
    return True


def _ensure_mojo_kl_refine_loaded() -> bool:
    """Load and type the Mojo shared KL-refinement library on first use."""
    global _HAS_MOJO_KL_REFINE, _mojo_kl_refine_lib
    if _mojo_kl_refine_lib is not None:
        return True

    library_path = _accel_path("mojo", "partition", "libpartition.so")
    if not library_path.is_file():
        return False
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError:
        return False
    function = getattr(library, "kl_refine_c", None)
    if function is None:
        return False
    function.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
    ]
    function.restype = ctypes.c_uint64
    _mojo_kl_refine_lib = library
    _HAS_MOJO_KL_REFINE = True
    return True


__all__: list[str] = []
