# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optional cortical-column native backend discovery

"""Discover optional native sparse-matrix kernels for the cortical-column model."""

from __future__ import annotations

import ctypes
import logging
import os as os
from typing import Any, Callable, NamedTuple, cast


class NativeBackends(NamedTuple):
    """Native kernels discovered for one import of the public module."""

    rust_spmv: Callable[..., Any] | None
    rust_multi_spmv: Callable[..., Any] | None
    julia_multi_spmv: Callable[..., Any] | None
    go_multi_spmv: Callable[..., Any] | None
    mojo_multi_spmv: Callable[..., Any] | None


def _load_ctypes_multi_spmv(library_path: str) -> tuple[Any, Callable[..., Any]] | None:
    """Load and configure a C-ABI batched SpMV symbol when its library exists."""
    if not os.path.exists(library_path):
        return None
    library = ctypes.CDLL(library_path)
    kernel = library.py_parallel_csr_multi_spmv_add_c
    kernel.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
    ]
    kernel.restype = None
    return library, cast(Callable[..., Any], kernel)


def discover_native_backends(
    public_module_file: str,
    logger: logging.Logger,
    import_module: Callable[[str], Any],
) -> NativeBackends:
    """Discover optional Rust, Julia, Go, and Mojo cortical-column kernels.

    Discovery is invoked by the public module on every import or reload so its
    historical module-level capability flags remain truthful and monkeypatchable.
    Missing optional runtimes fail closed to the Python implementation.
    """
    rust_spmv: Callable[..., Any] | None = None
    rust_multi_spmv: Callable[..., Any] | None = None
    try:
        rust_spmv = import_module(
            "sc_neurocore_engine.sc_neurocore_engine"
        ).py_parallel_csr_spmv_add
    except (ImportError, AttributeError):
        try:
            rust_spmv = import_module("sc_neurocore_engine").py_parallel_csr_spmv_add
        except (ImportError, AttributeError):
            pass
    try:
        rust_multi_spmv = import_module(
            "sc_neurocore_engine.sc_neurocore_engine"
        ).py_parallel_csr_multi_spmv_add
    except (ImportError, AttributeError):
        try:
            rust_multi_spmv = import_module("sc_neurocore_engine").py_parallel_csr_multi_spmv_add
        except (ImportError, AttributeError):
            pass

    julia_multi_spmv: Callable[..., Any] | None = None
    try:
        from juliacall import Main as jl

        julia_file = os.path.abspath(
            os.path.join(
                os.path.dirname(public_module_file),
                "..",
                "accel",
                "julia",
                "network",
                "cortical_column.jl",
            )
        )
        if os.path.exists(julia_file):
            jl.seval(f'include("{julia_file}")')
            julia_multi_spmv = cast(
                Callable[..., Any],
                jl.CorticalColumnAccel.py_parallel_csr_multi_spmv_add,
            )
    except Exception as julia_error:  # noqa: BLE001
        logger.debug("Julia multi-spmv accel unavailable: %r", julia_error)

    go_multi_spmv: Callable[..., Any] | None = None
    try:
        go_library_file = os.path.abspath(
            os.path.join(
                os.path.dirname(public_module_file),
                "..",
                "accel",
                "go",
                "cortical_column",
                "libcortical_column.so",
            )
        )
        loaded_go = _load_ctypes_multi_spmv(go_library_file)
        if loaded_go is not None:
            _go_library, go_multi_spmv = loaded_go
    except Exception as go_error:  # noqa: BLE001
        logger.debug("Go multi-spmv accel unavailable: %r", go_error)

    mojo_multi_spmv: Callable[..., Any] | None = None
    try:
        mojo_library_file = os.path.abspath(
            os.path.join(
                os.path.dirname(public_module_file),
                "..",
                "accel",
                "mojo",
                "kernels",
                "libcortical_column.so",
            )
        )
        loaded_mojo = _load_ctypes_multi_spmv(mojo_library_file)
        if loaded_mojo is not None:
            _mojo_library, mojo_multi_spmv = loaded_mojo
    except Exception as mojo_error:  # noqa: BLE001
        logger.debug("Mojo multi-spmv accel unavailable: %r", mojo_error)

    return NativeBackends(
        rust_spmv=rust_spmv,
        rust_multi_spmv=rust_multi_spmv,
        julia_multi_spmv=julia_multi_spmv,
        go_multi_spmv=go_multi_spmv,
        mojo_multi_spmv=mojo_multi_spmv,
    )
