# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native dense IF request layouts and owned result lifetime

"""C ABI-one descriptors and automatic native replay result lifetime."""

import ctypes
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast


class LayerSpec(ctypes.Structure):
    """Native ABI-one borrowed dense layer and optional initial-state descriptor."""

    _fields_ = [
        ("outputs", ctypes.c_size_t),
        ("inputs", ctypes.c_size_t),
        ("weights", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("bias_len", ctypes.c_size_t),
        ("threshold", ctypes.c_double),
        ("initial_fraction", ctypes.c_double),
        ("initial", ctypes.c_void_p),
        ("initial_len", ctypes.c_size_t),
    ]


class ReplayRequest(ctypes.Structure):
    """ABI-one request with retained caller-owned descriptors and arrays."""

    _fields_ = [
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("layers", ctypes.c_void_p),
        ("layer_count", ctypes.c_size_t),
        ("frames", ctypes.c_void_p),
        ("frames_len", ctypes.c_size_t),
        ("steps", ctypes.c_size_t),
        ("batch", ctypes.c_size_t),
        ("max_working_bytes", ctypes.c_size_t),
    ]


class BufferView(ctypes.Structure):
    """Borrowed row-major result storage, live until its opaque owner is freed."""

    _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_size_t)]


ReplayCall = Callable[[object, object], int]
BufferCall = Callable[[object, int, int, object], int]
FreeCall = Callable[[object], None]


@dataclass(frozen=True)
class NativeAPI:
    """Loaded library or managed-runtime root and typed ownership/buffer entry points."""

    library: object
    replay: ReplayCall
    buffer: BufferCall
    free: FreeCall


def _release(api: NativeAPI, handle: ctypes.c_void_p) -> None:
    """Keep the supplying library alive until its opaque result is freed exactly once."""
    api.free(handle)
    handle.value = None


class _FinalizerControl(Protocol):
    """Descriptor-backed CPython finalizer control; the stdlib stub declares it as a slotless field."""

    atexit: bool


class NativeOwner:
    """Release native storage only after all arrays retaining this owner have expired."""

    def __init__(self, api: NativeAPI, handle: ctypes.c_void_p) -> None:
        """Bind one successful native allocation to automatic final-owner cleanup."""
        self.api = api
        self.handle = handle
        self.finalizer = weakref.finalize(self, _release, api, handle)
        cast(_FinalizerControl, self.finalizer).atexit = False
