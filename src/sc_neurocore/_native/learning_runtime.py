# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning dynamic-library runtime

"""Load, type, and safely expose the autonomous-learning C ABI."""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
import platform
from threading import RLock
from typing import Any

from .learning_validation import require_u32_seed

_HAS_LEARNING = False
_lib: ct.CDLL | None = None
_DETERMINISTIC_SEED: int | None = None
_LOAD_LOCK = RLock()


class OnlineO1SnapshotFFI(ct.Structure):
    """ctypes representation of the bounded Rust online O(1) snapshot."""

    _fields_ = [
        ("weight", ct.c_uint32),
        ("pre_trace", ct.c_uint32),
        ("post_trace", ct.c_uint32),
        ("eligibility", ct.c_int32),
    ]


def _library_path() -> Path:
    """Return the configured or platform-default autonomous-learning library."""
    configured = os.getenv("SC_NEUROCORE_LIB_PATH")
    if configured:
        return Path(configured)
    system = platform.system()
    name = (
        "autonomous_learning.dll"
        if system == "Windows"
        else "libautonomous_learning.dylib"
        if system == "Darwin"
        else "libautonomous_learning.so"
    )
    return Path(__file__).parent.resolve() / name


def _bind_rule_api(lib: ct.CDLL) -> None:
    """Attach ctypes signatures for scalar and batched rule functions."""
    lib.create_rule.argtypes = [ct.c_uint32, ct.c_float, ct.c_float, ct.c_float]
    lib.create_rule.restype = ct.c_void_p
    lib.step_rule.argtypes = [ct.c_void_p, ct.c_bool, ct.c_bool, ct.c_float, ct.c_float]
    lib.step_rule.restype = None
    lib.get_rule_weight.argtypes = [ct.c_void_p]
    lib.get_rule_weight.restype = ct.c_float
    lib.reset_rule.argtypes = [ct.c_void_p]
    lib.reset_rule.restype = None
    lib.destroy_rule.argtypes = [ct.c_void_p]
    lib.destroy_rule.restype = None
    lib.step_rule_batched.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_bool),
        ct.POINTER(ct.c_bool),
        ct.POINTER(ct.c_float),
        ct.c_size_t,
        ct.c_float,
    ]
    lib.step_rule_batched.restype = None


def _bind_learner_api(lib: ct.CDLL) -> None:
    """Attach ctypes signatures for the legacy ELIGENT learner."""
    lib.create_learner.argtypes = [ct.c_float, ct.c_float, ct.c_float]
    lib.create_learner.restype = ct.c_void_p
    lib.step_learner.argtypes = [ct.c_void_p, ct.c_bool, ct.c_bool, ct.c_float, ct.c_float]
    lib.step_learner.restype = None
    lib.destroy_learner.argtypes = [ct.c_void_p]
    lib.destroy_learner.restype = None
    lib.step_learner_batched.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_bool),
        ct.POINTER(ct.c_bool),
        ct.POINTER(ct.c_float),
        ct.c_size_t,
        ct.c_float,
    ]
    lib.step_learner_batched.restype = None


def _bind_layer_api(lib: ct.CDLL) -> None:
    """Attach ctypes signatures for Rayon layers and state transport."""
    lib.create_rule_layer.argtypes = [
        ct.c_size_t,
        ct.c_uint32,
        ct.c_float,
        ct.c_float,
        ct.c_float,
    ]
    lib.create_rule_layer.restype = ct.c_void_p
    lib.step_rule_layer.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_bool),
        ct.POINTER(ct.c_bool),
        ct.POINTER(ct.c_float),
        ct.c_float,
    ]
    lib.step_rule_layer.restype = None
    lib.step_rule_layer_analog.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.c_uint64,
        ct.c_float,
    ]
    lib.step_rule_layer_analog.restype = None
    lib.get_rule_layer_weights.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float)]
    lib.get_rule_layer_weights.restype = None
    lib.destroy_rule_layer.argtypes = [ct.c_void_p]
    lib.destroy_rule_layer.restype = None
    lib.reset_rule_layer.argtypes = [ct.c_void_p]
    lib.reset_rule_layer.restype = None
    lib.save_rule_layer_batched.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.save_rule_layer_batched.restype = ct.c_bool
    lib.load_rule_layer_batched.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.load_rule_layer_batched.restype = ct.c_bool
    lib.get_rule_layer_state_size.argtypes = [ct.c_void_p]
    lib.get_rule_layer_state_size.restype = ct.c_size_t
    lib.get_rule_layer_state_mem.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint8)]
    lib.get_rule_layer_state_mem.restype = ct.c_bool
    lib.set_rule_layer_state_mem.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint8)]
    lib.set_rule_layer_state_mem.restype = ct.c_bool
    try:
        checked = lib.set_rule_layer_state_mem_checked
    except AttributeError:
        return
    checked.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint8), ct.c_size_t]
    checked.restype = ct.c_bool


def _bind_wgpu_api(lib: ct.CDLL) -> None:
    """Attach ctypes signatures for the optional WGPU implementation."""
    lib.create_wgpu_layer.argtypes = [
        ct.c_size_t,
        ct.c_uint32,
        ct.c_float,
        ct.c_float,
        ct.c_float,
        ct.c_float,
        ct.c_float,
        ct.c_float,
    ]
    lib.create_wgpu_layer.restype = ct.c_void_p
    lib.step_wgpu_layer.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.c_float,
    ]
    lib.step_wgpu_layer.restype = None
    lib.get_wgpu_weights.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float)]
    lib.get_wgpu_weights.restype = None
    lib.set_wgpu_layer_seed.argtypes = [ct.c_void_p, ct.c_uint32]
    lib.set_wgpu_layer_seed.restype = None
    lib.free_wgpu_layer.argtypes = [ct.c_void_p]
    lib.free_wgpu_layer.restype = None
    lib.reset_wgpu_layer.argtypes = [ct.c_void_p]
    lib.reset_wgpu_layer.restype = None
    try:
        create_weighted = lib.create_wgpu_layer_with_weight
        set_weights = lib.set_wgpu_weights
    except AttributeError:
        return
    create_weighted.argtypes = [
        ct.c_size_t,
        ct.c_uint32,
        ct.c_float,
        ct.c_float,
        ct.c_float,
        ct.c_float,
        ct.c_float,
        ct.c_float,
        ct.c_float,
    ]
    create_weighted.restype = ct.c_void_p
    set_weights.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_size_t]
    set_weights.restype = ct.c_bool


def _bind_online_o1_api(lib: ct.CDLL) -> None:
    """Bind optional fixed-point online O(1) symbols when present."""
    try:
        create = lib.create_online_o1_synapse
        step = lib.step_online_o1_synapse
        state_bits = lib.online_o1_per_synapse_state_bits
        destroy = lib.destroy_online_o1_synapse
    except AttributeError:
        return
    create.argtypes = [ct.c_uint8, ct.c_uint8, ct.c_uint8, ct.c_uint8, ct.c_uint8, ct.c_uint32]
    create.restype = ct.c_void_p
    step.argtypes = [ct.c_void_p, ct.c_bool, ct.c_bool, ct.c_int32]
    step.restype = OnlineO1SnapshotFFI
    state_bits.argtypes = [ct.c_void_p]
    state_bits.restype = ct.c_uint32
    destroy.argtypes = [ct.c_void_p]
    destroy.restype = None


def _load_native_library() -> bool:
    """Load and type the optional Rust learning library atomically."""
    global _HAS_LEARNING, _lib
    path = _library_path()
    with _LOAD_LOCK:
        if not path.is_file():
            _HAS_LEARNING = False
            _lib = None
            return False
        try:
            candidate = ct.CDLL(str(path))
            _bind_rule_api(candidate)
            _bind_learner_api(candidate)
            _bind_layer_api(candidate)
            _bind_wgpu_api(candidate)
            _bind_online_o1_api(candidate)
        except (AttributeError, OSError):
            _HAS_LEARNING = False
            _lib = None
            return False
        _lib = candidate
        _HAS_LEARNING = True
        return True


def _get_lib() -> ct.CDLL:
    """Return the loaded Rust library or raise an actionable error."""
    if _lib is None:
        raise RuntimeError(
            "libautonomous_learning is not loaded; build crates/autonomous_learning "
            "or set SC_NEUROCORE_LIB_PATH"
        )
    return _lib


def is_available() -> bool:
    """Return whether the required autonomous-learning ABI is loaded."""
    return _HAS_LEARNING and _lib is not None


def has_symbol(name: str) -> bool:
    """Return whether the loaded library exposes ``name``."""
    return _lib is not None and hasattr(_lib, name)


def require_symbol(name: str) -> Any:
    """Return a typed dynamic symbol or raise for an incompatible library."""
    lib = _get_lib()
    try:
        return getattr(lib, name)
    except AttributeError as exc:
        raise RuntimeError(f"libautonomous_learning lacks required symbol {name}") from exc


def destroy_noexcept(symbol: str, pointer: object) -> None:
    """Destroy a native handle without leaking exceptions from finalizers."""
    lib = _lib
    if lib is None or not pointer:
        return
    try:
        getattr(lib, symbol)(pointer)
    except (AttributeError, OSError, TypeError):
        return


def set_deterministic_mode(seed: int | None = None) -> None:
    """Set a shared deterministic seed for CPU analogue and WGPU paths."""
    global _DETERMINISTIC_SEED
    _DETERMINISTIC_SEED = None if seed is None else require_u32_seed(name="seed", value=seed)


def deterministic_seed() -> int | None:
    """Return the configured shared deterministic seed."""
    return _DETERMINISTIC_SEED


_load_native_library()
