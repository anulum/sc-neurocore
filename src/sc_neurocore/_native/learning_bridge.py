# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning compatibility facade

"""Stable public facade for modular autonomous-learning backends.

The implementation is split by responsibility so ABI loading, validation,
native ownership, WGPU execution, Torch dynamics, and backend selection remain
independently auditable.  Public object identities retain this historical
module path for pickle and downstream compatibility.
"""

from __future__ import annotations

from typing import Any

from . import learning_runtime as _runtime
from .learning_factory import create_plasticity_layer
from .learning_rust import RustEligentLearner, RustOnlineO1Synapse, RustPlasticityRule
from .learning_rust_layer import RustRuleLayer
from .learning_validation import RULE_BCM, RULE_ELIGENT, RULE_REWARD_STDP, RULE_STDP
from .learning_wgpu import RustWgpuRuleLayer

OnlineO1SnapshotFFI = _runtime.OnlineO1SnapshotFFI
_get_lib = _runtime._get_lib
_load_native_library = _runtime._load_native_library


def is_available() -> bool:
    """Return whether the required autonomous-learning Rust ABI is loaded."""
    return _runtime.is_available()


def set_deterministic_mode(seed: int | None = None) -> None:
    """Set the shared deterministic seed for analogue and WGPU paths."""
    _runtime.set_deterministic_mode(seed)


_PUBLIC_OBJECTS: list[Any] = [
    OnlineO1SnapshotFFI,
    RustOnlineO1Synapse,
    RustPlasticityRule,
    RustEligentLearner,
    RustRuleLayer,
    RustWgpuRuleLayer,
    create_plasticity_layer,
]

try:
    import torch as _torch  # noqa: F401
except ImportError:
    _TORCH_EXPORTS: list[str] = []
else:
    from .learning_torch import TorchRuleLayer
    from .learning_torch_autograd import _BiologicalAutogradFactory

    AutogradSTDPLayer = TorchRuleLayer
    _PUBLIC_OBJECTS.extend((TorchRuleLayer, AutogradSTDPLayer, _BiologicalAutogradFactory))
    _TORCH_EXPORTS = ["TorchRuleLayer", "AutogradSTDPLayer"]

for _public_object in _PUBLIC_OBJECTS:
    _public_object.__module__ = __name__

__all__ = [
    "OnlineO1SnapshotFFI",
    "RULE_ELIGENT",
    "RULE_STDP",
    "RULE_REWARD_STDP",
    "RULE_BCM",
    "set_deterministic_mode",
    "is_available",
    "RustOnlineO1Synapse",
    "RustPlasticityRule",
    "RustEligentLearner",
    "RustRuleLayer",
    "RustWgpuRuleLayer",
    "create_plasticity_layer",
    *_TORCH_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Expose read-only historical runtime diagnostics for compatibility."""
    if name == "_lib":
        return _runtime._lib
    if name == "_HAS_LEARNING":
        return _runtime._HAS_LEARNING
    if name == "_DETERMINISTIC_SEED":
        return _runtime.deterministic_seed()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
