# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Layers Package Init

"""Expose stochastic-computing layer primitives for package consumers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .sc_dense_layer import SCDenseLayer
from .sc_conv_layer import SCConv2DLayer
from .sc_learning_layer import SCLearningLayer
from .vectorized_layer import VectorizedSCLayer
from .recurrent import SCRecurrentLayer
from .memristive import MemristiveDenseLayer
from .fusion import SCFusionLayer
from .attention import StochasticAttention
from .hardware_aware import HardwareAwareSCLayer
from .predictive_coding import PredictiveCodingSCLayer
from .rall_dendrite import RallDendrite
from .circuit_primitives import LateralInhibition, WinnerTakeAll

if TYPE_CHECKING:
    from .jax_dense_layer import JaxSCDenseLayer

_LAZY_SYMBOL_SOURCES = {"JaxSCDenseLayer": "jax_dense_layer"}


def __getattr__(name: str) -> Any:
    """Lazily resolve optional-accelerator layer symbols."""
    source = _LAZY_SYMBOL_SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{source}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List eager and lazy public layer symbols."""
    return sorted(set(__all__) | set(globals()))


__all__ = [
    "SCDenseLayer",
    "SCConv2DLayer",
    "SCLearningLayer",
    "VectorizedSCLayer",
    "SCRecurrentLayer",
    "MemristiveDenseLayer",
    "SCFusionLayer",
    "StochasticAttention",
    "HardwareAwareSCLayer",
    "PredictiveCodingSCLayer",
    "RallDendrite",
    "LateralInhibition",
    "WinnerTakeAll",
    "JaxSCDenseLayer",
]
