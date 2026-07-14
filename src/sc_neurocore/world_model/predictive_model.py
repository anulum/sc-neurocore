# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Linear Gaussian state-space public facade

"""Linear Gaussian filtering, smoothing, learning, and state forecasting.

The model follows Kalman (1960), Rauch, Tung, and Striebel (1965), and the
controlled expectation-maximisation formulation of Shumway and Stoffer
(1982). Native acceleration applies to the forward Kalman filter; RTS
smoothing and the EM M-step remain NumPy implementations.
"""

from __future__ import annotations

from . import _lgssm_backends as _backend_runtime
from ._lgssm_backends import (
    _ensure_go_loaded as _ensure_go_loaded,
    _ensure_julia_loaded as _ensure_julia_loaded,
    _ensure_mojo_loaded as _ensure_mojo_loaded,
    _missing_rust_kalman_filter as _missing_rust_kalman_filter,
)
from ._lgssm_em import EMLearner
from ._lgssm_filter import KalmanFilter
from ._lgssm_smoothing import RTSSmoother
from ._lgssm_types import FilterResult, LinearGaussianSSM, SmoothResult
from ._predictive_world_model import PredictiveWorldModel

__all__ = [
    "LinearGaussianSSM",
    "FilterResult",
    "KalmanFilter",
    "SmoothResult",
    "RTSSmoother",
    "EMLearner",
    "PredictiveWorldModel",
]


def __getattr__(name: str) -> object:
    """Return the live historical Rust-availability flag on private access."""
    if name == "_HAS_RUST_LGSSM":
        return _backend_runtime._HAS_RUST_LGSSM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Preserve the historical public identity for introspection and serialized
# class references while implementations remain partitioned by responsibility.
for _public_class in (
    LinearGaussianSSM,
    FilterResult,
    KalmanFilter,
    SmoothResult,
    RTSSmoother,
    EMLearner,
    PredictiveWorldModel,
):
    _public_class.__module__ = __name__

del _public_class
