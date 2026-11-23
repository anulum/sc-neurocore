# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python facade for the Julia contrastive/ssl kernels

"""Python facade for the Julia ``SslAccel`` module (``ssl.jl``).

The dispatcher :mod:`sc_neurocore.accel.contrastive_ssl` validates every input
before it calls these functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.julia.neurons._runtime import JULIA_MAIN as _jl

_SOURCE = Path(__file__).with_name("ssl.jl")
_LOADED = False


def ensure_loaded() -> Any:
    """Include ``ssl.jl`` on first use and return its ``SslAccel`` module.

    Returns
    -------
    Any
        The Julia module object.

    Raises
    ------
    ImportError
        If juliacall is not installed.
    FileNotFoundError
        If ``ssl.jl`` is missing.
    """
    global _LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _LOADED:
        if not _SOURCE.is_file():
            raise FileNotFoundError(f"ssl.jl missing at {_SOURCE}")
        _jl.include(str(_SOURCE))
        _LOADED = True
    return _jl.SslAccel


def infonce_loss(
    view_a: npt.NDArray[np.float64],
    view_b: npt.NDArray[np.float64],
    temperature: float,
) -> float:
    """Return the Julia mean InfoNCE loss of two validated views.

    Parameters
    ----------
    view_a, view_b : ndarray of shape (batch, features), float64
        Finite views of the same shape.
    temperature : float
        Finite, positive temperature.

    Returns
    -------
    float
        The loss.
    """
    module = ensure_loaded()
    return float(module.compute(module.SpikeContrastiveLossState(temperature), view_a, view_b))


def csdp_update(
    weights: npt.NDArray[np.float64],
    pre: npt.NDArray[np.float64],
    post: npt.NDArray[np.float64],
    lr: float,
    decay: float,
    *,
    positive: bool,
) -> npt.NDArray[np.float64]:
    """Return one Julia CSDP phase applied to validated weights.

    Parameters
    ----------
    weights : ndarray of shape (n_post, n_pre), float64
        Finite weights.
    pre, post : ndarray of shape (n_pre,) and (n_post,), float64
        Finite spike-rate vectors.
    lr, decay : float
        Finite, non-negative rule parameters.
    positive : bool
        ``True`` for the Hebbian phase, ``False`` for the anti-Hebbian phase.

    Returns
    -------
    ndarray of shape (n_post, n_pre)
        The updated weights.
    """
    module = ensure_loaded()
    state = module.CSDPRuleState(lr, decay)
    phase = module.positive_update if positive else module.negative_update
    updated = np.array(phase(state, weights, pre, post), dtype=np.float64)
    return updated.reshape(weights.shape)


def goodness(values: npt.NDArray[np.float64]) -> float:
    """Return the Julia sum of squared validated activations.

    Parameters
    ----------
    values : ndarray, float64
        Finite activations of any shape.

    Returns
    -------
    float
        ``sum(values**2)``.
    """
    module = ensure_loaded()
    return float(module.goodness(module.CSDPRuleState(0.0, 0.0), values))
