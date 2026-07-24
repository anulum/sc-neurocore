# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_predictive_model_backends.py

from __future__ import annotations


"""Loader, selection, fail-closed, and FFI marshalling tests for LGSSM backends."""


import ctypes


import importlib


import importlib.util


from collections.abc import Callable


from pathlib import Path


from types import SimpleNamespace


from typing import Protocol, cast


import numpy as np


import pytest


from sc_neurocore.world_model import _lgssm_backends as backends


from sc_neurocore.world_model._lgssm_backends import ExplicitBackendName


from sc_neurocore.world_model._lgssm_types import FloatArray


from sc_neurocore.world_model.predictive_model import LinearGaussianSSM


class _DoublePointer(Protocol):
    contents: ctypes.c_double


def _model() -> LinearGaussianSSM:
    return LinearGaussianSSM(
        A=np.eye(2),
        B=np.zeros((2, 0)),
        C=np.eye(2),
        D=np.zeros((2, 0)),
        Q=np.eye(2) * 0.1,
        R=np.eye(2) * 0.2,
        mu_0=np.zeros(2),
        Sigma_0=np.eye(2),
    )


def _inputs() -> tuple[FloatArray, FloatArray]:
    return np.zeros((3, 2)), np.zeros((3, 0))


def _native_mapping() -> dict[str, object]:
    covariance = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    return {
        "means": np.zeros((3, 2)),
        "covariances": covariance,
        "pred_means": np.zeros((3, 2)),
        "pred_covariances": covariance,
        "log_likelihood": -4.0,
    }


def _write_c_outputs(
    time_steps: int,
    state_dim: int,
    means_address: int,
    covariances_address: int,
    pred_means_address: int,
    pred_covariances_address: int,
    likelihood_address: int,
) -> None:
    means = ctypes.cast(means_address, ctypes.POINTER(ctypes.c_double))
    covariances = ctypes.cast(covariances_address, ctypes.POINTER(ctypes.c_double))
    pred_means = ctypes.cast(pred_means_address, ctypes.POINTER(ctypes.c_double))
    pred_covariances = ctypes.cast(
        pred_covariances_address,
        ctypes.POINTER(ctypes.c_double),
    )
    likelihood = ctypes.cast(likelihood_address, ctypes.POINTER(ctypes.c_double))
    for index in range(time_steps * state_dim):
        means[index] = 0.0
        pred_means[index] = 0.0
    for time_index in range(time_steps):
        for row in range(state_dim):
            for column in range(state_dim):
                index = time_index * state_dim * state_dim + row * state_dim + column
                value = 1.0 if row == column else 0.0
                covariances[index] = value
                pred_covariances[index] = value
    likelihood[0] = -4.0


__all__ = [
    "ctypes",
    "importlib",
    "Callable",
    "Path",
    "SimpleNamespace",
    "Protocol",
    "cast",
    "np",
    "pytest",
    "backends",
    "ExplicitBackendName",
    "FloatArray",
    "LinearGaussianSSM",
    "_DoublePointer",
    "_model",
    "_inputs",
    "_native_mapping",
    "_write_c_outputs",
]
