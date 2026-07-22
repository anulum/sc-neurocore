# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LGSSM engine-binding contracts

"""Installed-extension contracts for the LGSSM Kalman-filter binding."""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np
import pytest

import sc_neurocore_engine as engine
from sc_neurocore.world_model import _lgssm_backends as backends
from sc_neurocore.world_model.predictive_model import KalmanFilter, LinearGaussianSSM

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _direct() -> dict[str, Any]:
    result = extension.py_lgssm_kalman_filter(
        obs_flat=[1.0, 2.0],
        controls_flat=[0.0, 0.0],
        t_len=2,
        p_dim=1,
        m_dim=1,
        a_flat=[1.0],
        b_flat=[0.0],
        c_flat=[1.0],
        d_flat=[0.0],
        q_flat=[0.1],
        r_flat=[0.2],
        mu_0=[0.0],
        sigma_0_flat=[1.0],
        d_dim=1,
    )
    return cast(dict[str, Any], result)


def test_exported_name_signature_and_loader_identity_are_stable() -> None:
    function = extension.py_lgssm_kalman_filter

    assert function.__name__ == "py_lgssm_kalman_filter"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(obs_flat, controls_flat, t_len, p_dim, m_dim, a_flat, b_flat, "
        "c_flat, d_flat, q_flat, r_flat, mu_0, sigma_0_flat, d_dim)"
    )
    assert engine.py_lgssm_kalman_filter is function
    assert "py_lgssm_kalman_filter" in engine.__all__
    assert backends._load_rust_kalman_filter() is function


def test_scalar_filter_mapping_is_exact() -> None:
    assert _direct() == {
        "means": [[0.8333333333333335], [1.5]],
        "covariances": [
            [[0.16666666666666669]],
            [[0.1142857142857143]],
        ],
        "pred_means": [[0.0], [0.8333333333333335]],
        "pred_covariances": [[[1.0]], [[0.2666666666666667]]],
        "log_likelihood": -3.422967818782874,
        "backend": "rust",
    }


def test_production_rust_dispatcher_matches_python_filter() -> None:
    model = LinearGaussianSSM(
        A=np.array([[0.8]]),
        B=np.array([[0.3]]),
        C=np.array([[1.1]]),
        D=np.array([[0.2]]),
        Q=np.array([[0.04]]),
        R=np.array([[0.06]]),
        mu_0=np.array([0.1]),
        Sigma_0=np.array([[0.4]]),
    )
    controls = np.linspace(-0.2, 0.3, 32, dtype=np.float64)[:, None]
    observations = np.cos(np.arange(32, dtype=np.float64) / 7.0)[:, None]

    rust = KalmanFilter(model).filter(observations, controls, backend="rust")
    python = KalmanFilter(model).filter(observations, controls, backend="python")

    np.testing.assert_allclose(rust.means, python.means, atol=1e-12)
    np.testing.assert_allclose(rust.covariances, python.covariances, atol=1e-12)
    np.testing.assert_allclose(rust.pred_means, python.pred_means, atol=1e-12)
    np.testing.assert_allclose(
        rust.pred_covariances,
        python.pred_covariances,
        atol=1e-12,
    )
    assert rust.log_likelihood == pytest.approx(python.log_likelihood, abs=1e-12)
