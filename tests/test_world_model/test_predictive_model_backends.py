# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Backend and wrapper tests for predictive world model

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import sc_neurocore.world_model.predictive_model as pm


def _model(control_dim: int = 0) -> pm.LinearGaussianSSM:
    d = 2
    m = control_dim
    return pm.LinearGaussianSSM(
        A=np.eye(d),
        B=np.zeros((d, m)),
        C=np.eye(d),
        D=np.zeros((d, m)),
        Q=np.eye(d) * 0.1,
        R=np.eye(d) * 0.2,
        mu_0=np.zeros(d),
        Sigma_0=np.eye(d),
    )


def test_missing_rust_kalman_filter_raises() -> None:
    with pytest.raises(RuntimeError, match="not available"):
        pm._missing_rust_kalman_filter()


def test_ensure_mojo_loaded_false_when_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    pm._mojo_lib = None
    pm._HAS_MOJO_LGSSM = False
    monkeypatch.setattr(pm, "__file__", str(Path("/tmp/fake_pkg/world_model/predictive_model.py")))
    monkeypatch.setattr("os.path.isfile", lambda _path: False)
    assert pm._ensure_mojo_loaded() is False


def test_ensure_mojo_loaded_false_on_cdll_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pm._mojo_lib = None
    pm._HAS_MOJO_LGSSM = False
    monkeypatch.setattr(pm, "__file__", str(Path("/tmp/fake_pkg/world_model/predictive_model.py")))
    monkeypatch.setattr("os.path.isfile", lambda _path: True)

    class _FakeCtypes:
        c_int64 = int

        @staticmethod
        def CDLL(_path: str) -> object:
            raise OSError("bad shared object")

    monkeypatch.setitem(pm._ensure_mojo_loaded.__globals__, "ctypes", _FakeCtypes)
    assert pm._ensure_mojo_loaded() is False


def test_ensure_go_loaded_false_without_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    pm._go_lib = None
    pm._HAS_GO_LGSSM = False
    monkeypatch.setattr(pm, "__file__", str(Path("/tmp/fake_pkg/world_model/predictive_model.py")))
    monkeypatch.setattr("os.path.isfile", lambda _path: True)

    class _FakeLib:
        pass

    class _FakeCtypes:
        c_double = float
        c_int = int
        POINTER = staticmethod(lambda _t: object)

        @staticmethod
        def CDLL(_path: str) -> object:
            return _FakeLib()

    monkeypatch.setitem(pm._ensure_go_loaded.__globals__, "ctypes", _FakeCtypes)
    assert pm._ensure_go_loaded() is False


def test_ensure_julia_loaded_false_without_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    pm._julia_module = None
    pm._HAS_JULIA_LGSSM = False

    class _FakeImportlibUtil:
        @staticmethod
        def find_spec(_name: str) -> None:
            return None

    monkeypatch.setitem(pm._ensure_julia_loaded.__globals__, "importlib", importlib)
    monkeypatch.setitem(pm._ensure_julia_loaded.__globals__, "importlib.util", _FakeImportlibUtil)
    assert pm._ensure_julia_loaded() is False


def test_linear_gaussian_ssm_rejects_negative_diagonal() -> None:
    with pytest.raises(ValueError, match="negative diagonal"):
        pm.LinearGaussianSSM(
            A=np.eye(2),
            B=np.zeros((2, 0)),
            C=np.eye(2),
            D=np.zeros((2, 0)),
            Q=np.array([[1.0, 0.0], [0.0, -0.1]]),
            R=np.eye(2),
            mu_0=np.zeros(2),
            Sigma_0=np.eye(2),
        )


def test_kalman_filter_rejects_missing_controls_for_controlled_model() -> None:
    model = _model(control_dim=1)
    obs = np.zeros((5, 2))
    with pytest.raises(ValueError, match="controls must have shape"):
        pm.KalmanFilter(model).filter(obs, controls=None)


def test_kalman_filter_rejects_unknown_backend() -> None:
    model = _model()
    obs = np.zeros((5, 2))
    with pytest.raises(ValueError, match="backend must be"):
        pm.KalmanFilter(model).filter(obs, backend="bogus")


@pytest.mark.parametrize(
    ("backend", "attr", "message"),
    [
        ("rust", "_HAS_RUST_LGSSM", "Rust LGSSM backend requested"),
        ("go", "_ensure_go_loaded", "Go LGSSM backend requested"),
        ("mojo", "_ensure_mojo_loaded", "Mojo LGSSM backend requested"),
        ("julia", "_ensure_julia_loaded", "Julia LGSSM backend requested"),
    ],
)
def test_kalman_filter_explicit_backend_errors(
    monkeypatch: pytest.MonkeyPatch, backend: str, attr: str, message: str
) -> None:
    model = _model()
    obs = np.zeros((5, 2))
    if attr.startswith("_ensure_"):
        monkeypatch.setattr(pm, attr, lambda: False)
    else:
        monkeypatch.setattr(pm, attr, False)
    with pytest.raises(RuntimeError, match=message):
        pm.KalmanFilter(model).filter(obs, backend=backend)


def test_filter_rust_raises_when_backend_probe_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    kf = pm.KalmanFilter(model)
    monkeypatch.setattr(pm, "_rust_kalman_filter", None)
    with pytest.raises(RuntimeError, match="cannot dispatch"):
        kf._filter_rust(np.zeros((3, 2)), np.zeros((3, 0)))


def test_filter_julia_raises_when_module_missing() -> None:
    model = _model()
    kf = pm.KalmanFilter(model)
    old = pm._julia_module
    pm._julia_module = None
    try:
        with pytest.raises(RuntimeError, match="not loaded"):
            kf._filter_julia(np.zeros((3, 2)), np.zeros((3, 0)))
    finally:
        pm._julia_module = old


def test_filter_mojo_raises_when_lib_missing() -> None:
    model = _model()
    kf = pm.KalmanFilter(model)
    old = pm._mojo_lib
    pm._mojo_lib = None
    try:
        with pytest.raises(RuntimeError, match="not loaded"):
            kf._filter_mojo(np.zeros((3, 2)), np.zeros((3, 0)))
    finally:
        pm._mojo_lib = old


def test_predictive_world_model_scalar_action_and_reset() -> None:
    pwm = pm.PredictiveWorldModel(state_dim=2, action_dim=1, seed=1)
    state = np.array([1.0, -1.0])
    predicted = pwm.predict_next_state(state, np.array(0.5))
    assert predicted.shape == (2,)

    pwm._mu = np.ones(2)
    pwm._Sigma = np.eye(2) * 2.0
    pwm.reset()
    np.testing.assert_allclose(pwm._mu, pwm.model.mu_0)
    np.testing.assert_allclose(pwm._Sigma, pwm.model.Sigma_0)
