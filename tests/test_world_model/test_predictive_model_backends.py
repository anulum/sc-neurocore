# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Backend and wrapper tests for predictive world model

from __future__ import annotations

import ctypes
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sc_neurocore.world_model.predictive_model as pm
from tests.module_reload import restore_module_namespace, snapshot_module_namespace


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


def test_import_time_rust_probe_falls_back_when_engine_symbol_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = pm._importlib.import_module

    def missing_rust_engine(name: str) -> object:
        if name in {"sc_neurocore_engine.world_model", "sc_neurocore_engine"}:
            raise ImportError(name)
        return real_import_module(name)

    monkeypatch.setattr(pm._importlib, "import_module", missing_rust_engine)
    try:
        _saved_ns = snapshot_module_namespace(pm)
        reloaded = importlib.reload(pm)
        assert reloaded._HAS_RUST_LGSSM is False
        with pytest.raises(RuntimeError, match="Rust LGSSM backend requested"):
            reloaded.KalmanFilter(_model()).filter(np.zeros((3, 2)), backend="rust")
    finally:
        monkeypatch.undo()
        restore_module_namespace(pm, _saved_ns)


def test_import_time_rust_probe_uses_world_model_submodule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = pm._importlib.import_module
    sentinel_filter = object()

    def submodule_rust_engine(name: str) -> object:
        if name == "sc_neurocore_engine.world_model":
            return SimpleNamespace(get_lgssm_kalman_filter=lambda: sentinel_filter)
        return real_import_module(name)

    monkeypatch.setattr(pm._importlib, "import_module", submodule_rust_engine)
    try:
        _saved_ns = snapshot_module_namespace(pm)
        reloaded = importlib.reload(pm)
        assert reloaded._HAS_RUST_LGSSM is True
        assert reloaded._rust_kalman_filter is sentinel_filter
    finally:
        monkeypatch.undo()
        restore_module_namespace(pm, _saved_ns)


def test_import_time_rust_probe_uses_root_package_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = pm._importlib.import_module
    sentinel_filter = object()

    def root_only_rust_engine(name: str) -> object:
        if name == "sc_neurocore_engine.world_model":
            raise ImportError(name)
        if name == "sc_neurocore_engine":
            return SimpleNamespace(py_lgssm_kalman_filter=sentinel_filter)
        return real_import_module(name)

    monkeypatch.setattr(pm._importlib, "import_module", root_only_rust_engine)
    try:
        _saved_ns = snapshot_module_namespace(pm)
        reloaded = importlib.reload(pm)
        assert reloaded._HAS_RUST_LGSSM is True
        assert reloaded._rust_kalman_filter is sentinel_filter
    finally:
        monkeypatch.undo()
        restore_module_namespace(pm, _saved_ns)


def test_ensure_mojo_loaded_returns_true_when_already_cached() -> None:
    old_lib = pm._mojo_lib
    old_flag = pm._HAS_MOJO_LGSSM
    sentinel = object()
    try:
        pm._mojo_lib = sentinel
        pm._HAS_MOJO_LGSSM = True
        assert pm._ensure_mojo_loaded() is True
        assert pm._mojo_lib is sentinel
    finally:
        pm._mojo_lib = old_lib
        pm._HAS_MOJO_LGSSM = old_flag


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


def test_ensure_mojo_loaded_false_without_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    old_lib = pm._mojo_lib
    old_flag = pm._HAS_MOJO_LGSSM
    pm._mojo_lib = None
    pm._HAS_MOJO_LGSSM = False
    monkeypatch.setattr(pm, "__file__", str(Path("/tmp/fake_pkg/world_model/predictive_model.py")))
    monkeypatch.setattr("os.path.isfile", lambda _path: True)

    try:
        monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())
        assert pm._ensure_mojo_loaded() is False
    finally:
        pm._mojo_lib = old_lib
        pm._HAS_MOJO_LGSSM = old_flag


def test_ensure_mojo_loaded_configures_symbol_and_caches_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_lib = pm._mojo_lib
    old_flag = pm._HAS_MOJO_LGSSM
    pm._mojo_lib = None
    pm._HAS_MOJO_LGSSM = False
    monkeypatch.setattr(pm, "__file__", str(Path("/tmp/fake_pkg/world_model/predictive_model.py")))
    monkeypatch.setattr("os.path.isfile", lambda _path: True)

    class _FakeFn:
        argtypes: list[object] | None = None
        restype: object | None = object()

    fake_fn = _FakeFn()
    fake_lib = SimpleNamespace(kalman_filter_c=fake_fn)

    try:
        monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_lib)
        assert pm._ensure_mojo_loaded() is True
        assert pm._mojo_lib is fake_lib
        assert fake_fn.argtypes == [ctypes.c_int64] * 19
        assert fake_fn.restype is None
        assert pm._HAS_MOJO_LGSSM is True
    finally:
        pm._mojo_lib = old_lib
        pm._HAS_MOJO_LGSSM = old_flag


def test_ensure_go_loaded_false_when_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    old_lib = pm._go_lib
    old_flag = pm._HAS_GO_LGSSM
    pm._go_lib = None
    pm._HAS_GO_LGSSM = False
    monkeypatch.setattr(pm, "__file__", str(Path("/tmp/fake_pkg/world_model/predictive_model.py")))
    monkeypatch.setattr("os.path.isfile", lambda _path: False)
    try:
        assert pm._ensure_go_loaded() is False
    finally:
        pm._go_lib = old_lib
        pm._HAS_GO_LGSSM = old_flag


def test_ensure_go_loaded_false_on_cdll_error(monkeypatch: pytest.MonkeyPatch) -> None:
    old_lib = pm._go_lib
    old_flag = pm._HAS_GO_LGSSM
    pm._go_lib = None
    pm._HAS_GO_LGSSM = False
    monkeypatch.setattr(pm, "__file__", str(Path("/tmp/fake_pkg/world_model/predictive_model.py")))
    monkeypatch.setattr("os.path.isfile", lambda _path: True)

    def raise_os_error(_path: str) -> object:
        raise OSError("bad Go shared object")

    try:
        monkeypatch.setattr(ctypes, "CDLL", raise_os_error)
        assert pm._ensure_go_loaded() is False
    finally:
        pm._go_lib = old_lib
        pm._HAS_GO_LGSSM = old_flag


def test_ensure_go_loaded_false_without_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    pm._go_lib = None
    pm._HAS_GO_LGSSM = False
    monkeypatch.setattr(pm, "__file__", str(Path("/tmp/fake_pkg/world_model/predictive_model.py")))
    monkeypatch.setattr("os.path.isfile", lambda _path: True)

    class _FakeLib:
        pass

    monkeypatch.setattr(ctypes, "CDLL", lambda _path: _FakeLib())
    assert pm._ensure_go_loaded() is False


def test_ensure_go_loaded_returns_true_when_already_cached() -> None:
    old_lib = pm._go_lib
    old_flag = pm._HAS_GO_LGSSM
    sentinel = object()
    try:
        pm._go_lib = sentinel
        pm._HAS_GO_LGSSM = True
        assert pm._ensure_go_loaded() is True
        assert pm._go_lib is sentinel
    finally:
        pm._go_lib = old_lib
        pm._HAS_GO_LGSSM = old_flag


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


def test_ensure_julia_loaded_false_when_module_file_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_module = pm._julia_module
    old_flag = pm._HAS_JULIA_LGSSM
    pm._julia_module = None
    pm._HAS_JULIA_LGSSM = False

    class _FakeImportlibUtil:
        @staticmethod
        def find_spec(_name: str) -> object:
            return object()

    class _FakeImportlib:
        @staticmethod
        def import_module(_name: str) -> object:
            return SimpleNamespace(Main=SimpleNamespace())

    monkeypatch.setitem(pm._ensure_julia_loaded.__globals__, "importlib", _FakeImportlib)
    monkeypatch.setitem(pm._ensure_julia_loaded.__globals__, "importlib.util", _FakeImportlibUtil)
    monkeypatch.setattr("os.path.isfile", lambda _path: False)
    try:
        assert pm._ensure_julia_loaded() is False
    finally:
        pm._julia_module = old_module
        pm._HAS_JULIA_LGSSM = old_flag


def test_ensure_julia_loaded_returns_true_when_already_cached() -> None:
    old_module = pm._julia_module
    old_flag = pm._HAS_JULIA_LGSSM
    sentinel = object()
    try:
        pm._julia_module = sentinel
        pm._HAS_JULIA_LGSSM = True
        assert pm._ensure_julia_loaded() is True
        assert pm._julia_module is sentinel
    finally:
        pm._julia_module = old_module
        pm._HAS_JULIA_LGSSM = old_flag


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"A": np.zeros((2, 3))}, "A must be"),
        ({"B": np.zeros((1, 1))}, "B must be"),
        ({"D": np.zeros((1, 0))}, "D must be"),
        ({"Q": np.eye(1)}, "Q must be"),
        ({"R": np.eye(1)}, "R must be"),
        ({"mu_0": np.zeros(1)}, "mu_0 must be"),
        ({"Sigma_0": np.eye(1)}, "Sigma_0 must be"),
    ],
)
def test_linear_gaussian_ssm_rejects_each_malformed_parameter_shape(
    kwargs: dict[str, np.ndarray],
    message: str,
) -> None:
    params = {
        "A": np.eye(2),
        "B": np.zeros((2, 0)),
        "C": np.eye(2),
        "D": np.zeros((2, 0)),
        "Q": np.eye(2),
        "R": np.eye(2),
        "mu_0": np.zeros(2),
        "Sigma_0": np.eye(2),
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        pm.LinearGaussianSSM(**params)


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


def test_filter_go_raises_when_lib_missing() -> None:
    model = _model()
    kf = pm.KalmanFilter(model)
    old = pm._go_lib
    pm._go_lib = None
    try:
        with pytest.raises(RuntimeError, match="not loaded"):
            kf._filter_go(np.zeros((3, 2)), np.zeros((3, 0)))
    finally:
        pm._go_lib = old


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


def test_filter_mojo_dispatches_raw_buffers_and_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMojoLib:
        @staticmethod
        def kalman_filter_c(
            _obs_addr: int,
            _ctl_addr: int,
            _a_addr: int,
            _b_addr: int,
            _c_addr: int,
            _d_addr: int,
            _q_addr: int,
            _r_addr: int,
            _mu0_addr: int,
            _sigma0_addr: int,
            t_len: int,
            _p_dim: int,
            _m_dim: int,
            d_dim: int,
            means_addr: int,
            covs_addr: int,
            pred_means_addr: int,
            pred_covs_addr: int,
            log_lik_addr: int,
        ) -> None:
            means = ctypes.cast(means_addr, ctypes.POINTER(ctypes.c_double))
            covs = ctypes.cast(covs_addr, ctypes.POINTER(ctypes.c_double))
            pred_means = ctypes.cast(pred_means_addr, ctypes.POINTER(ctypes.c_double))
            pred_covs = ctypes.cast(pred_covs_addr, ctypes.POINTER(ctypes.c_double))
            log_lik = ctypes.cast(log_lik_addr, ctypes.POINTER(ctypes.c_double))
            for i in range(t_len * d_dim):
                means[i] = float(i + 1)
                pred_means[i] = float(100 + i)
            for i in range(t_len * d_dim * d_dim):
                covs[i] = float(10 + i)
                pred_covs[i] = float(200 + i)
            log_lik[0] = -12.5

    old_lib = pm._mojo_lib
    old_flag = pm._HAS_MOJO_LGSSM
    try:
        pm._mojo_lib = _FakeMojoLib()
        pm._HAS_MOJO_LGSSM = True
        monkeypatch.setattr(pm, "_ensure_mojo_loaded", lambda: True)
        result = pm.KalmanFilter(_model()).filter(np.zeros((3, 2)), backend="mojo")
    finally:
        pm._mojo_lib = old_lib
        pm._HAS_MOJO_LGSSM = old_flag

    np.testing.assert_allclose(result.means, np.arange(1, 7, dtype=np.float64).reshape(3, 2))
    np.testing.assert_allclose(
        result.pred_means,
        np.arange(100, 106, dtype=np.float64).reshape(3, 2),
    )
    assert result.covariances.shape == (3, 2, 2)
    assert result.pred_covariances.shape == (3, 2, 2)
    assert result.log_likelihood == -12.5


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
