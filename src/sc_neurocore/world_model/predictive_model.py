# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Linear Gaussian State-Space Model with Kalman + RTS + EM

"""Probabilistic predictive world model: linear Gaussian state-space.

Implementation references:
  Kalman, R.E. (1960). "A New Approach to Linear Filtering and
    Prediction Problems." J. Basic Engineering 82(1): 35-45.
  Rauch, H.E., Tung, F. & Striebel, C.T. (1965). "Maximum
    likelihood estimates of linear dynamic systems." AIAA J 3(8):
    1445-1450.  (the RTS smoother)
  Shumway, R.H. & Stoffer, D.S. (1982). "An approach to time
    series smoothing and forecasting using the EM algorithm."
    J Time Series Analysis 3(4): 253-264.
  Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*,
    Springer. §13.3 (linear dynamical systems).
  Murphy, K.P. (2023). *Probabilistic Machine Learning: Advanced
    Topics*, MIT Press. §29 (state-space models).

The previous implementation was a deterministic linear matmul
on a randomly-initialised matrix that called itself "stochastic"
and admitted "(simplified)" in a comment. It was replaced
2026-04-17 per `feedback_sophisticated_from_start.md`.

Model
-----

Latent state x_t ∈ R^d, observation y_t ∈ R^p, control u_t ∈ R^m:

    x_{t+1} = A · x_t + B · u_t + w_t,    w_t ~ N(0, Q)
    y_t     = C · x_t + D · u_t + v_t,    v_t ~ N(0, R)

with prior x_0 ~ N(μ_0, Σ_0). All parameters {A, B, C, D, Q, R,
μ_0, Σ_0} are estimable from data via the EM algorithm
(Shumway & Stoffer 1982). When parameters are known, the Kalman
filter (forward pass) computes p(x_t | y_{1:t}, u_{1:t}) and the
RTS smoother (backward pass) computes p(x_t | y_{1:T}, u_{1:T}).

This module provides:
  - `LinearGaussianSSM` — the model + parameters
  - `KalmanFilter` — forward filtering + log-likelihood
  - `RTSSmoother` — backward smoothing
  - `EMLearner` — parameter estimation via EM
  - `PredictiveWorldModel` — backwards-compatible thin wrapper
    on the above (the legacy class name)
"""

from __future__ import annotations

import importlib as _importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt

_RustKalmanFilter = Callable[..., object]


def _missing_rust_kalman_filter(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("Rust LGSSM backend is not available")


def _load_rust_kalman_filter() -> _RustKalmanFilter:
    try:
        world_model_module = _importlib.import_module("sc_neurocore_engine.world_model")
        return cast(_RustKalmanFilter, world_model_module.get_lgssm_kalman_filter())
    except (AttributeError, ImportError):
        try:
            engine_module = _importlib.import_module("sc_neurocore_engine")
            return cast(_RustKalmanFilter, engine_module.py_lgssm_kalman_filter)
        except (AttributeError, ImportError) as exc:
            raise ImportError("Rust LGSSM backend is not available") from exc


# Detect Rust acceleration backend
try:
    _rust_kalman_filter = _load_rust_kalman_filter()

    _HAS_RUST_LGSSM = True
except (ImportError, AttributeError):
    _rust_kalman_filter = _missing_rust_kalman_filter
    _HAS_RUST_LGSSM = False


# Detect Julia acceleration backend (lazy — Julia startup is ~5 s)
_julia_module = None
_HAS_JULIA_LGSSM = False


# Detect Go acceleration backend (lazy — load shared library on first use)
_go_lib = None
_HAS_GO_LGSSM = False


# Detect Mojo acceleration backend (lazy — load shared library on first use)
_mojo_lib = None
_HAS_MOJO_LGSSM = False


def _ensure_mojo_loaded() -> bool:
    """Lazy-load the Mojo LGSSM shared library on first request.

    Built once via:
      cd src/sc_neurocore/accel/mojo/world_model
      mojo build --emit shared-lib -o liblgssm.so lgssm.mojo

    Per `feedback_mojo_026_ffi_pattern.md`, the @export signature
    accepts only non-parametric types — we pass every matrix as a
    raw `int64` address (numpy `arr.ctypes.data`) and the Mojo
    side reconstructs `UnsafePointer[Float64, MutAnyOrigin]`
    inside the function.
    """
    global _mojo_lib, _HAS_MOJO_LGSSM
    if _mojo_lib is not None:
        return True
    import ctypes
    import os as _os

    so_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "accel",
        "mojo",
        "world_model",
        "liblgssm.so",
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "kalman_filter_c", None)
    if fn is None:
        return False
    # 19 args: 10 input addresses + 4 size scalars + 5 output addresses
    fn.argtypes = [ctypes.c_int64] * 19
    fn.restype = None
    _mojo_lib = lib
    _HAS_MOJO_LGSSM = True
    return True


def _ensure_go_loaded() -> bool:
    """Lazy-load the Go LGSSM shared library on first request.

    The .so is built once via:
      cd src/sc_neurocore/accel/go/lgssm
      go build -buildmode=c-shared -o liblgssm.so lgssm.go

    Returns True if the .so loads + has the expected symbol;
    False otherwise (with reason logged into the caller's
    `unavailable_reason` channel).
    """
    global _go_lib, _HAS_GO_LGSSM
    if _go_lib is not None:
        return True
    import ctypes
    import os as _os

    so_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "accel",
        "go",
        "lgssm",
        "liblgssm.so",
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "kalman_filter_c", None)
    if fn is None:
        return False
    # Configure C signature: 18 pointers/int args, void return.
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # obs
        ctypes.POINTER(ctypes.c_double),  # ctl
        ctypes.POINTER(ctypes.c_double),  # A
        ctypes.POINTER(ctypes.c_double),  # B
        ctypes.POINTER(ctypes.c_double),  # C
        ctypes.POINTER(ctypes.c_double),  # D
        ctypes.POINTER(ctypes.c_double),  # Q
        ctypes.POINTER(ctypes.c_double),  # R
        ctypes.POINTER(ctypes.c_double),  # mu_0
        ctypes.POINTER(ctypes.c_double),  # Sigma_0
        ctypes.c_int,  # T
        ctypes.c_int,  # p
        ctypes.c_int,  # m
        ctypes.c_int,  # d
        ctypes.POINTER(ctypes.c_double),  # means_out
        ctypes.POINTER(ctypes.c_double),  # covs_out
        ctypes.POINTER(ctypes.c_double),  # pred_means_out
        ctypes.POINTER(ctypes.c_double),  # pred_covs_out
        ctypes.POINTER(ctypes.c_double),  # log_lik_out
    ]
    fn.restype = None
    _go_lib = lib
    _HAS_GO_LGSSM = True
    return True


def _ensure_julia_loaded() -> bool:
    """Lazy-load the Julia LGSSM module on first request.

    Returns True if Julia + the .jl module are available; False otherwise.
    Caches the loaded module in `_julia_module` for subsequent calls.
    Julia startup latency is ~5 s — never paid unless `backend='julia'`
    is explicitly requested.
    """
    global _julia_module, _HAS_JULIA_LGSSM
    if _julia_module is not None:
        return True
    import os as _os

    importlib_module = globals().get("importlib")
    if importlib_module is None:
        import importlib as importlib_module
    importlib_util = globals().get("importlib.util")
    if importlib_util is None:
        import importlib.util as importlib_util

    assert importlib_module is not None
    assert importlib_util is not None
    spec = importlib_util.find_spec("juliacall")
    if spec is None:
        return False
    juliacall = importlib_module.import_module("juliacall")
    jl = juliacall.Main
    jl_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "accel",
        "julia",
        "world_model",
        "predictive_model.jl",
    )
    if not _os.path.isfile(jl_path):
        return False
    jl.include(jl_path)
    _julia_module = jl.PredictiveModelAccel
    _HAS_JULIA_LGSSM = True
    return True


# ───────────────────────── model parameters ─────────────────────────


@dataclass
class LinearGaussianSSM:
    """Parameters of a discrete-time linear Gaussian state-space model.

    All matrices are NumPy arrays of dtype float64. Shapes:

    - ``A``: (d, d) state transition
    - ``B``: (d, m) control input
    - ``C``: (p, d) observation
    - ``D``: (p, m) feed-through
    - ``Q``: (d, d) process noise covariance (PSD)
    - ``R``: (p, p) observation noise covariance (PSD)
    - ``mu_0``: (d,) prior mean
    - ``Sigma_0``: (d, d) prior covariance
    """

    A: np.ndarray[Any, Any]
    B: np.ndarray[Any, Any]
    C: np.ndarray[Any, Any]
    D: np.ndarray[Any, Any]
    Q: np.ndarray[Any, Any]
    R: np.ndarray[Any, Any]
    mu_0: np.ndarray[Any, Any]
    Sigma_0: np.ndarray[Any, Any]

    @property
    def state_dim(self) -> int:
        """Return the latent state dimension."""
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        """Return the observation dimension."""
        return int(self.C.shape[0])

    @property
    def control_dim(self) -> int:
        """Return the control-input dimension."""
        return int(self.B.shape[1])

    def __post_init__(self) -> None:
        """Validate the state-space matrix dimensions."""
        d = self.state_dim
        p = self.obs_dim
        m = self.control_dim
        if self.A.shape != (d, d):
            raise ValueError(f"A must be ({d},{d}), got {self.A.shape}")
        if self.B.shape != (d, m):
            raise ValueError(f"B must be ({d},{m}), got {self.B.shape}")
        if self.C.shape != (p, d):
            raise ValueError(f"C must be ({p},{d}), got {self.C.shape}")
        if self.D.shape != (p, m):
            raise ValueError(f"D must be ({p},{m}), got {self.D.shape}")
        if self.Q.shape != (d, d):
            raise ValueError(f"Q must be ({d},{d}), got {self.Q.shape}")
        if self.R.shape != (p, p):
            raise ValueError(f"R must be ({p},{p}), got {self.R.shape}")
        if self.mu_0.shape != (d,):
            raise ValueError(f"mu_0 must be ({d},), got {self.mu_0.shape}")
        if self.Sigma_0.shape != (d, d):
            raise ValueError(f"Sigma_0 must be ({d},{d}), got {self.Sigma_0.shape}")
        # PSD checks (loose — accept symmetry + non-negative diag)
        for name, M in (("Q", self.Q), ("R", self.R), ("Sigma_0", self.Sigma_0)):
            if not np.allclose(M, M.T, atol=1e-9):
                raise ValueError(f"{name} must be symmetric")
            if np.any(np.diag(M) < -1e-9):
                raise ValueError(f"{name} has negative diagonal entry")

    @classmethod
    def random(
        cls,
        state_dim: int,
        obs_dim: int,
        control_dim: int = 0,
        seed: int = 42,
    ) -> LinearGaussianSSM:
        """Construct a stable random LGSSM for smoke tests / initialisation."""
        rng = np.random.default_rng(seed)
        d, p, m = state_dim, obs_dim, max(control_dim, 1)
        # Stable A: spectral radius < 1
        raw = rng.standard_normal((d, d)) * 0.5
        eigmax = float(np.max(np.abs(np.linalg.eigvals(raw))))
        A = raw * (0.95 / max(eigmax, 1e-12))
        B = rng.standard_normal((d, control_dim)) if control_dim > 0 else np.zeros((d, 0))
        C = rng.standard_normal((p, d)) * 0.5
        D = rng.standard_normal((p, control_dim)) if control_dim > 0 else np.zeros((p, 0))
        Q = np.eye(d) * 0.1
        R = np.eye(p) * 0.1
        mu_0 = np.zeros(d)
        Sigma_0 = np.eye(d)
        return cls(A=A, B=B, C=C, D=D, Q=Q, R=R, mu_0=mu_0, Sigma_0=Sigma_0)


# ───────────────────────── Kalman filter ─────────────────────────


@dataclass
class FilterResult:
    """Output of `KalmanFilter.filter()`.

    Attributes
    ----------
    means : np.ndarray, shape (T, d)
        Filtered means E[x_t | y_{1:t}, u_{1:t}].
    covariances : np.ndarray, shape (T, d, d)
        Filtered covariances Cov[x_t | y_{1:t}, u_{1:t}].
    pred_means : np.ndarray, shape (T, d)
        One-step-ahead predicted means E[x_t | y_{1:t-1}, u_{1:t-1}].
    pred_covariances : np.ndarray, shape (T, d, d)
        One-step-ahead predicted covariances.
    log_likelihood : float
        Log p(y_{1:T} | u_{1:T}) under the model — used by EM.
    """

    means: np.ndarray[Any, Any]
    covariances: np.ndarray[Any, Any]
    pred_means: np.ndarray[Any, Any]
    pred_covariances: np.ndarray[Any, Any]
    log_likelihood: float


class KalmanFilter:
    """Forward Kalman filter for `LinearGaussianSSM`.

    Computes the filtering distribution p(x_t | y_{1:t}, u_{1:t})
    for a fully-observed sequence of observations + (optional)
    controls. Algorithm: standard Kalman update equations
    (Bishop 2006 §13.3.1).
    """

    def __init__(self, model: LinearGaussianSSM) -> None:
        self.model = model

    def filter(
        self,
        observations: np.ndarray[Any, Any],
        controls: Optional[np.ndarray[Any, Any]] = None,
        backend: str = "auto",
    ) -> FilterResult:
        """Run forward filtering on a sequence.

        Parameters
        ----------
        observations : np.ndarray, shape (T, p)
            Observation sequence.
        controls : np.ndarray, shape (T, m), optional
            Control input sequence. Required iff `model.control_dim > 0`.
        backend : {"auto", "rust", "julia", "go", "python"}
            Acceleration backend selector.
            - "auto" picks Rust > Julia > Go > Python in priority
              order (Rust if `sc_neurocore_engine` wheel is built,
              else Julia if `juliacall` is installed, else Go if
              `liblgssm.so` is built, else Python).
            - "rust" / "julia" / "go" / "python" force the named
              backend (raises RuntimeError if the named backend
              is not available).

        Returns
        -------
        FilterResult
        """
        T, p = observations.shape
        if p != self.model.obs_dim:
            raise ValueError(f"obs dim {p} ≠ model.obs_dim {self.model.obs_dim}")
        m = self.model.control_dim
        if m > 0:
            if controls is None or controls.shape != (T, m):
                raise ValueError(f"controls must have shape ({T},{m})")
        else:
            controls = np.zeros((T, 0)) if controls is None else controls

        # Backend dispatch
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        if backend == "rust" and not _HAS_RUST_LGSSM:
            raise RuntimeError(
                "Rust LGSSM backend requested but py_lgssm_kalman_filter "
                "is not available; install sc_neurocore_engine wheel."
            )
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError(
                "Julia LGSSM backend requested but juliacall + the "
                "predictive_model.jl module is not available; "
                "install juliacall (pip install juliacall)."
            )
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go LGSSM backend requested but liblgssm.so is not "
                "built; run `cd src/sc_neurocore/accel/go/lgssm && "
                "go build -buildmode=c-shared -o liblgssm.so lgssm.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo LGSSM backend requested but liblgssm.so is not "
                "built; run `cd src/sc_neurocore/accel/mojo/world_model "
                "&& mojo build --emit shared-lib -o liblgssm.so lgssm.mojo`."
            )
        if (backend == "auto" and _HAS_RUST_LGSSM) or backend == "rust":
            return self._filter_rust(observations, controls)
        if backend == "julia":
            return self._filter_julia(observations, controls)
        if backend == "go":
            return self._filter_go(observations, controls)
        if backend == "mojo":
            return self._filter_mojo(observations, controls)

        d = self.model.state_dim
        A, B, C, D = self.model.A, self.model.B, self.model.C, self.model.D
        Q, R = self.model.Q, self.model.R

        means = np.zeros((T, d))
        covs = np.zeros((T, d, d))
        pred_means = np.zeros((T, d))
        pred_covs = np.zeros((T, d, d))

        # Initial prediction (from prior)
        x_pred = self.model.mu_0.copy()
        P_pred = self.model.Sigma_0.copy()

        log_lik = 0.0
        for t in range(T):
            pred_means[t] = x_pred
            pred_covs[t] = P_pred

            y_t = observations[t]
            u_t = controls[t] if m > 0 else None

            # Innovation (residual)
            y_hat = C @ x_pred + (D @ u_t if m > 0 else 0.0)
            innov = y_t - y_hat
            S = C @ P_pred @ C.T + R  # innovation covariance

            # Log-likelihood contribution: log N(y_t | y_hat, S)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                # Should not happen for PSD R; defensive only.
                raise np.linalg.LinAlgError("non-PSD innovation covariance")
            S_inv_innov = np.linalg.solve(S, innov)
            log_lik += -0.5 * (p * np.log(2 * np.pi) + logdet + float(innov @ S_inv_innov))

            # Kalman gain
            K = P_pred @ C.T @ np.linalg.inv(S)

            # Update (filtered estimate)
            x_filt = x_pred + K @ innov
            # Joseph form for numerical stability of P_filt:
            #   (I - K C) P_pred (I - K C)^T + K R K^T
            I = np.eye(d)
            P_filt = (I - K @ C) @ P_pred @ (I - K @ C).T + K @ R @ K.T

            means[t] = x_filt
            covs[t] = P_filt

            # Predict next
            x_pred = A @ x_filt + (B @ u_t if m > 0 else 0.0)
            P_pred = A @ P_filt @ A.T + Q

        return FilterResult(
            means=means,
            covariances=covs,
            pred_means=pred_means,
            pred_covariances=pred_covs,
            log_likelihood=log_lik,
        )

    def _filter_rust(
        self,
        observations: np.ndarray[Any, Any],
        controls: np.ndarray[Any, Any],
    ) -> FilterResult:
        """Forward-pass dispatch to the Rust LGSSM Kalman filter.

        Marshals all matrices as flat row-major Vec<f64> across the
        PyO3 boundary; the Rust side reconstructs ndarray::Array2
        via `Array2::from_shape_vec`. Result identical to the Python
        path within float64 round-off (verified by the parity test
        in tests/test_world_model/test_predictive_model.py).
        """
        if _rust_kalman_filter is None:
            raise RuntimeError("Rust backend probed False; cannot dispatch")

        T, p = observations.shape
        m = self.model.control_dim
        d = self.model.state_dim
        A, B, C, D = self.model.A, self.model.B, self.model.C, self.model.D
        Q, R = self.model.Q, self.model.R

        # Flatten to row-major Vec<f64> for the PyO3 marshalling.
        result = cast(
            Mapping[str, object],
            _rust_kalman_filter(
                obs_flat=observations.astype(np.float64).ravel(order="C").tolist(),
                controls_flat=controls.astype(np.float64).ravel(order="C").tolist(),
                t_len=T,
                p_dim=p,
                m_dim=m,
                a_flat=A.ravel(order="C").tolist(),
                b_flat=B.ravel(order="C").tolist(),
                c_flat=C.ravel(order="C").tolist(),
                d_flat=D.ravel(order="C").tolist(),
                q_flat=Q.ravel(order="C").tolist(),
                r_flat=R.ravel(order="C").tolist(),
                mu_0=self.model.mu_0.tolist(),
                sigma_0_flat=self.model.Sigma_0.ravel(order="C").tolist(),
                d_dim=d,
            ),
        )
        return FilterResult(
            means=np.array(result["means"], dtype=np.float64),
            covariances=np.array(result["covariances"], dtype=np.float64),
            pred_means=np.array(result["pred_means"], dtype=np.float64),
            pred_covariances=np.array(result["pred_covariances"], dtype=np.float64),
            log_likelihood=float(cast(float, result["log_likelihood"])),
        )

    def _filter_julia(
        self,
        observations: np.ndarray[Any, Any],
        controls: np.ndarray[Any, Any],
    ) -> FilterResult:
        """Forward-pass dispatch to the Julia LGSSM Kalman filter.

        Calls the `kalman_filter` function from
        `accel/julia/world_model/predictive_model.jl` via juliacall.
        Result identical to Python + Rust paths within float64
        round-off (verified by parity tests).
        """
        if _julia_module is None:
            raise RuntimeError("Julia module not loaded; cannot dispatch")

        result = _julia_module.kalman_filter(
            np.ascontiguousarray(observations, dtype=np.float64),
            np.ascontiguousarray(controls, dtype=np.float64),
            np.ascontiguousarray(self.model.A, dtype=np.float64),
            np.ascontiguousarray(self.model.B, dtype=np.float64),
            np.ascontiguousarray(self.model.C, dtype=np.float64),
            np.ascontiguousarray(self.model.D, dtype=np.float64),
            np.ascontiguousarray(self.model.Q, dtype=np.float64),
            np.ascontiguousarray(self.model.R, dtype=np.float64),
            np.ascontiguousarray(self.model.mu_0, dtype=np.float64),
            np.ascontiguousarray(self.model.Sigma_0, dtype=np.float64),
        )
        # juliacall returns NamedTuple → convert fields to NumPy
        return FilterResult(
            means=np.asarray(result.means, dtype=np.float64),
            covariances=np.asarray(result.covariances, dtype=np.float64),
            pred_means=np.asarray(result.pred_means, dtype=np.float64),
            pred_covariances=np.asarray(result.pred_covs, dtype=np.float64),
            log_likelihood=float(result.log_lik),
        )

    def _filter_go(
        self,
        observations: np.ndarray[Any, Any],
        controls: np.ndarray[Any, Any],
    ) -> FilterResult:
        """Forward-pass dispatch to the Go LGSSM Kalman filter.

        Calls the C-ABI `kalman_filter_c` function exported by the
        Go shared library `accel/go/lgssm/liblgssm.so`. Buffers are
        passed as numpy arrays; the Go side fills `means_out`,
        `covs_out`, `pred_means_out`, `pred_covs_out`, and a
        single-element `log_lik_out`.
        """
        if _go_lib is None:
            raise RuntimeError("Go shared library not loaded; cannot dispatch")

        import ctypes

        T, p = observations.shape
        d = self.model.state_dim
        m = self.model.control_dim

        # Input buffers (must remain alive — keep Python refs)
        obs_arr = np.ascontiguousarray(observations, dtype=np.float64)
        ctl_arr = np.ascontiguousarray(controls, dtype=np.float64)
        a_arr = np.ascontiguousarray(self.model.A, dtype=np.float64)
        b_arr = np.ascontiguousarray(self.model.B, dtype=np.float64)
        c_arr = np.ascontiguousarray(self.model.C, dtype=np.float64)
        d_arr = np.ascontiguousarray(self.model.D, dtype=np.float64)
        q_arr = np.ascontiguousarray(self.model.Q, dtype=np.float64)
        r_arr = np.ascontiguousarray(self.model.R, dtype=np.float64)
        mu0_arr = np.ascontiguousarray(self.model.mu_0, dtype=np.float64)
        sigma0_arr = np.ascontiguousarray(self.model.Sigma_0, dtype=np.float64)

        # Output buffers
        means_out = np.zeros((T, d), dtype=np.float64, order="C")
        covs_out = np.zeros((T, d, d), dtype=np.float64, order="C")
        pred_means_out = np.zeros((T, d), dtype=np.float64, order="C")
        pred_covs_out = np.zeros((T, d, d), dtype=np.float64, order="C")
        log_lik_out: npt.NDArray[np.float64] = np.zeros(1, dtype=np.float64, order="C")

        _go_lib.kalman_filter_c(
            obs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctl_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            q_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            r_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            mu0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sigma0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(T),
            ctypes.c_int(p),
            ctypes.c_int(m),
            ctypes.c_int(d),
            means_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            covs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pred_means_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pred_covs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            log_lik_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return FilterResult(
            means=means_out,
            covariances=covs_out,
            pred_means=pred_means_out,
            pred_covariances=pred_covs_out,
            log_likelihood=float(log_lik_out[0]),
        )

    def _filter_mojo(
        self,
        observations: np.ndarray[Any, Any],
        controls: np.ndarray[Any, Any],
    ) -> FilterResult:
        """Forward-pass dispatch to the Mojo LGSSM Kalman filter.

        Calls the C-ABI `kalman_filter_c` function exported by the
        Mojo shared library `accel/mojo/world_model/liblgssm.so`.
        Per `feedback_mojo_026_ffi_pattern.md`, every matrix is
        passed as a raw int64 address (no parametric pointer types
        in the @export signature); the Mojo side reconstructs
        `UnsafePointer[Float64, MutAnyOrigin]` inside.
        """
        if _mojo_lib is None:
            raise RuntimeError("Mojo shared library not loaded; cannot dispatch")

        T, p = observations.shape
        d = self.model.state_dim
        m = self.model.control_dim

        obs_arr = np.ascontiguousarray(observations, dtype=np.float64)
        ctl_arr = np.ascontiguousarray(controls, dtype=np.float64)
        a_arr = np.ascontiguousarray(self.model.A, dtype=np.float64)
        b_arr = np.ascontiguousarray(self.model.B, dtype=np.float64)
        c_arr = np.ascontiguousarray(self.model.C, dtype=np.float64)
        d_arr = np.ascontiguousarray(self.model.D, dtype=np.float64)
        q_arr = np.ascontiguousarray(self.model.Q, dtype=np.float64)
        r_arr = np.ascontiguousarray(self.model.R, dtype=np.float64)
        mu0_arr = np.ascontiguousarray(self.model.mu_0, dtype=np.float64)
        sigma0_arr = np.ascontiguousarray(self.model.Sigma_0, dtype=np.float64)

        means_out = np.zeros((T, d), dtype=np.float64, order="C")
        covs_out = np.zeros((T, d, d), dtype=np.float64, order="C")
        pred_means_out = np.zeros((T, d), dtype=np.float64, order="C")
        pred_covs_out = np.zeros((T, d, d), dtype=np.float64, order="C")
        log_lik_out: npt.NDArray[np.float64] = np.zeros(1, dtype=np.float64, order="C")

        _mojo_lib.kalman_filter_c(
            obs_arr.ctypes.data,
            ctl_arr.ctypes.data,
            a_arr.ctypes.data,
            b_arr.ctypes.data,
            c_arr.ctypes.data,
            d_arr.ctypes.data,
            q_arr.ctypes.data,
            r_arr.ctypes.data,
            mu0_arr.ctypes.data,
            sigma0_arr.ctypes.data,
            T,
            p,
            m,
            d,
            means_out.ctypes.data,
            covs_out.ctypes.data,
            pred_means_out.ctypes.data,
            pred_covs_out.ctypes.data,
            log_lik_out.ctypes.data,
        )
        return FilterResult(
            means=means_out,
            covariances=covs_out,
            pred_means=pred_means_out,
            pred_covariances=pred_covs_out,
            log_likelihood=float(log_lik_out[0]),
        )


# ───────────────────────── RTS smoother ─────────────────────────


@dataclass
class SmoothResult:
    """Output of `RTSSmoother.smooth()`.

    Attributes
    ----------
    means : np.ndarray, shape (T, d)
        Smoothed means E[x_t | y_{1:T}, u_{1:T}].
    covariances : np.ndarray, shape (T, d, d)
        Smoothed covariances Cov[x_t | y_{1:T}, u_{1:T}].
    cross_covariances : np.ndarray, shape (T-1, d, d)
        Lag-1 smoothed cross-covariances Cov[x_t, x_{t+1} | y_{1:T}].
        Required by the EM M-step.
    """

    means: np.ndarray[Any, Any]
    covariances: np.ndarray[Any, Any]
    cross_covariances: np.ndarray[Any, Any]


class RTSSmoother:
    """Rauch-Tung-Striebel backward smoother (1965).

    Consumes a `FilterResult` (forward pass) and produces the
    smoothing distribution p(x_t | y_{1:T}, u_{1:T}) for every t.
    """

    def __init__(self, model: LinearGaussianSSM) -> None:
        self.model = model

    def smooth(self, filter_result: FilterResult) -> SmoothResult:
        """Run RTS smoothing over a forward-filter result."""
        T, d = filter_result.means.shape
        A = self.model.A

        smoothed_means = filter_result.means.copy()
        smoothed_covs = filter_result.covariances.copy()
        cross_covs = np.zeros((T - 1, d, d))

        for t in range(T - 2, -1, -1):
            P_pred_next = filter_result.pred_covariances[t + 1]
            # RTS gain
            J = filter_result.covariances[t] @ A.T @ np.linalg.inv(P_pred_next)
            smoothed_means[t] = filter_result.means[t] + J @ (
                smoothed_means[t + 1] - filter_result.pred_means[t + 1]
            )
            smoothed_covs[t] = (
                filter_result.covariances[t] + J @ (smoothed_covs[t + 1] - P_pred_next) @ J.T
            )
            # Lag-1 smoothed covariance: Cov(x_t, x_{t+1} | y_{1:T})
            cross_covs[t] = J @ smoothed_covs[t + 1]

        return SmoothResult(
            means=smoothed_means,
            covariances=smoothed_covs,
            cross_covariances=cross_covs,
        )


# ───────────────────────── EM learner ─────────────────────────


class EMLearner:
    """Expectation-Maximisation parameter estimator for LGSSM.

    Reference: Shumway & Stoffer (1982), Bishop (2006) §13.3.2.

    Per iteration:
      E-step: run Kalman filter + RTS smoother → posterior means,
              covariances, cross-covariances.
      M-step: closed-form update for {A, C, Q, R, mu_0, Sigma_0}
              from the smoothed posterior. (B and D are treated
              as known; estimating them simultaneously requires
              a more involved derivation.)

    Convergence: log-likelihood is monotone non-decreasing
    across iterations under exact arithmetic.
    """

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.log_likelihood_history: list[float] = []

    def fit(
        self,
        observations: np.ndarray[Any, Any],
        initial_model: LinearGaussianSSM,
        controls: Optional[np.ndarray[Any, Any]] = None,
    ) -> LinearGaussianSSM:
        """Estimate model parameters from a single observation sequence."""
        T, p = observations.shape
        d = initial_model.state_dim
        m = initial_model.control_dim
        model = initial_model
        prev_ll = -np.inf
        self.log_likelihood_history = []

        for it in range(self.max_iter):
            # E-step
            kf = KalmanFilter(model)
            fr = kf.filter(observations, controls=controls)
            sr = RTSSmoother(model).smooth(fr)
            self.log_likelihood_history.append(fr.log_likelihood)

            # M-step (B and D held fixed)
            # Sufficient statistics
            x = sr.means  # (T, d)
            P = sr.covariances  # (T, d, d)
            P_lag = sr.cross_covariances  # (T-1, d, d)

            # E[x_t x_t^T] = P_t + x_t x_t^T
            Exx = P + np.einsum("ti,tj->tij", x, x)
            # E[x_{t+1} x_t^T] = P_lag_t + x_{t+1} x_t^T
            Ex1x = P_lag + np.einsum("ti,tj->tij", x[1:], x[:-1])

            # A_new = (Σ_{t=1..T-1} E[x_{t+1} x_t^T]) ·
            #         (Σ_{t=0..T-2} E[x_t x_t^T])^{-1}
            num = Ex1x.sum(axis=0)
            den = Exx[:-1].sum(axis=0)
            A_new = num @ np.linalg.inv(den)

            # Q_new = (1/(T-1)) Σ_{t=1..T-1} (E[x_t x_t^T]
            #         - A E[x_t x_{t-1}^T]^T - E[x_t x_{t-1}^T] A^T
            #         + A E[x_{t-1} x_{t-1}^T] A^T)
            Q_acc = np.zeros((d, d))
            for t in range(T - 1):
                Q_acc += (
                    Exx[t + 1] - A_new @ Ex1x[t].T - Ex1x[t] @ A_new.T + A_new @ Exx[t] @ A_new.T
                )
            Q_new = Q_acc / (T - 1)
            Q_new = 0.5 * (Q_new + Q_new.T)  # symmetrise

            # C_new = (Σ_t y_t x_t^T) (Σ_t E[x_t x_t^T])^{-1}
            num_C = np.einsum("ti,tj->ij", observations, x)
            den_C = Exx.sum(axis=0)
            C_new = num_C @ np.linalg.inv(den_C)

            # R_new = (1/T) Σ_t (y_t y_t^T - C y_t x_t^T x_t y_t^T C^T
            #         + ... )  collapsed:
            #         (1/T) Σ_t (y_t - C x_t)(y_t - C x_t)^T
            #         + C (Σ_t P_t) C^T
            R_new = np.zeros((p, p))
            residuals = observations - x @ C_new.T
            R_new = residuals.T @ residuals
            for t in range(T):
                R_new += C_new @ P[t] @ C_new.T
            R_new /= T
            R_new = 0.5 * (R_new + R_new.T)

            # μ_0, Σ_0 from smoothed first state
            mu_0_new = sr.means[0].copy()
            Sigma_0_new = sr.covariances[0].copy()

            # Update model
            model = LinearGaussianSSM(
                A=A_new,
                B=model.B,
                C=C_new,
                D=model.D,
                Q=Q_new,
                R=R_new,
                mu_0=mu_0_new,
                Sigma_0=Sigma_0_new,
            )

            if abs(fr.log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = fr.log_likelihood

        return model


# ───────────────────────── legacy wrapper ─────────────────────────


@dataclass
class PredictiveWorldModel:
    """Probabilistic predictive world model based on a Linear Gaussian SSM.

    The legacy 65-LOC `predict_next_state` / `forecast` API is
    preserved as a thin wrapper on the proper `LinearGaussianSSM`
    + `KalmanFilter` infrastructure above. The previous
    deterministic linear matmul + clip implementation was replaced
    2026-04-17 per `feedback_sophisticated_from_start.md`.
    """

    state_dim: int
    action_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialise the underlying linear-Gaussian state-space model."""
        self.model: LinearGaussianSSM = LinearGaussianSSM.random(
            state_dim=self.state_dim,
            obs_dim=self.state_dim,  # observe the state directly
            control_dim=self.action_dim,
            seed=self.seed,
        )
        # Filtered posterior moments — updated by `predict_next_state`.
        self._mu: np.ndarray[Any, Any] = self.model.mu_0.copy()
        self._Sigma: np.ndarray[Any, Any] = self.model.Sigma_0.copy()

    def reset(self) -> None:
        """Reset the running belief state to the prior mean."""
        self._mu = self.model.mu_0.copy()
        self._Sigma = self.model.Sigma_0.copy()

    def predict_next_state(
        self,
        current_state: np.ndarray[Any, Any],
        action: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Predict E[x_{t+1} | x_t, u_t] under the SSM dynamics.

        Returns the deterministic mean prediction; for a full
        probabilistic forecast use `predict_next_state_with_cov`.
        """
        u: npt.NDArray[np.float64] = action.astype(np.float64)
        if u.shape == ():
            u = u[np.newaxis]
        prediction: npt.NDArray[np.float64] = self.model.A @ current_state + (
            self.model.B @ u if self.model.control_dim > 0 else 0.0
        )
        return prediction

    def predict_next_state_with_cov(
        self,
        current_state: np.ndarray[Any, Any],
        current_cov: np.ndarray[Any, Any],
        action: np.ndarray[Any, Any],
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Predict mean + covariance of x_{t+1} given (x_t, Σ_t, u_t)."""
        mu_next = self.predict_next_state(current_state, action)
        Sigma_next = self.model.A @ current_cov @ self.model.A.T + self.model.Q
        return mu_next, Sigma_next

    def forecast(
        self,
        initial_state: np.ndarray[Any, Any],
        actions: list[np.ndarray[Any, Any]],
    ) -> list[np.ndarray[Any, Any]]:
        """Multi-step deterministic forecast (mean trajectory)."""
        traj: list[np.ndarray[Any, Any]] = []
        x: npt.NDArray[np.float64] = initial_state.astype(np.float64)
        for a in actions:
            x = self.predict_next_state(x, np.asarray(a, dtype=np.float64))
            traj.append(x.copy())
        return traj

    def forecast_with_cov(
        self,
        initial_state: np.ndarray[Any, Any],
        initial_cov: np.ndarray[Any, Any],
        actions: list[np.ndarray[Any, Any]],
    ) -> list[Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        """Multi-step probabilistic forecast (mean + cov trajectory)."""
        traj: list[Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
        x: npt.NDArray[np.float64] = initial_state.astype(np.float64)
        P: npt.NDArray[np.float64] = initial_cov.astype(np.float64)
        for a in actions:
            x, P = self.predict_next_state_with_cov(
                x,
                P,
                np.asarray(a, dtype=np.float64),
            )
            traj.append((x.copy(), P.copy()))
        return traj
