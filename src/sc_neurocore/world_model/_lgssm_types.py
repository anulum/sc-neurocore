# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Linear Gaussian state-space data contracts

"""Validated parameter and result types for linear Gaussian state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_MATRIX_ROUNDOFF_MULTIPLIER = 32.0


def _matrix_tolerance(matrix: FloatArray) -> float:
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    return float(_MATRIX_ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * matrix.shape[0] * scale)


def _as_float_array(value: npt.ArrayLike, *, name: str, ndim: int) -> FloatArray:
    try:
        array = np.array(value, dtype=np.float64, order="C", copy=True)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_dimension(value: int, *, name: str, allow_zero: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    dimension = int(value)
    lower_bound = 0 if allow_zero else 1
    if dimension < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return dimension


def _symmetrise(matrix: FloatArray) -> FloatArray:
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def _require_symmetric(matrix: FloatArray, *, name: str) -> None:
    tolerance = _matrix_tolerance(matrix)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")


def _require_positive_semidefinite(matrix: FloatArray, *, name: str) -> None:
    _require_symmetric(matrix, name=name)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(_symmetrise(matrix))))
    if minimum_eigenvalue < -_matrix_tolerance(matrix):
        raise ValueError(f"{name} must be positive semidefinite")


def _require_positive_semidefinite_stack(
    matrices: FloatArray,
    *,
    name: str,
) -> None:
    """Validate a non-empty covariance stack without a Python matrix loop."""
    transposed = np.swapaxes(matrices, -1, -2)
    scales = np.maximum(
        1.0,
        np.linalg.norm(matrices, ord=2, axis=(-2, -1)),
    )
    tolerances = (
        _MATRIX_ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * matrices.shape[-1] * scales
    )
    symmetry_errors = np.max(np.abs(matrices - transposed), axis=(-2, -1))
    if np.any(symmetry_errors > tolerances):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (matrices + transposed)
    minimum_eigenvalues = np.min(np.linalg.eigvalsh(symmetric), axis=-1)
    if np.any(minimum_eigenvalues < -tolerances):
        raise ValueError(f"{name} must be positive semidefinite")


def _require_positive_definite(matrix: FloatArray, *, name: str) -> None:
    _require_symmetric(matrix, name=name)
    try:
        np.linalg.cholesky(_symmetrise(matrix))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


def _stabilise_covariance(matrix: FloatArray, *, positive_definite: bool) -> FloatArray:
    symmetric = _symmetrise(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = _matrix_tolerance(symmetric) if positive_definite else 0.0
    adjusted = np.maximum(eigenvalues, floor)
    return _symmetrise((eigenvectors * adjusted) @ eigenvectors.T)


def _solve_positive_definite(matrix: FloatArray, right_hand_side: FloatArray) -> FloatArray:
    symmetric = _symmetrise(matrix)
    try:
        lower = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("matrix must be positive definite") from exc
    intermediate = np.linalg.solve(lower, right_hand_side)
    return np.asarray(np.linalg.solve(lower.T, intermediate), dtype=np.float64)


def _right_solve_positive_definite(matrix: FloatArray, numerator: FloatArray) -> FloatArray:
    return np.asarray(_solve_positive_definite(matrix, numerator.T).T, dtype=np.float64)


def _normalise_observations(observations: npt.ArrayLike, *, obs_dim: int) -> FloatArray:
    array = _as_float_array(observations, name="observations", ndim=2)
    if array.shape[0] == 0:
        raise ValueError("observations must contain at least one time step")
    if array.shape[1] != obs_dim:
        raise ValueError(f"observations must have shape (T, {obs_dim}), got {array.shape}")
    return array


def _normalise_controls(
    controls: npt.ArrayLike | None,
    *,
    time_steps: int,
    control_dim: int,
) -> FloatArray:
    expected_shape = (time_steps, control_dim)
    if controls is None:
        if control_dim > 0:
            raise ValueError(f"controls must have shape {expected_shape}")
        return np.zeros(expected_shape, dtype=np.float64)
    array = _as_float_array(controls, name="controls", ndim=2)
    if array.shape != expected_shape:
        raise ValueError(f"controls must have shape {expected_shape}, got {array.shape}")
    return array


def _normalise_vector(
    value: npt.ArrayLike,
    *,
    name: str,
    length: int,
    allow_scalar: bool = False,
) -> FloatArray:
    try:
        array = np.array(value, dtype=np.float64, order="C", copy=True)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if allow_scalar and array.ndim == 0 and length == 1:
        array = array.reshape(1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalise_state_covariance(
    covariance: npt.ArrayLike,
    *,
    state_dim: int,
    name: str,
) -> FloatArray:
    array = _as_float_array(covariance, name=name, ndim=2)
    if array.shape != (state_dim, state_dim):
        raise ValueError(f"{name} must have shape ({state_dim}, {state_dim}), got {array.shape}")
    _require_positive_semidefinite(array, name=name)
    return array


@dataclass
class LinearGaussianSSM:
    """Parameters of a discrete-time linear Gaussian state-space model.

    Parameters
    ----------
    A : numpy.ndarray, shape (d, d)
        State-transition matrix.
    B : numpy.ndarray, shape (d, m)
        Control-input matrix. Use an empty second dimension when ``m = 0``.
    C : numpy.ndarray, shape (p, d)
        Observation matrix.
    D : numpy.ndarray, shape (p, m)
        Direct control-to-observation matrix.
    Q : numpy.ndarray, shape (d, d)
        Symmetric positive-semidefinite process covariance.
    R : numpy.ndarray, shape (p, p)
        Symmetric positive-definite observation covariance.
    mu_0 : numpy.ndarray, shape (d,)
        Prior state mean.
    Sigma_0 : numpy.ndarray, shape (d, d)
        Symmetric positive-definite prior covariance.

    Raises
    ------
    ValueError
        If a parameter has an incompatible shape, non-finite value, or invalid
        covariance contract.

    """

    A: FloatArray
    B: FloatArray
    C: FloatArray
    D: FloatArray
    Q: FloatArray
    R: FloatArray
    mu_0: FloatArray
    Sigma_0: FloatArray

    def __post_init__(self) -> None:
        """Copy parameters into finite, C-contiguous float64 arrays and validate them."""
        self.A = _as_float_array(self.A, name="A", ndim=2)
        self.B = _as_float_array(self.B, name="B", ndim=2)
        self.C = _as_float_array(self.C, name="C", ndim=2)
        self.D = _as_float_array(self.D, name="D", ndim=2)
        self.Q = _as_float_array(self.Q, name="Q", ndim=2)
        self.R = _as_float_array(self.R, name="R", ndim=2)
        self.mu_0 = _as_float_array(self.mu_0, name="mu_0", ndim=1)
        self.Sigma_0 = _as_float_array(self.Sigma_0, name="Sigma_0", ndim=2)

        d = self.A.shape[0]
        if d == 0 or self.A.shape != (d, d):
            raise ValueError(f"A must be a non-empty square matrix, got {self.A.shape}")
        if self.B.shape[0] != d:
            raise ValueError(f"B must have {d} rows, got {self.B.shape}")
        m = self.B.shape[1]
        p = self.C.shape[0]
        if p == 0 or self.C.shape != (p, d):
            raise ValueError(f"C must have shape (p, {d}) with p > 0, got {self.C.shape}")

        expected_shapes = {
            "D": (p, m),
            "Q": (d, d),
            "R": (p, p),
            "mu_0": (d,),
            "Sigma_0": (d, d),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = getattr(self, name).shape
            if actual_shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {actual_shape}")

        _require_positive_semidefinite(self.Q, name="Q")
        _require_positive_definite(self.R, name="R")
        _require_positive_definite(self.Sigma_0, name="Sigma_0")

    @property
    def state_dim(self) -> int:
        """Return the latent-state dimension ``d``."""
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        """Return the observation dimension ``p``."""
        return int(self.C.shape[0])

    @property
    def control_dim(self) -> int:
        """Return the control-input dimension ``m``."""
        return int(self.B.shape[1])

    @classmethod
    def random(
        cls,
        state_dim: int,
        obs_dim: int,
        control_dim: int = 0,
        seed: int = 42,
    ) -> LinearGaussianSSM:
        """Construct a stable random model for initialisation and examples.

        Parameters
        ----------
        state_dim : int
            Positive latent-state dimension.
        obs_dim : int
            Positive observation dimension.
        control_dim : int, default=0
            Non-negative control dimension.
        seed : int, default=42
            NumPy random-generator seed.

        Returns
        -------
        LinearGaussianSSM
            Model whose state-transition spectral radius is ``0.95``.

        Raises
        ------
        ValueError
            If a dimension is not an integer in its documented domain.

        """
        d = _require_dimension(state_dim, name="state_dim", allow_zero=False)
        p = _require_dimension(obs_dim, name="obs_dim", allow_zero=False)
        m = _require_dimension(control_dim, name="control_dim", allow_zero=True)
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((d, d)) * 0.5
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(raw))))
        minimum_scale = float(np.finfo(np.float64).tiny)
        A = np.asarray(raw * (0.95 / max(spectral_radius, minimum_scale)))
        B = rng.standard_normal((d, m)) if m > 0 else np.zeros((d, 0))
        C = rng.standard_normal((p, d)) * 0.5
        D = rng.standard_normal((p, m)) if m > 0 else np.zeros((p, 0))
        return cls(
            A=A,
            B=B,
            C=C,
            D=D,
            Q=np.eye(d) * 0.1,
            R=np.eye(p) * 0.1,
            mu_0=np.zeros(d),
            Sigma_0=np.eye(d),
        )


@dataclass
class FilterResult:
    """Forward-filter posterior and one-step prediction moments.

    Parameters
    ----------
    means : numpy.ndarray, shape (T, d)
        Filtered state means.
    covariances : numpy.ndarray, shape (T, d, d)
        Filtered state covariances.
    pred_means : numpy.ndarray, shape (T, d)
        One-step predicted state means before observing each sample.
    pred_covariances : numpy.ndarray, shape (T, d, d)
        One-step predicted state covariances.
    log_likelihood : float
        Sequence log-likelihood under the model.

    """

    means: FloatArray
    covariances: FloatArray
    pred_means: FloatArray
    pred_covariances: FloatArray
    log_likelihood: float

    def __post_init__(self) -> None:
        """Validate result shapes, finiteness, symmetry, and covariance signs."""
        self.means = _as_float_array(self.means, name="means", ndim=2)
        self.covariances = _as_float_array(self.covariances, name="covariances", ndim=3)
        self.pred_means = _as_float_array(self.pred_means, name="pred_means", ndim=2)
        self.pred_covariances = _as_float_array(
            self.pred_covariances, name="pred_covariances", ndim=3
        )
        time_steps, state_dim = self.means.shape
        if time_steps == 0 or state_dim == 0:
            raise ValueError("means must have non-zero time and state dimensions")
        expected_vector_shape = (time_steps, state_dim)
        expected_matrix_shape = (time_steps, state_dim, state_dim)
        if self.pred_means.shape != expected_vector_shape:
            raise ValueError(
                f"pred_means must have shape {expected_vector_shape}, got {self.pred_means.shape}"
            )
        for name in ("covariances", "pred_covariances"):
            covariance_stack = getattr(self, name)
            if covariance_stack.shape != expected_matrix_shape:
                raise ValueError(
                    f"{name} must have shape {expected_matrix_shape}, got {covariance_stack.shape}"
                )
            _require_positive_semidefinite_stack(covariance_stack, name=name)
        self.log_likelihood = float(self.log_likelihood)
        if not isfinite(self.log_likelihood):
            raise ValueError("log_likelihood must be finite")


@dataclass
class SmoothResult:
    """Rauch-Tung-Striebel smoothed state moments.

    Parameters
    ----------
    means : numpy.ndarray, shape (T, d)
        Smoothed state means.
    covariances : numpy.ndarray, shape (T, d, d)
        Smoothed state covariances.
    cross_covariances : numpy.ndarray, shape (T - 1, d, d)
        Lag-one covariances ``Cov[x_t, x_{t+1} | y_{0:T-1}]``.

    """

    means: FloatArray
    covariances: FloatArray
    cross_covariances: FloatArray

    def __post_init__(self) -> None:
        """Validate smoothed moment shapes, finiteness, and covariance signs."""
        self.means = _as_float_array(self.means, name="means", ndim=2)
        self.covariances = _as_float_array(self.covariances, name="covariances", ndim=3)
        self.cross_covariances = _as_float_array(
            self.cross_covariances, name="cross_covariances", ndim=3
        )
        time_steps, state_dim = self.means.shape
        if time_steps == 0 or state_dim == 0:
            raise ValueError("means must have non-zero time and state dimensions")
        covariance_shape = (time_steps, state_dim, state_dim)
        cross_shape = (time_steps - 1, state_dim, state_dim)
        if self.covariances.shape != covariance_shape:
            raise ValueError(
                f"covariances must have shape {covariance_shape}, got {self.covariances.shape}"
            )
        if self.cross_covariances.shape != cross_shape:
            raise ValueError(
                "cross_covariances must have shape "
                f"{cross_shape}, got {self.cross_covariances.shape}"
            )
        _require_positive_semidefinite_stack(
            self.covariances,
            name="covariances",
        )
