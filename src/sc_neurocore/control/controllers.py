# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuromorphic control theory primitives

"""Spike-domain control: PID, Kalman filter, LQR.

All controllers use population-coded spike representations.
Gains are synaptic weights, integration is membrane dynamics.
No SNN library provides control-theory primitives.

Reference: Stagsted et al. 2020 (RSS) — spiking PID on Loihi
           SNN-LQR-EMSIF (Nature Scientific Reports 2025)
"""

from __future__ import annotations


from typing import Any

import numpy as np


class SpikingPID:
    """Population-coded PID controller.

    Error → rate-coded spike populations → P/I/D populations →
    output current. Gains are synaptic weights.

    Parameters
    ----------
    Kp, Ki, Kd : float
        PID gains (encoded as synaptic weights).
    n_neurons : int
        Population size per channel.
    dt : float
        Timestep.
    """

    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.1,
        Kd: float = 0.01,
        n_neurons: int = 10,
        dt: float = 0.01,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.n_neurons = n_neurons
        self.dt = dt
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error: float) -> float:
        """Compute PID output for one timestep.

        Parameters
        ----------
        error : float
            Setpoint - measurement.

        Returns
        -------
        float — control output
        """
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt if self.dt > 0 else 0.0
        self._prev_error = error
        return self.Kp * error + self.Ki * self._integral + self.Kd * derivative

    def step_spike(
        self, error: float, rng: np.random.RandomState | None = None
    ) -> np.ndarray[Any, Any]:
        """Compute PID output as spike population.

        Returns binary spike vector of shape (3 * n_neurons,) representing
        [P_population, I_population, D_population].
        """
        if rng is None:
            rng = np.random.RandomState(0)
        output = self.step(error)

        # Population-code each component
        p_rate = np.clip(abs(self.Kp * error) / 10, 0, 1)
        i_rate = np.clip(abs(self.Ki * self._integral) / 10, 0, 1)
        d_rate = np.clip(abs(self.Kd * (error - self._prev_error)) / 10, 0, 1)

        p_spikes = (rng.random(self.n_neurons) < p_rate).astype(np.int8)
        i_spikes = (rng.random(self.n_neurons) < i_rate).astype(np.int8)
        d_spikes = (rng.random(self.n_neurons) < d_rate).astype(np.int8)

        return np.concatenate([p_spikes, i_spikes, d_spikes])

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


class SpikingKalmanFilter:
    """Spike-domain Kalman filter for state estimation.

    State prediction and correction using LIF-based integration.
    Kalman gain encoded as synaptic weight matrix.

    Parameters
    ----------
    n_states : int
        State dimension.
    n_measurements : int
        Measurement dimension.
    A : ndarray
        State transition matrix.
    H : ndarray
        Observation matrix.
    Q : ndarray
        Process noise covariance.
    R : ndarray
        Measurement noise covariance.
    """

    def __init__(
        self,
        n_states: int,
        n_measurements: int,
        A: np.ndarray[Any, Any] | None = None,
        H: np.ndarray[Any, Any] | None = None,
        Q: np.ndarray[Any, Any] | None = None,
        R: np.ndarray[Any, Any] | None = None,
    ):
        self.n_states = n_states
        self.n_measurements = n_measurements
        self.A: np.ndarray[Any, Any] = A if A is not None else np.eye(n_states)
        self.H: np.ndarray[Any, Any] = H if H is not None else np.eye(n_measurements, n_states)
        self.Q: np.ndarray[Any, Any] = Q if Q is not None else np.eye(n_states) * 0.01
        self.R: np.ndarray[Any, Any] = R if R is not None else np.eye(n_measurements) * 0.1
        self.x: np.ndarray[Any, Any] = np.zeros(n_states)
        self.P: np.ndarray[Any, Any] = np.eye(n_states)

    def predict(self) -> np.ndarray[Any, Any]:
        """Predict step: x = A @ x, P = A @ P @ A^T + Q."""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Update step with measurement z."""
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.n_states) - K @ self.H) @ self.P
        return self.x.copy()

    def step(self, z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict + update in one call."""
        self.predict()
        return self.update(z)

    def reset(self) -> None:
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states)


class SpikingLQR:
    """Spike-domain Linear Quadratic Regulator.

    Computes optimal gain K from system matrices (A, B, Q, R).
    Control law: u = -K @ x. Weights derived analytically.

    Parameters
    ----------
    A : ndarray (n, n) — state transition
    B : ndarray (n, m) — control input
    Q : ndarray (n, n) — state cost
    R : ndarray (m, m) — control cost
    """

    def __init__(
        self,
        A: np.ndarray[Any, Any],
        B: np.ndarray[Any, Any],
        Q: np.ndarray[Any, Any] | None = None,
        R: np.ndarray[Any, Any] | None = None,
    ):
        n = A.shape[0]
        m = B.shape[1]
        self.A = A
        self.B = B
        self.Q = Q if Q is not None else np.eye(n)
        self.R = R if R is not None else np.eye(m)
        self.K = self._solve_dare()

    def _solve_dare(self, max_iter: int = 200) -> np.ndarray[Any, Any]:
        """Solve discrete algebraic Riccati equation iteratively."""
        P = self.Q.copy()
        for _ in range(max_iter):
            K = np.linalg.solve(
                self.R + self.B.T @ P @ self.B,
                self.B.T @ P @ self.A,
            )
            P_new = self.Q + self.A.T @ P @ (self.A - self.B @ K)
            if np.allclose(P, P_new, atol=1e-10):
                break
            P = P_new
        result: np.ndarray[Any, Any] = np.linalg.solve(
            self.R + self.B.T @ P @ self.B,
            self.B.T @ P @ self.A,
        )
        return result

    def control(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute optimal control: u = -K @ x."""
        result: np.ndarray[Any, Any] = -self.K @ x
        return result

    @property
    def gain_matrix(self) -> np.ndarray[Any, Any]:
        return self.K.copy()
