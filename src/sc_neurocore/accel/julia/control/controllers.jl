# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for control/controllers

module ControllersAccel

using Statistics, LinearAlgebra

mutable struct SpikingLQRState
    Kp::Float64
    Ki::Float64
    Kd::Float64
    n_neurons::Float64
    dt::Float64
    _integral::Float64
    _prev_error::Float64
    n_states::Float64
    n_measurements::Float64
    A::Float64
    B::Float64
    Q::Float64
    R::Float64
    K::Float64
end

function SpikingLQRState()
    SpikingLQRState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function step(s::SpikingLQRState, error)
    s._integral += error * s.dt
    derivative = (error - s._prev_error) / s.dt if s.dt > 0 else 0.0
    s._prev_error = error
    return s.Kp * error + s.Ki * s._integral + s.Kd * derivative
end

function step_spike(s::SpikingLQRState)
    self, error: float, rng: np.random.RandomState | nothing = nothing
    ) -> np.ndarray[Any, Any]
    if rng is nothing
        rng = np.random.RandomState(0)
    output = s.step(error)
    # Population-code each component
    p_rate = clamp(abs(s.Kp * error) / 10, 0, 1)
    i_rate = clamp(abs(s.Ki * s._integral) / 10, 0, 1)
    d_rate = clamp(abs(s.Kd * (error - s._prev_error)) / 10, 0, 1)
    p_spikes = (rng.random(s.n_neurons) < p_rate).astype(np.int8)
    i_spikes = (rng.random(s.n_neurons) < i_rate).astype(np.int8)
    d_spikes = (rng.random(s.n_neurons) < d_rate).astype(np.int8)
    return vcat([p_spikes, i_spikes, d_spikes])
end

function reset(s::SpikingLQRState)
    s._integral = 0.0
    s._prev_error = 0.0
end

function predict(s::SpikingLQRState)
    s.x = s.A @ s.x
    s.P = s.A @ s.P @ s.A.T + s.Q
    return s.x.copy()
end

function update(s::SpikingLQRState, z, Any])
    S = s.H @ s.P @ s.H.T + s.R
    K = s.P @ s.H.T @ np.linalg.inv(S)
    innovation = z - s.H @ s.x
    s.x = s.x + K @ innovation
    s.P = (np.eye(s.n_states) - K @ s.H) @ s.P
    return s.x.copy()
end

function step(s::SpikingLQRState, z, Any])
    s.predict()
    return s.update(z)
end

function reset(s::SpikingLQRState)
    s.x = zeros(s.n_states)
    s.P = np.eye(s.n_states)
end

function _solve_dare(s::SpikingLQRState, max_iter)
    P = s.Q.copy()
    for _ in 1:max_iter
        K = np.linalg.solve(
            s.R + s.B.T @ P @ s.B,
            s.B.T @ P @ s.A,
        )
        P_new = s.Q + s.A.T @ P @ (s.A - s.B @ K)
        if np.allclose(P, P_new, atol=1e-10)
            break
        P = P_new
    result: np.ndarray[Any, Any] = np.linalg.solve(
        s.R + s.B.T @ P @ s.B,
        s.B.T @ P @ s.A,
    )
    return result
end

function control(s::SpikingLQRState, x, Any])
    result: np.ndarray[Any, Any] = -s.K @ x
    return result
end

function gain_matrix(s::SpikingLQRState)
    return s.K.copy()
end

end # module ControllersAccel
