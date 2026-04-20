# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for reservoir/auto_reservoir

module AutoReservoirAccel

using Statistics, LinearAlgebra

mutable struct AutoCriticalReservoirState
    firing_fraction::Float64
    criticality_error::Float64
    kernel_quality::Float64
    spectral_radius::Float64
    n_inputs::Float64
    n_neurons::Float64
    n_outputs::Float64
    threshold::Float64
    leak::Float64
    connectivity::Float64
    w_critical::Float64
    W_res::Float64
    W_in::Float64
    W_out::Float64
    _v::Float64
end

function AutoCriticalReservoirState()
    AutoCriticalReservoirState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::AutoCriticalReservoirState)
    return (
        f"Reservoir: firing={s.firing_fraction:.3f}, "
        f"criticality_err={s.criticality_error:.4f}, "
        f"kernel_q={s.kernel_quality:.3f}, "
        f"spectral_r={s.spectral_radius:.3f}"
    )
end

function spectral_radius(s::AutoCriticalReservoirState)
    eigvals = abs(np.linalg.eigvals(s.W_res))
    return float(eigvals.max()) if length(eigvals) > 0 else 0.0
end

function reset(s::AutoCriticalReservoirState)
    s._v = zeros(s.n_neurons)
    s._spikes = zeros(s.n_neurons)
end

function step(s::AutoCriticalReservoirState, x)
    current = s.W_in @ x + s.W_res @ s._spikes
    s._v = (1 - s.leak) * s._v + s.leak * current
    s._spikes = (s._v >= s.threshold).astype(np.float64)  # type: ignore[assignment]
    s._v -= s._spikes * s.threshold
    return s._spikes.copy()
end

function run(s::AutoCriticalReservoirState, inputs)
    s.reset()
    T = inputs.shape[0]
    states = zeros((T, s.n_neurons))
    for t in 1:T
        states[t] = s.step(inputs[t])
    return states
end

function fit_readout(s::AutoCriticalReservoirState, states, targets, ridge)
    # W_out = targets^T @ states @ (states^T @ states + ridge*I)^{-1}
    S = states
    reg = ridge * np.eye(s.n_neurons)
    s.W_out = np.linalg.solve(S.T @ S + reg, S.T @ targets).T
end

function predict(s::AutoCriticalReservoirState, states)
    return states @ s.W_out.T
end

function train_and_predict(s::AutoCriticalReservoirState)
    self, train_inputs: np.ndarray, train_targets: np.ndarray, test_inputs: np.ndarray
    ) -> np.ndarray
    train_states = s.run(train_inputs)
    s.fit_readout(train_states, train_targets)
    test_states = s.run(test_inputs)
    return s.predict(test_states)
end

function metrics(s::AutoCriticalReservoirState, inputs)
    states = s.run(inputs)
    firing_fraction = float(states.mean())
    criticality_error = abs(firing_fraction - 0.5)
    # Kernel quality: rank of state matrix normalized by timesteps
    rank = np.linalg.matrix_rank(states)
    kernel_quality = rank / max(states.shape[0], 1)
    return ReservoirMetrics(
        firing_fraction=firing_fraction,
        criticality_error=criticality_error,
        kernel_quality=kernel_quality,
        spectral_radius=s.spectral_radius,
    )
end

end # module AutoReservoirAccel
