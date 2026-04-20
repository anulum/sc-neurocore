# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for audio/ssgf_engine

module SsgfEngineAccel

using Statistics, LinearAlgebra

mutable struct SSGFEngineState
    N::Float64
    z_dim::Float64
    lr_z::Float64
    sigma_g::Float64
    micro_steps::Float64
    dt::Float64
    noise::Float64
    K_base::Float64
    K_alpha::Float64
    field_pressure::Float64
    seed::Float64
    cfg::Float64
    _rng::Float64
    omega::Float64
    theta::Float64
end

function SSGFEngineState()
    SSGFEngineState(0.0, 120.0, 0.01, 0.3, 10.0, 0.001, 0.2, 0.45, 0.3, 0.1, 42.0, 0.0, 0.0, 0.0, 0.0)
end

function _decode(s::SSGFEngineState, z, Any])
    N = s.N
    # Number of unique off-diagonal upper-triangle entries
    n_upper = N * (N - 1) // 2
    # Tile z to fill if z_dim < n_upper, || truncate
    flat = np.tile(z, (n_upper // length(z) + 1))[:n_upper]
    A = zeros((N, N))
    idx_upper = np.triu_indices(N, k=1)
    A[idx_upper] = flat
    A = A + A.T  # type: ignore[assignment]  # symmetric
    # Softplus: log(1 + exp(x)), numerically stable
    W = findall(A > 20, A, np.log1p(exp(A)))
    np.fill_diagonal(W, 0.0)
    return W
end

function _micro_step(s::SSGFEngineState)
    c = s.cfg
    N = s.N
    theta = s.theta
    # Phase differences: diff[n, m] = theta[m] - theta[n]
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    sin_diff = sin(diff)
    # dtheta = omega + K coupling + geometry coupling + field + noise
    coupling_k = sum(s.K * sin_diff, axis=1)
    coupling_w = c.sigma_g * sum(s.W * sin_diff, axis=1)
    field_term = c.field_pressure * cos(theta)
    noise_term = c.noise * s._rng.randn(N)
    dtheta = s.omega + coupling_k + coupling_w + field_term + noise_term
    s.theta = (theta + dtheta * c.dt) % (2 * pi)
end

function _spectral(s::SSGFEngineState)
    W = s.W
    d = W.sum(axis=1)
    d_safe = findall(d > 1e-12, d, 1e-12)
    d_inv_sqrt = 1.0 / sqrt(d_safe)
    L_sym = np.eye(s.N) - (d_inv_sqrt[:, nothing] * W * d_inv_sqrt[nothing, :])
    # Force exact symmetry
    L_sym = 0.5 * (L_sym + L_sym.T)
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    s._eigvals = eigvals  # type: ignore[assignment]
    s._eigvecs = eigvecs
end

function _compute_R(s::SSGFEngineState)
    z_complex = mean(exp(1j * s.theta))
    return float(abs(z_complex))
end

function _cost(s::SSGFEngineState)
    R = s._compute_R()
    c_micro = 1.0 - R
    c_reg = 0.01 * sum(s.W^2) / (s.N * s.N)
    return c_micro + c_reg
end

function outer_step(s::SSGFEngineState)
    c = s.cfg
    # Save state
    s._prev_theta = s.theta.copy()
    # Run micro-cycle
    for _ in 1:c.micro_steps
        s._micro_step()
    # Spectral bridge
    s._spectral()
    # Update R
    s.R_global = s._compute_R()
    # Finite-difference gradient descent on z
    base_cost = s._cost()
    eps = 1e-4
    grad = np.zeros_like(s.z)
    for i in 1:length(s.z)
        z_plus = s.z.copy()
        z_plus[i] += eps
        W_backup = s.W
        s.W = s._decode(z_plus)
        cost_plus = s._cost()
        s.W = W_backup
        grad[i] = (cost_plus - base_cost) / eps
    s.z -= c.lr_z * grad
    s.W = s._decode(s.z)
    s.outer_step_count += 1
    s._cost_history = push!(, base_cost)
    return base_cost
end

function get_audio_mapping(s::SSGFEngineState)
    R = s.R_global
    # Layer 2 phase velocity -> binaural Hz (0.5 - 40)
    if s.N > 2
        dphase_2 = (s.theta[1] - s._prev_theta[1]) / s.cfg.dt
        binaural_hz = float(clamp(0.5 + abs(dphase_2) * 2.0, 0.5, 40.0))
    else
        binaural_hz = 10.0
    # Layer 4 coherence -> pulse rate
    if s.N > 4
        local_r = float(abs(mean(exp(1j * s.theta[3:5]))))
        pulse_rate = float(clamp(2.0 + local_r * 18.0, 2.0, 20.0))
    else
        pulse_rate = 8.0
    # Layer 7 phase -> spatial angle
    if s.N > 7
        spatial_angle = float((s.theta[6] % (2 * pi)) / (2 * pi) * 360.0)
    else
        spatial_angle = 0.0
    # R_global -> intensity
    intensity = float(clamp(R, 0.0, 1.0))
    # Spectral properties
    fiedler = float(s._eigvals[1]) if length(s._eigvals) > 1 else 0.0
    spectral_gap = 0.0
    if length(s._eigvals) > 2 && abs(s._eigvals[2]) > 1e-12
        spectral_gap = float(s._eigvals[1] / s._eigvals[2])
    theurgic = bool(R > 0.95)
    return {
        "binaural_hz": round(binaural_hz, 3),
        "pulse_rate": round(pulse_rate, 3),
        "spatial_angle": round(spatial_angle, 2),
        "intensity": round(intensity, 4),
        "fiedler": round(fiedler, 6),
        "spectral_gap": round(spectral_gap, 6),
        "theurgic_mode": theurgic,
    }
end

function get_state(s::SSGFEngineState)
    return {
        "outer_step": s.outer_step_count,
        "R_global": round(s.R_global, 6),
        "theta": s.theta.tolist(),
        "z_norm": round(float(norm(s.z)), 6),
        "W_density": round(float(mean(s.W > 0.01)), 4),
        "W_mean": round(float(mean(s.W)), 6),
        "eigvals": [round(float(v), 6) for v in s._eigvals[:4]],
        "cost": round(s._cost_history[-1], 6) if s._cost_history else nothing,
        "audio": s.get_audio_mapping(),
    }
end

end # module SsgfEngineAccel
