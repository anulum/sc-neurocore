# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia LGSSM Kalman filter (parity with predictive_model.py)

"""
Julia implementation of the forward Kalman filter for a Linear
Gaussian State-Space Model. Matches the Python KalmanFilter and
the Rust py_lgssm_kalman_filter step-for-step so the three
backends are observationally equivalent.

References (matches the Python module):
  Kalman 1960; Bishop 2006 §13.3.1.

Algorithm per timestep t:
    x_pred = A x_filt + B u
    P_pred = A P_filt A' + Q
    e      = y - C x_pred - D u
    S      = C P_pred C' + R
    K      = P_pred C' / S
    x_filt = x_pred + K e
    P_filt = (I - K C) P_pred (I - K C)' + K R K'   (Joseph form)
    log-lik += -0.5 (p log 2π + log |S| + e' S^{-1} e)

Usage (from Python via juliacall):

    from juliacall import Main as jl
    jl.include("src/sc_neurocore/accel/julia/world_model/predictive_model.jl")
    result = jl.PredictiveModelAccel.kalman_filter(
        observations, controls, A, B, C, D, Q, R, mu_0, Sigma_0,
    )
"""

module PredictiveModelAccel

using LinearAlgebra

export kalman_filter

"""
    kalman_filter(observations, controls, A, B, C, D, Q, R, mu_0, Sigma_0)
        -> NamedTuple

Returns a NamedTuple with fields:
- means        :: Matrix{Float64}  (T × d)  filtered means
- covariances  :: Array{Float64,3} (T × d × d) filtered covariances
- pred_means   :: Matrix{Float64}  (T × d) predicted means
- pred_covs    :: Array{Float64,3} (T × d × d) predicted covariances
- log_lik      :: Float64         total log-likelihood
- backend      :: String          always "julia"

All matrix arguments must be `Matrix{Float64}` with the standard
shapes from the Python LGSSM specification:
- observations: (T, p)
- controls:     (T, m) — pass an `(T, 0)` matrix when no controls
- A: (d, d), B: (d, m), C: (p, d), D: (p, m), Q: (d, d), R: (p, p)
- mu_0: Vector{Float64} of length d
- Sigma_0: (d, d)
"""
function kalman_filter(
    observations::AbstractMatrix{<:Real},
    controls::AbstractMatrix{<:Real},
    A::AbstractMatrix{<:Real},
    B::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    D::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
    R::AbstractMatrix{<:Real},
    mu_0::AbstractVector{<:Real},
    Sigma_0::AbstractMatrix{<:Real},
)
    T = size(observations, 1)
    p = size(observations, 2)
    d = size(A, 1)
    m = size(B, 2)
    has_control = m > 0

    means        = zeros(Float64, T, d)
    covariances  = zeros(Float64, T, d, d)
    pred_means   = zeros(Float64, T, d)
    pred_covs    = zeros(Float64, T, d, d)

    x_pred = Vector{Float64}(mu_0)
    P_pred = Matrix{Float64}(Sigma_0)

    log_lik = 0.0
    two_pi_log = log(2 * π)
    Id = Matrix{Float64}(I, d, d)

    for t in 1:T
        # Record predicted state for this step
        pred_means[t, :] .= x_pred
        pred_covs[t, :, :] .= P_pred

        y_t = view(observations, t, :)

        # Innovation: e = y - C x_pred - D u
        y_hat = C * x_pred
        if has_control
            u_t = view(controls, t, :)
            y_hat .+= D * u_t
        end
        innov = collect(y_t) .- y_hat

        # Innovation covariance: S = C P_pred C' + R
        S_mat = C * P_pred * transpose(C) .+ R

        # Symmetrise (defensive — round-off can break Cholesky on tight PSD)
        S_sym = 0.5 .* (S_mat .+ transpose(S_mat))
        S_chol = cholesky(Symmetric(S_sym))
        s_inv_innov = S_chol \ innov

        # Log-determinant via Cholesky: 2 sum log diag(L)
        logdet_S = 2 * sum(log.(diag(S_chol.L)))
        quad_form = dot(innov, s_inv_innov)
        log_lik += -0.5 * (p * two_pi_log + logdet_S + quad_form)

        # Kalman gain: K = P_pred C' S^{-1}
        K_gain = (P_pred * transpose(C)) / S_chol

        # Filtered state: x_filt = x_pred + K e
        x_filt = x_pred .+ K_gain * innov

        # Joseph form: P_filt = (I - K C) P_pred (I - K C)' + K R K'
        I_minus_KC = Id .- K_gain * C
        P_filt = I_minus_KC * P_pred * transpose(I_minus_KC) .+ K_gain * R * transpose(K_gain)

        means[t, :] .= x_filt
        covariances[t, :, :] .= P_filt

        # Predict next
        x_next = A * x_filt
        if has_control
            u_t = view(controls, t, :)
            x_next .+= B * u_t
        end
        P_next = A * P_filt * transpose(A) .+ Q

        x_pred = x_next
        P_pred = P_next
    end

    return (
        means = means,
        covariances = covariances,
        pred_means = pred_means,
        pred_covs = pred_covs,
        log_lik = log_lik,
        backend = "julia",
    )
end

end  # module PredictiveModelAccel
