# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis/gpfa (Gaussian Process Factor Analysis)
#
# Runs the GPFA EM loop from a caller-supplied deterministic initialisation,
# matching the NumPy reference in src/sc_neurocore/analysis/spike_stats/gpfa.py
# (block-structured E-step, exact marginal Gaussian log-likelihood). Results agree
# with the Python, Rust, Go and Mojo backends within float64 round-off.

module GpfaAccel

using LinearAlgebra

function gp_kernel(n_bins::Int, tau::Float64)
    t = collect(0.0:(n_bins - 1))
    diff = t .- t'
    return exp.(-0.5 .* diff .^ 2 ./ (tau^2 + 1e-12))
end

function e_step(Y, C, d, r_diag, k_all, n_neurons::Int, n_bins::Int, n_latents::Int)
    kt = n_latents * n_bins
    r_inv = 1.0 ./ (r_diag .+ 1e-10)
    ct_rinv = C' .* r_inv'                         # (n_latents, n_neurons)
    ct_rinv_c = ct_rinv * C                        # (n_latents, n_latents)

    prec = zeros(Float64, kt, kt)
    eye_bins = Matrix{Float64}(I, n_bins, n_bins)
    for j in 1:n_latents
        slj = ((j - 1) * n_bins + 1):(j * n_bins)
        k_inv = (k_all[j] + 1e-6 .* eye_bins) \ eye_bins
        prec[slj, slj] = k_inv + ct_rinv_c[j, j] .* eye_bins
        for k in 1:n_latents
            if k != j
                slk = ((k - 1) * n_bins + 1):(k * n_bins)
                prec[slj, slk] = ct_rinv_c[j, k] .* eye_bins
            end
        end
    end

    y_centered = Y .- d
    rhs = zeros(Float64, kt)
    for t in 1:n_bins
        v = ct_rinv * y_centered[:, t]
        for j in 1:n_latents
            rhs[(j - 1) * n_bins + t] = v[j]
        end
    end

    prec_reg = prec + 1e-8 .* Matrix{Float64}(I, kt, kt)
    x_vec = prec_reg \ rhs
    sigma_post = prec_reg \ Matrix{Float64}(I, kt, kt)

    x_post = zeros(Float64, n_latents, n_bins)
    for j in 1:n_latents, t in 1:n_bins
        x_post[j, t] = x_vec[(j - 1) * n_bins + t]
    end

    xx_post = zeros(Float64, n_latents, n_latents)
    for t in 1:n_bins
        for j in 1:n_latents
            for k in 1:n_latents
                xx_post[j, k] +=
                    x_post[j, t] * x_post[k, t] +
                    sigma_post[(j - 1) * n_bins + t, (k - 1) * n_bins + t]
            end
        end
    end
    return x_post, xx_post
end

function m_step(Y, x_post, xx_post, n_bins::Int, n_latents::Int)
    d_new = vec(sum(Y; dims=2) ./ n_bins)
    y_centered = Y .- d_new
    yx = y_centered * x_post'
    c_new = ((xx_post' + 1e-8 .* Matrix{Float64}(I, n_latents, n_latents)) \ yx')'
    yyt = (y_centered * y_centered') ./ n_bins
    cxyt = (c_new * x_post * y_centered') ./ n_bins
    r_new = max.(diag(yyt - cxyt), 1e-6)
    return Matrix{Float64}(c_new), d_new, r_new
end

function log_likelihood(Y, C, d, r_diag, k_all, n_neurons::Int, n_bins::Int, n_latents::Int)
    n_obs = n_neurons * n_bins
    n_state = n_latents * n_bins
    a = zeros(Float64, n_obs, n_state)
    for t in 1:n_bins
        row_start = (t - 1) * n_neurons
        for j in 1:n_latents
            col = (j - 1) * n_bins + t
            a[(row_start + 1):(row_start + n_neurons), col] = C[:, j]
        end
    end
    k_big = zeros(Float64, n_state, n_state)
    for j in 1:n_latents
        sl = ((j - 1) * n_bins + 1):(j * n_bins)
        k_big[sl, sl] = k_all[j]
    end
    cov = a * k_big * a'
    for t in 1:n_bins
        for i in 1:n_neurons
            idx = (t - 1) * n_neurons + i
            cov[idx, idx] += r_diag[i]
        end
    end
    cov += 1e-8 .* Matrix{Float64}(I, n_obs, n_obs)
    yc = zeros(Float64, n_obs)
    for t in 1:n_bins
        for i in 1:n_neurons
            yc[(t - 1) * n_neurons + i] = Y[i, t] - d[i]
        end
    end
    quad = dot(yc, cov \ yc)
    return -0.5 * (quad + logdet(cov) + n_obs * log(2.0 * pi))
end

"""
    gpfa_em(Y, C0, d0, R0_diag, tau, max_iter, tol)

Run the GPFA EM loop from a fixed initialisation. Returns a NamedTuple with
`trajectories` (n_latents × n_bins), `C` (n_neurons × n_latents), `d`
(n_neurons), `R_diag` (n_neurons), `log_liks` (vector) and `backend` ("julia").
"""
function gpfa_em(Y_in, C0_in, d0_in, R0_diag_in, tau_in, max_iter, tol)
    Y = Matrix{Float64}(Y_in)
    C = Matrix{Float64}(C0_in)
    d = Vector{Float64}(d0_in)
    r = Vector{Float64}(R0_diag_in)
    tau = Vector{Float64}(tau_in)
    max_iter = Int(max_iter)
    tol = Float64(tol)

    n_neurons = size(Y, 1)
    n_bins = size(Y, 2)
    n_latents = size(C, 2)
    k_all = [gp_kernel(n_bins, tau[j]) for j in 1:n_latents]

    log_liks = Float64[]
    x_post = zeros(Float64, n_latents, n_bins)
    for _ in 1:max_iter
        x_post, xx_post = e_step(Y, C, d, r, k_all, n_neurons, n_bins, n_latents)
        C, d, r = m_step(Y, x_post, xx_post, n_bins, n_latents)
        ll = log_likelihood(Y, C, d, r, k_all, n_neurons, n_bins, n_latents)
        push!(log_liks, ll)
        if length(log_liks) > 1 && abs(log_liks[end] - log_liks[end - 1]) < tol
            break
        end
    end

    return (
        trajectories = x_post,
        C = C,
        d = d,
        R_diag = r,
        log_liks = log_liks,
        backend = "julia",
    )
end

end # module GpfaAccel
