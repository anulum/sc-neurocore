# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis/phi_estimation
#
# Geometric integrated information Phi* (Barrett & Seth 2011) under the Gaussian
# assumption. Mutual information is a difference of covariance log-determinants
# taken from Cholesky factors (LAPACK via Julia's LinearAlgebra), matching the
# NumPy reference and the Rust, Go and Mojo backends within float64 round-off.

module PhiAccel

using Statistics, LinearAlgebra

# log|A| for a symmetric positive-definite matrix via Cholesky.
logdet_spd(cov) = logdet(cholesky(Symmetric(cov)))

# Gaussian mutual information MI(X;Y) = 0.5 (log|Cov_X| + log|Cov_Y| - log|Cov_XY|).
# Covariances use the unbiased (ddof=1) estimator with a 1e-10 diagonal jitter.
function gaussian_mi(x, y)
    eps = 1e-10
    cov_x = cov(x; dims=2) + eps * I
    cov_y = cov(y; dims=2) + eps * I
    cov_xy = cov(vcat(x, y); dims=2) + eps * I
    mi = 0.5 * (logdet_spd(cov_x) + logdet_spd(cov_y) - logdet_spd(cov_xy))
    return max(0.0, mi)
end

"""
    phi_star(data, tau)

Geometric Phi* for a `(n_channels, n_timesteps)` matrix. Returns a non-negative
`Float64`; `0` means fully reducible. Bipartitions are the contiguous splits
`{1..k} | {k+1..n}`.
"""
function phi_star(data_in, tau_in)
    data = Matrix{Float64}(data_in)
    tau = Int(tau_in)
    n, t = size(data)
    if n < 2 || 2 * tau >= t
        return 0.0
    end

    past = data[:, 1:(t - tau)]
    future = data[:, (1 + tau):t]

    mi_whole = gaussian_mi(past, future)
    mi_parts_min = Inf
    for k in 1:(n - 1)
        mi_a = gaussian_mi(past[1:k, :], future[1:k, :])
        mi_b = gaussian_mi(past[(k + 1):n, :], future[(k + 1):n, :])
        mi_parts_min = min(mi_parts_min, mi_a + mi_b)
    end

    return max(0.0, mi_whole - mi_parts_min)
end

end # module PhiAccel
