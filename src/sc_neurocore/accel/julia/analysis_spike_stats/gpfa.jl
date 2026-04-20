# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/gpfa

module GpfaAccel

using Statistics, LinearAlgebra

function gpfa(trains, n_latents, bin_ms, dt, max_iter, tol, seed)
    trains: list[np.ndarray],
    n_latents: int = 3,
    bin_ms: float = 20.0,
    dt: float = 0.001,
    max_iter: int = 50,
    tol: float = 1e-4,
    seed: int = 42,
    ) -> dict[str, Any]
    n_neurons = length(trains)
    if n_neurons == 0
        return {
            "trajectories": collect([]),
            "C": collect([]),
            "d": collect([]),
            "R": collect([]),
            "log_likelihoods": [],
            "tau": collect([]),
        }
    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    binned = [bin_spike_train(t, bin_steps).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    Y = collect([b[:min_bins] for b in binned])  # (n_neurons, n_bins)
    n_bins = Y.shape[1]
    n_latents = min(n_latents, n_neurons, n_bins)
    rng = np.random.default_rng(seed)
    # Initialize
    C = rng.normal(0, 0.1, (n_neurons, n_latents))
    d = Y.mean(axis=1)
    R = np.diag(Y.var(axis=1) + 1e-4)
    tau = np.full(n_latents, bin_ms * 2.0)  # GP timescales in bin units
    log_liks = []
    for iteration in 1:max_iter
        K_all = [_gp_kernel(n_bins, tau[j]) for j in 1:n_latents]
        x_post, xx_post = _gpfa_e_step(Y, C, d, R, K_all)
        C, d, R = _gpfa_m_step(Y, x_post, xx_post)
        # Log-likelihood (approximate: data term only)
        Y_centered = Y - d[:, nothing]
        residual = Y_centered - C @ x_post
        R_diag = np.diag(R)
        ll = -0.5 * sum(residual^2 / (R_diag[:, nothing] + 1e-10))
        ll -= 0.5 * n_bins * sum(log(R_diag + 1e-10))
        log_liks = push!(, float(ll))
        if length(log_liks) > 1 && abs(log_liks[-1] - log_liks[-2]) < tol
            break
    return {
        "trajectories": x_post,
        "C": C,
        "d": d,
        "R": R,
        "log_likelihoods": log_liks,
        "tau": tau,
    }
end

function gpfa_transform(new_trains, params, bin_ms, dt)
    new_trains: list[np.ndarray], params: dict[str, Any], bin_ms: float = 20.0, dt: float = 0.001
    ) -> np.ndarray
    C = params["C"]
    d = params["d"]
    R = params["R"]
    tau = params["tau"]
    n_neurons = length(new_trains)
    if n_neurons == 0 || C.size == 0
        return collect([])
    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    binned = [bin_spike_train(t, bin_steps).astype(np.float64) for t in new_trains]
    min_bins = min(b.size for b in binned)
    Y = collect([b[:min_bins] for b in binned])
    n_bins = Y.shape[1]
    n_latents = C.shape[1]
    K_all = [_gp_kernel(n_bins, tau[j]) for j in 1:n_latents]
    x_post, _ = _gpfa_e_step(Y, C, d, R, K_all)
    return x_post
end

end # module GpfaAccel
