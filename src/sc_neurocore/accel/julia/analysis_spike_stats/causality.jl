# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/causality

module CausalityAccel

using Statistics, LinearAlgebra

function pairwise_granger_causality(source, target, bin_size, order)
    source: np.ndarray[Any, Any], target: np.ndarray[Any, Any], bin_size: int = 10, order: int = 5
    ) -> float
    cs = bin_spike_train(source, bin_size).astype(np.float64)
    ct = bin_spike_train(target, bin_size).astype(np.float64)
    n = min(cs.size, ct.size)
    if n <= 2 * order
        return 0.0
    cs, ct = cs[:n], ct[:n]
    y = ct[order:]
    n_pts = y.size
    x_r = np.column_stack([ct[order - k - 1 : n - k - 1] for k in 1:order])
    x_f = np.column_stack([x_r] + [cs[order - k - 1 : n - k - 1] for k in 1:order])
        xtx = x.T @ x
        reg = 1e-8 * np.eye(xtx.shape[0])
        beta = np.linalg.solve(xtx + reg, x.T @ yy)
        residuals = yy - x @ beta
        return float(sum(residuals^2))
    sse_r = _sse(x_r, y)
    sse_f = _sse(x_f, y)
    if sse_f <= 0
        return 0.0
    return float(log(max(sse_r, 1e-30) / max(sse_f, 1e-30)))
end

function conditional_granger_causality(source, target, condition, bin_size, order)
    source: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    condition: np.ndarray[Any, Any],
    bin_size: int = 10,
    order: int = 5,
    ) -> float
    cs = bin_spike_train(source, bin_size).astype(np.float64)
    ct = bin_spike_train(target, bin_size).astype(np.float64)
    cc = bin_spike_train(condition, bin_size).astype(np.float64)
    n = min(cs.size, ct.size, cc.size)
    if n <= 2 * order
        return 0.0
    cs, ct, cc = cs[:n], ct[:n], cc[:n]
    y = ct[order:]
    x_cond = np.column_stack(
        [ct[order - k - 1 : n - k - 1] for k in 1:order]
        + [cc[order - k - 1 : n - k - 1] for k in 1:order]
    )
    x_full = np.column_stack([x_cond] + [cs[order - k - 1 : n - k - 1] for k in 1:order])
        reg = 1e-8 * np.eye(x.shape[1])
        beta = np.linalg.solve(x.T @ x + reg, x.T @ yy)
        return float(sum((yy - x @ beta) ^ 2))
    sse_c = _sse(x_cond, y)
    sse_f = _sse(x_full, y)
    if sse_f <= 0
        return 0.0
    return float(log(max(sse_c, 1e-30) / max(sse_f, 1e-30)))
end

function spectral_granger_causality(trains, bin_size, order, n_freqs)
    trains: list[np.ndarray[Any, Any]], bin_size: int = 10, order: int = 5, n_freqs: int = 64
    ) -> np.ndarray[Any, Any]
    binned = collect([bin_spike_train(t, bin_size).astype(np.float64) for t in trains])
    d = binned.shape[0]
    beta, sigma = _var_coefficients(binned, order)
    freqs = range(0, 0.5, n_freqs)
    gc = zeros((d, d, n_freqs))
    for fi, f in enumerate(freqs)
        a_f = np.eye(d, dtype=complex)
        for k in 1:order
            coeff_block = beta[k * d : (k + 1) * d, :].T
            a_f -= coeff_block * exp(-2j * pi * f * (k + 1))
        det_a = np.linalg.det(a_f)
        if abs(det_a) < 1e-30
            continue
        h = np.linalg.inv(a_f)
        s = h @ sigma @ h.conj().T
        for i in 1:d
            for j in 1:d
                if i == j
                    continue
                if abs(s[i, i]) > 1e-30
                    gc[i, j, fi] = max(
                        0.0,
                        log(
                            abs(s[i, i]) / abs(s[i, i] - sigma[j, j] * abs(h[i, j]) ^ 2 + 1e-30)
                        ).real,
                    )
    return gc
end

function partial_directed_coherence(trains, bin_size, order, n_freqs)
    trains: list[np.ndarray[Any, Any]], bin_size: int = 10, order: int = 5, n_freqs: int = 64
    ) -> np.ndarray[Any, Any]
    binned = collect([bin_spike_train(t, bin_size).astype(np.float64) for t in trains])
    d = binned.shape[0]
    beta, _ = _var_coefficients(binned, order)
    freqs = range(0, 0.5, n_freqs)
    pdc = zeros((d, d, n_freqs))
    for fi, f in enumerate(freqs)
        a_f = np.eye(d, dtype=complex)
        for k in 1:order
            coeff_block = beta[k * d : (k + 1) * d, :].T
            a_f -= coeff_block * exp(-2j * pi * f * (k + 1))
        for j in 1:d
            norm = sqrt(sum(abs(a_f[:, j]) ^ 2))
            if norm > 0
                for i in 1:d
                    pdc[i, j, fi] = abs(a_f[i, j]) / norm
    return pdc
end

function directed_transfer_function(trains, bin_size, order, n_freqs)
    trains: list[np.ndarray[Any, Any]], bin_size: int = 10, order: int = 5, n_freqs: int = 64
    ) -> np.ndarray[Any, Any]
    binned = collect([bin_spike_train(t, bin_size).astype(np.float64) for t in trains])
    d = binned.shape[0]
    beta, sigma = _var_coefficients(binned, order)
    freqs = range(0, 0.5, n_freqs)
    dtf = zeros((d, d, n_freqs))
    for fi, f in enumerate(freqs)
        a_f = np.eye(d, dtype=complex)
        for k in 1:order
            coeff_block = beta[k * d : (k + 1) * d, :].T
            a_f -= coeff_block * exp(-2j * pi * f * (k + 1))
        det_a = np.linalg.det(a_f)
        if abs(det_a) < 1e-30
            continue
        h = np.linalg.inv(a_f)
        for i in 1:d
            norm = sqrt(sum(abs(h[i, :]) ^ 2))
            if norm > 0
                for j in 1:d
                    dtf[i, j, fi] = abs(h[i, j]) / norm
    return dtf
end

end # module CausalityAccel
