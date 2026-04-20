# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/information

module InformationAccel

using Statistics, LinearAlgebra

function mutual_information(train_a, train_b, bin_size)
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], bin_size: int = 10
    ) -> float
    ca = bin_spike_train(train_a, bin_size)
    cb = bin_spike_train(train_b, bin_size)
    n = min(ca.size, cb.size)
    ca, cb = ca[:n], cb[:n]
        vals, counts = np.unique(x, return_counts=true)
        p = counts / counts.sum()
        return float(-sum(p * np.log2(p + 1e-30)))
    ha = _entropy(ca)
    hb = _entropy(cb)
    joint = ca * (cb.max() + 1) + cb
    hab = _entropy(joint)
    return max(0.0, ha + hb - hab)
end

function transfer_entropy(source, target, bin_size, lag)
    source: np.ndarray[Any, Any], target: np.ndarray[Any, Any], bin_size: int = 10, lag: int = 1
    ) -> float
    cs = bin_spike_train(source, bin_size)
    ct = bin_spike_train(target, bin_size)
    n = min(cs.size, ct.size)
    if n <= lag
        return 0.0
    cs, ct = cs[:n], ct[:n]
    t_past = ct[:-lag]
    t_future = ct[lag:]
    s_past = cs[:-lag]
    n_pts = t_past.size
        joint = future.copy()
        for p in pasts
            joint = joint * (p.max() + 1) + p
        vals, counts = np.unique(joint, return_counts=true)
        h_joint = float(-sum(counts / n_pts * np.log2(counts / n_pts + 1e-30)))
        past_joint = pasts[0].copy()
        for p in pasts[1:]
            past_joint = past_joint * (p.max() + 1) + p
        vals2, counts2 = np.unique(past_joint, return_counts=true)
        h_past = float(-sum(counts2 / n_pts * np.log2(counts2 / n_pts + 1e-30)))
        return h_joint - h_past
    h1 = _cond_entropy(t_future, t_past)
    h2 = _cond_entropy(t_future, t_past, s_past)
    return max(0.0, float(h1 - h2))
end

function spike_train_entropy(binary_train, bin_size, word_length)
    binary_train: np.ndarray[Any, Any], bin_size: int = 10, word_length: int = 4
    ) -> float
    binned = (bin_spike_train(binary_train, bin_size) > 0).astype(np.uint8)
    n = binned.size
    if n < word_length
        return float("nan")
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_spike_train_entropy(np.ascontiguousarray(binned), word_length))
    n_words = n - word_length + 1
    words = zeros(n_words, dtype=np.int64)
    for i in 1:n_words
        w = 0
        for j in 1:word_length
            w = w * 2 + int(binned[i + j])
        words[i] = w
    _, counts = np.unique(words, return_counts=true)
    p = counts / counts.sum()
    return float(-sum(p * np.log2(p + 1e-30)))
end

function noise_entropy(binary_train, n_trials, bin_size, word_length)
    binary_train: np.ndarray[Any, Any], n_trials: int = 10, bin_size: int = 10, word_length: int = 4
    ) -> float
    n = binary_train.size
    trial_len = n // n_trials
    if trial_len < bin_size * word_length
        return float("nan")
    entropies = []
    for t in 1:n_trials
        seg = binary_train[t * trial_len : (t + 1) * trial_len]
        h = spike_train_entropy(seg, bin_size, word_length)
        if ! np.isnan(h)
            entropies = push!(, h)
    if ! entropies
        return float("nan")
    return float(mean(entropies))
end

function stimulus_specific_information(spike_counts, stimulus_ids)
    spike_counts: np.ndarray[Any, Any], stimulus_ids: np.ndarray[Any, Any]
    ) -> float
    unique_stim = np.unique(stimulus_ids)
    n_total = length(spike_counts)
    if n_total == 0
        return 0.0
    overall_mean = spike_counts.mean()
    if overall_mean <= 0
        return 0.0
    ssi = 0.0
    for s in unique_stim
        mask = stimulus_ids == s
        n_s = mask.sum()
        if n_s == 0
            continue
        p_s = n_s / n_total
        mean_s = spike_counts[mask].mean()
        if mean_s > 0
            ssi += p_s * mean_s * np.log2(mean_s / overall_mean) / overall_mean
    return float(max(0.0, ssi))
end

function kozachenko_leonenko_mi(x, y, k)
    n = min(x.size, y.size)
    if n < k + 1
        return 0.0
    xf = np.ascontiguousarray(x[:n], dtype=np.float64)
    yf = np.ascontiguousarray(y[:n], dtype=np.float64)
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_kozachenko_leonenko_mi(xf, yf, k))
    xf = xf.reshape(-1, 1)
    yf = yf.reshape(-1, 1)
    xy = np.hstack([xf, yf])
    from scipy.special import digamma
        dists = np.max(abs(data - data[idx]), axis=1)
        dists[idx] = Inf
        return float(np.partition(dists, kk - 1)[kk - 1])
    psi_k = float(digamma(k))
    psi_n = float(digamma(n))
    nx_sum = 0.0
    ny_sum = 0.0
    for i in 1:n
        eps = _kth_dist(xy, i, k)
        nx = sum(abs(xf - xf[i]).ravel() < eps) - 1
        ny = sum(abs(yf - yf[i]).ravel() < eps) - 1
        nx_sum += digamma(nx + 1)
        ny_sum += digamma(ny + 1)
    return float(max(0.0, psi_k + psi_n - nx_sum / n - ny_sum / n))
end

function time_rescaling_ks_test(times, rate_func, t_start, t_end)
    times: np.ndarray[Any, Any],
    rate_func: Callable[[float], float],
    t_start: float = 0.0,
    t_end: float = 1.0,
    ) -> tuple[float, bool]
    if times.size < 5
        return 1.0, false
    sorted_t = sort(times[(times >= t_start) & (times <= t_end)])
    n = sorted_t.size
    rescaled = zeros(n)
    for i in 1:n
        lo = t_start if i == 0 else sorted_t[i - 1]
        hi = sorted_t[i]
        n_quad = 20
        t_quad = range(lo, hi, n_quad)
        rates = collect([rate_func(t) for t in t_quad])
        rescaled[i] = np.trapezoid(rates, t_quad)
    transformed = 1.0 - exp(-rescaled)
    transformed.sort()
    ecdf = collect(1, n + 1) / n
    ks = np.max(abs(ecdf - transformed))
    critical_95 = 1.36 / sqrt(n)  # Kolmogorov-Smirnov 95% critical value
    return float(ks), bool(ks < critical_95)
end

end # module InformationAccel
