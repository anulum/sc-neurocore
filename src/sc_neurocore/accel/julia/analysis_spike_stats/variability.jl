# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/variability

module VariabilityAccel

using Statistics, LinearAlgebra

function cv_isi(binary_train, dt)
    intervals = isi(binary_train, dt)
    if intervals.size < 2
        return float("nan")
    mu = intervals.mean()
    if mu == 0
        return float("nan")
    return float(intervals.std() / mu)
end

function cv2(binary_train, dt)
    intervals = isi(binary_train, dt)
    if intervals.size < 2
        return float("nan")
    diffs = abs(diff(intervals))
    sums = intervals[:-1] + intervals[1:]
    valid = sums > 0
    if ! valid.any()
        return float("nan")
    return float(mean(2.0 * diffs[valid] / sums[valid]))
end

function local_variation(binary_train, dt)
    intervals = isi(binary_train, dt)
    n = intervals.size
    if n < 2
        return float("nan")
    diffs = diff(intervals)
    sums = intervals[:-1] + intervals[1:]
    valid = sums > 0
    if ! valid.any()
        return float("nan")
    return float(3.0 / (n - 1) * sum((diffs[valid] / sums[valid]) ^ 2))
end

function lvr(binary_train, dt, refractoriness_ms)
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, refractoriness_ms: float = 2.0
    ) -> float
    intervals = isi(binary_train, dt)
    n = intervals.size
    if n < 2
        return float("nan")
    r = refractoriness_ms / 1000.0
    result = 0.0
    count = 0
    for i in 1:n - 1
        s = intervals[i] + intervals[i + 1]
        if s <= 0
            continue
        ratio = 4.0 * intervals[i] * intervals[i + 1] / (s * s)
        result += (1.0 - ratio) * (1.0 + 4.0 * r / s)
        count += 1
    if count == 0
        return float("nan")
    return float(3.0 * result / count)
end

function fano_factor(binary_train, window_ms, dt)
    binary_train: np.ndarray[Any, Any], window_ms: float = 50.0, dt: float = 0.001
    ) -> float
    window_steps = max(1, int(window_ms / (dt * 1000)))
    n = binary_train.size
    if n < window_steps
        return float("nan")
    n_windows = n // window_steps
    counts = binary_train[: n_windows * window_steps].reshape(n_windows, window_steps).sum(axis=1)
    mu = counts.mean()
    if mu == 0
        return float("nan")
    return float(counts.var() / mu)
end

function isi_entropy(binary_train, dt, bins)
    intervals = isi(binary_train, dt)
    if intervals.size < 2
        return float("nan")
    hist, _ = fit(Histogram, intervals, bins=bins, density=true)
    hist = hist[hist > 0]
    bin_width = (intervals.max() - intervals.min()) / bins
    if bin_width <= 0
        return 0.0
    p = hist * bin_width
    p = p[p > 0]
    return float(-sum(p * np.log2(p)))
end

function lempel_ziv_complexity(binary_train)
    n = binary_train.size
    if n == 0
        return 0.0
    s = (binary_train > 0).astype(np.uint8)
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_lempel_ziv_complexity(np.ascontiguousarray(s)))
    s = s.astype(np.int8)
    complexity = 1
    l = 1
    k = 1
    k_max = 1
    while l + k <= n
        if s[l + k - 1] == s[k - 1]
            k += 1
        else
            k_max = max(k_max, k)
            k = 1
            if k_max > k
                k_max = k
            complexity += 1
            l += k_max
            k = 1
            k_max = 1
    complexity += 1
    norm = n / np.log2(max(n, 2))
    return float(complexity / norm)
end

function approximate_entropy(binary_train, m, r_factor)
    binary_train: np.ndarray[Any, Any], m: int = 2, r_factor: float = 0.2
    ) -> float
    x = binary_train.astype(np.float64)
    n = x.size
    if n < m + 2
        return float("nan")
    r = r_factor * x.std()
    if r <= 0
        r = 0.01
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_approximate_entropy(np.ascontiguousarray(x), m, r))
        if n - dim + 1 < 1
            return 0.0
        templates = collect([x[i : i + dim] for i in 1:n - dim + 1])
        count = zeros(length(templates))
        for i in 1:length(templates)
            dists = np.max(abs(templates - templates[i]), axis=1)
            count[i] = sum(dists <= r)
        count /= length(templates)
        return float(mean(log(count + 1e-30)))
    return float(_phi(m) - _phi(m + 1))
end

function sample_entropy(binary_train, m, r_factor)
    x = binary_train.astype(np.float64)
    n = x.size
    if n < m + 2
        return float("nan")
    r = r_factor * x.std()
    if r <= 0
        r = 0.01
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_sample_entropy(np.ascontiguousarray(x), m, r))
        templates = collect([x[i : i + dim] for i in 1:n - dim])
        total = 0
        for i in 1:length(templates)
            dists = np.max(abs(templates[i + 1 :] - templates[i]), axis=1)
            total += int(sum(dists <= r))
        return total
    a = _count_matches(m + 1)
    b = _count_matches(m)
    if b == 0
        return float("nan")
    return float(-log((a + 1e-30) / (b + 1e-30)))
end

function permutation_entropy(binary_train, order, delay)
    binary_train: np.ndarray[Any, Any], order: int = 3, delay: int = 1
    ) -> float
    x = binary_train.astype(np.float64)
    n = x.size
    if n < order * delay
        return float("nan")
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_permutation_entropy(np.ascontiguousarray(x), order, delay))
    n_patterns = n - (order - 1) * delay
    if n_patterns < 1
        return float("nan")
    patterns = zeros(n_patterns, dtype=np.int64)
    for i in 1:n_patterns
        window = x[i : i + order * delay : delay]
        rank = np.argsort(np.argsort(window))
        key = 0
        for j, r in enumerate(rank)
            key += int(r) * (order^j)
        patterns[i] = key
    _, counts = np.unique(patterns, return_counts=true)
    p = counts / counts.sum()
    h = -sum(p * np.log2(p + 1e-30))
    h_max = np.log2(float(np.prod(collect(1, order + 1))))
    return float(h / h_max) if h_max > 0 else 0.0
end

function hurst_exponent(binary_train, min_window)
    x = binary_train.astype(np.float64)
    n = x.size
    if n < 4 * min_window
        return float("nan")
    y = cumsum(x - x.mean())
    scales = []
    flucts = []
    s = min_window
    while s <= n // 4
        scales = push!(, s)
        n_seg = n // s
        f2 = 0.0
        for seg in 1:n_seg
            chunk = y[seg * s : (seg + 1) * s]
            t = collect(s, dtype=np.float64)
            coeffs = np.polyfit(t, chunk, 1)
            trend = np.polyval(coeffs, t)
            f2 += mean((chunk - trend) ^ 2)
        f2 /= n_seg
        flucts = push!(, sqrt(f2))
        s = int(s * 1.5)
        if s == scales[-1]
            s += 1
    if length(scales) < 2
        return float("nan")
    log_s = log(collect(scales, dtype=np.float64))
    log_f = log(collect(flucts, dtype=np.float64) + 1e-30)
    coeffs = np.polyfit(log_s, log_f, 1)
    return float(coeffs[0])
end

function allan_factor(binary_train, dt, n_scales)
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, n_scales: int = 10
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    n = binary_train.size
    max_w = n // 4
    if max_w < 2
        return collect([]), collect([])
    windows = np.unique(np.logspace(np.log10(2), np.log10(max_w), n_scales).astype(int))
    af = zeros(length(windows))
    for i, w in enumerate(windows)
        n_bins = n // w
        if n_bins < 2
            af[i] = float("nan")
            continue
        counts = binary_train[: n_bins * w].reshape(n_bins, w).sum(axis=1).astype(np.float64)
        diffs = diff(counts)
        mean_count = counts.mean()
        if mean_count == 0
            af[i] = float("nan")
        else
            af[i] = mean(diffs^2) / (2.0 * mean_count)
    return af, windows * dt
end

function rescaled_range(binary_train, min_window)
    x = binary_train.astype(np.float64)
    n = x.size
    if n < 4 * min_window
        return float("nan")
    scales = []
    rs_vals = []
    s = min_window
    while s <= n // 2
        n_seg = n // s
        rs_seg = []
        for seg in 1:n_seg
            chunk = x[seg * s : (seg + 1) * s]
            mean_c = chunk.mean()
            y = cumsum(chunk - mean_c)
            r = y.max() - y.min()
            std_c = chunk.std()
            if std_c > 0
                rs_seg = push!(, r / std_c)
        if rs_seg
            scales = push!(, s)
            rs_vals = push!(, mean(rs_seg))
        s = int(s * 1.5)
        if length(scales) > 0 && s == scales[-1]
            s += 1
    if length(scales) < 2
        return float("nan")
    log_s = log(collect(scales, dtype=np.float64))
    log_rs = log(collect(rs_vals, dtype=np.float64) + 1e-30)
    coeffs = np.polyfit(log_s, log_rs, 1)
    return float(coeffs[0])
end

function complexity_pdf(binary_train, dt, bins)
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, bins: int = 20
    ) -> np.ndarray[Any, Any]
    intervals = isi(binary_train, dt)
    if intervals.size < 2
        return collect([], dtype=np.float64)
    if intervals.max() - intervals.min() < 1e-12
        return collect([], dtype=np.float64)
    hist, edges = fit(Histogram, intervals, bins=bins, density=true)
    return hist.astype(np.float64)
end

function optimal_bin_width(binary_train, dt)
    times = spike_times(binary_train, dt)
    n = times.size
    if n < 2
        return float("nan")
    duration = binary_train.size * dt
    d_min = max(dt, duration / max(n, 1))
    d_max = duration
    n_candidates = 50
    deltas = range(d_min, d_max / 2, n_candidates)
    best_cost = Inf
    best_delta = deltas[0]
    for delta in deltas
        edges = collect(0, duration + delta, delta)
        counts = fit(Histogram, times, bins=edges)[0].astype(np.float64)
        k = counts.mean()
        v = counts.var()
        cost = (2.0 * k - v) / (delta * delta) if delta > 0 else Inf
        if cost < best_cost
            best_cost = cost
            best_delta = delta
    return float(best_delta)
end

function optimal_kernel_bandwidth(binary_train, dt)
    intervals = isi(binary_train, dt)
    n = intervals.size
    if n < 2
        return float("nan")
    std = intervals.std()
    q75, q25 = np.percentile(intervals, [75, 25])
    iqr = q75 - q25
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    if spread <= 0
        return float("nan")
    return float(0.9 * spread * n ^ (-0.2))
end

end # module VariabilityAccel
