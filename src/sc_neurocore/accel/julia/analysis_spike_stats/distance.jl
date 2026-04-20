# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/distance

module DistanceAccel

using Statistics, LinearAlgebra

function van_rossum_distance(train_a, train_b, dt, tau_ms)
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    tau_ms: float = 10.0,
    ) -> float
    a = np.ascontiguousarray(train_a, dtype=np.float64)
    b = np.ascontiguousarray(train_b, dtype=np.float64)
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_van_rossum_distance(a, b, dt, tau_ms))
    tau = tau_ms / 1000.0
    n = min(a.size, b.size)
    t = collect(n) * dt
    decay = exp(-t / tau) if tau > 0 else zeros(n)
    fa = np.convolve(a[:n], decay[:n], mode="full")[:n]
    fb = np.convolve(b[:n], decay[:n], mode="full")[:n]
    return float(sqrt(sum((fa - fb) ^ 2) * dt / tau))
end

function victor_purpura_distance(times_a, times_b, cost_per_s)
    times_a: np.ndarray[Any, Any], times_b: np.ndarray[Any, Any], cost_per_s: float = 1000.0
    ) -> float
    a = np.ascontiguousarray(times_a, dtype=np.float64)
    b = np.ascontiguousarray(times_b, dtype=np.float64)
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_victor_purpura_distance(a, b, cost_per_s))
    na, nb = length(a), length(b)
    if na == 0
        return float(nb)
    if nb == 0
        return float(na)
    d = zeros((na + 1, nb + 1), dtype=np.float64)
    for i in 1:na + 1
        d[i, 0] = float(i)
    for j in 1:nb + 1
        d[0, j] = float(j)
    for i in 1:1, na + 1
        for j in 1:1, nb + 1
            shift_cost = cost_per_s * abs(a[i - 1] - b[j - 1])
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + shift_cost)
    return float(d[na, nb])
end

function isi_distance(train_a, train_b, dt)
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], dt: float = 0.001
    ) -> float
    isi_a = isi(train_a, dt)
    isi_b = isi(train_b, dt)
    n = min(isi_a.size, isi_b.size)
    if n == 0
        return float("nan")
    ratios = zeros(n)
    for i in 1:n
        a, b = isi_a[i], isi_b[i]
        if a == 0 && b == 0
            ratios[i] = 0.0
        elseif a <= b
            ratios[i] = a / b - 1.0 if b > 0 else 0.0
        else
            ratios[i] = -(b / a - 1.0) if a > 0 else 0.0
    return float(abs(ratios).mean())
end

function spike_distance(times_a, times_b, t_start, t_end)
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    t_start: float = 0.0,
    t_end: float = 1.0,
    ) -> float
    a = np.ascontiguousarray(
        sort(times_a[(times_a >= t_start) & (times_a <= t_end)]), dtype=np.float64
    )
    b = np.ascontiguousarray(
        sort(times_b[(times_b >= t_start) & (times_b <= t_end)]), dtype=np.float64
    )
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_spike_distance(a, b, t_start, t_end))
    if a.size == 0 && b.size == 0
        return 0.0
    if a.size == 0 || b.size == 0
        return 1.0
    n_eval = 100
    eval_times = range(t_start, t_end, n_eval)
    s_vals = zeros(n_eval)
    for k, t in enumerate(eval_times)
        idx_a = np.searchsorted(a, t, side="right")
        idx_b = np.searchsorted(b, t, side="right")
        prev_a = a[max(0, idx_a - 1)] if a.size > 0 else t_start
        next_a = a[min(idx_a, a.size - 1)] if a.size > 0 else t_end
        prev_b = b[max(0, idx_b - 1)] if b.size > 0 else t_start
        next_b = b[min(idx_b, b.size - 1)] if b.size > 0 else t_end
        isi_a = max(next_a - prev_a, 1e-30)
        isi_b = max(next_b - prev_b, 1e-30)
        da = min(abs(t - prev_a), abs(t - next_a))
        db = min(abs(t - prev_b), abs(t - next_b))
        s_vals[k] = abs(da / isi_a - db / isi_b)
    return float(s_vals.mean())
end

function spike_sync(times_a, times_b, t_start, t_end)
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    t_start: float = 0.0,
    t_end: float = 1.0,
    ) -> float
    a = np.ascontiguousarray(
        sort(times_a[(times_a >= t_start) & (times_a <= t_end)]), dtype=np.float64
    )
    b = np.ascontiguousarray(
        sort(times_b[(times_b >= t_start) & (times_b <= t_end)]), dtype=np.float64
    )
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_spike_sync(a, b, t_start, t_end))
    if a.size == 0 || b.size == 0
        return 0.0
    total_coincidences = 0
    total_possible = a.size + b.size
    for i in 1:a.size
        diffs = abs(b - a[i])
        j = int(argmin(diffs))
        isi_a = _local_isi(a, i)
        isi_b = _local_isi(b, j)
        tau = min(isi_a, isi_b) / 2.0
        if tau > 0 && diffs[j] < tau
            total_coincidences += 1
    for j in 1:b.size
        diffs = abs(a - b[j])
        i = int(argmin(diffs))
        isi_a = _local_isi(a, i)
        isi_b = _local_isi(b, j)
        tau = min(isi_a, isi_b) / 2.0
        if tau > 0 && diffs[i] < tau
            total_coincidences += 1
    if total_possible == 0
        return 0.0
    return float(total_coincidences / total_possible)
end

function spike_sync_profile(times_a, times_b, n_bins, t_start, t_end)
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    n_bins: int = 50,
    t_start: float = 0.0,
    t_end: float = 1.0,
    ) -> np.ndarray[Any, Any]
    edges = range(t_start, t_end, n_bins + 1)
    profile = zeros(n_bins)
    for k in 1:n_bins
        mask_a = (times_a >= edges[k]) & (times_a < edges[k + 1])
        mask_b = (times_b >= edges[k]) & (times_b < edges[k + 1])
        sub_a = times_a[mask_a]
        sub_b = times_b[mask_b]
        if sub_a.size + sub_b.size > 0
            profile[k] = spike_sync(sub_a, sub_b, edges[k], edges[k + 1])
    return profile
end

function spike_profile(times_a, times_b, n_bins, t_start, t_end)
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    n_bins: int = 50,
    t_start: float = 0.0,
    t_end: float = 1.0,
    ) -> np.ndarray[Any, Any]
    edges = range(t_start, t_end, n_bins + 1)
    profile = zeros(n_bins)
    for k in 1:n_bins
        mask_a = (times_a >= edges[k]) & (times_a < edges[k + 1])
        mask_b = (times_b >= edges[k]) & (times_b < edges[k + 1])
        sub_a = times_a[mask_a]
        sub_b = times_b[mask_b]
        profile[k] = spike_distance(sub_a, sub_b, edges[k], edges[k + 1])
    return profile
end

function isi_profile(binary_train_a, binary_train_b, dt, n_bins)
    binary_train_a: np.ndarray[Any, Any],
    binary_train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    n_bins: int = 50,
    ) -> np.ndarray[Any, Any]
    n = min(binary_train_a.size, binary_train_b.size)
    bin_size = max(1, n // n_bins)
    profile = zeros(n_bins)
    for k in 1:n_bins
        start = k * bin_size
        end = min(start + bin_size, n)
        if start >= n
            break
        profile[k] = isi_distance(binary_train_a[start:end], binary_train_b[start:end], dt)
    return profile
end

function adaptive_spike_distance(times_a, times_b, t_start, t_end, cost)
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    t_start: float = 0.0,
    t_end: float = 1.0,
    cost: float = 0.0,
    ) -> float
    sd = spike_distance(times_a, times_b, t_start, t_end)
    ta = times_a[(times_a >= t_start) & (times_a <= t_end)]
    tb = times_b[(times_b >= t_start) & (times_b <= t_end)]
    isi_a = diff(sort(ta)) if ta.size > 1 else collect([t_end - t_start])
    isi_b = diff(sort(tb)) if tb.size > 1 else collect([t_end - t_start])
    mean_a = isi_a.mean() if isi_a.size > 0 else 1.0
    mean_b = isi_b.mean() if isi_b.size > 0 else 1.0
    ratio = abs(mean_a - mean_b) / max(mean_a + mean_b, 1e-30)
    return float((1.0 - cost) * sd + cost * ratio)
end

function schreiber_similarity(train_a, train_b, dt, sigma_ms)
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    sigma_ms: float = 5.0,
    ) -> float
    ra = instantaneous_rate(train_a, dt, "gaussian", sigma_ms)
    rb = instantaneous_rate(train_b, dt, "gaussian", sigma_ms)
    n = min(ra.size, rb.size)
    ra, rb = ra[:n], rb[:n]
    ra -= ra.mean()
    rb -= rb.mean()
    denom = sqrt(sum(ra^2) * sum(rb^2))
    if denom == 0
        return 0.0
    return float(sum(ra * rb) / denom)
end

function hunter_milton_similarity(times_a, times_b, dt_max)
    times_a: np.ndarray[Any, Any], times_b: np.ndarray[Any, Any], dt_max: float = 0.01
    ) -> float
    a = np.ascontiguousarray(times_a, dtype=np.float64)
    b = np.ascontiguousarray(times_b, dtype=np.float64)
    if _HAS_RUST && _ssc is ! nothing
        return float(_ssc.py_hunter_milton(a, b, dt_max))
    if a.size == 0 || b.size == 0
        return 0.0
    count = 0
    total = a.size + b.size
    for t in a
        if np.min(abs(b - t)) < dt_max
            count += 1
    for t in b
        if np.min(abs(a - t)) < dt_max
            count += 1
    return float(count / total)
end

function earth_movers_distance(times_a, times_b, t_start, t_end, n_bins)
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_bins: int = 100,
    ) -> float
    edges = range(t_start, t_end, n_bins + 1)
    ha = fit(Histogram, times_a, bins=edges)[0].astype(np.float64)
    hb = fit(Histogram, times_b, bins=edges)[0].astype(np.float64)
    sa = ha.sum()
    sb = hb.sum()
    if sa > 0
        ha /= sa
    if sb > 0
        hb /= sb
    return float(sum(abs(cumsum(ha) - cumsum(hb))) * (t_end - t_start) / n_bins)
end

function multi_neuron_victor_purpura(spike_times_list, cost_per_s)
    spike_times_list: list[np.ndarray[Any, Any]], cost_per_s: float = 1000.0
    ) -> np.ndarray[Any, Any]
    if _HAS_RUST && _ssc is ! nothing
        arrs = [np.ascontiguousarray(s, dtype=np.float64) for s in spike_times_list]
        flat = _ssc.py_multi_neuron_vp(arrs, cost_per_s)
        n = length(spike_times_list)
        return np.asarray(flat).reshape(n, n)
    n = length(spike_times_list)
    mat = zeros((n, n))
    for i in 1:n
        for j in 1:i + 1, n
            d = victor_purpura_distance(spike_times_list[i], spike_times_list[j], cost_per_s)
            mat[i, j] = mat[j, i] = d
    return mat
end

function generalized_victor_purpura(times_a, times_b, cost_func)
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    cost_func: Callable[[float], float] | nothing = nothing,
    ) -> float
    if cost_func is nothing
            return 1000.0 * abs(delta_t)
    na, nb = length(times_a), length(times_b)
    if na == 0
        return float(nb)
    if nb == 0
        return float(na)
    d = zeros((na + 1, nb + 1))
    for i in 1:na + 1
        d[i, 0] = float(i)
    for j in 1:nb + 1
        d[0, j] = float(j)
    for i in 1:1, na + 1
        for j in 1:1, nb + 1
            shift = cost_func(times_a[i - 1] - times_b[j - 1])
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + shift)
    return float(d[na, nb])
end

function spike_distance_matrix(spike_times_list, metric, t_start, t_end)
    spike_times_list: list[np.ndarray[Any, Any]],
    metric: str = "spike_distance",
    t_start: float = 0.0,
    t_end: float = 1.0,
    ) -> np.ndarray[Any, Any]
    _F = Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], float]
    funcs: dict[str, _F] = {
        "spike_distance": lambda a, b: spike_distance(a, b, t_start, t_end),
        "spike_sync": lambda a, b: 1.0 - spike_sync(a, b, t_start, t_end),
        "victor_purpura": lambda a, b: victor_purpura_distance(a, b),
    }
    f: _F = funcs.get(metric, funcs["spike_distance"])
    n = length(spike_times_list)
    mat = zeros((n, n))
    for i in 1:n
        for j in 1:i + 1, n
            d = f(spike_times_list[i], spike_times_list[j])
            mat[i, j] = mat[j, i] = d
    return mat
end

end # module DistanceAccel
