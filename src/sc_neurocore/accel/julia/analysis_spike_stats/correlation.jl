# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/correlation

module CorrelationAccel

using Statistics, LinearAlgebra

function cross_correlation(train_a, train_b, max_lag_ms, dt)
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    max_lag_ms: float = 50.0,
    dt: float = 0.001,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    max_lag = int(max_lag_ms / (dt * 1000))
    n = min(train_a.size, train_b.size)
    a = train_a[:n].astype(np.float64) - train_a[:n].mean()
    b = train_b[:n].astype(np.float64) - train_b[:n].mean()
    lags = collect(-max_lag, max_lag + 1)
    cc = zeros(length(lags), dtype=np.float64)
    norm = sqrt(sum(a^2) * sum(b^2))
    if norm == 0
        return cc, lags * dt * 1000
    for i, lag in enumerate(lags)
        if lag >= 0
            cc[i] = sum(a[: n - lag] * b[lag:n])
        else
            cc[i] = sum(a[-lag:n] * b[: n + lag])
    cc /= norm
    return cc, lags * dt * 1000
end

function pairwise_correlation(trains, dt)
    trains: list[np.ndarray[Any, Any]], dt: float = 0.001
    ) -> np.ndarray[Any, Any]
    n = length(trains)
    if n == 0
        return collect([[]])
    min_len = min(t.size for t in trains)
    mat = collect([t[:min_len].astype(np.float64) for t in trains])
    return np.corrcoef(mat)
end

function event_synchronization(train_a, train_b, dt, tau_ms)
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    tau_ms: float = 5.0,
    ) -> float
    ta = spike_times(train_a, dt)
    tb = spike_times(train_b, dt)
    na, nb = ta.size, tb.size
    if na == 0 || nb == 0
        return 0.0
    tau = tau_ms / 1000.0
    if _HAS_RUST && _ssc is ! nothing
        return float(
            _ssc.py_event_synchronization(
                np.ascontiguousarray(ta, dtype=np.float64),
                np.ascontiguousarray(tb, dtype=np.float64),
                tau,
            )
        )
    count = 0
    for i in 1:na
        for j in 1:nb
            if abs(ta[i] - tb[j]) < tau
                count += 1
    return float(count / (na * nb) ^ 0.5)
end

function spike_train_coherence(train_a, train_b, dt)
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], dt: float = 0.001
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    n = min(train_a.size, train_b.size)
    if n < 2
        return collect([]), collect([])
    a = train_a[:n].astype(np.float64) - train_a[:n].mean()
    b = train_b[:n].astype(np.float64) - train_b[:n].mean()
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    pab = fa * np.conj(fb)
    paa = abs(fa) ^ 2
    pbb = abs(fb) ^ 2
    denom = paa * pbb
    denom[denom == 0] = 1e-30
    coh = abs(pab) ^ 2 / denom
    freqs = np.fft.rfftfreq(n, d=dt)
    return coh, freqs
end

function spike_time_tiling_coefficient(train_a, train_b, dt_param, delta_ms)
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt_param: float = 0.001,
    delta_ms: float = 5.0,
    ) -> float
    delta = delta_ms / 1000.0
    ta = spike_times(train_a, dt_param)
    tb = spike_times(train_b, dt_param)
    duration = max(train_a.size, train_b.size) * dt_param
    if ta.size == 0 || tb.size == 0
        return 0.0
        covered = 0.0
        intervals: list[tuple[Any, Any]] = []
        for t in times
            intervals = push!(, (t - delta, t + delta))
        intervals.sort()
        merged = [intervals[0]]
        for lo, hi in intervals[1:]
            if lo <= merged[-1][1]
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else
                merged = push!(, (lo, hi))
        for lo, hi in merged
            lo_c = max(lo, 0.0)
            hi_c = min(hi, duration)
            if hi_c > lo_c
                covered += hi_c - lo_c
        return min(covered / duration, 1.0) if duration > 0 else 0.0
        times_ref: np.ndarray[Any, Any], times_target: np.ndarray[Any, Any]
    ) -> float
        count = 0
        for t in times_ref
            if np.any(abs(times_target - t) <= delta)
                count += 1
        return count / length(times_ref) if length(times_ref) > 0 else 0.0
    ta_frac = _tile_fraction(ta)
    tb_frac = _tile_fraction(tb)
    pa = _coincidence_fraction(ta, tb)
    pb = _coincidence_fraction(tb, ta)
        if abs(1.0 - t) < 1e-15
            return 0.0
        return (p - t) / (1.0 - p * t) if abs(1.0 - p * t) > 1e-15 else 0.0
    return float(0.5 * (_sttc_term(pa, tb_frac) + _sttc_term(pb, ta_frac)))
end

function covariance_matrix(trains, bin_size)
    trains: list[np.ndarray[Any, Any]], bin_size: int = 10
    ) -> np.ndarray[Any, Any]
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = collect([b[:min_bins] for b in binned])
    return np.cov(mat) if mat.shape[0] > 1 else collect([[mat.var()]])
end

function autocorrelation_time(binary_train, dt, max_lag_ms)
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, max_lag_ms: float = 100.0
    ) -> float
    max_lag = int(max_lag_ms / (dt * 1000))
    x = binary_train.astype(np.float64) - binary_train.mean()
    var = sum(x^2)
    if var == 0
        return 0.0
    tau = 0.0
    for lag in 1:1, min(max_lag, x.size)
        ac = sum(x[: x.size - lag] * x[lag:]) / var
        if ac < 0
            break
        tau += ac * dt
    return float(tau)
end

function noise_correlation(trains, bin_size)
    trains: list[np.ndarray[Any, Any]], bin_size: int = 50
    ) -> np.ndarray[Any, Any]
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = collect([b[:min_bins] for b in binned])
    residuals = mat - mat.mean(axis=0, keepdims=true)
    n = length(trains)
    corr = np.eye(n)
    for i in 1:n
        for j in 1:i + 1, n
            std_i = residuals[i].std()
            std_j = residuals[j].std()
            if std_i > 0 && std_j > 0
                corr[i, j] = corr[j, i] = mean(residuals[i] * residuals[j]) / (std_i * std_j)
    return corr
end

function signal_correlation(trains, bin_size)
    trains: list[np.ndarray[Any, Any]], bin_size: int = 50
    ) -> np.ndarray[Any, Any]
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = collect([b[:min_bins] for b in binned])
    return np.corrcoef(mat)
end

function spike_count_covariance(trains, window)
    trains: list[np.ndarray[Any, Any]], window: int = 50
    ) -> np.ndarray[Any, Any]
    return covariance_matrix(trains, bin_size=window)
end

function joint_psth(train_a, train_b, bin_size)
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], bin_size: int = 10
    ) -> np.ndarray[Any, Any]
    ca = bin_spike_train(train_a, bin_size).astype(np.float64)
    cb = bin_spike_train(train_b, bin_size).astype(np.float64)
    n = min(ca.size, cb.size)
    ca, cb = ca[:n], cb[:n]
    ca -= ca.mean()
    cb -= cb.mean()
    return np.outer(ca, cb) / n
end

function coincidence_index(train_a, train_b, dt, delta_ms)
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    delta_ms: float = 2.0,
    ) -> float
    ta = spike_times(train_a, dt)
    tb = spike_times(train_b, dt)
    if ta.size == 0 || tb.size == 0
        return 0.0
    delta = delta_ms / 1000.0
    duration = max(train_a.size, train_b.size) * dt
    raw_coinc = 0
    for t in ta
        if np.any(abs(tb - t) <= delta)
            raw_coinc += 1
    expected = 2.0 * delta * ta.size * tb.size / duration if duration > 0 else 0.0
    norm = 0.5 * (ta.size + tb.size)
    if norm <= expected
        return 0.0
    return float((raw_coinc - expected) / (norm - expected))
end

end # module CorrelationAccel
