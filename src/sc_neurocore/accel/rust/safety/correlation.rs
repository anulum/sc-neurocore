// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety acceleration for correlation

pub fn cross_correlation(train_a: f64, train_b: f64, max_lag_ms: f64, dt: f64) -> f64 {
    // train_a: ndarray[Any, Any],
    // train_b: ndarray[Any, Any],
    // max_lag_ms: float = 50.0,
    // dt: float = 0.001,
    // ) -> tuple[ndarray[Any, Any], ndarray[Any, Any]] {
    // max_lag = int(max_lag_ms / (dt * 1000))
    // n = min(train_a.size, train_b.size)
    // a = train_a[:n].astype(float64) - train_a[:n].mean()
    // b = train_b[:n].astype(float64) - train_b[:n].mean()
    // lags = arange(-max_lag, max_lag + 1)
    // cc = zeros(len(lags), dtype=float64)
    // norm = (sum(a.powi2 as f64).sqrt() * sum(b.powi2))
    // if norm == 0 {
    // return cc, lags * dt * 1000
    // for i, lag in enumerate(lags) {
    // if lag >= 0 {
    // cc[i] = sum(a[: n - lag] * b[lag:n])
    // else {
    // cc[i] = sum(a[-lag:n] * b[: n + lag])
    // cc /= norm
    0.0
}

pub fn pairwise_correlation(trains: f64, dt: f64) -> f64 {
    // trains: list[ndarray[Any, Any]], dt: float = 0.001
    // ) -> ndarray[Any, Any] {
    // n = len(trains)
    // if n == 0 {
    // return array([[]])
    // min_len = min(t.size for t in trains)
    // mat = array([t[:min_len].astype(float64) for t in trains])
    // return corrcoef(mat)
    0.0
}

pub fn event_synchronization(train_a: f64, train_b: f64, dt: f64, tau_ms: f64) -> f64 {
    // train_a: ndarray[Any, Any],
    // train_b: ndarray[Any, Any],
    // dt: float = 0.001,
    // tau_ms: float = 5.0,
    // ) -> float {
    // ta = spike_times(train_a, dt)
    // tb = spike_times(train_b, dt)
    // na, nb = ta.size, tb.size
    // if na == 0 or nb == 0 {
    // return 0.0
    // tau = tau_ms / 1000.0
    // if _HAS_RUST and _ssc is not 0 {
    // return float(
    // _ssc.py_event_synchronization(
    // ascontiguousarray(ta, dtype=float64),
    // ascontiguousarray(tb, dtype=float64),
    // tau,
    // )
    // )
    // count = 0
    0.0
}

pub fn spike_train_coherence(train_a: f64, train_b: f64, dt: f64) -> f64 {
    // train_a: ndarray[Any, Any], train_b: ndarray[Any, Any], dt: float = 0.
    // ) -> tuple[ndarray[Any, Any], ndarray[Any, Any]] {
    // n = min(train_a.size, train_b.size)
    // if n < 2 {
    // return array([]), array([])
    // a = train_a[:n].astype(float64) - train_a[:n].mean()
    // b = train_b[:n].astype(float64) - train_b[:n].mean()
    // fa = fft.rfft(a)
    // fb = fft.rfft(b)
    // pab = fa * conj(fb)
    // paa = (fa as f64).abs() .powi 2
    // pbb = (fb as f64).abs() .powi 2
    // denom = paa * pbb
    // denom[denom == 0] = 1e-30
    // coh = (pab as f64).abs() .powi 2 / denom
    // freqs = fft.rfftfreq(n, d=dt)
    // return coh, freqs
    0.0
}

pub fn spike_time_tiling_coefficient(train_a: f64, train_b: f64, dt_param: f64, delta_ms: f64) -> f64 {
    // train_a: ndarray[Any, Any],
    // train_b: ndarray[Any, Any],
    // dt_param: float = 0.001,
    // delta_ms: float = 5.0,
    // ) -> float {
    // delta = delta_ms / 1000.0
    // ta = spike_times(train_a, dt_param)
    // tb = spike_times(train_b, dt_param)
    // duration = max(train_a.size, train_b.size) * dt_param
    // if ta.size == 0 or tb.size == 0 {
    // return 0.0
    // covered = 0.0
    // intervals: list[tuple[Any, Any]] = []
    // for t in times {
    // intervals.append((t - delta, t + delta))
    // intervals.sort()
    // merged = [intervals[0]]
    // for lo, hi in intervals[1:] {
    // if lo <= merged[-1][1] {
    // merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    0.0
}

pub fn covariance_matrix(trains: f64, bin_size: f64) -> f64 {
    // trains: list[ndarray[Any, Any]], bin_size: int = 10
    // ) -> ndarray[Any, Any] {
    // binned = [bin_spike_train(t, bin_size).astype(float64) for t in trains
    // min_bins = min(b.size for b in binned)
    // mat = array([b[:min_bins] for b in binned])
    // return cov(mat) if mat.shape[0] > 1 else array([[mat.var()]])
    0.0
}

pub fn autocorrelation_time(binary_train: f64, dt: f64, max_lag_ms: f64) -> f64 {
    // binary_train: ndarray[Any, Any], dt: float = 0.001, max_lag_ms: float
    // ) -> float {
    // max_lag = int(max_lag_ms / (dt * 1000))
    // x = binary_train.astype(float64) - binary_train.mean()
    // var = sum(x.powi2)
    // if var == 0 {
    // return 0.0
    // tau = 0.0
    // for lag in range(1, min(max_lag, x.size)) {
    // ac = sum(x[: x.size - lag] * x[lag:]) / var
    // if ac < 0 {
    // break
    // tau += ac * dt
    // return float(tau)
    0.0
}

pub fn noise_correlation(trains: f64, bin_size: f64) -> f64 {
    // trains: list[ndarray[Any, Any]], bin_size: int = 50
    // ) -> ndarray[Any, Any] {
    // binned = [bin_spike_train(t, bin_size).astype(float64) for t in trains
    // min_bins = min(b.size for b in binned)
    // mat = array([b[:min_bins] for b in binned])
    // residuals = mat - mat.mean(axis=0, keepdims=true)
    // n = len(trains)
    // corr = eye(n)
    // for i in range(n) {
    // for j in range(i + 1, n) {
    // std_i = residuals[i].std()
    // std_j = residuals[j].std()
    // if std_i > 0 and std_j > 0 {
    // corr[i, j] = corr[j, i] = mean(residuals[i] * residuals[j]) / (std_i *
    // return corr
    0.0
}

pub fn signal_correlation(trains: f64, bin_size: f64) -> f64 {
    // trains: list[ndarray[Any, Any]], bin_size: int = 50
    // ) -> ndarray[Any, Any] {
    // binned = [bin_spike_train(t, bin_size).astype(float64) for t in trains
    // min_bins = min(b.size for b in binned)
    // mat = array([b[:min_bins] for b in binned])
    // return corrcoef(mat)
    0.0
}

pub fn spike_count_covariance(trains: f64, window: f64) -> f64 {
    // trains: list[ndarray[Any, Any]], window: int = 50
    // ) -> ndarray[Any, Any] {
    // return covariance_matrix(trains, bin_size=window)
    0.0
}

pub fn joint_psth(train_a: f64, train_b: f64, bin_size: f64) -> f64 {
    // train_a: ndarray[Any, Any], train_b: ndarray[Any, Any], bin_size: int
    // ) -> ndarray[Any, Any] {
    // ca = bin_spike_train(train_a, bin_size).astype(float64)
    // cb = bin_spike_train(train_b, bin_size).astype(float64)
    // n = min(ca.size, cb.size)
    // ca, cb = ca[:n], cb[:n]
    // ca -= ca.mean()
    // cb -= cb.mean()
    // return outer(ca, cb) / n
    0.0
}

pub fn coincidence_index(train_a: f64, train_b: f64, dt: f64, delta_ms: f64) -> f64 {
    // train_a: ndarray[Any, Any],
    // train_b: ndarray[Any, Any],
    // dt: float = 0.001,
    // delta_ms: float = 2.0,
    // ) -> float {
    // ta = spike_times(train_a, dt)
    // tb = spike_times(train_b, dt)
    // if ta.size == 0 or tb.size == 0 {
    // return 0.0
    // delta = delta_ms / 1000.0
    // duration = max(train_a.size, train_b.size) * dt
    // raw_coinc = 0
    // for t in ta {
    // if any((tb - t as f64).abs() <= delta) {
    // raw_coinc += 1
    // expected = 2.0 * delta * ta.size * tb.size / duration if duration > 0
    // norm = 0.5 * (ta.size + tb.size)
    // if norm <= expected {
    // return 0.0
    // return float((raw_coinc - expected) / (norm - expected))
    0.0
}

pub fn _tile_fraction(times: f64) -> f64 {
    // covered = 0.0
    // intervals: list[tuple[Any, Any]] = []
    // for t in times {
    // intervals.append((t - delta, t + delta))
    // intervals.sort()
    // merged = [intervals[0]]
    // for lo, hi in intervals[1:] {
    // if lo <= merged[-1][1] {
    // merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    // else {
    // merged.append((lo, hi))
    // for lo, hi in merged {
    // lo_c = max(lo, 0.0)
    // hi_c = min(hi, duration)
    // if hi_c > lo_c {
    // covered += hi_c - lo_c
    // return min(covered / duration, 1.0) if duration > 0 else 0.0
    0.0
}

pub fn _coincidence_fraction(times_ref: f64, times_target: f64) -> f64 {
    // times_ref: ndarray[Any, Any], times_target: ndarray[Any, Any]
    // ) -> float {
    // count = 0
    // for t in times_ref {
    // if any((times_target - t as f64).abs() <= delta) {
    // count += 1
    // return count / len(times_ref) if len(times_ref) > 0 else 0.0
    0.0
}

pub fn _sttc_term(p: f64, t: f64) -> f64 {
    // if abs(1.0 - t) < 1e-15 {
    // return 0.0
    // return (p - t) / (1.0 - p * t) if abs(1.0 - p * t) > 1e-15 else 0.0
    0.0
}
