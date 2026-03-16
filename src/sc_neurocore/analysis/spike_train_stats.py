# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spike train statistics — standard neuroscience analysis toolkit.

Provides the metrics researchers expect when comparing SNN simulators:
CV (coefficient of variation), Fano factor, cross-correlation,
power spectral density, pairwise correlation, and PSTH.

All functions accept spike trains as 1-D numpy arrays of 0/1 (binary)
or spike-time arrays (float, in seconds).
"""

from __future__ import annotations

import numpy as np


def spike_times(binary_train: np.ndarray, dt: float = 0.001) -> np.ndarray:
    """Extract spike times (seconds) from a binary 0/1 array."""
    return np.where(binary_train > 0)[0] * dt


def isi(binary_train: np.ndarray, dt: float = 0.001) -> np.ndarray:
    """Inter-spike intervals (seconds) from a binary train."""
    times = spike_times(binary_train, dt)
    if times.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(times)


def firing_rate(binary_train: np.ndarray, dt: float = 0.001) -> float:
    """Mean firing rate (Hz)."""
    duration = binary_train.size * dt
    if duration <= 0:
        return 0.0
    return float(np.sum(binary_train) / duration)


def cv_isi(binary_train: np.ndarray, dt: float = 0.001) -> float:
    """Coefficient of variation of ISI. CV=1 for Poisson, <1 for regular."""
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return float("nan")
    mu = intervals.mean()
    if mu == 0:
        return float("nan")
    return float(intervals.std() / mu)


def fano_factor(
    binary_train: np.ndarray, window_ms: float = 50.0, dt: float = 0.001
) -> float:
    """Fano factor: variance/mean of spike counts in sliding windows."""
    window_steps = max(1, int(window_ms / (dt * 1000)))
    n = binary_train.size
    if n < window_steps:
        return float("nan")
    n_windows = n // window_steps
    counts = binary_train[: n_windows * window_steps].reshape(n_windows, window_steps).sum(axis=1)
    mu = counts.mean()
    if mu == 0:
        return float("nan")
    return float(counts.var() / mu)


def spike_count(binary_train: np.ndarray) -> int:
    """Total spike count."""
    return int(np.sum(binary_train))


def psth(
    trials: list[np.ndarray], bin_ms: float = 10.0, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Peri-stimulus time histogram across trials.

    Returns (rates_hz, bin_centers_ms).
    """
    if not trials:
        return np.array([]), np.array([])
    max_len = max(t.size for t in trials)
    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    n_bins = max_len // bin_steps
    if n_bins == 0:
        return np.array([]), np.array([])
    counts = np.zeros(n_bins, dtype=np.float64)
    for trial in trials:
        trimmed = trial[: n_bins * bin_steps]
        if trimmed.size == 0:
            continue
        reshaped = trimmed.reshape(-1, bin_steps) if trimmed.size >= bin_steps else trimmed[None, :]
        if reshaped.shape[0] <= n_bins:
            counts[: reshaped.shape[0]] += reshaped.sum(axis=1)
    rates = counts / (len(trials) * bin_ms / 1000.0)
    centers = (np.arange(n_bins) + 0.5) * bin_ms
    return rates, centers


def cross_correlation(
    train_a: np.ndarray, train_b: np.ndarray, max_lag_ms: float = 50.0, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlogram between two binary spike trains.

    Returns (correlation, lags_ms).
    """
    max_lag = int(max_lag_ms / (dt * 1000))
    n = min(train_a.size, train_b.size)
    a = train_a[:n].astype(np.float64) - train_a[:n].mean()
    b = train_b[:n].astype(np.float64) - train_b[:n].mean()
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.zeros(len(lags), dtype=np.float64)
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if norm == 0:
        return cc, lags * dt * 1000
    for i, lag in enumerate(lags):
        if lag >= 0:
            cc[i] = np.sum(a[: n - lag] * b[lag:n])
        else:
            cc[i] = np.sum(a[-lag:n] * b[: n + lag])
    cc /= norm
    return cc, lags * dt * 1000


def pairwise_correlation(trains: list[np.ndarray], dt: float = 0.001) -> np.ndarray:
    """Pairwise Pearson correlation matrix across neurons."""
    n = len(trains)
    if n == 0:
        return np.array([[]])
    min_len = min(t.size for t in trains)
    mat = np.array([t[:min_len].astype(np.float64) for t in trains])
    return np.corrcoef(mat)


def power_spectrum(
    binary_train: np.ndarray, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density of a binary spike train.

    Returns (psd, freqs_hz).
    """
    n = binary_train.size
    if n < 2:
        return np.array([]), np.array([])
    x = binary_train.astype(np.float64) - binary_train.mean()
    fft_vals = np.fft.rfft(x)
    psd = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return psd, freqs


def burst_detection(
    binary_train: np.ndarray, dt: float = 0.001, max_isi_ms: float = 10.0, min_spikes: int = 3
) -> list[tuple[float, float, int]]:
    """Detect bursts: consecutive spikes with ISI < max_isi_ms.

    Returns list of (start_time_s, end_time_s, spike_count).
    """
    times = spike_times(binary_train, dt)
    if times.size < min_spikes:
        return []
    max_isi = max_isi_ms / 1000.0
    intervals = np.diff(times)
    bursts = []
    i = 0
    while i < intervals.size:
        if intervals[i] < max_isi:
            start = i
            while i < intervals.size and intervals[i] < max_isi:
                i += 1
            n_spikes = i - start + 1
            if n_spikes >= min_spikes:
                bursts.append((float(times[start]), float(times[i]), n_spikes))
        else:
            i += 1
    return bursts


# ── Instantaneous rate via kernel density ──────────────────────────


def instantaneous_rate(
    binary_train: np.ndarray,
    dt: float = 0.001,
    kernel: str = "gaussian",
    sigma_ms: float = 10.0,
) -> np.ndarray:
    """Instantaneous firing rate via kernel convolution (Hz).

    Kernels: 'gaussian', 'exponential', 'rectangular'.
    """
    n = binary_train.size
    sigma_steps = max(1, int(sigma_ms / (dt * 1000)))
    if kernel == "gaussian":
        hw = 3 * sigma_steps
        x = np.arange(-hw, hw + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / sigma_steps) ** 2)
    elif kernel == "exponential":
        hw = 5 * sigma_steps
        x = np.arange(0, hw, dtype=np.float64)
        k = np.exp(-x / sigma_steps)
    elif kernel == "rectangular":
        hw = sigma_steps
        k = np.ones(2 * hw + 1, dtype=np.float64)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    k /= k.sum() * dt
    return np.convolve(binary_train.astype(np.float64), k, mode="same")


# ── Spike train distance metrics ──────────────────────────────────


def van_rossum_distance(
    train_a: np.ndarray, train_b: np.ndarray, dt: float = 0.001, tau_ms: float = 10.0
) -> float:
    """Van Rossum 2001 — exponential-kernel spike train distance."""
    tau = tau_ms / 1000.0
    n = min(train_a.size, train_b.size)
    t = np.arange(n) * dt
    decay = np.exp(-t / tau) if tau > 0 else np.zeros(n)
    fa = np.convolve(train_a[:n].astype(np.float64), decay[:n], mode="full")[:n]
    fb = np.convolve(train_b[:n].astype(np.float64), decay[:n], mode="full")[:n]
    return float(np.sqrt(np.sum((fa - fb) ** 2) * dt / tau))


def victor_purpura_distance(
    times_a: np.ndarray, times_b: np.ndarray, cost_per_s: float = 1000.0
) -> float:
    """Victor-Purpura 1996 — edit distance between spike time arrays.

    cost_per_s: cost of shifting a spike by 1 second (q parameter).
    """
    na, nb = len(times_a), len(times_b)
    if na == 0:
        return float(nb)
    if nb == 0:
        return float(na)
    d = np.zeros((na + 1, nb + 1), dtype=np.float64)
    for i in range(na + 1):
        d[i, 0] = float(i)
    for j in range(nb + 1):
        d[0, j] = float(j)
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            shift_cost = cost_per_s * abs(times_a[i - 1] - times_b[j - 1])
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + shift_cost)
    return float(d[na, nb])


def isi_distance(train_a: np.ndarray, train_b: np.ndarray, dt: float = 0.001) -> float:
    """ISI-distance (Kreuz et al. 2007) — ratio-based ISI comparison."""
    isi_a = isi(train_a, dt)
    isi_b = isi(train_b, dt)
    n = min(isi_a.size, isi_b.size)
    if n == 0:
        return float("nan")
    ratios = np.zeros(n)
    for i in range(n):
        a, b = isi_a[i], isi_b[i]
        if a == 0 and b == 0:
            ratios[i] = 0.0
        elif a <= b:
            ratios[i] = a / b - 1.0 if b > 0 else 0.0
        else:
            ratios[i] = -(b / a - 1.0) if a > 0 else 0.0
    return float(np.abs(ratios).mean())


# ── Regularity / variability measures ─────────────────────────────


def cv2(binary_train: np.ndarray, dt: float = 0.001) -> float:
    """Local coefficient of variation CV2. Holt et al. 1996.

    CV2 = mean(2|ISI_{i+1} - ISI_i| / (ISI_{i+1} + ISI_i)).
    Less sensitive to firing rate changes than global CV.
    """
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return float("nan")
    diffs = np.abs(np.diff(intervals))
    sums = intervals[:-1] + intervals[1:]
    valid = sums > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(2.0 * diffs[valid] / sums[valid]))


def local_variation(binary_train: np.ndarray, dt: float = 0.001) -> float:
    """Local variation LV. Shinomoto et al. 2003.

    LV = (3/(N-1)) * sum((ISI_i - ISI_{i+1})^2 / (ISI_i + ISI_{i+1})^2).
    LV=1 for Poisson, <1 for regular, >1 for bursty.
    """
    intervals = isi(binary_train, dt)
    n = intervals.size
    if n < 2:
        return float("nan")
    diffs = np.diff(intervals)
    sums = intervals[:-1] + intervals[1:]
    valid = sums > 0
    if not valid.any():
        return float("nan")
    return float(3.0 / (n - 1) * np.sum((diffs[valid] / sums[valid]) ** 2))


def isi_entropy(binary_train: np.ndarray, dt: float = 0.001, bins: int = 20) -> float:
    """Shannon entropy of the ISI distribution (bits).

    Higher entropy = more irregular. Poisson has maximum entropy
    for a given rate.
    """
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return float("nan")
    hist, _ = np.histogram(intervals, bins=bins, density=True)
    hist = hist[hist > 0]
    bin_width = (intervals.max() - intervals.min()) / bins
    if bin_width <= 0:
        return 0.0
    p = hist * bin_width
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ── Synchrony measures ────────────────────────────────────────────


def event_synchronization(
    train_a: np.ndarray, train_b: np.ndarray, dt: float = 0.001, tau_ms: float = 5.0
) -> float:
    """Quian Quiroga et al. 2002 — event synchronization.

    Returns synchrony score in [0, 1]. Based on coincidence counting
    within adaptive windows.
    """
    ta = spike_times(train_a, dt)
    tb = spike_times(train_b, dt)
    na, nb = ta.size, tb.size
    if na == 0 or nb == 0:
        return 0.0
    tau = tau_ms / 1000.0
    count = 0
    for i in range(na):
        for j in range(nb):
            if abs(ta[i] - tb[j]) < tau:
                count += 1
    return float(count / (na * nb) ** 0.5)


def spike_train_coherence(
    train_a: np.ndarray, train_b: np.ndarray, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude-squared coherence between two binary spike trains.

    Returns (coherence, freqs_hz).
    """
    n = min(train_a.size, train_b.size)
    if n < 2:
        return np.array([]), np.array([])
    a = train_a[:n].astype(np.float64) - train_a[:n].mean()
    b = train_b[:n].astype(np.float64) - train_b[:n].mean()
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    pab = fa * np.conj(fb)
    paa = np.abs(fa) ** 2
    pbb = np.abs(fb) ** 2
    denom = paa * pbb
    denom[denom == 0] = 1e-30
    coh = np.abs(pab) ** 2 / denom
    freqs = np.fft.rfftfreq(n, d=dt)
    return coh, freqs


# ── Latency analysis ──────────────────────────────────────────────


def first_spike_latency(
    binary_train: np.ndarray, dt: float = 0.001
) -> float:
    """Time to first spike (seconds). Returns nan if no spike."""
    idx = np.argmax(binary_train > 0)
    if binary_train[idx] == 0:
        return float("nan")
    return float(idx * dt)


def response_onset(
    binary_train: np.ndarray,
    baseline_steps: int = 100,
    dt: float = 0.001,
    threshold_sigma: float = 3.0,
) -> float:
    """Detect response onset as first bin exceeding baseline + threshold_sigma * std.

    Returns onset time (seconds), or nan if no response detected.
    """
    if binary_train.size <= baseline_steps:
        return float("nan")
    baseline_rate = binary_train[:baseline_steps].mean()
    baseline_std = binary_train[:baseline_steps].std()
    if baseline_std == 0:
        baseline_std = 1e-10
    threshold = baseline_rate + threshold_sigma * baseline_std
    post = binary_train[baseline_steps:]
    above = np.where(post > threshold)[0]
    if above.size == 0:
        return float("nan")
    return float((baseline_steps + above[0]) * dt)


# ── Spike-triggered analysis ──────────────────────────────────────


def spike_triggered_average(
    stimulus: np.ndarray, binary_train: np.ndarray, window_steps: int = 50
) -> np.ndarray:
    """Spike-triggered average (STA) of a stimulus signal.

    Returns the average stimulus snippet preceding each spike.
    """
    times = np.where(binary_train > 0)[0]
    valid = times[times >= window_steps]
    if valid.size == 0:
        return np.zeros(window_steps, dtype=np.float64)
    snippets = np.array([stimulus[t - window_steps : t] for t in valid])
    return snippets.mean(axis=0)


# ── Binned representation ─────────────────────────────────────────


def bin_spike_train(
    binary_train: np.ndarray, bin_size: int = 10
) -> np.ndarray:
    """Bin a binary spike train into spike counts per bin."""
    n = binary_train.size
    n_bins = n // bin_size
    if n_bins == 0:
        return np.array([int(binary_train.sum())])
    trimmed = binary_train[: n_bins * bin_size]
    return trimmed.reshape(n_bins, bin_size).sum(axis=1)


def population_rate(
    trains: list[np.ndarray], dt: float = 0.001, sigma_ms: float = 10.0
) -> np.ndarray:
    """Population-level instantaneous firing rate (Hz).

    Sums all trains then applies Gaussian kernel smoothing.
    """
    if not trains:
        return np.array([])
    min_len = min(t.size for t in trains)
    total = np.zeros(min_len, dtype=np.float64)
    for t in trains:
        total += t[:min_len].astype(np.float64)
    return instantaneous_rate(total, dt=dt, kernel="gaussian", sigma_ms=sigma_ms)


# ── Surrogate generation (significance testing) ───────────────────


def surrogate_isi_shuffle(
    binary_train: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Generate surrogate by shuffling ISIs. Preserves rate + ISI distribution."""
    intervals = np.diff(np.where(binary_train > 0)[0])
    if intervals.size < 2:
        return binary_train.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(intervals)
    out = np.zeros_like(binary_train)
    idx = np.where(binary_train > 0)[0][0]
    out[idx] = 1
    for gap in intervals:
        idx += gap
        if idx < out.size:
            out[idx] = 1
    return out


def surrogate_dither(
    binary_train: np.ndarray, dither_ms: float = 5.0, dt: float = 0.001, seed: int = 0
) -> np.ndarray:
    """Generate surrogate by jittering each spike time ±dither_ms."""
    rng = np.random.default_rng(seed)
    dither_steps = int(dither_ms / (dt * 1000))
    times = np.where(binary_train > 0)[0]
    out = np.zeros_like(binary_train)
    for t in times:
        jittered = t + rng.integers(-dither_steps, dither_steps + 1)
        jittered = np.clip(jittered, 0, out.size - 1)
        out[jittered] = 1
    return out


def surrogate_trial_shuffle(
    trains: list[np.ndarray], seed: int = 0
) -> list[np.ndarray]:
    """Shuffle trial order. Destroys trial-to-trial correlation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trains))
    return [trains[i] for i in idx]


# ── Information theory ────────────────────────────────────────────


def mutual_information(
    train_a: np.ndarray, train_b: np.ndarray, bin_size: int = 10
) -> float:
    """Mutual information between two binned spike trains (bits).

    MI = H(A) + H(B) - H(A,B) using binned spike counts.
    """
    ca = bin_spike_train(train_a, bin_size)
    cb = bin_spike_train(train_b, bin_size)
    n = min(ca.size, cb.size)
    ca, cb = ca[:n], cb[:n]

    def _entropy(x):
        vals, counts = np.unique(x, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log2(p + 1e-30)))

    ha = _entropy(ca)
    hb = _entropy(cb)
    joint = ca * (cb.max() + 1) + cb
    hab = _entropy(joint)
    return max(0.0, ha + hb - hab)


def transfer_entropy(
    source: np.ndarray, target: np.ndarray, bin_size: int = 10, lag: int = 1
) -> float:
    """Transfer entropy from source to target spike train (bits).

    TE = H(target_future | target_past) - H(target_future | target_past, source_past)
    """
    cs = bin_spike_train(source, bin_size)
    ct = bin_spike_train(target, bin_size)
    n = min(cs.size, ct.size)
    if n <= lag:
        return 0.0
    cs, ct = cs[:n], ct[:n]
    t_past = ct[:-lag]
    t_future = ct[lag:]
    s_past = cs[:-lag]
    n_pts = t_past.size

    def _cond_entropy(future, *pasts):
        joint = future.copy()
        for p in pasts:
            joint = joint * (p.max() + 1) + p
        vals, counts = np.unique(joint, return_counts=True)
        h_joint = -np.sum(counts / n_pts * np.log2(counts / n_pts + 1e-30))
        past_joint = pasts[0].copy()
        for p in pasts[1:]:
            past_joint = past_joint * (p.max() + 1) + p
        vals2, counts2 = np.unique(past_joint, return_counts=True)
        h_past = -np.sum(counts2 / n_pts * np.log2(counts2 / n_pts + 1e-30))
        return h_joint - h_past

    h1 = _cond_entropy(t_future, t_past)
    h2 = _cond_entropy(t_future, t_past, s_past)
    return max(0.0, float(h1 - h2))


# ── LFP coupling ─────────────────────────────────────────────────


def phase_locking_value(
    binary_train: np.ndarray, lfp_signal: np.ndarray
) -> float:
    """Phase locking value (PLV) between spikes and LFP phase.

    Extracts instantaneous phase of LFP via Hilbert transform,
    then computes PLV = |mean(exp(j*phase_at_spikes))|.
    """
    n = min(binary_train.size, lfp_signal.size)
    analytic = np.fft.ifft(
        np.fft.fft(lfp_signal[:n].astype(np.float64))
        * 2 * (np.arange(n) > 0).astype(np.float64)
    )
    phase = np.angle(analytic)
    spike_idx = np.where(binary_train[:n] > 0)[0]
    if spike_idx.size == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * phase[spike_idx]))))


def spike_field_coherence(
    binary_train: np.ndarray, lfp_signal: np.ndarray, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Spike-field coherence (SFC) between binary train and LFP.

    Returns (coherence, freqs_hz). SFC = |S_xy|^2 / (S_xx * S_yy).
    """
    n = min(binary_train.size, lfp_signal.size)
    if n < 2:
        return np.array([]), np.array([])
    a = binary_train[:n].astype(np.float64) - binary_train[:n].mean()
    b = lfp_signal[:n].astype(np.float64) - lfp_signal[:n].mean()
    fa, fb = np.fft.rfft(a), np.fft.rfft(b)
    sab = fa * np.conj(fb)
    saa = np.abs(fa) ** 2
    sbb = np.abs(fb) ** 2
    denom = saa * sbb
    denom[denom == 0] = 1e-30
    sfc = np.abs(sab) ** 2 / denom
    return sfc, np.fft.rfftfreq(n, d=dt)


def spike_phase_histogram(
    binary_train: np.ndarray, lfp_signal: np.ndarray, n_bins: int = 36
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of LFP phase at spike times.

    Returns (counts, bin_centers_rad) with bins spanning [-pi, pi].
    """
    n = min(binary_train.size, lfp_signal.size)
    analytic = np.fft.ifft(
        np.fft.fft(lfp_signal[:n].astype(np.float64))
        * 2 * (np.arange(n) > 0).astype(np.float64)
    )
    phase = np.angle(analytic)
    spike_phases = phase[binary_train[:n] > 0]
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(spike_phases, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return hist, centers


# ── Dimensionality reduction ─────────────────────────────────────


def spike_train_pca(
    trains: list[np.ndarray], n_components: int = 3, bin_size: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """PCA on binned spike count matrix (neurons x time_bins).

    Returns (projected, explained_variance_ratio).
    """
    if not trains:
        return np.array([[]]), np.array([])
    binned = np.array([bin_spike_train(t, bin_size).astype(np.float64) for t in trains])
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    mat -= mat.mean(axis=1, keepdims=True)
    cov = np.cov(mat)
    if cov.ndim < 2:
        return mat[:1], np.array([1.0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    components = eigvecs[:, idx]
    projected = components.T @ mat
    total_var = eigvals.sum()
    explained = eigvals[idx] / total_var if total_var > 0 else eigvals[idx]
    return projected, explained


# ── Decoding ──────────────────────────────────────────────────────


def population_vector_decode(
    trains: list[np.ndarray],
    preferred_directions: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """Georgopoulos population vector decoding.

    Each neuron i has a preferred direction (angle in radians).
    Decoded direction per time bin = weighted sum of preferred directions.
    Returns decoded angles per time bin.
    """
    if not trains:
        return np.array([])
    min_len = min(t.size for t in trains)
    n_bins = min_len // window
    if n_bins == 0:
        return np.array([])
    decoded = np.zeros(n_bins)
    for b in range(n_bins):
        sx, sy = 0.0, 0.0
        for i, t in enumerate(trains):
            count = t[b * window : (b + 1) * window].sum()
            sx += count * np.cos(preferred_directions[i])
            sy += count * np.sin(preferred_directions[i])
        decoded[b] = np.arctan2(sy, sx)
    return decoded


# ── Network inference ─────────────────────────────────────────────


def functional_connectivity(
    trains: list[np.ndarray], max_lag_ms: float = 20.0, dt: float = 0.001
) -> np.ndarray:
    """Infer functional connectivity matrix from peak cross-correlation.

    Returns NxN matrix where entry (i,j) is max |cross-correlation|
    between neuron i and neuron j within ±max_lag.
    """
    n = len(trains)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mat[i, j] = 1.0
                continue
            cc, _ = cross_correlation(trains[i], trains[j], max_lag_ms=max_lag_ms, dt=dt)
            peak = np.abs(cc).max() if cc.size > 0 else 0.0
            mat[i, j] = mat[j, i] = peak
    return mat


# ── Statistical testing ───────────────────────────────────────────


def significance_bootstrap(
    statistic_func,
    train_a: np.ndarray,
    train_b: np.ndarray,
    n_surrogates: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap significance test for a pairwise statistic.

    Returns (observed_value, p_value).
    statistic_func(a, b) -> float.
    """
    observed = statistic_func(train_a, train_b)
    rng = np.random.default_rng(seed)
    combined = np.concatenate([train_a, train_b])
    n_a = train_a.size
    count_extreme = 0
    for _ in range(n_surrogates):
        perm = rng.permutation(combined.size)
        surr_a = combined[perm[:n_a]]
        surr_b = combined[perm[n_a:]]
        surr_val = statistic_func(surr_a, surr_b)
        if abs(surr_val) >= abs(observed):
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_surrogates + 1)
    return float(observed), float(p_value)
