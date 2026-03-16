# Tutorial 23: Spike Train Analysis

SC-NeuroCore ships 125 spike train analysis functions across 22 modules,
covering the combined scope of Elephant, PySpike, SpikeInterface, and
NeuroTools. All functions are pure NumPy with zero external dependencies.

This tutorial walks through the major analysis categories using spike
trains generated from SC-NeuroCore neuron models.

## Recording Spike Trains from Neuron Models

```python
import numpy as np
from sc_neurocore import StochasticLIFNeuron

neuron = StochasticLIFNeuron(tau_mem=10.0, dt=1.0, v_threshold=1.0, noise_std=0.05)
train = np.zeros(10000)
for t in range(10000):
    spike = neuron.step(0.12)
    train[t] = float(spike)

dt = 0.001  # 1 ms timestep
print(f"Recorded {int(train.sum())} spikes in {len(train) * dt:.1f} s")
```

For the examples below we use Poisson generators for reproducibility:

```python
from sc_neurocore.analysis import homogeneous_poisson, gamma_process

train_a = homogeneous_poisson(rate_hz=40.0, duration_s=10.0, seed=0)
train_b = homogeneous_poisson(rate_hz=40.0, duration_s=10.0, seed=1)
train_regular = gamma_process(rate_hz=40.0, shape=5.0, duration_s=10.0, seed=2)
```

## Basic Statistics

```python
from sc_neurocore.analysis import (
    firing_rate, spike_count, cv_isi, cv2, fano_factor,
    local_variation, isi_entropy, hurst_exponent,
)

print(f"Rate: {firing_rate(train_a):.1f} Hz")
print(f"Count: {spike_count(train_a)}")
print(f"CV(ISI): {cv_isi(train_a):.3f}")        # ~1.0 for Poisson
print(f"CV2: {cv2(train_a):.3f}")                # ~1.0 for Poisson
print(f"Fano factor: {fano_factor(train_a):.3f}")
print(f"LV: {local_variation(train_a):.3f}")
print(f"ISI entropy: {isi_entropy(train_a):.2f} bits")
print(f"Hurst H: {hurst_exponent(train_a):.3f}")

# Compare with regular train
print(f"\nRegular train CV(ISI): {cv_isi(train_regular):.3f}")  # <1.0
print(f"Regular train Fano: {fano_factor(train_regular):.3f}")  # <1.0
```

## Comparing Spike Trains: Distance Metrics

```python
from sc_neurocore.analysis import (
    victor_purpura_distance, van_rossum_distance,
    spike_distance, spike_sync, isi_distance,
    spike_distance_matrix, spike_times,
)

# Extract spike times for time-based metrics
times_a = spike_times(train_a)
times_b = spike_times(train_b)

vp = victor_purpura_distance(times_a[:100], times_b[:100], cost_per_s=500.0)
vr = van_rossum_distance(train_a, train_b, tau_ms=10.0)
sd = spike_distance(times_a, times_b, t_start=0.0, t_end=10.0)
ss = spike_sync(times_a, times_b, t_start=0.0, t_end=10.0)

print(f"Victor-Purpura: {vp:.1f}")
print(f"Van Rossum: {vr:.3f}")
print(f"SPIKE-distance: {sd:.3f}")
print(f"SPIKE-sync: {ss:.3f}")
```

## Synchrony Analysis

```python
from sc_neurocore.analysis import (
    cross_correlation, spike_time_tiling_coefficient,
    event_synchronization, coincidence_index,
    pairwise_correlation, covariance_matrix,
)

cc, lags = cross_correlation(train_a, train_b, max_lag_ms=50.0)
sttc = spike_time_tiling_coefficient(train_a, train_b, delta_ms=5.0)
es = event_synchronization(train_a, train_b, tau_ms=5.0)
ci = coincidence_index(train_a, train_b, delta_ms=2.0)

print(f"Peak cross-correlation: {cc.max():.3f}")
print(f"STTC: {sttc:.3f}")
print(f"Event synchronization: {es:.3f}")
print(f"Coincidence index: {ci:.3f}")

# Population-level
trains = [train_a, train_b, train_regular]
corr_mat = pairwise_correlation(trains)
print(f"Correlation matrix shape: {corr_mat.shape}")
```

## Spectral Analysis

```python
from sc_neurocore.analysis import power_spectrum, spike_train_coherence

psd, freqs = power_spectrum(train_a)
coh, coh_freqs = spike_train_coherence(train_a, train_b)

# Find dominant frequency (skip DC)
if psd.size > 1:
    peak_idx = np.argmax(psd[1:]) + 1
    print(f"Dominant frequency: {freqs[peak_idx]:.1f} Hz")
    print(f"Peak PSD: {psd[peak_idx]:.4f}")
```

## LFP Coupling

```python
from sc_neurocore.analysis import (
    phase_locking_value, spike_field_coherence,
    spike_phase_histogram,
)

# Simulate an LFP oscillation at 10 Hz
t = np.arange(10000) * 0.001
lfp = np.sin(2 * np.pi * 10.0 * t)

plv = phase_locking_value(train_a, lfp)
sfc, sfc_freqs = spike_field_coherence(train_a, lfp)
phase_hist, phase_bins = spike_phase_histogram(train_a, lfp)

print(f"PLV: {plv:.3f}")
print(f"SFC peak: {sfc.max():.3f}")
print(f"Phase histogram bins: {len(phase_bins)}")
```

## Information Theory

```python
from sc_neurocore.analysis import (
    mutual_information, transfer_entropy,
    spike_train_entropy, kozachenko_leonenko_mi,
)

mi = mutual_information(train_a, train_b, bin_size=10)
te = transfer_entropy(train_a, train_b, bin_size=10, lag=1)
h = spike_train_entropy(train_a, bin_size=10, word_length=4)

print(f"Mutual information: {mi:.3f} bits")
print(f"Transfer entropy A->B: {te:.3f} bits")
print(f"Spike train entropy: {h:.3f} bits")

# Continuous MI estimator
binned_a = np.convolve(train_a, np.ones(50), mode='same')
binned_b = np.convolve(train_b, np.ones(50), mode='same')
kl_mi = kozachenko_leonenko_mi(binned_a[:1000], binned_b[:1000], k=3)
print(f"KL MI estimator: {kl_mi:.3f} nats")
```

## Causal Inference

```python
from sc_neurocore.analysis import (
    pairwise_granger_causality, partial_directed_coherence,
    directed_transfer_function,
)

gc_ab = pairwise_granger_causality(train_a, train_b, bin_size=10, order=5)
gc_b_to_a = pairwise_granger_causality(train_b, train_a, bin_size=10, order=5)
print(f"GC A->B: {gc_ab:.4f}")
print(f"GC B->A: {gc_b_to_a:.4f}")

# Multivariate directed connectivity
pdc = partial_directed_coherence(trains, bin_size=10, order=5, n_freqs=32)
dtf = directed_transfer_function(trains, bin_size=10, order=5, n_freqs=32)
print(f"PDC shape: {pdc.shape}")  # (n_neurons, n_neurons, n_freqs)
```

## Dimensionality Reduction

```python
from sc_neurocore.analysis import spike_train_pca, factor_analysis

projected, explained = spike_train_pca(trains, n_components=2, bin_size=20)
print(f"PCA projected shape: {projected.shape}")
print(f"Variance explained: {explained}")

loadings, uniquenesses = factor_analysis(trains, n_factors=2, bin_size=20)
print(f"FA loadings shape: {loadings.shape}")
```

## Decoding

```python
from sc_neurocore.analysis import (
    population_vector_decode, bayesian_decode,
    linear_discriminant_decode,
)

# Population vector decoding
preferred = np.array([0.0, np.pi / 2, np.pi])  # preferred directions
decoded = population_vector_decode(trains, preferred, window=50)
print(f"Decoded {decoded.size} time bins")

# Bayesian decoding
tuning = np.array([[10.0, 5.0, 2.0], [2.0, 8.0, 12.0]])  # 2 stimuli x 3 neurons
counts = np.array([8, 6, 3])
stim = bayesian_decode(counts, tuning)
print(f"MAP stimulus: {stim}")
```

## Pattern Detection (SPADE)

```python
from sc_neurocore.analysis import spade_detect

# Create correlated trains: inject shared spikes
rng = np.random.default_rng(42)
shared = (rng.random(10000) < 0.005).astype(float)
t1 = np.clip(train_a + shared, 0, 1)
t2 = np.clip(train_b + np.roll(shared, 3), 0, 1)
t3 = np.clip(train_regular + np.roll(shared, 6), 0, 1)

patterns = spade_detect(
    [t1, t2, t3],
    bin_ms=5.0, min_support=3, max_pattern_size=3,
    n_surrogates=50, alpha=0.05,
)
print(f"Detected {len(patterns)} significant patterns")
for p in patterns[:3]:
    print(f"  neurons={p['neurons']}, lags={p['lags']}, count={p['count']}, p={p['p_value']:.3f}")
```

## GPFA: Smooth Latent Trajectories

```python
from sc_neurocore.analysis import gpfa, gpfa_transform

result = gpfa(
    [t1, t2, t3],
    n_latents=2, bin_ms=20.0, max_iter=20,
)
traj = result["trajectories"]
print(f"Latent trajectories shape: {traj.shape}")
print(f"EM converged in {len(result['log_likelihoods'])} iterations")

# Project new data
new_traj = gpfa_transform([train_a, train_b, train_regular], result)
print(f"New trajectories shape: {new_traj.shape}")
```

## Significance Testing

```python
from sc_neurocore.analysis import (
    significance_bootstrap, surrogate_isi_shuffle,
    surrogate_dither,
)

# Bootstrap test for cross-correlation peak
def peak_cc(a, b):
    cc, _ = cross_correlation(a, b, max_lag_ms=20.0)
    return cc.max() if cc.size > 0 else 0.0

obs, pval = significance_bootstrap(peak_cc, train_a, train_b, n_surrogates=100)
print(f"Observed peak CC: {obs:.3f}, p-value: {pval:.3f}")

# Generate surrogates for custom tests
surr = surrogate_isi_shuffle(train_a, seed=0)
surr_dither = surrogate_dither(train_a, dither_ms=5.0, seed=0)
print(f"Original rate: {firing_rate(train_a):.1f} Hz")
print(f"ISI-shuffle rate: {firing_rate(surr):.1f} Hz")
print(f"Dither rate: {firing_rate(surr_dither):.1f} Hz")
```

## Temporal Pattern Analysis

```python
from sc_neurocore.analysis import (
    burst_detection, first_spike_latency,
    change_point_detection, response_onset,
)

bursts = burst_detection(train_a, max_isi_ms=10.0, min_spikes=3)
print(f"Detected {len(bursts)} bursts")
for start, end, count in bursts[:3]:
    print(f"  {start:.3f}-{end:.3f} s, {count} spikes")

latency = first_spike_latency(train_a)
print(f"First spike latency: {latency*1000:.1f} ms")

cps = change_point_detection(train_a, bin_size=50, threshold=3.0)
print(f"Change points at bins: {cps[:5]}")
```

## Waveform Analysis

```python
from sc_neurocore.analysis import (
    waveform_width, waveform_amplitude,
    waveform_halfwidth, waveform_pt_ratio,
)

# Synthetic extracellular waveform (30 kHz sampling)
t_wf = np.arange(60) / 30000.0
wf = -np.exp(-((t_wf - 0.0005) ** 2) / (2 * 0.0001 ** 2)) + \
      0.4 * np.exp(-((t_wf - 0.001) ** 2) / (2 * 0.0002 ** 2))

width = waveform_width(wf, dt=1.0 / 30000)
amp = waveform_amplitude(wf)
hw = waveform_halfwidth(wf, dt=1.0 / 30000)
pt = waveform_pt_ratio(wf)

print(f"Trough-to-peak width: {width * 1000:.2f} ms")
print(f"Amplitude: {amp:.3f}")
print(f"Halfwidth: {hw * 1000:.2f} ms")
print(f"P/T ratio: {pt:.2f}")
```

## Sorting Quality Metrics

```python
from sc_neurocore.analysis import (
    isi_violation_rate, presence_ratio,
    isolation_distance, snr,
)

viol = isi_violation_rate(train_a, refractory_ms=1.5)
pres = presence_ratio(train_a, n_bins=100)
print(f"ISI violation rate: {viol:.4f}")
print(f"Presence ratio: {pres:.2f}")

# Cluster quality (synthetic features)
cluster = rng.normal(0, 1, (50, 4))
noise = rng.normal(3, 1.5, (100, 4))
iso = isolation_distance(cluster, noise)
print(f"Isolation distance: {iso:.1f}")
```

## Next Steps

- [API Reference](../api/analysis.md) -- full 125-function listing with signatures
- [Neuron Model Selection](22_neuron_model_selection.md) -- choosing models for your simulation
- [Building Your First SNN](02_building_your_first_snn.md) -- network construction basics
