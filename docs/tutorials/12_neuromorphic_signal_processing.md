<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Neuromorphic Signal Processing

Process real-world signals (audio, sensor data, time series) using
SC-NeuroCore's stochastic bitstream pipeline. Neuromorphic signal
processing replaces DSP multiply-accumulate operations with single-gate
logic, achieving orders-of-magnitude power savings on FPGA.

**Prerequisites**: `pip install sc-neurocore scipy matplotlib`

## 1. SC as a signal processing primitive

Traditional DSP computes weighted sums with multipliers and adders.
In stochastic computing:

| DSP operation | SC equivalent | Gate count |
|---------------|---------------|------------|
| Multiply | AND gate | 1 |
| Weighted sum | MUX tree | N + 1 |
| Integration | Counter | log₂(L) |
| Filtering | Shift-register + AND | N + 1 |

SC trades precision for area: a 16-bit multiplier needs ~256 gates;
an SC multiplier (AND) needs 1. The precision scales as 1/√L with
bitstream length L.

## 2. Encode an audio signal

Convert a 1 kHz sine wave into a unipolar bitstream:

```python
import numpy as np
from sc_neurocore import BitstreamEncoder

# Generate 1 kHz sine at 16 kHz sample rate
fs = 16000
duration = 0.01  # 10 ms segment
t = np.arange(int(fs * duration)) / fs
signal = 0.5 + 0.4 * np.sin(2 * np.pi * 1000 * t)  # unipolar [0.1, 0.9]

L = 256  # bitstream length per sample
encoder = BitstreamEncoder(x_min=0.0, x_max=1.0, length=L, seed=42)

# Encode each sample as an L-bit stream
bitstreams = np.zeros((len(signal), L), dtype=np.uint8)
for i, s in enumerate(signal):
    bitstreams[i] = encoder.encode(float(s))

print(f"Signal: {len(signal)} samples")
print(f"Bitstreams: {bitstreams.shape} (samples × bits)")
print(f"Mean bit density: {bitstreams.mean():.3f} (expected: {signal.mean():.3f})")
```

## 3. SC low-pass filter

A first-order IIR low-pass filter in SC uses a shift register to
maintain state and an AND gate for multiplication:

```python
from sc_neurocore import BitstreamSynapse

def sc_lowpass(bitstreams, alpha=0.9):
    """First-order IIR filter: y[n] = α·y[n-1] + (1-α)·x[n].

    In SC: α is a bitstream, multiplication is AND.
    """
    n_samples, L = bitstreams.shape
    alpha_bits = (np.random.rand(L) < alpha).astype(np.uint8)
    output = np.zeros((n_samples, L), dtype=np.uint8)
    state = np.zeros(L, dtype=np.uint8)

    for n in range(n_samples):
        # y[n] = α·y[n-1] (AND gate)
        feedback = state & alpha_bits
        # (1-α)·x[n]: use inverted α mask
        new_input = bitstreams[n] & (~alpha_bits & 1)
        # Combine via OR (approximate addition for low duty cycles)
        output[n] = feedback | new_input
        state = output[n]

    return output

filtered = sc_lowpass(bitstreams, alpha=0.95)
decoded_filtered = filtered.mean(axis=1)
print(f"Filtered signal range: [{decoded_filtered.min():.3f}, {decoded_filtered.max():.3f}]")
```

## 4. SC FIR filter

An N-tap FIR filter uses N AND gates (one per tap) and a MUX tree
for summation:

```python
def sc_fir(bitstreams, taps):
    """SC FIR filter: y[n] = Σ h[k] · x[n-k].

    Each tap weight h[k] is a bitstream probability.
    Multiplication by AND, summation by MUX (random select).
    """
    n_samples, L = bitstreams.shape
    n_taps = len(taps)
    output = np.zeros((n_samples, L), dtype=np.uint8)

    for n in range(n_taps - 1, n_samples):
        # For each bit position, randomly select one tap (MUX)
        tap_select = np.random.randint(0, n_taps, size=L)
        for b in range(L):
            k = tap_select[b]
            tap_bits = (np.random.rand() < taps[k])
            output[n, b] = bitstreams[n - k, b] & int(tap_bits)

    return output

# 5-tap moving average (equal weights = 1/5 = 0.2)
taps = [0.2] * 5
fir_out = sc_fir(bitstreams, taps)
decoded_fir = fir_out.mean(axis=1)
print(f"FIR output mean: {decoded_fir[10:].mean():.3f}")
```

## 5. Spike-based edge detection

Detect edges (transients) in a signal using the difference between
fast and slow LIF neurons — the neuromorphic equivalent of a
band-pass filter:

```python
from sc_neurocore import StochasticLIFNeuron

fast_neuron = StochasticLIFNeuron(length=L)
slow_neuron = StochasticLIFNeuron(length=L)

fast_spikes = []
slow_spikes = []
edge_signal = []

# Test signal: step function with noise
test_signal = np.concatenate([
    np.full(50, 0.2),
    np.full(50, 0.8),
    np.full(50, 0.2),
    np.full(50, 0.8),
])
test_signal += np.random.normal(0, 0.02, len(test_signal))
test_signal = np.clip(test_signal, 0.01, 0.99)

for t in range(len(test_signal)):
    x = int(test_signal[t] * 255)
    fs, _ = fast_neuron.step(x_value=x)
    ss, _ = slow_neuron.step(x_value=max(1, x // 2))
    fast_spikes.append(fs)
    slow_spikes.append(ss)
    edge_signal.append(abs(fs - ss))

fast_rate = np.convolve(fast_spikes, np.ones(10) / 10, mode="same")
slow_rate = np.convolve(slow_spikes, np.ones(10) / 10, mode="same")
edge_rate = np.convolve(edge_signal, np.ones(10) / 10, mode="same")

print(f"Edge peaks at steps: {np.where(edge_rate > 0.3)[0]}")
```

## 6. Spectral analysis via spike trains

Extract frequency content from spike trains using inter-spike
interval (ISI) histograms:

```python
from scipy import signal as scipy_signal

# Generate a two-tone signal
fs = 16000
t = np.arange(8000) / fs
two_tone = 0.5 + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 880 * t)
two_tone = np.clip(two_tone, 0.01, 0.99)

# Encode and pass through LIF neuron
neuron = StochasticLIFNeuron(length=128)
spike_times = []
for step in range(len(two_tone)):
    spike, _ = neuron.step(x_value=int(two_tone[step] * 255))
    if spike:
        spike_times.append(step)

# ISI histogram → frequency estimate
isis = np.diff(spike_times)
if len(isis) > 10:
    freqs_hz = fs / isis  # convert ISI to frequency
    hist, bin_edges = np.histogram(freqs_hz, bins=100, range=(100, 2000))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_freq = bin_centers[hist.argmax()]
    print(f"Dominant ISI frequency: {peak_freq:.0f} Hz (expected: 440 Hz)")
```

## 7. Comparison: SC vs conventional DSP

```python
# Conventional FIR for reference
from scipy.signal import firwin, lfilter

# Design 5-tap low-pass at 2 kHz cutoff, 16 kHz sample rate
h = firwin(5, 2000, fs=16000)
conventional = lfilter(h, 1.0, signal)

# SC version (decode back to float for comparison)
sc_version = decoded_fir[:len(conventional)]

# NMSE
valid = slice(10, None)  # skip transient
nmse = np.mean((sc_version[valid] - conventional[valid]) ** 2) / np.var(conventional[valid])
print(f"SC vs conventional NMSE: {nmse:.4f}")
print(f"SC uses {5} AND gates + 1 MUX vs {5} multipliers + {4} adders")
```

## 8. Hardware cost comparison

| Component | Conventional (16-bit) | SC (L=256) |
|-----------|----------------------|------------|
| Multiplier | ~256 LUTs | 1 LUT (AND) |
| 5-tap FIR | ~1280 LUTs | ~10 LUTs |
| Power (est.) | ~50 mW | ~0.5 mW |
| Precision | 16-bit exact | ~8-bit effective |
| Latency | 1 cycle | L=256 cycles |

SC trades latency for area and power. For applications where
latency tolerance > 256 clock cycles (audio, slow sensors),
SC wins on power by 100x.

## 9. Real-time processing pipeline

Combine encoding, filtering, and spike-based feature extraction
into a streaming pipeline:

```python
class NeuromorphicDSP:
    """Streaming SC signal processing pipeline."""

    def __init__(self, n_filters=8, length=128):
        self.encoder = BitstreamEncoder(length=length, seed=0)
        self.length = length
        self.filters = [StochasticLIFNeuron(length=length) for _ in range(n_filters)]

    def process_sample(self, sample):
        """Process one sample, return n_filters spike outputs."""
        bits = self.encoder.encode(float(sample))
        x_val = int(np.clip(sample, 0, 1) * 255)
        return [n.step(x_value=x_val)[0] for n in self.filters]

pipeline = NeuromorphicDSP(n_filters=8, length=128)
features = []
for s in signal[:100]:
    features.append(pipeline.process_sample(s))

features = np.array(features)
print(f"Feature matrix: {features.shape} (samples × filters)")
print(f"Active filters per step: {features.sum(axis=1).mean():.1f}")
```

## What you learned

- SC replaces DSP multipliers with AND gates (1 LUT vs ~256)
- Low-pass IIR filter: shift register + AND + OR
- FIR filter: N AND gates + MUX tree
- Spike-based edge detection: fast LIF minus slow LIF
- ISI histograms extract frequency content from spike trains
- SC trades latency for 100x power reduction — ideal for always-on sensors
- Streaming pipeline: encode → filter → spike features

## Next steps

- Implement a full SC filterbank for audio preprocessing
- Deploy the FIR filter on FPGA and measure real power consumption
- Use neuromorphic features as input to an SC classifier (Tutorial 07)
- Compare SC filterbank against Mel filterbank for speech recognition
