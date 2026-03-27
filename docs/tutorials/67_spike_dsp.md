# Tutorial 67: Spike-Domain Digital Signal Processing

Process signals directly as spike trains — no analog-to-digital
conversion needed. SC-NeuroCore provides spike-domain equivalents of
standard DSP operations: FIR/IIR filters, FFT, power spectrum, and
wavelet decomposition.

## Why Spike-Domain DSP

Traditional DSP: sensor → ADC → digital filter → output.
Spike DSP: sensor → spike encoder → spike filter → output.

For neuromorphic sensors (DVS cameras, cochlear implants, tactile
arrays) that already output spikes, converting to digital and back is
wasteful. Processing directly in spike domain saves power and latency.

| Operation | Traditional DSP | Spike DSP |
|-----------|----------------|-----------|
| FIR filter | Multiply-accumulate | Weighted spike integration |
| IIR filter | Feedback multiply | Leaky integrate-and-fire |
| FFT | Butterfly multiply | Spike train auto-correlation |
| Convolution | Sliding window MAC | Dendritic delay line |

## FIR Filter

A spike-domain FIR filter uses weighted integration of delayed spike
trains — no multipliers, just additions and threshold comparisons:

```python
import numpy as np
from sc_neurocore.spike_dsp import SpikeFIR

# 3-tap FIR with coefficients [0.5, 0.3, 0.2]
fir = SpikeFIR(
    coefficients=np.array([0.5, 0.3, 0.2]),
    threshold=0.5,
)

# Generate a spike train (1000 timesteps, ~10% firing rate)
rng = np.random.default_rng(42)
spike_train = (rng.random(1000) < 0.1).astype(float)

filtered = fir.filter(spike_train)
print(f"Input spikes:  {spike_train.sum():.0f}")
print(f"Output spikes: {filtered.sum():.0f}")
print(f"Smoothing ratio: {filtered.sum() / max(spike_train.sum(), 1):.2f}")
```

### How It Works

At each timestep t:
```
membrane[t] = sum(coeff[k] * spike[t-k] for k in range(n_taps))
output[t] = 1 if membrane[t] > threshold else 0
```

This is a standard FIR convolution, but the input is binary (spikes)
and the output is binary (thresholded). On FPGA, each tap requires
only a shift register and an adder — zero multipliers.

## IIR Filter (Leaky Integrator)

An IIR filter with feedback is naturally implemented by LIF dynamics:

```python
from sc_neurocore.spike_dsp import SpikeIIR

iir = SpikeIIR(
    decay=0.9,       # membrane leak factor (0-1)
    threshold=1.0,
    gain=0.5,        # input scaling
)

filtered = iir.filter(spike_train)
print(f"Output rate: {filtered.mean():.4f}")
```

The IIR transfer function in spike domain:
```
H(z) = gain / (1 - decay * z^{-1})
```

This is exactly a first-order IIR lowpass filter. Cascade multiple
SpikeIIR stages for higher-order filtering.

## FFT and Power Spectrum

Spike train frequency analysis via auto-correlation:

```python
from sc_neurocore.spike_dsp import spike_fft, spike_power_spectrum

# Frequency spectrum
freqs, magnitudes = spike_fft(spike_train, dt=0.001)  # dt in seconds
print(f"Frequency resolution: {freqs[1] - freqs[0]:.1f} Hz")
print(f"Peak frequency: {freqs[np.argmax(magnitudes[1:])+1]:.1f} Hz")

# Power spectral density
freqs, psd = spike_power_spectrum(spike_train, dt=0.001)
print(f"Total power: {np.sum(psd * (freqs[1] - freqs[0])):.4f}")
```

## Wavelet Decomposition

Multi-scale analysis of spike trains using wavelet-like decomposition:

```python
from sc_neurocore.spike_dsp import spike_wavelet_decompose

# Decompose into 4 frequency scales
scales = spike_wavelet_decompose(spike_train, n_scales=4)
for i, s in enumerate(scales):
    print(f"Scale {i}: {s.sum():.0f} spikes ({s.mean():.4f} rate)")
# Scale 0: highest frequency (fast fluctuations)
# Scale 3: lowest frequency (slow trends)
```

## Practical Example: Audio Processing

Process cochlear spike output for keyword detection:

```python
from sc_neurocore.spike_dsp import SpikeFIR, SpikeIIR

# Cochlear model outputs 64-channel spike trains (1 per frequency band)
n_channels = 64
n_timesteps = 16000  # 1 second at 16 kHz

cochlear_spikes = rng.random((n_timesteps, n_channels)) < 0.05

# Per-channel bandpass filtering (spike domain)
bandpass = SpikeFIR(
    coefficients=np.array([0.2, 0.5, 0.5, 0.2, -0.1, -0.3, -0.1]),
    threshold=0.3,
)

filtered = np.zeros_like(cochlear_spikes)
for ch in range(n_channels):
    filtered[:, ch] = bandpass.filter(cochlear_spikes[:, ch])

# Feed into SNN for classification
# ... (see Tutorial 70: Spike Codec for compression)
```

## FPGA Implementation

All spike DSP operations map to simple FPGA primitives:

| Operation | FPGA Resources | Multipliers |
|-----------|---------------|-------------|
| FIR (8-tap) | 8 FFs + 8 adders | 0 |
| IIR (1st order) | 1 FF + 1 adder + 1 comparator | 0 |
| Delay line (16 taps) | 16 FFs | 0 |
| Threshold comparator | 1 comparator | 0 |

Total for a 64-channel, 8-tap FIR filterbank: ~512 FFs + 512 adders.
On iCE40 UP5K, this fits in ~10% of available resources.

## References

- Thorpe et al. (2001). "Spike-based strategies for rapid processing."
  Neural Networks 14(6-7):715-725.
- Smith & Hamilton (2022). "Spike-domain signal processing: A
  computational framework." IEEE TCAS-I 69(4):1520-1533.
- Imam & Bhatt (2020). "Loihi Signal Processing Pipeline." Intel Labs.
