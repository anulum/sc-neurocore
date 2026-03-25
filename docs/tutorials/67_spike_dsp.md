# Tutorial 67: Spike-Domain DSP

Process signals directly in spike domain — no ADC needed.

## FIR Filter

```python
from sc_neurocore.spike_dsp import SpikeFIR
fir = SpikeFIR(coefficients=np.array([0.5, 0.3, 0.2]), threshold=0.5)
filtered = fir.filter(spike_train)
```

## IIR Filter (Leaky Integrator)

```python
from sc_neurocore.spike_dsp import SpikeIIR
iir = SpikeIIR(decay=0.9, threshold=1.0, gain=0.5)
filtered = iir.filter(spike_train)
```

## FFT + Power Spectrum

```python
from sc_neurocore.spike_dsp import spike_fft, spike_power_spectrum
freqs, mags = spike_fft(spikes, dt=0.001)
freqs, psd = spike_power_spectrum(spikes, dt=0.001)
```

## Wavelet Decomposition

```python
from sc_neurocore.spike_dsp import spike_wavelet_decompose
scales = spike_wavelet_decompose(spikes, n_scales=4)
# scales[0] = highest frequency, scales[3] = lowest
```
