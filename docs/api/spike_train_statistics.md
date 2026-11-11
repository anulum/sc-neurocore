# Spike-Train Statistics Core

`sc_neurocore.spike_train_statistics` holds nine spike-train measures that ship
in the wheel. NumPy is their only dependency. Shipped modules use it to measure
spike trains in an installed copy. The identity monitors
(`IdentitySubstrate.health_check`, `DirectorController.monitor`) are the first.

The wider toolkit, [`sc_neurocore.analysis`](analysis.md), ships only in a
source checkout. It re-exports these nine functions under its existing names, so
`sc_neurocore.analysis.firing_rate` and
`sc_neurocore.spike_train_statistics.firing_rate` are the same object. The core
imports nothing from `sc_neurocore.analysis`.

## Functions

Every function takes a binary spike train: a one-dimensional array with one
entry per time step, where a positive entry marks a spike. `dt` is the time step
in seconds.

| Function | Returns |
| --- | --- |
| `spike_times(train, dt)` | Spike times in seconds |
| `isi(train, dt)` | Inter-spike intervals in seconds; empty below two spikes |
| `firing_rate(train, dt)` | Mean rate in hertz; `0.0` for a zero-length train |
| `spike_count(train)` | Number of spikes |
| `bin_spike_train(train, bin_size)` | Spike count per complete bin of `bin_size` steps |
| `power_spectrum(train, dt)` | `(psd, freqs_hz)` of the mean-removed train |
| `cv_isi(train, dt)` | Coefficient of variation of the intervals; `nan` below two intervals |
| `fano_factor(train, window_ms, dt)` | Variance over mean of the window counts; `nan` for a silent or too short train |
| `permutation_entropy(train, order, delay)` | Bandt-Pompe entropy in `[0, 1]`; ties ranked by position; `order` and `delay` below 1 are refused |

```python
import numpy as np
from sc_neurocore.spike_train_statistics import cv_isi, firing_rate, permutation_entropy

train = (np.random.default_rng(0).random(1000) < 0.05).astype(np.int8)
print(firing_rate(train), cv_isi(train), permutation_entropy(train, order=3))
```

## Backends

The core is the NumPy implementation. In a source checkout where the Rust
`spike_stats_core` helper is built, `sc_neurocore.analysis.permutation_entropy`
calls the helper for valid requests. The tests check that the helper and the
core agree to a relative 1e-12 on binary trains.

## Reference

::: sc_neurocore.spike_train_statistics.basic

::: sc_neurocore.spike_train_statistics.spectral

::: sc_neurocore.spike_train_statistics.variability
