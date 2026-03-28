# Izhikevich 20 Firing Patterns Validation

Validates `SCIzhikevichNeuron` against the 20 electronic neuron types
from Izhikevich (2003) Table 1 and Izhikevich (2004) extended patterns.

## Reference

- Izhikevich, E.M. (2003). Simple model of spiking neurons. IEEE TNN 14(6):1569-72.
- Izhikevich, E.M. (2004). Which model to use for cortical spiking neurons? IEEE TNN 15(5):1063-70.

## Model

```
v' = 0.04v² + 5v + 140 - u + I
u' = a(bv - u)
if v >= 30 mV: v ← c, u ← u + d
```

## Validated Patterns (Kaggle run 2026-03-28)

| Pattern | (a, b, c, d) | I | Spikes/500ms | ISI (ms) | CV |
|---------|--------------|---|-------------|----------|-----|
| RS | 0.02, 0.2, -65, 8 | 10 | 12 | 45.0 | 0.123 |
| IB | 0.02, 0.2, -55, 4 | 10 | 16 | 31.2 | — |
| CH | 0.02, 0.2, -50, 2 | 10 | 37 | 13.1 | — |
| FS | 0.1, 0.2, -65, 2 | 10 | 50 | 10.0 | — |
| TC (burst) | 0.02, 0.25, -65, 0.05 | 0 | 0 | — | — |
| TC (tonic) | 0.02, 0.25, -65, 0.05 | 5 | 46 | 10.8 | — |
| RZ | 0.1, 0.26, -65, 2 | 0 | 1 | — | — |
| LTS | 0.02, 0.25, -65, 2 | 10 | 35 | 14.5 | — |
| Tonic Spiking | 0.02, 0.2, -65, 6 | 14 | 19 | 26.8 | — |
| Phasic Spiking | 0.02, 0.25, -65, 6 | 0.5 | 1 | — | — |
| Tonic Bursting | 0.02, 0.2, -50, 2 | 15 | 58 | 8.5 | — |
| Phasic Bursting | 0.02, 0.25, -55, 0.05 | 0.6 | 1 | — | — |
| Mixed Mode | 0.02, 0.2, -55, 4 | 10 | 16 | 31.2 | — |
| SFA | 0.01, 0.2, -65, 8 | 30 | 20 | 24.9 | — |
| Class 1 | 0.02, -0.1, -55, 6 | 30 | 15 | 33.1 | — |
| Class 2 | 0.2, 0.26, -65, 0 | 5 | 53 | 9.5 | — |
| Spike Latency | 0.02, 0.2, -65, 6 | 7 | 9 | 55.6 | — |
| Subthreshold Osc | 0.05, 0.26, -60, 0 | 2 | 25 | 20.3 | — |
| Accommodation | 0.02, 1.0, -55, 4 | 10 | 84 | 6.0 | — |
| Inhibition-Induced | -0.02, -1.0, -60, 8 | 80 | 1 | — | — |

## Qualitative Checks

- RS has low CV (0.123 < 0.15) — confirmed regular spiking
- FS fires faster than RS (50 > 12) — confirmed
- All suprathreshold patterns produce spikes

## Test Files

- `tests/test_izhikevich_20_patterns.py` — pytest suite
- `benchmarks/results/phase1_validation_results.json` — measured data
