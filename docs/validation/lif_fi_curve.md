# LIF f-I Curve Validation

Validates `StochasticLIFNeuron` (with noise disabled) against the
analytical firing rate for a deterministic leaky integrate-and-fire neuron.

## Analytical Solution

For a LIF with membrane equation dv/dt = -(v - v_rest)/tau + R*I:

```
v_ss = v_rest + R * I * tau
ISI = tau * ln((v_ss - v_reset) / (v_ss - v_threshold))
f = 1 / (ISI + t_ref)
```

With normalised units (v_rest=0, v_th=1, v_reset=0, R=1, tau=20ms):
- Rheobase: I_rh = V_th / (R * tau) = 0.05
- ISI at I=0.1: 20 * ln(2) = 13.863 ms

## Reference

Gerstner, W. & Kistler, W.M. (2002). *Spiking Neuron Models*. Cambridge
University Press. Ch. 1.

## Measured Results (Kaggle run 2026-03-28, dt=0.1ms)

| Current | Analytical (sp/ms) | Simulated | Error |
|---------|-------------------|-----------|-------|
| 0.06 | 0.027906 | 0.027800 | 0.38% |
| 0.08 | 0.050977 | 0.051000 | 0.04% |
| 0.10 | 0.072135 | 0.071800 | 0.46% |
| 0.15 | 0.123315 | 0.123400 | 0.07% |
| 0.20 | 0.173803 | 0.172400 | 0.81% |
| 0.30 | 0.274241 | 0.270200 | 1.47% |
| 0.50 | 0.474561 | 0.454400 | 4.25% |

All within 5% tolerance. Error increases with current because higher
firing rates approach the dt=0.1ms discretisation limit.

## Additional Validations

- Subthreshold (I < 0.05): zero firing rate confirmed
- Rheobase (I = 0.05): rate < 0.01 confirmed
- Monotonicity: verified across 8 current levels
- Refractory period: reduces rate, matches analytical (<5%)
- Multiple tau (10, 20, 40 ms): all within 5%
- Native `StochasticLIFNeuron` matches Euler: 0.00% error

## Test Files

- `tests/test_lif_fi_curve.py` — pytest suite (dt=1ms for fast CI)
- `benchmarks/results/phase1_validation_results.json` — Kaggle measured data (dt=0.1ms)
