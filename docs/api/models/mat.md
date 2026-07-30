# Kobayashi MAT* non-resetting neuron

**Class:** `sc_neurocore.neurons.models.mat.MATNeuron`
**Source:** Kobayashi, Tsubo, and Shinomoto (2009),
[*Made-to-order spiking neuron model equipped with a multi-timescale adaptive threshold*](https://doi.org/10.3389/neuro.10.009.2009)

## Identity

`MATNeuron` is the paper's non-resetting MAT* point neuron. Voltage is measured
relative to rest. A spike changes the adaptive threshold and starts a 2 ms
absolute refractory interval, but it does not reset voltage. The historical
SC-NeuroCore reset/RK4 recurrence is retained separately as
[`SCResettingMATNeuron`](sc_resetting_mat.md).

## Maintained equations

The membrane follows Equation 1:

$$
\tau_m \frac{dV}{dt} = -V + RI.
$$

One maintained step uses the paper's 0.001 ms forward-Euler voltage update:

$$
V_{n+1}=V_n+\frac{\Delta t}{\tau_m}(-V_n+RI_n).
$$

The MAT* threshold is the baseline plus two spike-history memories:

$$
\theta(t)=\omega+\sum_k\left[
\alpha_1e^{-(t-t_k)/\tau_1}+\alpha_2e^{-(t-t_k)/\tau_2}
\right].
$$

The implementation carries those histories as `theta1` and `theta2`, decays
them exactly over each sampled interval, and adds `alpha_1` / `alpha_2` after an
eligible level event. During refractory, voltage and both histories continue to
evolve. A voltage still above threshold may emit again when refractory expires.
No within-step root localisation is claimed.

## Source profiles

| Profile | `omega` | `alpha_1` | `alpha_2` |
|---|---:|---:|---:|
| regular spiking (default) | 19 | 37 | 2 |
| intrinsically bursting | 26 | 1.7 | 2 |
| fast spiking | 11 | 10 | 0.002 |

All profiles retain `tau_m=5 ms`, `R=50 MOhm`, `tau_1=10 ms`,
`tau_2=200 ms`, `dt=0.001 ms`, and a 2 ms refractory interval unless explicitly
overridden. These are named examples from the paper, not universal cortical-cell
calibrations.

```python
from sc_neurocore.neurons.models.mat import MATNeuron

rs = MATNeuron.regular_spiking()
ib = MATNeuron.intrinsically_bursting()
fs = MATNeuron.fast_spiking()

events = [rs.step(0.7) for _ in range(5000)]
print(sum(events), rs.v, rs.threshold)
```

## Executable runtimes

| Lane | Surface | Enrolled parity |
|---|---|---|
| Python | `MATNeuron.step` | reference |
| Rust engine | `MATNeuron` and `py_mat_simulate` | complete trace within `2e-12` |
| Rust safety | `accel/rust/safety/mat.rs` | complete trace within `2e-12` |
| Julia | `MatAccel` through the dispatcher | complete trace within `2e-12` |
| Go | `accel/go/mat/libmat.so` | complete trace within `2e-12` |
| Mojo | `accel/mojo/mat/libmat.so` | complete trace within `2e-12` |

`sc_neurocore.accel.mat.simulate_mat(currents, backend="auto")` chooses the
fastest available complete runtime. Explicitly requesting an unavailable or
unknown backend fails; it never substitutes a surrogate.

## Reproducibility and hardware boundary

The independent 10,272-step source receipt is
`reference_trace_data/mat_2009_rs.json`. It records one event at index 3945 and
canonical trace SHA-256
`3382c1a73215026c1c1f41749cbc2061ff338c044e336e8a25ca59b8ac139de8`.
Paired TOML/JSON schemas reproduce the same discrete contract.

The hand-written signed Q32.32 RTL is a bounded hardware specialization. It is
cycle-exact to its independent integer oracle and preserves the Python event
vector over the enrolled trace. Yosys synthesis passes; the optimized netlist is
cycle-exact over that trace; and a depth-12 CVC5 job proves voltage, threshold,
refractory, reset, and event safety under bounded input. This is H1 evidence:
timing, PPA, device measurements, and universal binary64 equivalence are not
claimed.

The five-runtime 200,000-step result is
`benchmarks/results/bench_mat.json`. It is loaded-host local regression evidence,
not a production-speed or hardware-performance claim.

See [source and runtime fidelity evidence](../../validation/mat_source_fidelity.md).
