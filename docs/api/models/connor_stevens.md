# ConnorStevensNeuron

**Module:** `sc_neurocore.neurons.models.connor_stevens`
**Rust:** `sc_neurocore_engine::neurons::biophysical::ConnorStevensNeuron`
**Reference:** Connor & Stevens (1971); Connor, Walter & McKown (1977)
**Family:** Biophysical conductance model with A-type potassium current
**State:** `v`, `m`, `h`, `n`, `a`, `b`

`ConnorStevensNeuron` models type-I excitability by adding a transient A-type
potassium current to Hodgkin-Huxley-style sodium and delayed-rectifier
potassium dynamics. The maintained runtime contract is candidate-first RK4 over
sub-steps; invalid state, parameters, current, rates, or candidates fail before
state mutation.

## Equations

Membrane voltage:

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_A - I_L + I$$

Ionic currents:

$$I_{Na} = g_{Na} m^3 h (V - E_{Na})$$
$$I_K = g_K n^4 (V - E_K)$$
$$I_A = g_A a^3 b (V - E_A)$$
$$I_L = g_L (V - E_L)$$

Sodium rates:

$$\alpha_m = \frac{0.38(V + 29.7)}{1 - \exp(-(V + 29.7)/10)}$$
$$\beta_m = 15.2 \exp(-(V + 54.7)/18)$$
$$\alpha_h = 0.266 \exp(-(V + 48)/20)$$
$$\beta_h = \frac{3.8}{1 + \exp(-(V + 18)/10)}$$

Delayed-rectifier potassium rates:

$$\alpha_n = \frac{0.02(V + 45.7)}{1 - \exp(-(V + 45.7)/10)}$$
$$\beta_n = 0.25 \exp(-(V + 55.7)/80)$$

A-type potassium gates:

$$a_\infty = \left(\frac{0.0761 \exp((V + 94.22)/31.84)}{1 + \exp((V + 1.17)/28.93)}\right)^{1/3}$$
$$\tau_a = 0.3632 + \frac{1.158}{1 + \exp((V + 55.96)/20.12)}$$
$$b_\infty = \frac{1}{(1 + \exp((V + 53.3)/14.54))^4}$$
$$\tau_b = 1.24 + \frac{2.678}{1 + \exp((V + 50)/16.027)}$$

Gate derivatives:

$$\frac{dm}{dt} = \alpha_m(1-m) - \beta_m m$$
$$\frac{dh}{dt} = \alpha_h(1-h) - \beta_h h$$
$$\frac{dn}{dt} = \alpha_n(1-n) - \beta_n n$$
$$\frac{da}{dt} = \frac{a_\infty-a}{\tau_a}, \qquad \frac{db}{dt} = \frac{b_\infty-b}{\tau_b}$$

## Numerical contract

Each public `step(current)` call advances one macro-step using:

- candidate-first fourth-order Runge-Kutta integration,
- `int(1.0 / max(dt, 0.001))` sub-steps, which is 100 with the default `dt=0.01`,
- singularity-safe rate evaluation for `alpha_m` and `alpha_n`,
- finite-domain checks before derivative evaluation,
- candidate gate envelope checks before mutation,
- upward threshold-crossing spike semantics: return `1` only when `v` crosses `v_threshold` from below.

The fail-closed mutation rule is important for adapter and hardware paths: if an
invalid current, corrupted state, invalid parameter, overflowing rate, or
non-finite candidate is encountered, the previous state remains unchanged. The
Python reference raises a typed exception on invalid runtime input. Non-throwing
mirrors return an error signal or no-spike sentinel while preserving state.

## Parameters

| Parameter | Default | Unit | Description |
|-----------|--------:|------|-------------|
| `v` | -68.0 | mV | Membrane potential |
| `m` | 0.01 | unitless | Sodium activation gate |
| `h` | 0.99 | unitless | Sodium inactivation gate |
| `n` | 0.1 | unitless | Delayed-rectifier potassium activation gate |
| `a` | 0.5 | unitless | A-type potassium activation gate |
| `b` | 0.1 | unitless | A-type potassium inactivation gate |
| `g_na` | 120.0 | mS/cm^2 | Sodium conductance |
| `g_k` | 20.0 | mS/cm^2 | Delayed-rectifier potassium conductance |
| `g_a` | 47.7 | mS/cm^2 | A-type potassium conductance |
| `g_l` | 0.3 | mS/cm^2 | Leak conductance |
| `e_na` | 55.0 | mV | Sodium reversal potential |
| `e_k` | -72.0 | mV | Delayed-rectifier potassium reversal potential |
| `e_a` | -75.0 | mV | A-type potassium reversal potential |
| `e_l` | -17.0 | mV | Leak reversal potential |
| `c_m` | 1.0 | uF/cm^2 | Membrane capacitance |
| `dt` | 0.01 | ms | RK4 sub-step size |
| `v_threshold` | 0.0 | mV | Spike detection threshold |

Valid conductances are finite and non-negative. `c_m` and `dt` must be finite
and positive. Constructor gate values must be finite and physically bounded.

## Maintained implementation surfaces

| Surface | Contract |
|---------|----------|
| Python reference | Candidate-first RK4, typed invalid-input exceptions, module-specific tests |
| Rust engine | Candidate-first RK4, no-spike sentinel on invalid runtime input, engine tests |
| Go service | Candidate-first RK4, `(spike, error)` step contract, module-specific tests |
| Julia mirror | Candidate-first RK4, `-1` invalid-input sentinel, direct local validation |
| Rust safety mirror | Candidate-first RK4, no-spike sentinel, standalone `rustc --test` coverage |
| Mojo notes | Non-executable parity notes documenting the state order and RK4 contract |

## Behavioural tests

Module-specific tests in `tests/test_model_connor_stevens.py` verify:

- default parameters and six-state reset semantics,
- finite long-run trajectories,
- deterministic traces,
- A-type conductance dominance and onset-delay behaviour,
- bounded voltage and gate trajectories,
- parameter sweeps for `g_a` and `g_na`,
- population/network/spike-stat integration wiring,
- independent RK4 reference parity for one macro-step,
- invalid constructor parameters and fail-closed runtime mutation boundaries.

Additional module-owned checks cover the Go service, Rust engine, Julia mirror,
and Rust safety mirror.

## Benchmarks

Measured locally on 2026-05-31 after RK4 hardening.

| Runtime | Benchmark | Median | Per step | Artefact |
|---------|-----------|-------:|---------:|----------|
| Python | 500 macro-steps x 5 repeats | 2.26 s per 500 steps | 4.52 ms | `benchmarks/results/local_i5_11600k_python_2026-05-31_connor_stevens.json` |
| Rust Criterion | `connor_stevens_1k_steps` | 71.13 ms per 1k steps | 71.13 us | `benchmarks/results/local_i5_11600k_criterion_2026-05-31_connor_stevens.json` |

The RK4 path is intentionally slower than the prior raw-Euler path because each
sub-step evaluates four derivative stages and rejects invalid candidates before
commit. The benchmark is retained as evidence of runtime cost, not as a target
for numerical shortcuts.

## Example

```python
from sc_neurocore.neurons.models.connor_stevens import ConnorStevensNeuron

neuron = ConnorStevensNeuron()
spikes = 0
for _ in range(500):
    spikes += neuron.step(10.0)
print(neuron.v, spikes)
```

## Notes

The A-type current remains the defining mechanism: `g_a=47.7` exceeds the
standard delayed-rectifier `g_k=20.0`, so transient potassium activation delays
threshold crossing near rheobase. Removing the A-current with `g_a=0.0` is still
supported for controlled comparisons and is covered by module-specific tests.
