# GIFPopulationNeuron

**Module:** `sc_neurocore.neurons.models.gif_population`
**Rust:** `sc_neurocore_engine::neurons::biophysical::GIFPopulationNeuron`
**Reference:** Mensi, Naud, Pozzorini, Avermann, Petersen & Gerstner (2012), *Journal of Neurophysiology* 107(6), 1756-1775.
**Family:** Generalized integrate-and-fire with stochastic escape-rate firing.
**State variables:** `v` membrane voltage, `eta` spike-triggered adaptation current.

## Contract implemented

`GIFPopulationNeuron` models the Mensi et al. generalized integrate-and-fire population mechanism with three production contracts:

1. The subthreshold `v, eta` flow is the exact fixed-current solution of the coupled linear equations over one time step.
2. Spikes are sampled from the exact Poisson interval probability `1 - exp(-lambda * dt)` using a bounded escape-rate hazard.
3. Seeded construction and reset make stochastic trajectories reproducible for tests, benchmarks, and audits.

## Equations

The continuous-time subthreshold system is:

```text
tau_m dV/dt = -(V - V_rest) - eta + I_ext
deta/dt = -eta / tau_eta
```

For constant input over `dt`, define:

```text
x = V - V_rest - I_ext
eta_decay = exp(-dt / tau_eta)
membrane_decay = exp(-dt / tau_m)
```

The implemented exact update is:

```text
eta_next = eta * eta_decay
```

For `tau_m != tau_eta`:

```text
x_next = x * membrane_decay
         - eta * tau_eta / (tau_eta - tau_m) * (eta_decay - membrane_decay)
```

For the equal-time-constant limit:

```text
x_next = membrane_decay * (x - eta * dt / tau_m)
```

Then:

```text
V_next = V_rest + I_ext + x_next
```

The spike hazard is:

```text
lambda(V) = lambda_0 * exp(clamp((V - theta) / delta_v, -745, 20))
P(spike) = clamp(1 - exp(-lambda(V) * dt), 0, 1)
```

On spike:

```text
V <- V_reset
eta <- eta + eta_increment
```

Invalid inputs or invalid runtime parameters fail closed: the method returns `0` and preserves the dynamic state.

## Default parameters

| Parameter | Default | Unit | Role |
| --- | ---: | --- | --- |
| `v` | -65.0 | mV | Membrane voltage |
| `theta` | -50.0 | mV | Escape-rate midpoint |
| `eta` | 0.0 | model current | Adaptation current |
| `tau_m` | 20.0 | ms | Membrane time constant |
| `tau_eta` | 100.0 | ms | Adaptation decay time constant |
| `delta_v` | 2.0 | mV | Escape-rate sharpness |
| `lambda_0` | 0.001 | ms^-1 | Baseline hazard |
| `eta_increment` | 5.0 | model current | Spike adaptation kick |
| `v_rest` | -65.0 | mV | Resting voltage |
| `v_reset` | -65.0 | mV | Reset voltage |
| `dt` | 0.5 | ms | Integration step |
| `seed` | 42 | integer | Reproducible random stream |

## Backend surfaces

| Surface | File | Status |
| --- | --- | --- |
| Python reference | `src/sc_neurocore/neurons/models/gif_population.py` | Exact subthreshold flow and seeded escape-rate sampling |
| Rust engine | `engine/src/neurons/biophysical.rs` | Same exact flow and bounded probability |
| PyO3 class | `sc_neurocore_engine.GIFPopulationNeuron(seed=42)` | Rust engine exposure |
| Go service | `src/sc_neurocore/accel/go/services/gif_population.go` | Same scalar contract with deterministic seed |
| Julia kernel | `src/sc_neurocore/accel/julia/neurons/gif_population.jl` | Same scalar contract |
| Rust safety mirror | `src/sc_neurocore/accel/rust/safety/gif_population.rs` | Fail-closed validation mirror |
| Mojo contract | `src/sc_neurocore/accel/mojo/kernels/gif_population.mojo` | Scalar contract notes for kernel promotion |

## Behavioural tests

The dedicated test file `tests/test_model_gif_population.py` checks:

- exact coupled subthreshold reference point,
- equal-time-constant limit,
- forced-spike adaptation kick,
- zero baseline hazard,
- invalid input and invalid parameter preservation,
- nonfinite candidate rejection,
- seeded reproducibility and seed separation,
- reset replay of the seeded stochastic stream,
- population construction for the model surface.

Only the dedicated model test file is used.

## Benchmark

The benchmark script is `benchmarks/bench_gif_population.py`. It records Python reference throughput, Rust PyO3 throughput when available, and a hyperpolarized silent-path parity check. The current evidence file is `benchmarks/results/bench_gif_population.json`.

Run:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_gif_population.py
```

## Minimal use

```python
from sc_neurocore.neurons.models.gif_population import GIFPopulationNeuron

neuron = GIFPopulationNeuron(seed=123)
spikes = [neuron.step(40.0) for _ in range(1000)]
state = {"v": neuron.v, "eta": neuron.eta}
```

## Production notes

- `tau_m`, `tau_eta`, `delta_v`, and `dt` must be positive.
- `lambda_0` must be nonnegative.
- Hazard exponent bounds prevent overflow and underflow from escaping the probability interval.
- `reset()` restores `v`, clears `eta`, and rewinds the seeded random stream.
