# NeuroGridNeuron

**Module:** `sc_neurocore.neurons.models.neurogrid`
**Reference:** Benjamin et al. 2014 / Boahen Neurogrid
**Family:** Hardware-inspired analog neuromorphic, two-compartment EIF
**State variables:** `v_s`, `v_d`

`NeuroGridNeuron` models a passive dendrite coupled to an EIF-like soma. The
production path uses candidate-first RK4 over the continuous two-state flow and
applies the discrete spike/reset rule once to the accepted public-step
candidate. Python keeps `integrator="baseline_euler"` as a regression-only
comparison path.

---

## Equations

Dendrite:

$$\tau_d \frac{dV_d}{dt} = -(V_d - V_r) + I - g_c(V_d - V_s)$$

Soma:

$$\tau_s \frac{dV_s}{dt} = -(V_s - V_r) + \Delta_T \exp\left(\frac{V_s - V_\theta}{\Delta_T}\right) + g_c(V_d - V_s)$$

For RK4 stages, the effective soma voltage is capped at `v_peak` before it is
used in the exponential and coupling terms. This prevents uncommitted
super-peak RK4 stages from injecting unphysical charge into the dendrite. After
the candidate is accepted, `v_s >= v_peak` emits a spike and resets only the
soma to `v_reset`.

---

## Parameters

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `v_s` | -65.0 | Soma voltage |
| `v_d` | -65.0 | Dendrite voltage |
| `tau_s` | 20.0 | Soma time constant |
| `tau_d` | 50.0 | Dendrite time constant |
| `g_c` | 0.5 | Inter-compartment coupling |
| `delta_t` | 2.0 | EIF exponential slope |
| `v_rest` | -65.0 | Resting voltage |
| `v_threshold` | -50.0 | EIF rheobase voltage |
| `v_peak` | 20.0 | Spike/reset threshold |
| `v_reset` | -65.0 | Post-spike soma reset |
| `dt` | 0.1 | Public-step integration interval |
| `integrator` | `"rk4"` | Python production path; `"baseline_euler"` is comparison-only |

---

## Implementation Surfaces

| Surface | File |
|---------|------|
| Python reference | `src/sc_neurocore/neurons/models/neurogrid.py` |
| Rust engine | `engine/src/neurons/hardware/neurogrid.rs` |
| Rust safety mirror | `src/sc_neurocore/accel/rust/safety/neurogrid.rs` |
| Go service | `src/sc_neurocore/accel/go/services/neurogrid.go` |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/neurogrid.jl` |
| Mojo kernel | `src/sc_neurocore/accel/mojo/kernels/neurogrid.mojo` |
| Benchmark driver | `benchmarks/bench_model_neurogrid.py` |
| Rust benchmark example | `engine/examples/bench_neurogrid_rk4.rs` |
| Python tests | `tests/test_model_neurogrid.py` |

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|------------------|
| Isolation | 7 | defaults, compartments, binary output, compartment evolution, long-run finite state, reset, determinism |
| Analytical | 7 | RK4 candidate commit, baseline Euler formula, coupling signs, EIF exponential, exponent cap, spike reset, dendritic drive |
| Compartments | 3 | dendrite slower than soma, dendritic accumulation, coupling independence when `g_c=0` |
| Dynamics | 4 | subthreshold silence, driven spiking, rate monotonicity, FI sweep finite state |
| Parameters | 4 | coupling, `delta_t`, `tau_d`, and `dt` sweeps |
| Performance | 2 | isolation and network throughput guardrails |
| Pipeline | 7 | Population, Projection, Network, spike_count, ISI, firing_rate, rate cross-check |
| RK4 hardening | 7 | default RK4, rejected integrator, RK4/Euler split, cross-backend anchor, invalid input/state/config preservation |
| **Total** | **55** | **All passed locally** |

Focused verification command:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_neurogrid.py -q
```

---

## Five-Backend Regression Benchmark

Command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_model_neurogrid.py
```

Artefact:
`benchmarks/results/local_python_2026-06-26_neurogrid_rk4.json`.

Pinned anchor:

| Backend | Steps | Current | Spike count |
|---------|------:|--------:|------------:|
| Python | 20,000 | 100.0 | 94 |
| Rust engine | 20,000 | 100.0 | 94 |
| Go | 20,000 | 100.0 | 94 |
| Julia | 20,000 | 100.0 | 94 |
| Mojo | 20,000 | 100.0 | 94 |

The Python baseline-Euler comparison path reports `93` spikes at the same
anchor. The RK4 value is the production cross-language contract.

Measured local regression results from
`benchmarks/results/local_python_2026-06-26_neurogrid_rk4.json`:

| Backend | Median ns/step | Min ns/step | Max ns/step | Spike anchor |
|---------|---------------:|------------:|------------:|-------------:|
| Python | 7,829.917 | 7,542.574 | 9,481.517 | 94 |
| Rust engine | 106.279 | 105.583 | 107.269 | 94 |
| Go | 205.200 | 183.700 | 209.500 | 94 |
| Julia | 139.799 | 132.648 | 141.947 | 94 |
| Mojo | spike-only | spike-only | spike-only | 94 |

Benchmark timing is local, non-isolated workstation context only. Use the
recorded medians as regression signals, not published throughput claims.

---

## Usage

```python
from sc_neurocore.neurons.models.neurogrid import NeuroGridNeuron

neuron = NeuroGridNeuron()
spikes = sum(neuron.step(100.0) for _ in range(20_000))

print(spikes)   # 94 under the pinned RK4 regression anchor
print(neuron.v_s, neuron.v_d)
```

Euler comparison:

```python
rk4 = NeuroGridNeuron()
euler = NeuroGridNeuron(integrator="baseline_euler")

print(sum(rk4.step(100.0) for _ in range(20_000)))    # 94
print(sum(euler.step(100.0) for _ in range(20_000)))  # 93
```

---

## Verification Snapshot

Measured 2026-06-26:

- Python model tests passed: 55 tests.
- Strict mypy passed on the Python model, tests, and benchmark driver.
- Ruff passed on the changed Python files.
- Go service tests passed.
- Rust safety standalone tests passed.
- Rust engine `neurogrid` tests passed.
- Julia and Mojo parity paths reported the pinned spike anchor.
- Five-backend benchmark artefact generated at
  `benchmarks/results/local_python_2026-06-26_neurogrid_rk4.json`.

---

## Citations

1. Benjamin BV, Gao P, McQuinn E, Choudhary S, Chandrasekaran AR, Bussat JM,
   Alvarez-Icaza R, Arthur JV, Merolla PA, Boahen K (2014). Neurogrid: A
   mixed-analog-digital multichip system for large-scale neural simulations.
   *Proceedings of the IEEE* 102(5):699-716.
   DOI: [10.1109/JPROC.2014.2313565](https://doi.org/10.1109/JPROC.2014.2313565)

**Model status:** production RK4 surface with Python/Rust/Go/Julia/Mojo parity
at the pinned regression anchor.
