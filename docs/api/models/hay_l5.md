# HayL5PyramidalNeuron

**Module:** `sc_neurocore.neurons.models.hay_l5`
**Reference:** Hay, Hill, Schürmann, Markram & Segev 2011 (PLoS Comput Biol)
**Family:** Conductance-based multi-compartment model
**State variables:** `v_s`, `h_na`, `n_k`, `v_t`, `m_ca`, `h_ca`, `m_ih`, `v_a`, `ca_a`

`HayL5PyramidalNeuron` is SC-NeuroCore's maintained reduced
three-compartment Layer 5 thick-tufted pyramidal-cell model. It exposes soma,
apical-trunk, and apical-tuft voltages, fast somatic Na/K gating, trunk Ca/Ih
gating, and tuft calcium dynamics. The public API remains:

```python
from sc_neurocore.neurons.models.hay_l5 import HayL5PyramidalNeuron

neuron = HayL5PyramidalNeuron()
spike = neuron.step(current_soma=10.0, current_tuft=0.0)
```

The production integrator is candidate-first RK4 with four internal sub-steps
per public `step()` call. Python keeps `integrator="baseline_euler"` only as a
regression-comparison mode; Rust engine, Rust safety, Go, Julia, and Mojo use
the RK4 path.

---

## Equations

### Compartments

Soma:

$$C_m \frac{dV_s}{dt} = -I_{Na} - I_K - I_{L,s} - I_{s \to t} + I_s/p_s$$

Apical trunk:

$$C_m \frac{dV_t}{dt} = -I_{Ca,t} - I_{Ih} - I_{L,t} - I_{t \to s} - I_{t \to a}$$

Apical tuft:

$$C_m \frac{dV_a}{dt} = -I_{Ca,a} - I_{KCa} - I_{L,a} - I_{a \to t} + I_a/p_a$$

Tuft calcium:

$$\frac{dCa_a}{dt} = -f_{Ca} I_{Ca,a} - Ca_a/\tau_{Ca}$$

Spike detection is an upward soma crossing of `v_threshold = -30.0` mV.
Calcium candidates are clipped to `>= 0` before the next sub-step.

### Currents

| Current | Formula |
|---------|---------|
| `I_Na` | `g_na * m_na_inf^3 * h_na * (v_s - e_na)` |
| `I_K` | `g_k * n_k^4 * (v_s - e_k)` |
| `I_Ca,t` | `g_ca_t * m_ca^2 * h_ca * (v_t - e_ca)` |
| `I_Ih` | `g_ih * m_ih * (v_t - e_ih)` |
| `I_Ca,a` | `g_ca_a * m_ca_a_inf^2 * (v_a - e_ca)` |
| `I_KCa` | `g_kca * (ca_a / (ca_a + 0.001)) * (v_a - e_k)` |
| Leak | compartment-specific `g_l_* * (v - e_l)` |
| Coupling | `g_st` between soma/trunk, `g_ta` between trunk/tuft |

### Numerical Contract

- Four RK4 sub-steps per public call.
- Every RK4 stage evaluates all nine derivatives from one consistent candidate
  state.
- Inputs, parameters, and runtime state must be finite.
- `p_s`, `p_t`, `p_a`, `ca_decay`, `dt`, and `c_m` must be positive.
- Conductances and `f_ca` must be non-negative.
- The public state commits only after all four candidate sub-steps are finite.

---

## Parameters

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `v_s`, `v_t`, `v_a` | -75.0 | Soma, trunk, tuft voltages |
| `h_na` | 0.9 | Somatic Na inactivation |
| `n_k` | 0.1 | Somatic K activation |
| `m_ca` | 0.0 | Trunk Ca activation |
| `h_ca` | 1.0 | Trunk Ca inactivation |
| `m_ih` | 0.0 | Trunk Ih activation |
| `ca_a` | 0.0001 | Tuft calcium |
| `g_na` | 300.0 | Soma Na conductance |
| `g_k` | 40.0 | Soma K conductance |
| `g_l_s`, `g_l_t`, `g_l_a` | 0.03 | Compartment leak conductances |
| `g_ca_t` | 2.0 | Trunk Ca conductance |
| `g_ih` | 0.02 | Trunk Ih conductance |
| `g_ca_a` | 1.5 | Tuft Ca conductance |
| `g_kca` | 2.5 | Tuft Ca-activated K conductance |
| `g_st` | 1.5 | Soma-trunk coupling |
| `g_ta` | 0.8 | Trunk-tuft coupling |
| `p_s`, `p_t`, `p_a` | 0.15, 0.25, 0.60 | Compartment area fractions |
| `ca_decay` | 200.0 | Tuft calcium decay constant |
| `f_ca` | 0.0002 | Calcium influx coupling |
| `dt` | 0.025 | Internal sub-step timestep |
| `v_threshold` | -30.0 | Soma spike threshold |
| `c_m` | 1.0 | Membrane capacitance scale |
| `integrator` | `"rk4"` | Python production path; `"baseline_euler"` is comparison-only |

---

## Implementation Surfaces

| Surface | File |
|---------|------|
| Python reference | `src/sc_neurocore/neurons/models/hay_l5.py` |
| Rust engine | `engine/src/neurons/multi_compartment/hay_l5.rs` |
| Rust safety mirror | `src/sc_neurocore/accel/rust/safety/hay_l5.rs` |
| Go service | `src/sc_neurocore/accel/go/services/hay_l5.go` |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/hay_l5.jl` |
| Mojo kernel | `src/sc_neurocore/accel/mojo/kernels/hay_l5.mojo` |
| Benchmark driver | `benchmarks/bench_model_hay_l5.py` |
| Rust benchmark example | `engine/examples/bench_hay_l5_rk4.rs` |
| Python tests | `tests/test_model_hay_l5.py` |

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|------------------|
| Isolation | 7 | defaults, nine state variables, binary output, dual input, long-run finite state, reset, determinism |
| Analytical | 10 | sub-steps, compartment structure, area fractions, couplings, calcium, reversal ordering, channel groups |
| Compartment dynamics | 4 | somatic spiking, tuft depolarisation, soma-to-trunk propagation, all compartments evolve |
| Dynamics | 4 | somatic-drive spike, subthreshold silence, rate sweep, finite FI sweep |
| Parameters | 3 | Na, trunk Ca, and coupling sweeps |
| Performance | 2 | isolation and network throughput guardrails |
| Pipeline | 6 | Population, Projection, Network, spike_count, ISI, firing_rate |
| RK4 hardening | 8 | default RK4, rejected integrator, RK4/Euler regression split, cross-backend anchors, invalid input/state preservation, invalid `dt` rejection |
| **Total** | **53** | **All passed locally** |

Focused verification command:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_hay_l5.py -q
```

---

## Five-Backend Regression Benchmark

Command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_model_hay_l5.py
```

Artefact:
`benchmarks/results/local_python_2026-06-26_hay_l5_rk4.json`.

Pinned anchor:

| Backend | Steps | Current soma | Current tuft | Spike count |
|---------|------:|-------------:|-------------:|------------:|
| Python | 20,000 | 10.0 | 0.0 | 1 |
| Rust engine | 20,000 | 10.0 | 0.0 | 1 |
| Go | 20,000 | 10.0 | 0.0 | 1 |
| Julia | 20,000 | 10.0 | 0.0 | 1 |
| Mojo | 20,000 | 10.0 | 0.0 | 1 |

The dual-input anchor is `4` spikes at 20,000 steps with
`current_soma=5.0` and `current_tuft=5.0`; it is covered by the Python, Rust
engine, Rust safety, and Go tests.

Measured local regression results from
`benchmarks/results/local_python_2026-06-26_hay_l5_rk4.json`:

| Backend | Median ns/step | Min ns/step | Max ns/step | Spike anchor |
|---------|---------------:|------------:|------------:|-------------:|
| Python | 99,490.680 | 88,847.402 | 103,916.734 | 1 |
| Rust engine | 1,318.765 | 1,289.627 | 1,430.737 | 1 |
| Go | 2,706.000 | 2,277.000 | 2,967.000 | 1 |
| Julia | 1,229.921 | 1,159.662 | 1,301.917 | 1 |
| Mojo | spike-only | spike-only | spike-only | 1 |

Benchmark timing is local, non-isolated workstation context only. Use the
recorded medians as regression signals, not published throughput claims.

---

## Usage Examples

### Somatic drive

```python
from sc_neurocore.neurons.models.hay_l5 import HayL5PyramidalNeuron

neuron = HayL5PyramidalNeuron()
spikes = sum(neuron.step(10.0) for _ in range(20_000))

print(spikes)      # 1 under the pinned RK4 regression anchor
print(neuron.v_s)  # final soma voltage
print(neuron.ca_a) # non-negative tuft calcium
```

### Dual soma/tuft drive

```python
from sc_neurocore.neurons.models.hay_l5 import HayL5PyramidalNeuron

neuron = HayL5PyramidalNeuron()
spikes = sum(neuron.step(5.0, 5.0) for _ in range(20_000))

print(spikes)  # 4 under the dual-input RK4 anchor
```

### Euler comparison path

```python
from sc_neurocore.neurons.models.hay_l5 import HayL5PyramidalNeuron

rk4 = HayL5PyramidalNeuron()
euler = HayL5PyramidalNeuron(integrator="baseline_euler")

rk4_spikes = sum(rk4.step(10.0) for _ in range(20_000))
euler_spikes = sum(euler.step(10.0) for _ in range(20_000))

print(rk4_spikes, euler_spikes)  # 1, 10
```

---

## Verification Snapshot

Measured 2026-06-26:

- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_hay_l5.py -q`
  passed: 53 tests.
- Strict mypy passed on the Python model, tests, and benchmark driver.
- Ruff passed on the changed Python files.
- Go service tests passed.
- Rust safety standalone tests passed.
- Rust engine `hay_` tests passed.
- Julia and Mojo parity paths both reported the pinned spike anchor.
- Five-backend benchmark artefact generated at
  `benchmarks/results/local_python_2026-06-26_hay_l5_rk4.json`.

---

## Citations

1. Hay E, Hill S, Schürmann F, Markram H, Segev I (2011). Models of
   neocortical Layer 5b pyramidal cells capturing a wide range of dendritic and
   perisomatic active properties. *PLoS Computational Biology* 7(7):e1002107.
   DOI: [10.1371/journal.pcbi.1002107](https://doi.org/10.1371/journal.pcbi.1002107)

2. Larkum ME, Zhu JJ, Sakmann B (1999). A new cellular mechanism for coupling
   inputs arriving at different cortical layers. *Nature* 398:338-341.
   DOI: [10.1038/18686](https://doi.org/10.1038/18686)

**Model status:** production RK4 surface with Python/Rust/Go/Julia/Mojo parity
at the pinned regression anchors.
