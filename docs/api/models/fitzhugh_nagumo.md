# SPDX-License-Identifier: AGPL-3.0-or-later
# FitzHughNagumoNeuron

**Module:** `sc_neurocore.neurons.models.fitzhugh_nagumo`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::FitzHughNagumoNeuron`
**Reference:** FitzHugh, Biophys. J. 1(6), 1961; Nagumo, Arimoto & Yoshizawa, Proc. IRE 50(10), 1962
**Family:** two-dimensional qualitative reduction of Hodgkin-Huxley excitability
**State variables:** `v` (fast membrane-like variable), `w` (slow recovery variable)

---

## Equations

The maintained runtime surfaces integrate the published two-state FitzHugh-Nagumo ODE:

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + I$$

$$\frac{dw}{dt} = \varepsilon(v + a - bw)$$

Spike events are threshold crossings of the continuous limit-cycle variable:

$$v_t \geq v_{threshold} \;\text{and}\; v_{t-1} < v_{threshold}$$

There is no artificial reset after a spike. The trajectory continues through the phase plane and returns through the cubic recovery dynamics.

---

## Production Integration Contract

The production default is RK4 over `(v, w)` with current held constant during one call. Python keeps `integrator="baseline_euler"` only as an explicit compatibility path; the default Python path, Rust engine, Julia mirror, Go mirror, and Rust safety mirror use RK4.

```python
def step(self, current: float) -> int:
    current = self._finite_float("current", current)
    self._validate_runtime_configuration()
    v_prev = self.v
    candidate = self._rk4_candidate(current)
    self.v, self.w = self._validate_candidate(*candidate)
    return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0
```

Fail-closed boundaries:

- non-finite state, parameters, or current are rejected before integration;
- `b`, `epsilon`, and `dt` must remain positive;
- overflow or non-finite derivatives reject before mutation;
- non-finite RK4 candidates reject before mutation;
- invalid safety-mirror calls return the documented failure value or raise the local language error without advancing state.

---

## Parameters

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `v` | -1.0 | fast membrane-like variable |
| `w` | -0.5 | slow recovery variable |
| `a` | 0.7 | recovery nullcline offset |
| `b` | 0.8 | recovery nullcline slope factor |
| `epsilon` | 0.08 | recovery timescale separation |
| `dt` | 0.1 | integration timestep |
| `v_threshold` | 1.0 | upward-crossing spike threshold |
| `integrator` | `rk4` | Python production integrator; `baseline_euler` and `rosenbrock` are explicit options |

`epsilon = 0.08` makes the recovery variable 12.5 times slower than the fast variable, giving the model its standard fast-slow phase-plane structure.

---

## Analytical Properties

### Nullclines

The `v` nullcline is the cubic curve:

$$w = v - v^3/3 + I$$

The `w` nullcline is the line:

$$w = (v + a)/b$$

Their intersections determine fixed points. Depending on input current, the fixed point can be stable, unstable with a surrounding limit cycle, or stabilised on the depolarised branch.

### Oscillatory band

The module-owned RK4 tests validate the default deterministic current regimes over 10,000 steps:

| Current | Spikes | Interpretation |
|--------:|-------:|----------------|
| 0.0 | 0 | stable low-current fixed point |
| 0.5 | 26 | oscillatory regime |
| 0.8 | 28 | oscillatory regime |
| 1.0 | 28 | oscillatory regime |
| 2.0 | 1 | initial crossing followed by depolarised suppression |

The tested trajectories remain finite and bounded, and repeated runs are bit-exact for the same parameters.

---

## Pipeline Position

```text
sc_neurocore pipeline
├── Python reference: sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron
│   ├── step(current) -> int {0, 1}
│   ├── reset() -> None
│   ├── Population(FitzHughNagumoNeuron, n=N)
│   ├── Network(population, drive, monitor)
│   └── Analysis: spike_count(), firing_rate(), isi()
├── Rust engine: sc_neurocore_engine::neurons::simple_spiking::FitzHughNagumoNeuron
├── PyO3 binding: sc_neurocore_engine.FitzHughNagumoNeuron
├── Julia mirror: src/sc_neurocore/accel/julia/neurons/fitzhugh_nagumo.jl
├── Go mirror: src/sc_neurocore/accel/go/services/fitzhugh_nagumo.go
└── Rust safety mirror: src/sc_neurocore/accel/rust/safety/fitzhugh_nagumo.rs
```

---

## Verification Evidence (Measured 2026-05-31)

### Python module tests and coverage

```text
PYTHONPATH=src .venv/bin/python -m coverage run --rcfile=/dev/null --source=src/sc_neurocore/neurons/models -m pytest tests/test_model_fitzhugh_nagumo.py -q
53 passed in 33.91s
src/sc_neurocore/neurons/models/fitzhugh_nagumo.py: 100%
```

### Polyglot checks

| Surface | Verification |
|---------|--------------|
| Python reference | module-specific pytest suite, 53 passed, 100% model coverage |
| Rust engine | `cargo test --manifest-path engine/Cargo.toml fhn_ -- --nocapture`, 8 FHN tests passed |
| Julia mirror | `include(...)`; one RK4 step produced finite valid state |
| Go mirror | `go test src/sc_neurocore/accel/go/services/fitzhugh_nagumo.go` |
| Rust safety mirror | `rustc --test .../fitzhugh_nagumo.rs`, 5 tests passed |

---

## Performance Benchmarks

### Python RK4 reference (local i5-11600K, measured 2026-05-31)

Artifact: `benchmarks/results/local_i5_11600k_python_2026-05-31_fitzhugh_nagumo.json`

| Metric | Value |
|--------|------:|
| Workload | 100,000 RK4 steps, 5 repeats |
| Median | 449,020,314 ns |
| Per-step median | 4,490.20314 ns |
| Throughput median | 222,707.0733 steps/s |
| Spikes per repeat | `[274, 274, 274, 274, 274]` |

### Rust engine RK4 Criterion (local i5-11600K, measured 2026-05-31)

Artifact: `benchmarks/results/local_i5_11600k_criterion_2026-05-31_fitzhugh_nagumo.json`

| Metric | Value |
|--------|------:|
| Workload | `fhn_10k_steps`, 10 samples |
| Criterion interval | 487.96 us to 518.98 us |
| Point estimate | 508.24 us per 10,000 steps |
| Point-estimate per-step | 50.824 ns |
| Point-estimate throughput | 19,675,743.7431 steps/s |

The new Rust number reflects four RK4 derivative stages per step, replacing the previous single-stage engine benchmark.

---

## Usage Examples

### Basic oscillation

```python
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron

neuron = FitzHughNagumoNeuron()
spikes = []
for t in range(10_000):
    if neuron.step(current=0.8):
        spikes.append(t)

print(f"Spike count: {len(spikes)}")
```

### Explicit legacy comparison

```python
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron

rk4 = FitzHughNagumoNeuron()
legacy = FitzHughNagumoNeuron(integrator="baseline_euler")

rk4.step(0.8)
legacy.step(0.8)
print((rk4.v, rk4.w), (legacy.v, legacy.w))
```

---

## Citations

1. **FitzHugh, R.** (1961). Impulses and physiological states in theoretical models of nerve membrane. *Biophysical Journal*, 1(6), 445-466. DOI: [10.1016/S0006-3495(61)86902-6](https://doi.org/10.1016/S0006-3495(61)86902-6)
2. **Nagumo, J., Arimoto, S., & Yoshizawa, S.** (1962). An active pulse transmission line simulating nerve axon. *Proceedings of the IRE*, 50(10), 2061-2070. DOI: [10.1109/JRPROC.1962.288235](https://doi.org/10.1109/JRPROC.1962.288235)
3. **Hodgkin, A. L. & Huxley, A. F.** (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*, 117(4), 500-544. DOI: [10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764)
4. **Izhikevich, E. M.** (2007). *Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting.* MIT Press.
5. **Ermentrout, G. B. & Terman, D. H.** (2010). *Mathematical Foundations of Neuroscience.* Springer.
