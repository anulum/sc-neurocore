<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# FitzHughNagumoNeuron

!!! info "Fidelity evidence"
    The source transformation, five-runtime measurement boundary, Q16.16 RTL
    co-simulation, synthesis, and formal-safety limits are recorded in the
    [FitzHugh-Nagumo source-fidelity boundary](../../validation/fitzhugh_nagumo_source_fidelity.md).

**Module:** `sc_neurocore.neurons.models.fitzhugh_nagumo`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::FitzHughNagumoNeuron`
**Reference:** FitzHugh, Biophys. J. 1(6), 1961; Nagumo, Arimoto & Yoshizawa, Proc. IRE 50(10), 1962
**Family:** two-dimensional qualitative reduction of Hodgkin-Huxley excitability
**State variables:** `v` (fast membrane-like variable), `w` (slow recovery variable)

---

## Equations

FitzHugh's equations (1)-(3) use `(x, y, z, c)`. Under the explicit
transformation `v = -x`, `w = y`, `I = -z`, `tau = c t`, and
`epsilon = 1/c^2`, the maintained runtime surfaces integrate:

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + I$$

$$\frac{dw}{dt} = \varepsilon(v + a - bw)$$

Spike events are threshold crossings of the continuous limit-cycle variable:

$$v_t \geq v_{threshold} \;\text{and}\; v_{t-1} < v_{threshold}$$

There is no artificial reset after a spike. The trajectory continues through
the phase plane and returns through the cubic recovery dynamics. RK4, the
maintained parameter profile, the observation grid, and the threshold event are
repository numerical specializations rather than claims about the paper's
analogue solver.

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
│   ├── step(current) -> i32
│   └── simulate(n_steps, current) -> (Vec<f64>, i64)   [RK4]
├── PyO3 bindings: sc_neurocore_engine.FitzHughNagumoNeuron,
│                  sc_neurocore_engine.py_fitzhugh_nagumo_simulate
├── Polyglot simulate chain (RK4): see "Polyglot acceleration" below
│   ├── Julia: src/sc_neurocore/accel/julia/neurons/fitzhugh_nagumo.jl
│   ├── Go:    src/sc_neurocore/accel/go/neurons/fitzhugh_nagumo/fitzhugh_nagumo.go (c-shared)
│   └── Mojo:  src/sc_neurocore/accel/mojo/neurons/fitzhugh_nagumo.mojo (FFI)
├── Rust safety mirror: src/sc_neurocore/accel/rust/safety/fitzhugh_nagumo.rs
├── NetworkRunner: NeuronVariant::FitzHughNagumo
└── Hardware evidence: committed signed-Q16.16 compiler lowering
    ├── exact eight-event Icarus/VVP receipt co-simulation
    ├── Yosys coarse synthesis
    └── depth-4 public-port reset-safety BMC
```

---

## Verification evidence

The maintained evidence is organized by contract rather than brittle test
counts:

| Contract | Evidence |
|----------|----------|
| Source transformation and independent RK4 features | `tests/test_reference_fitzhugh_nagumo.py` and `fitzhugh_nagumo_driven_oscillation_doi.json` |
| Replayable complete state plus event receipt | `tests/test_bench_fitzhugh_nagumo.py` |
| Python dynamics, parameters, integrators, atomicity, and network pipeline | `tests/test_model_fitzhugh_nagumo_*.py`, `tests/test_fitzhugh_nagumo_integrator_paths.py` |
| Production Rust/PyO3 and NetworkRunner | engine model, binding, export, and runner tests |
| Safety Rust, Go, Julia, and executable Mojo | native kernel tests plus the source-hashed five-runtime packet |
| Committed Q16.16 object and event co-simulation | `tests/test_cosim_fitzhugh_nagumo_catalogue.py` |
| Yosys and formal reset safety | the same catalogue test and `hdl/formal/catalogue/sc_fitzhugh_nagumo.sby` |

The compact receipt at
`src/sc_neurocore/neurons/reference_receipts/fitzhugh_nagumo_1961.json`
records all eight event indices, final `(v,w)` state, and a binary digest over
3,000 RK4 steps at current 0.5.

---

## Benchmark evidence

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

These historical packets remain reproducible historical evidence. The current
five-runtime closure packet below supersedes them for cross-runtime parity; its
timings are still not production claims.

---

## Polyglot acceleration

`step` runs one RK4 update, but `simulate(n_steps, current, backend=...)` is a
sequential recurrence (each step depends on the previous) that does not
vectorise — a compiled inner loop genuinely beats Python. The kernel carries a
full polyglot chain over the **RK4** integrator (the production default;
`simulate` raises for the `baseline_euler` / `rosenbrock` integrators, which stay
on the per-step `step()` path):

```python
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron

neuron = FitzHughNagumoNeuron()                                    # integrator="rk4"
trace, spikes = neuron.simulate(2_000_000, current=0.5)            # auto -> Rust
trace, spikes = neuron.simulate(2_000_000, 0.5, backend="go")     # force a backend
```

`backend` accepts `"auto" | "rust" | "julia" | "go" | "mojo" | "python"`. `auto`
prefers Rust (it ships in the `sc_neurocore_engine` wheel) and falls back to the
pure-NumPy reference. `trace[t]` is `v` after step `t`; `spikes` counts upward
crossings of `v_threshold`.

The RK4 right-hand side is **exact arithmetic** — the cube is written `v*v*v`
(bit-identical to Rust `v.powi(3)`, Julia `v^3` and Go/Mojo `v*v*v`), with no
transcendental functions — and FitzHugh-Nagumo is a two-dimensional flow, so by
Poincaré-Bendixson it cannot be chaotic. **Rust, Julia and Go reproduce the NumPy
trace bit-for-bit**, even over a 50,000-step limit cycle. Mojo's release build
fuses some of the RK4 multiply-adds into FMAs (one rounding rather than two), so
each step agrees to within a couple of ULP; being non-chaotic, that gap does not
amplify and the spike counts stay identical.

> Aligning the cube to `v*v*v` (from the historical `v**3`) made the Python
> reference bit-identical to the engine's existing `v.powi(3)` and is what lets
> the whole chain agree to the last bit.

### Measured backends

Reproduce with `python benchmarks/bench_fitzhugh_nagumo_simulate.py --json
benchmarks/results/bench_fitzhugh_nagumo_simulate.json`. Workload: 2,000,000 RK4
steps, default parameters, current = 0.5, median of 5 repeats. **Non-isolated**
(loaded workstation, Python 3.12 / NumPy 2.3) — functional/regression evidence,
not isolated-core release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 1352.64 | 1.00× | 0 |
| mojo | 104.60 | 12.93× | 7.16e-13 (non-amplifying FMA) |
| go | 109.93 | 12.30× | 0 (bit-exact) |
| rust | 113.61 | 11.91× | 0 (bit-exact) |
| julia | 113.72 | 11.89× | 0 (bit-exact) |

Mojo is fastest in raw throughput (the FMA contraction helps the arithmetic-heavy
RK4), but because it is not bit-exact it is **not** chosen by `auto`; `auto`
selects Rust — the fastest backend that is both bit-exact and ships in the wheel.
All five rows record 5,067 events and the same final state within the declared
per-backend tolerance. The four derivative stages per step make the absolute
per-step cost higher than the map kernels.

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
