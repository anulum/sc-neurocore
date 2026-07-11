# RulkovMapNeuron

**Module:** `sc_neurocore.neurons.models.rulkov_map`
**Reference:** Rulkov 2002
**Family:** Map-based (discrete iteration, no ODE)
**State variables:** `x` (fast), `y` (slow)

---

## Equations

### Fast variable (piecewise map)

$$x_{n+1} = \begin{cases}
\frac{\alpha}{1 - x_n} + y_n + I & \text{if } x_n \leq 0 \\
\alpha + y_n + I & \text{if } 0 < x_n < \alpha + y_n + I \\
-1 & \text{otherwise}
\end{cases}$$

### Slow variable

$$y_{n+1} = y_n - \mu(x_n + 1) + \mu\sigma$$

### Spike detection

Upward crossing of x_threshold: $x_n \geq \theta$ and $x_{n-1} < \theta$.

### Implementation (as coded)

```python
def step(self, current: float = 0.0) -> int:
    x_prev = self.x
    if self.x <= 0:
        x_new = self.alpha / (1.0 - self.x) + self.y + current
    elif self.x < self.alpha + self.y + current:
        x_new = self.alpha + self.y + current
    else:
        x_new = -1.0
    y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma
    self.x = x_new
    self.y = y_new
    return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0
```

No numerical integration — pure discrete map iteration. O(1) per step.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x` | −1.0 | Fast variable (membrane analogue) |
| `y` | −3.0 | Slow variable (adaptation/recovery) |
| `alpha` | 4.0 | Controls spike amplitude and map nonlinearity |
| `sigma` | −1.6 | Slow variable offset (controls excitability) |
| `mu` | 0.001 | Slow variable timescale (mu ≪ 1 for timescale separation) |
| `x_threshold` | 0.0 | Spike detection threshold on x |

---

## Behaviour

### Three-branch piecewise structure

1. **Branch 1** ($x \leq 0$): $x_{new} = \alpha/(1-x) + y + I$. This is the
   subthreshold regime. At the initial state $x = -1$, $y = -3$, $I = 0$,
   the first fast update returns $x_{new} = 4/2 + (-3) = -1$ exactly.

2. **Branch 2** ($0 < x < \alpha + y + I$): $x_{new} = \alpha + y + I$. This
   is the spike plateau — x stays at the maximum value for one step.

3. **Branch 3** ($x \geq \alpha + y + I$): $x_{new} = -1$. Hard reset to
   the resting state.

### Verified first fast update

At default params (x=−1, y=−3, I=0): $x_{new} = 4/(1-(-1)) + (-3) + 0 = 2 - 3 = -1$.
The fast coordinate returns exactly to $x = -1$ on that first step. The full
two-state point is not fixed: the simultaneous slow update moves `y` by
$\mu\sigma=-0.0016$. Both facts are pinned by the model tests.

### Current-driven spiking

Adding current shifts x upward via all three branches. At I=0 with default
params, the slow state drifts while the neuron remains silent over the tested window.
At I=0.5, current pushes x above threshold, triggering rapid spike clusters.

### Measured dynamics (constant current)

| Current | Spikes (2,000 iterations) | Mean ISI | Regime |
|---------|-------------|----------|--------|
| 0.0 | 0 | — | Silent slow drift |
| 0.5 | 34 | 6 | Sparse bursting |
| 1.0 | 77 | 5 | Regular bursting |
| 2.0 | 179 | 4 | Fast bursting |
| 5.0 | 400 | 3 | Very fast spiking |

### Slow variable y dynamics

y evolves on the timescale $\mu = 0.001$ (1000× slower than x). At the initial
fast coordinate ($x = -1$): $\Delta y = -\mu(-1+1) + \mu\sigma = \mu\sigma = -0.0016$.
This slow drift of y modulates the burst pattern over long timescales.

### Sigma controls excitability

sigma shifts the y-nullcline. At sigma=−1.6 (default), the operating point
is in the silent regime at I=0. At sigma=1.0, the neuron fires spontaneously
(8,308 spikes/50k). This is the primary excitability control parameter.

### Alpha controls spike nonlinearity

alpha appears in branch 1 as the numerator of the nonlinear term $\alpha/(1-x)$
and in branch 2 as the spike plateau value. Higher alpha (6.0, 8.0) can trigger
spontaneous firing even at I=0. At alpha=2.0, the neuron is silent at I=0.

### x is bounded

Measured x range over 10k steps at I=0.5: approximately [−2.6, 1.4963]. The branch-3 reset
to $x = -1$ prevents divergence. The lower bound comes from branch-1 dynamics
where $\alpha/(1-x) + y$ can go negative when y is sufficiently negative.

### Bursting pattern

At moderate current, spikes come in rapid clusters (ISI 3–6 steps) separated
by quiescent intervals. The ISI shows variability (CV > 0.1), consistent with
the interaction between the fast map dynamics and the slow y modulation.

---

## Comparison with ODE-Based Models

| Property | Rulkov Map | Izhikevich 2003 | HH |
|----------|-----------|-----------------|-----|
| Integration | None (discrete map) | Euler ODE | ODE (stiff) |
| Cost per step | O(1), no multiply chain | O(1), 2 multiplies | O(1), 4+ exp() |
| Bursting | Built-in (3 branches) | Via parameter choice | Via slow K/Ca |
| Timescales | Separate (mu) | Separate (a, b) | Coupled (gating) |
| FPGA datapath | Rational divider plus branch selects | Polynomial arithmetic | Exponential look-up tables plus gating arithmetic |

---

## Numerical Considerations

- **No dt parameter:** This is a discrete map, not an ODE. Each call to `step()`
  advances one iteration. The "time step" is implicit in the map definition.
- **No numerical instability:** The map is bounded by construction — branch 3
  resets x to −1 whenever x exceeds the plateau value.
- **mu ≪ 1 required:** The timescale separation between x (fast) and y (slow)
  requires mu to be small. At mu=0.01, y evolves 10× faster, changing the
  burst pattern. At mu=0.0001, y barely moves.
- **Division by zero protection:** Branch 1 uses $\alpha/(1-x)$. At $x = 1$,
  this diverges. However, the branch condition $x \leq 0$ ensures $1-x \geq 1$,
  so division by zero cannot occur in branch 1.

---

## Python↔Verilog co-simulation and synthesis

The paired schema files use `method = "map"`, the same simultaneous recurrence
as `RulkovMapNeuron`, and rising `x >= 0` crossing detection. At `I=1.5`, the
30-iteration evidence window visits the rational, plateau, and hard-reset
branches ten times each. Hand model and both schema formats have exact states
and event decisions. The generated Q16.16 RTL reproduces the complete ten-event
vector while each committed `x`/`y` state remains within `0.001` absolute error
of the float64 hand trajectory.

This bounded trajectory is the declared metric for a sensitive discrete map.
Long-window spike-count identity is intentionally not used: fixed-point rounding
can move a sensitive map onto a different orbit even when the short-window
lowering is faithful.

The Q16.16 core also passes Yosys 0.33 `synth_xilinx`; the raw synthesis report
is committed at `hdl/reports/yosys_rulkov_map_q1616_2026-07-11.json`, satisfying
the model's documented H2 terminal. The formal catalogue carries a separate
Q8.8 port-only reset-spike safety job at BMC depth 4.

---

## Polyglot acceleration

A single `step` is trivial, but `simulate(n_steps, current, backend=...)` is a
sequential recurrence (each step depends on the previous, and an upward-crossing
spike depends on the previous `x`) that does not vectorise — a compiled inner
loop genuinely beats Python. The kernel carries a full polyglot chain:

```python
neuron = RulkovMapNeuron()
trace, spikes = neuron.simulate(2_000_000, current=0.5)            # auto -> Rust
trace, spikes = neuron.simulate(2_000_000, 0.5, backend="go")     # force a backend
```

`backend` accepts `"auto" | "rust" | "julia" | "go" | "mojo" | "python"`. `auto`
prefers Rust (it ships in the `sc_neurocore_engine` wheel) and falls back to the
pure-NumPy reference. `trace[t]` is `x` after step `t`; `spikes` counts upward
crossings of `x_threshold`; the instance `(x, y)` is left at the final step.

Because the fast map is exact floating-point arithmetic (one division,
additions and multiplications, no transcendental functions), **Rust, Julia and
Go reproduce the NumPy trace bit-for-bit** across the silent, bursting and
spontaneous regimes. Mojo's release build can contract the slow-variable update
`y - mu*(x+1) + mu*sigma` into fused multiply-adds (one rounding rather than
two); each step therefore agrees to within a couple of ULP. Unlike a freely
chaotic map, the branch resets (`x` to exactly `-1`, or to the plateau value)
periodically resynchronise the trajectory, so the whole-trace gap stays at the
per-step ULP level rather than diverging. This is the documented Mojo FMA-parity
behaviour, not a defect, and the spike counts still match exactly.

### Measured backends

Reproduce with `PYTHONPATH=src .venv/bin/python benchmarks/bench_rulkov_map.py --json
benchmarks/results/bench_rulkov_map.json`. Workload: 2,000,000 steps, default
parameters, current = 0.5, median of 5 repeats. **Non-isolated** (loaded
workstation, Python 3.12 / NumPy 2.3) — functional/regression evidence, not
isolated-core release numbers. The committed artefact includes source SHA-256
hashes for the Python benchmark, Python model, Rust engine entry points, Go
cgo kernel, Julia kernel, and Mojo kernel; `tools/benchmark_evidence_gate.py`
validates those hashes plus backend numeric metrics before release use.

| backend | median (ms) | min (ms) | speedup vs NumPy | parity Δ vs NumPy | spikes |
|---|---:|---:|---:|---:|---:|
| python (NumPy) | 357.54 | 328.32 | 1.00× | 0 | 34 |
| go | 14.66 | 14.18 | 24.39× | 0 | 34 |
| mojo | 15.84 | 15.48 | 22.57× | 1.78e-15 (sub-ULP FMA) | 34 |
| rust | 16.33 | 16.02 | 21.89× | 0 | 34 |
| julia | 17.71 | 17.24 | 20.19× | 0 | 34 |

The speedups are more modest than for the branch-free Cazelles map (~22× versus
~82×): the three-branch conditional limits instruction-level parallelism in
every backend, and the NumPy reference inner loop is correspondingly cheaper per
step. Go and Mojo lead by filling a preallocated NumPy buffer over the C ABI;
Rust returns a NumPy array directly (avoiding a multi-million-element
Python-list marshal); `auto` selects Rust as the always-available wheel backend
within ~1.11× of the fastest locally-built backend. The benchmark gate
additionally requires all five backend spike counts to remain equal; the
refreshed artefact reports 34 spikes for every backend.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/rulkov_map.py`.
- **Backends:** Rust (`engine/src/neurons/maps.rs` + `py_rulkov_map_simulate`),
  Julia (`accel/julia/neurons/rulkov_map.jl`), Go
  (`accel/go/neurons/rulkov_map/rulkov_map.go`, c-shared), Mojo
  (`accel/mojo/neurons/rulkov_map.mojo`, FFI). Each reproduces the NumPy
  reference bit-for-bit (Rust/Julia/Go) or to a documented per-step ULP bound
  (Mojo); see *Polyglot acceleration* above.
- **Rust wiring:** `RulkovMapNeuron::step(f64) → i32` and
  `RulkovMapNeuron::simulate(n_steps, current) → (Vec<f64>, i64)`; two f64 state
  variables, six f64 parameters.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, state evolution, finite 50k, reset |
| Map dynamics | 5 | branch-1 first update (x=−1 exactly), branch-1 current shift, branch-3 reset to −1, y slow drift (μσ), x bounded |
| f–I curve | 4 | silent at I=0, I=0.5 triggers spikes, rate increases, monotonic 4-point |
| Bursting | 2 | short ISI (median <10), ISI variability (CV>0.1) |
| Parameters | 4 | sigma excitability, alpha amplitude, mu timescale, upward crossing |
| Determinism | 1 | bit-exact (300 steps) |
| Network | 2 | Population(n=10), Network spikes |
| Analysis | 2 | spike_count, consistency |
| Polyglot parity | 33 | rust/julia/go bit-exact (4 regimes + empty/single + high-current branches 2/3), mojo ULP-bounded trace + per-step + spike count, dispatch/validation, simulate==repeated-step, final-state advance |
| Schema/RTL trajectory | 1 | exact hand/TOML/JSON states, all three branches, exact Q16.16 event vector, bounded x/y error |
| Silicon evidence | 2 | Q16.16 Yosys synthesis report and Q8.8 depth-4 formal safety job |

The step-level categories above are listed by intent; parametrisation expands
them at collection time. The two files collect **67 tests** in total (34 in
`tests/test_model_rulkov_map.py`, 33 in `tests/test_rulkov_map_backends.py`),
all passing.

---

## Findings

1. **Initial fast update verified exactly:** At x=−1, y=−3, I=0: x_new = −1.0
   to machine precision while the slow coordinate advances.
2. **y slow drift = μσ at the initial x:** Measured Δy = −0.0016 = μ × σ
   exactly. The slow variable dynamics are correct.
3. **Branch-3 reset to −1 confirmed:** When x=5 with alpha+y+I=1,
   x_new = −1.0 exactly.
4. **Sigma is the excitability switch:** sigma=−1.6 → silent at I=0;
   sigma=1.0 → 8,308 spontaneous spikes. The y offset determines whether
   the operating point sits in the excitable or oscillatory regime.
5. **Bursting confirmed:** ISI at I=0.5 shows median ≈ 6 steps with
   CV > 0.1, consistent with rapid spike clusters separated by
   quiescent intervals.
6. **No division-by-zero risk:** Branch-1 condition $x \leq 0$ guarantees
   $1 - x \geq 1$. Verified by the x range never exceeding 0 in branch 1.
7. **Current shift verified:** At x=−1, y=−3, I=2: x_new = 4/2 + (−3) + 2 = 1.0.
   Measured to within 1e-10 of exact.
8. **mu timescale separation:** At mu=0.01, y drifts 10× faster than at
   mu=0.001. Measured: |y − y_0| is larger with mu=0.01 after 1k steps.


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~108K steps/s |
| Spikes (10K steps, I=5.0) | 400 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`RulkovMapNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
400 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(RulkovMapNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~108K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
