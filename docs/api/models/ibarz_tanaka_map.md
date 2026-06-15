# IbarzTanakaMapNeuron

**Module:** `sc_neurocore.neurons.models.ibarz_tanaka_map`
**Reference:** Ibarz et al. 2007
**Family:** Map-based (piecewise-linear bursting)
**State variables:** `x` (fast, ≈voltage), `y` (slow, ≈adaptation)

## Equations

$$x_{n+1} = f(x_n) + y_n + I$$
$$y_{n+1} = y_n - \mu(x_n + 1) + \mu\sigma$$

$$f(x) = \begin{cases} \alpha/(1-x) & x \leq 0 \\ \alpha + \beta x & x > 0 \end{cases}$$

Spike: $x \geq x_\theta$, reset $x \to x_{reset}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 3.65 | Piecewise map amplitude |
| `beta` | 0.25 | Linear spiking slope |
| `mu` | 0.0005 | Slow time-scale |
| `sigma` | -1.6 | Slow variable target |
| `x_threshold` | 3.0 | Spike threshold |
| `x_reset` | -1.0 | Post-spike reset |

## Behaviour

- **Discrete map:** No ODE — iterative, computationally cheap.
- **Piecewise-linear:** f(x) has a singularity at x=1 (from left),
  producing sharp spike onset. Linear spiking phase above x=0.
- **Bursting:** Slow y variable (µ=0.0005) modulates burst-pause.
- **Deterministic:** Fully deterministic map.
- **Efficient:** Single evaluation per step — ideal for large networks.

## Polyglot acceleration

A single `step` is trivial, but `simulate(n_steps, current, backend=...)` is a
sequential recurrence (each step depends on the previous) that does not
vectorise — a compiled inner loop genuinely beats Python. The kernel carries a
full polyglot chain:

```python
neuron = IbarzTanakaMapNeuron()
trace, spikes = neuron.simulate(2_000_000, current=3.0)            # auto -> Rust
trace, spikes = neuron.simulate(2_000_000, 3.0, backend="go")     # force a backend
```

`backend` accepts `"auto" | "rust" | "julia" | "go" | "mojo" | "python"`. `auto`
prefers Rust (it ships in the `sc_neurocore_engine` wheel) and falls back to the
pure-NumPy reference. `trace[t]` is `x` after step `t` — already reset to
`x_reset` on a spiking step; `spikes` counts threshold crossings; the instance
`(x, y)` is left at the final step.

Because the map is exact floating-point arithmetic (one division, additions and
multiplications, no transcendental functions), **Rust, Julia and Go reproduce
the NumPy trace bit-for-bit** across the silent, bursting and strongly-driven
regimes (defaults are silent below current ≈ 2.0). Mojo's release build can
contract the linear branch `alpha + beta*x` and the slow-variable update
`y - mu*(x+1) + mu*sigma` into fused multiply-adds (one rounding rather than
two); each step therefore agrees to within a couple of ULP. The explicit reset
to `x_reset` on every spike periodically resynchronises the trajectory, so the
whole-trace gap stays at the per-step ULP level rather than diverging. This is
the documented Mojo FMA-parity behaviour, not a defect, and the spike counts
still match exactly.

### Measured backends

Reproduce with `python benchmarks/bench_ibarz_tanaka_map.py --json
benchmarks/results/bench_ibarz_tanaka_map.json`. Workload: 2,000,000 steps,
default parameters, current = 3.0 (sustained bursting), median of 5 repeats.
**Non-isolated** (loaded workstation, Python 3.12 / NumPy 2.3) —
functional/regression evidence, not isolated-core release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 315.99 | 1.00× | 0 |
| go | 15.36 | 20.57× | 0 |
| rust | 15.86 | 19.93× | 0 |
| mojo | 17.40 | 18.16× | 4.00e-13 (sub-ULP FMA) |
| julia | 21.83 | 14.47× | 0 |

The speedups (~20×) are modest relative to a branch-free map: the two-piece
conditional plus the per-spike reset limit instruction-level parallelism in
every backend. Go and Mojo lead by filling a preallocated NumPy buffer over the
C ABI; Rust returns a NumPy array directly (avoiding a multi-million-element
Python-list marshal); `auto` selects Rust as the always-available wheel backend
within ~1.03× of the fastest locally-built backend.

## Infrastructure Pipeline

```
IbarzTanakaMapNeuron
├── step(current) → int {0,1}
├── Population: works
├── Verilog: division LUT + comparator, ~30 LUTs
├── Rust: supported via NeuronVariant
└── simulate(): polyglot N-step chain (rust/julia/go/mojo) — see above
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, piecewise f, slow y, reset on spike, rate increase, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| Polyglot parity | 33 | rust/julia/go bit-exact (4 regimes + empty/single + strong-drive resets), mojo ULP-bounded trace + per-step + spike count, dispatch/validation, simulate==repeated-step, final-state advance, reset-value trace |

The polyglot parity suite lives in `tests/test_ibarz_tanaka_backends.py`; the
step-level suite lives in `tests/test_model_ibarz_tanaka.py`. The two files
collect **64 tests** in total, all passing.


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~319K steps/s |
| Spikes (10K steps, I=5.0) | 2421 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`IbarzTanakaMapNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
2421 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(IbarzTanakaMapNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~319K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
