# MedvedevMapNeuron

**Module:** `engine/src/neurons/maps.rs`
**Rust struct:** `MedvedevMapNeuron` (line 164)
**Reference:** Medvedev, SIAM J Appl Dyn Syst 4:1228, 2005
**Family:** Map-based (1D chaotic, piecewise-monotone)
**State variables:** `x` (phase variable, mod 1)

---

## Biological Context

Map-based neuron models replace continuous-time differential equations with discrete-time
iterative maps. They sacrifice biophysical detail for extreme computational efficiency —
a single map iteration replaces the entire integration loop of a conductance-based model.

The Medvedev map belongs to the class of **piecewise-monotone expanding maps** that
generate chaotic dynamics. These maps are studied in mathematical neuroscience for their
ability to produce:

1. **Chaotic spike trains:** The sensitive dependence on initial conditions generates
   aperiodic, statistically complex firing patterns similar to those observed in cortical
   neurons in vivo.

2. **Information encoding:** Chaotic dynamics can encode information in the precise timing
   of spikes, with the inherent noise-like variability providing a rich temporal code.

3. **Rapid switching:** As discrete-time maps, transitions between firing patterns occur
   instantaneously (one iteration), modelling the abrupt mode transitions observed in
   some neuronal circuits.

### Theoretical background

The Medvedev map is a **tent-map variant** — a piecewise-linear expanding map on the
unit interval [0, 1). The tent map is one of the simplest dynamical systems exhibiting
chaotic behaviour and is closely related to the logistic map through a topological
conjugacy.

The key property of the Medvedev map is that for expansion rate α > 2, the map is
**uniformly expanding** (|f'(x)| > 1 everywhere), which guarantees:
- Positive Lyapunov exponent (chaotic dynamics)
- Ergodicity with respect to an absolutely continuous invariant measure
- Dense periodic orbits
- Topological transitivity (mixing)

These are the hallmarks of deterministic chaos, making the Medvedev map a minimal model
for studying chaos in neural systems.

### Relation to other map models in SC-NeuroCore

| Model | Dimensions | Nonlinearity | Chaos regime |
|-------|-----------|-------------|-------------|
| MedvedevMap | 1D | Piecewise-linear (tent) | α > 2 |
| AiharaMap | 2D | Sigmoid | k_f > 0.7, α > 1.5 |
| ErmentroutKopellMap | 1D | Trigonometric (theta) | No chaos (λ=0) |
| CazellesMap | 2D | Logistic quadratic | a > 3.57 |
| CourageNekorkinMap | 2D | Piecewise-linear Lorenz | α > 2 |
| IbarzTanakaMap | 2D | Quadratic + slow | Model-specific |
| RulekovMap | 2D | 1/(1-x) | σ-dependent |

The Medvedev map is the simplest of all map models: 1D, piecewise-linear, no slow
variable. This makes it the fastest to compute but also the least biophysically detailed.

---

## Mathematical Model

### Map equation

The Medvedev map is a piecewise-linear tent map with external input:

$$x_{n+1} = \begin{cases}
\alpha \, x_n + I & \text{if } x_n < \beta \\
\alpha \, (1 - x_n) + I & \text{if } x_n \geq \beta
\end{cases} \pmod{1}$$

where:
- $x_n \in [0, 1)$ is the phase variable at step n
- $\alpha = 3.5$ is the expansion rate (slope magnitude)
- $\beta = 0.5$ is the branch point (fold location)
- $I$ is the external input current
- The modular arithmetic $\text{rem\_euclid}(1)$ constrains $x$ to $[0, 1)$

### Geometric interpretation

The map consists of two linear branches meeting at $x = \beta$:

1. **Left branch** ($x < \beta$): $f(x) = \alpha x + I$ — ascending with slope $\alpha$
2. **Right branch** ($x \geq \beta$): $f(x) = \alpha(1-x) + I$ — descending with slope $-\alpha$

This creates a **tent shape** centered at $x = \beta$:
- Maximum value: $f(\beta) = \alpha \beta + I$ (from left) = $\alpha(1-\beta) + I$ (from right)
- At the default $\beta = 0.5$: maximum = $0.5\alpha + I = 1.75 + I$

The mod 1 operation wraps the output back to [0, 1), preventing divergence regardless of
α or I values.

### Spike detection

A spike is detected when x crosses the threshold $x_\theta = 0.9$ **upward**:

$$\text{spike}_n = \begin{cases}
1 & \text{if } x_n \geq x_\theta \text{ and } x_{n-1} < x_\theta \\
0 & \text{otherwise}
\end{cases}$$

The upward-crossing requirement prevents double-counting when x remains above threshold
for multiple consecutive iterations.

### Lyapunov exponent

For the tent map without input (I = 0) and with $\beta = 0.5$:

$$\lambda = \ln|\alpha|$$

At the default $\alpha = 3.5$: $\lambda = \ln(3.5) \approx 1.253$

This positive Lyapunov exponent confirms chaotic dynamics. The exponent is constant
(independent of the trajectory) because the slope magnitude is uniform across both
branches.

| α | λ | Behaviour |
|---|---|-----------|
| 0.5 | -0.693 | Stable fixed point (contracting) |
| 1.0 | 0.0 | Marginal (neutral) |
| 2.0 | 0.693 | Chaotic onset |
| 3.5 (default) | 1.253 | Strongly chaotic |
| 5.0 | 1.609 | Very strong chaos |

### Fixed points

Without input (I = 0), the fixed points satisfy $f(x^*) = x^*$:

**Left branch** ($x^* < \beta$): $\alpha x^* = x^* \pmod{1}$ → $x^* = 0$ (for $\alpha \neq 1$)

**Right branch** ($x^* \geq \beta$): $\alpha(1-x^*) = x^* \pmod{1}$ → $x^* = \alpha/(1+\alpha)$

At $\alpha = 3.5$: $x^* = 3.5/4.5 \approx 0.778$

Both fixed points are **unstable** for $\alpha > 1$ (|f'| = α > 1), so nearby orbits
diverge exponentially. This is why the dynamics are chaotic — no stable attracting point
exists.

### Invariant measure

For the symmetric tent map ($\beta = 0.5$) with $\alpha > 2$ and I = 0, the invariant
measure is the **Lebesgue measure** on [0, 1). This means that long-run time averages
equal spatial averages (ergodicity), and the trajectory fills [0, 1) uniformly.

With external input $I \neq 0$, the invariant measure shifts but the chaotic character
is preserved as long as the map remains expanding after mod 1 wrapping.

---

## Effect of Input Current on Dynamics

The external input I acts as a **vertical shift** of the tent map:

| I | Effect |
|---|--------|
| 0.0 | Pure tent map, symmetric about β |
| 0.1 | Slight upward shift, changes spike timing |
| 0.5 | Significant shift, different ergodic measure |
| 1.0 | Large shift, many wrappings per iteration |

Because of the mod 1 operation, increasing I doesn't change the fundamental chaotic
character (the expansion rate α is unchanged), but it alters:
- The specific orbit sequence (butterfly effect)
- The spike timing pattern (different threshold crossings)
- The time-averaged firing rate

### Approximate firing rate

In the ergodic regime (α > 2), the fraction of time x spends above threshold is
approximately the measure of the set $\{x : f^{-1}([x_\theta, 1)) \cap [0, 1)\}$.
For the symmetric tent map with threshold crossing detection:

The expected spike rate depends on the probability of an upward crossing of x_θ,
which requires $x_{n-1} < x_\theta$ and $x_n \geq x_\theta$. For the uniform invariant
measure, this probability is approximately:

$$P(\text{spike}) \approx (1 - x_\theta) \cdot x_\theta = 0.1 \times 0.9 = 0.09$$

So roughly ~9% of iterations produce a spike (every ~11 steps). With external input,
this rate changes.

---

## Comparison with Other 1D Maps

### Medvedev vs Logistic map

| Property | Medvedev (tent) | Logistic $x \to rx(1-x)$ |
|----------|----------------|--------------------------|
| Linearity | Piecewise-linear | Quadratic |
| Derivative | Constant ±α | Varies: r(1-2x) |
| Lyapunov | ln α (constant) | Varies with orbit |
| Chaos onset | α > 1 | r > 3.5699... |
| Period doubling | No (direct chaos) | Yes (cascade) |
| Computation | 1 multiply + 1 add | 2 multiplies + 1 add |
| mod 1 needed | Yes | No (naturally bounded for r ≤ 4) |

### Medvedev vs Bernoulli shift

The Medvedev map with $\beta = 0.5$ and $I = 0$ is a **generalised Bernoulli shift**:
- Bernoulli shift: $x \to 2x \pmod{1}$ (binary shift, α = 2)
- Medvedev: $x \to \alpha x \pmod{1}$ with folding at β (α = 3.5)

The Bernoulli shift is the simplest chaotic map; the Medvedev map adds the piecewise
structure and adjustable expansion rate.

---

## Effect of Parameters on Behaviour

### Expansion rate (α)

| α | Dynamics | Lyapunov |
|---|----------|----------|
| < 1 | Contracting → stable fixed point | < 0 |
| 1 | Neutral | 0 |
| 1 < α < 2 | Expanding but orbits may be periodic | > 0 |
| ≥ 2 | Uniformly expanding, ergodic chaos | ≥ ln 2 |
| 3.5 (default) | Strongly chaotic | 1.253 |

### Branch point (β)

| β | Effect |
|---|--------|
| 0.0 | Degenerate: right branch only |
| 0.25 | Asymmetric: shorter left branch |
| 0.5 (default) | Symmetric tent |
| 0.75 | Asymmetric: longer left branch |
| 1.0 | Degenerate: left branch only |

Asymmetric β values change the invariant measure from uniform to non-uniform,
creating preferred regions in the phase space.

### Spike threshold (x_θ)

| x_θ | Approximate spike probability |
|-----|------------------------------|
| 0.5 | High (~25% of steps) |
| 0.7 | Moderate (~15%) |
| 0.9 (default) | Low (~9%) |
| 0.95 | Very low (~5%) |

Lower thresholds produce higher spike rates; the upward-crossing requirement
prevents saturation.

---

## Parameters

All defaults from `MedvedevMapNeuron::new()` in `maps.rs:172`:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `x` | 0.0 | — | Phase variable [0, 1) |
| `alpha` | 3.5 | — | Expansion rate (slope magnitude) |
| `beta` | 0.5 | — | Piecewise branch point |
| `x_threshold` | 0.9 | — | Spike detection threshold |

Note: There is no `dt` parameter — map models operate in discrete time (one iteration
per `step()` call). There is also no `gain` parameter — the input current enters
additively.

---

## Implementation Details

### Code structure (`maps.rs:180–193`)

```
step(current) → i32:
    x_prev = x

    if x < β:
        x = α × x + current
    else:
        x = α × (1 - x) + current

    x = x.rem_euclid(1.0)   // Euclidean remainder: x ∈ [0, 1)

    if x ≥ x_threshold AND x_prev < x_threshold:
        return 1  // spike (upward crossing)
    return 0
```

### Key implementation notes

1. **rem_euclid vs fmod:** Rust's `rem_euclid(1.0)` returns a value in [0, 1) even
   for negative inputs (unlike C's fmod which preserves sign). This ensures x is always
   non-negative.

2. **Upward crossing detection:** The spike condition requires **both** the current x
   to be above threshold **and** the previous x to be below. This prevents counting
   a spike when x remains above threshold for consecutive iterations.

3. **No safety clamps:** Unlike conductance-based models, there are no NaN checks or
   voltage clamps. The mod 1 operation inherently bounds x to [0, 1), preventing
   divergence. If current produces NaN, x will become NaN and remain so — there is no
   NaN recovery mechanism.

4. **No sub-stepping:** Map models compute one iteration per `step()` call. The concept
   of sub-steps does not apply to discrete-time maps.

5. **Reset:** `reset()` sets x = 0.0 (initial value from `new()`).

6. **Input is additive:** Unlike gain-modulated models, the external current I is added
   directly to the map value before mod 1 wrapping.

---

## Numerical Example

**Setup:** Default parameters (α=3.5, β=0.5, x_θ=0.9), I = 0.0.

Starting from x₀ = 0.0:

| Step | x_prev | Branch | f(x) = α×... | +I | mod 1 | x_new | Spike? |
|------|--------|--------|-------------|-----|-------|-------|--------|
| 1 | 0.000 | Left | 3.5×0.0 = 0.0 | 0.0 | 0.000 | 0.000 | No (0<0.9, 0<0.9) |
| | | | | | | | |
| *With I = 0.1:* | | | | | | | |
| 1 | 0.000 | Left | 3.5×0.0 = 0.0 | 0.1 | 0.100 | 0.100 | No |
| 2 | 0.100 | Left | 3.5×0.1 = 0.35 | 0.1 | 0.450 | 0.450 | No |
| 3 | 0.450 | Left | 3.5×0.45 = 1.575 | 0.1 | 0.675 | 0.675 | No |
| 4 | 0.675 | Right | 3.5×(1-0.675) = 1.1375 | 0.1 | 0.2375 | 0.2375 | No |
| 5 | 0.238 | Left | 3.5×0.238 = 0.831 | 0.1 | 0.931 | 0.931 | Yes! (0.238<0.9, 0.931≥0.9) |

The orbit visits different regions of [0, 1) aperiodically, with occasional threshold
crossings generating spikes.

---

## Ergodic Properties

### Time averages

For the symmetric tent map (β = 0.5, I = 0) with α ≥ 2, the Birkhoff ergodic theorem
guarantees that time averages converge to the spatial average:

$$\lim_{N \to \infty} \frac{1}{N} \sum_{n=0}^{N-1} g(x_n) = \int_0^1 g(x) \, dx$$

for any integrable function g and almost all initial conditions x₀.

This means:
- Time-averaged x: $\langle x \rangle = 0.5$
- Time-averaged x²: $\langle x^2 \rangle = 1/3$
- Variance: $\text{Var}(x) = 1/3 - 1/4 = 1/12 \approx 0.083$

### Autocorrelation

The autocorrelation function of the tent map decays exponentially:

$$C(k) = \langle x_n x_{n+k} \rangle - \langle x \rangle^2 \propto \alpha^{-k}$$

At α = 3.5: $C(k) \propto 3.5^{-k}$, so correlations decay by ~71% per step.
After 3 steps, autocorrelation is $< 2\%$ — the map produces nearly uncorrelated
sequences on timescales longer than a few iterations.

### Entropy

The metric (Kolmogorov-Sinai) entropy equals the positive Lyapunov exponent for
expanding maps:

$$h = \lambda = \ln \alpha = \ln 3.5 \approx 1.253 \; \text{bits/iteration}$$

This quantifies the rate of information production: each iteration generates ~1.25 bits
of unpredictable information.

---

## Sensitive Dependence on Initial Conditions

Two orbits starting from nearby initial conditions diverge exponentially:

$$|x_n^{(1)} - x_n^{(2)}| \sim |x_0^{(1)} - x_0^{(2)}| \cdot e^{\lambda n}$$

At α = 3.5 (λ ≈ 1.253): a perturbation of $\delta = 10^{-15}$ (double precision limit)
grows to O(1) in approximately:

$$n^* \approx \frac{\ln(1/\delta)}{\lambda} = \frac{15 \ln 10}{1.253} \approx 27.6 \; \text{steps}$$

After ~28 iterations, two orbits with machine-epsilon difference are completely
decorrelated. This has implications for:
- Reproducibility: identical seeds are required for deterministic replay
- Network simulations: small numerical differences (e.g., GPU vs CPU) produce
  completely different spike trains after ~30 steps

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| Multipliers | DSP48E1 | 1 slice |
| State registers | Flip-flops | ~64 bits (1 × 64-bit state) |
| Comparator | LUT | ~32 LUTs (x < β) |
| Modular arithmetic | LUT | ~64 LUTs (rem_euclid) |
| Total LUTs | | ~150–250 |
| Pipeline depth | Cycles | 2–3 |
| Latency at 100 MHz | | 20–30 ns |
| Throughput | Neurons/s | ~33–50 M |

**Key advantages for FPGA:**
- Single multiply per step (α × x or α × (1-x))
- No exponentials, no transcendental functions
- No sub-stepping — 1 clock pipeline
- mod 1 is trivially implemented as keeping the fractional part
- In fixed-point (Q1.15 or Q1.31): mod 1 = mask off integer part

This is the **cheapest model in SC-NeuroCore** for FPGA — a single DSP slice plus
minimal logic can implement the entire neuron. A Zynq-7020 could potentially run
>10,000 Medvedev neurons per clock cycle in a systolic array.

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Python model + dispatch | `src/sc_neurocore/neurons/models/medvedev_map.py` |
| Rust implementation | `engine/src/neurons/maps.rs` (`step` + `simulate`) |
| PyO3 wrappers | `py_neuron_default!` (state: x) + `py_medvedev_map_simulate` |
| Polyglot `simulate` chain | rust / julia / go / mojo (see below) |
| NetworkRunner wired | `NeuronVariant::MedvedevMap` |
| `create_neuron("MedvedevMapNeuron")` | Yes |
| coverage tests | step-level `tests/test_model_medvedev_map.py` + polyglot parity `tests/test_medvedev_map_backends.py` (71 collected across both, all passing) |
| Benchmark | `benchmarks/bench_medvedev_map.py` (+ committed JSON) |

---

## Polyglot acceleration

`step` is a single iteration, but `simulate(n_steps, current, backend=...)` is a
sequential recurrence (each step depends on the previous) that does not
vectorise — a compiled inner loop genuinely beats Python. The kernel carries a
full polyglot chain:

```python
from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron

neuron = MedvedevMapNeuron()
trace, spikes = neuron.simulate(2_000_000, current=0.1)            # auto -> Rust
trace, spikes = neuron.simulate(2_000_000, 0.1, backend="go")     # force a backend
```

`backend` accepts `"auto" | "rust" | "julia" | "go" | "mojo" | "python"`. `auto`
prefers Rust (it ships in the `sc_neurocore_engine` wheel) and falls back to the
pure-NumPy reference. `trace[t]` is `x` after step `t` (folded into `[0, 1)`);
`spikes` counts upward crossings of `x_threshold`; the instance `x` is left at
the final step.

The fold uses the **Euclidean remainder**, which is bit-identical across the
chain: Python `x % 1.0` = Rust `rem_euclid(1.0)` = Julia `mod(x, 1.0)` =
Go/Mojo `x - floor(x)` (note Julia's `%` operator is `rem`, truncated, and must
not be used). Because every step is exact floating-point arithmetic, **Rust,
Julia and Go reproduce the NumPy trace bit-for-bit** across the chaotic regime.

Mojo's release build contracts `alpha*x + current` into a fused multiply-add
(one rounding rather than two), so each step agrees only to within a couple of
ULP. This is an expanding chaotic map (λ = ln 3.5 ≈ 1.25 > 0), so a single ULP
is amplified into a visibly different whole trace and a slightly different spike
count over long horizons — by design, exactly the sensitive-dependence property
documented above. Mojo is therefore validated on the per-step ULP bound and the
`[0, 1)` structural invariant, not on whole-trace or exact-spike equality; when
bit-exactness is required, use `auto` (Rust), `julia` or `go`.

### Measured backends

Reproduce with `python benchmarks/bench_medvedev_map.py --json
benchmarks/results/bench_medvedev_map.json`. Workload: 2,000,000 steps, default
parameters, current = 0.1, median of 5 repeats. **Non-isolated** (loaded
workstation, Python 3.12 / NumPy 2.3) — functional/regression evidence, not
isolated-core release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 220.72 | 1.00× | 0 |
| mojo | 11.07 | 19.94× | 9.99e-01 (chaotic FMA divergence) |
| rust | 17.21 | 12.82× | 0 |
| julia | 32.16 | 6.86× | 0 |
| go | 33.58 | 6.57× | 0 |

Mojo is fastest in raw throughput, but because its FMA contraction diverges on
this chaotic map it is **not** chosen by `auto`; `auto` selects Rust — the
fastest **bit-exact** backend and the one that ships in the wheel. Julia and Go
trail here because the per-step work is tiny (one multiply, one add, one fold),
so the `mod`/`math.Mod` call and FFI marshalling dominate the loop.

---

## Benchmark

### Polyglot throughput (measured 2026-06-14)

See *Polyglot acceleration* above for the full `simulate` backend table
(2,000,000-step workload). Single-step `step()` throughput is dominated by PyO3
call overhead; the `simulate` path removes that per-call cost, which is why the
NumPy inner loop alone reaches ~9 M steps/s and Rust ~116 M steps/s on this
2,000,000-step workload.

### Performance context

The Medvedev map is one of the fastest neuron models in SC-NeuroCore:
- **1 multiply + 1 add + 1 fold** per step (vs ~100 operations for WB models)
- No sub-stepping (vs 50 sub-steps for conductance-based models)
- Rust/Julia/Go parity is **bit-exact** (the fold's Euclidean remainder is
  order-independent); Mojo diverges only through FMA contraction on the chaotic
  orbit.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import MedvedevMapNeuron

neuron = MedvedevMapNeuron()

# Generate chaotic spike train
spikes = []
x_trace = []
for step in range(10000):
    fired = neuron.step(0.1)  # Small constant input
    if fired:
        spikes.append(step)
    x_trace.append(neuron.x)

print(f"Spikes: {len(spikes)} in 10K steps")
print(f"Mean ISI: {10000/max(len(spikes),1):.1f} steps")

# Demonstrate sensitive dependence
neuron1 = MedvedevMapNeuron()
neuron2 = MedvedevMapNeuron()
neuron2.x = 1e-15  # Tiny perturbation
for step in range(50):
    neuron1.step(0.1)
    neuron2.step(0.1)
    diff = abs(neuron1.x - neuron2.x)
    if step % 10 == 0:
        print(f"Step {step}: |Δx| = {diff:.2e}")
# Expected: diff grows exponentially, reaching O(1) by step ~28
```

### Rust

```rust
use sc_neurocore_engine::neurons::maps::MedvedevMapNeuron;

let mut neuron = MedvedevMapNeuron::new();
let mut spike_count = 0;

for _ in 0..10000 {
    spike_count += neuron.step(0.1);
}

println!("Spikes: {}, x: {:.6}", spike_count, neuron.x);
```

---

## Findings

1. **Chaotic dynamics.** For α = 3.5, the orbit fills [0, 1) aperiodically with positive
   Lyapunov exponent λ = ln(3.5) ≈ 1.253. Verified.
2. **x bounded.** rem_euclid(1.0) constrains x to [0, 1) regardless of input. Verified.
3. **Piecewise branches.** Both branches (x < β and x ≥ β) are exercised during typical
   orbits. Verified.
4. **Spike detection.** Upward crossing of x_θ = 0.9 correctly detected. Verified.
5. **Sensitive dependence.** Two orbits with 10⁻¹⁵ initial difference diverge to O(1)
   within ~28 steps. Verified.
6. **Rust parity.** Python and Rust produce identical spike trains (EXACT). Verified.
7. **State stability.** 20K steps without NaN or divergence. Verified.
8. **Reset.** x returns to 0.0 after `reset()`. Verified.
9. **Deterministic.** Same initial condition + same input = identical output. Verified.

---

## References

1. Medvedev GS (2005). Reduction of a model of an excitable cell to a one-dimensional
   map. *SIAM J Appl Dyn Syst* 4:1228–1262.

2. Medvedev GS (2006). Transition to bursting via deterministic chaos. *Phys Rev Lett*
   97:048102.

3. Ibarz B, Casado JM, Sanjuán MAF (2011). Map-based models in neuronal dynamics.
   *Phys Rep* 501:1–74.

4. Lasota A, Mackey MC (1994). *Chaos, Fractals, and Noise: Stochastic Aspects of
   Dynamics.* Springer-Verlag, New York.

5. Devaney RL (1989). *An Introduction to Chaotic Dynamical Systems.* Addison-Wesley,
   Redwood City, CA.

6. Boyarsky A, Góra P (1997). *Laws of Chaos: Invariant Measures and Dynamical Systems
   in One Dimension.* Birkhäuser, Boston.

7. Rulkov NF (2002). Modeling of spiking-bursting neural behavior using two-dimensional
   map. *Phys Rev E* 65:041922.

8. Courbage M, Nekorkin VI (2010). Map-based models in neuroscience. *Int J Bifurc
   Chaos* 20:1631–1651.

9. Aihara K, Takabe T, Toyoda M (1990). Chaotic neural networks. *Phys Lett A*
   144:333–340.

10. Izhikevich EM (2007). *Dynamical Systems in Neuroscience: The Geometry of
    Excitability and Bursting.* MIT Press, Cambridge, MA.

11. Collet P, Eckmann J-P (1980). *Iterated Maps on the Interval as Dynamical Systems.*
    Birkhäuser, Basel.

12. Lichtenberg AJ, Lieberman MA (1992). *Regular and Chaotic Dynamics.* 2nd ed.
    Springer-Verlag, New York.

---

*Document verified against Rust source `engine/src/neurons/maps.rs:164–197`.
All equations, parameters, and default values read directly from the implementation.*
