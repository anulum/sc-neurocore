# SPDX-License-Identifier: AGPL-3.0-or-later
# BertramPhantomBurster

**Module:** `sc_neurocore.neurons.models.bertram_phantom`
**Reference:** Bertram, Satin, Zhang, Smolen & Sherman, Biophys. J. 87(5), 2004; updated 2008
**Family:** Biophysical conductance-based (3-ODE, pancreatic β-cell, phantom bursting)
**State variables:** `v` (membrane potential), `s1` (slow variable 1), `s2` (slow variable 2, ultra-slow)

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -I_{Ca} - I_K - I_{s1} - I_{s2} - I_L + I_{ext}$$

### Five ionic currents

$$I_{Ca} = g_{Ca} \, m_\infty(V) \, (V - E_{Ca})$$
$$I_K = g_K \, n_\infty(V) \, (V - E_K)$$
$$I_{s1} = g_{s1} \, s_1 \, (V - E_K)$$
$$I_{s2} = g_{s2} \, s_2 \, (V - E_K)$$
$$I_L = g_L \, (V - E_L)$$

### Boltzmann activation functions

$$m_\infty(V) = \frac{1}{1 + \exp((V_m - V)/s_m)}$$
$$n_\infty(V) = \frac{1}{1 + \exp((V_n - V)/s_n)}$$
$$s_{1,\infty}(V) = \frac{1}{1 + \exp((V_{s1} - V)/s_{s1})}$$
$$s_{2,\infty}(V) = \frac{1}{1 + \exp((V_{s2} - V)/s_{s2})}$$

### Dual slow variables

$$\frac{ds_1}{dt} = \frac{s_{1,\infty}(V) - s_1}{\tau_{s1}}$$

$$\frac{ds_2}{dt} = \frac{s_{2,\infty}(V) - s_2}{\tau_{s2}}$$

### Three timescales

| Variable | τ | Role |
|----------|---|------|
| V | ~C_m/g = ~0.5 ms | Fast: spike dynamics |
| s1 | 20,000 ms (20 s) | Slow: burst modulation |
| s2 | 100,000 ms (100 s) | Ultra-slow: episode modulation |

The 200,000:1 ratio between τ_s2 and the effective V timescale is
the widest timescale separation in SC-NeuroCore. The ultra-slow s2
operates on the **minute** timescale.

### Implementation

```python
def step(self, current: float) -> int:
    current = _finite_float("current", current)
    self._validate_state()
    v_prev = self.v
    v, s1, s2 = self._validate_candidate(*self._rk4_candidate(current))
    self.v = v
    self.s1 = s1
    self.s2 = s2
    return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0
```

The production step uses classical RK4 over the three-state ODE
`(V, s1, s2)` with current held constant during the call. Parameters and
runtime state fail closed before integration; candidate voltage and gate
states are checked before mutation. The Boltzmann activation is evaluated in
overflow-stable form, which is required for the steep `s2` switch.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −50.0 | mV | Membrane potential |
| `s1` | 0.1 | — | Slow variable 1 |
| `s2` | 0.1 | — | Ultra-slow variable 2 |
| `g_ca` | 3.6 | nS | Ca²⁺ conductance |
| `g_k` | 10.0 | nS | K⁺ delayed rectifier |
| `g_s1` | 4.0 | nS | Slow K⁺ conductance 1 |
| `g_s2` | 4.0 | nS | Slow K⁺ conductance 2 |
| `g_l` | 0.2 | nS | Leak conductance |
| `e_ca` | 25.0 | mV | Ca²⁺ reversal |
| `e_k` | −75.0 | mV | K⁺ reversal (shared by K, s1, s2) |
| `e_l` | −40.0 | mV | Leak reversal |
| `c_m` | 5.3 | pF | Membrane capacitance |
| `v_m` | −20.0 | mV | m_∞ half-activation |
| `s_m` | 12.0 | mV | m_∞ slope factor |
| `v_n` | −16.0 | mV | n_∞ half-activation |
| `s_n` | 5.6 | mV | n_∞ slope factor |
| `v_s1` | −40.0 | mV | s1_∞ half-activation |
| `s_s1` | 10.0 | mV | s1_∞ slope factor |
| `v_s2` | −42.0 | mV | s2_∞ half-activation |
| `s_s2` | 0.4 | mV | s2_∞ slope factor (**extremely steep**) |
| `tau_s1` | 20,000 | ms | Slow timescale (20 s) |
| `tau_s2` | 100,000 | ms | Ultra-slow timescale (100 s) |
| `dt` | 0.5 | ms | Integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### s2 is extremely steep

s_s2 = 0.4 mV — the steepest Boltzmann in the entire library. The s2_∞
function transitions from 0 to 1 over a **0.8 mV range** (10%–90% ≈
2.2 × s_s2). This means s2 acts as a near-binary switch: either fully
off (V < −42.4) or fully on (V > −41.6).

---

## Analytical Properties

### Phantom bursting mechanism (Bertram et al. 2004)

The model produces bursting through a **phantom slow manifold** — a
geometric mechanism distinct from standard slow-variable bursting:

1. **Neither s1 alone nor s2 alone can produce bursting.** With only s1
   (g_s2=0), the system shows only tonic spiking. With only s2 (g_s1=0),
   the system also shows tonic spiking.

2. **Together, s1 and s2 create a phantom slow manifold.** The combined
   slow dynamics trace a path in (s1, s2) space that passes near the
   bifurcation point of the fast subsystem — close enough to cause
   bursting, but not through an actual slow manifold.

3. **The "phantom" is geometrical:** The slow trajectory in (s1, s2)
   space passes near a region where a slow manifold *would exist* if
   the parameters were slightly different. The proximity to this
   "phantom manifold" creates the alternation between spiking and silence.

### Dual timescale separation

$$\tau_{s2} / \tau_{s1} = 100{,}000 / 20{,}000 = 5$$

The two slow variables operate on different timescales:
- s1 (20 s): modulates individual burst duration and inter-burst interval
- s2 (100 s): modulates **episodic** patterns — groups of bursts separated
  by long silences

This creates a **three-level hierarchy:**
1. Spikes within bursts (~10 ms)
2. Bursts within episodes (~20 s, controlled by s1)
3. Episodes within the recording (~100 s, controlled by s2)

### Both slow currents share E_K reversal

I_s1 and I_s2 both reverse at E_K = −75 mV. They are both **outward
(hyperpolarising)** when V > E_K (which is always true during spiking,
since V_threshold = −20 > −75). This means both slow variables act as
negative feedback — they accumulate during spiking and suppress further
activity.

### Boltzmann midpoints

| Function | Midpoint | Slope | Steepness |
|----------|----------|-------|-----------|
| m_∞ | −20 mV | 12 mV | Moderate (Ca²⁺ activation) |
| n_∞ | −16 mV | 5.6 mV | Steep (K⁺ activation) |
| s1_∞ | −40 mV | 10 mV | Moderate (slow 1) |
| s2_∞ | −42 mV | 0.4 mV | **Extreme** (slow 2, near-binary) |

---

## Behaviour

### Validated RK4 runtime contract

The production implementation advances the Bertram phantom-burster ODE with
classical RK4 over `(V, s1, s2)`. The default state is not documented as an
endogenous compound-bursting fixture; the validated runtime contract is:

- finite, bounded RK4 evolution for the published three-state conductance ODE;
- fail-closed validation for non-finite parameters, invalid gates, invalid
  capacitance/timescales, invalid current, and non-physical candidate states;
- deterministic upward-threshold spike detection from `v_threshold`;
- parity of the Python reference, Julia accelerator mirror, Go service mirror,
  and Rust safety mirror for the same ODE and validation envelope.

The module-owned test suite validates driven threshold-crossing regimes,
including one upward crossing for `current=200.0` over 50,000 steps from the
default state and monotone non-reduction of the crossing count in the tested
high-current sweep.

### Biological context: insulin secretion

Pancreatic beta cells can exhibit compound bursting in response to glucose:

- individual bursts -> calcium oscillations -> pulsatile insulin release;
- episodes -> slow metabolic oscillations -> ultradian insulin rhythm;
- the Bertram phantom-burster equations provide a mechanistic model for this
  phenomenon when parameterized into the appropriate bifurcation regime.

### s_s2 = 0.4 creates ultrasensitive switching

The extreme steepness of s2_inf means that a 1 mV change around V = -42 mV
switches s2 from fully off to fully on. This binary-like behaviour creates
sharp transitions between episodes in parameter regimes that traverse the
phantom slow manifold.

---

## Comparison with Related Models

| Property | BertramPhantom | Chay | ChayKeizer | ShermanRinzelKeizer |
|----------|---------------|------|-----------|-------------------|
| ODEs | 3 | 3 | 3 | 3 |
| Slow variables | 2 (s1, s2) | 1 (Ca2+) | 1 (Ca2+) | 2 (n, s) |
| Ultra-slow | Yes (tau_s2=100s) | No | No | Yes (tau_s) |
| Phantom manifold | Yes | No | No | Related |
| s_s2 steepness | 0.4 mV | n/a | n/a | about 5 mV |
| Reference | Bertram 2004/2008 | Chay 1985 | Chay-Keizer 1983 | Sherman 1988 |

The BertramPhantomBurster remains the temporally widest beta-cell model in
SC-NeuroCore because it contains both slow and ultra-slow gates. Runtime
documentation distinguishes that mathematical capability from the specific
default parameter regime validated in the current tests.

---

## Verification Evidence (Measured 2026-05-31)

### Module-specific test execution

```text
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_bertram_phantom.py -q
76 passed in 217.33s
```

### Pipeline and invariant coverage

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | PASS | default `v=-50`, `s1=0.1`, `s2=0.1` |
| `step()` return contract | PASS | integer upward-threshold crossing flag |
| RK4 derivative contract | PASS | independent reference RK4 candidate matches production |
| State validation | PASS | invalid parameters and corrupted runtime state rejected |
| Candidate validation | PASS | non-finite and out-of-envelope candidates preserve state |
| Boltzmann functions | PASS | midpoint, monotonicity, and steep `s2` switch verified |
| Current balance | PASS | ionic current signs and reversal ordering verified |
| Population/network path | PASS | population, projection, monitor, and analysis surfaces exercised |
| Determinism | PASS | identical initial conditions produce identical trajectories |

### Polyglot parity checks

| Surface | Verification |
|---------|--------------|
| Python reference | module-owned pytest suite, 76 passed |
| Julia accelerator mirror | `include(...)`; one RK4 step validates finite state |
| Go service mirror | `go test src/sc_neurocore/accel/go/services/bertram_phantom.go` |
| Rust safety mirror | `rustc --test .../bertram_phantom.rs`; 3 tests passed |

### Local benchmark artifact

`benchmarks/results/local_i5_11600k_python_2026-05-31_bertram_phantom.json`
records the Python RK4 reference benchmark on a local Intel i5-11600K host:

| Metric | Value |
|--------|------:|
| Workload | 50,000 RK4 steps, 5 repeats |
| Median | 861,425,040 ns |
| Per-step median | 17,228.5008 ns |
| Throughput median | 58,043.3557 steps/s |
| Spikes per repeat | `[1, 1, 1, 1, 1]` |

The benchmark should be rerun whenever the Bertram phantom ODE, validation
envelope, or polyglot mirrors change.

---

## Numerical Considerations

- **RK4 timestep:** `dt = 0.5 ms` is integrated through four ODE stages before
  any state mutation. This removes the single-stage truncation path from the
  production reference.
- **Candidate validation:** `V` must remain finite and within the model
  envelope; `s1` and `s2` must remain finite gates in `[0, 1]` up to roundoff
  tolerance. Invalid candidates fail before mutating state.
- **Boltzmann stability:** activation functions use an overflow-stable sigmoid
  form, required by the steep `s2` slope.
- **Stage cost:** each RK4 call evaluates four derivative stages, so the Python
  path performs 16 Boltzmann evaluations and 20 ionic-current evaluations per
  committed step.
- **Ultra-slow gate:** `tau_s2 = 100,000 ms`; at `dt = 0.5 ms`, substantial
  `s2` movement requires long simulations.

---

## Implementation Notes

- **Python reference:** `src/sc_neurocore/neurons/models/bertram_phantom.py`
- **Julia accelerator mirror:** `src/sc_neurocore/accel/julia/neurons/bertram_phantom.jl`
- **Go service mirror:** `src/sc_neurocore/accel/go/services/bertram_phantom.go`
- **Rust safety mirror:** `src/sc_neurocore/accel/rust/safety/bertram_phantom.rs`
- **Three state variables:** `v`, `s1`, `s2`
- **Mutation policy:** validate parameters and current, compute RK4 candidate,
  validate candidate, then commit state and detect upward threshold crossing

---

## Theoretical Context

### Phantom bursting mechanism

Bertram, Butte, Kiemel & Sherman (1995) introduced the concept of
"phantom bursting" — a bursting mechanism that arises from the
interaction of two slow variables ($s_1$ and $s_2$) operating on
different timescales. Neither variable alone is sufficient to
produce bursting; the burst emerges from their combined effect —
hence "phantom" (the bursting is not attributable to a single slow
process).

### Three-level temporal hierarchy

The model produces oscillations at three timescales:

1. **Spikes** (~ms): Fast Na⁺/K⁺-like dynamics via Boltzmann
   activation/inactivation ($m_\infty$, $n_\infty$)
2. **Bursts** (~seconds): Modulated by $s_1$ ($\tau_{s_1} = 20$ s)
3. **Episodes** (~minutes): Modulated by $s_2$ ($\tau_{s_2} = 100$ s)

This nested structure produces "compound bursting" — bursts of
bursts, or episodes of bursting separated by long silent periods.

### Pancreatic beta cell application

The model was originally developed to explain electrical activity
in pancreatic beta cells, which produce bursting patterns with
periods ranging from seconds to minutes. The glucose-dependent
transition from fast bursting (high glucose) to slow bursting
(moderate glucose) is captured by adjusting the balance between
$s_1$ and $s_2$.

The slow variable $s_2$ corresponds to a very slow metabolic
oscillator (possibly ATP/ADP ratio or glycolytic oscillations),
while $s_1$ corresponds to a faster ionic mechanism (such as
Ca²⁺-dependent K⁺ channels or endoplasmic reticulum Ca²⁺ cycling).

### Diabetes and insulin secretion

Insulin secretion from beta cells is pulsatile — tightly coupled
to the electrical bursting pattern. Glucose raises ATP/ADP ratio →
closes K_ATP channels → depolarises the cell → triggers bursting →
Ca²⁺ influx → insulin exocytosis. The burst period determines
the insulin pulse frequency (~5 min in humans). Type 2 diabetes
is associated with disrupted bursting — loss of the compound
pattern and transition to irregular or continuous spiking. The
Bertram model provides a framework for studying how changes in
the slow metabolic oscillator ($s_2$) affect insulin pulsatility.

### Bursting classification

In the Izhikevich (2000) taxonomy, phantom bursting is a
**fold/fold** burster — the active phase begins and ends at fold
(saddle-node) bifurcations. The compound pattern arises because
the two slow variables move through the bifurcation structure at
different rates, creating nested fold cycles.

### Ultra-steep Boltzmann

The $s_2$ activation has slope $k_{s_2} = 0.4$ mV — the steepest
Boltzmann in the SC-NeuroCore library. This creates a near-binary
switch: $s_2$ transitions from ~0 to ~1 over less than 1 mV. This
sharpness is critical for the episode-level switching — $s_2$ acts
as an almost digital gate on the bursting mechanism.

---

## Usage Examples

### Example 1: Driven RK4 threshold crossing

```python
from sc_neurocore.neurons.models.bertram_phantom import (
    BertramPhantomBurster,
)

neuron = BertramPhantomBurster()
spike_times = []

for t in range(50000):  # 25 seconds at 0.5 ms/step
    spike = neuron.step(200.0)  # driven RK4 regime
    if spike:
        spike_times.append(t * 0.5)  # ms

print(f"Spikes: {len(spike_times)}")
if len(spike_times) > 2:
    isis = [
        spike_times[i + 1] - spike_times[i]
        for i in range(len(spike_times) - 1)
    ]
    print(f"Mean ISI: {sum(isis) / len(isis):.3f} ms")
```

### Example 2: Slow variable dynamics

```python
from sc_neurocore.neurons.models.bertram_phantom import (
    BertramPhantomBurster,
)

neuron = BertramPhantomBurster()
for _ in range(200000):  # 100 seconds
    neuron.step(0.0)

print(f"V = {neuron.v:.1f} mV")
print(f"s1 = {neuron.s1:.4f} (tau = 20 s)")
print(f"s2 = {neuron.s2:.4f} (tau = 100 s)")
```

### Example 3: Beta cell network

```python
from sc_neurocore.network import Network, Population
from sc_neurocore.neurons.models.bertram_phantom import (
    BertramPhantomBurster,
)
from sc_neurocore.monitors import SpikeMonitor
from sc_neurocore.analysis import spike_count

islet = Population(BertramPhantomBurster, n=10)
net = Network()
net.add_population("beta_cells", islet)

mon = SpikeMonitor()
net.add_monitor("spikes", mon, source="beta_cells")

net.run(duration=60.0)  # 1 minute
print(f"Total spikes: {spike_count(mon)}")
```

---

## Technical Reference

### Polyglot parity

| Aspect | Python | Julia | Go | Rust safety |
|--------|--------|-------|----|-------------|
| State variables | `v`, `s1`, `s2` | same | same | same |
| Integrator | RK4 | RK4 | RK4 | RK4 |
| Boltzmann | overflow-stable sigmoid | same | same | same |
| Validation | parameters, state, candidate | same envelope | same envelope | same envelope |
| Mutation | candidate validated before commit | same | same | same |

Parity is maintained across the Python reference and the accelerator/safety
mirrors for the ODE, default parameters, validation envelope, and spike flag
contract.

### Source files

| File | Description |
|------|-------------|
| `src/sc_neurocore/neurons/models/bertram_phantom.py` | Python reference |
| `src/sc_neurocore/accel/julia/neurons/bertram_phantom.jl` | Julia accelerator mirror |
| `src/sc_neurocore/accel/go/services/bertram_phantom.go` | Go service mirror |
| `src/sc_neurocore/accel/rust/safety/bertram_phantom.rs` | Rust safety mirror |
| `tests/test_model_bertram_phantom.py` | Module-owned behavioural and numerical tests |

---

## Performance Benchmarks

### Local Python RK4 benchmark (i5-11600K, measured 2026-05-31)

| Metric | Value |
|--------|------:|
| Workload | `bertram_phantom_rk4_50k_steps` |
| Repeats | 5 |
| Median | 861,425,040 ns |
| Per-step median | 17,228.5008 ns |
| Throughput median | 58,043.3557 steps/s |

The stored benchmark artifact is
`benchmarks/results/local_i5_11600k_python_2026-05-31_bertram_phantom.json`.
It measures the Python reference RK4 path, not the Rust engine Criterion path.

---

## Citations

1. Bertram R, Butte MJ, Kiemel T, Sherman A (1995). Topological and
   phenomenological classification of bursting oscillations. *Bull
   Math Biol* 57(3):413–439.
   DOI: [10.1007/BF02460633](https://doi.org/10.1007/BF02460633)

2. Sherman A, Rinzel J, Keizer J (1988). Emergence of organized
   bursting in clusters of pancreatic beta-cells by channel sharing.
   *Biophys J* 54(3):411–425.
   DOI: [10.1016/S0006-3495(88)82975-0](https://doi.org/10.1016/S0006-3495(88)82975-0)

3. Bertram R, Sherman A (2004). A calcium-based phantom bursting
   model for pancreatic islets. *Bull Math Biol* 66(5):1313–1344.
   DOI: [10.1016/j.bulm.2003.12.005](https://doi.org/10.1016/j.bulm.2003.12.005)

4. Izhikevich EM (2000). Neural excitability, spiking and bursting.
   *Int J Bifurcat Chaos* 10(6):1171–1266.
   DOI: [10.1142/S0218127400000840](https://doi.org/10.1142/S0218127400000840)

5. Rinzel J (1987). A formal classification of bursting mechanisms
   in excitable systems. Springer, pp. 267–281.
   DOI: [10.1007/978-3-642-93360-8_26](https://doi.org/10.1007/978-3-642-93360-8_26)

6. Rorsman P, Ashcroft FM (2018). Pancreatic β-cell electrical
   activity and insulin secretion: of mice and men. *Physiol Rev*
   98(1):117–214.
   DOI: [10.1152/physrev.00008.2017](https://doi.org/10.1152/physrev.00008.2017)
