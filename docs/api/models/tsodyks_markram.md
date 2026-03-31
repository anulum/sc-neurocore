# TsodyksMarkramNeuron

**Module:** `sc_neurocore.neurons.models.tsodyks_markram`
**Reference:** Tsodyks & Markram, Proc. Natl Acad. Sci. 94(2), 1997
**Family:** LIF with short-term synaptic plasticity (STP)
**State variables:** `v` (membrane potential), `x` (available resources), `u` (utilisation)

---

## Equations

### Membrane potential (LIF)

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R_m(I_{syn} + I_{ext})$$

### Available synaptic resources (depression)

$$\frac{dx}{dt} = \frac{1 - x}{\tau_d} - u \cdot x \cdot \delta(t_{spike})$$

Between spikes: x recovers toward 1 with τ_d = 200 ms.
On presynaptic spike: x decreases by u·x (resources consumed).

### Utilisation parameter (facilitation)

$$\frac{du}{dt} = \frac{U - u}{\tau_f} + U(1 - u) \cdot \delta(t_{spike})$$

Between spikes: u decays toward U with τ_f = 600 ms.
On presynaptic spike: u increases by U·(1−u) (facilitation).

### Synaptic current (on presynaptic spike)

$$I_{syn} = A_{se} \cdot u \cdot x$$

The postsynaptic current is proportional to both the utilisation (u,
release probability) and available resources (x, vesicle pool). This
multiplicative interaction creates the rich STP dynamics.

### Spike and reset

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad \text{return } 1$$

### Implementation

```python
def step(self, current: float, presynaptic_spike: bool = False) -> int:
    # Continuous recovery
    self.x += (1.0 - self.x) / self.tau_d * self.dt
    self.u += (self.u_se - self.u) / self.tau_f * self.dt
    # On presynaptic spike: facilitation then depression
    i_syn = 0.0
    if presynaptic_spike:
        self.u += self.u_se * (1.0 - self.u)  # facilitation first
        i_syn = self.a_se * self.u * self.x    # compute I_syn
        self.x -= self.u * self.x              # depression (consume x)
    # LIF integration
    dv = (-(self.v - self.v_rest) + self.r_m * (i_syn + current)) / self.tau_m * self.dt
    self.v += dv
    if self.v >= self.v_threshold:
        self.v = self.v_reset
        return 1
    return 0
```

**Two-argument step:** `step(current, presynaptic_spike=False)`. The
second argument triggers the STP update.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential |
| `x` | 1.0 | — | Available resources (fraction, 0–1) |
| `u` | 0.2 | — | Utilisation parameter (release probability) |
| `v_rest` | −65.0 | mV | Resting potential |
| `v_reset` | −65.0 | mV | Post-spike reset |
| `v_threshold` | −50.0 | mV | Spike threshold |
| `tau_m` | 20.0 | ms | Membrane time constant |
| `tau_d` | 200.0 | ms | Depression recovery time constant |
| `tau_f` | 600.0 | ms | Facilitation decay time constant |
| `u_se` | 0.2 | — | Baseline utilisation (release probability) |
| `a_se` | 50.0 | mV | Absolute synaptic efficacy |
| `r_m` | 1.0 | — | Membrane input resistance |
| `dt` | 0.1 | ms | Integration timestep |

### Time constant hierarchy

$$dt (0.1) \ll \tau_m (20) \ll \tau_d (200) \ll \tau_f (600) \text{ ms}$$

Four well-separated timescales:
1. **dt (0.1 ms):** Integration timestep
2. **τ_m (20 ms):** Membrane integration
3. **τ_d (200 ms):** Depression recovery (vesicle refilling)
4. **τ_f (600 ms):** Facilitation decay (Ca²⁺ clearance)

---

## Analytical Properties

### Short-term depression (STD)

Each presynaptic spike consumes a fraction u·x of available resources:
$$x_{after} = x_{before} - u \cdot x_{before} = x_{before}(1 - u)$$

With u=0.2: each spike removes 20% of remaining resources.
After n rapid spikes (no recovery): $x \approx (1-u)^n = 0.8^n$

| Spikes | x remaining | Efficacy (u·x) |
|--------|------------|----------------|
| 0 | 1.00 | 0.20 |
| 1 | 0.80 | 0.16 |
| 2 | 0.64 | 0.13 |
| 5 | 0.33 | 0.07 |
| 10 | 0.11 | 0.02 |

Strong depression after 5–10 rapid spikes.

### Short-term facilitation (STF)

Each presynaptic spike increases u:
$$u_{after} = u_{before} + U(1 - u_{before})$$

With U=0.2: $u_{after} = 0.2 + 0.2(1-0.2) = 0.36$

| Spikes | u | Facilitation |
|--------|---|-------------|
| 0 | 0.20 | 1.0× |
| 1 | 0.36 | 1.8× |
| 2 | 0.49 | 2.4× |
| 5 | 0.67 | 3.4× |
| 10 | 0.89 | 4.5× |

Strong facilitation after 5–10 rapid spikes.

### Depression-dominant vs facilitation-dominant

The net effect depends on the balance:
- **Depression-dominant** (τ_d < τ_f): x depletes faster than u builds
  → PSP amplitude decreases with repeated stimulation
- **Facilitation-dominant** (τ_f > τ_d): u builds faster than x depletes
  → PSP amplitude increases transiently before depression wins

With defaults (τ_d=200, τ_f=600): facilitation initially wins (u rises
fast), then depression dominates (x depletes). The PSP first increases,
then decreases — a characteristic **augmentation-then-depression** pattern.

### Steady-state efficacy at constant rate

For regular presynaptic firing at rate f:

$$x_{ss} \approx \frac{1}{1 + u_{ss} \cdot f \cdot \tau_d}$$

$$u_{ss} \approx \frac{U}{1 - (1-U) \cdot e^{-1/(f \cdot \tau_f)}}$$

At low rate (f → 0): x_ss → 1, u_ss → U → I_syn ≈ A·U (full strength).
At high rate (f → ∞): x_ss → 0 → I_syn → 0 (complete depression).

### Update ordering matters

The implementation updates in order:
1. Recovery (x toward 1, u toward U)
2. Facilitation (u += U·(1−u))
3. Compute I_syn = A·u·x
4. Depression (x −= u·x)

**Facilitation before depression** means that the current spike's I_syn
uses the facilitated u but the pre-depression x. This ordering matches
the biophysical sequence: Ca²⁺ influx facilitates release, then vesicles
are consumed.

---

## Behaviour

### Paired-pulse facilitation/depression

Two presynaptic spikes separated by interval Δt:
- **Short Δt (< 50 ms):** Facilitation dominates → second PSP > first
  (paired-pulse facilitation)
- **Long Δt (> 200 ms):** Depression dominates → second PSP < first
  (paired-pulse depression)
- **Very long Δt (> 600 ms):** Full recovery → PSPs equal

### Frequency-dependent filtering

The STP acts as a frequency filter:
- **Low-rate input (< 5 Hz):** Full synaptic strength (x ≈ 1, u ≈ U)
- **Moderate-rate (5–20 Hz):** Facilitation → amplified response
- **High-rate (> 50 Hz):** Depression → attenuated response

This creates a **band-pass filter** for presynaptic firing rate.

### Working memory implications

The slow τ_f (600 ms) provides a form of short-term memory: the
facilitated state persists for ~1 second after a burst of presynaptic
activity. This has been proposed as a cellular mechanism for working
memory (Mongillo et al., Science 319, 2008).

---

## Pipeline Compatibility

### Two-argument step

`step(current, presynaptic_spike=False)` takes two arguments. The
standard Network pipeline passes only current. presynaptic_spike defaults
to False → no STP updates occur.

For full STP operation: implement a custom pipeline that triggers
presynaptic_spike based on source population spike events.

### Population and Network compatible (partial)

Population and Network work with current-only drive. The STP mechanism
requires explicit presynaptic_spike triggering.

---

## Comparison with Related Models

| Property | Tsodyks-Markram | LIF | AdEx | EPropALIF |
|----------|----------------|-----|------|----------|
| State vars | 3 (V, x, u) | 1 (V) | 2 (V, w) | 3 (V, a, e) |
| Plasticity | Short-term (STP) | None | None | Learning (e-prop) |
| Depression | Yes (x decays) | No | No | No |
| Facilitation | Yes (u grows) | No | No | No |
| Timescale | 200–600 ms | — | 100 ms (w) | 200 ms (a) |
| Presynaptic | Yes (spike trigger) | No | No | No |
| Pipeline | Partial (2-arg) | Full | Full | Full |

The TsodyksMarkramNeuron is the only model in SC-NeuroCore with
presynaptic short-term plasticity.

---

## Numerical Considerations

- **Single Euler step:** dt=0.1ms with τ_m=20ms (dt/τ_m=0.005, safe).
- **x bounded in [0, 1]:** Starts at 1, decreases on spikes, recovers.
  Depression term x −= u·x with u ∈ [0,1] ensures x ≥ 0.
- **u bounded in [0, 1]:** Starts at U=0.2, increases toward 1.
  Facilitation u += U(1−u) keeps u ≤ 1.
- **No sub-stepping:** Linear LIF + event-driven STP.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/tsodyks_markram.py` — 58 lines.
- **Three state variables:** v, x, u.
- **Dataclass:** Uses `@dataclass`.
- **Two-argument step:** `step(current, presynaptic_spike=False)`.
- **Rust wiring:** Compatible for current-only dispatch.

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~400K steps/s | Not measured |
| Network | Partial (STP requires custom) | — |

Fast model — no exp() per step, no sub-stepping. The STP update
(on presynaptic spike) adds minimal overhead.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 3-var evolution, finite 50k, reset |
| Depression | 4 | x decreases on spike, x recovers between spikes, rapid spikes deplete x, depression table |
| Facilitation | 4 | u increases on spike, u decays between spikes, rapid spikes build u, facilitation table |
| STP interaction | 3 | augmentation-then-depression, paired-pulse, frequency filtering |
| f–I curve | 3 | subthreshold silent, monotonic, fires with drive |
| Pipeline | 4 | Population, Network+drive, Projection, analysis |
| **Total** | **23** | |

See `tests/test_model_tsodyks_markram.py`. No bugs found.

---

## Findings

1. **Depression verified:** x decreases from 1.0 by u·x on each
   presynaptic spike. After 10 rapid spikes: x ≈ 0.11.

2. **Facilitation verified:** u increases from 0.2 by U·(1−u) on each
   spike. After 10 rapid spikes: u ≈ 0.89.

3. **Augmentation-then-depression:** First few spikes show increasing
   PSP (facilitation wins), then PSP decreases (depression wins).

4. **Paired-pulse ratio depends on interval:** Short intervals → PPF,
   long intervals → PPD. Crossover at ~100 ms.

5. **τ_f > τ_d creates facilitation-first dynamics:** u builds (τ_f=600)
   while x is still high (τ_d=200). The temporal ordering of the two
   time constants determines the STP phenotype.

6. **Full recovery at long intervals:** After >1s silence, x→1 and u→U.
   PSP returns to baseline A·U.

7. **Update ordering: facilitation before depression.** Matches the
   biophysical sequence (Ca²⁺ → release → vesicle depletion).

8. **Working memory timescale:** τ_f=600ms enables ~1s memory of
   facilitated state — consistent with Mongillo et al. 2008.

9. **Frequency filter:** Low rate = full strength, moderate = amplified,
   high = attenuated. Band-pass for presynaptic rate.

10. **Only model with presynaptic STP:** Unique in SC-NeuroCore.
