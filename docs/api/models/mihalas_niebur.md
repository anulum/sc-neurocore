# MihalasNieburNeuron

**Module:** `sc_neurocore.neurons.models.mihalas_niebur`
**Reference:** Mihalas & Niebur 2009
**Family:** Integrate-and-fire (generalised, 20 patterns)
**State variables:** `v`, `theta` (threshold), `i1` (fast current), `i2` (slow current)

## Equations

$$\tau_v \frac{dV}{dt} = -(V - V_r) + I_1 + I_2 + I_{ext}$$
$$\tau_\theta \frac{d\theta}{dt} = \theta_\infty - \theta + a(V - V_r)$$
$$\tau_1 \frac{dI_1}{dt} = -I_1$$
$$\tau_2 \frac{dI_2}{dt} = -I_2$$

On spike: $V \to V_{reset}$, $\theta \leftarrow \max(\theta, \theta_{reset})$,
$I_1 \leftarrow I_1 + r_1$, $I_2 \leftarrow I_2 + r_2$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_v` | 10.0 | Membrane time constant |
| `tau_theta` | 100.0 | Threshold adaptation time |
| `tau_1` | 10.0 | Fast current decay |
| `tau_2` | 200.0 | Slow current decay |
| `a` | 0.0 | Voltage → threshold coupling |
| `r1` | 0.0 | Spike → fast current increment |
| `r2` | 0.0 | Spike → slow current increment |

## Behaviour

- **20 spike patterns:** Different (a, r1, r2) configs produce tonic,
  phasic, bursting, accommodation, rebound, etc.
- **Dynamic threshold:** a > 0 raises threshold with voltage (accommodating).
- **Two adaptation currents:** Fast (tau_1=10) and slow (tau_2=200)
  adaptation, independently configurable.
- **Default config:** No adaptation (a=r1=r2=0) → pure LIF behaviour.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 13 | construction, step binary, subthreshold, spikes, dynamic theta, adaptation i1, i1 decay, i2 slower, rate increase, tonic config, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **16** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~185K steps/s |
| Spikes (10K steps, I=5.0) | 3333 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`MihalasNieburNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
3333 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(MihalasNieburNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~185K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
