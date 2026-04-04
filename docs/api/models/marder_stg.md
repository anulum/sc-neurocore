# MarderSTGNeuron

**Module:** `sc_neurocore.neurons.models.marder_stg`
**Reference:** Marder & Selverston 1992
**Family:** Conductance-based (CPG oscillator)
**State variables:** `v`, `m_na`, `h_na`, `m_cat`, `h_cat`, `m_cas`, `m_a`, `h_a`, `m_kd`, `m_h`, `ca`

## Equations

8 ionic currents: Na, CaT, CaS, A, KCa, Kd, Ih, leak.

$$I_{total} = -I_{Na} - I_{CaT} - I_{CaS} - I_A - I_{KCa} - I_{Kd} - I_H - I_L + I_{ext}$$

Calcium dynamics: $d[Ca]/dt = -f_{Ca}(I_{CaT}+I_{CaS}) - d_{Ca}[Ca]$.

Boltzmann activation: $m_\infty = 1/(1 + \exp((V_{1/2}-V)/k))$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_na` | 200.0 | Na conductance |
| `g_cat` | 2.5 | T-type Ca conductance |
| `g_cas` | 4.0 | S-type Ca conductance |
| `g_a` | 50.0 | A-type K conductance |
| `g_kca` | 25.0 | Ca-activated K conductance |
| `g_kd` | 75.0 | Delayed-rectifier K |
| `g_h` | 0.01 | Ih (HCN) conductance |
| `g_l` | 0.01 | Leak conductance |
| `ca_decay` | 0.02 | Ca clearance rate |

## Behaviour

- **Intrinsic oscillator:** Fires at I=0 — central pattern generator (CPG)
  for the crustacean pyloric rhythm.
- **8 ionic currents:** Most biophysically detailed model in the library.
- **Ca-dependent KCa:** [Ca] accumulates via CaT + CaS currents and
  activates KCa, providing slow burst modulation.
- **11 state variables:** v + 8 gating + [Ca].

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, intrinsic oscillation, Ca dynamics, Ca non-negative, 8 currents, Boltzmann, 11 state vars, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **13** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~37K steps/s |
| Spikes (10K steps, I=5.0) | 26 |
| State stability (20K steps) | PASS |
| Rust parity | PASS |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`MarderSTGNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
26 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(MarderSTGNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**PASS** — spike counts within 15% tolerance.

---

## Findings (measured 2026-04-04)

1. Throughput: ~37K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: PASS
4. Numerical stability confirmed over 20K steps
