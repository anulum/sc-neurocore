# COBALIFNeuron

**Module:** `sc_neurocore.neurons.models.coba_lif`
**Reference:** Destexhe, Rudolph & Paré, Nat. Rev. Neurosci. 4, 2003; Brette et al., J. Comput. Neurosci. 23(3), 2007
**Family:** Integrate-and-fire (conductance-based synaptic input)
**State variables:** `v` (membrane potential), `g_e` (excitatory conductance), `g_i` (inhibitory conductance)

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -g_L(V - E_L) - g_e(V - E_e) - g_i(V - E_i) + I$$

### Synaptic conductance decay

$$\frac{dg_e}{dt} = -\frac{g_e}{\tau_e}, \quad \frac{dg_i}{dt} = -\frac{g_i}{\tau_i}$$

### Spike and reset

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}$$

### Conductance injection

Synaptic events are modelled as instantaneous conductance increments:

$$g_e \leftarrow g_e + \Delta g_e, \quad g_i \leftarrow g_i + \Delta g_i$$

These delta values are passed as extra parameters to `step()`.

### Implementation Contract

```python
def step(self, current: float, delta_ge: float = 0.0, delta_gi: float = 0.0) -> int:
    g_e_pre = self.g_e + delta_ge
    g_i_pre = self.g_i + delta_gi
    v_next, g_e_next, g_i_next = self._rk4_candidate(
        self.v, g_e_pre, g_i_pre, current
    )
    if v_next >= self.v_threshold:
        self.v = self.v_reset
        self.g_e = g_e_next
        self.g_i = g_i_next
        return 1
    self.v = v_next
    self.g_e = g_e_next
    self.g_i = g_i_next
    return 0
```

The implementation applies conductance injections before integration, advances
the coupled `(v, g_e, g_i)` ODE with a fourth-order Runge-Kutta candidate, and
commits that candidate only after finite-value and safety-envelope checks pass.
A spike is evaluated against the RK4 voltage candidate; reset changes only
`v`, while the RK4 conductance candidates are retained.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential |
| `g_e` | 0.0 | nS | Excitatory synaptic conductance |
| `g_i` | 0.0 | nS | Inhibitory synaptic conductance |
| `c_m` | 200.0 | pF | Membrane capacitance |
| `g_l` | 10.0 | nS | Leak conductance |
| `e_l` | −65.0 | mV | Leak reversal potential |
| `e_e` | 0.0 | mV | Excitatory reversal potential |
| `e_i` | −80.0 | mV | Inhibitory reversal potential |
| `tau_e` | 5.0 | ms | Excitatory conductance time constant |
| `tau_i` | 10.0 | ms | Inhibitory conductance time constant |
| `v_threshold` | −50.0 | mV | Spike threshold |
| `v_reset` | −65.0 | mV | Post-spike reset potential |
| `dt` | 0.1 | ms | Integration timestep |

### C_m = 200 pF (large capacitance)

The membrane capacitance of 200 pF is typical for cortical pyramidal
neurons (large soma + dendrites). This requires substantial current
(≈500+ pA) or strong synaptic conductance to reach threshold. Smaller
neurons (interneurons) typically have C_m ≈ 50–100 pF.

### Reversal potentials

- **E_e = 0 mV:** AMPA/NMDA receptor channels (Na⁺/K⁺ mixed). The
  driving force g_e(V − 0) is always negative when V < 0 → inward
  (depolarising) current.
- **E_i = −80 mV:** GABA_A receptor channels (Cl⁻). The driving force
  g_i(V − (−80)) is positive when V > −80 → outward (hyperpolarising)
  current. At V = −80, inhibition has zero effect (shunting).
- **E_l = −65 mV:** Resting leak. No net current at V = E_l.

### Timescale separation

- **τ_e = 5 ms:** AMPA-like fast excitation. Conductance decays to 37%
  in 5 ms.
- **τ_i = 10 ms:** GABA_A-like slower inhibition. Conductance decays to
  37% in 10 ms. Inhibition outlasts excitation — biophysically realistic.

---

## Analytical Properties

### Conductance-based vs current-based LIF

The key difference from standard (current-based) LIF:

| Feature | Current-based LIF | COBA LIF |
|---------|-------------------|----------|
| Synaptic current | I_syn = w × spike | I_syn = g × (V − E_rev) |
| V dependence | None | Yes (driving force) |
| Near threshold | Full effect | Reduced (V close to E_e) |
| Near E_i | Full effect | Zero (shunting) |
| Biophysics | Abstract | Realistic |

The driving force (V − E_rev) means that:
- Excitation weakens as V → E_e (depolarisation reduces driving force)
- Inhibition weakens as V → E_i (hyperpolarisation reduces driving force)
- This creates **automatic gain control** — the closer V is to the
  reversal, the less effect additional conductance has

### Effective membrane time constant

With active conductances, the effective time constant changes:

$$\tau_{eff} = \frac{C_m}{g_L + g_e + g_i}$$

At rest (g_e = g_i = 0): τ_eff = 200/10 = 20 ms
With g_e = 10 nS: τ_eff = 200/20 = 10 ms (faster)
With g_e = 10, g_i = 10: τ_eff = 200/30 = 6.7 ms (even faster)

High synaptic conductance **shortens the integration time** — the
neuron responds faster but is harder to drive to threshold (because
the driving force decreases near reversal). This is the "high-
conductance state" described by Destexhe et al.

### Steady-state voltage

For constant conductances and current:

$$V_{ss} = \frac{g_L E_L + g_e E_e + g_i E_i + I}{g_L + g_e + g_i}$$

This is a conductance-weighted average of reversal potentials plus
current. With g_e = 0, g_i = 0, I = 0: V_ss = E_L = −65 mV.

### Subthreshold at I = 100

Despite constant input of I = 100, the neuron does NOT spike (verified
by test: 0 spikes in 5000 steps). The steady-state voltage:

$$V_{ss} = \frac{10 \times (-65) + 100}{10} = \frac{-550}{10} = -55 \text{ mV}$$

This is below the threshold of −50 mV. The neuron needs I ≥ 150 for
V_ss to reach threshold. The test uses I=500 for reliable spiking.

### Conductance decay

The conductance equations remain first-order exponential decays:
$$g_e(t + dt) = g_e(t) \cdot \exp(-dt/\tau_e)$$

The production step now computes those conductance candidates through the same
RK4 pass as membrane voltage so the voltage-dependent synaptic current and
conductance decay are integrated as one coupled candidate. At `dt=0.1` and
`tau_e=5`, RK4 retains approximately the same 98.02% excitatory conductance as
the closed-form exponential while preserving one shared integration contract
across Python, Rust, Go, Julia, and Mojo.

---

## Behaviour

### Conductance-based synaptic drive

The COBA model receives synaptic input via conductance increments
(delta_ge, delta_gi) rather than direct current. Each presynaptic
spike adds a fixed conductance quantum, which then decays exponentially.
The resulting current depends on the postsynaptic membrane voltage:

$$I_{syn,e} = g_e \times (V - E_e)$$

This is more biophysically realistic than current injection because:
1. The effect of a synapse depends on the neuron's current state
2. Near reversal potential, additional input has diminishing returns
3. Multiple synaptic inputs interact nonlinearly through voltage

### High-conductance state

When many synapses are active simultaneously, total conductance
(g_L + g_e + g_i) increases dramatically. This produces the
"high-conductance state" characteristic of cortical neurons in vivo:
- Shorter effective time constant
- Reduced input resistance
- Narrower spike threshold window
- More temporally precise spike timing

### step() interface

The `step()` method takes three parameters:
1. `current` (float): Direct current injection (pA)
2. `delta_ge` (float, default 0): Excitatory conductance increment (nS)
3. `delta_gi` (float, default 0): Inhibitory conductance increment (nS)

In the standard Pipeline (Population → Network), only `current` is
used via PoissonInput. The delta_ge/delta_gi parameters are available
for direct neuron manipulation or custom network implementations.

---

## Comparison with Related Models

| Property | COBA LIF | LIF | AdEx | HH |
|----------|----------|-----|------|-------|
| State vars | 3 (V, g_e, g_i) | 1 (V) | 2 (V, w) | 4 (V, m, h, n) |
| Synaptic | Conductance-based | Current-based | Current-based | Conductance-based |
| Driving force | Yes (V − E_rev) | No | No | Yes |
| Exp per step | 2 (g_e, g_i decay) | 0–1 | 1 | 4+ |
| Parameters | 13 | 6–8 | 12 | 15+ |
| Biophysics | Intermediate | Minimal | Minimal | Full |
| step() args | 3 (I, Δg_e, Δg_i) | 1 (I) | 1 (I) | 1 (I) |

COBA LIF sits between simple LIF and full HH — it captures the
essential biophysics of synaptic conductance without modelling
individual ion channel kinetics.

---

## Polyglot Surfaces

The Python model, Rust safety mirror, Go service, Julia kernel, and Mojo
accelerator contract use the same candidate-first RK4 equations:

$$\dot V = \frac{-g_L(V - E_L) - g_e(V - E_e) - g_i(V - E_i) + I}{C_m}$$

$$\dot g_e = -g_e/\tau_e,\quad \dot g_i = -g_i/\tau_i$$

Invalid state, invalid step input, non-finite candidates, negative
conductance candidates, and voltage candidates outside the configured safety
envelope fail closed before mutation on stateful surfaces.

## Local Measured Performance

Measured on `aaarthuus` on 2026-06-18 with
`benchmarks/results/local_python_2026-06-18_coba_lif_rk4.json`. This is a
local, non-isolated regression artefact and is not a production speed claim.

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 4402.640635 | 4346.726280 | 5184.713550 | 2777 |
| Rust safety | 38.702560 | 38.626330 | 39.219430 | 2777 |
| Go service | 53.780000 | 52.530000 | 55.500000 | 2777 |
| Julia kernel | 47.083040 | 46.466800 | 47.172330 | 2777 |
| Mojo kernel | 40.720870 | 39.896560 | 41.193015 | 2777 |

All measured mirrors emitted exactly 2,777 spikes over 200,000 steps at
`current=500.0`, giving zero-tolerance spike parity across the maintained
polyglot surfaces.

---

## Numerical Considerations

- **Coupled RK4 candidate:** V, g_e, and g_i are advanced together so the
  voltage-dependent conductance current is integrated against the same
  candidate trajectory as conductance decay.
- **No exp() calls in the production step:** Conductance decay is represented
  by RK4 derivative evaluations instead of separate exponential decay factors.
- **Fail-closed envelope:** V is not clipped after mutation. Candidate voltage
  must remain in the configured safety envelope before it can be committed.
- **dt = 0.1 ms:** Conservative timestep for the LIF-class ODE.
  The effective membrane time constant (20 ms at rest) is 200× larger
  than dt, ensuring numerical stability.
- **Conductance non-negativity:** delta_ge and delta_gi are typically
  non-negative. Candidate conductances must remain non-negative before commit.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/coba_lif.py`.
- **Three state variables:** v, g_e, g_i.
- **Dataclass:** Uses `@dataclass`.
- **Multi-argument step():** Takes 3 parameters (current, delta_ge,
  delta_gi). In Population context, only current is used.
- **Polyglot wiring:** Python, Rust safety, Go service, Julia, and Mojo share
  the same RK4 derivative equations and spike/reset contract.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Dynamics | 5 | finite rest state, candidate-first RK4 injection, excitation, inhibition, spike reset |
| Reset | 1 | reset restores leak voltage and clears conductances without changing dt |
| Validation | 19 | invalid runtime state, invalid parameters, invalid step inputs, voltage envelope, conductance envelope |
| Polyglot mirrors | 2 | Go and Rust RK4 candidate tests run in their native test harnesses |
| **Python total** | **27** | **Passed on 2026-06-18** |

See `tests/test_model_coba_lif.py`.

---

## Findings (Measured 2026-06-18)

1. **27 Python behavior tests passed.** The current module-specific suite
   covers RK4 conductance injection, excitation, inhibition, spike reset,
   reset semantics, invalid runtime state, invalid parameters, invalid step
   input, and candidate-envelope rejection.

2. **Subthreshold at I=100 confirmed.** V_ss ≈ −55 mV < V_threshold = −50.
   This remains below threshold by the analytical steady-state expression.

3. **Spiking at I=500 is parity-measured.** Python, Rust, Go, Julia, and Mojo
   each emitted 2,777 spikes over 200,000 steps.

4. **Conductance injection is candidate-first.** `delta_ge` and `delta_gi`
   are added to the pre-step state, then the full `(v, g_e, g_i)` candidate is
   advanced with RK4 before any mutation occurs.

5. **Spike reset retains conductance candidates.** When the RK4 voltage
   candidate crosses threshold, only voltage resets; the RK4 conductance
   candidates remain committed.

6. **Invalid state does not mutate.** Non-finite values, negative
   conductances, invalid positive parameters, and out-of-envelope candidates
   fail before state commit.

7. **Native mirror tests pass.** Go and Rust safety tests verify RK4 candidate
   parity, invalid-state preservation, and suprathreshold reset behavior.

8. **Reset restores initial state.** v → E_L (−65), g_e → 0, g_i → 0.

---

## Pipeline Verification

### Test execution

```
27/27 Python behavior tests passed on 2026-06-18.
Go native RK4 tests passed.
Rust safety RK4 tests passed.
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | v=-65, g_e=0, g_i=0 |
| step(I) → int {0,1} | ✓ PASS | Standard binary output |
| step(I, Δg_e, Δg_i) | ✓ PASS | Candidate-first RK4 multi-argument interface |
| Subthreshold (I=100) | ✓ PASS | V_ss < threshold by analytical steady-state |
| Spiking (I=500) | ✓ PASS | 2,777 spikes in 200,000-step parity benchmark |
| g_e/g_i decay | ✓ PASS | Coupled RK4 conductance candidates |
| delta_ge injection | ✓ PASS | Applied before RK4 candidate |
| delta_gi injection | ✓ PASS | Applied before RK4 candidate |
| State finite | ✓ PASS | V, g_e, g_i finite after accepted candidates |
| reset() | ✓ PASS | v→E_L, g_e→0, g_i→0 |
| Python/Rust/Go/Julia/Mojo parity | ✓ PASS | Zero-tolerance spike parity in local benchmark |

### Note on conductance injection

The Pipeline uses only the `current` parameter of step(). The
delta_ge/delta_gi parameters are not utilised by Population/Network —
they exist for direct neuron access or custom network implementations.
To fully exploit COBA's conductance-based synapses, a custom Network
loop that passes conductance deltas per synapse would be needed.

The current local evidence covers the single-neuron physics contract and its
maintained polyglot mirrors. Network-level conductance-routing work remains a
separate pipeline surface because `Population` currently drives this model via
the scalar `current` argument.

---

## Theoretical Context

### The Brette et al. 2007 benchmark

The COBA LIF model is one of two standard benchmarks defined by Brette
et al. (2007) for comparing neural simulation platforms:

1. **CUBA (Current-Based):** LIF with current-based synapses (simpler)
2. **COBA (Conductance-Based):** LIF with conductance-based synapses (this)

The COBA benchmark specifies 4000 excitatory + 1000 inhibitory neurons
with random sparse connectivity. SC-NeuroCore's COBALIFNeuron implements
the single-neuron component of this benchmark.

### Destexhe's high-conductance state

Destexhe, Rudolph & Paré (2003) showed that in vivo cortical neurons
operate in a "high-conductance state" where synaptic conductance
significantly exceeds leak conductance. This state:
- Reduces the effective time constant from ~20 ms to ~5 ms
- Increases temporal precision of spike timing
- Creates a "fluctuation-driven" regime where spikes are triggered by
  transient conductance fluctuations rather than steady depolarisation
- Is fundamentally different from the "mean-driven" regime of in vitro
  preparations

The COBALIFNeuron captures this regime when driven by balanced
excitatory and inhibitory conductance inputs.

### Conductance-based vs current-based in practice

For large network simulations:
- Current-based (CUBA) is faster and simpler (no driving force)
- Conductance-based (COBA) is more realistic but requires knowing
  reversal potentials and synaptic conductance magnitudes
- The choice depends on whether voltage-dependent synaptic effects
  matter for the phenomenon being studied
- For E/I balance studies, COBA is preferred because inhibition
  naturally shunts near E_i, preventing pathological hyperpolarisation

---

## Usage Examples

### Example 1: Basic Python — conductance-driven spiking

```python
from sc_neurocore.neurons.models.coba_lif import COBALIFNeuron

neuron = COBALIFNeuron()

# Drive with constant current
spike_count = 0
for t in range(10000):
    spike = neuron.step(current=500.0)
    spike_count += spike

print(f"Spikes: {spike_count} in 10 s")
print(f"V = {neuron.v:.1f} mV, g_e = {neuron.g_e:.4f}, g_i = {neuron.g_i:.4f}")
```

### Example 2: Advanced Python — excitatory/inhibitory conductance injection

```python
from sc_neurocore.neurons.models.coba_lif import COBALIFNeuron
import numpy as np

neuron = COBALIFNeuron()
rng = np.random.default_rng(42)

# Simulate Poisson-like E/I conductance bombardment
voltages, spikes = [], []
for t in range(5000):
    # Excitatory: ~100 Hz Poisson × 0.5 nS per event
    delta_ge = rng.poisson(0.01) * 0.5
    # Inhibitory: ~25 Hz Poisson × 2.0 nS per event
    delta_gi = rng.poisson(0.0025) * 2.0
    spike = neuron.step(0.0, delta_ge=delta_ge, delta_gi=delta_gi)
    voltages.append(neuron.v)
    spikes.append(spike)

total = sum(spikes)
print(f"Spikes: {total}, rate: {total / 0.5:.1f} Hz")
print(f"Mean V: {np.mean(voltages):.1f} mV (high-conductance state)")
```

### Example 3: PyO3 Rust — high-performance stepping

```rust
use sc_neurocore_engine::neurons::COBALIFNeuron;

let mut neuron = COBALIFNeuron::new();

// Pure current drive (delta_ge=0, delta_gi=0)
let mut spikes = 0;
for _ in 0..100_000 {
    spikes += neuron.step(500.0, 0.0, 0.0);
}
println!("COBA LIF: {spikes} spikes in 100K steps");
println!("V = {:.2} mV, g_e = {:.4}, g_i = {:.4}",
    neuron.v, neuron.g_e, neuron.g_i);

neuron.reset();
assert!((neuron.v - (-65.0)).abs() < 1e-12);
```

---

## Technical Reference

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `step(current, delta_ge=0, delta_gi=0) → int` | 0 or 1 | Advance dt ms, inject conductances, return spike |
| `reset` | `reset() → None` | — | Restore v=E_L, g_e=0, g_i=0 |

### Polyglot RK4 Parity

| Property | Python | Rust/Go/Julia/Mojo | Match |
|----------|--------|--------------------|-------|
| Conductance injection | add Δg_e/Δg_i before candidate | same pre-candidate timing | EXACT |
| Synaptic current | `g_e·(V-E_e) + g_i·(V-E_i)` | same equation | EXACT |
| Voltage derivative | `(-g_L·(V-E_L) - I_syn + I) / C_m` | same equation | EXACT |
| Conductance derivatives | `-g_e/τ_e`, `-g_i/τ_i` | same equations | EXACT |
| Integrator | candidate-first RK4 | candidate-first RK4 | EXACT |
| Spike condition | RK4 `V_next ≥ θ` | RK4 `V_next ≥ θ` | EXACT |
| Reset | `V = V_reset`; retain conductance candidates | same semantics | EXACT |
| Parameters (13) | finite `float` values | finite `f64`/`Float64` values | EXACT |

### NetworkRunner wrapper

The NetworkRunner uses `wrap_3arg!` macro to adapt the 3-argument
`step(current, delta_ge, delta_gi)` to a 1-argument interface by
hardcoding `delta_ge=0.0, delta_gi=0.0`. This means conductance
injection is only available through the Python API, not the
NetworkRunner.

### Supported operations

| Operation | Supported | Notes |
|-----------|-----------|-------|
| Population | Yes | Standard interface |
| Projection | Yes | Self-recurrent tested |
| NetworkRunner | Yes (current only) | `WrCOBALIFCell` wrapper, Δg_e/Δg_i = 0 |
| SpikeMonitor | Yes | Binary spike output |
| PoissonInput | Yes | Tested at 500 Hz |
| PyO3 bridge | Yes | Full 3-arg step() with defaults |
| Conductance injection | Python/Rust/Go/Julia/Mojo | NetworkRunner wrapper remains current-only |

---

## Performance Benchmarks

The current measured benchmark is
`benchmarks/results/local_python_2026-06-18_coba_lif_rk4.json`. It is local
non-isolated regression evidence, not a production throughput claim. The
current table is recorded in [Local Measured Performance](#local-measured-performance).

---

## Citations

1. Destexhe, A., Rudolph, M. & Paré, D. (2003). The high-conductance
   state of neocortical neurons in vivo. *Nature Reviews Neuroscience*,
   4(9), 739–751.
   DOI: [10.1038/nrn1198](https://doi.org/10.1038/nrn1198)

2. Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D.,
   Bower, J. M., … & Destexhe, A. (2007). Simulation of networks of
   spiking neurons: a review of tools and strategies. *Journal of
   Computational Neuroscience*, 23(3), 349–398.
   DOI: [10.1007/s10827-007-0038-6](https://doi.org/10.1007/s10827-007-0038-6)

3. Vogels, T. P. & Abbott, L. F. (2005). Signal propagation and logic
   gating in networks of integrate-and-fire neurons. *Journal of
   Neuroscience*, 25(46), 10786–10795.
   DOI: [10.1523/JNEUROSCI.3508-05.2005](https://doi.org/10.1523/JNEUROSCI.3508-05.2005)

4. Brunel, N. (2000). Dynamics of sparsely connected networks of
   excitatory and inhibitory spiking neurons. *Journal of Computational
   Neuroscience*, 8(3), 183–208.
   DOI: [10.1023/A:1008925309027](https://doi.org/10.1023/A:1008925309027)

5. Kuhn, A., Aertsen, A. & Rotter, S. (2004). Neuronal integration of
   synaptic input in the fluctuation-driven regime. *Journal of
   Neuroscience*, 24(10), 2345–2356.
   DOI: [10.1523/JNEUROSCI.3349-03.2004](https://doi.org/10.1523/JNEUROSCI.3349-03.2004)

6. Rudolph, M. & Destexhe, A. (2003). A fast-conducting, stochastic
   integrative mode for neocortical neurons in vivo. *Journal of
   Neuroscience*, 23(6), 2466–2476.
   DOI: [10.1523/JNEUROSCI.23-06-02466.2003](https://doi.org/10.1523/JNEUROSCI.23-06-02466.2003)
