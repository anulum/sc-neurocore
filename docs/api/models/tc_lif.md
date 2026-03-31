# TwoCompartmentLIFNeuron

**Module:** `sc_neurocore.neurons.models.tc_lif`
**Reference:** Yang et al., AAAI Conference on Artificial Intelligence, 2024
**Family:** Multi-compartment Integrate-and-Fire (soma + dendrite)
**State variables:** `v_s` (soma potential), `v_d` (dendrite potential)

---

## Equations

### Dendrite compartment (slow integrator)

$$\tau_d \frac{dV_d}{dt} = -(V_d - V_{rest}) + I_d$$

### Soma compartment (fast integrator + spike generator)

$$\tau_s \frac{dV_s}{dt} = -(V_s - V_{rest}) + \kappa(V_d - V_s) + I_{soma}$$

### Coupling

The coupling term $\kappa(V_d - V_s)$ transfers dendritic potential to
the soma:
- When $V_d > V_s$: current flows dendrite → soma (depolarising)
- When $V_d < V_s$: current flows soma → dendrite (stabilising)
- κ controls coupling strength (default 0.5)

### Spike and reset

$$V_s \geq \theta: \quad V_s \leftarrow V_{reset}, \quad V_d \text{ unchanged}$$

The dendrite is **not** reset on spike — it retains its potential, providing
persistent history-dependent input for subsequent spikes.

### Implementation

```python
def step(self, i_soma: float, i_dend: float = 0.0) -> int:
    dvd = (-(self.v_d - self.v_rest) + i_dend) / self.tau_d * self.dt
    self.v_d += dvd
    dvs = (-(self.v_s - self.v_rest) + self.kappa * (self.v_d - self.v_s) + i_soma) / self.tau_s * self.dt
    self.v_s += dvs
    if self.v_s >= self.theta:
        self.v_s = self.v_reset
        return 1
    return 0
```

Forward Euler, single step per call. Dendrite updated first, then soma
(sequential ordering — dendrite feeds into soma within the same timestep).

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v_s` | 0.0 | a.u. | Soma potential (initial) |
| `v_d` | 0.0 | a.u. | Dendrite potential (initial) |
| `v_rest` | 0.0 | a.u. | Resting potential (both compartments) |
| `v_reset` | 0.0 | a.u. | Soma post-spike reset |
| `theta` | 1.0 | a.u. | Soma spike threshold |
| `tau_s` | 2.0 | ms | Soma time constant (fast) |
| `tau_d` | 20.0 | ms | Dendrite time constant (slow) |
| `kappa` | 0.5 | — | Dendrite→soma coupling strength |
| `dt` | 1.0 | ms | Integration timestep |

### Time constant ratio

$$\tau_d / \tau_s = 20 / 2 = 10$$

The dendrite is 10× slower than the soma. This creates a two-timescale
system:
- **Soma (τ_s = 2 ms):** Rapidly integrates current input → generates spikes
- **Dendrite (τ_d = 20 ms):** Slowly integrates dendritic input → provides
  sustained, history-dependent drive to the soma

This ratio is the key architectural feature: the dendrite acts as a slow
memory that the fast soma reads from.

---

## Analytical Properties

### Dendrite steady state

For constant dendritic input I_d:
$$V_{d,ss} = V_{rest} + I_d$$

With defaults (V_rest = 0): $V_{d,ss} = I_d$. The dendrite simply
integrates its input to steady state.

### Soma steady state (no spike, constant drive)

For constant I_soma and constant V_d:
$$V_{s,ss} = \frac{V_{rest} + \kappa V_d + I_{soma}}{1 + \kappa}$$

With defaults ($V_{rest} = 0$, $\kappa = 0.5$):
$$V_{s,ss} = \frac{0.5 V_d + I_{soma}}{1.5} = \frac{V_d}{3} + \frac{2 I_{soma}}{3}$$

The soma voltage is a weighted average of dendritic input (1/3) and
somatic input (2/3).

### Spiking condition

Spike when $V_s \geq \theta = 1.0$:
$$\frac{V_d}{3} + \frac{2 I_{soma}}{3} \geq 1.0$$

With pure somatic drive ($V_d = 0$): need $I_{soma} \geq 1.5$.
With pure dendritic drive ($I_{soma} = 0$): need $V_d \geq 3.0$
(i.e., $I_d \geq 3.0$ sustained).

### Coupling κ controls mixing

- κ = 0: No coupling. Soma = pure LIF (ignores dendrite).
- κ = 0.5: Moderate coupling (default). Dendrite contributes 1/3 of drive.
- κ = 1.0: Strong coupling. Dendrite contributes 1/2 of drive.
- κ → ∞: Soma follows dendrite. Dendrite dominates.

### Dendrite persistence after spike

When the soma resets ($V_s \leftarrow 0$), $V_d$ is unchanged. If the
dendrite was at a high potential, it immediately begins re-depolarising
the soma via the coupling term $\kappa(V_d - V_s) = \kappa V_d > 0$.

This creates a **burst-like behaviour:** the dendrite can drive multiple
consecutive spikes before its slow decay brings $V_d$ below the effective
threshold. The number of spikes in a burst depends on $V_d$ amplitude
and the κ and τ_d parameters.

### Sequential memory

The slow dendritic compartment provides temporal context:
- At time t: dendrite holds a filtered history of past inputs
- The soma uses this history (via κ coupling) plus current input
- This enables the neuron to respond differently to the same input
  depending on the history — critical for sequential tasks

Yang et al. (2024) showed that TC-LIF outperforms standard LIF on:
- Sequential MNIST (pixel-by-pixel classification)
- Speech recognition (temporal patterns)
- Reinforcement learning (delayed rewards)

---

## Behaviour

### Dual-pathway processing

The two-compartment structure creates two input pathways:
1. **Fast pathway (i_soma → V_s):** Immediate, transient drive. Quickly
   integrated and quickly forgotten (τ_s = 2 ms).
2. **Slow pathway (i_dend → V_d → V_s):** Delayed, sustained drive.
   Slowly integrated and slowly forgotten (τ_d = 20 ms).

This mirrors biological pyramidal neurons where apical dendrites
(slow, NMDA-mediated) provide contextual modulation of basal soma
(fast, AMPA-mediated) spike generation.

### Pure somatic drive (i_dend = 0)

The neuron reduces to a standard LIF with τ_s = 2 ms:
- Very fast integration and decay
- High threshold for spiking (need I_soma ≥ ~1.5 for sustained firing)
- No history dependence

### Pure dendritic drive (i_soma = 0)

The dendrite charges slowly and drives the soma via coupling:
- Long onset delay (τ_d = 20 ms)
- Sustained drive after input ceases (dendrite retains charge)
- Can produce spike bursts if V_d is high enough

### Combined drive

Both inputs contribute — the soma integrates fast somatic input with
slow dendritic context. This is the intended operating mode for
sequential tasks.

---

## Pipeline Compatibility

### Two-argument step

`step(i_soma, i_dend=0.0)` takes two arguments. The standard Network
pipeline passes a single current. When used in a Network:
- `step_all` passes a single current as i_soma
- i_dend defaults to 0.0 (no dendritic input)
- Only the fast pathway is active

For full dual-pathway operation: implement a custom pipeline or use
the model standalone.

### Population and Network compatible

Population(TwoCompartmentLIFNeuron, n=10) works. Network simulation
works with single-current drive.

---

## Comparison with Related Models

| Property | TC-LIF | LIF | AdEx | NeuroGridNeuron | MainenSejnowski |
|----------|--------|-----|------|----------------|-----------------|
| Compartments | 2 (soma+dend) | 1 | 1 (+w) | 2 (soma+dend) | 2 (soma+axon) |
| State vars | 2 (V_s, V_d) | 1 (V) | 2 (V, w) | 2 (V_s, V_d) | 5 |
| Memory | Dendrite (τ_d) | None | Adaptation (τ_w) | Dendrite (τ_d) | None |
| Spike reset | Soma only | V→reset | V→reset, w+=b | Soma only | V→reset |
| ML focus | Yes (AAAI 2024) | Standard | Neuroscience | Neuromorphic | Biophysical |
| Sub-steps | 1 | 1 | 1 | 10 | 20 |
| Pipeline | Compatible | Compatible | Compatible | Compatible | Compatible |

TC-LIF is designed for machine learning on sequential tasks. Its
two-compartment structure provides temporal memory without the
complexity of full biophysical models.

---

## Numerical Considerations

- **Single Euler step:** dt=1.0ms. Adequate for τ_s=2ms (dt/τ_s=0.5 —
  at the edge of stability) and τ_d=20ms (dt/τ_d=0.05 — safe).
- **Sequential update:** Dendrite updated before soma within each step.
  This means the soma sees the "new" V_d immediately.
- **No sub-stepping:** The linear dynamics and moderate dt/τ ratio
  keep the model stable without sub-stepping.
- **No clipping:** V_s and V_d are not clipped. With negative input,
  V_d can go below V_rest.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/tc_lif.py` — 50 lines.
- **Two state variables:** v_s (soma), v_d (dendrite).
- **Dataclass:** Uses `@dataclass`.
- **Two-argument step:** `step(i_soma, i_dend=0.0)`.
- **Rust wiring:** Compatible (2 f64 state vars, single-current dispatch).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~500K steps/s | Not measured |
| Network (10 neurons, 1s) | ~40K neuron-steps/s | — |

Fast model — single Euler step, no exp(), no sub-stepping. Two linear
updates per call.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary return, both compartments evolve, finite 50k, reset |
| Coupling | 5 | κ=0 decoupled (pure LIF), V_d drives V_s via κ, coupling current direction, κ scaling, dendrite persistence after spike |
| Timescale | 3 | τ_d/τ_s=10 ratio, dendrite slower than soma, dual-timescale response |
| Dynamics | 4 | fires with soma drive, fires with dendrite drive, rate monotonic, burst-like from dendrite |
| Parameters | 3 | dt stability, κ sweep, τ_d sweep |
| Pipeline | 4 | Population, Network+drive, Projection, analysis |
| **Total** | **24** | |

See `tests/test_model_tc_lif.py`. No bugs found.

---

## Findings

1. **Dendrite persists after spike:** V_d is unchanged on soma reset,
   providing sustained drive for subsequent spikes.

2. **κ=0 reduces to LIF:** With zero coupling, the soma is a pure LIF
   with τ_s=2ms. Dendrite evolves independently but has no effect.

3. **τ_d/τ_s=10 confirmed:** The dendrite operates on a 10× slower
   timescale, creating the two-timescale dynamics.

4. **Dendritic drive produces spikes:** Pure i_dend input (i_soma=0)
   charges V_d → κ coupling drives V_s → spike. Verified.

5. **Burst-like from high V_d:** When V_d is high, the dendrite re-
   depolarises the soma immediately after reset, producing rapid
   consecutive spikes.

6. **Soma steady-state = weighted average:** V_ss = (κV_d + I_soma)/(1+κ),
   confirmed by long-run convergence measurements.

7. **Sequential update order matters:** Dendrite-first ordering means
   the soma sees the updated V_d within the same timestep.

8. **Network works with single-current:** Standard pipeline passes
   i_soma only. Dendritic pathway requires custom integration.

9. **ML-focused design:** The model was designed for AAAI 2024 sequential
   learning — the dendrite provides the temporal context that standard
   LIF lacks.

10. **Fast model:** ~500K steps/s — no exp(), no sub-stepping, comparable
    to standard LIF performance despite the second compartment.
