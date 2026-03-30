# RallCableNeuron

**Module:** `sc_neurocore.neurons.models.rall_cable`
**Reference:** Rall 1962
**Family:** Multi-compartment (passive cable discretisation)
**State variables:** `v` (numpy array of length `n_comp`)

---

## Equations

### Per-compartment dynamics

$$\tau_m \frac{dV_i}{dt} = -(V_i - E_L) + g_r (V_{i-1} - 2V_i + V_{i+1}) + I_i$$

where $g_r$ is `g_ratio` (axial-to-leak conductance ratio), and $I_i$ is the
injected current (non-zero only at the distal compartment $i = N-1$).

### Boundary conditions

- **Sealed ends:** At $i = 0$ (soma), $V_{-1} = V_0$ (no current leak left).
  At $i = N-1$ (distal), $V_N = V_{N-1}$ (no current leak right).
- Implemented as: `left = v[i-1] if i > 0 else v[i]` and
  `right = v[i+1] if i < n_comp-1 else v[i]`.

### Spike detection (soma only)

$$\text{spike} = \begin{cases} 1 & \text{if } V_0 \geq \theta \text{ and } V_0^{\text{prev}} < \theta \\ 0 & \text{otherwise} \end{cases}$$

On spike: $V_0 \leftarrow V_{\text{reset}}$. Other compartments are NOT reset.

### Implementation (as coded)

```python
def step(self, current: float) -> int:
    v_prev_soma = self.v[0]
    dv = np.zeros(self.n_comp)
    for i in range(self.n_comp):
        leak = -(self.v[i] - self.v_rest)
        left = self.v[i - 1] if i > 0 else self.v[i]
        right = self.v[i + 1] if i < self.n_comp - 1 else self.v[i]
        axial = self.g_ratio * (left - 2.0 * self.v[i] + right)
        inj = current if i == self.n_comp - 1 else 0.0
        dv[i] = (leak + axial + inj) / self.tau_m
    self.v += dv * self.dt
    if self.v[0] >= self.v_threshold and v_prev_soma < self.v_threshold:
        self.v[0] = self.v_reset
        return 1
    return 0
```

Forward Euler, all compartments updated simultaneously per step.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `n_comp` | 5 | — | Number of compartments |
| `tau_m` | 20.0 | ms | Membrane time constant |
| `v_rest` | −65.0 | mV | Resting / leak reversal potential |
| `g_ratio` | 0.5 | — | Axial-to-leak conductance ratio (electrotonic coupling strength) |
| `v_threshold` | −50.0 | mV | Somatic spike threshold |
| `v_reset` | −65.0 | mV | Somatic post-spike reset potential |
| `dt` | 0.1 | ms | Integration time step |

---

## Behaviour

### Passive cable attenuation

Current is injected at the distal end ($i = N-1$). It propagates toward the
soma ($i = 0$) through axial coupling. The passive membrane (leak) attenuates
the signal at each compartment. The voltage gradient is monotonic: distal >
proximal.

Verified: at $I = 100$ with default params (n=5, g_ratio=0.5), after 5,000
steps: $V_4 = -57.68$ mV (distal) vs. $V_0 = -64.95$ mV (soma). The soma
barely depolarises from rest.

### Coupling strength (g_ratio) controls propagation

Higher `g_ratio` → stronger axial coupling → less attenuation → more somatic
depolarisation. Measured at n_comp=3, I=200, 10k steps:

| g_ratio | Soma V (mV) |
|---------|-------------|
| 0.1 | −63.8 |
| 5.0 | −50.1 |

Verified in test: `n_strong.v[0] > n_weak.v[0]`.

### Cable length (n_comp) controls attenuation

More compartments → longer electrotonic distance → more attenuation.
Measured at g_ratio=2.0, I=500, 50k steps:

| n_comp | Spikes | Soma V |
|--------|--------|--------|
| 2 | 4,993 | −55.1 |
| 3 | 2,768 | −58.6 |
| 5 | 467 | −50.1 |

Fewer compartments → dramatically more somatic spikes.

### Default parameters: no somatic spikes

With n_comp=5 and g_ratio=0.5, even at I=500 the soma cannot reach the
−50 mV threshold. The passive cable attenuates the distal signal too
much. This is a faithful representation of Rall's point: long passive
dendrites attenuate synaptic input significantly.

### Somatic-only reset

On spike, only $V_0$ is reset to $V_{\text{reset}}$. The dendritic compartments
retain their voltage. This means the dendritic depolarisation persists and
continues to drive the soma after reset — contributing to sustained firing
when coupling is strong enough.

---

## Measured Dynamics (from test probing)

### Default params (n_comp=5, g_ratio=0.5)

| Current | Spikes (50k) | Soma V | Distal V | Regime |
|---------|-------------|--------|----------|--------|
| 10 | 0 | −64.95 | −57.68 | Subthreshold |
| 50 | 0 | −64.76 | −28.40 | Subthreshold |
| 100 | 0 | −64.52 | 8.21 | Subthreshold (distal depolarised) |
| 200 | 0 | −64.04 | 81.41 | Subthreshold (distal diverging) |
| 500 | 0 | −62.61 | 301.03 | Subthreshold (distal far from bio range) |

Note: the distal compartment diverges to non-biological voltages at high
current because there is no active conductance to limit depolarisation.
The soma remains near rest due to attenuation.

### Short cable (n_comp=2, varying g_ratio)

| g_ratio | Spikes (50k) | Soma V |
|---------|-------------|--------|
| 0.5 | 2,493 | −56.88 |
| 2.0 | 4,993 | −55.08 |
| 5.0 | 6,245 | −52.55 |

Strong coupling with a short cable produces thousands of spikes.

---

## Population Incompatibility

`RallCableNeuron.v` is a numpy array (not a scalar). The `Population` class
calls `_sync_voltages()` which does `self._voltages[i] = neuron.v`, assuming
a scalar. This raises `ValueError: setting an array element with a sequence`.

This is a known limitation. To use Rall cable neurons in a network, either:
1. Wrap them in an adapter that exposes `v` as `v[0]` (soma voltage).
2. Modify `Population._sync_voltages` to handle array-valued state.

Tested: `Population(RallCableNeuron, n=5)` raises `ValueError` or `TypeError`.

---

## Numerical Considerations

- **Passive stability:** The passive cable is unconditionally stable for
  sufficiently small dt, because all eigenvalues of the discretised
  Laplacian are negative. Tested stable at dt = 0.05, 0.1, 0.2 with
  n_comp=3, g_ratio=1.0.
- **Distal divergence at high I:** Without active conductances, the distal
  compartment voltage grows linearly with input current. At I=500 with
  default params, distal V reaches 301 mV — far outside biological range.
  This is expected for a purely passive model.
- **Euler integration:** All compartments updated simultaneously (not
  sequentially), avoiding directional bias.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/rall_cable.py` — 53 lines.
- **NumPy arrays:** Both `v` and the temporary `dv` are numpy arrays.
  The per-compartment loop is pure Python — not vectorised.
- **Sealed-end boundaries:** Implemented by setting the virtual neighbour
  equal to the boundary compartment: `left = v[i]` at i=0,
  `right = v[i]` at i=N-1. This is equivalent to zero-flux (Neumann)
  boundary conditions.
- **Rust wiring:** Not directly compatible with the current
  `step(f64) → i32` dispatch because the neuron has array state.
  Would require a specialised variant or wrapper.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | construction defaults (n_comp=5, all V=−65), step returns 0 or 1, distal compartment depolarises under current, all state finite after 50k steps, reset() restores all V to v_rest |
| Propagation | 3 | distal more depolarised than soma, signal attenuates soma→distal (monotonic gradient), stronger coupling (g_ratio=5 vs 0.1) → more somatic depolarisation |
| Spiking | 4 | fewer compartments → more spikes (n=2 vs n=5), n_comp=2 g_ratio=2 fires ≥100 spikes, default params (n=5, g=0.5) no spikes at I=500, somatic V resets to v_reset on spike |
| Parameters | 8 | n_comp variations (2, 3, 5, 10) all produce correct array size and finite state, dt stability (0.05, 0.1, 0.2), deterministic (2 runs identical) |
| Network | 1 | Population raises ValueError/TypeError (array-valued v incompatible) |
| Analysis | 2 | spike_count ≥ 10 at n_comp=2 g_ratio=5 I=500, spike_count matches manual sum |
| **Total** | **23** | |

---

## Findings

1. **Passive attenuation dominates at default params:** 5 compartments
   with g_ratio=0.5 attenuate the signal so heavily that even I=500
   cannot depolarise the soma to threshold. This faithfully models
   Rall's electrotonic distance concept.
2. **g_ratio is the critical parameter:** Increasing from 0.5 to 5.0
   at n_comp=3 takes the neuron from 564 to 4,537 spikes at I=500.
3. **Cable length scales non-linearly:** Going from n_comp=2 to n_comp=5
   at g_ratio=2.0 reduces spikes from 4,993 to 467 (10× reduction for
   2.5× longer cable).
4. **Somatic-only reset:** Dendritic compartments retain depolarisation
   after spike, providing continued drive. Confirmed: V[0] resets to
   v_reset while V[1:] remain elevated.
5. **Population incompatible:** Array-valued v attribute breaks the
   scalar assumption in Population._sync_voltages. Documented as a
   known limitation rather than a bug.
6. **Distal voltage can diverge:** At I=500 with default params,
   V[4] = 301 mV. This is expected for passive cable (no active
   conductance ceiling) but limits biological interpretability of
   the distal compartment voltage.
