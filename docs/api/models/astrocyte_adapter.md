# AstrocyteNeuron (Adapter)

**Module:** `sc_neurocore.neurons.models.astrocyte_adapter`
**Reference:** Li & Rinzel 1994 (wrapped model); Adapter pattern for pipeline integration
**Family:** Adapter (converts AstrocyteModel Ca²⁺ output to spiking interface)
**State variables:** Delegates to `AstrocyteModel` (ca, h, ip3) + `v` (pseudo-voltage = Ca²⁺)

---

## Purpose

The AstrocyteModel (Li-Rinzel 1994) returns Ca²⁺ concentration (float),
which is incompatible with the SC-NeuroCore spiking pipeline (expects
`step() → int`). The AstrocyteNeuron adapter wraps the AstrocyteModel
and converts its output:

- **Ca²⁺ > ca_threshold** → return 1 (spike = gliotransmitter release)
- **Ca²⁺ ≤ ca_threshold** → return 0 (no release)
- **v attribute** → reports Ca²⁺ as pseudo-voltage (for monitor/plot)

This enables astrocytes to be used in Population, Projection, Network,
and SpikeMonitor — the full SC-NeuroCore pipeline.

---

## Equations

All dynamics are delegated to the wrapped AstrocyteModel. The adapter
adds only:

### Threshold conversion

$$\text{spike} = \begin{cases} 1 & \text{if } [Ca^{2+}] > \theta_{Ca} \\ 0 & \text{otherwise} \end{cases}$$

### Pseudo-voltage

$$v = [Ca^{2+}]$$

The `v` attribute is updated after each step to report the current Ca²⁺
concentration. This allows StateMonitor and voltage-based analysis tools
to work with astrocyte data.

### Implementation

```python
def step(self, current: float) -> int:
    ca = self._astro.step(current)
    self.v = ca
    return 1 if ca > self.ca_threshold else 0
```

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `ca_threshold` | 0.3 | µM | Ca²⁺ threshold for "spike" |
| `dt` | 0.01 | s | Timestep (passed to AstrocyteModel) |

All other parameters (g_ca, v_er, k_er, d1–d5, etc.) are inherited from
AstrocyteModel with its defaults. To customise: create an AstrocyteModel
with custom parameters and assign to `_astro`, or pass params through
Population constructor.

### ca_threshold = 0.3 µM

The default 0.3 µM corresponds to the Ca²⁺ level at which astrocytes
are observed to release gliotransmitter (Parpura & Bhatt 1998). Below
0.3 µM: resting. Above 0.3 µM: active release.

---

## Analytical Properties

### Adapter pattern

The AstrocyteNeuron follows the **Adapter design pattern** (GoF):
- **Target interface:** `step(current) → int`, `v` attribute, `reset()`
- **Adaptee:** AstrocyteModel (step → float)
- **Adapter:** AstrocyteNeuron (thresholds Ca²⁺ → int)

### Ca²⁺ oscillation → spike bursts

When the underlying AstrocyteModel oscillates:
- Rising phase: Ca²⁺ crosses 0.3 µM → spike = 1
- Peak: Ca²⁺ >> 0.3 µM → continuous spikes
- Falling phase: Ca²⁺ drops below 0.3 µM → spike = 0
- Trough: Ca²⁺ ≈ 0.05 µM → silent

The resulting spike train is a **rectangular wave** — 1 during the
Ca²⁺ peak, 0 during the trough. The duty cycle depends on the
oscillation waveform and ca_threshold.

### Properties accessible

The adapter exposes:
- `ca` (property): current Ca²⁺ concentration
- `ip3` (property): current IP3 concentration
- `v` (attribute): = Ca²⁺ (pseudo-voltage)

These allow monitoring astrocyte-specific state while using the
standard pipeline.

### Spike semantics

| Neuron | spike = 1 means: |
|--------|------------------|
| LIF | Action potential (Na⁺ spike) |
| HH | Action potential (threshold crossing) |
| AstrocyteNeuron | **Gliotransmitter release** (Ca²⁺ above threshold) |

The "spike" from an AstrocyteNeuron represents a fundamentally different
biological event — it is not an electrical spike but a biochemical
secretion event lasting ~10–100 ms (vs ~1 ms for neuronal spikes).

---

## Behaviour

### Pipeline-compatible astrocyte

```python
from sc_neurocore.neurons.models.astrocyte_adapter import AstrocyteNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

pop = Population(AstrocyteNeuron, n=10, label="astro",
                 params={"ca_threshold": 0.3})
drive = PoissonInput(n=10, rate_hz=100.0, weight=0.1, dt=0.01)
mon = SpikeMonitor(pop)
net = Network(pop, drive, mon)
net.run(duration=10.0, dt=0.01, backend="python")
print(f"Gliotransmitter release events: {mon.count}")
```

### Tripartite synapse wiring

With the adapter, astrocytes can participate in Network wiring:
- **Neuron → Astrocyte:** Excitatory projection carries glutamate →
  drives IP3 production → Ca²⁺ oscillation → "spikes"
- **Astrocyte → Neuron:** Projection carries gliotransmitter effect →
  modulates neuronal excitability

### ca_threshold tunes sensitivity

- ca_threshold = 0.1 µM: very sensitive (fires during small Ca²⁺ transients)
- ca_threshold = 0.3 µM: moderate (default, matches experimental threshold)
- ca_threshold = 0.5 µM: selective (fires only during large Ca²⁺ peaks)

---

## Comparison with Related Models

| Property | AstrocyteNeuron | AstrocyteModel | LIF | SigmoidRate |
|----------|----------------|---------------|-----|------------|
| Output | int (spike) | float (Ca²⁺) | int (spike) | float (rate) |
| Pipeline | Compatible | Limited (float) | Compatible | Limited (float) |
| Biology | Gliotransmitter release | Ca²⁺ signalling | Action potential | Firing rate |
| Timescale | 10–100 s | 10–100 s | 1–100 ms | Variable |
| v meaning | Ca²⁺ concentration | — | Membrane potential | Rate |

The adapter makes the AstrocyteModel pipeline-compatible while preserving
access to the underlying Ca²⁺/IP3 dynamics.

---

## Numerical Considerations

- All numerical properties inherited from AstrocyteModel (dt=0.01s, no exp()).
- The threshold comparison (ca > ca_threshold) is exact — no numerical
  sensitivity.
- v is updated each step (single assignment, no computation).

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/astrocyte_adapter.py` — 67 lines.
- **Composition:** Contains an AstrocyteModel instance (`_astro`).
- **Properties:** `ca` and `ip3` delegate to `_astro`.
- **__post_init__:** Creates AstrocyteModel with configured dt.
- **Dataclass:** Uses `@dataclass`.
- **Rust wiring:** Compatible via the standard step(f64) → i32 dispatch.

---

## Infrastructure Pipeline

```
AstrocyteNeuron (adapter)
├── step(current) → int {0, 1}
│   └── delegates to AstrocyteModel.step()
│       └── Ca²⁺ > ca_threshold → 1 (gliotransmitter release)
├── v = Ca²⁺ (pseudo-voltage)
├── .ca, .ip3 properties for direct state access
├── Population, Network, SpikeMonitor: fully compatible
│   PoissonInput(weight=0.1, rate=100Hz)
├── Projection: bidirectional neuron↔astrocyte wiring
└── Analysis: spike_count = gliotransmitter release count
```

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~300K steps/s | Not measured |
| Network (10 astrocytes, 10s) | ~3K neuron-steps/s | — |

Same as AstrocyteModel — the adapter adds negligible overhead
(one comparison per step).

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary return, v=Ca, reset, ca/ip3 properties |
| Threshold | 4 | fires when Ca>0.3, silent when Ca<0.3, ca_threshold configurable, duty cycle |
| Pipeline | 4 | Population, Network+drive, SpikeMonitor counts, Projection wiring |
| Behaviour | 3 | Ca oscillation → spike bursts, glutamate drive, stimulus response |
| **Total** | **16** | |

See `tests/test_model_astrocyte_adapter.py`. No bugs found.

---

## Findings

1. **Threshold conversion works:** Ca²⁺ > 0.3 µM → spike=1. Below → 0.
   Binary conversion correctly applied each step.

2. **v = Ca²⁺:** The pseudo-voltage accurately tracks cytosolic Ca²⁺.
   StateMonitor can plot Ca²⁺ dynamics using standard voltage tools.

3. **Pipeline fully functional:** Population, Network, SpikeMonitor,
   PoissonInput, Projection all work. "Spikes" represent gliotransmitter
   release events.

4. **Ca²⁺ oscillation → rectangular spike train:** During Ca²⁺ peaks,
   continuous spike=1. During troughs, spike=0.

5. **ca/ip3 properties accessible:** Direct access to underlying
   AstrocyteModel state for analysis.

6. **reset() delegates correctly:** Resetting the adapter resets the
   underlying AstrocyteModel and updates v.

7. **Adapter pattern bridges the float/int gap:** The fundamental design
   contribution — enables non-spiking models to participate in the
   spiking pipeline.

8. **Only adapter model:** Unique in SC-NeuroCore — demonstrates the
   pattern for wrapping any continuous-output model for pipeline use.
