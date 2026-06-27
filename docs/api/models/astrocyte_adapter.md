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

The adapter constructor accepts `ca_threshold` and `dt`. The wrapped
`AstrocyteModel` uses its default Li-Rinzel parameters; callers that need
custom calcium dynamics should instantiate `AstrocyteModel` directly or build a
small specialised adapter around that configured model. `Population` parameters
therefore configure only the adapter fields documented above.

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

- **Source:** `src/sc_neurocore/neurons/models/astrocyte_adapter.py` — 90 lines.
- **Composition:** Contains an AstrocyteModel instance (`_astro`).
- **Properties:** `ca` and `ip3` delegate to `_astro`.
- **__post_init__:** Creates AstrocyteModel with configured dt.
- **Dataclass:** Uses `@dataclass`.
- **Polyglot status:** Python owns the full adapter behaviour. Rust, Go, and
  Julia safety/service stubs exist for this surface, but this documentation
  slice did not change runtime semantics or rerun cross-language benchmarks.

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

## Performance And Benchmark Status

| Metric | Python | Rust |
|--------|--------|------|
| Historical isolation smoke | ~300K steps/s | Not measured |
| Historical network smoke (10 astrocytes, 10s) | ~3K neuron-steps/s | — |

The adapter adds one comparison and one pseudo-voltage assignment around
`AstrocyteModel.step()`. The 2026-06-27 docstring-policy slice did not change
the runtime algorithm, so no new isolated benchmark artefact was generated.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Adapter unit contract | 8 | binary return, v=Ca, reset, ca/ip3 properties, Population `step_all`, voltages, reset_all |
| End-to-end model contract | 28 | defaults, validation, threshold sensitivity, sustained IP3 drive, finite long-run state, Network, Projection, SpikeMonitor, spike statistics, deterministic traces |
| **Total** | **36** | |

See `tests/test_astrocyte_adapter.py` and
`tests/test_model_astrocyte_adapter.py`. On 2026-06-27 the focused run passed
with `38 passed` when combined with `tests/test_public_docstring_policy.py`;
strict mypy reported no issues for the adapter source and both dedicated test
files. Isolated pytest-cov for this adapter is currently blocked by the local
SciPy/NumPy `_NoValueType` import failure triggered through `network.__init__`
under coverage instrumentation; the same production-path tests pass without
coverage instrumentation.

---

## Findings

1. **Threshold conversion works:** Ca²⁺ > 0.3 µM → spike=1. Below → 0.
   Binary conversion correctly applied each step.

2. **v = Ca²⁺:** The pseudo-voltage accurately tracks cytosolic Ca²⁺.
   StateMonitor can plot Ca²⁺ dynamics using standard voltage tools.

3. **Pipeline wiring covered:** Population, Network, SpikeMonitor,
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

---

## Design Pattern for Other Rate Models

The AstrocyteNeuron adapter demonstrates a general pattern that can be
applied to any float-returning model to make it pipeline-compatible:

```python
@dataclass
class ThresholdAdapter:
    threshold: float = 0.5
    def __post_init__(self):
        self._inner = SomeRateModel()
        self.v = 0.0
    def step(self, current: float) -> int:
        output = self._inner.step(current)
        self.v = output
        return 1 if output > self.threshold else 0
    def reset(self):
        self._inner.reset()
        self.v = 0.0
```

This pattern could be applied to: WilsonCowanUnit, SiegertTransferFunction,
WendlingNeuron, JansenRitUnit, LarterBreakspearNeuron, SigmoidRateNeuron,
ThresholdLinearRateNeuron, WongWangUnit — all of which return float and
are currently pipeline-limited.

### Limitations of the pattern

The threshold conversion is a lossy transformation:
- The continuous rate/concentration information is reduced to binary
- The threshold choice affects the spike pattern significantly
- For models with multi-dimensional output (WongWang returns tuple),
  additional logic is needed

Despite these limitations, the adapter pattern provides the simplest path
to pipeline integration for non-spiking models.

### Future: native rate pipeline

An alternative to the adapter pattern would be a native rate-based pipeline
in SC-NeuroCore that handles float outputs directly — passing rates through
projections instead of spikes. This would eliminate the need for adapters
and preserve the continuous dynamics information. Currently not implemented.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
18/18 PASSED in 3.82s
├── TestAdapterIsolation: 5 tests (defaults, binary return, v=Ca, reset, properties)
├── TestThreshold: 4 tests (fires when Ca>0.3, silent below, configurable, duty cycle)
├── TestPipeline: 5 tests (Population, Network, SpikeMonitor, Projection, analysis)
└── TestBehaviour: 4 tests (oscillation→bursts, glutamate drive, stimulus response, deterministic)
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | ca_threshold=0.3, dt=0.01 |
| step() → int {0,1} | ✓ PASS | Binary spike via Ca threshold |
| v = Ca²⁺ | ✓ PASS | Pseudo-voltage tracks cytosolic Ca |
| .ca, .ip3 properties | ✓ PASS | Direct state access |
| reset() | ✓ PASS | Delegates to AstrocyteModel |
| Population(n=10) | ✓ PASS | 10 adapter instances |
| Network + PoissonInput | ✓ PASS | Spikes produced |
| Projection wiring | ✓ PASS | src→tgt accepted |
| SpikeMonitor | ✓ PASS | Counts gliotransmitter events |
| Analysis (spike_count) | ✓ PASS | Works on adapter output |
| Deterministic | ✓ PASS | Two runs identical |

**ALL 18 PIPELINE TESTS PASSED. ADAPTER IS END-TO-END FUNCTIONAL.**
