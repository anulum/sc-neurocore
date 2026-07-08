<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Quantum Cognition API Reference

**Module:** `sc_neurocore.quantum_cognition` — Tier: experimental
**Source:** `src/sc_neurocore/quantum_cognition/`
**Status:** experimental research tooling.  Hardware validation requires explicit physical inputs and live IBM execution; simulator artefacts are not hardware evidence.

This page covers the **quantum cognition layer** — an experimental subpackage for Fisher-Posner inspired spin dynamics, ATP-gated LIF coupling, radical-pair singlet-yield calculations, and explicit simulator or IBM Quantum dispatch.

> **Warning:** This module implements speculative neuroscience.  The Fisher-Posner hypothesis (Fisher 2015) is not experimentally confirmed.  Results should not be interpreted as validated quantum biology.

---

## 1. Mathematical Formalism

### 1.1 Spin Hamiltonian

`SpinPoolMPS.evolve_exact()` builds the two-spin coupling Hamiltonian directly:

$$
H = 2\pi \sum_{i,j,a,b} A_{ij}^{ab} S_i^a S_j^b
$$

where each `SpinCouplingTensor.tensor_mhz` supplies the explicit 3×3 coupling tensor in MHz.  The dense exact path applies `exp(-iHt)` and then converts the result back to MPS only if the configured `bond_dim` can represent the state without truncation.

### 1.2 Hybrid LIF Coupling

The hybrid neuron couples ATP state to a classical membrane equation:

$$
\frac{dV_m}{dt} = \frac{-(V_m - V_{\mathrm{rest}}) + I_{\mathrm{in}} + I_{\mathrm{pump}}}{\tau_m}
$$

where the pump current depends on quantum efficiency $\eta$:

$$
I_{\mathrm{pump}} = (\eta - 0.5) \cdot 2 \cdot \text{ATP}_{\mathrm{level}}
$$

### 1.3 Measurement and ATP Efficiency

`apply_measurement()` performs projective Born-rule measurement at one site.  `get_local_atp_efficiency()` returns the adjacent two-site singlet probability:

$$
\eta(s) = \text{Tr}(\rho_{s,s+1} P_{\text{singlet}})
$$

The returned value is clipped only to the physical probability interval `[0, 1]`.  The telemetry `entanglement_map` is not a physics multiplier.

### 1.4 Spiking Condition

Spike emission requires both voltage threshold crossing AND sufficient metabolic energy:

$$
\text{spike} \iff V_m \geq V_{\text{th}} \;\wedge\; \text{ATP} \geq \text{ATP}_{\text{consumption}}
$$

When ATP is depleted, the neuron experiences *metabolic failure* — it cannot fire despite suprathreshold voltage.

---

## 2. Public Surface

The module re-exports the quantum cognition public surface from `sc_neurocore.quantum_cognition.__init__`:

| Symbol | Source file | Role |
|--------|-------------|------|
| `SpinPoolMPS` | `spin_pool.py` | Pure-state MPS spin storage with bounded exact Hamiltonian evolution |
| `SpinCouplingTensor` | `spin_pool.py` | Explicit two-site 3×3 coupling tensor in MHz |
| `HybridFisherPosnerLIF` | `fisher_posner.py` | LIF neuron with quantum-metabolic coupling |
| `RadicalPairModel` | `radical_pair.py` | Density-matrix radical-pair singlet-yield model |
| `RadicalPairParams` | `radical_pair.py` | Radical-pair rates, lifetime, exchange, and hyperfine tensors |
| `FisherPosnerQuantumBridge` | `bridge_adapter.py` | PennyLane, explicit Aer, IBM Quantum, or emulated bridge |
| `QuantumStudioHook` | `studio_hook.py` | Telemetry hook for SNN Visual Studio |
| `QuantumCognitionLayerMetadata` | `studio_hook.py` | Frozen dataclass for layer metadata |
| `ContentChunk` | `content_indexer.py` | Indexed text chunk with provenance metadata |
| `GOTMBrain` | `gotm_brain.py` | Self-learning brain composing quantum cognition + LLM |
| `HAS_PENNYLANE` | `bridge_adapter.py` | `bool` — was PennyLane importable? |

Install: `pip install sc-neurocore[quantum-cognition]`

This extra is research-grade and opt-in. It inherits the `[quantum]`
dependency set, including Qiskit and PennyLane, and is not part of the default
wheel install. Use the emulated backend for deterministic local development;
use PennyLane, Aer, or IBM Quantum backends only when the experiment explicitly
requires those optional runtimes and records the corresponding provenance.

---

## 3. `SpinPoolMPS` — Non-local spin storage

`SpinPoolMPS` stores a pure MPS state for spin-1/2 nuclei.  It starts in `|00...0>`.  Thermal mixed states are not represented by random phases; use the density-matrix radical-pair path for finite-temperature calculations.

### 3.1 Constructor

```python
SpinPoolMPS(
    n_sites: int = 8,
    bond_dim: int = 16,
    correlation_length: float = 2.0,
    update_rate: float = 0.1,
    seed: int | None = 42,
)
```

- `n_sites` — number of ³¹P nuclear spin sites
- `bond_dim` — nominal MPS bond dimension (reserved for future full tensor network)
- `correlation_length` — telemetry scale for derived map summaries
- `update_rate` — telemetry update scale
- `seed` — random generator seed for Born-rule measurement sampling

Raises `ValueError` for `n_sites < 1`, `bond_dim < 1`, `correlation_length ≤ 0`, or `update_rate ∉ (0, 1]`.

### 3.2 Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_statevector` | `(*, max_sites: int = 16) → np.ndarray` | Exact full-state extraction for bounded systems |
| `set_statevector` | `(statevector, *, atol=1e-12) → None` | Load a state only if MPS conversion needs no truncation |
| `evolve_exact` | `(couplings: list[SpinCouplingTensor], time_us: float, *, max_sites=12) → None` | Exact dense Hamiltonian evolution |
| `apply_measurement` | `(site_idx: int, intensity: float = 1.0) → None` | Born-rule projective measurement |
| `get_local_atp_efficiency` | `(site_idx: int) → float` | Adjacent two-site singlet probability $\eta(s) \in [0, 1]$ |
| `get_status` | `() → dict` | Summary for telemetry (n_sites, avg/max/min entanglement, count) |
| `get_state` | `() → dict` | Full state for checkpointing |
| `set_state` | `(state: dict) → None` | Restore from checkpoint |
| `reset` | `() → None` | Reset to uniform entanglement |
| `to_scpn_payload` | `() → dict` | Produce SCPNDatastream-compatible metadata dict |

### 3.3 Key Properties

- `tensors: list[np.ndarray]` — MPS tensors `A[left, physical, right]`
- `entanglement_map: np.ndarray` — telemetry, shape `(n_sites,)`, sums to 1.0
- `P_singlet_2site: np.ndarray` — two-site singlet projection operator, shape `(4, 4)`

### 3.4 Exact Evolution Example

```python
import numpy as np
from sc_neurocore.quantum_cognition import SpinCouplingTensor, SpinPoolMPS

pool = SpinPoolMPS(n_sites=2, bond_dim=4)
pool.evolve_exact(
    [SpinCouplingTensor(0, 1, np.eye(3))],
    time_us=0.25,
)
```

Non-adjacent dynamics must be expressed through explicit coupling tensors.  The event-driven cortical layer does not invent distance-kernel physics.

---

## 4. `HybridFisherPosnerLIF` — Quantum-metabolic neuron

### 4.1 Constructor

```python
HybridFisherPosnerLIF(
    neuron_id: int,
    spin_pool: SpinPoolMPS,
    dt: float = 1.0,
    v_rest: float = -70.0,
    v_threshold: float = -50.0,
    v_reset: float = -70.0,
    tau_m: float = 20.0,
    atp_initial: float = 1.0,
    atp_consumption: float = 0.05,
    atp_basal_regeneration: float = 0.001,
)
```

`neuron_id` must be an integer site index in the provided `spin_pool`;
booleans and non-integer aliases are rejected. Voltage parameters and input
currents must be finite, `dt` and `tau_m` must be strictly positive,
`atp_initial` must be in `[0, 1]`, `atp_consumption` must be in `(0, 1]`,
and `atp_basal_regeneration` must be non-negative. Invalid `step(I_in)` calls
fail before membrane voltage, ATP, counters, or spin-pool measurements mutate.

### 4.2 `step` — integrate one timestep

```python
def step(self, I_in: float) -> tuple[float, bool]:
```

Returns `(Vm, is_spiking)`.  The step pipeline:

1. **ATP regeneration** — quantum efficiency $\eta$ determines regeneration rate
2. **Pump current** — $I_{\text{pump}} = (\eta - 0.5) \cdot 2 \cdot \text{ATP}$
3. **LIF integration** — forward Euler with external + pump current
4. **Spike decision** — threshold + metabolic gate

On successful spike, calls `spin_pool.apply_measurement(neuron_id, 1.0)` creating bidirectional quantum-classical feedback.

### 4.3 NeuronProtocol Compatibility

| Method | sc-neurocore protocol |
|--------|----------------------|
| `step(current)` → spike indicator | ✅ via `tuple[float, bool]` |
| `get_state()` → dict | ✅ returns Vm, ATP, counters |
| `reset_state()` / `reset()` | ✅ both aliases |

### 4.4 Telemetry Counters

- `_total_spikes` — successful action potentials
- `_metabolic_failures` — suprathreshold events blocked by ATP depletion
- `_total_steps` — integration steps since last reset

---

## 5. `RadicalPairModel` — Density-Matrix Singlet Yield

`RadicalPairModel` computes radical-pair singlet yield from an explicit Hamiltonian over two electron spins and a finite nuclear bath.

```python
model = RadicalPairModel.from_hyperfine_tensors(
    tensors_1=[A1, A2, A3],
    tensors_2=[B1, B2, B3],
    exchange_j=1.0,
    recombination_rate=0.1,
    lifetime_us=100.0,
)
yield_s = model.singlet_yield(b_local=0.0)
```

The yield is integrated as:

$$
\Phi_S = k \int_0^T e^{-kt} \operatorname{Tr}[P_S \rho(t)]\,dt.
$$

Dense exact evolution is limited to small nuclear baths and raises for unsupported sizes instead of switching to a scalar proxy.  Nonzero `entanglement_boost` arguments are rejected.

---

## 6. `FisherPosnerQuantumBridge` — QPU interface

### 6.1 Constructor

```python
FisherPosnerQuantumBridge(
    n_qubits: int,
    backend: str = "auto",
)
```

Backend selection:

| `backend` | Behaviour |
|-----------|-----------|
| `"auto"` | PennyLane if available, else emulated |
| `"pennylane"` | Requires PennyLane; raises `ImportError` if absent |
| `"ibm_aer"` | Explicit local Aer simulator path |
| `"ibm_qiskit"` | Requires `SC_NEUROCORE_IBM_TOKEN`, `QISKIT_IBM_TOKEN`, or `IBM_QUANTUM_TOKEN`; never falls back to a simulator |
| `"emulated"` | Pure NumPy; no external dependencies |

### 6.2 `execute_non_local_sync`

```python
def execute_non_local_sync(
    self, entangle_pairs: list[tuple[int, int]]
) -> np.ndarray:
```

Creates Bell pairs (H + CNOT) for specified qubit pairs and returns PauliZ expectations.  Entangled qubits yield `⟨Z⟩ = 0`.

### 6.3 `optimize_phases`

```python
def optimize_phases(
    self,
    target_coherence: float,
    learning_rate: float = 0.05,
    n_steps: int = 1,
) -> np.ndarray | None:
```

Gradient-based optimisation of qubit phases via PennyLane autograd.  Returns `None` in emulated mode.

**Fixed regression:** The prototype used `float()` around the cost function, which broke PennyLane's `ArrayBox` autograd tracing.  The current implementation removes the cast, allowing gradients to flow correctly through `qml.RZ` rotations.

### 6.4 `apply_orchestrator_bias`

```python
def apply_orchestrator_bias(
    self,
    global_phases: np.ndarray,
    target_coherence: float,
    learning_rate: float = 0.2,
) -> np.ndarray | None:
```

Accepts phase vectors from `scpn_phase_orchestrator.adapters.quantum_control_bridge.QuantumControlBridge.export_artifact()` and uses them as initial conditions for local gradient descent.

### 6.5 Cross-Repo Integration Points

| SC-NEUROCORE method | External consumer |
|---------------------|-------------------|
| `to_qpu_artifact_metadata()` | `scpn_neurocore.bridge.QPUBridgeArtifact` schema v1 |
| `apply_orchestrator_bias()` | PO `QuantumControlBridge.export_artifact()` format |
| `execute_non_local_sync()` | QC `bridge/snn_adapter.py` entangle-pairs pattern |

---

## 7. `QuantumStudioHook` — Telemetry

### 7.1 Constructor

```python
QuantumStudioHook(
    spin_pool: SpinPoolMPS,
    bridge: FisherPosnerQuantumBridge,
)
```

### 7.2 Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_layer_metadata()` | `QuantumCognitionLayerMetadata` | Frozen dataclass for UI layer panel |
| `get_layer_metadata_dict()` | `dict` | JSON-serialisable version |
| `get_realtime_data()` | `dict` | Streaming entanglement map + ATP efficiencies |
| `get_entanglement_snapshot()` | `dict` | Timestamped archive/stream snapshot with status summary |
| `to_json_event()` | `str` | Compact single-line JSON event for NDJSON or SSE bridges |

### 7.3 `QuantumCognitionLayerMetadata` Dataclass

```python
@dataclass(frozen=True)
class QuantumCognitionLayerMetadata:
    layer_name: str = "Quantum Cognition (Fisher-Posner)"
    status: str = "stable"
    avg_entanglement: float = 0.0
    n_sites: int = 0
    color: str = "#00f2ff"
    node_style: str = "glow"
```

---

## 7. Polyglot Acceleration Kernels

The Python spin-pool path is the publication reference for `get_local_atp_efficiency()` because it computes the two-site singlet RDM.  Rust, Mojo, and Julia are benchmark telemetry kernels; they do not infer ATP efficiency from the telemetry map.

| Language | File | LOC | API Status |
|----------|------|-----|------------|
| Python | `spin_pool.py` | 216 | Reference implementation |
| Mojo 0.26 | `spin_pool.mojo` | 145 | Telemetry benchmark kernel |
| Rust | `spin_pool.rs` | 180 | Telemetry benchmark kernel; ATP efficiency fails closed |
| Julia | `spin_pool.jl` | 124 | Telemetry benchmark kernel; ATP efficiency fails closed |

### 7.1 Mojo Kernel — `QuantumSpinChainMPS`

```mojo
struct QuantumSpinChainMPS:
    var sites: Int
    var bond_dim: Int
    var entanglement_map: List[Float64]

    fn __init__(out self, sites: Int, bond_dim: Int)
    fn apply_measurement(mut self, site_idx: Int, intensity: Float64)
    fn get_local_atp_telemetry(self, site_idx: Int) -> Float64
    fn apply_phase_shift(mut self, phi: Float64)
    fn get_avg_entanglement(self) -> Float64
    fn reset(mut self)
```

**Fixed from prototype:** Replaced deprecated `from tensor import Tensor` with `List[Float64]`, `inout self` → `out self` / `mut self`, `import math` → `import std.math`.

### 7.2 Rust Kernel — `QuantumSpinChainMPS`

```rust
pub struct QuantumSpinChainMPS {
    pub sites: usize,
    pub bond_dim: usize,
    pub correlation_length: f64,
    pub update_rate: f64,
    pub entanglement_map: Vec<f64>,
    pub measurement_count: u64,
}

impl QuantumSpinChainMPS {
    pub fn new(sites: usize, bond_dim: usize) -> Self;
    pub fn apply_measurement(&mut self, site_idx: usize, intensity: f64);
    pub fn get_local_atp_efficiency(&self, site_idx: usize) -> Result<f64, &'static str>;
    pub fn get_local_atp_telemetry(&self, site_idx: usize) -> f64;
    pub fn apply_phase_shift(&mut self, phi: f64);
    pub fn get_avg_entanglement(&self) -> f64;
    pub fn reset(&mut self);
}
```

### 7.3 Julia Kernel — `QuantumSpinPoolAccel`

```julia
module QuantumSpinPoolAccel
    apply_measurement!(entanglement_map, site_idx, intensity, ξ, α)
    get_local_atp_efficiency(entanglement_map, site_idx) -> error
    get_local_atp_telemetry(entanglement_map, site_idx) -> Float64
    apply_phase_shift!(entanglement_map, phi)
    get_avg_entanglement(entanglement_map) -> Float64
    benchmark_spin_chain(sites, n_steps) -> Float64
end
```

### 7.4 API Parity Matrix

| Operation | Python | Mojo | Rust | Julia |
|-----------|:------:|:----:|:----:|:-----:|
| Init (uniform map) | ✅ | ✅ | ✅ | ✅ |
| apply_measurement | ✅ | ✅ | ✅ | ✅ |
| get_local_atp_efficiency | ✅ | — | fail-closed | fail-closed |
| get_local_atp_telemetry | — | ✅ | ✅ | ✅ |
| apply_phase_shift | — | ✅ | ✅ | ✅ |
| Normalisation | ✅ | ✅ | ✅ | ✅ |
| Reset | ✅ | ✅ | ✅ | — |
| State serialisation | ✅ | — | — | — |
| SCPN payload | ✅ | — | — | — |

---

## 8. Pipeline Wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.quantum_cognition import *` | `__init__.py` re-exports 4 classes | `tests/test_quantum_cognition.py::TestPackageImport` |
| `pip install sc-neurocore[quantum-cognition]` | Pulls `[quantum]` (Qiskit + PennyLane) | `pyproject.toml` line 94 |
| `FisherPosnerQuantumBridge(backend="auto")` | Falls back to emulated without PennyLane | `test_init_emulated` |
| `HybridFisherPosnerLIF.step()` → `spin_pool.apply_measurement()` | Bidirectional quantum-classical feedback | `test_spike_feedback_to_spin_pool` |
| `FisherPosnerQuantumBridge.to_qpu_artifact_metadata()` | Dict compatible with QPUBridgeArtifact | `test_qpu_artifact_metadata` |
| Non-locality: spike at A → efficiency change at B | Exponential decay kernel | `TestNonLocality` (2 tests) |

### 8.1 Cross-Repo Integration Map

```
SC-NEUROCORE                    SCPN-QUANTUM-CONTROL         SCPN-PHASE-ORCHESTRATOR
─────────────                   ────────────────────         ───────────────────────
quantum_cognition/              bridge/snn_adapter.py        adapters/neurocore_bridge.py
├─ SpinPoolMPS                  ├─ ArcaneNeuronBridge        ├─ NeurocoreBridge
├─ HybridFisherPosnerLIF          │  imports ArcaneNeuron      │  imports StochasticLIF
├─ FisherPosnerQuantumBridge    bridge/orchestrator_adapter   adapters/quantum_control_bridge
│  ├─ to_qpu_artifact_metadata  ├─ PhaseOrchestratorAdapter  ├─ QuantumControlBridge
│  └─ apply_orchestrator_bias   └─ UPDEPhaseArtifact         └─ import_artifact/export_artifact
├─ QuantumStudioHook
├─ content_indexer.py           (reads all 26 GOTM repos)
│  ├─ ContentChunk
│  ├─ index_gotm_repo()
│  └─ embed_chunks()
└─ gotm_brain.py                (composes all above)
   ├─ GOTMBrain
   └─ LearningStep              ←── agentic-shared/llm.py (local LLM)
```

No cross-repo code changes were required — the existing bridges already support the data contracts used by quantum_cognition.

---

## 8B. CLI SNN Stimuli

The module CLI writes optional SNN stimulus records when `learn` or `daemon`
processes content:

```bash
python -m sc_neurocore.quantum_cognition learn /path/to/repo --snn-dir ./snn_stimuli
```

The default repository root and stimulus directory point at the Samsung ext4
GOTM working tree:

- `/media/anulum/GOTM/aaa_God_of_the_Math_Collection`
- `/media/anulum/GOTM/aaa_God_of_the_Math_Collection/04_ARCANE_SAPIENCE/snn_stimuli`

Every emitted `qc_*.json` stimulus uses the fleet memory-write schema:

| Field | Meaning |
|-------|---------|
| `content` | Human-readable learning-step summary, prefixed with `QC step <n>`. |
| `project` | Fixed uppercase project slug: `SC-NEUROCORE`. |
| `actor` | Controlled producer role: `system`. |
| `timestamp` | Unix timestamp in whole seconds from the writing process. |
| `entities` | Stable linking entities: `SC-NEUROCORE`, `quantum_cognition`. |
| `kind` | Event classification, currently `event`. |
| `source_ref` | Producing code surface: `sc_neurocore.quantum_cognition.__main__:_emit_snn_stimulus`. |

The stimulus writer no longer emits the legacy `text` or `source` aliases.

---

## 8A. `ContentChunk` & Content Indexer

### 8A.1 `ContentChunk` Dataclass

```python
@dataclass
class ContentChunk:
    repo_name: str        # e.g. "SC-NEUROCORE"
    file_path: str        # relative path within repo
    chunk_index: int      # sequential index within file
    text: str             # raw text content (~2000 chars)
    content_type: str     # "docstring" | "comment" | "markdown" | "code" | "metadata"
    weight: float         # priority weight (markdown=1.2, python=1.0, etc.)
    sha256: str           # auto-computed, 16-char hex provenance hash
```

### 8A.2 `index_gotm_repo`

```python
def index_gotm_repo(
    repo_path: str | Path,
    repo_name: str | None = None,
) -> list[ContentChunk]:
```

Walks a repository tree, extracts content from 13 supported file types (`.py`, `.md`, `.rs`, `.jl`, `.mojo`, `.go`, `.toml`, `.yaml`, `.yml`, `.json`, `.sv`, `.v`, `.lean`), skips `__pycache__`, `.git`, `build`, etc.  Returns chunks sorted by weight (descending).

### 8A.3 `embed_chunks`

```python
def embed_chunks(
    chunks: list[ContentChunk],
    n_dims: int = 32,
    seed: int = 42,
) -> np.ndarray:  # shape (len(chunks), n_dims)
```

Deterministic, offline embedding — no neural model required.  Feature dimensions:

| Dims | Feature | Source |
|------|---------|--------|
| 0–25 | Character frequency (a–z) | Statistical |
| 26 | Text length (log scale) | Statistical |
| 27 | Vocabulary richness | unique/total words |
| 28 | Content weight | From file type |
| 29 | Content type encoding | docstring=0.9 … metadata=0.3 |
| 30–31 | Hash-derived features | SHA-256 deterministic |

Adapts gracefully when `n_dims < 32` — only populates available dimensions.

---

## 8B. `GOTMBrain` — Self-Learning System

### 8B.1 Constructor

```python
GOTMBrain(
    n_neurons: int = 32,
    bridge_backend: str = "emulated",
    seed: int | None = 42,
)
```

Creates `n_neurons` `HybridFisherPosnerLIF` neurons sharing a single `SpinPoolMPS`, plus a `FisherPosnerQuantumBridge` for phase optimisation.  The default uses explicit emulation for deterministic repository-learning runs; pass `"pennylane"`, `"ibm_aer"`, or `"ibm_qiskit"` when that backend is intentionally part of the experiment.

### 8B.2 Learning Loop

```python
# Single step
step = brain.learn_step(chunk, vector) -> LearningStep

# Full repository
steps = brain.learn_from_repo("/path/to/repo", max_chunks=50) -> list[LearningStep]
```

The loop:
1. `get_llm_guidance(chunk.summary)` → FOCUS / EXPLORE / STABILIZE; without an explicit local LLM endpoint it deterministically returns STABILIZE
2. `process_content(vector, directive)` → spike indices
3. Spikes feed back into SpinPoolMPS via `apply_measurement()`

### 8B.3 Directive Mapping

| Directive | Target coherence | Learning rate | Behaviour |
|-----------|:----------------:|:------------:|-----------|
| FOCUS | 0.8 | 0.2 | Deep engagement with content |
| EXPLORE | 0.4 | 0.05 | Broad, shallow scanning |
| STABILIZE | 0.6 | 0.1 | Consolidation (default fallback) |

### 8B.4 State Inspection

| Method | Returns | Description |
|--------|---------|-------------|
| `get_learning_state()` | dict | Full state: n_neurons, total_spikes, ATP, entanglement, pool_state |
| `get_history()` | list[dict] | All `LearningStep` records as JSON dicts |
| `reset()` | None | Clear neurons, pool, and history |

---

## 9. Verification Notes

The focused Python tests cover initialisation, validation, projective measurement, exact coupling evolution, ATP gating, backend selection, and radical-pair density-matrix behaviour.  IBM hardware evidence is intentionally separate from simulator tests and requires explicit hyperfine tensors plus a live token.

---

## 10. Known Issues & Follow-ups

### 10.1 Cross-language benchmark (P2)

The Mojo, Rust, and Julia kernels exist but have not been benchmarked against the Python reference for wall-clock comparison.  The Mojo `benchmark_spin_chain` function and Rust `pub fn benchmark_spin_chain` are ready; a benchmark driver script is needed to time all 4 backends on identical workloads.

### 10.2 Larger Exact Spin Evolution (P3)

`SpinPoolMPS` has an exact dense Hamiltonian path for bounded systems.  Larger Posner-scale sweeps need a validated sparse or tensor-network evolution path before they can replace the dense reference.

### 10.3 GOTM Brain integration (P3 — experimental separate project)

`GOTMBrain` is an experimental research surface for local content indexing and
local guidance-loop learning traces.  It is not hardware validation evidence,
not a biological claim, and not a production decision system.  Future
self-learning brain promotion remains gated on explicit hardware-validation
evidence, reproducible run packs, and separate public release criteria.

### 10.4 Population integration (P2)

`HybridFisherPosnerLIF` implements `get_state()`, `reset_state()`, and `step()` matching `NeuronProtocol`, but is not yet registered in the `Population(model=...)` lazy registry.  This would enable `Population(model="HybridFisherPosnerLIF", n=100)` with automatic network integration.

---

## 11. Example Usage

### 11.1 Basic Non-Local Coupling

```python
from sc_neurocore.quantum_cognition import SpinPoolMPS, HybridFisherPosnerLIF

# Create shared spin pool (8 nuclear spin sites)
pool = SpinPoolMPS(n_sites=8)

# Create two neurons sharing the same quantum substrate
neuron_A = HybridFisherPosnerLIF(0, pool)
neuron_B = HybridFisherPosnerLIF(7, pool)

# Drive neuron A — observe non-local effects on neuron B
eff_B_before = pool.get_local_atp_efficiency(7)
for _ in range(200):
    neuron_A.step(50.0)  # Strong input → spikes → measurements
eff_B_after = pool.get_local_atp_efficiency(7)

print(f"Neuron A spikes: {neuron_A._total_spikes}")
print(f"Neuron B ATP efficiency: {eff_B_before:.4f} → {eff_B_after:.4f}")
# Non-local effect: efficiency at site 7 changed due to spikes at site 0
```

### 11.2 Quantum Bridge with PennyLane

```python
from sc_neurocore.quantum_cognition import FisherPosnerQuantumBridge

bridge = FisherPosnerQuantumBridge(4, backend="auto")
print(f"Backend: {bridge.backend}")

# Create Bell pairs
expectations = bridge.execute_non_local_sync([(0, 1), (2, 3)])
print(f"PauliZ expectations: {expectations}")

# Optimise phases (PennyLane only)
params = bridge.optimize_phases(target_coherence=0.8, n_steps=10)
if params is not None:
    print(f"Optimised phases: {params}")
```

### 11.3 Studio Telemetry

```python
from sc_neurocore.quantum_cognition import (
    SpinPoolMPS, FisherPosnerQuantumBridge, QuantumStudioHook,
)

pool = SpinPoolMPS(n_sites=4)
bridge = FisherPosnerQuantumBridge(4, backend="emulated")
hook = QuantumStudioHook(pool, bridge)

# Layer metadata for UI
meta = hook.get_layer_metadata_dict()
print(meta["layer_name"])     # "Quantum Cognition (Fisher-Posner)"
print(meta["visual_config"])  # {"color": "#00f2ff", "node_style": "glow"}

# Streaming data for live graphs
data = hook.get_realtime_data()
print(data["entanglement_map"])   # [0.25, 0.25, 0.25, 0.25]
print(data["atp_efficiencies"])   # [0.625, 0.625, 0.625, 0.625]

# Single-line telemetry event for NDJSON/SSE transport
event = hook.to_json_event("quantum_snapshot")
assert "\n" not in event
```

---

## 12. Tests

```bash
# Focused quantum cognition suite
PYTHONPATH=src python -m pytest tests/test_quantum_cognition.py -q
# 43 passed  (verified 2026-07-02)
```

| Test file | Test class | Tests | What's covered |
|-----------|------------|:-----:|----------------|
| `test_quantum_cognition.py` | `TestSpinPoolMPS` | 13 | Init, validation, measurement, ATP efficiency, state roundtrip, reset, SCPN payload |
| | `TestHybridFisherPosnerLIF` | 9 | Spiking, metabolic failure, ATP regeneration, spike feedback, state, repr |
| | `TestNonLocality` | 2 | Distal efficiency change, proximity gradient |
| | `TestFisherPosnerQuantumBridge` | 9 | Emulated + PennyLane backends, sync, gradient, validation |
| | `TestQuantumStudioHook` | 6 | Metadata, realtime data, snapshot payload, compact JSON event, repr |
| | `TestPackageImport` | 2 | All symbols importable, tier label |
| `test_gotm_brain.py` | `TestContentChunk` | 2 | Create, to_dict |
| | `TestTextExtraction` | 5 | Python docstrings, Rust doc comments, chunking, skip dirs |
| | `TestIndexFile` | 4 | Python, markdown, Rust indexing, unknown ext |
| | `TestIndexRepo` | 4 | Full repo walk, __pycache__ skip, error handling, weight sorting |
| | `TestEmbedChunks` | 4 | Shape, normalisation, determinism, content discrimination |
| | `TestGOTMBrain` | 13 | Init, validation, LLM guidance, process, learn_step, learn_from_repo, state, history, reset, entanglement evolution |
| | `TestLearningStep` | 1 | to_dict serialisation |
| | `TestPackageImport` | 1 | New symbols importable |
| | **Total** | **74** | |

---

## 13. Physical Evidence Boundary

The quantum cognition API remains experimental, but the molecular-input lane is
now backed by first-principles ORCA evidence rather than placeholder constants.
As of 2026-06-26, the neutral dry Posner cluster, cation-radical EPR/HFC pass,
neutral NMR pass, hydrated/dimer follow-up pass, and tier-2 optimized
hydration/dimer physics pass have all completed on the ML350 with normal ORCA
termination in the checked outputs.

Processed tier-2 values include an optimized hydrated-cluster energy of
`-10412.778244957428 Eh`, an optimized neutral-dimer energy of
`-19908.244951484612 Eh`, a counterpoise-corrected dimer binding estimate of
`-193.284716 kcal/mol`, a CPCM(Water) hydrated single point of
`-10412.897506075209 Eh`, and PBE0 cross-check single points for the hydrated
cluster and dimer. These values are processed as internal deterministic
evidence artifacts; they are not yet promoted into runtime model constants.

The active continuation is a tier-3 validation run: hydrated optimized
frequency/IR first, then optimized dimer frequency/IR. Those jobs determine
whether the promoted geometries are true minima and whether the vibrational
evidence is suitable for stronger Fisher-Posner control inputs. Until that
frequency validation, uncertainty propagation, and explicit model-constant
injection are complete, public API behavior and IBM hardware claims remain
gated.

---

## 14. References

Theoretical basis:

- Fisher, M. P. A. "Quantum cognition: The possibility of processing with nuclear spins in the brain." *Annals of Physics* 362, 593–602 (2015). doi:10.1016/j.aop.2015.08.020
- Swift, M. W. *et al.* "Posner molecules: from atomic structure to nuclear spins." *Phys. Chem. Chem. Phys.* 20, 12373–12380 (2018).
- Weingarten, C. P. *et al.* "A new spin on neural processing: Quantum cognition." *Front. Hum. Neurosci.* 10, 541 (2016).

Neuron model basis:

- Gerstner, W. & Kistler, W. M. *Spiking Neuron Models.* Cambridge University Press (2002).

Cross-repo references:

- SCPN-QUANTUM-CONTROL NB26-28: FIM alone synchronises at K=0, λ ≥ 8
- SCPN-QUANTUM-CONTROL NB19: autonomic → cortical directional bias = 1.36

Related public surfaces:

- Quantum hardware layer: [`api/quantum.md`](quantum.md) (if exists)
- Network integration: [`api/network.md`](network.md)
- IBM hardware claims remain gated on ORCA-derived molecular inputs and live
  backend calibration/run evidence; internal verification notes are intentionally
  not linked from public API pages.

---

## 15. Auto-rendered API

::: sc_neurocore.quantum_cognition
    options:
      show_root_heading: true
      show_source: true
      members:
        - SpinPoolMPS
        - HybridFisherPosnerLIF
        - FisherPosnerQuantumBridge
        - QuantumStudioHook
        - QuantumCognitionLayerMetadata
