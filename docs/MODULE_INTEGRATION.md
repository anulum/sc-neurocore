# SC-NEUROCORE — Module Integration Reference

> **Status:** Historical integration inventory; use the live system map and
> focused module evidence for current status
> **Snapshot date:** 2026-04-16
> **Snapshot tests:** 1,173 across 19 modules
> **Languages:** Python, Go, Rust, SystemVerilog

---

## Overview

This document records the April 2026 migration from the research sandbox into
the SC-NEUROCORE tree. Paths, tests, and readiness have continued to evolve;
this inventory must not be used as a live certification or production-readiness
claim.

At that snapshot, modules were:

1. Copied (not moved) into `src/sc_neurocore/` and `tests/`
2. Given clean module names (no `phase` prefixes)
3. Verified with full regression from the new locations
4. Documented with class/function inventories for downstream documentation



---

## Module Registry

### Safety & Certification

#### safety_cert — fail-closed safety-evidence organisation
- **Source:** ten responsibility modules plus the compatibility facade under
  `src/sc_neurocore/safety_cert/`
- **Tests:** `tests/test_safety_cert/` — **456 focused tests**
- **Key Classes:**
  - `Requirement` and `TraceabilityMatrix` — explicit implementation and
    verification links
  - `FailureMode`, `FMEDA`, and `ReliabilityMetrics` — arithmetic over
    caller-supplied FIT/DC records
  - `FormalProperty` and `FormalProofCertificate` — full-field evidence
    hashing and reports
  - `CertificationGenerator` and `CertificationPackage` — fail-closed
    assembly and atomic materialisation
  - `EvidenceItem` and `EvidenceBag` — path-safe SHA-256 manifests and
    on-disk verification

This library organises evidence. It does not grant certification, approval, or
conformity.

#### fault_injection — Resilience Testing
- **Source:** `src/sc_neurocore/fault_injection/fault_injection.py`
- **Tests:** `tests/test_fault_injection/test_fault_injection.py` — **22 tests**
- **Key Classes:**
  - `FaultInjector` — Configurable bit-flip / stuck-at fault injector
  - `FaultModel` — Fault type definitions (SEU, MBU, stuck-at)
  - `RadiationProfile` — Space/terrestrial radiation models
  - `ResilienceBenchmark` — Automated resilience scoring

---

### ASIC & Hardware

#### asic_flow — Multi-PDK ASIC Generation
- **Source:** `src/sc_neurocore/asic_flow/asic_flow.py`
- **Tests:** `tests/test_asic_flow/test_asic_flow.py` — **67 tests**
- **Key Classes:**
  - `PDKConfig` — Process Design Kit configuration (Sky130, GF12, TSMC)
  - `FloorplanGenerator` — Automated floorplanning
  - `TimingAnalyzer` — Static timing analysis
  - `PowerEstimator` — Dynamic/leakage power estimation
  - `DesignRuleChecker` — DRC violation detection
  - `FormalPropertyLink` — SymbiYosys integration

#### uvm_gen — UVM Testbench Generator
- **Source:** `src/sc_neurocore/uvm_gen/uvm_gen.py`
- **Tests:** `tests/test_uvm_gen/test_uvm_gen.py` — **72 tests**
- **Key Classes:**
  - `UVMGenerator` — Full UVM testbench emission
  - `RTLModule` — RTL module abstraction
  - `StimulusConfig` — Constrained-random stimulus
  - `CoverageSpec` — Functional coverage specification
  - `ScoreboardConfig` — Self-checking scoreboard

#### chiplet — Multi-Die Chiplet Generator
- **Source:** `src/sc_neurocore/chiplet/chiplet_gen.py`
- **Tests:** `tests/test_chiplet/test_chiplet_gen.py` — **94 tests**
- **Key Classes:**
  - `ChipletGenerator` — Die-level generation with UCIe/BoW links
  - `InterconnectTopo` — Die-to-die topology (mesh, ring, star)
  - `ThermalModel` — Per-die thermal estimation
  - `YieldEstimator` — Multi-die yield modelling

---

### Physical Substrates

#### spintronic — Magnetic-Domain SC Mapper
- **Source:** `src/sc_neurocore/spintronic/spintronic_mapper.py`
- **Tests:** `tests/test_spintronic/test_spintronic_mapper.py` — **66 tests**
- **Key Classes:**
  - `SpintronicMapper` — Maps SC networks to MTJ arrays
  - `MagneticDomainSim` — Micromagnetic co-simulation bridge
  - `SpinTorqueModel` — STT/SOT write models
  - `TMRCalculator` — Tunnelling magnetoresistance estimation

#### memristor — Memristor Crossbar Mapper
- **Source:** `src/sc_neurocore/memristor/memristor_mapper.py`
- **Tests:** `tests/test_memristor/test_memristor_mapper.py` — **70 tests**
- **Key Classes:**
  - `CrossbarArray` — Memristive crossbar abstraction
  - `ConductanceModel` — Non-linear conductance model
  - `AgingSimulator` — Device degradation simulation
  - `CrossbarEstimator` — Power/area estimation

#### analog_bridge — Stochastic-to-Analog Bridge
- **Source:** `src/sc_neurocore/analog_bridge/analog_bridge.py`
- **Tests:** `tests/test_analog_bridge/test_analog_bridge.py` — **27 tests**
- **Key Classes:**
  - `AnalogBridge` — SC-to-analog conversion layer
  - `DACModel` — Digital-to-analog output model
  - `ADCModel` — Analog-to-digital input model

---

### Exascale & Runtime

#### hypervisor — Neuromorphic Multi-Tenant Hypervisor
- **Source:** `src/sc_neurocore/hypervisor/hypervisor.py`
- **Tests:** `tests/test_hypervisor/test_hypervisor.py` — **78 tests**
- **Key Classes:**
  - `Hypervisor` — Multi-tenant SC workload manager
  - `Tenant` — Isolated workload with resource quotas
  - `ResourceAllocator` — FPGA tile / memory partitioning
  - `PreemptionPolicy` — Priority-based preemption
  - `HealthMonitor` — Watchdog and heartbeat monitoring

#### digital_twin/twinsync — Digital Twin Synchronization
- **Source:** `src/sc_neurocore/digital_twin/twinsync.py`
- **Tests:** `tests/test_twinsync/test_twinsync.py` — **72 tests**
- **Key Classes:**
  - `TwinSession` — Time-warp optimistic simulation
  - `NullMessageOptimizer` — Conservative sync lookahead
  - `DeltaCheckpoint` — Memory-efficient state diffs
  - `ReplayVerifier` — Deterministic replay verification
  - `DriftAutoCorrector` — Real-time drift compensation
  - `CheckpointAuditChain` — SHA-256 tamper-evident chain
  - `TwinFederation` — Multi-twin GVT coordination

---

### Frontiers

#### evo_substrate — Self-Replicating Evolutionary Substrate
- **Source:** `src/sc_neurocore/evo_substrate/evo_substrate.py`
- **Tests:** `tests/test_evo_substrate/test_evo_substrate.py` — **91 tests**
- **Key Classes:**
  - `Genome`, `TopologyGene`, `NeuronGene`, `PlasticityGene` — Genetic encoding
  - `MutationEngine` — Point, structural, duplication, swap mutations
  - `CrossoverEngine` — Multi-point crossover
  - `FitnessEvaluator` — Multi-objective fitness
  - `ReplicationEngine` — Generation-level evolution loop
  - `OrganismEmitter` — NIR + Verilog organism emission
  - `SafetyBounds` — Mutation space constraints
  - `TileDeploymentTracker` — FPGA tile allocation
  - `HallOfFame` — Historical best tracking
  - `IslandModel` — Multi-deme with migration
  - `NoveltyArchive` — Behavioural novelty search
  - `ExtinctionDetector` — Stagnation-triggered mass extinction
  - `CoevolutionArena` — Predator-prey dynamics
  - `FormalSafetyGuard` — Pre-deployment validation
  - `TournamentSelector` — Configurable selection pressure
  - `ParetoFront` — NSGA-II style non-dominated front
  - `AgeRegulator` — Maximum lifespan enforcement
  - `BloatPenalizer` — Genome complexity regularization
  - `CPPNGenome` — Compositional Pattern Producing Networks
  - `HWFitnessCollector` — FPGA execution feedback
  - `EvoStatisticsTracker` — Per-generation analytics
  - `ComplexityTracker` — Open-ended complexity metric

#### meta_plasticity — Mutable Plasticity Rules
- **Source:** `src/sc_neurocore/meta_plasticity/meta_plasticity.py`
- **Tests:** `tests/test_meta_plasticity/test_meta_plasticity.py` — **72 tests**
- **Key Classes:**
  - `MetaPlasticityEngine` — Self-modifying learning rules
  - `PlasticityRule` — Parameterised STDP/triplet rules
  - `RuleEvolver` — Evolutionary plasticity rule search
  - `ConsolidationScheduler` — Memory consolidation timing

#### autonomous_learning — High-Performance Online Plasticity Engine
- **Source:** `crates/autonomous_learning/`, `src/sc_neurocore/_native/`, `src/sc_neurocore/accel/go/autonomous_learning/`
- **Tests:** `tests/test_learning/` — **3 tests (Parity & Unit)**
- **Key Classes:**
  - `RustPlasticityRule` — FFI Python class mapping STDP/BCM/R-STDP to Rust backend
  - `LearningBridgeAccel` — Julia C-FFI wrapper
  - `autonomous_learning.PlasticityRule` (Go) — Go Cgo wrapper

#### bioware — Biological-Hardware Interface
- **Source:** `src/sc_neurocore/bioware/bioware.py`
- **Tests:** `tests/test_bioware/test_bioware.py` — **121 tests**
- **Key Classes:**
  - `BiowareInterface` — Wet-lab / in-silico bridge
  - `OrganoidModel` — Cerebral organoid abstraction
  - `MEAAdapter` — Multi-electrode array interface
  - `SpikeProtocol` — Biological spike encoding

#### federated — Federated SC Learning
- **Source:** `src/sc_neurocore/federated/federated_sc.py`
- **Tests:** `tests/test_federated/test_federated_sc.py` — **93 tests**
- **Key Classes:**
  - `FederatedCoordinator` — Privacy-preserving aggregation
  - `SecureAggregator` — Secure multi-party computation
  - `DifferentialPrivacy` — DP-SGD noise injection
  - `ModelCompressor` — Communication-efficient compression

#### bci_studio — BCI Closed-Loop Control
- **Source:** `src/sc_neurocore/bci_studio/bci_primitives.py`, `bci_studio.py`
- **Tests:** `tests/test_bci/` — **32 tests**
- **Key Classes:**
  - `BCIClosedLoopEngine` — Real-time neural decoding
  - `StimulusGenerator` — Closed-loop stimulation
  - `SafetyMonitor` — Charge density limiter

---

### Unification

#### explainability — SC Explainability Tools
- **Source:** `src/sc_neurocore/explainability/explainability.py`
- **Tests:** `tests/test_explainability/test_explainability.py` — **71 tests**
- **Key Classes:**
  - `ExplainabilityEngine` — Bitstream-level explanation generation
  - `CausalAttributor` — Causal attribution for SC decisions
  - `FormalPropertyLink` — Formal verification anchoring

#### neuro_symbolic — Predictive Coding Primitives
- **Source:** `src/sc_neurocore/neuro_symbolic/predictive_coding.py`
- **Tests:** `tests/test_neuro_symbolic/test_predictive_coding.py` — **34 tests**
- **Key Classes:**
  - `PredictiveCodingLayer` — Hierarchical prediction error
  - `SymbolEncoder` — Hyperdimensional symbol binding
  - `VerifiableInference` — Formally verifiable reasoning

#### stochastic_doctor — Bitstream Diagnostics
- **Source:** `src/sc_neurocore/stochastic_doctor/diagnostics.py`
- **Tests:** `tests/test_stochastic_doctor/test_diagnostics.py` — **16 tests**
- **Key Classes:**
  - `BitstreamDoctor` — Health diagnostics engine
  - `CorrelationDetector` — Inter-stream correlation analysis
  - `BitstreamAuditReport` — Full audit trail

#### model_zoo — Auto-Verilog Model Zoo
- **Source:** `src/sc_neurocore/model_zoo/model_zoo.py`
- **Tests:** `tests/test_model_zoo/test_model_zoo.py` — **37 tests**
- **Key Classes:**
  - `PluginRegistry` — Neuron model plugin system
  - `VerilogGenerator` — One-command Verilog emission
  - `DocGenerator` — Auto-documentation from plugins
  - `LIFPlugin`, `IzhikevichPlugin`, `AdExPlugin`, `HodgkinHuxleyPlugin`

---

## Test Summary

| Module | Location | Tests |
|--------|----------|------:|
| safety_cert | `tests/test_safety_cert/` | 456 |
| asic_flow | `tests/test_asic_flow/` | 67 |
| fault_injection | `tests/test_fault_injection/` | 22 |
| uvm_gen | `tests/test_uvm_gen/` | 72 |
| hypervisor | `tests/test_hypervisor/` | 78 |
| twinsync | `tests/test_twinsync/` | 72 |
| spintronic | `tests/test_spintronic/` | 66 |
| chiplet | `tests/test_chiplet/` | 94 |
| memristor | `tests/test_memristor/` | 70 |
| analog_bridge | `tests/test_analog_bridge/` | 27 |
| bioware | `tests/test_bioware/` | 121 |
| meta_plasticity | `tests/test_meta_plasticity/` | 72 |
| evo_substrate | `tests/test_evo_substrate/` | 91 |
| federated | `tests/test_federated/` | 93 |
| bci_studio | `tests/test_bci/` | 32 |
| explainability | `tests/test_explainability/` | 71 |
| neuro_symbolic | `tests/test_neuro_symbolic/` | 34 |
| stochastic_doctor | `tests/test_stochastic_doctor/` | 16 |
| model_zoo | `tests/test_model_zoo/` | 37 |
| **TOTAL** | | **1,173** |

## Running Tests

```bash
# Single module
python3 -m pytest tests/test_evo_substrate/ -q

# All integrated modules
python3 -m pytest tests/test_safety_cert tests/test_asic_flow tests/test_fault_injection \
  tests/test_uvm_gen tests/test_hypervisor tests/test_twinsync tests/test_spintronic \
  tests/test_chiplet tests/test_memristor tests/test_analog_bridge tests/test_bioware \
  tests/test_meta_plasticity tests/test_evo_substrate tests/test_federated tests/test_bci \
  tests/test_explainability tests/test_neuro_symbolic tests/test_stochastic_doctor \
  tests/test_model_zoo -q
```

## Provenance

All modules were industrialized
through a multi-pass gap analysis and hardening process. This document serves as the single source
of truth for what was integrated, where it lives, and how it is tested.

---

## Rust Crate Architecture

The Research Rust crates (`tinysc_riscv`, `core_engine`, `autonomous_learning`, `neuro_symbolic`)
remain as standalone crates and are **NOT** merged into `engine/`. Rationale:

1. **`engine/`** is a single PyO3/maturin crate (`sc_neurocore_engine`) with `cdylib` + `rlib`
   output, custom build config, and `bridge/` Python source path. Adding unrelated crates
   would break the `maturin develop` / `pip install -e .` workflow.
2. The Research Rust crates are standalone research crates with no Python binding.
3. When any of them matures to production, it should be added as a **separate workspace member**
   under a new top-level `crates/` directory (not inside `engine/`).

### Future Workspace Migration (when ready)

```toml
# SC-NEUROCORE/Cargo.toml (new workspace root)
[workspace]
members = [
    "engine",
    "crates/tinysc_riscv",
    "crates/neuro_symbolic",
]
```

This preserves the engine's build integrity while enabling shared dependency resolution.
