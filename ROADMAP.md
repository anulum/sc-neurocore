# Roadmap

> Last updated: 2026-05-25. Priorities may shift based on
> validation results and community feedback.

## Current Maintenance Snapshot — 2026-05-25

- Mainline CI was verified green on 2026-05-25 at `edc35c11934f`;
  obsolete completed failed/cancelled repair-sequence Actions runs were purged
  only after replacement green evidence was present.
- GitHub deployment hygiene was refreshed on 2026-05-25: inactive stale Pages
  deployment records were removed, while current successful Pages and package
  release deployments were retained as evidence.
- Dependabot, code-scanning, and secret-scanning alert surfaces were rechecked
  on 2026-05-25 and reported zero open alerts.
- Coverage is being recovered in staged slices. The active branch raises the
  Python gate to 96%; 100% remains the target, not the current release claim.
- Open production blockers are tracked in internal audit and roadmap files.
  Do not use public docs as the source of truth for detailed task queues.
- Cross-repository validation must confirm the SCPN datastream contract with
  SCPN-QUANTUM-CONTROL and SCPN-PHASE-ORCHESTRATOR before claiming bridge
  readiness.
- 2026-05-21: Hardened `safety_cert/safety_cert.py` and
  `tests/test_safety_cert/test_safety_cert.py` with consolidated fail-closed
  validation updates across traceability, FMEDA aggregates, formal certificate
  state guards, certification-package checklist identity/contracts, evidence
  digest and uniqueness checks, cross-standard mapping/overlap integrity, WCET
  and change-impact state checks, and proof-coverage/property-gap identifier
  normalisation.
- 2026-05-21: Added 12 additional production hardening slices in
  `safety_cert/safety_cert.py` with matched tests: traceability requirement
  identity invariants, canonical/idempotent linking semantics, status downgrade
  correctness, deterministic traceability report ordering, FMEDA duplicate-seed
  prevention and guarded seed insertion, per-entry failure-rate and
  diagnostic-coverage validation, component identity checks, and deterministic
  FMEDA report ordering.

## v3.8 — Hardening & Edge AI Readiness ✓

### Coverage gate recovery

Historical releases reached higher local gates, but the current production
path uses a staged recovery plan: keep CI green, raise the gate only after
measured full-suite coverage supports it, and converge back to 100%.

### ~~NumPy 2.x full compatibility~~ ✓

Audit complete — zero deprecated calls found.

### ~~Enterprise CI/CD & supply chain hardening~~ ✓

13 CI workflows, all SHA-pinned. Bandit SAST, CodeQL, OpenSSF Scorecard.
Preflight gate with pre-push hook. PyPI OIDC trusted publisher. Python
minimum raised to 3.10.

### ~~Python API documentation~~ ✓

Live at GitHub Pages via mkdocstrings. Deploys on push to main.

### ~~Stale issue automation~~ ✓

`.github/workflows/stale.yml` — labels after 60 days, closes after 14 more.

### ~~Rust engine feature parity~~ ✓

Attention kernel: multi-head softmax with SIMD dispatch (475 lines).
Graph layer: CSR sparse backend (461 lines). MLIR emitter: CIRCT
hw/comb dialect output from IR graphs. 111 Rust neuron models with PyO3 bindings
(neurons, synapses, layers, networks, compiler IR).

### ~~Expanded SIMD kernels (issue #28)~~ ✓

AVX-512, AVX2, ARM NEON, ARM SVE, RISC-V RVV backends with
runtime dispatch. Operations: popcount, pack/unpack, fused
AND/XOR+popcount, dot/max/sum/scale f64, hamming distance,
softmax. SVE/RVV use portable fallbacks pending intrinsic
stabilisation in Rust.

## v3.9 — Quantum, SCPN, Benchmarks ✓

### ~~SCPN L1-L16 stack~~ ✓

16-layer SCPN stack complete. `create_full_stack()` returns all 16 layers.
`run_integrated_step()` chains L1→L16 with inter-layer coupling.

### ~~Formal verification~~ ✓

SymbiYosys proofs across 7 HDL modules: LIF neuron (6), bitstream synapse
(8), encoder (3), dense layer core (7), dotproduct (5), firing rate
bank (22), AXI-Lite config (13). 67 formal properties total.

### ~~Brunel balanced-network benchmark~~ ✓

20-variant translator suite. Brian2 comparison with honest framing.
NeuroBench-aligned metrics (up to 847 MOP/s).

### ~~Co-simulation parity~~ ✓

Python golden model → Icarus Verilog → bit-exact checker.

### ~~Quantum backend stabilisation~~ ✓

Qiskit Aer + PennyLane backends validated (Python 3.10+).
IBM Heron r2 noise model (depolarizing, amplitude/phase damping,
asymmetric readout). Parameter-shift gradient rule for variational
circuits. Hybrid quantum-classical VQE pipeline with scipy optimizer.
QEC shield noise integration with surface code thresholds.

### ~~Holonomic adapter ecosystem~~ ✓

Per-adapter benchmark suite (`benchmarks/adapter_benchmark.py`):
L1-L16 latency, memory, throughput with/without JAX JIT.
All 16 adapters registered in ComponentRegistry with factory
function. Plugin discovery via `importlib.metadata` entry points.

## v3.10 — JOSS Paper & FPGA Demo

### JOSS paper

`paper/paper.md` is maintained as a JOSS-format pre-submission draft. Formal
submission is postponed until the open production-hardening, validation,
coverage, documentation, and hardware-evidence TODOs are completed and
reverified. Numeric claims in the paper must be regenerated from current CI,
release, benchmark, and artefact evidence before submission.

### ~~MNIST-on-FPGA demo~~ ✓

`examples/mnist_fpga/demo.py` — train → Q8.8 → SC → Verilog export.
Float 94.2%, Q8.8 94.2%, SC 94.0%. 10 synthesisable HDL modules.

### ~~Vivado tooling~~ ✓

`tools/vivado_impl.tcl` + `tools/vivado_report.py`. Non-project flow
targeting Xilinx 7-series.

### ~~Documentation overhaul~~ ✓

README benchmark wording, HDL-module summaries, and Zenodo DOI references were
updated historically. Public docs must now treat the current coverage gate,
hardware measurements, and model-fidelity audit state as the evidence boundary.

## v3.12 — Competitive Sprint ✓

122 Python + 111 Rust neuron models were present in this release slice, with
PyO3 bindings for extended model categories, JAX training support, CuPy sparse
GPU paths, FMEA + traceability-matrix work, broad Python/Rust test coverage,
13 CI workflows, and conda-forge recipe work. Treat exact test counts as
release-time evidence that must be regenerated before public reuse.

New in this release:

- **ArcaneNeuron + 8 AI-optimized models**: self-referential cognition, attention-gated, predictive coding, phase-binding, meta-plastic, and more
- **Identity substrate**: persistent spiking network with checkpointing, trace encoding, L16 Director control
- **Network simulation engine**: Population-Projection-Network with
  3 backends (Python, Rust NetworkRunner, MPI)
- **Rust NetworkRunner**: 111-model fused loop (was 80), Rayon parallel, 100K+ neurons
- **MPI distributed simulation**: billion-neuron scale via mpi4py
- **Model zoo**: 10 configurations + 3 pre-trained weight sets (MNIST, SHD, DVS)
- **12 visualization plots**, **13 advanced plasticity rules**, **6 topology generators**
- **125 analysis functions** across 23 modules
- **conda-forge recipe draft** prepared for staged-recipes submission; not yet
  published on conda-forge

## v3.13 — NIR Interop, Equation Compiler, Import Speed ✓ (current)

- **NIR bridge**: 18/18 primitives (was 11), full roundtrip verified
- **Cross-framework interop**: verified with Norse, snnTorch, SpikingJelly, Sinabs, Rockpool
- **Equation-to-Verilog compiler**: any `EquationNeuron` ODE string → synthesizable Q8.8 RTL
- **Surrogate gradient MNIST**: 99.49% (ConvSpikingNet with learnable beta/threshold)
- **Import time optimization**: 200s → 10s via lazy-load neuron models + deferred scipy
- **SC arithmetic**: CORDIV division, adaptive bitstream length, Sobol/Halton decorrelation
- **Learning rules**: BCM, voltage STDP, TBPTT, EWC (real implementation), learnable beta/threshold
- **Biological circuits**: tripartite synapse, Rall dendrite, cortical microcircuit, astrocyte adapter
- **Hardware**: AXI-Stream interface, DMA controller, parameterized AXI-Lite, CDC primitives
- **Deep audit**: 15 bugs + 7 concerns fixed across 942 files

## v3.14 — SHD FPGA Deployment + GPU Backend ✓ (current)

### SHD end-to-end FPGA pipeline

Pre-silicon train → quantise → synthesise → bitstream flow for Spiking
Heidelberg Digits (SHD) speech classification targeting Zynq XC7Z020
(PYNQ-Z2). Physical board validation remains open:

- **Training:** DCLS max (Hammouamri 2024) on Vertex AI T4. The historical
  75.2% `dcls_max` result is now treated as exploratory because it selected
  checkpoints under native validation rather than the deployable rounded-delay
  validation condition.
- **Current deployable evidence:** corrected deployable-selector runs across
  seeds 0-4 reached 72.3990% mean test accuracy with 1.9672 percentage-point
  sample standard deviation and 0.0 percentage-point rounding drop in all five
  runs.
- **Verilog:** 5 new modules (sc_shd_top, sc_vmin_lif_neuron, sc_axonal_delay,
  sc_dense_int8_sparse, sc_shd_axi_wrapper) — 25 total HDL modules, 5 455 lines
- **Vivado synthesis:** 1 317 LUT (2.5%), 848 FF (0.8%), 0 BRAM, 0 DSP,
  WNS +4.048 ns at 100 MHz (~168 MHz achievable)
- **Bitstream generated** via Vivado v2025.2 (Zynq PS + AXI-Lite block design)
- **PYNQ deployment package:** driver, demo, .bit, .hwh generated for the
  pre-silicon deployment package
- **Q8.8 co-simulation:** bit-true Python reference matches Verilog, 4% gap vs PyTorch
- **Collaboration:** Joint work with T. Masquelier, A. Queant, B. Cottereau (CNRS/CerCo)

### ~~GPU compute backend (wgpu)~~ ✓

Feature-gated wgpu backend for DenseLayer stochastic computing:

- Philox 4x32-10 GPU-native RNG (no PCIe bandwidth bottleneck)
- Two-kernel architecture: encode (Bernoulli sampling) + accumulate (AND+popcount)
- Cross-platform via Vulkan (AMD RDNA2, NVIDIA, Metal, DX12)
- PyO3 GpuDenseLayer class with forward_fast() and forward_batch_numpy()

### ~~Project Zenith: Autonomous Learning Subsystem~~ ✓

- Exact bit-identical parity mapped bridging `PyTorch Surrogate Autograd` and `Rust Spintronic Emulation`.
- `sc_neurocore.plasticity.create_plasticity_layer` fully integrates supervised biological convergence loops mapped securely to deterministic `SymbiYosys` formal proofs.
- 4 Metaplasticity rules (STDP, BCM, R-STDP, ELIGENT).
- **Available now via v4.1**: Fully functional, verified natively on `rust-wgpu` parallel cross-platform frameworks supporting massive scale edge deployments natively.

### Model documentation upgrade — in progress

Per-model documentation pages (567+ lines each) with equations, parameters,
defaults, benchmarks, tests, and benchmark artefacts. Current machine-audited
coverage is tracked by timestamped manifests in `docs/internal/`; human
scientific review remains the promotion gate.

### ~~WaveformCodec~~ ✓

Neural data compression for BCI: 24x reduction on Neuralink-scale data,
Rust + Verilog paths, bit-true guarantees, mode parameter (background/snippet).

### ~~CI fixes and dependency updates~~ ✓

ruff 0.15.9, mkdocs strict mode, typos exceptions, Vivado gitignore,
dependabot PRs merged.

## v3.14 — ArcaneZenith Cognitive Core & Hardened Interfaces (ship now) ✓

### ~~ArcaneZenith Cognitive Core Primitive~~ ✓

- Shipped unified `sc_neurocore.arcane_zenith.create_arcane_neuron_with_zenith_plasticity` factory combining the novelty-driven `ArcaneNeuron` identity structure with Project Zenith plasticity rules driving the internal boundaries natively (tau_deep, novelty threshold, confidence, and lr_base bounds).
- Extracted and tracked `identity_drift` across continuous execution lifetimes for explicit verification bounds predictability.

### ~~Zenith Backend API Parity and WGPU Determinism~~ ✓

- Configured Python TorchRuleLayer natively parsing exact keyword parameter boundaries (tau_plus, tau_minus, tau_e) uniformly across framework targets.
- Enforced complete WGSL global seed matching via `set_wgpu_layer_seed` native ABI to fully guarantee cross-platform execution predictability.
- Finalized explicit state serialization boundaries integrating `get_state_dict()` proxies over Torch, Rust, and Wgpu backends to support familiar PyTorch-style checkpoint loading protocols.

### ~~HIL Debugger Telemetry Server~~ ✓

- Added HIL Debugger: real-time FPGA telemetry server with WebSocket broadcast, lock-free ring buffer, per-layer stats, triggers, and Python orchestration daemon (experimental).

### ~~Quantum Cognition (Fisher-Posner)~~ ✓

- Experimental subpackage implementing the Fisher-Posner quantum cognition
  hypothesis: `SpinPoolMPS`, `HybridFisherPosnerLIF`, `FisherPosnerQuantumBridge`,
  `QuantumStudioHook`, `ContentChunk`, `GOTMBrain` (8 public exports).
- 7 Python modules (1,586 LOC) + 3 polyglot acceleration kernels
  (Rust 312 LOC, Mojo 180 LOC, Julia 248 LOC).
- 74 Python tests + 10 Rust inline tests = 84 total, all passing.
- Cross-language benchmarks: Rust 200–340× Python, Mojo 240–500× Python.
- GOTM Brain self-learning module with local LLM guidance loop.
- API documentation updated with experimental scope and hardware-validation boundaries.
- Cross-repo bridges verified: SCPN-QUANTUM-CONTROL, SCPN-PHASE-ORCHESTRATOR.

## v4.0 — Physical FPGA Demos + Production (target: Q3 2026)

### Repository split and stable API freeze

- Freeze the stable public API for the promoted runtime, compiler, hardware,
  and bridge surfaces.
- Split the current broad source tree into several focused repositories after
  the ongoing experimental verification campaigns identify which modules are
  production paths, retained research paths, or retired exploratory paths.
- Keep the v3.x repository intentionally broad until that evidence is complete;
  the current kitchen-sink shape is transitional, not the intended long-term
  project layout.
- Publish migration notes mapping v3.x modules to the v4.0 repository layout
  before the split is completed.

### FPGA deployment proof (P0 blocker) PARTIALLY DONE

- ~~Generate Zynq 7020 deployment artefacts~~ ✓ (SHD bitstream generated,
  2.5% LUT in Vivado reports)
- ~~Measure: LUT count, BRAM, DSP, Fmax~~ ✓ (Vivado reports committed)
- Verify on physical PYNQ-Z2 board (on order)
- Measure dynamic power on silicon
- ~~Emit FPGA power and thermal digital-twin JSON beside deployable bitstream
  artefacts, seeded from synthesis reports and board profile metadata~~ ✓
  (`sc_neurocore.edge.power_thermal` now supports pre-silicon estimates and
  Vivado routed report-derived JSON).
- Deploy MNIST classifier as second demo
- Latency target: < 1 us neuron update (achieved: 2.83 us per 250-step sample)

### Per-model documentation (P1)

- Complete remaining 84 model doc pages (567+ lines each)
- Auto-generate benchmark tables from existing pipeline results
- Publish per-model Verilog mapping status

### JOSS submission & review

- Submit via https://joss.theoj.org/papers/new
- Respond to reviewer feedback (estimated 4-8 weeks)

### Wheel trimming

- Remove frontier/speculative tiers from `pip install sc-neurocore`
- ~~Add `pip install sc-neurocore[core]` install flag~~ ✓
- Fewer modules = stronger signal for core SC+SNN+FPGA story

### ~~Sparse weight matrices~~ ✓

CuPy CSR path added in `vectorized_layer.py` for N>1K networks.

### ~~JAX JIT compilation~~ ✓

`jax_forward_pass` + `jax_surrogate_gradient_step` added.

### ~~Tool Qualification Kit (TQK)~~ ✓

FMEA + traceability matrix in `docs/safety/`.

### ~~Network simulation engine~~ ✓

Population-Projection-Network with Python, Rust, MPI backends. Moved to v3.12.

### ~~Pre-trained model zoo~~ ✓

10 configurations + 3 pre-trained weight sets. Moved to v3.12.

### ~~conda-forge recipe draft~~ ✓

Recipe draft prepared for staged-recipes submission. The package is not yet
published on conda-forge, so public install claims stay gated until a feedstock
and Anaconda package page exist. Moved to v3.12.

### Mixed-precision plasticity (Zenith)

Allow per-rule or per-synapse bit-width selection inside the plasticity update (e.g., 8-bit traces for ELIGENT, 16-bit for BCM). Native support over verifying surfaces.

Progress (2026-05-21): Torch plasticity backend now supports fail-closed mixed-precision controls with scalar or per-synapse bit-width vectors (`mixed_precision_bits`, `weight_bits`, `trace_bits`, `eligibility_bits`, `theta_bits`, `act_avg_bits`) and explicit clip bounds. Coverage includes quantisation-grid assertions and malformed-spec rejection in `tests/test_learning/test_autonomous_learning.py`.

### Online meta-learning loop (ArcaneNeuron)

Built-in outer loop using Zenith’s mapping controlling internal thresholds adapting on dynamic environments continuously. Full demo notebook deployment scheduled.

Progress (2026-05-21): `ArcaneZenithCognitiveCore` now exposes a built-in outer-loop API (`run_meta_learning_episode`) with deterministic per-step adaptation traces, bounded-parameter contract checks, fail-closed empty-input validation, and compact symbolic trace export (`export_reasoning_trace`) for downstream introspection.

## v4.1 — Community & Ecosystem (target: Q4 2026)

### Community seeding

- ~~Awesome-neuromorphic listing~~ drafted, PR pending
- Conference lightning talk (NICE, ICONS, or Telluride)
- ~~Lab outreach~~ templates ready in `docs/internal/`
- GitHub Discussions with seeded categories
- Publish `sc_neurocore_engine` wheels through the trusted-publisher release
  path. This remains gated on current wheel publication and install-smoke
  evidence in the internal TODO.

### Silicon partnerships

- Intel Loihi 2: LAVA framework backend adapter
- SpiNNaker2: SpiNNTools compilation target
- One-click NIR mapping reports for Loihi 2 and SpiNNaker2, including
  supported-node lowering, unsupported-node diagnostics, resource estimates,
  and hardware-noise back-annotation hooks.
- Target: default middleware layer for neuromorphic silicon

### Self-hosted NeuroCore Hub

- Dockerised Studio bundle for private labs with local model zoo indexing,
  benchmark runner, and offline artefact cache.
- Air-gapped deployment mode with no telemetry egress and explicit local-only
  dependency mirrors.
- Reference compose profile for BCI/medical teams that need repeatable private
  validation runs.
  Progress (2026-05-21): Hub bundle config now enforces fail-closed offline contracts: `offline=True` requires at least one local dependency mirror directory, manifests expose the air-gapped mirror contract, and bundle generation materialises mirror directories for operator provisioning.

### Zenith BCI & Neuro-symbolic Primitives

- **BCI closed-loop primitive**: `ZenithBCILoop` module translating Neuralink/Neuropixels continuous streams dropping latency guarantees beneath 10 ms constraints via parallel GPU interfaces.
  Progress (2026-05-21): Added `sc_neurocore.interfaces.ZenithBCILoop` with deterministic Neuropixels/Neuralink-style stream ingestion, closed-loop waveform→spike→feedback processing, explicit per-stage latency ledger (`ingest/codec/decode/feedback`), and budget verdict (`latency_budget_met`) against a configurable sub-10 ms target. Reproducibility evidence is now codified in `notebooks/40_zenith_bci_loop_evidence.ipynb` with executable guardrails in `tests/test_notebooks/test_zenith_bci_loop_notebook.py`.
- **Online fault-injection + resilience mode**: Add radiation-hard bit-flip verification logic directly integrating across biological pathways for satellite deployment.
  Progress (2026-05-21): `ArcaneZenithCognitiveCore` now exposes `evaluate_bio_pathway_resilience(...)`, which deterministically converts biological pathway firing-rate maps into pathway bitstreams, runs seeded resilience-mode fault-injection, and emits pathway-labelled replay-ready reports (`layer_id`, channels, radiation profile, recommended degradation action).
- **Neuro-symbolic self-verification trace**: Leverage ArcaneNeuron identity compartments to export a short symbolic "reasoning log" capturing novelty/internal shift.
  Progress (2026-05-21): `ArcaneZenithCognitiveCore` now emits a schema-stamped symbolic reasoning log (`export_symbolic_reasoning_log`) with deterministic labels for novelty level/shift, confidence trend, identity drift regime, and adaptation regime, including numeric evidence payloads and per-step embedding in `run_meta_learning_episode` traces.

### Industrial applications

- Robotics: SNN-based reactive controllers with formal timing
- Smart grids: stochastic load prediction with bounded latency
- Fusion control: real-time plasma state estimation (SCPN-Fusion-Core bridge)
  Progress (2026-05-21): Added first-class industrial readiness domains in `sc_neurocore.industrial_applications` for `robotics`, `smart_grid`, and `fusion_control`, including explicit timing-evidence gates (`EvidenceCategory.TIMING` with latency/timing alias normalisation) and domain-specific mandatory evidence coverage tests.

### Formal SNN verification standard

Position SC-NeuroCore as reference implementation for formal SNN
verification:
- Liveness: every reachable marking can fire at least one transition
- Boundedness: token counts within proven upper bounds
- Deadlock freedom: compiler output is provably deadlock-free
  Progress (2026-05-21): `sc_neurocore.verification.publication_grade_snn_standard_profile()` now encodes these three roadmap claims as explicit mandatory conformance requirements (`liveness_reachable_transition_fireability`, `boundedness_token_bounds`, `deadlock_freedom_compiler_output`) with fail-closed assessment alongside implementation-equivalence and external-proof gates.

## Research Roadmap Intake — 2026-04-30

The following items are accepted as roadmap candidates. They require scoped
design docs, milestone split, and evidence gates before claims move into the
public feature set.

### Auto-adaptive precision optimizer

Per-synapse bit-length selection driven by sensitivity analysis and formal
error bounds. This extends the existing adaptive precision, stochastic
computing, and Zenith/ArcaneZenith plasticity pieces into a first-class
optimiser with proof-carrying precision assignments.

2026-04-30 implementation slice: `assign_synapse_precisions(...)` now produces
per-synapse bit-width and SC bitstream-length plans with conservative
quantisation, Hoeffding stochastic, and total error bounds. The companion
`precision_plan_manifest(...)` gives downstream compiler and verification
passes a deterministic evidence surface.

### Production HIL daemon and real-time digital twin

Closed-loop BCI path from Neuropixels/Neuralink-scale telemetry into live SC
network adaptation. Build from the HIL debugger, AER routing, bioware signal
handling, and digital-twin synchronization primitives, with sub-10 ms latency
as the design target.

### MLIR/CIRCT native backend

Add MLIR/CIRCT emission alongside the current Verilog path so compiler output
can target next-generation open EDA flows without replacing the proven RTL
backend.

2026-04-30 implementation slice: `generate_mlir_bundle(...)` now validates
HDL-facing identifiers and writes CIRCT-ready `.mlir` plus
`mlir_bundle_manifest.json` with operation counts and `firtool` availability.
The manifest keeps downstream Verilog, timing, area, and power claims gated
until explicit CIRCT/OpenROAD execution evidence is attached.

### One-command multi-PDK ASIC flow

Expose OpenROAD-backed multi-PDK ASIC compilation through the Python API with
area, timing, and power estimates attached to generated evidence manifests.
Claims remain gated on exact OpenROAD binary/container digest and PDK revision.

2026-04-30 implementation slice: `generate_asic_flow_bundle(...)` writes the
Yosys/OpenROAD deck bundle and `asic_flow_manifest.json` in one Python call,
including PDK-resolution blockers and pre-synthesis area/power/timing
estimates while explicitly marking physical PPA claims as not allowed until
real EDA evidence is attached.

### Stochastic photonic co-design loop

Promote the photonic bridge into a first-class SC-to-photonics workflow:
stochastic bitstream mapping, FDTD validation loop, and GDSII generation with
auto-placed optical pulse modulators.

### Federated edge neuromorphic hypervisor

Multi-tenant edge-cluster orchestration built on AER routing and deployment
contracts. Target isolation, scheduling, telemetry, and per-tenant evidence
manifests for distributed neuromorphic deployments.

## Focused Roadmap Priorities — 2026-05-01

### Studio as default path (drag-drop → train → deploy)

- Studio must become the primary UX with one default flow from network canvas
  to deployment: drag-drop construction, surrogate-training loop, adaptive
  precision compile, one-click deploy, and live co-simulation.
- Outcome: a first-time user can execute a full end-to-end SNN deployment in
  one Studio session without hand-written scripts.

### Model documentation and evidence sprint

- Finalise the per-model documentation sprint: generate the remaining 84 model
  pages with:
  - equations,
  - benchmarks,
  - Verilog mapping status.
- This is the current highest user-facing documentation debt and must be
  executed before additional feature headlines.

### Hardware validation before expansion

- Before any new feature lane, run physical validation on a physical
  PYNQ-Z2 board: power, latency, and dynamic measurements, then publish
  measured numbers and a reproducible run pack.
- A single measured demo must be the default credibility baseline for claims.

### Packaging polish and offline deployment

- Add shipping profile for `hdl install` + Docker + conda that supports
  pre-built wheels with static primitive bitstreams so Vivado is not required
  for baseline FPGA targets.
- Keep this as the default package path for standard users and CI validation.
  Progress (2026-05-21): `tools/install_profile_audit.py` now fail-closes packaging readiness on the hub offline mirror contract, requiring the self-hosted hub manifest to expose local dependency mirrors (`mirrors/wheelhouse`, `mirrors/huggingface`) and `requires_local_dependency_mirrors=true` in addition to static primitive and wheel/conda checks.

### Full tape-out API and formal-aware precision flow

- Extend `generate_asic_flow_bundle(target_pdk="asap7")` so it emits:
  - GDSII,
  - power and area reports,
  - formal evidence bundle.
- Keep claims gated until manifests include full CIRCT/OpenROAD run evidence.
  Progress (2026-05-21): ASIC flow manifests now encode formal-evidence attachment state (`formal_evidence_attached`, `formal_evidence_complete_for_claim`) with explicit required artefact types, keeping physical/tape-out claims fail-closed unless formal bundle evidence is attached alongside external EDA execution evidence.

### Adaptive precision optimiser as API surface

- Promote adaptive precision with formal error-bounds as a first-class API:
  `assign_synapse_precisions(...)` with an explicit UI action in Studio:
  “auto-tune for <0.1 % error at minimal LUTs”.
- Generate and bundle SymbiYosys proof evidence for the bounded-error claims.
  Progress (2026-05-21): Added `auto_tune_synapse_precisions(...)` as a first-class percent-target API (`target_error_percent`) that wraps the synapse planner and emits deterministic action metadata (`auto_tune_adaptive_precision`, target fraction, optimisation objective) for Studio/CI integration.
  Progress (2026-05-21): Added `write_precision_formal_evidence_bundle(...)` to materialise deterministic SymbiYosys-ready artefacts (`.sva`, `.sby`, formal manifest) with fail-closed claim flags (`symbiyosys_executed=false`, `formal_proof_passed=false`, `hardware_measurement_claimed=false`) until external proof execution evidence is attached.
