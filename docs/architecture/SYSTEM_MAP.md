# SC-NeuroCore — System Architecture Map

> Canonical architecture map. Package version `3.16.0`; refreshed against the
> current tree on 2026-07-12. Supersedes the earlier `architecture.md` and
> `COMPONENT_INVENTORY.md` in this directory, which are stale (see *Provenance*).
>
> Purpose: a single, factual reference to SC-NeuroCore's capabilities, data flow,
> inputs/outputs, processing models, backends and their wiring — for sibling
> repositories and for the STUDIO integration layer. Every structural claim is
> anchored to the source tree; quantitative or model claims carry a file
> reference. No marketing language, no speculative naming.

---

## 1. What SC-NeuroCore is

A three-tier stochastic-computing (SC) spiking-neural-network framework that takes
a network defined in Python down to bit-true RTL on FPGA/ASIC silicon:

| Tier | Location | Substance |
|------|----------|-----------|
| **Software simulation** | `src/sc_neurocore/` (121 subpackages, ~57k LoC Python) | Neuron/synapse models, SC arithmetic, network engine, learning, analysis |
| **Compiled engine** | `engine/` (Rust + PyO3 → `sc_neurocore_engine`), `crates/` (7 research crates), `src/sc_neurocore/accel/` (Julia/Go/Mojo/JAX kernels) | SIMD SC inference, fixed-point integrators, spike-train analysis, IR emit |
| **Hardware** | `hdl/` (Verilog RTL, 43 top-level non-testbench modules), `hdl_gen/`, `compiler/`, `asic_flow/` | Synthesisable RTL, FPGA bitstreams, ASIC flow |

The defining design choice is **stochastic computing**: values are encoded as
Bernoulli/quasi-random bitstreams, so arithmetic becomes single-gate logic
(multiply = AND/XNOR, scaled-add = MUX) — directly mappable to extremely small,
low-power digital hardware. The SHD pipeline is proven end-to-end on a Zynq
XC7Z020 (`ARCHITECTURE.md` §SHD; 1 317 LUT / 2.5 %).

### Module tiers (from `pyproject.toml` + `__init__.py`)

- **Core (ships in the wheel)** — imported by default; the production surface:
  `neurons`, `synapses`, `layers`, `sources`, `recorders`, `core`, `utils`,
  `accel`, `network`, `learning`, `compiler`, `ir`, `hdl_gen`, `hardware`,
  `nir_bridge`, `math`, `solvers`, `scpn`, `adapters`, `spike_codec`, and more.
- **Research (source-available, excluded from the built wheel)** — explicitly
  excluded in `[tool.setuptools.packages.find] exclude`: `audio`, `generative`,
  `world_model`, `dashboard`, `viz`, `analysis`, `experiments`, `drivers`,
  `interfaces`, `swarm`, `sleep`, `optics`, `chaos`, `bio`, `robotics`,
  `physics`. Functional but not part of the supported binary distribution.
- **Contrib (not a package)** — speculative material lives under the repo-root
  `research/` tree (`eschaton/`, `speculative/`, `meta/`, `transcendent/`,
  `exotic/`, `post_silicon/`); it is **not** importable as `sc_neurocore.*` and
  is out of scope for this map.

---

## 2. Public API surface (the contract)

`import sc_neurocore` is lazy (PEP 562, `__init__.py:154`) and torch-free; symbols
load on first access. The exported, supported surface is:

| Group | Symbols | Defining subpackage |
|-------|---------|---------------------|
| Neurons | `BaseNeuron`, `StochasticLIFNeuron`, `FixedPointLIFNeuron`, `FixedPointLFSR`, `FixedPointBitstreamEncoder`, `HomeostaticLIFNeuron`, `StochasticDendriticNeuron`, `SCIzhikevichNeuron` | `neurons` |
| Synapses | `BitstreamSynapse`, `BitstreamDotProduct`, `StochasticSTDPSynapse`, `RewardModulatedSTDPSynapse` | `synapses` |
| Layers | `SCDenseLayer`, `SCConv2DLayer`, `SCLearningLayer`, `VectorizedSCLayer`, `SCRecurrentLayer`, `MemristiveDenseLayer`, `SCFusionLayer`, `StochasticAttention` | `layers` |
| Sources / Recorders | `BitstreamCurrentSource` / `BitstreamSpikeRecorder` | `sources` / `recorders` |
| Encoding utils | `RNG`, `BitstreamEncoder`, `BitstreamAverager`, `generate_bernoulli_bitstream`, `generate_sobol_bitstream`, `bitstream_to_probability`, `estimate_memory` | `utils` |
| Licensing | `get_license_status`, `set_license_key`, `validate_license_key`, … | `license` |
| Exceptions | `SCNeuroError`, `SCEncodingError`, `SCConfigError`, `SCWeightError`, `SCDependencyError`, `SCHardwareError`, `SCCompilerError` | `exceptions` |
| Lazy submodules | `plasticity`, `datasets` | — |

CLI entry point: `sc-neurocore = sc_neurocore.cli:main` (`pyproject.toml`).
`src/sc_neurocore/cli/parser.py` is the composition root; focused modules under
`src/sc_neurocore/cli/commands/` own compilation, deployment, formal, SC-NIR,
Studio, synthesis-evidence, mapping, serving, hub, status, and maintenance
commands. Top-level help exposes Model, Hardware, Studio, and Maintain modes,
while `COMMAND --help` progressively discloses command-specific options. Heavy
optional dependencies are imported inside handlers rather than by the parser.

---

## 3. The spine: end-to-end data flow

```
 float input
    │  BitstreamEncoder (LFSR x^16+x^14+x^13+x^11+1, or Sobol/Halton quasi-random)
    ▼
 bitstream  (uint8 0/1, or packed uint64 words)
    │  ── multiply = AND (unipolar) / XNOR (bipolar [-1,1]);  scaled-add = MUX
    ▼
 SC primitives ── neurons (fixed-point LIF, Q8.8) · synapses · SC layers
    │              (packed 64-bit AND + popcount in VectorizedSCLayer)
    ▼
 Network engine  (network.Network: clock-driven step loop)
    │  per timestep: apply stimuli → CSR projection matvec → population.step_all
    │               → record → STDP plasticity update → optional FIM sync
    ▼
 outputs ── spikes / membrane voltages / firing rates / packed bitstreams
    │
    ▼
 analysis & decode (spike-train statistics, decoders) · spike codecs (BCI) · export
```

Bipolar `[-1, 1]` SC (XNOR multiply) is the fundamental signed primitive — it is
what allows signed-weight networks on SC hardware (`core/bipolar.py`).

### Network engine step loop (verified)

`network.Network` (`network/network.py:59`) is the central simulation entry point.
`run(...)` (`:102`) routes to one of three backends; the Python loop
`_run_python` (`:203`) per timestep:

1. zero the per-population current accumulator;
2. `_apply_stimuli()` (`:220`) — inject configured input currents;
3. `_apply_projections()` (`:221`) — CSR sparse weight matvec across projections;
4. `population.step_all(currents, spike_gating)` (`:225`) — advance every neuron;
5. `_record(...)` (`:227`) — spike/voltage recorders;
6. `_update_plasticity()` (`:229`) — STDP weight updates;
7. optional `_apply_fim()` (`:231`) — Fisher-Information-Metric synchronisation.

Backends: **Rust engine** (high throughput), **MPI** (`network/mpi_runner.py`,
distributed), **pure NumPy** (full feature parity: plasticity, FIM, spike-gating).
Per the source, plasticity / FIM / spike-gating run on the Python path; the Rust
and MPI paths cover a subset (see Audit for the exact gap).

---

## 4. Processing models (what is actually implemented)

### Neuron models — 152 model files (`neurons/models/*.py`, verified count)

Model-agnostic `step`/`reset` contract; each model is a fixed mathematical
formulation, not a simplification. Families present include integrate-and-fire
(LIF, ALIF, adaptive, vmin), Izhikevich (incl. `Izhikevich2007Neuron` with
Rust+Julia+Go+Mojo backends, `neurons/models/izhikevich2007.py:34`),
Hodgkin–Huxley-class conductance models, FitzHugh–Nagumo, resonate-and-fire,
map-based, and bit-true fixed-point variants (`FixedPointLIFNeuron`,
`sc_izhikevich.py`) for RTL verification. (Full per-model SOTA classification is
in the private audit.)

### Synapses & plasticity

- Synapses: static `BitstreamSynapse`, `BitstreamDotProduct`, `StochasticSTDPSynapse`,
  `RewardModulatedSTDPSynapse`.
- Plasticity (`learning/`, `plasticity.py`, `meta_plasticity/`, `online_learning/`):
  pair/triplet/voltage STDP, BCM, STP, e-prop, R-STDP, BPTT/TBPTT, EWC, MAML,
  homeostatic, structural. `_native/learning_bridge.py` exposes Rust STDP/BCM/
  Reward-STDP/ELIGENT rules; a Torch autograd path (`learning_bridge.py:719`)
  computes exact Jacobians of biological traces.

### SC arithmetic & encoders (`core/`, `math/`, `utils/`, `encoding/`)

Unipolar & bipolar SC encoding, LFSR vs Sobol/Halton quasi-random sources,
CORDIV stochastic division, MUX scaled-add, Q8.8 fixed-point throughout
(`constants.py`). Cited literature appears in docstrings (e.g. Gaines 1969,
Alaghi & Hayes 2013, Diehl et al. 2015).

### Spike coding / DSP / normalisation

`spike_codec/` (6 codecs: ISI+varint, predictive, delta, streaming, AER,
waveform — BCI-grade, up to ~50–200× / 24× on 1024-ch), `spike_dsp/` (FIR/IIR on
spike trains), `spike_norm/` (5 SNN normalisers citing 2021–2024 venues).

### Network-level models

Population-Projection-Network with per-synapse learnable delays
(`projection.py:187`, ring-buffer spike history), auto-critical reservoir
(`reservoir/auto_reservoir.py`, mean-field `W_c = θ/(2βNp)`), event-driven
priority-queue simulation (`event_driven/`), topology generators & graph metrics
(`topology/`), MS-ResNet membrane-shortcut deep SNNs (`residual/`).

---

## 5. Backends and their wiring

### 5.1 Canonical SC-inference selector — `accel/backend.py`

`get_backend(name="auto")` (`accel/backend.py:123`) returns a handle whose
`sc_forward(weights_packed, input_probs, *, length, seed)` is the stable SC
inference contract; the Rust and NumPy paths are **bit-identical for a fixed
seed** (`:13`).

- `ACCELERATORS = ("rust", "mojo", "julia", "go")`, `PRIORITY = (*ACCELERATORS, "numpy")`
  (`:29`).
- `_probe()` (`:96`) has working implementations **only for `rust` and `numpy`**;
  for `mojo`/`julia`/`go` it returns `None` (`:109`). So `auto` resolves to
  **Rust if `sc_neurocore_engine` is importable, otherwise NumPy**. `mojo`,
  `julia`, `go` are reserved names in the priority order but are **not yet
  registered** in this selector — see Audit for the gap and the per-package
  kernels that exist outside it.

### 5.2 Rust engine — `engine/` → `sc_neurocore_engine` (PyO3)

The compiled engine exposes SIMD SC inference (`py_sc_forward_packed`),
fixed-point neuron integrators, a 24-module spike-train **analysis** suite
(`engine/src/analysis/`: GPFA, SPADE, point-process, neural decoders, spectral,
information, …), an **IR** with SystemVerilog and MLIR emitters
(`engine/src/ir/`), a **grad** surrogate-gradient path, an optional **wgpu** GPU
backend, and a **scpn** Kuramoto phase solver. Consumed (per the import graph) by
`neurons`, `scpn`, `optimizer`, `chiplet`, `spike_codec`, `sensors`, `profiler`,
`nas`, `network`.

### 5.3 Per-package polyglot kernels — `accel/{julia,go,mojo,rust}/`

A broad polyglot mirror exists: `accel/julia/` (~140 per-package subdirs),
`accel/go/` (~25), `accel/mojo/` (~13). Sampled kernels are **real algorithms,
not shims** (e.g. `accel/julia/.../phi_estimation.jl` Cholesky Φ*;
`accel/go/.../wong_wang.go` Wong–Wang 2006 decision dynamics;
`accel/mojo/kernels.mojo` SIMD popcount + AND/OR/XOR/MUX). They are **research-
stage**: invoked by their own per-package dispatchers (e.g. neuron `HAS_RUST`
flags, `accel/mojo_dispatch.py`), **not** through the canonical
`get_backend`/`sc_forward` path.

### 5.4 Optional accelerators (feature-gated by import guards)

| Backend | Where | Gate | Used by |
|---------|-------|------|---------|
| Numba JIT | `accel/jit_kernels.py` | `accel` extra | bit-pack / MAC hot paths |
| CuPy GPU | `accel/`, `layers/` | `gpu` extra (`HAS_CUPY`) | device arrays for layers |
| JAX | `accel/jax_backend.py`, `adapters/` holonomic | `jax` extra | dense layer, SCPN holonomic adapters |
| PyTorch | `training/`, `conversion/`, `qat/` | `training` extra | surrogate-gradient training, ANN→SNN |
| MPI | `network/mpi_runner.py` | `mpi` extra | distributed network |
| Julia (juliacall) | `network/`, `chiplet/` | `julia` extra | reference ODEs / solvers |

### 5.5 Research crates — `crates/` (7)

`tinysc_riscv`, `core_engine`, `autonomous_learning`, `neuro_symbolic`,
`spike_stats_core`, `evo_substrate_core`, `stochastic_doctor_core`. Mostly
standalone (C-FFI / Criterion-benched); `autonomous_learning` is reachable from
Python via `_native/learning_bridge.py`. (Wiring completeness: see Audit.)

### 5.6 Hardware tier — `hdl/`

25 Verilog modules (LIF Q8.8, axonal delay, CSR int8 sparse matvec, AER
encoder/router, CORDIV, on-chip STDP, SHD 3-stage core + AXI wrapper), formal
(`hdl/formal/` SymbiYosys), PYNQ-Z2 deployment (`hdl/pynq/`). Proven on silicon
via the SHD pipeline.

### 5.7 Benchmarks

A measured corpus exists under `benchmarks/results/` (e.g.
`local_i5_11600k_criterion_2026-05-31_*` per-neuron Rust criterion runs,
`brian2_comparison_results.json`). Note: some per-language speed claims in
docstrings (e.g. Mojo "27–214×", `accel/mojo_dispatch.py:13`) do **not** yet have
a stored measured artifact — treat as unverified until benchmarked.

---

## 6. Hardware compiler / EDA pipeline (Python → ASIC)

Ordered stages, each with its entry symbol (verified end-to-end):

1. **Python model** → neuron equations (strings / IR graph).
2. **SC-NIR build** — `build_scnir_from_neuron_graph()` (`ir/scnir_convert.py`).
3. **Precision analysis** — `compile_for_chip()` (`chip_compiler/compiler.py:70`).
4. **Fixed-point quantisation** — `compile_adaptive_precision()`
   (`compiler/compiler_impl.py:25`, dual-datapath with hysteresis).
5. **AST → Verilog** — `compile_to_verilog()` / `_emit_expr()`
   (`compiler/verilog_compiler.py:19`, `verilog_expr_emitter.py:19`).
6. **HDL bundle** — `build_scnir_source_bundle()` (`ir/scnir_hdl.py:87`).
7. **Top-level assembly** — `VerilogGenerator.generate()`
   (`hdl_gen/verilog_generator.py:59`).
8. **Formal** — `compile_dense_lif_fixture_rtl()` (`formal/property_compiler.py`),
   `generate_sby_script()` (SymbiYosys BMC), Lean 4 proof inventory
   (`formal/lean_bridge.py`).
9. **MLIR** — `CompilerPipeline.compile_mlir_to_verilog()` (`compiler/pipeline.py:47`,
   shells `firtool`).
10. **Synthesis / P&R** — `run_synthesis()` (Yosys), `run_pnr()` (nextpnr)
    (`compiler/pipeline.py:80,105`).
11. **ASIC deck** — `OpenSourcePDKResolver.resolve()` (`asic_flow/asic_flow.py:200`);
    PDKs SKY130, GF180MCU, TSMC28, Intel16, custom.
12. **Chiplet** — interposer models UCIe / BoW / EMIB / CoWoS
    (`chiplet/chiplet_gen.py:73`).
13. **Safety evidence** — fail-closed traceability, caller-supplied FMEDA and
    formal-property records, timing assumptions, checklist bookkeeping, and
    atomic hash-manifest materialisation (`safety_cert/`, library-only; not a
    certification or approval service).

External tools shelled out: `firtool`, `yosys`, `nextpnr`, `openroad`, `iverilog`,
`vivado`, `sby`, `lean`.

---

## 7. Capability catalogue (all 121 subpackages)

Grouped by domain. "Tier" = ships in wheel (**C**ore) or **R**esearch
(source-only). Maturity for the five deep-mapped domains (§3–6) is evidence-based;
the remainder is classified from the structural inventory and refined in the
private audit.

### A. SC substrate & arithmetic — *Core*
`core` · `math` · `utils` · `encoding` · `conversion` · `compression` ·
`stochastic_doctor` · `doctor` · `constants.py` · `exceptions.py`
→ Bitstreams, SC arithmetic, encoders, codecs, diagnostics. `exceptions` is the
most-imported module (in-degree 10). **MAINTAINED.**

### B. Spike coding / DSP — *Core*
`spike_codec` · `spike_dsp` · `spike_norm` · `spike_ode` · `spike_gnn`
→ BCI-grade compression, spike-train filtering, normalisation. **MAINTAINED**
(codecs) / DRAFT (`spike_gnn`, `spike_ode`).

### C. Neurons & synapses — *Core*
`neurons` (152 models) · `synapses` · `sources` · `recorders`
→ The model library + I/O. **MAINTAINED.**

### D. Plasticity & continual learning — *Core / Research*
`learning` · `plasticity.py` · `meta_plasticity` · `homeostasis` ·
`online_learning` · `continual` · `contrastive` · `few_shot` · `transfer` ·
`distillation`
→ STDP family, e-prop, BCM, EWC/MAML, meta-plasticity. **MAINTAINED** (learning,
meta_plasticity, few_shot) / DRAFT (continual, contrastive, transfer, distillation).

### E. Network & dynamics — *Core*
`network` · `layers` · `reservoir` · `topology` · `graphs` · `ensembles` ·
`residual` · `temporal_hierarchy` · `spatial` · `event_driven` · `fusion`
→ Simulation engine + SC layers. **MAINTAINED** (network, layers, reservoir,
topology) / DRAFT (graphs, event_driven, temporal_hierarchy, fusion, residual,
spatial, ensembles — research prototypes not yet wired into the `Network` core).

### F. SCPN integration — *Core*
`scpn` (16-layer L1–L16 stack) · `adapters` (holonomic L1–L16, JAX)
→ Bridge to the SCPN ecosystem. **MAINTAINED.**

### G. Hardware compiler & IR — *Core*
`compiler` · `chip_compiler` · `ir` · `hdl` · `hdl_gen` · `hardware`
→ IR → SV/MLIR → RTL (see §6). **MAINTAINED.**

### H. EDA / ASIC / verification — *Core*
`asic_flow` · `chiplet` · `uvm_gen` · `verification` · `formal` · `safety_cert`
→ ASIC flow, chiplet, formal proofs, UVM, and safety-evidence organisation.
**MAINTAINED** (asic_flow, chiplet, formal; safety_cert as a library-only,
non-certifying evidence surface) / DRAFT (uvm_gen, verification).

### I. Acceleration & polyglot — *Core*
`accel` (rust/julia/go/mojo/jax) · `_native` · `bridges`
→ Backend selection + compiled kernels (see §5). **MAINTAINED** (rust path,
`_native`) / Research (julia/go/mojo per-package surfaces are mixed prototypes,
not canonical dispatch; each needs its own compile/parity evidence).

### J. Interop, serving & deployment — *Core / Research*
`nir_bridge` (18/18 NIR primitives; Norse/snnTorch/SpikingJelly/Sinabs/Rockpool) ·
`export` (ONNX) · `datasets` · `model_zoo` · `models` · `studio` (FastAPI web) ·
`serve` · `hub` · `integrations` (Lava) · `interfaces` · `comm` · `federated` ·
`hypervisor` · `pipeline`
→ Framework interop + serving. **MAINTAINED** (nir_bridge, export, model_zoo,
studio) / DRAFT (serve, hub, federated, hypervisor, comm, pipeline).

### K. Training & optimisation — *Core / Research*
`training` (Torch surrogate-gradient) · `qat` · `nas` · `optimizer` ·
`snn_optimizer` · `augmentation` · `autofit`
→ Gradient training (6 surrogate-gradient paths, 11 neuron cells), QAT, NAS.
**MAINTAINED** (training, optimizer) / library-only (`nas`: design space rich, but
evolutionary search is delegated to the Rust engine with no Python fallback) / DRAFT
(qat, snn_optimizer, augmentation, autofit).

### L. Neuromorphic substrates & physical I/O — *mostly Research*
`bio` · `bioware` · `memristor` · `spintronic` · `optics` · `analog_bridge` ·
`sensors` · `drivers` · `edge` · `robotics` · `world_model` · `transformers` ·
`hdc` · `symbolic` · `neuro_symbolic` · `evo_substrate` · `digital_twin` ·
`audio` · `sensors`
→ Device-substrate models (memristor/spintronic/photonic), MEA bio-hybrid,
edge AER, HDC/VSA, self-replicating SC substrate. Mixed **MAINTAINED / DRAFT**;
several are Research-tier (excluded from wheel). Detailed classification in audit.

### M. Analysis, explainability, profiling, energy — *Research (most)*
`analysis` (spike-train toolkit, 132 functions) · `explain` · `explainability` ·
`debug` · `profiler` · `profiling` · `viz` · `dashboard` · `energy` ·
`energy_accounting`
→ Spike statistics, explanation, energy/CO₂e accounting, HIL debug. **MAINTAINED**
(analysis, profiler, energy_accounting) / DRAFT (others).

### N. Quantum & cognition — *Research (experimental)*
`quantum` (Qiskit/PennyLane VQC bridge) · `quantum_cognition` (Fisher–Posner,
experimental) · `chaos` · `physics`
→ Experimental; `quantum_cognition` and `physics`/`chaos` are research-tier.
**EXPERIMENTAL** — flagged in audit.

### O. Safety, security, resilience — *Core / Research*
`security` · `privacy` · `resilience` · `fault_injection` · `safety_cert` ·
`identity` · `control` · `swarm` · `sleep` · `generative`
→ Threat detection, privacy, fault injection, persistent-identity substrate
(`identity`: 3-population SNN + Lazarus checkpoint), governance. Mixed maturity;
some Research-tier. Detailed in audit.

### P. Wire contracts & misc — *Core*
`proto` (core.proto / telemetry.proto, HIL wire contract) · `experimental` ·
`experiments` · `benchmarks`
→ Protocol buffers + experiment harnesses.

---

## 8. Dependency backbone (intra-package wiring)

Most-depended-on modules (import in-degree, absolute imports only):
`exceptions` (10) · `neurons` (6) · `_native` (6) · `accel` (5) · `energy` (5) ·
`utils` (4) · `compiler` (4) · `hdl_gen` (4) · `scpn` (3) · `learning` (3) ·
`training` (3) · `core` (3) · `solvers` (3) · `analysis` (3).

These are the load-bearing nodes: a change to `exceptions`, `neurons`, `_native`,
`accel`, `utils`, or `compiler` ripples widest. (Relative-import edges are not
captured in this count; the figure is a lower bound.)

---

## 9. Integration points for siblings & STUDIO

- **Stable compute entry**: `from sc_neurocore.accel import get_backend` →
  `sc_forward(...)` (used by SCPN-CONTROL's Petri-net compiler fast path).
- **Framework interop**: `nir_bridge.from_nir/to_nir` (NIR graph in/out).
- **Hardware artifacts**: `compiler` / `hdl_gen` emit `.v`/`.sv`; `asic_flow`
  emits PDK decks; `hdl/pynq/` ships a PYNQ driver.
- **Serving**: `studio` (FastAPI) exposes a web surface; `serve`/`hub` are the
  programmatic serving stubs (DRAFT).
- **SCPN bridge**: `scpn` + `adapters` map the L1–L16 layer stack.
- **Wire contract**: `proto/` defines the HIL telemetry/core protobuf schema.

STUDIO should consume this map's §2 (public API), §5 (backend contract), §6
(hardware pipeline), and §9; it should **not** surface Research/contrib-tier
modules as production capabilities.

---

## Provenance

- Built from a mechanical AST inventory of all 121 subpackages plus five
  evidence-based depth scans (neurons, SC core, network, compiler/EDA,
  acceleration) and direct source verification of the dispatch selector, neuron
  count, and benchmark/FIM claims.
- The earlier `architecture.md` (counts: 174 neurons) and `COMPONENT_INVENTORY.md`
  (lists classes absent from `src/`, e.g. `DigitalSoul`, `PhiEvaluator`) are
  **stale**; this document is the corrected reference. The per-component SOTA
  assessment, maturity ledger, documentation-drift findings, and the per-lane
  remediation plan are maintained separately in the internal audit
  (`docs/internal/audit/`).
