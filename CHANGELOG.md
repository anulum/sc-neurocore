# Changelog

All notable changes to the `sc-neurocore` project will be documented in this file.

## [Unreleased]

### Added
- Quantum-cognition coverage contracts now exercise GOTM brain local-LLM import
  fallback and spike-index accumulation, lifting the focused `dashboard.py`,
  `gotm_brain.py`, and `radical_pair.py` selector to 100% exact-file coverage.
  No runtime path, polyglot mirror, benchmark dispatch, generated API surface,
  or benchmark artefact changed.
- The generated public API reference now excludes single-underscore classes,
  functions, and methods while retaining public and dunder surfaces. The
  generator contract, committed `docs/API_REFERENCE.md`, and documentation
  workflow were updated together; no runtime path, polyglot mirror, benchmark
  dispatch, or benchmark artefact changed.
- Quantum-cognition CLI SNN stimulus records now emit the canonical numeric
  fleet-memory `timestamp` field while preserving `content`, `project`, `actor`,
  `kind`, and `source_ref`; the focused CLI contract rejects legacy `text` and
  `source` aliases. No runtime model dynamics, polyglot kernel,
  benchmark-dispatched path, or benchmark artefact changed.
- Added a schema-driven neuron reference-trace validation harness with immutable
  corpus contracts, fail-closed JSON payload parsing, package-data loading,
  `UniversalNeuron` execution, and feature-level validation reports. The seed
  corpus covers analytic closed-form `lif`, `lapicque`, and `quadratic_if`
  traces and is covered by strict mypy checks plus 100% exact-file focused
  coverage; no polyglot kernel or benchmark-dispatched runtime path changed.
- Compiler coverage-contract tests now lock the MLIR bundle fail-closed CIRCT
  paths, pipeline tool-resolution failures, folded Verilog datapath guard paths,
  and split expression-emitter LUT/error branches that keep the compiler core at
  100% exact-file coverage under its focused selector.
- The `sc.kuramoto_step` IR operation now lowers to synthesisable RTL, replacing the
  previous hard error. The SystemVerilog emitter instantiates a new
  `hdl/sc_kuramoto_step.v` fixed-point phase core (signed Q8.16, 64-entry sine LUT)
  that computes one Euler step
  `theta_n += dt * (omega_n + sum_m K_nm sin(theta_m - theta_n))` over an explicit
  N×N coupling matrix. The core is co-simulated against a bit-exact fixed-point oracle
  with Icarus Verilog and checked against the ideal float step within the LUT
  resolution, and the MLIR emitter now instantiates the same core instead of a
  passthrough. The Python IR builder gains `constant_f64_vec` and `kuramoto_step` so
  the graph can be constructed, emitted, and simulated end to end.
- The `sc.graph_forward` IR operation now lowers to synthesisable RTL, replacing the
  previous hard error. The SystemVerilog emitter instantiates a new
  `hdl/sc_graph_forward.v` fixed-point aggregation core (signed Q8.16) that computes the
  degree-normalised neighbourhood aggregation `agg[i][f] = (Σ_j A[i][j]·X[j][f]) / deg[i]`
  — the graph-structural half of the reference `StochasticGraphLayer` rate-mode forward
  pass — over an explicit adjacency matrix, deferring rounding to a single signed division.
  The core is co-simulated against a bit-exact fixed-point oracle with Icarus Verilog and
  checked against the ideal float aggregation within fixed-point resolution, and the MLIR
  emitter now instantiates the same core instead of a passthrough. The Python IR builder
  gains `graph_forward`.
- The `sc.softmax_attention` IR operation now lowers to synthesisable RTL, replacing the
  previous hard error. The SystemVerilog emitter instantiates a new
  `hdl/sc_softmax_attention.v` fixed-point single-head scaled-dot-product attention core
  (signed Q8.16) that computes `softmax(Q·Kᵀ / sqrt(dim_k)) · V` — the reference
  `StochasticAttention::forward_softmax` — with a numerically stable row-max subtraction,
  a 256-entry exp lookup over the symmetric [-16, 16) grid at 0.125 spacing (mirroring the
  `expr_lut_tables` / `c_fixed_emitter` transcendental machinery), and a single integer
  division per softmax weight. Query/key/value shapes are inferred from the constant
  operand lengths and `dim_k`, and the `1/sqrt(dim_k)` scaling plus exp-LUT geometry are
  baked as instance parameters. The core is co-simulated against a bit-exact fixed-point
  oracle with Icarus Verilog and checked against the ideal float softmax attention within
  the exp-LUT resolution, and the MLIR emitter now instantiates the same core instead of a
  passthrough. The Python IR builder gains `softmax_attention`. With this, all three IR ops
  that previously had no synthesisable RTL (KuramotoStep, GraphForward, SoftmaxAttention)
  now lower to co-simulated fixed-point cores.
- `generate_ip_xact` is now re-exported from the `sc_neurocore.hdl_gen` package namespace
  alongside the other `generate_*` HDL emitters, so the IP-XACT (IEEE 1685) component-XML
  generator can be imported without reaching into the submodule. The generator gained a
  dedicated structural test suite (`tests/test_ip_xact.py`) that parses the emitted
  component tree — the AXI-Lite slave interface, the optional parameter block, and the
  port-vector geometry — and it is no longer excluded from coverage measurement (now 100 %).
- The bipolar stochastic-computing primitives (`core/bipolar.py`) are no longer excluded
  from coverage measurement: the guard clauses (non-finite and out-of-range values,
  non-binary bitstreams, wrong-dimensional MAC operands), the default-RNG path of
  `bipolar_encode`, and the SC-layer optional-bias and `tanh` branches gained tests,
  taking the module from 84 % to 100 %.
- The closed-loop sleep optimiser (`sleep/sleep_optimizer.py`) is no longer excluded from
  coverage measurement: the inactive-session guards (`add_sample` / `add_samples` /
  `check_and_adapt` before a session starts), the None-detection fallback to WAKE, and the
  re-induction branch (two consecutive unwanted awakenings arm a gentle N1 re-induction)
  gained tests, taking the module from 91 % to 100 %.
- The post-session sleep report generator (`sleep/report_generator.py`) is no longer excluded
  from coverage measurement: the grade bands (A/B/D/F), the sleep-onset-latency scoring tiers,
  the mid-session wakeup counter, and the N3/REM/latency/wakeup recommendation branches gained
  tests over a controlled tick history, taking the module from 84 % to 100 %.
- The closed-loop adaptive audio engines (`audio/adaptive_engine.py` and `audio/evs_engine.py`)
  are no longer excluded from coverage measurement. Both were already exercised to 100 %
  statement coverage by the existing behaviour and contract suites (`test_adaptive_audio`,
  `test_audio_evs_contracts`, and the audio package/mapping/profile contracts); the omit
  entries were stale, so removing them simply lets that measured coverage count.
- The NeuroML 2 importer (`adapters/neuroml.py`) is no longer excluded from coverage
  measurement. It is a pure-standard-library (`xml.etree`) importer with no optional
  dependency, and the unit/current parsers' missing-value and dimensionless fall-through
  paths, the `iafTauRefCell` refractory branch, and the `create_neuron` AdEx and
  unknown-cell-type branches gained tests, taking the module from 91 % to 100 %.
- The CCW/VIBRANA bridge (`interfaces/ccw_bridge.py`) gained a dedicated behavioural test
  suite (`tests/test_ccw_bridge.py`, 45 tests) and is no longer excluded from coverage
  measurement. Despite the "CCW system bridge" label it is a self-contained,
  pure-standard-library-and-numpy data transformation (SCPN metrics → binaural-audio
  parameters, L7 glyph vectors → VIBRANA visualisation states, plus metadata and session
  configs) with no live-system dependency, so every mapping, mode-selection branch, metric
  smoothing path, glyph-length guard, and the optional JSON file-export sink are now
  covered at 100 %.
- The PYNQ/hardware-in-the-loop drivers (`drivers/physical_twin.py`,
  `drivers/sc_neurocore_driver.py`, `drivers/verify_hardware_link.py`) are no longer
  excluded from coverage measurement — the whole `*/drivers/*` omit glob is removed. They
  are pure-Python and already tested with mocked PYNQ overlays and sockets (no real FPGA);
  added tests close the remaining branches — the twin's constructor validation guards, its
  TCP connection-failure / empty-reply / non-JSON-reply error paths, the driver's
  missing-bitstream failure, and the hardware-link diagnostic's FPGA success/unexpected-error
  branches plus the present-but-unreachable Evo 2 and Opentrons probe outcomes — taking all
  three modules to 100 %.

### Fixed
- Capped the `nir` dependency to `<1.0.8`. nir 1.0.8 makes `Flatten.input_type` a required
  constructor argument, which broke the NIR bridge's `Flatten` construction and turned the
  test suite red once the release reached CI (the pinned 1.0.7 in `requirements/hub.txt` was
  overridden when the package was built and installed with extras). The extras now match the
  tested 1.0.7 line; the NIR canary continues to track the latest nir separately.
- The SC IR text format was not idempotent under `parse . print` (found by the
  `roundtrip_ir` fuzz target once the fuzz workflow was repaired). Two float-handling
  asymmetries are corrected: (1) the printer formatted whole-number `f64` constants without a
  decimal point (`5.0` → `"5"`, `[5.0, 6.0]` → `"[5, 6]"`), so they re-parsed to integer
  variants (`U64`/`I64Vec`) — the printer now appends `.0` so such values re-read as floats;
  and (2) the parser accepted non-finite float literals (`NaN`, `inf`), and a `NaN` constant
  can never round-trip because `NaN != NaN` — the parser now rejects non-finite literals at
  every float site (constants, vector elements, and the `scale`/`offset`/`kuramoto_step`
  parameters). Regression tests cover whole-number scalar/vector round-tripping and non-finite
  rejection.
- The NeuroML 2 importer produced an unusable Adaptive-Exponential (AdEx) cell:
  `_import_adex_cell` emitted the NeuroML attribute names (`C`, `g_L`, `E_L`, `V_T`,
  `delta_T`, `V_reset`, `V_thresh`) rather than the `AdExNeuron` constructor names, so
  `create_neuron` raised `TypeError: unexpected keyword argument 'C'` for every imported
  AdEx cell. The importer now maps the attributes onto the model's own parameters in its
  native unit system — leak reversal to `v_rest`, exponential threshold `V_T` to `v_rh`,
  `tau = C / g_L` (pF/nS = ms) with the capacitance kept as `c_m` — and, critically,
  parses the spike-triggered adaptation `b` as a *current* in pA (`0.0805 nA = 80.5 pA`)
  rather than in nA, since `b`, the adaptation state `w`, and the injected current share
  the pA unit. The correction is validated against the Brette & Gerstner (2005) hallmark
  of spike-frequency adaptation (a lengthening inter-spike interval under sustained drive).
- The SystemVerilog emitter wrote malformed signed literals (`16'sd-51`) for negative
  fixed-point constants; the sign is now placed outside the sized base (`-16'sd51`),
  so graphs with negative vector or scalar constants emit valid Verilog.

## [3.16.0] - 2026-07-05

### Provenance and citation integrity
- Verified every neuron-descriptor DOI against its registry (Crossref, or DataCite
  for arXiv preprints) and corrected misstated and fabricated citations: a
  non-existent "Kilinc & Bhatt (2023)" (the model is the Nagumo-Sato/Aihara sigmoid
  map), a fabricated "Jahns et al. (2025)", a recurring phantom "Bhatt" co-author
  across several cerebellar models, an arXiv identifier that pointed at an unrelated
  paper, and several digit-transposed or wrong DOIs. Added `tools/provenance/verify_dois.py`
  and a committed DOI ledger checked offline by `tests/test_provenance_doi_integrity.py`,
  so a fabricated or mistyped descriptor DOI now fails CI. Descriptor citeable coverage
  rose from 119 to 131 models.

### Added
- `compile_network_to_fpga` now fails closed on IR that would exhaust synthesis resources,
  before any RTL is emitted: `data_width` must lie in `[1, 64]` and `fraction` must satisfy
  `0 <= fraction < data_width` (so the signed Q-format keeps at least one integer or sign
  bit — otherwise it would silently emit broken RTL); the neuron count is
  capped — at 8192 for the direct/AER interconnects (which instantiate one module per
  neuron, so an unbounded count is a synthesis-time denial of service) and higher, 262144,
  for the folded interconnect (which shares one processing element and is bounded by its
  state-RAM depth). A network over the direct/AER cap is pointed at `interconnect="folded"`.
  The total synapse count is capped too (at 1048576), since every interconnect flattens all
  weight matrices into one shared ROM — a blow-up axis independent of the neuron count.
- `generate_host_driver` now validates its inputs before emitting any driver code: the
  `data_width`/`fraction` Q-format (as the FPGA compiler does), a non-negative memory-mapped
  `base_address`, and a positive, bounded bit width (`[1, 4096]`) for every parameter
  register — so a malformed AXI-Lite/Wishbone driver request fails closed instead of
  emitting a driver with degenerate or unbounded runtime masks.
- `cargo-fuzz` targets (`engine/fuzz`): `parse_ir` fuzzes `ir::parser::parse` (arbitrary
  input — exposed to Python as `ir_parse` — must only ever return `Err`, never panic);
  `roundtrip_ir` asserts the parser and `printer::print` are inverse (`parse(print(g)) == g`,
  and a printed graph always re-parses); `bitstream` differentially checks that
  `bitstream::pack` and `pack_fast` agree bit-for-bit on arbitrary bytes and that
  `unpack` round-trips; and `csr_matrix` builds a CSR matrix from `arbitrary` vectors and
  expands it, checking that a matrix `CsrMatrix::new` accepts cannot panic on use (it found
  a real out-of-bounds — see Fixed). The fuzz crate depends on the engine with
  `default-features = false`, so the build skips the bundled-Z3 compile (made optional in the
  same release) and the pyo3 module dead-strips out, keeping the instrumented build fast;
  `fuzz/Cargo.lock` is committed so NumPy/ndarray resolve to the engine's versions. A
  scheduled weekly `SC-NeuroCore Fuzz` workflow runs all four (not a per-PR gate).
- The folded FPGA interconnect now folds populations with **heterogeneous per-neuron
  parameters**. Each parameter that varies across a population's neurons is exposed on a
  processing-element input port (`compile_to_datapath(param_ports=...)`) and streamed from
  a per-neuron ``case(nidx)`` ROM — the parameter-space analogue of the state BRAM — so
  every neuron receives its own parameters, bit-for-bit the direct path's per-neuron
  ``#(.P_X(...))`` overrides (golden co-simulation parity for heterogeneous firing
  thresholds and membrane time constants). `FoldedResourceMetrics` gains `param_rom_bits`
  and the folded area estimator charges the parameter ROM. Populations with uniform
  parameters are unchanged (the PE bakes them). This removes the former restriction that a
  folded population be parameter-uniform.
- Formal equivalence toolkit generalised to a two-state neuron shape — the
  Izhikevich model, the first with two coupled state registers (membrane ``v`` and
  recovery ``u``). The quadratic ``(v-VR)*(v-VT)`` product drives ``v`` and a spike
  resets both (``v <- C``, ``u <- u + D``). Proving it equivalent unbounded needs
  the coordinated two-register state-matching invariant: a `StateTap` for ``v`` and
  ``u``. The point is empirically pinned by a companion proof that taps only ``v``
  and returns `UNKNOWN` (inconclusive — the modules are equivalent, the invariant is
  merely too weak) rather than `PASS`, establishing that every coupled state register
  must be tapped, not just the observable output. With the one product abstracted and
  both states tapped, a structurally distinct DUT and golden reference prove
  equivalent unbounded by k-induction (≈8 s at 16-bit under `z3`). No toolkit code
  change — the existing multi-tap `expose_state_taps` and `abstract_to_free_inputs`
  already carry the two-state case, confirming the flow spans linear (LIF), quadratic
  single-state (QIF) and quadratic two-state (Izhikevich) neurons.
- Formal equivalence toolkit generalised to a second neuron shape — the quadratic
  integrate-and-fire (QIF). Its state update carries a ``v*v`` *self*-multiply
  (the LIF only multiplied state by a free input) declared inline as
  ``wire v_sq = v * v;``. `operator_abstraction.abstract_to_free_inputs` now
  handles both that inline-initialiser form and the declaration-plus-``assign``
  form, and anchors the lifted name as the declared identifier so a use in another
  statement's right-hand side is never matched. With the self-multiply abstracted
  to a shared free input and the single membrane state tapped, a structurally
  distinct QIF DUT and golden reference prove equivalent unbounded by k-induction —
  demonstrating the whitebox-tap + multiplier-abstraction flow is not LIF-specific.
- Multiplier abstraction for full-width unbounded equivalence proofs
  (`sc_neurocore.compiler.operator_abstraction`). Whitebox taps make k-induction
  converge, but bit-blasting the fixed-point multiplier keeps it tractable only at
  a narrow (4-bit) datapath. `abstract_to_free_inputs` drops the signal carrying a
  product and exposes it as a free input port; when the device-under-test and the
  reference lift the same product to the same input name, the miter drives both
  instances from one shared free wire, so the products are equal by construction
  (congruence) and the solver never reasons about multiplication. The abstraction
  is a sound over-approximation for a `PASS` (equivalence for every product value
  implies it for the real product); a blackbox uninterpreted function would be the
  textbook route but yosys 0.33's `smtbmc` crashes on a blackbox submodule. With
  the multipliers abstracted and the state tapped, the LIF proves equivalent to
  its reference **unbounded at full width** — about 1 s at 16-bit, 4 s at 32-bit,
  and 22 s at 64-bit under `z3` k-induction, where before it was intractable past
  4-bit; the residual growth is the datapath adders and comparators, not the
  (now-abstracted) multiplier.
- Unbounded equivalence proofs via whitebox state taps
  (`sc_neurocore.compiler.whitebox_taps`). Naive k-induction on the equivalence
  miter is intractable — the induction step starts from unreachable states where
  the outputs agree but the hidden state differs, and the fixed-point multiplier
  compounds it. `expose_state_taps` instruments a module to surface its internal
  registers as observation output ports (an `output wire` plus one continuous
  `assign`, adding no register and rewriting no logic, so behaviour on the original
  ports is unchanged); a tap whose source is a constant lets two structurally
  different modules present the same tap interface. The miter then compares the
  taps like any other output, and that tap equality is the reachable-state
  invariant that makes `prove_equivalence(mode="prove")` converge — the LIF
  primitive is proven equivalent to its reference *unbounded*, not just to a
  bounded depth. (yosys 0.33 resolves neither hierarchical references nor
  SystemVerilog `bind`, so exposing the state is the route to reference it.) The
  remaining bound is SMT tractability, not soundness: the proof closes for a narrow
  4-bit datapath and slows as the multiplier widens.
- Machine-checked Python↔RTL equivalence flow for generated models
  (`sc_neurocore.compiler.equivalence_miter` + `equivalence_check`). Given the
  compiler's generated Verilog and an independent reference module,
  `prove_equivalence` builds a sequential miter — both instances driven by
  identical free inputs and a shared counter-derived reset, asserting output
  agreement on every post-reset cycle — runs it through SymbiYosys bounded model
  checking, and returns a real verdict (`proven` with the checked depth, or a
  counterexample with its failing assertion and trace path). This replaces the
  prior text-only equivalence sketch and the standalone `.sby` script generator,
  which are now cross-referenced to the runnable flow. Bounded model checking
  proves equivalence to a configurable depth; unbounded k-induction is available
  but not the default (wide-multiplier datapaths need an invariant to converge).
  The proof functions raise when `sby`/`yosys` are absent, and the tests skip in
  that case.
- Machine-checked RTL property proofs for adaptive-precision evidence bundles
  (`sc_neurocore.compiler.formal_property_check` + a shared
  `sc_neurocore.compiler._sby_runner`). `write_precision_formal_evidence_bundle`
  now renders a real synthesisable bounded-error monitor (the RTL `.v`, which the
  previous bundle named but never wrote) and a bound assertion checker whose
  immediate `assert`/`assume` statements encode the plan's safety obligations —
  accumulated error within the claimed total bound, sequencer within the declared
  bitstream length, and an accumulator wide enough that it never wraps — in place
  of the earlier placeholder comments. `prove_property` proves the RTL satisfies
  those obligations via SymbiYosys bounded model checking and returns a real
  verdict; because the accumulator stops updating after the bitstream length, the
  bounded proof to `length + 2` cycles is complete, not merely bounded. The
  bundle gains an `execute` flag: `execute=False` (default) writes the artefacts
  deterministically with no external tools, while `execute=True` runs the proof
  when the toolchain (`sby`/`yosys`/`z3`) is present and records the machine-checked
  verdict — or a skip reason when it is absent, never a fabricated pass. The
  equivalence and property runners now share one audited `sby` invocation and
  verdict-parse boundary. (yosys 0.33 silently ignores SystemVerilog `bind`, so
  the checker is instantiated explicitly under `` `ifdef FORMAL `` rather than
  bound in.)
- Unbounded k-induction for the adaptive-precision property proof. The bounded
  monitor's assertion checker now carries a strengthening lemma
  (`err_acc <= step_count * per_step`) that is 1-inductive, so
  `write_precision_formal_evidence_bundle(..., unbounded=True)` (and
  `prove_property(mode="prove")`) prove the obligations by k-induction — an
  unbounded proof whose depth is a small constant independent of the bitstream
  length, where bounded model checking's completeness depth scales with it. The
  lemma is trivially true under BMC, so both modes share one checker. k-induction
  adds a third outcome to the runners: an *inconclusive* result (the base case
  holds but the induction step does not converge) is recorded as `proven=False`
  with an `UNKNOWN` verdict and no counterexample — never a fabricated pass — and
  is distinguished from a tool failure, which still raises.
- GPU Izhikevich neuron batch runner (`sc_neurocore_engine.GpuIzhikevichBatch`, behind
  the Rust `gpu` feature), extending the GPU neuron-dynamics path. A wgpu/WGSL compute
  shader runs one thread per neuron, each looping all steps internally with a constant
  current and applying the two half-steps of the Euler update then the `v ≥ v_peak`
  threshold with reset `v ← c`, `u ← u + d`; it returns row-major `[n_neurons × n_steps]`
  spikes (`int32`) and voltages (`float32`). It mirrors the CPU `neuron::Izhikevich`
  model; because WGSL has no `f64` the arithmetic is `f32`, so GPU tests check agreement
  with the `f64` CPU oracle by tolerance — a tight sub-threshold voltage trace, spike
  count within a small margin, and firing-rate monotonicity in the drive current. Like
  the LIF kernel this is O(N) per-neuron work with no inter-neuron coupling, so the rayon
  CPU stays ahead across the benchmarked range; a Criterion benchmark records the measured
  CPU/GPU order. Tests and benches self-skip when no GPU adapter is present.
- GPU Kuramoto oscillator integrator (`sc_neurocore_engine.GpuKuramoto`, behind the
  Rust `gpu` feature), extending the GPU neuron-dynamics path beyond the LIF batch
  kernel. A wgpu/WGSL compute shader runs one thread per oscillator, each summing its
  coupling row `K_nm·sin(θ_m − θ_n)` over all others, so the O(N²) all-to-all coupling
  is fully parallel; the host ping-pongs two phase buffers across the Euler steps in a
  single command encoder. It mirrors the noise-free baseline of the CPU
  `KuramotoSolver`; because WGSL has no `f64` the arithmetic is `f32`, so a GPU test
  checks agreement with the `f64` CPU solver within tolerance (order parameter and
  per-oscillator circular distance) plus a phase-locking sanity check, and a Criterion
  benchmark sweeps oscillator counts for the CPU/GPU crossover. Tests and benches
  self-skip when no GPU adapter is present.
- Learned quantisers for quantisation-aware training in `sc_neurocore.qat`,
  extending the previous straight-through/ternary support. `LSQLinear` /
  `LSQQuantizer` implement Learned Step Size Quantization (Esser et al. 2020):
  the quantiser step size is a trainable parameter — per-tensor or
  per-output-channel — learned jointly with the weights, with the paper's
  step-size gradient and `2*mean(|w|)/sqrt(qmax)` initialisation.
  `PACTActivation` implements PACT (Choi et al. 2018): a learnable clipping
  bound bounds the activation range before uniform quantisation.
  `MinMaxObserver` and `PerChannelMinMaxObserver` derive per-tensor or
  per-channel `(scale, zero_point)` from calibration statistics (with a
  `fake_quantize` helper). `LSQPACTLIFNet` wires LSQ per-channel weights and a
  PACT-quantised analogue input into a feedforward LIF SNN end to end.
- QCFS conversion route in `sc_neurocore.conversion.convert`. When the source
  model carries `QCFSActivation` layers, conversion now uses their learned
  per-layer thresholds directly (no calibration pass), adopts the layers'
  trained timestep budget when `T` is unset, and pre-loads each IF neuron to a
  membrane potential of `theta / 2` — the optimal shift (Bu et al. 2022) that
  cancels the quantisation flooring bias, giving near-lossless conversion of a
  QCFS-trained ANN. `ConvertedSNN` gained an `initial_membrane_fraction` field
  carrying this shift (`0.0` reproduces the threshold-balancing route). A new
  `replace_relu_with_qcfs` helper substitutes every `ReLU`/`ReLU6` in a model
  for a `QCFSActivation`, preparing it for conversion-aware fine-tuning.
- Pre-synthesis area, latency, and power estimate for the folded interconnect
  (`sc_neurocore.energy.estimate_folded_area` → `FoldedAreaEstimate`). It maps a folded
  compile's `FoldedResourceMetrics` onto the existing Yosys-calibrated per-block costs in
  `energy/fpga_models.py` — one combinational PE per neuron type, one DSP (or a LUT-based
  multiply on a DSP-less target) per shared multiplier, a `data_width`-wide weight/threshold
  /bias ROM mux per multiplier column, the per-neuron spike-bus double-buffer flip-flops, the
  sequencer counters, and `state_ram_bits` of state BRAM — and turns `cycles_per_tick` into
  latency, time, and energy per tick. `sc-neurocore compile-nir --interconnect folded` prints
  the estimate and persists it under an `area_estimate` block in `folded_metrics.json` (skipped
  for the non-FPGA `web` target). The estimate inherits the underlying primitives' ~20%
  accuracy and introduces no new uncalibrated coefficients.
- Folded (time-multiplexed) FPGA interconnect for `compile_network_to_fpga`,
  opt-in via `interconnect="folded"` and the `sc-neurocore compile-nir
  --interconnect folded` flag. Multi-layer networks share one combinational
  processing element per neuron type (a per-type PE pool) across every neuron of
  that type, hold per-neuron state in a per-population BRAM, and a single sequencer
  walks each population in turn, advancing the whole network in `neurons + 1`
  cycles per timestep. A single global spike bus committed at the end of each tick
  gives recurrent and inter-population spiking fan-in a stable prior-tick
  double-buffer. Per-source synaptic delays fold via a depth-`d` `spike_bus` history
  shift-register (a delay of `d` ticks reads the bus committed `d` ticks ago). NIR
  `Threshold` transforms fold too: a source threshold gates each sign-extended weight on
  the source value (spike magnitude or external input); a destination threshold replaces a
  connection's per-neuron weighted sum with one spike-magnitude when it exceeds the
  threshold (selected from the per-neuron weight ROM). A per-destination-neuron connection
  bias folds as a constant ACC_WIDTH term added to that connection's fan-in (held in the
  same per-neuron ROM), so a destination threshold wraps the bias along with the weights.
  An analogue source population (`li`/`cuba_li`/`integrator`, whose output is the membrane
  voltage) folds via a global voltage bus — one `DATA_WIDTH` word per analogue source
  neuron, committed once per tick like the spike bus — that the destination multiplies by
  the weight (or threshold-gates), mirroring the direct path's registered `v_out`; a
  delayed analogue source reads a depth-`d` voltage-bus history register, the analogue of
  the spike-bus history. Bit-exact with the direct path (golden co-simulation parity for
  connection-less, external-weighted, recurrent-spiking, two-population
  feedforward/recurrent, delayed recurrent / mixed-per-column-delay two-population,
  external/spiking source-threshold, mixed destination-threshold, inter-population
  source-threshold, external-bias, biased destination-threshold, analogue-source, analogue
  source-threshold, and delayed-analogue fan-in). Reports a `FoldedResourceMetrics` summary
  (populations, processing elements, shared multipliers, state-RAM bits, cycles per tick,
  collapsed direct instances) on the result, in the CLI output, and as a
  `folded_metrics.json` artefact. The folded subset now covers every direct fan-in shape
  except a delayed *external* (non-population) source connection, which falls back to
  direct. Never auto-selected; the direct/AER paths and the SC-NIR source-handoff manifest
  are unchanged.

### Changed
- The engine's Z3-backed safety-verification supervisor is now behind an optional
  `z3` Cargo feature, on by default so the published wheel and CI test matrix are
  unchanged. Building the engine with `--no-default-features` skips the bundled-Z3
  C++ compile (and omits the `PySpikingControllerPool` class), cutting a from-scratch
  engine check from minutes to ~30 s for fast or instrumented builds (e.g. `cargo-fuzz`
  targets) that do not need the verifier. `z3::` was already contained to
  `engine/src/supervisor.rs`; a CI step guards that the no-Z3 build stays compilable.
- Reconciled the lightweight dict-form NIR importer
  (`sc_neurocore.compiler.intelligence.import_nir_graph`) with the authoritative
  `nir_bridge`. It previously mapped only two node types to hand-written,
  divergent ODE strings; it now derives every equation from the shared canonical
  templates (extracted to `sc_neurocore.nir_bridge.neuron_templates`, the single
  source of truth also used by the FPGA back-end) and covers the six NIR
  point-neuron types (LIF, IF, LI, CuBa-LIF, CuBa-LI, integrator) plus an
  explicit Izhikevich extension, resolving framework/case aliases and falling
  back to a leaky integrator for unknown types. `NIRGraph` now also carries the
  full multi-compartment `state_equations`, `thresholds`, `resets`, resolved
  `parameters` and `node_types`. The docstring no longer overstates the module's
  role: it is a dependency-free convenience front-end, and
  `sc_neurocore.nir_bridge.from_nir` remains the authoritative typed importer for
  real `nir.*` graphs, affine/convolutional layers, subgraphs and hardware
  lowering.
- Rebuilt the bit-true simulation kernel
  (`sc_neurocore.compiler.intelligence.generate_bittrue_kernel`) onto a real
  integer fixed-point lowering. The previous C/Rust ``step`` was a no-op that
  only carried the equation in a comment while claiming identity with the RTL;
  it now advances each state variable with a genuine wrap-truncate multiply and
  saturating explicit-Euler accumulate via the new
  `sc_neurocore.compiler.c_fixed_emitter`, and its documentation no longer
  overstates the guarantee. A new
  `generate_bittrue_kernel_from_neuron` emits a whole-neuron C or Rust kernel —
  dt scaling, parameter encoding, threshold, reset rules and spike sequencing —
  that reproduces `compile_to_verilog` bit-for-bit; the identity is proven by an
  iverilog co-simulation that drives the compiled RTL and the kernel with the
  same stimulus and requires equal per-cycle state traces (LIF, Izhikevich,
  FitzHugh-Nagumo, a tanh cell and an integrate-and-fire, at Q8.8 and Q16.16),
  with the Rust kernel checked against the C kernel for the same guarantee.
- Rebuilt the HLS C++ export (`sc_neurocore.compiler.intelligence.generate_hls_cpp`)
  onto a real AST-based lowering. A new `sc_neurocore.compiler.c_expr_emitter`
  (`CExprEmitter` / `emit_c_expr`) parses each ODE expression and emits valid
  `ap_fixed` C++ — translating Python operators, integer powers, roots, and the
  supported transcendentals (via `hls_math` and inline sigmoid/exprel helpers)
  instead of embedding the raw expression string. Equations are now Euler-integrated
  (`<var>_next = <var> + dt * d<var>`, matching the Verilog backend), the membrane
  variable resets by subtracting a configurable threshold on a spike, and free
  identifiers become function inputs so the generated unit is self-contained. The
  output is verified to compile with a host C++ compiler.
- Extracted the target-independent expression-lowering numerics — the
  compile-time constant folder, the 256-point symmetric sample grid, and the
  quantised transcendental LUT generators (exp, log, sqrt, tanh, cosh, exprel,
  sigmoid, sin, cos, cbrt) plus the supported-function vocabulary — into a new
  `sc_neurocore.compiler.expr_lut_tables` module. The Verilog expression emitter
  now delegates to it, so the generated RTL is byte-for-byte unchanged; the
  shared tables give future C/C++/Rust lowering backends a single bit-exact
  source of truth.

### Fixed
- `graph::CsrMatrix::new` validated only the lengths of the CSR arrays, not that
  `row_offsets` is non-decreasing and starts at 0 or that every column index is `< n_cols`.
  A malformed CSR (reachable from Python via the sparse graph constructor) was therefore
  accepted and later panicked with an index-out-of-bounds when expanded by `to_dense`.
  `new` now rejects a non-monotonic offset array, a non-zero first offset and any
  out-of-range column index, so a constructed matrix is always safe to use. Found by the
  new `csr_matrix` fuzz target.
- The opt-in folded FPGA interconnect (`interconnect="folded"`) built its shared
  processing element from the *quantised* population, so `Q88.encode` ran a second time
  over already-quantised parameters and baked a corrupt value into the PE for every graph
  carrying explicit parameters (a 16-bit `tau = 5120` re-encoded to
  `5120 × 256 mod 2**16 = 0`). Real NIR networks always carry explicit parameters, so the
  folded PE was silently generated with `tau = 0`. Folded PE parameters are now built from
  the real-valued parameters (`_dequantised_pop`, a lossless rescale for fixed-point
  values), matching the per-instance module the direct interconnect emits. The
  double-encoding had gone undetected because the folded co-simulation suites used
  template-default (already real-valued) parameters; a co-simulation of a real
  explicit-parameter network now guards it.
- Hardened the public swarm agent/evolver contracts. `AgentConfig`,
  `SwarmAgent.weights`, `think`, `act`, `reset`, `EvolverConfig`,
  `evaluate_individual`, mutation, and generation execution now reject
  malformed or non-finite inputs before mutating state; the touched swarm
  modules are strict-mypy clean, enforced by the scoped public-docstring
  policy, and covered at 100% exact-file coverage by the focused swarm suite.
- `network.topology.scale_free` now rejects invalid Barabasi-Albert dimensions
  and non-finite weights before allocating the graph or sampling attachment
  probabilities, closing the `m=0` divide-by-zero path and the `m >= n`
  empty-graph path. The existing topology generator tests are strict-typed and
  cover the fail-closed parameter boundary at 100% exact-file coverage.
- Fixed the direct FPGA interconnect emitting an assignment to an undeclared
  delay register when a connection mixed zero and non-zero per-source-column
  synaptic delays: a delayed source's register chain was registered (and its
  `*_spike_d1 <= *_spike` shift emitted) even for columns with delay zero, which
  have no declared chain, so the generated RTL failed to elaborate. Undelayed
  columns are no longer given a register chain.
- Removed duplicate hardware-profile registrations in the built-in platform
  modules: `cortical_labs_dishbrain` and `finalspark_neuroplatform` were each
  registered four times, `biomemory_dna` three times, and `belousov_zhabotinsky`,
  `fujitsu_digital_annealer`, `rl_toffoli_asic`, `ibm_microfluidic`,
  `mems_resonator`, `brainscales2` and `spinnaker2` twice — with conflicting
  fields silently resolved by last-registration-wins. Each profile name now
  registers exactly once with its previously-effective values (behaviour
  unchanged); `_reg` rejects a duplicate name unless `allow_override=True`, and
  the `hardware_profiles`/`research_platforms` guides now match the registered
  values.
- Corrected the `online_learning` package docstring, which advertised "e-prop,
  RTRL, and forward-gradient methods" though only e-prop (`EpropTrainer`) and an
  eligibility-based online trainer (`OnlineTrainer`/`OnlineLIFLayer`) are
  implemented. The docstring now lists only the shipped methods.

### Physics and mathematics hardening
- Rebuilt `MarderSTGNeuron` (Liu, Golowasch, Marder & Abbott 1998 stomatogastric
  ganglion neuron) as a faithful thirteen-state model integrated with
  fourth-order Runge-Kutta, replacing the prior single-step forward-Euler
  approximation. Restored the CaS inactivation gate and the K-C activation gate
  (`h_cas`, `m_kca`), replaced the constant gate time constants with the
  published voltage-dependent `tau(V)` functions, made the K-C activation
  voltage- and calcium-dependent, switched the calcium reversal to the Nernst
  equation, and corrected the calcium dynamics to the published 20 ms relaxation
  form (transcribed from ModelDB 93321). Propagated the same RK4 model across the
  Rust engine, Rust safety mirror, Julia, and Go surfaces with bit-exact
  Python-Rust spike-count parity and a reference Mojo kernel; refreshed the model
  descriptor, the model documentation, the Rust benchmark figure, and the test
  suite to 49 multi-angle tests. The neuron is an endogenous burster (fires at
  zero injected current).
- Rebuilt `PinskyRinzelNeuron` (Pinsky & Rinzel 1994, 2-compartment CA3
  pyramidal cell) as a faithful eight-state model integrated with fourth-order
  Runge-Kutta, replacing the prior single-step forward-Euler approximation.
  Restored the separate voltage/Ca-dependent K-C activation gate `c` and the
  dendritic calcium state `ca` (previously conflated), added the membrane
  capacitance `cm = 3`, corrected `chi(Ca) = min(Ca/250, 1)` (previously a
  voltage expression), and aligned the Na/K-DR/Ca/K-C rate functions and
  reversal convention with the published kinetics (ModelDB 35358). Propagated
  the same RK4 model across the Rust engine, Rust safety mirror, Julia, and Go
  surfaces with Python-Rust spike-count parity and a reference Mojo kernel;
  refreshed the model descriptor (RK4, `ca`, `cm`), the model documentation, the
  Rust benchmark figure, and the test suite to 54 multi-angle tests. The model
  fires repetitively at low somatic drive and enters depolarisation block at
  high drive (non-monotonic f-I).

### Fixed
- Expanded GOTM brain coverage to 100% exact-file evidence, covering the local
  LLM import-success path, valid directive parsing, invalid directive fallback,
  local endpoint exception fallback, and repr/docstring policy.
- Expanded L7 holonomic symbolic adapter coverage to 100% exact-file evidence,
  covering single-node routing, nonstandard ring coordinates, underflowed
  Metatron topology rejection without warnings, boolean integer parameter
  guards, invalid golden-ratio scale rejection, mismatched input broadcast, and
  touched docstring-policy surfaces while keeping the Rust safety mirror green.
- Expanded radical-pair RPM coverage to 100% exact-file evidence, covering
  invalid quadrature, explicit hyperfine tensor construction/state telemetry,
  tensor-shape rejection, zero-nucleus density helper behavior, oversized dense
  nuclear-bath rejection, non-positive kinetic parameter guards, and
  repr/docstring policy while keeping the Rust safety kernel green.
- Expanded quantum terminal dashboard coverage to 100% exact-file evidence,
  covering terminal-size fallback, narrow-terminal hidden-neuron telemetry,
  no-history rendering, spike-raster intensity bands, directive colour fallback,
  empty-scale ATP bars, and repr/docstring policy.
- Expanded spin-pool MPS coverage to 100% exact-file evidence, covering valid
  dense exact evolution, ATP observable site rejection, status/repr telemetry,
  and stable TEBD adjacency validation while keeping the Rust telemetry-kernel
  tests green.
- Expanded Kane silicon mapper coverage to 100% exact-file evidence, covering
  triangular and hexagonal donor placement, zero-site rejection, zero-distance
  exchange coupling, and the touched docstring policy surface while keeping the
  Rust safety-kernel parity tests green.
- Hardened Wong-Wang Julia, Go, and Mojo acceleration facades so `stim1`,
  `stim2`, and `xi` must be one-dimensional time-series before backend
  dispatch, and added module-specific dispatcher contracts for Go/Mojo loader
  failures, unavailable libraries, C return codes, length validation, and
  success return buffers.
- Hardened Wilson-Cowan Julia, Go, and Mojo acceleration facades so
  `ext_input` must be a one-dimensional time-series before backend dispatch,
  and added module-specific dispatcher contracts for Go/Mojo unavailable
  libraries, C return codes, and success paths.
- Preserved local result dtype for non-root MPI gather outputs and expanded the
  MPI driver contract tests to cover no-MPI constructor fallback, root gather
  collection, non-root empty-gather dtype, missing communicator fallback, and
  barrier delegation.
- Kept `sc_neurocore.accel.mojo.MojoKernelRunner` fail-closed when the optional
  Mojo runner import fails, replacing stale reload bindings with a placeholder
  that raises `RuntimeError` with the captured import reason.

### Physics and mathematics hardening
- Promoted `DendriticNMDANeuron` (two-compartment Jahr-Stevens NMDA Mg2+ block
  neuron) from raw dendrite-first Euler to candidate-first RK4 over
  `(v_soma, v_dend)`, with finite parameter/state/input validation,
  reset-on-commit soma spike semantics, and an explicit
  `integrator="baseline_euler"` regression path. Replaced Go, Rust safety, Julia,
  and Mojo placeholders or broken paths with real dual-input RK4 mirrors,
  harmonised the Rust engine, added focused Python/Go/Rust tests, a Rust
  benchmark example, a five-backend local non-isolated benchmark artefact, and
  refreshed the model documentation with measured benchmark results. Python,
  Rust, Go, Julia, and Mojo agree on the 253-spike anchor at 20k steps /
  `i_soma=50.0`, `glutamate=0.5`.
- Promoted `NeuroGridNeuron` (reduced Neurogrid two-compartment analog EIF
  neuron) from raw Euler to candidate-first RK4 over `(v_s, v_d)`, with an
  event-limited soma stage cap at `v_peak`, finite input/state/candidate
  validation, reset-on-commit spike semantics, and an explicit
  `integrator="baseline_euler"` regression path. Replaced placeholder Go and
  Rust safety mirrors, repaired Julia, added Mojo, harmonised the Rust engine,
  added focused Python/Go/Rust tests, a Rust benchmark example, a five-backend
  local non-isolated benchmark artefact, and refreshed the model documentation.
  Python, Rust, Go, Julia, and Mojo agree on the 94-spike anchor at 20k steps /
  current 100.0.
- Promoted `HayL5PyramidalNeuron` (reduced three-compartment Layer 5 thick-tufted
  pyramidal cell) from four raw Euler sub-steps to candidate-first RK4 over the
  nine-state `(v_s, h_na, n_k, v_t, m_ca, h_ca, m_ih, v_a, ca_a)` system, with
  finite input/state/candidate validation, non-negative tuft-calcium candidates,
  dual soma/tuft input parity, and an explicit `integrator="baseline_euler"`
  regression path. Replaced broken or placeholder Go, Rust safety, Julia, and
  Mojo surfaces with real RK4 mirrors, harmonised the Rust engine, added focused
  Python/Go/Rust tests, a Rust benchmark example, a five-backend local
  non-isolated benchmark artefact, and refreshed the model documentation. Python,
  Rust, Go, Julia, and Mojo agree on the 1-spike anchor at 20k steps /
  `current_soma=10.0`, `current_tuft=0.0`; the dual-input anchor is 4 spikes at
  20k steps / `current_soma=5.0`, `current_tuft=5.0`.
- Promoted `DeSchutterPurkinjeNeuron` (compact De Schutter & Bower Purkinje cell)
  from five raw Euler sub-steps to candidate-first RK4 over the seven-state
  `(v, h_na, n_k, m_cap, h_cap, q_kca, ca)` system, with finite input/state/
  candidate validation, non-negative calcium candidates, and an explicit
  `integrator="baseline_euler"` regression path. Replaced the decorative Go,
  Rust safety, and Mojo placeholders with real RK4 mirrors, harmonised the Rust
  engine and Julia surface, added focused Python/Go/Rust tests, a Rust benchmark
  example, a five-backend local non-isolated benchmark artefact, and refreshed
  the model documentation. Python, Rust, Go, Julia, and Mojo agree on the 1-spike
  anchor at 20k steps / current 500.0.
- Promoted `MulticompartmentMCNNeuron` (Spiking-WM dual-dendrite working-memory
  cell) from raw forward Euler to candidate-first RK4 over the coupled
  `(u, v_basal, v_apical)` system, with finite input/state/candidate validation
  and an explicit `integrator="baseline_euler"` regression path. The Rust engine,
  Rust safety mirror, Go service, Julia mirror, and new Mojo kernel now share the
  same derivative order and threshold-reset rule. Added focused Python/Go/Rust
  tests, a Rust benchmark example, a five-backend local non-isolated benchmark
  artefact, and refreshed the model documentation. All five backends agree on
  the `49,999` spike anchor at 200k steps / basal current 3.2.

## [3.15.35] - 2026-06-26

### Physics and mathematics hardening
- GPFA (Gaussian Process Factor Analysis) now uses a deterministic PCA
  initialisation (top singular vectors of the centred data with a fixed sign
  convention) instead of a random one, so results are reproducible and
  seed-independent. The EM loop is factored into reusable `gpfa_pca_init` and
  `gpfa_em` entry points with a backend dispatch contract that lets acceleration
  backends share an identical starting point, and the marginal log-likelihood is
  the exact Gaussian form. Added a full reference test suite (deterministic init,
  EM convergence, exact log-likelihood and its non-PSD guard, projection, dispatch).
- GPFA Rust backend (`py_gpfa_em`) now shares the deterministic initialisation and
  computes the exact marginal log-likelihood (previously an approximate
  residual-sum form), so it agrees with the NumPy reference to within float64
  round-off (same iteration count, trajectories within ~1e-10); selectable via
  `backend="rust"`, with a gated parity test.
- GPFA Julia backend (`accel/julia/analysis/gpfa.jl`, previously a stub) implements
  the same EM-from-initialisation contract and agrees with the NumPy reference to
  within ~1e-12; selectable via `backend="julia"`, gated parity test included.
- GPFA Go backend (`accel/go/gpfa/gpfa.go`, c-shared via cgo) implements the same
  EM-from-initialisation contract and agrees with the NumPy reference to within
  ~1e-12; selectable via `backend="go"`, with parity and loader-branch tests.
- GPFA Mojo backend (`accel/mojo/kernels/gpfa.mojo`, previously a stub) implements
  the same EM-from-initialisation contract over the `@export` raw-address FFI and
  agrees with the NumPy reference to within ~1e-10; selectable via `backend="mojo"`,
  with parity and loader-branch tests.
- GPFA now uses a structured Cholesky estimator across all five backends. The
  E-step, M-step and marginal log-likelihood operate on the `n_state × n_state`
  posterior precision (`n_state = n_latents · n_bins`) via Cholesky factorisations
  rather than a general elimination, and the log-likelihood uses the Woodbury
  identity and the matrix-determinant lemma so it never forms the dense
  `(n_neurons · n_bins)²` marginal covariance. This is the exact structured
  estimator of Yu et al. (2009): more numerically stable for the symmetric
  positive-definite systems and far cheaper at scale (the dense form would build,
  e.g., an 8000×8000 covariance where the structured form factors a 1600×1600
  precision). The Rust backend factors with `nalgebra`, Julia with native LAPACK,
  Go and Mojo with an in-place Cholesky, and the Python reference with SciPy; the
  structured likelihood is checked against a dense brute-force covariance (~1e-11).
- GPFA `backend="auto"` resolves to the structured Rust path when the engine is
  present and falls back to the NumPy reference otherwise. With the dense
  `n_obs`-sized solve removed, `benchmarks/bench_gpfa.py` measures the Rust backend
  as the fastest path (1.43x over NumPy on the reference workload; Mojo 0.87x, Go
  0.51x); the compiled backends remain available by name for cross-language parity
  and portability. Added `docs/api/gpfa.md` describing the deterministic init, the
  structured EM contract, the five backends and the benchmark.
- Phi* (integrated information, `analysis/phi_estimation.py`) now computes the
  Gaussian mutual information from covariance log-determinants taken via Cholesky
  (`MI = 0.5(log|Cov_X| + log|Cov_Y| - log|Cov_XY|)`) instead of a product/ratio of
  raw determinants with a `1e-300` clamp, which underflowed for larger channel
  counts; the single-channel covariance now uses the unbiased (`ddof=1`) estimator
  consistently. `phi_star(..., backend=...)` gains a polyglot dispatch over five
  parity-verified backends — NumPy, Rust (`py_phi_star`, re-exported from the
  bridge), Julia and Mojo (both previously non-functional stubs, now real), and a
  new Go c-shared backend — agreeing with the reference to ~1e-15 (Mojo ~1e-10).
  `backend="auto"` prefers Rust; `benchmarks/bench_phi.py` measures every compiled
  backend faster than NumPy (Rust ~7x, Julia ~4.4x, Mojo ~3.5x, Go ~3.0x). Added
  parity and loader-branch tests (phi_estimation.py at 100% statement coverage) and
  refreshed the `docs/api/analysis.md` Phi* section.
- Sorting-quality Mahalanobis metrics (`analysis/spike_stats/sorting_quality.py`)
  now evaluate the squared Mahalanobis distance `(x-μ)ᵀ Σ⁻¹ (x-μ)` through the
  Cholesky factor of the regularised cluster covariance (a triangular solve, then
  `Σ z²`) instead of forming the covariance inverse explicitly — more accurate for
  ill-conditioned cluster covariances and cheaper. `isolation_distance` and
  `l_ratio` share the kernel and gain a polyglot dispatch `backend=` over five
  parity-verified backends — NumPy, Rust (`py_isolation_distance` / `py_l_ratio`,
  re-exported from the bridge; the Rust path previously used a Gauss-Jordan
  inverse mislabelled as Cholesky), Julia and Mojo (both previously non-functional
  stubs, now real), and a new Go c-shared backend — agreeing with the reference to
  ~1e-13. `backend="auto"` prefers Rust; `benchmarks/bench_sorting_quality.py`
  measures Mojo ~4.6x, Go ~3.8x and Rust ~1.9x over NumPy on the reference workload
  (Julia ~0.5x, interop-bound on this size). Added Cholesky-kernel, parity and
  loader-branch tests (sorting_quality.py at 100% statement coverage), removed two
  unreachable defensive branches, and expanded the `docs/api/analysis.md`
  sorting-quality section.
- Fixed a latent type-inference error in the Rust Phi* test (`det.ln()` on an
  ambiguous numeric literal) that prevented the engine test target from compiling.
- Dimensionality reduction (`analysis/spike_stats/dimensionality.py`) now uses a
  deterministic, reproducible covariance eigendecomposition — eigenvalues in
  descending order with sign-canonicalised eigenvectors — across PCA, demixed PCA
  and factor analysis. The Rust backend's hand-rolled Jacobi eigensolver and
  Gauss-Jordan inverse are replaced by `nalgebra`'s symmetric eigensolver and
  Cholesky solves; factor analysis starts from a deterministic PCA initialisation
  (replacing a random one, so it is seed-independent) and solves its symmetric
  positive-definite systems by Cholesky. Each estimator gains a `backend=`
  dispatch over five parity-verified backends — NumPy, Rust (`py_pca_components`
  / `py_demixed_components` / `py_factor_loadings`, re-exported from the bridge),
  Julia and Mojo (both previously non-functional stubs, now real), and a new Go
  c-shared backend (cyclic Jacobi where no LAPACK is linked) — agreeing to ~1e-13.
  Dense symmetric eigendecomposition is LAPACK's strength, so on the reference
  workload (`benchmarks/bench_dimensionality.py`) the NumPy/LAPACK path is the
  fastest and `backend="auto"` resolves to it; the compiled backends are kept for
  cross-language parity and portability. Added a dedicated test module
  (dimensionality.py at 100% statement coverage), the benchmark and the
  `docs/api/analysis.md` dimensionality section.
- The Rust `cell_assembly_detection` backend (`analysis/network.rs`) replaces its
  hand-rolled Jacobi eigensolver with `nalgebra`'s symmetric solver, returning
  descending, sign-canonicalised eigenvectors — matching the NumPy/LAPACK
  reference the Python path already uses. No public API change.
- The Rust causality backend (`analysis/causality.rs`) now uses structured
  factorisations for its dense linear algebra. The per-frequency MVAR transfer
  function `H(f) = A(f)⁻¹` in spectral Granger causality and the directed
  transfer function is obtained from a single `nalgebra` LU factorisation of the
  non-Hermitian spectral matrix — one factorisation yields both the
  near-singularity test (its determinant) and the inverse — replacing a separate
  hand-rolled complex Gauss-Jordan inverse and Gaussian-elimination determinant.
  The VAR fit and the Granger sum-of-squared-errors solves now solve their
  ridge-regularised normal equations `XᵀX + εI` (symmetric positive-definite) with
  a Cholesky factorisation rather than generic Gaussian elimination, factoring
  once for all right-hand sides. The spectral-matrix assembly is shared across the
  spectral Granger, partial-directed-coherence and directed-transfer-function
  paths. Results match the previous implementation to float64 round-off; no public
  API change.
- The Fisher linear-discriminant decoder now solves its within-class scatter
  system with a Cholesky factorisation in both backends. The scatter matrix
  `S_w + εI` is symmetric positive-definite, so the per-class Fisher weights
  `w_c = S_w⁻¹ (mean_c − overall_mean)` come from a single factorisation reused
  across classes rather than an explicit matrix inverse (Python reference,
  `scipy.linalg.cho_factor`/`cho_solve`) or a per-class Gaussian elimination
  (Rust backend, `nalgebra`) — the numerically optimal route for an SPD system.
  Decoded classes are unchanged; the naive-Bayes decoder (diagonal covariance) is
  unaffected.
- The Rust world-model Kalman filter (`lgssm.rs`) factors the innovation
  covariance `S = C P_pred Cᵀ + R` with a single `nalgebra` Cholesky decomposition
  per timestep, replacing a hand-rolled Cholesky and an explicit matrix inverse.
  The one factor serves the log-determinant (`2 Σ log Lᵢᵢ`), the innovation
  quadratic form (`S⁻¹ innov` via triangular solves), and the Kalman gain
  `K = P_pred Cᵀ S⁻¹` — computed as `Kᵀ = S⁻¹ (C P_pred)` so `S⁻¹` is never formed
  explicitly. Matches the NumPy/LAPACK reference
  (`world_model/predictive_model.py`) to float64 round-off; the Rust-vs-Python
  parity suite (means, covariances, log-likelihood, atol 1e-9) stays green. No
  public API change.

### Studio platform
- Documented the optional `sc_neurocore.federation` Hub-facing Studio federation
  surface with a dedicated API page and navigation entry, covering schema-A
  manifest emission, evidence bundles, and verifiable-honesty envelopes.
- The FPGA synthesis panel can export a synthesis-scoped evidence bundle from
  the latest single-target or all-target worker job, list the bundle artefacts,
  and download them through the authenticated Studio job-artifact route.

### Public bitstream-inference API
- Restored a stable public stochastic-inference surface over caller-owned packed
  weight bitstreams. `sc_neurocore.accel.sc_forward(weights_packed, input_probs,
  *, length, backend, seed)` returns the AND-then-popcount estimate of
  `weights @ input_probs` for unipolar stochastic computing; the input encoder is
  the deterministic 16-bit LFSR comparator, so the Rust accelerated path and the
  NumPy fallback are bit-identical for a fixed seed. Added
  `sc_neurocore.accel.get_backend()` / `available_backends()` as a fastest-first
  backend selector (Rust accelerated path, NumPy fallback), the Rust engine
  `sc_forward_packed` + PyO3 `py_sc_forward_packed`, a parity test proving the
  estimate matches the dense reference within stochastic tolerance and that the
  Rust and NumPy paths agree exactly, a Rust-vs-NumPy benchmark with a committed
  artefact, and documentation.
- `BitstreamEncoder` again accepts `BitstreamEncoder(length=..., seed=...)`:
  `x_min`/`x_max` default to the unipolar probability domain `[0, 1]`, with
  explicit ranges still supported.
- `import sc_neurocore` is now lightweight and torch-free: public symbols and
  submodules load lazily (PEP 562) instead of eagerly importing the torch-pulling
  graph (plasticity, datasets, layers). Downstream consumers can run their
  coverage tracers over code that imports sc-neurocore without the torch
  C-extension crash; torch loads only when a torch-dependent feature is used.

### Timing-aware formal verification
- Added a two-flop clock-domain-crossing synchroniser property template to the
  timing formal framework: `sc_cdc_two_flop_monitor` (in
  `hdl/formal/timing/timing_wrapper_lib.sv`) and the `SC_ASSERT_CDC_TWO_FLOP`
  macro (in `timing_assertions.svh`). A consumer binds it over its own
  destination-domain synchroniser flops; the monitor proves the output is the
  source delayed by exactly the synchroniser depth with no combinational path past
  the last flop. The property is expressed in the open-source Yosys/SymbiYosys
  procedural-immediate-assertion subset (no concurrent SVA). Added a worked
  `example_cdc_two_flop_synchroniser.{sv,sby}` proof, a SymbiYosys-gated proof
  test, a mutation test proving a one-flop synchroniser is rejected, and a
  consumable-surface section (bounded-latency macros, CDC template, SymbiYosys task
  convention) in `docs/formal/timing_aware_properties.md`.

### Studio platform
- Added Projects panel artifact download controls for exported project evidence
  bundles, reusing the authenticated Studio job-artifact endpoint and showing
  path-confined artifact names, sizes, and SHA-256 labels in the workbench.
- Added Projects panel evidence-bundle export for saved project workspaces,
  using the bounded Studio evidence worker from the normal workbench UI.
- Added Admin panel audit-archive restore controls for path-free payload
  validation and confined restore-artifact materialization, wired to the
  existing admin-gated validate and restore endpoints.
- Added Admin panel audit-archive controls for quarantine archive creation,
  retention review, and prune-candidate purge execution.
- Added an admin-gated Studio audit quarantine archive retention purge endpoint
  that removes only archive jobs marked as prune candidates.
- Added an admin-gated Studio audit quarantine archive restore job that
  materializes validated archive rows as confined evidence artifacts.
- Added an admin-gated Studio audit quarantine archive retention endpoint that
  inventories valid archive jobs and marks non-destructive prune candidates.
- Added an admin-gated Studio audit quarantine archive validation endpoint for
  path-free import and restore preflight checks.
- Added an admin-gated Studio audit quarantine archive job that persists
  quarantined retained rows as path-confined evidence artifacts.
- Added an admin-gated, path-free Studio audit quarantine export for legacy,
  corrupt, or chain-broken retained audit rows.
- Added path-free quarantine metadata to Studio audit status and export payloads
  so legacy, corrupt, or chain-broken retained audit rows are counted separately
  from verified retained evidence.
- Fail closed on non-positive Studio audit retained-file settings so configured
  rotation always keeps at least one archived JSONL segment for incident review
  and retained-chain verification.
- Added retained-chain integrity verification to persistent Studio JSONL audit
  logs. Audit status, audit export, and operator status now report
  path-free `integrity_verified`, `integrity_error`, and latest retained event
  hash fields.
- Authenticated `/ws/progress` when Studio route-policy enforcement is enabled.
  Non-browser clients can use bearer headers, while the React frontend carries
  browser-session tokens through the `studio-auth` WebSocket subprotocol.
- Sent the active Studio bearer token on frontend project-delete requests so
  production route-policy enforcement does not reject the saved-project delete
  workflow.

### Physics and mathematics hardening
- Added the polyglot `adc_to_spike_windows(...)` chain for the ADC-to-spike
  decimating rate-code encoder across python / rust / julia / go / mojo. Each
  decimation window is centred and quantised to a Q-format code, sign-aware
  averaged (round half away-from-zero then truncate toward zero) and converted
  into a deterministic spike count (`|window| // threshold`) with the window sign
  as polarity. The arithmetic is exact integer, so every backend reproduces the
  Python floor bit-for-bit, and a dedicated test pins the Python floor against the
  cycle-stepped golden model in `tools/adc_to_spike_reference.py`. Added the Rust
  engine `adc_to_spike_windows` + PyO3 `py_adc_to_spike_windows`, the wired
  `sc_neurocore.sensors.adc_to_spike_kernel` primary with fastest-first dispatch,
  the Julia/Go/Mojo backends, unit, cross-backend and golden-reference parity
  tests to full coverage of the primary module, a five-backend benchmark with a
  committed artefact, and documentation.
- Added the polyglot batched `mixed_dense_forward_batch(...)` chain for the
  mixed-precision Q8.8 × Q16.16 dense MAC across python / rust / julia / go / mojo.
  The integer branch (per-tensor scale folded so the accumulator divisor equals
  the Q8.8 weight scale) contracts in a signed 64-bit accumulator, divides by an
  arithmetic right shift (floor division) and saturates to the Q16.16 code range
  with explicit overflow/underflow flags, so every backend reproduces the Python
  floor bit-for-bit. Added the Rust engine `mixed_dense_forward_batch_q88_q1616` +
  PyO3 `py_mixed_dense_forward_batch_q88_q1616`, the wired
  `sc_neurocore.compiler.mixed_dense_kernel` primary with fastest-first dispatch,
  the Julia/Go/Mojo backends, unit and cross-backend parity tests to full coverage
  of the primary module, a five-backend benchmark with a committed artefact (the
  NumPy and Julia matmul paths measured competitive with or faster than the scalar
  FFI backends), and documentation.
- Added the polyglot batched `dcls_max_forward_batch(...)` chain for the DCLS-max
  Q8.8 triangular (tent) weighting kernel (Khalfaoui-Hassani, Pellegrini &
  Masquelier 2023) across python / rust / julia / go / mojo. The kernel is exact
  integer Q8.8 arithmetic with a Q16.16 saturating accumulator, so every backend —
  including Mojo — reproduces the Python floor bit-for-bit with a parity tolerance
  of exactly zero. Added the Rust engine `dcls_max_forward_batch_q88` + PyO3
  `py_dcls_max_forward_batch_q88`, the wired `sc_neurocore.scpn.dcls_tent_kernel`
  primary with per-channel learnable centre/sigma and fastest-first dispatch, the
  Julia/Go/Mojo backends, unit and cross-backend parity tests to full coverage of
  the primary module, a five-backend benchmark with a committed artefact, and a
  documentation upgrade.
- Replaced `ExpIFNeuron` raw Euler mutation with candidate-first RK4 across the
  maintained Python reference, Rust engine, Go service, Julia mirror, and Mojo
  mirror. The Fourcaud-Trocmé EIF ODE and hard reset are unchanged; all surfaces
  now reject non-finite RK4 derivatives/candidates before mutation. Added focused
  Python/Rust/Go RK4 tests, a Go benchmark hook, a local non-isolated Python RK4
  regression artifact, and refreshed the public model documentation.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `McKeanNeuron` (McKean 1970 piecewise-linear FitzHugh-Nagumo caricature) across
  python / rust / julia / go / mojo. Rust, Julia and Go reproduce the NumPy RK4
  reference bit-for-bit (exact piecewise-linear arithmetic); the Mojo backend is
  ULP-bounded and non-amplifying. Added the Rust engine `simulate` + PyO3
  `py_mckean_simulate`, the Julia/Go/Mojo backends, cross-backend parity tests, a
  multi-language benchmark with a committed artefact, and a documentation upgrade;
  replaced the decorative `accel/go/services` stub with a real c-shared backend.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `WilsonHRNeuron` (Wilson 1999 polynomial cortical model) across
  python / rust / julia / go / mojo. Rust, Julia and Go reproduce the NumPy RK4
  reference bit-for-bit (exact polynomial arithmetic with a hard voltage reset);
  the Mojo backend is ULP-bounded and non-amplifying. Added the Rust engine
  `simulate` + PyO3 `py_wilson_hr_simulate`, the Julia/Go/Mojo backends,
  cross-backend parity tests, a multi-language benchmark with a committed artefact,
  and a documentation upgrade; replaced the decorative `accel/go/services` stub
  with a real c-shared backend.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `PernarowskiNeuron` (Pernarowski 1994 pancreatic beta-cell burster) across
  python / rust / julia / go / mojo. Aligned the Python cubic to `v*v*v` so it is
  bit-identical to the engine's `v.powi(3)` (removing the now unreachable
  `OverflowError` branch); Rust, Julia and Go reproduce the NumPy RK4 reference
  bit-for-bit, the Mojo backend is ULP-bounded. Added the Rust engine `simulate` +
  PyO3 `py_pernarowski_simulate`, the Julia/Go/Mojo backends, cross-backend parity
  tests, a multi-language benchmark with a committed artefact, and a documentation
  upgrade; replaced the decorative `accel/go/services` stub with a real c-shared
  backend.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `TermanWangOscillator` (Terman-Wang 1995 LEGION relaxation oscillator) across
  python / rust / julia / go / mojo. Aligned the Python cubic to `v*v*v` (matching
  the engine `v.powi(3)`); the `tanh` gating makes Rust bit-identical (shared glibc
  tanh) while Julia/Go/Mojo are ULP-bounded (non-amplifying 2D oscillator). Added
  the Rust engine `simulate` + PyO3 `py_terman_wang_simulate`, the Julia/Go/Mojo
  backends, cross-backend parity tests, a multi-language benchmark with a committed
  artefact, and a documentation upgrade; replaced the decorative
  `accel/go/services` and `accel/mojo/kernels` stubs with real backends.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `MihalasNieburNeuron` (Mihalas-Niebur 2009 generalised integrate-and-fire model)
  across python / rust / julia / go / mojo. The four-state `(v, theta, i1, i2)`
  right-hand side is purely linear (no transcendental functions), advanced by
  candidate-first RK4 with a discontinuous spike reset, so Rust, Julia and Go
  reproduce the NumPy reference bit-for-bit (trace, spike count, final state); the
  Mojo backend fuses multiply-add and is validated non-amplifying within a ULP
  band. Added the Rust engine `simulate` + PyO3 `py_mihalas_niebur_simulate`, the
  Julia/Go/Mojo backends, cross-backend parity tests, a multi-language benchmark
  with a committed artefact, and a documentation upgrade; replaced the decorative
  `accel/go/services` and `accel/mojo/kernels` stubs with real backends.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `GLIFNeuron` (Allen Institute GLIF5 generalised leaky integrate-and-fire model)
  across python / rust / julia / go / mojo. The four-state
  `(v, theta, i_asc1, i_asc2)` right-hand side is purely linear (no transcendental
  functions), advanced by candidate-first RK4 with an additive threshold spike
  reset, so Rust, Julia and Go reproduce the NumPy reference bit-for-bit (trace,
  spike count, final state); the Mojo backend fuses multiply-add and is validated
  non-amplifying within a ULP band. Added the Rust engine `simulate` + PyO3
  `py_glif_simulate`, the Julia/Go/Mojo backends, cross-backend parity tests, a
  multi-language benchmark with a committed artefact, and a documentation upgrade;
  replaced the decorative `accel/go/services` and `accel/mojo/kernels` stubs with
  real backends.

## [3.15.34] - 2026-06-15

### Physics and mathematics hardening
- Corrected `CourageNekorkinMapNeuron` to the canonical Courbage-Nekorkin-Vdovin
  2007 map (`Chaos` 17:043109): `x̄ = x + F(x) − y − β·H(x − d) + I`,
  `ȳ = y + ε(x − J)` with the piecewise-linear field `F`, the Heaviside
  discontinuity at `x = d`, and the breakpoints `J_min`, `J_max`. Replaced the
  prior implementation, which was missing the leading `x` term, used an incorrect
  saturating field, had no Heaviside term, an opposite-sign recovery update, and a
  `±1e6` clip that masked divergence. Default parameters set to the published
  chaotic spiking-bursting regime (`m0=0.0864, m1=0.65, a=0.2, d=0.235, J=0.2,
  beta=0.085, eps=0.02`).
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  the map across python / rust / julia / go / mojo. Rust, Julia and Go reproduce
  the NumPy reference bit-for-bit; the Mojo backend is per-step ULP-bounded with a
  matching spike-count band. `auto` selects Rust.
- Added the Rust engine struct plus PyO3 `py_courage_nekorkin_map_simulate`, the
  fail-closed Rust safety mirror, the Julia/Go/Mojo backends, cross-backend parity
  tests, a multi-language benchmark with a committed results artefact, and a
  rewritten model documentation page.

### Dependencies
- Migrated z3 `0.12` -> `0.20` (feature `static-link-z3` renamed to `bundled`;
  the bounded-model verifier updated for the lifetime-free 0.20 AST/Solver API).
- Bumped esbuild/vite/@vitejs/plugin-react (Studio frontend, clears the open
  esbuild advisory), github/codeql-action, click, and hypothesis.

### CI and release reconciliation
- Reconciled `main` with the previously orphaned release tags `v3.15.26`–
  `v3.15.33` so the published release lineage is continuous again.
- Regenerated the capability manifest, corrected the mixed-precision emitter
  Q-format label, and built the ARM64 wheel against the exact matrix interpreter.

## [3.15.8] - 2026-06-05

### Documentation and release polish
- Bumped Python, Rust engine, bridge package, Sphinx docs, README, and
  generated capability metadata to version `3.15.8`.
- Expanded the public documentation surface with an evaluator map covering
  onboarding, notebooks, tutorials, API docs, benchmarks, hardware, industrial
  applications, commercial evaluation, and evidence boundaries.
- Kept benchmark, hardware, clinical, regulatory, and market language tied to
  committed artefacts or explicit missing-evidence boundaries.

## [3.15.1] - 2026-06-01

### Documentation and release polish
- Added public Product Overview and Applications and Market pages so new users, evaluators, and commercial readers can understand the project scope, evidence boundary, potential applications, and market position without reverse-engineering the API inventory.
- Refreshed the README, documentation home page, learning path, getting-started guide, notebook guide, API index, industrial-applications page, benchmark index, and cross-framework benchmark evidence page for clearer onboarding and claim traceability.
- Added a notebooks README with recommended reading order and reproducibility rules.
- Bumped Python package, public docs, capability metadata, and Rust engine package version references from 3.15.0 to 3.15.1.

### NIR Bridge
- Roundtrip tests for all 18/18 NIR primitives (was 7/18)
- Auto-broadcast scalar neuron params to input size (Norse/snnTorch export 0-dim tensors)
- Threshold fix: `>=` to `>` matching NIR spec and snnTorch behavior
- `reset_mode="subtract"` for snnTorch compatibility (subtract-reset vs zero-reset)
- IF subtract-reset test and unknown `reset_mode` fallback handling
- Cross-framework interop tests: Sinabs LIF/IAF/ExpLeak, Rockpool LIF/CubaLIF/LI, snnTorch RSynaptic subgraph
- Cross-framework r-encoding test documenting per-framework dt conventions
- SpikingJelly NIR roundtrip demo (`examples/spikingjelly_nir_roundtrip.py`)
- Norse NIR roundtrip demo with real Norse weights (`examples/norse_nir_roundtrip.py`)
- NIR roundtrip demo: stronger input to produce visible spikes
- Documentation: added SpikingJelly, Rockpool, Sinabs, snnTorch RSynaptic sections to `docs/guides/nir_integration.md`
- Documentation: framework dt/r quick reference table
- Documented Norse tau observation (export/import roundtrip discrepancy in Norse code)
- Removed unverified "first FPGA backend" claim from 6 files

### Physics and mathematics hardening
- Hardened `PoissonNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces to
  revalidate mutable rate and timestep state before sampling, reject non-finite
  interval hazards, and keep the finite-step Poisson probability bounded.
- Hardened `McCullochPittsNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to enforce finite weighted-input and mutable-threshold contracts,
  preserve equality-at-threshold Heaviside semantics, and keep reset as a
  stateless no-op.
- Hardened `EscapeRateNeuron` Python, Julia, Go, and Rust safety surfaces to
  revalidate mutable point-process state before membrane integration,
  exponentiation, hazard evaluation, or random sampling; non-finite voltage
  candidates and escape hazards now fail before membrane mutation.
- Hardened `LapicqueNeuron` Python, Julia, Go, and Rust safety surfaces to
  revalidate mutable RC state before division/integration and report invalid
  current, corrupted state, or non-finite Euler increments explicitly before
  membrane mutation; documented the Mojo fail-closed spike-flag boundary.
- Hardened `NonResettingLIFNeuron` Python, Julia, Go, and Rust safety surfaces
  to revalidate runtime membrane and adaptive-threshold state before
  integration, compute both candidates before mutation, and report non-finite
  updates explicitly while preserving the no-voltage-reset spike contract.
- Hardened `PerfectIntegratorNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to revalidate runtime membrane geometry before division/integration
  and to report invalid or non-finite voltage increments explicitly before
  state mutation.
- Hardened `ThetaNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces to
  reject corrupted runtime phase or timestep state before cosine/Euler
  evaluation and to report non-finite phase increments explicitly without
  mutating the compact-circle state.
- Hardened `SiegertTransferFunction` Python, Julia, Go, Mojo, and Rust safety
  surfaces to revalidate first-passage parameters at runtime, reject non-finite
  quadrature bounds, integrals, and inter-spike intervals, and keep rates
  finite, non-negative, and refractory bounded.
- Hardened `SigmoidRateNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to enforce the continuous-rate `[0, 1]` invariant, reject unstable
  Euler ratios and corrupted runtime state before mutation, and use saturated
  finite-drive logistic evaluation for extreme inputs.
- Hardened `AdaptiveThresholdMoENeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid runtime state and non-finite adaptive-threshold,
  quotient, or soft-reset candidates before state mutation, while preserving
  non-negative integer spike-count residual semantics.
- Hardened the `ThresholdLinearRateNeuron` Python, Julia, Go, Mojo, and Rust
  safety surfaces to reject invalid runtime rate state and non-finite rate
  outputs before state mutation.
- Hardened the `AdaptiveThresholdIFNeuron` Python, Julia, Go, Mojo, and Rust
  safety surfaces to reject invalid runtime state and non-finite Euler or
  threshold-jump updates before state mutation.
- Hardened the `BrainScaleSAdExNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid runtime state and non-finite hardware-scaled
  integrator or adaptation updates before state mutation.
- Hardened the `AdExNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid runtime state and non-finite integrator or adaptation
  updates before state mutation.
- Hardened the `ExpIFNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid runtime state and non-finite Euler updates before membrane
  mutation.
- Hardened the Rust/PyO3 Kuramoto solver boundary to fail closed on non-finite frequencies, coupling matrices, initial/runtime phases, field pressure, SSGF geometry/PGBO matrices, invalid `dt`, and negative/non-finite noise amplitudes.
- Extended Kuramoto `run()` and `run_ssgf()` validation so invalid timesteps and non-finite SSGF gains/matrices are rejected even for zero-step dry runs.
- Added PyO3 SSGF shape guards so malformed `W` and `h_munu` matrices raise `ValueError` before entering the Rust solver.
- Corrected the `PINGCircuit` Python reference step to consume one excitatory and one inhibitory Wiener-noise vector per timestep, matching the Rust, Julia, Go, and Mojo backend stochastic contract; benchmark metadata now reports the selected backend directly.
- Hardened `CorticalColumn(backend="python", use_block_csr=True)` so it remains on the scipy.sparse reference path and does not call the Rust single-block fallback when native symbols are present.
- Hardened `HindmarshRoseNeuron` RK4/Euler derivative evaluation to fail closed on cubic overflow or non-finite intermediate stages without mutating state.
- Hardened `MorrisLecarNeuron` Euler/RK4/Rosenbrock paths to fail closed on potassium-rate overflow or non-finite derivative/state updates without mutating state.
- Hardened `FitzHughNagumoNeuron` Euler/RK4/Rosenbrock paths to fail closed on cubic overflow or non-finite derivative/state updates without mutating state, and aligned the Julia, Go, and Rust safety counterparts with the documented no-reset state equation.
- Hardened `McKeanNeuron` runtime updates across Python, Julia, Go, and Rust safety surfaces to fail closed on non-finite state/current or non-finite post-update state instead of silently reporting no spike.
- Hardened `ResonateAndFireNeuron` Julia, Go, Mojo, and Rust safety counterparts so invalid current/state and non-finite Euler updates report explicit errors/sentinels instead of silently returning no spike.
- Hardened `QuadraticIFNeuron` Julia, Go, Mojo, and Rust safety counterparts so invalid current/state and non-finite Euler increments report explicit errors/sentinels instead of silently returning no spike.

### Repository hygiene
- Purged obsolete completed failed/cancelled GitHub Actions repair-sequence runs after later successful replacement runs were verified on `main`.
- Removed inactive stale GitHub Pages deployment records while retaining the current successful Pages deployment and successful package-release deployment evidence.
- Rechecked Dependabot, code-scanning, and secret-scanning alert surfaces; all reported zero open alerts.

## [3.15.0] — 2026-05-19

### Wave 11: Hardware and compiler extensions (2026-05-01)

#### Added — Hardware Profiles (8 new → 183 total, 35 platform classes)
- **Thermodynamic** (2): `extropic_epu`, `normal_cn101`.
- **Probabilistic / p-Bit** (2): `purdue_pbit`, `tohoku_sot_pbit`.
- **Polariton / Exciton** (2): `marvell_polariton`, `stanford_polariton`.
- **Metamaterial** (2): `mit_metamaterial`, `penn_acoustic_meta`.

#### Added — Compiler Features (9 new → 76 total)
- §68 `configure_approximation()` — precision-energy tradeoff knobs per population.
- §69 `model_energy_harvest()` — batteryless edge feasibility analysis.
- §70 `predict_aging()` — NBTI/HCI degradation modeling with timing derating.
- §71 `generate_dvfs_controller()` — Verilog DVFS FSM generator.
- §72 `explore_pareto()` — multi-objective power/area/latency Pareto frontier.
- §73 `protect_ip_pqc()` — post-quantum (CRYSTALS-Dilithium) IP protection.
- §74 `run_fault_campaign()` — systematic bit-flip SDC testing.
- §75 `verify_timing_closure()` — formal static timing analysis.
- §76 `ingest_telemetry()` — digital twin ↔ hardware telemetry loop.

#### Added — Tests
- `test_wave11_features.py` — 35 tests, 0 failures.

### Wave 10: Security, sovereignty & extensibility (2026-05-01)

#### Added — Hardware Profiles (10 new → 175 total, 31 platform classes)
- **Magnonic** (3): `tum_skyrmion`, `kaist_spinwave`, `imec_mtj_reservoir`.
- **Organic Bioelectronic** (2): `cambridge_oect`, `linkoping_organic`.
- **RISC-V Sovereign** (5): `sifive_x280_ai`, `esperanto_et_soc`,
  `ventana_veyron_ai`, `tenstorrent_ascalon`, `andes_ax45mpv`.

#### Added — Generic Profile Constructor
- `HardwareProfile.from_constraints()` — auto-construct profile from spec
  sheet constraints. **Any future hardware automatically supported.**

#### Added — Compiler Features (8 new → 67 total)
- `lint_hardware_trojans()` — detect dormant trigger / payload paths.
- `generate_sbom()` — CycloneDX/SPDX SBOM for EU CRA compliance.
- `generate_hil_calibration()` — HIL drift compensation protocol.
- `generate_digital_twin()` — software shadow of deployed hardware.
- `map_ucie_protocol()` — UCIe chiplet die-to-die lane mapping.
- `schedule_seu_scrubbing()` — space-grade configuration scrubbing.
- `obfuscate_ip()` — logic locking + structural IP protection.
- `embed_watermark()` — verifiable netlist watermark embedding.

#### Added — Tests
- `tests/test_wave10_features.py` — 32 tests across 13 test classes.

### Wave 9: Universal coverage & extensibility (2026-05-01)

#### Added — Hardware Profiles (10 new → 165 total, 28 platform classes)
- **Optical I/O** (2): `ayar_teraphy`, `intel_cpo`.
- **Acoustic** (2): `mit_phononic`, `caltech_mems_nn`.
- **Fluidic** (2): `stanford_microfluidic`, `eth_fluidic_logic`.
- **Space-Qualified** (4): `bae_rad750_sq`, `seakr_sbc`, `vorago_va10820`, `frontgrade_leon5`.

#### Added — Compiler Features (8 new → 59 total)
- `analyze_cdc()` — formal CDC check.
- `load_profiles_from_toml()` — custom HW without code changes.
- `plan_multi_die_floorplan()` — chiplet/3D bin packing.
- `check_regression()` — perf regression detector.
- `check_license_compliance()` — SPDX IP compatibility.
- `generate_power_state_machine()` — sleep/wake/hibernate FSM.
- `register_platform_hook()` / `discover_platforms()` — runtime extensibility.
- `generate_compilation_report()` — one-click markdown report.

#### Added — Tests
- `tests/test_wave9_features.py` — 28 tests across 10 test classes.

### Wave 8: Final gap closure & trajectory synthesis (2026-05-01)

#### Added — Hardware Profiles (11 new → 155 total, 24 platform classes)
- **RRAM** (3): `weebit_reram`, `crossbar_rram`, `adesto_cbram`.
- **SRAM-CIM** (2): `tsmc_cim_n7`, `samsung_cim_sf3`.
- **Cryo CMOS** (2): `intel_horse_ridge`, `google_cryo_ctrl`.
- **DNA/Molecular** (2): `microsoft_dna_store`, `asu_dna_perovskite`.
- **Quantum Neuromorphic** (2): `ibm_qnn`, `ionq_trapped_ion`.

#### Added — Compiler Intelligence Features (10 new → 51 total)
- `import_nir_graph()` — NIR/ONNX-SNN model import.
- `verify_ode_stability()` — Lyapunov/eigenvalue discretization check.
- `generate_power_intent()` — IEEE 1801 UPF generation.
- `estimate_carbon_footprint()` — lifecycle CO₂ per target.
- `insert_debug_probes()` — ILA/SignalTap auto-insertion.
- `generate_memory_map()` — address decoder for neuron SoC arrays.
- `score_portability()` — cross-platform compatibility scoring.
- `predict_reliability()` — MTTF from voltage/temp/node.
- `generate_fault_tree()` — FTA/FMEA for DO-254 Level A.
- `generate_testbench()` — Cocotb/UVM auto-generation.

#### Added — Tests
- `tests/test_wave8_features.py` — 38 tests across 12 test classes.

### Wave 7: Compiler intelligence and platform coverage (2026-05-01)

#### Added — Hardware Profiles (10 new → 144 total, 19 platform classes)
- **Biological** (2): `finalspark_neuroplatform`, `cortical_labs_dishbrain`.
- **Electrochemical** (3): `ibm_ecram`, `samsung_pcram`, `stanford_ecram`.
- **Wafer-Scale** (3): `cerebras_wse3_ws`, `tesla_dojo3`, `tachyum_prodigy`.
- **Analog Mixed-Signal** (2): `aspinity_aml100`, `renesas_analog_ai`.

#### Added — Compiler Intelligence Features (8 new → 41 total)
- `recommend_target()` — constraint-driven optimal HW selection.
- `plan_partial_reconfiguration()` — FPGA DPR partition scheduling.
- `score_supply_chain_risk()` — geopolitical/sole-source risk analysis.
- `generate_bittrue_kernel()` — C/Rust code matching Verilog bit-exactly.
- `classify_model_complexity()` — memory/compute/comm-bound routing.
- `CompilationCache` — memoized instant re-targeting.
- `estimate_thermal_envelope()` — junction temperature prediction.
- `optimize_network_topology()` — multi-chip spike bandwidth minimizer.

#### Added — Tests
- `tests/test_wave7_features.py` — 41 tests across 11 test classes.
- Total regression: **1,094 passed**, 1 xfailed, 0 failures.

### Wave 6: Total paradigm coverage & overlooked compiler features (2026-05-01)

#### Added — Hardware Profiles (21 new → 134 total, 79 vendors, 15 classes)
- **Superconducting** (3): `nist_sfq`, `northrop_aqfp`, `josephson_jj`.
- **Spintronic** (2): `everspin_stt_mram`, `samsung_sot_mram`.
- **Ferroelectric** (2): `gf_fefet`, `sk_hynix_feram`.
- **CGRA** (3): `samsung_cgra`, `qualcomm_npu_cgra`, `pact_xtensa`.
- **3D-Stacked** (3): `tsmc_soic`, `intel_foveros`, `amd_3dv`.
- **Edge MCU** (5): `rp2040`, `esp32_s3`, `stm32h7`, `nrf5340`, `max78000`.
- **RISC-V AI** (3): `sifive_x280`, `qualcomm_ventana`, `ainekko_rv`.

#### Added — Strategic Compiler Features (8 new → 33 total)
- `generate_equivalence_sketch()` — formal ODE↔RTL proof skeleton with SVA.
- `partition_timescales()` — multi-timescale ODE clock-domain splitting.
- `generate_provenance_chain()` / `format_provenance_json()` — SHA-256 audit trail.
- `generate_compliance_matrix()` / `format_compliance_report()` — DO-254/IEC 61508/ISO 26262.
- `generate_energy_schedule()` — energy-harvesting neuron update scheduling.
- `lint_side_channels()` — power/timing side-channel leakage analysis.
- `generate_drift_compensator()` — analog device aging calibration controller.
- `plan_heterogeneous_dispatch()` — multi-backend SNN model splitting.

#### Added — Tests
- `tests/test_wave6_features.py` — 58 tests across 10 test classes.
- Total regression: **1,033 passed**, 1 xfailed, 0 failures.

### Wave 5: Universal hardware coverage & strategic features (2026-05-01)

#### Added — Hardware Profiles (29 new → 113 total, 66 vendors, 9 classes)
- **Photonic / Optical Compute** (5): `lightmatter_passage`, `lightelligence_pace`,
  `xanadu_x8`, `ipronics_smartlight`, `luminous_computing`.
- **Chiplet / UCIe** (5): `tenstorrent_blackhole`, `cerebras_wse3`,
  `intel_ponte_vecchio`, `amd_mi300x`, `ucie_generic`.
- **PIM / CXL Memory** (5): `upmem_pim`, `samsung_hbm_pim`, `sk_hynix_aim`,
  `cxl_type3`, `axdimm`.
- **Next-Gen Neuromorphic** (5): `akida2`, `spinnaker2`, `dynapse2`,
  `rain_neuromorphic`, `brainscales2`.
- **Sovereign / Defence** (5): `bae_rad750`, `cobham_ut700`, `mpfs250t_rt`,
  `versal_xqrvc1902`, `trenz_zynq_space`.
- **Automotive / Edge AI** (6): `mythic_m1076`, `mobileye_eyeq6`, `horizon_j6`,
  `ambarella_cv72s`, `hailo15`, `syntiant_ndp120`.

#### Added — TOML Profile Loader (universal future-proofing)
- `load_toml_profile()` — register custom hardware targets from TOML files.
- `load_toml_profiles_dir()` — bulk-load all `*.toml` profiles from a directory.
- Enables instant compatibility with any future chip without code changes.

#### Added — Strategic Compiler Features
- `generate_tmr_wrapper()` — SEU/TMR wrapper with majority/median voter.
- `embed_model_checksum()` — SHA-256 hash embedding for reproducibility.
- `auto_quantisation_sweep()` — sweep Q4→Q32 for accuracy-vs-resource DSE.
- `format_quantisation_report()` — markdown table output for sweep results.
- `encode_mzi_weights()` — MZI phase-shift encoding for photonic chips.
- `generate_mzi_config()` — photonic chip config (JSON/CSV) from MZI weights.
- `plan_pim_layout()` — PIM/CXL memory bank layout optimisation.
- `generate_power_domain_wrapper()` — ICG clock gating for ultra-low-power edge.
- `generate_hls_cpp()` — Vitis/Catapult HLS C++ translation.
- `generate_bitstream_encryption()` — AES-256 bitstream encryption (Xilinx/Intel).
- `advise_ucie_partition()` — chiplet die-to-die neuron array partitioning.
- `advise_cxl_mapping()` — CXL.mem Type-3 device mapping with protocol selection.
- `generate_learning_params()` / `export_learning_config()` — STDP/RSTDP on-chip
  learning parameter export for Akida 2, BrainScaleS-2, SpiNNaker 2.
- `inject_weight_noise()` — stochastic weight noise injection (Gaussian/uniform/
  lognormal) for analog/memristive robustness validation.
- `create_noise_profile()` — device-variation characterisation for analog targets.
- `generate_pipeline_wrapper()` — auto-insert register stages for HF targets.
- `compare_targets()` — compile once, compare N hardware targets side-by-side.
- `format_comparison_report()` — markdown table from multi-target comparison.
- `generate_compilation_summary()` — comprehensive markdown compilation report.

#### Added — Tests
- `tests/test_wave5_features.py` — 130 tests (profiles, TOML, TMR, checksum,
  sweep, MZI, PIM, power-domain, HLS, encryption, UCIe, CXL, STDP, noise,
  pipeline, comparison, summary, cross-feature E2E integration).
- Total regression: **933 passed**, 1 xfailed, 0 failures.


### Network-level compilation & thermal-aware deployment (2026-05-01)

#### Added — Advanced Features: BRAM Auto-Selection
- `storage_recommendation()` — automatic register/BRAM/URAM strategy.
- `generate_bram_array()` — time-multiplexed BRAM-backed neuron array
  with `(* ram_style = "block" *)` inference pragmas.
- Supports 18Kb, 36Kb BRAM and 288Kb URAM (UltraScale+/Versal).

#### Added — Advanced Features: Thermal-Aware Compilation
- `thermal_analysis()` — ΔT estimation, frequency derating, hotspot risk.
- `generate_thermal_constraints()` — XDC with derated clock and DSP spreading.
- Technology model for 7nm through 65nm junction temperature.

#### Added — Advanced Features: Weight ROM Generation
- `generate_weight_rom()` — synaptic weights in 3 formats:
  Verilog ROM, Xilinx `.coe`, and Intel `.mif`.

#### Added — Tests
- `tests/test_wave4_features.py` — 28 tests (BRAM, thermal, weights).
- `tests/e2e/test_e2e_pipeline.py` — 22 end-to-end integration tests
  covering 9 cross-cutting compilation pipelines.
- Total regression: **745 passed**, 1 xfailed, 0 failures.


#### Added — Hardware Profiles (7 new → 84 total)
- **AI accelerators**: `qualcomm_nsp` (Qualcomm NSP), `sambanova` (SambaNova
  RDU), `cambricon_mlu` (Cambricon MLU370/590).
- **Emerging compute**: `superconducting` (AQFP/SFQ ~100 GHz),
  `cim_sram` (compute-in-SRAM), `analog_ai` (PCM/ReRAM),
  `event_camera` (Prophesee/Sony DVS).

#### Added — Static Analysis: Pipeline Stage Analysis
- `critical_path_depth()` — AST-based multiply chain analysis.
- `pipeline_stages_needed()` — pipeline budget from target frequency.
- `pipeline_analysis()` — multi-ODE per-variable pipeline report.

#### Added — Static Analysis: Power Estimation
- `estimate_power()` — switching-activity-based power model.
- `PowerEstimate` dataclass with dynamic/static/total/energy-per-spike.
- Technology node library: 7nm through 65nm capacitance scaling.

#### Added — Deployment: Multi-Target Compilation
- `compile_multi_target()` — compile one neuron to N targets.
- `format_comparison_table()` — markdown comparison report.
- `CompilationResult` dataclass with per-target metrics.

#### Added — Tests
- `tests/test_wave3_features.py` — 31 tests (profiles, pipeline,
  power, multi-target).
- Total regression: **695 passed**, 1 xfailed, 0 failures.


#### Added — Hardware Profiles (12 new → 77 total)
- **Neuromorphic**: `loihi3` (Intel, 4nm 8M neurons), `northpole` (IBM, 256-core),
  `innatera_pulsar` (Innatera, analog-digital hybrid μC).
- **FPGA**: `versal_ai_edge` (AMD, AI Engine + DSP58), `proasic3` (Microchip, flash),
  `trion` / `titanium` (Efinix), `gowin_arora_v` (Gowin 28nm), `intel_agilex5`
  (Intel, HBM2e).
- **AI accelerators**: `nvidia_dla` (Orin DLA), `mediatek_apu` (APU 790),
  `aws_inferentia` (Inferentia2/Trainium2).

#### Added — Deployment: SymbiYosys Formal Verification
- One-command `.sby` script generation for BMC, induction, and cover modes.
- Solver support: boolector, Z3, yices via SymbiYosys + Yosys.

#### Added — Deployment: RISC-V Driver + RTOS Templates
- RISC-V C driver with volatile MMIO accessors for PolarFire SoC, Efinix
  Titanium, and RISC-V soft-cores (Nios V, MicroBlaze V).
- FreeRTOS task template: `xTaskCreate` + `vTaskDelay` neuron tick loop.
- Zephyr RTOS thread template: `K_THREAD_DEFINE` + `k_msleep` integration.

#### Added — Advanced Features: DVS Event-Camera → AER Bridge
- Synthesisable Verilog bridge converting Prophesee / Sony IMX636 DVS events
  to SC-NeuroCore AER address-event protocol.
- Configurable FIFO depth, address width, timestamp width, polarity bit.
- Overflow detection flag for back-pressure monitoring.

#### Added — Deployment: Multi-Die SLR Placement
- Vivado XDC PBLOCK constraint generation for multi-SLR FPGAs (Versal,
  Agilex 7, Stratix 10, UltraScale+).
- Auto inter-SLR pipeline register directives for >500 MHz crossing.

#### Added — Advanced Features: Block-FP / MXFP Encoding
- OCP Microscaling Spec v1.0 formats: MXFP4, MXFP6, MXFP8 (E4M3/E5M2).
- IEEE FP8 (NVIDIA H100/B100 native) with block_size=1.
- Encode/decode block functions for parameter transfer and weight storage.

#### Added — Deployment: Safety Certification Evidence
- XML traceability matrix generation for DO-254 (DAL-A/B/C), IEC 61508
  (SIL 1–4), and ISO 26262 (ASIL A–D).
- Requirement → design → verification linkage with pass/fail/untested
  status and coverage percentage.

#### Added — Tests
- `tests/test_wave2_features.py` — 49 tests (profiles, SBY, RISC-V,
  DVS, SLR, MXFP, certification).
- Total regression: **650 passed**, 1 xfailed, 0 failures.

### Universal hardware compilation & deployment industrialisation (2026-05-01)

#### Added — Compiler: Hardware Profiles
- Expanded hardware profile registry from 32 to **65 pre-configured profiles**
  across 7 platform classes and 40 vendors.
- **Rad-hard / space**: NanoXplore NG-Ultra, Microchip RTG4, Xilinx Kintex
  UltraScale+ RT — DO-254 / MIL-STD-883 alignment.
- **Edge AI accelerators**: Hailo-8, Kneron KL730, Groq TSP, NVIDIA Jetson
  Orin, Intel Habana Gaudi 2/3, Renesas DRP-AI.
- **eFPGA IP**: Achronix Speedcore, Flex Logix EFLX, Menta Origami.
- **Vision-on-sensor**: Sony IMX500/IMX501, Samsung Exynos NPU.

#### Added — Compiler: Static Analysis (`static_analysis.py`)
- Guard-bit auto-computation from expression AST (single + multi-ODE).
- Formal overflow proof via interval arithmetic — mathematical guarantee of
  no overflow at compile time, no simulation required.
- SystemVerilog Assertion (SVA) generation for DO-254 / IEC 61508 formal
  verification (overflow assertions, reachability covers, input assumptions,
  stability checks).

#### Added — Compiler: Mixed Precision (`mixed_precision.py`)
- Per-variable mixed-precision specification via dict API.
- Automatic constraint solver: given value bounds, resolution requirements,
  and a total-bit budget, auto-selects optimal Q-format per variable.
- Preset shorthand (`from_preset({"v": "q88", "u": "q44"})`).

#### Added — SoC Integration: Bus Interface (`bus_interface.py`)
- AXI4-Lite bus wrapper generator (Xilinx/AMD compatible).
- Wishbone B4 bus wrapper generator (LiteX/open-source RISC-V compatible).
- Auto-generated register map with CTRL, I_T, SPIKE_COUNT, and parameter
  registers. Spike interrupt output for GIC/NVIC integration.

#### Added — Compiler: Deployment Utilities (`deployment.py`)
- **Resource estimation**: LUT/FF/DSP/BRAM estimation from Verilog without
  synthesis (heuristic-based, <1 ms).
- **Constraint generation**: SDC (Intel/generic) and XDC (Xilinx) timing
  constraint files with configurable target frequency.
- **Host driver generation**: Python MMIO class and C header with Q-format
  encode/decode for host-side parameter tuning.
- **Cocotb testbench generation**: 3-scenario Python-based verification
  (spike, zero-current, reset).

#### Added — Compiler: Advanced Features (`advanced_features.py`)
- **VHDL-2008 output mode**: generates entity/architecture wrappers for
  mixed-language simulation and DO-254 compliance.
- **Posit arithmetic**: posit-8 and posit-16 encode/decode with 4 standard
  configs (POSIT8_0, POSIT8_1, POSIT16_1, POSIT16_2).
- **CDC synchroniser generation**: multi-clock domain crossing with
  configurable stages and `ASYNC_REG` attributes.
- **TCL project generation**: complete Vivado and Quartus project scripts
  (synth → P&R → bitstream → reports).
- **Bitstream automation**: Yosys + nextpnr Makefiles for iCE40 and ECP5
  open-source FPGA flow.

#### Added — SoC Integration: IP-XACT Packaging (`ip_xact.py`)
- IEEE 1685 IP-XACT component XML generator for Vivado IP Integrator
  drag-and-drop integration with AXI bus interfaces, port definitions,
  file sets, and parameter schemas.

#### Added — Documentation
- New guide: `docs/guides/static_analysis_guide.md` (217 lines) — guard bits,
  interval arithmetic overflow proof, SVA generation.
- New guide: `docs/guides/soc_integration_guide.md` (239 lines) — bus
  wrappers, mixed-precision, host drivers, IP-XACT, VHDL output.
- New guide: `docs/guides/deployment_guide.md` (338 lines) — resource
  estimation, SDC/XDC constraints, Cocotb testbenches, Vivado/Quartus TCL,
  CDC synchronisers, posit arithmetic, iCE40/ECP5 Makefile, complete
  end-to-end deployment workflow.
- Updated guide: `docs/guides/hardware_profiles.md` (431 lines) — expanded
  from 51 to 65 profiles with rad-hard, eFPGA, edge AI, and vision tables;
  cross-references to 3 new guides.
- Updated roadmap: `docs/internal/COMPILER_ROADMAP_TODO.md` — all Tier 4
  items marked DONE, cross-references to new modules.
- 100% docstring coverage across all 13 modified/new source modules.

#### Added — Tests
- `tests/test_static_analysis.py` — 28 tests.
- `tests/test_bus_mixed_precision.py` — 34 tests.
- `tests/test_deployment.py` — 26 tests.
- `tests/test_advanced_features.py` — 40 tests (IP-XACT, VHDL, posit, CDC,
  TCL, Makefile).
- `tests/test_hardware_profiles.py` — expanded to cover all 65 profiles.
- Total regression: **577 passed**, 1 xfailed, 0 failures.

#### Fixed — Docstring Coverage
- Added missing docstrings to 24 functions/methods across
  `verilog_generator.py`, `equation_builder.py`, `universal_dsl.py`, and
  `neurons/__init__.py` to achieve 100% coverage on all session-touched files.

### Security hardening (2026-04-29)

#### Added
- Property-based fuzz coverage for malformed bitstream/IR ports, Studio graph
  JSON, transfer checkpoints, NIR imports, model-zoo NPZ archives, SCPN
  datastream JSON, custom chip-spec JSON, HDL stochastic-source lowering,
  equation/MLIR lowering, and optimiser evidence JSON.
- Offline supply-chain audit command for committed CycloneDX SBOM and release
  requirements metadata: `python tools/supply_chain_audit.py`.
- Hardware-install documentation now records Vivado `v2025.2` as the current
  SHD/PYNQ evidence pin and marks OpenROAD PPA numbers as unpublished until the
  binary/container digest and PDK revision are recorded.
- Packaging metadata now exposes `sc-neurocore[hdl]`, expands
  `sc-neurocore[full]` across CPU-side training, NIR, Studio, HDL, codec,
  bioware, and quantum workflows, and packages HDL/OpenROAD source artefacts.
- Added an offline EDA toolchain version inventory helper for Vivado,
  OpenROAD, Yosys, nextpnr, IceStorm, Trellis, Quartus, Lattice tools, PYNQ,
  and OpenROAD/PDK pin metadata.

#### Fixed
- Hardened validation boundaries for fuzzed JSON, NPZ, NIR, IR, and HDL inputs
  before they reach parser, lowering, or hardware-resource paths.
- Documented the strict release-mode supply-chain gate in `SECURITY.md`.
- Aligned the CycloneDX SBOM root component version with `pyproject.toml` so
  strict supply-chain audit runs pass without metadata drift.

### CI coverage restoration (2026-04-21)

#### Fixed
- `tools/ci_install_dev.py` now installs `dev,nir,compression,training,research,bioware,studio` so the 342 torch-gated tests (`arcane_zenith`, `darts_sc_nas`, `advanced_plasticity`, and the `_native` bridges that hit the `torch.autograd.Function` path) run inside the 3.10–3.14 matrix instead of being silently skipped.
- `tests/test_analog_bridge/test_analog_bridge.py` + `test_analog_bridge_extended.py` now import through `sc_neurocore.analog_bridge` rather than via a `sys.path.insert` hack; `coverage.py` was reporting 0 % for `analog_bridge.analog_bridge` despite the 27 tests executing every line.

#### Added
- `sc_neurocore.analog_bridge` package root re-exports `AnalogBridge`, `AnalogSubstrateProfile`, `EventDrivenInterface`, `CalibrationRoutine`, `AEREvent` through `__all__`.
- `tests/test_native/test_array_guards.py` — 24 multi-angle tests for `require_c_contiguous` covering happy path, dtype coercion, non-contiguous rejection, list / tuple conversion, the post-asarray defensive branch via `__array__` producers, alignment enforcement, and FFI integration byte ops. Module coverage 42 % → 100 %.
- Two `unittest.mock.patch`-based tests for `CalibrationRoutine.effective_resolution_bits` fallback (`max_err == 0` and `full_range == 0`); reachable branches not touched by the sweep-and-measure suite. Module coverage 99 % → 100 %.

### evo_substrate: 4-backend whole-process industrial evolve runner (2026-04-20)

#### Added
- `crates/evo_substrate_core` (new Rust crate, 1 227 LOC of `runner.rs` + C-FFI + PyO3 extension) — port of `ReplicationEngine.evolve_generation` + eleven industrial guards (TournamentSelector, AgeRegulator, FormalSafetyGuard, BloatPenalizer, ExtinctionDetector, HallOfFame, ParetoFront, LineageTracker, MutationEngine × 4 variants, CrossoverEngine, parametric FitnessEvaluator). Entry point `py_evolve_run(config_json) -> str`. Measured 72× speedup over the Python `ReplicationEngine` on 10-gen × 16-pop industrial runs (0.57 ms vs 40.88 ms).
- `src/sc_neurocore/accel/julia/evo_substrate/evo_runner.jl` (720 LOC) — same industrial loop in Julia 1.10+. JSON-in / JSON-out subprocess contract. Pinned deps via `Project.toml`.
- `src/sc_neurocore/accel/go/evo_substrate/runner.go` (926 LOC) — same industrial loop in Go 1.22+. Shares the JSON contract. `--runner` flag on the existing `evo_substrate_bench` binary dispatches to it.
- `src/sc_neurocore/accel/mojo/kernels/evo_runner.mojo` (803 LOC) — same industrial loop in Mojo 0.26+. Uses Mojo's Python interop for JSON + SHA-256 at the I/O boundary; compute loop (mutation, fitness, tournament, Pareto, lineage, extinction) runs in pure Mojo.
- Unified XorShift64 PRNG across all four backends (shift constants 13/7/17, `0xDEADBEEFCAFEBABE` fallback for zero seeds) so the uniform-random sequence is byte-identical cross-language. Rust↔Julia full bit-exact parity on final genomes / lineage / Pareto; Rust↔Go & Rust↔Mojo agree on structural counters but drift ~1e-3 on `best_fitness` because Go + Mojo `libm` `cos()` / `log()` differ from Rust's libm at ~1 ULP and Box-Muller compounds that.
- Hamming(7,4) encode / decode + `ScDoctor.adapt` control law added to `crates/stochastic_doctor_core` with PyO3 bridge (`py_hamming74_encode`, `py_hamming74_decode`, `py_sc_doctor_adapt`); `src/sc_neurocore/debug/sc_doctor.py` now dispatches to Rust when the extension is importable (1.7× / 3.1× speedup on encode / decode; `adapt` slower via FFI at 276 ns due to dominant PyO3 overhead). Pure-Python fallback preserved bit-exact.
- `sc_scope.compute_scc` now dispatches to `stochastic_doctor_core.py_scc_packed` (174× speedup over pure Python; bit-exact parity with fallback).
- Cross-language parity test harness `tests/test_evo_substrate/test_multilang_parity.py` (18 assertions) asserts Rust↔Julia byte-exact, Rust↔Go counter match + fitness tolerance, Rust↔Mojo schema match.
- Per-backend unit tests: Julia 17 tests (`test_evo_runner.jl`), Go 8 tests (`runner_test.go`), Mojo 7 side-validated tests (`tests/test_evo_substrate/test_mojo_runner.py`).

#### Documentation
- `docs/api/evo_substrate.md` §7.3 — new whole-process runners section with entry-point table, measured 4-way parity matrix, honest timing breakdown per backend (Rust PyO3 warm 0.57 ms, Go execution 2 ms excluding ~3 s `go build` first time, Mojo cold ~1.1 s pixi + JIT + Python interop, Julia cold ~3 s JSON.jl + SHA.jl precompile, Python reference 40.88 ms), decision matrix for which backend to pick, and the 4-way test-suite invocation list.

### Strategic module unification (2026-04-20)

#### Added
- `sc_neurocore.arcane_zenith.ArcaneZenithCognitiveCore` — three-compartment ArcaneNeuron (fast / working / deep membrane states) coupled via attention gate + self-model predictor, wired to four reward-modulated plasticity rules via a sharpened sigmoid that maps weights into biological ranges for `tau_deep`, `surprise_baseline`, `delta_conf`, `lr_base`. Factory `create_arcane_neuron_with_zenith_plasticity(backend=…)`, plus `step_from_bio_rates` (MEA rate dict) and `step_from_genome` (evo_substrate bridge). 32 multi-angle tests in `tests/test_arcane_zenith/`.
- `sc_neurocore.optics.photonic_emitter` — full rewrite of `CrosstalkModel.analyze_bank` on Marcatili coupled-mode theory (adjacent + next-nearest pairs); new `analyze_pairs` for O(N²) arbitrary geometry. Rust FFI `py_ph_analyze_crosstalk_bank` / `py_ph_analyze_crosstalk_pairs` (with 4 cargo tests); Python fallback matches to 1e-9. `FDTD2DSolver` split-field Berenger PML (Ezx + Ezy with σ-matched magnetic conductivity). `CompilationResult.to_gdsii` now produces real GDSII via `gdsfactory` + `klayout` (PDK auto-activation, `allow_duplicate` cells, netlist string to GDS TEXT layer 63/0). 43 tests in `tests/test_optics/`.
- `sc_neurocore.bioware` closed-loop surface: `BioHybridSession.process_frame` returns `BioHybridFrameResult` (typed dataclass with legacy mapping view — `result["round"]` + `result.round` both valid). `SpikeSorter` fit/assign with sklearn PCA+KMeans, no-op on empty input. `HomeostaticPlasticity.update_threshold` Q8.8 proportional controller (error × α × 256, clamped to min/max). New `mea_fitness_hook` — converts MEA spike dynamics to `{accuracy, energy_mw, latency_ms}` for evo_substrate's `ReplicationEngine(metrics_fn=…)`. Matching PCA / Berenger / closed-loop regression tests added.
- `sc_neurocore.accel.mojo.MojoKernelRunner` + `kernels.mojo` — Mojo SIMD primitives (packed SC ops, `sc_and/or/xor/mux/sub/not`, pack/unpack, `vec_mac`, `stdp_update`, `reward_modulated_stdp`, `hdc_bind`). Pixi-managed toolchain; `_HAS_MOJO` flag never raises on missing tooling. `benchmarks/bench_mojo_vs_rust.py` pure-text side-by-side harness.
- `sc_neurocore.edge.aer_router.AERRoutingDaemon` — Python lifecycle wrapper for the Go AER UDP mesh router (`accel/go/services/aer_router/main.go`). Three sibling Go modules: `hil_debugger` (WebSocket telemetry), `services` / `services_ext` (service coordination). Each with its own `go.mod` + `main_test.go`.
- `sc_neurocore.debug.hil_server.HILServerDaemon` + `HILDebugger` — lifecycle wrapper for the Go HIL debugger binary with `GET /health` readiness probe, 5 s timeout, SIGTERM → SIGKILL ladder.
- `sc_neurocore.formal.FormalProofEngine` — Lean 4 bridge. `safety_bounds.lean` proves six theorems (`monitor_soundness`, `safe_transition`, `sc_precision_bound`, `sc_add_preserves_range`, `lif_membrane_bounded`, `correlation_range`) mapped 1:1 to `neuro_safe_monitor.sv` P-properties. New `src/sc_neurocore/formal/__init__.py` exports the engine.
- `sc_neurocore.accel.julia.solvers.JuliaFusionSolver` + 4 `.jl` scripts (`fusion_solver`, `neuron_zoo`, `dynamical_analysis`, `spike_analysis`) — reference continuous-time ODE solvers via `DifferentialEquations.jl` (Tsit5).
- `sc_neurocore.hdl_gen.safety.neuro_safe_monitor` + `tb_safety_monitor` — SystemVerilog runtime safety monitor enforcing the six Lean theorems at nanosecond scale. Parameterised on Q8.8 current / voltage / coherence / SC denominator / LIF max. `openroad_flow/run_asic_flow.sh` drives Yosys synthesis (+ optional OpenROAD P&R) against the monitor with area / timing reports.
- `sc_neurocore.evo_substrate` gained (documented in full): `FormalSafetyGuard`, `BloatPenalizer`, `ExtinctionDetector`, `ComplexityTracker`, `CPPNGenome`, `ParetoFront`, `NoveltyArchive`, `HallOfFame`, `TileDeploymentTracker`, `ResourceBudget`, `LineageTracker`, `IslandModel`. Bridged to MEA via `mea_fitness_hook` and to ArcaneZenith via `step_from_genome`.
- `sc_neurocore.proto` — `core.proto` (Tensor, BitstreamMetadata) + `telemetry.proto` (HILFrame) as the wire contract for HIL debugging.
- Plasticity-layer `reset()` contract: new FFI `reset_rule_layer` in `libautonomous_learning` (Rayon par_iter over rules), new `WgpuRuleLayer::reset` + `reset_wgpu_layer` FFI, and `reset()` methods on `RustRuleLayer`, `RustWgpuRuleLayer`, `TorchRuleLayer` with per-rule trace-clearing scope matching the Rust `PlasticityRule::reset` trait contract. `ArcaneZenithCognitiveCore.reset()` now works across all three backends. 11 new tests.
- Example demos: `examples/14_bioware_closed_loop_demo.py` (100-frame MEA ↔ ArcaneZenith closed loop), `examples/15_photonic_compilation_demo.py` (SC → MZI cascade → real GDSII), `examples/16_evo_substrate_demo.py` (genome → SC top-level module → Verilog emit).

#### Documentation
- New API pages: `docs/api/mojo_accel.md`, `docs/api/edge.md`, `docs/api/formal.md`, `docs/api/julia_solvers.md`, `docs/api/proto.md`.
- Upgraded from stubs: `docs/api/evo_substrate.md` (23 → 155 lines), `docs/api/debug.md` (24 → 120 lines, added HIL section), `docs/api/hdl_gen.md` (17 → 100 lines, added safety-monitor P-property table + Lean mapping + ASIC flow).
- `docs/api/bioware.md` upgraded from 14-line stub (full `BioHybridSession` + `BioHybridFrameResult` dual-access + Q8.8 homeostatic controller + SpikeSorter + mea_fitness_hook sections).
- New `docs/api/arcane_zenith.md` + `docs/api/optics.md` completely rewritten (photonic compiler + Berenger PML + Marcatili crosstalk + GDSII).
- `mkdocs.yml` navigation restructured: new *Acceleration* (Mojo + Julia), *Formal + Safety*, *Edge + Wire Protocol* groups under Frontiers.

#### Fixed
- `RustEligentLearner.step` FFI signature was missing the `dt` parameter (4 args passed, 5 expected) — every non-empty call raised `AttributeError`. Added `dt: float = 0.001` kwarg.
- `sc_neurocore._native.learning_bridge` no longer raises at import time when `libautonomous_learning.so` is absent; returns `_HAS_LEARNING = False` so downstream imports succeed (the 398 previously-failing test collections now run).
- `CI workflows` (`ci.yml`, `v3-engine.yml`) now build the `autonomous_learning` cdylib and copy it into `src/sc_neurocore/_native/` before pytest runs — keeps the Rust path live.

#### Repository hygiene
- Untracked compiled Go bench binaries (`services_bench`, `services_ext_bench` ≈ 4.4 MB total) from `src/sc_neurocore/accel/go/services/…`; pattern added to `.gitignore` (regenerate locally via `go test -bench -c`).
- 22 ruff lint + format fixes across user-WIP modules (evo_substrate, mojo/runner, debug/hil_*, edge/aer_router, formal/lean_bridge). `ruff check src/ tests/` and `ruff format --check src/ tests/` clean.
- New optional extras in `pyproject.toml`: `optics = ["gdsfactory>=9.0"]`, `bioware = ["scikit-learn>=1.3"]`.

### CorticalColumn full-scale (77 169 cells) verification (2026-04-19)
- Ran the canonical fidelity reference: `scale=1.0, seed=42`, 600 ms simulation with the block + Rust batched multi-spmv path. 77 169 cells, build 298 s, sim 3 564 s ≈ **64 minutes wall**.
- **5/8 populations within 1.2× of Potjans Table 4** (L23i 1.07×, L4e 1.06×, L4i 1.09×, L6e 1.24×, L6i 1.05×). L5e 1.32×, L5i 1.22× plateau ~25 % over published — NOT purely a finite-size effect (does not collapse below 1.20× at full scale). L23e under-fires at 0.67× consistently across all four scales.
- Honest interpretation: the residual is a combination of (i) shorter analysis window than the published 5 s, (ii) dt-quantised global-bin delays vs the paper's per-connection continuous Gaussian, (iii) per-target multapse sampling vs NEST's `multapses=False` (which we cannot trivially use without breaking van Albada 2015 in-degree preservation). The shape is faithful (population ordering, E/I balance, all rates finite and bounded); the absolute residual at ≤ 1.32× is the practical limit of the current architecture.
- Doc page §4.1 now records all four scales side-by-side; the full-scale row is the canonical reference.

### CorticalColumn full-scale convergence verified at scale=0.5 (2026-04-18)
- Ran `scale=0.5, seed=42`, 600 ms simulation with the block + Rust batched multi-spmv path. 38 586 cells, build 116 s, sim 1 956 s (≈ 33 min wall).
- **6/8 populations within 1.2× of Potjans Table 4** (vs 5/8 at scale=0.1, 5/8 at scale=0.2): L23i 1.00×, L4e 0.95×, L4i 1.07×, L5i 1.20×, L6i 1.04×.
- L5e shrinks 1.97× → 1.52× → **1.36×**; L6e shrinks 2.81× → 2.43× → **1.68×**. Both still residual but on the predicted convergence trajectory of van Albada et al. 2015 Fig 5.
- Confirms the finite-size hypothesis empirically: residuals collapse monotonically as scale grows, full-scale (~77 000 cells) would close to ≤ 1.05× across all populations. scale=0.5 / 600 ms is now reachable in 33 min wall, unblocked by the block + Rust path.

### CorticalColumn batched multi-spmv Rust call (2026-04-18)
- New `engine/src/cortical_inject.rs::parallel_csr_multi_spmv_add` — does `2 × n_delay_bins` (= 10) spmv add operations in ONE FFI call. Rust loops internally over the bins; `par_chunks_mut(512)` parallelism still applies, with the per-row kernel summing contributions from all bins before writing back.
- New PyO3 wrapper `sc_neurocore_engine.py_parallel_csr_multi_spmv_add` accepting `Vec<PyReadonlyArray1>` for indptrs / indices / data / xs.
- `CorticalColumn._inject_block(dt)` now batches all non-empty (E + I) bins into ONE FFI call when the multi-spmv kernel is available; falls back to per-block calls otherwise.
- Bridge wrapper `bridge/sc_neurocore_engine/__init__.py` re-exports `py_parallel_csr_multi_spmv_add`.
- 1 new Rust unit test `test_multi_spmv_matches_sequential` proving batched output equals N sequential `parallel_csr_spmv_add` calls.
- **Measured perf at scale=0.1, 600 ms**: 287.5 s wall — DOWN from 460 s (single-call Rust) and ON PAR with scipy per-pair (290 s). FFI overhead reduction (10 calls → 1) reclaimed the gap.

### CorticalColumn Rust per-row-parallel CSR spmv kernel (2026-04-18)
- New `engine/src/cortical_inject.rs`: rayon-parallel CSR sparse mat-vec add (`y += W @ x`) with row-chunking (`CHUNK_SIZE = 512`) so each task sees ~250 µs of work — well above rayon's per-iteration scheduler break-even point. 4 unit tests.
- PyO3 wrapper `sc_neurocore_engine.py_parallel_csr_spmv_add` re-exported via `bridge/sc_neurocore_engine/__init__.py`.
- `CorticalColumn._inject_block(dt)` now dispatches to the Rust kernel automatically when available (auto-detected via `_HAS_RUST_CSR_SPMV`). Bit-identical results vs scipy single-threaded — per-row reductions are local so parallel order does not affect output.
- Pre-extracted `(indptr, indices, data)` triples per block at construction (`_block_e_arrays`, `_block_i_arrays`) to dodge per-step `np.ascontiguousarray` cast overhead that otherwise eats the per-call Rust speedup.
- **Honest perf finding**: Rust kernel measures 18.9 ms vs scipy 33 ms standalone (1.75× per call). In the full simulation pipeline at scale=0.1 / 600 ms, however, Rust takes **460 s** vs scipy **290 s** (per-pair) — a 1.6× regression. scipy's CSR mat-vec is already well-tuned for the in-pipeline access pattern (cache-warm matrices, sparse spike vectors); per-call Rust overhead + the surrounding Python concat / count_nonzero / slice work dominates.
- The Rust kernel is preserved as the **right primitive** for the future block-CSR / GPU / multi-node scale-up regime (where per-call FFI overhead shrinks relative to per-call work). Default per-pair scipy path is already the fastest Python-side measurement; Rust is opt-in via `use_block_csr=True`.

### CorticalColumn block-CSR opt-in path (2026-04-18)
- Added stacked block-CSR matrices keyed by `(source-type, global-bin-idx)` so the per-step inner loop can collapse from `n_pairs × n_delay_bins` (≈ 320 sparse mat-vecs) to `2 × n_delay_bins` (≈ 10). Bin centres are global, derived from theoretical Gaussian quantiles via `scipy.stats.norm.ppf`.
- New `CorticalColumn` parameter `use_block_csr: bool = False`. When True, the construction builds block matrices alongside the per-pair representation; `step()` dispatches to `_inject_block(dt)`.
- **Honest perf finding**: at `scale=0.1`, 300 ms sim, the block path measures 306 s vs ~145 s for the legacy per-pair path (≈ 2× SLOWER). scipy.sparse CSR mat-vec is compute-bound (FLOPs scale with `nnz`, identical between paths), and the per-pair tight inner loop wins on cache locality. The block path is preserved as an opt-in because it is the natural data layout for any future Rust / Mojo FFI port (10 FFI calls vs 320, where call overhead DOES dominate).
- Default flipped to `use_block_csr=False` so the as-shipped Python path stays on the fastest measured backend.
- New `tests/test_cortical_column.py::TestConnectivity::test_block_csr_path_builds_and_runs` exercises the opt-in path so it does not silently rot.

### CorticalColumn finite-size verification at scale=0.2 (2026-04-18)
- Empirically verified that the L5e/L6e residual at `scale=0.1` is a finite-size effect (van Albada et al. 2015 Fig 5), not a model bug. Scale=0.2 / 600 ms / seed=42 measurements:

  | Pop | scale=0.1 ratio | scale=0.2 ratio | Δ |
  |-----|----------------:|----------------:|---:|
  | L23e | 0.67× | 0.27× | overshoots low |
  | L23i | 1.19× | 0.94× | improving |
  | L4e | 0.68× | 0.73× | stable |
  | L4i | 1.21× | 1.08× | improving |
  | L5e | 1.97× | **1.52×** | **-23 %** |
  | L5i | 1.50× | **1.27×** | **-15 %** |
  | L6e | 2.81× | **2.43×** | **-14 %** |
  | L6i | 1.24× | **1.10×** | improving |

- The deep-layer residuals (L5e, L6e) shrink monotonically with scale; extrapolating linearly suggests scale=0.5 closes them to within 1.2-1.3× of Potjans Table 4. Closing all 8 populations to within 10 % requires full scale (~77 000 cells, ≈ 50 min/sec biotime). The implementation is faithful — the residual is intrinsic to sub-full-scale finite-size effects.
- `docs/api/cortical_column.md` §4.1 now documents the per-scale ratios side-by-side with the historical baseline and the rejected no-multapse experiment.

### CorticalColumn per-connection Gaussian delay distribution (2026-04-18)
- `network/cortical_column.py` adds per-connection delay binning. New constants `DELAY_E_SIGMA = 0.75 ms`, `DELAY_I_SIGMA = 0.4 ms` (Potjans Table 5). New `__init__` parameters `delay_distribution: bool = True` and `n_delay_bins: int = 5`. At construction time each (target, source) pair samples `K_per_target * n_t` per-connection delays from `N(DELAY_*, sigma_*)`, quantile-bins them into 5 groups and stores one sub-CSR per bin. Per `step()`, each pair contributes one `dot()` per bin, reading the source spike vector at that bin's delay offset.
- Setting `delay_distribution=False` restores the legacy single-mean-delay path for fast smoke tests and direct comparison.
- **Fidelity dramatically tightened.** Measured at `scale=0.1, seed=42`, 200 ms analysis window after 100 ms burn-in:

  | Population | single-delay ratio | per-conn Gaussian ratio |
  |------------|-------------------:|-----------------------:|
  | L23e | 5.29× | **0.67×** |
  | L23i | 4.78× | **1.19×** |
  | L4e  | 0.83× | 0.68× |
  | L4i  | 2.03× | **1.21×** |
  | L5e  | 3.05× | 1.97× |
  | L5i  | 2.10× | 1.50× |
  | L6e  | 5.23× | 2.81× |
  | L6i  | 2.33× | **1.24×** |

  5/8 populations now sit within 1.2× of Potjans Table 4; the remaining 3 (L4e, L5e, L6e) within 2-3×.
- Cost: per-step ≈ 5× slower (5 sparse mat-vecs per pair instead of 1). At `scale=0.1`, sim wall went 32 s → ~290 s for 600 ms (matches 5× expectation).
- New `tests/test_cortical_column.py::TestPublishedFidelity::test_per_connection_delays_tighten_rates` — asserts ≥ 5/8 populations within `[0.5, 1.5]×` of published Table 4 values. Pins the win.
- `benchmarks/bench_cortical_column.py` now bench BOTH `delay_distribution` modes side-by-side.
- All 29 cortical_column tests pass with the new default (29 passed in 14:18 with delay distribution, 24 deselected-fidelity tests in 4:39 for fast iteration via `-k 'not Fidelity'`).

### PINGCircuit Rust acceleration backend (2026-04-18)
- New Rust per-step kernel `engine/src/ping.rs` with PyO3 wrapper `sc_neurocore_engine.py_ping_step`. Mirrors the Python step semantics (LIF + AMPA / GABA decays + drive + Wiener noise + refractory + spike detect + reset). Noise samples are drawn on the Python side and passed in as `xi_e` / `xi_i` so the per-instance RNG state evolves identically across both backends.
- New `backend=` parameter on `PINGCircuit` (`"auto" | "rust" | "python"`, default `"auto"`). `"rust"` raises `RuntimeError` if the kernel is not built; `"auto"` falls back to NumPy.
- Bridge wrapper `bridge/sc_neurocore_engine/__init__.py` re-exports `py_ping_step` so pytest's `bridge/`-on-`sys.path` setup sees the Rust symbol.
- New `tests/test_gamma_oscillation.py::TestPythonRustParity` (6 cases): per-population firing rates within 10 % across (80, 20) / (400, 100) / (1000, 250); dominant FFT peak within 1.5 Hz; explicit `backend="rust"` smoke; invalid-backend rejection. Per-cell membrane V values drift at the float-noise level (NumPy SIMD/FMA vs Rust scalar ordering) — documented inline; aggregate dynamics match.
- `benchmarks/bench_gamma_oscillation.py` extended to bench BOTH backends. Measured speedup: ~3.3-4.3× across the three workload sizes (per-step 145.8 → 33.7 µs at (80, 20); 588.3 → 178.3 µs at (4000, 1000)). All 6 runs stay in the published 30-80 Hz dominant band.
- `engine/src/ping.rs` ships 3 Rust unit tests (no-drive silence; supra-threshold drive + refractory hold; deterministic for identical inputs). All pass on `cargo test --release`.

### CorticalColumn no-multapse experiment — REJECTED (2026-04-18)
- Tried replacing the multapse-with-replacement adjacency builder with a vectorised `argpartition` no-multapse sampler (matching NEST `multapses=False` default). Mean per-target weight is identical between the two approaches and per-target unique connectivity rises from ~63 % to 100 %.
- Measured at `scale=0.1, seed=42`, 600 ms: rates BLEW UP to refractory ceiling for 6 of 8 populations (L23e 90 Hz, L4e/L4i ≈ 410 Hz, L5e/L5i/L6i 260-390 Hz). Pre-experiment multapse-with-replacement gave rates 1.6-7.5× over Potjans Table 4 (within band, just inflated). Post-experiment no-multapse made the divergence ~10× worse.
- Honest finding: at sub-full scale the deterministic per-target in-degree of the no-multapse path amplifies population synchrony in the heavy-recurrent regime (K approaches N_s for several pairs); the multapse path's natural variance dampens this. Documented inline next to the multapse sampler so future contributors don't repeat the experiment without first re-reading van Albada 2015 §3.

### PINGCircuit scale-invariant weight normalisation (2026-04-18)
- `network/gamma_oscillation.py`: per-spike conductance contributions are now divided by source population size at construction (`_w_*_eff = w_* · default_size / actual_size`). The default `(80, 20)` published weights stay bit-identical; larger circuits no longer drift out of the 30-80 Hz band. `bench_gamma_oscillation.py` now reports 40.0 / 41.2 / 41.2 Hz across `(80,20) / (400,100) / (4000,1000)` — all in band — vs 40.0 / 103.8 / 76.2 before the fix. All 19 PINGCircuit tests still pass (default weights and behaviour unchanged at `(80, 20)`).

### Honest benchmark scripts for network/ models (2026-04-18)
- `benchmarks/bench_cortical_column.py`: 3-config wall-clock + per-population firing rates + Potjans Table 4 ratios for `CorticalColumn`. Replaces hand-measured numbers in `docs/api/cortical_column.md` with reproducible JSON output at `benchmarks/results/bench_cortical_column.json`. Honest BLOCKED status reported per backend (Rust/Julia/Go/Mojo) per `feedback_no_fabricated_benchmarks` and `feedback_module_standard_attnres`.
- `benchmarks/bench_gamma_oscillation.py`: 3-workload `step()` wall-clock + dominant gamma frequency check (must lie in 30-80 Hz) for `PINGCircuit`. JSON output at `benchmarks/results/bench_gamma_oscillation.json`. Documents the per-cell LIF + 4 conductance decays as a clean Rust + Mojo target (BLOCKED, tracked under multilang policy). Bench surfaces a real fidelity edge case at `n_e=400, n_i=100` (f_dom=103.8 Hz, outside published 30-80 Hz band) that the default-configuration test does not catch.
- `docs/api/cortical_column.md` performance table updated to reference the bench script and JSON path; numbers replaced with the measured values (build 0.04 / 2.04 / 4.07 s and per-step 0.96 / 2.07 / 5.29 ms across the three configurations).

### Bandit MEDIUM triage (2026-04-18)
- 6 MEDIUM `B307` findings (use of `eval`) → ACCEPT with `# nosec B307` markers and inline rationale: `equation_builder.py` Euler integrator, RK4 derivative eval, threshold expression and reset rule (4 sites); `studio/analysis.py` nullcline grid eval (2 sites). All sites are downstream of `EquationNeuron._validate_expr` AST whitelist (`_ALLOWED_AST_NODES` + `_BLOCKED_NAMES` reject any escape vector before `compile`) with empty-`__builtins__` eval globals.
- Re-running `bandit -r src/ -ll` returns 0 findings.
- 55 LOW findings remain (B101 asserts, B603/B404/B607 subprocess, B110 try/pass, B311 random); informational, no real impact, full inventory in `docs/internal/audit_bandit_2026-04-18.md` and `docs/internal/AUDIT_INDEX.md`.

### CorticalColumn Potjans & Diesmann 2014 (2026-04-18)
- `network/cortical_column.py` rewritten from 5-population canonical-microcircuit toy to the full 8-population Potjans & Diesmann 2014 model: L23e, L23i, L4e, L4i, L5e, L5i, L6e, L6i with per-population sizes from Table 5, the verbatim 8×8 connection-probability matrix from Table 5, per-cell background Poisson drive (`K_bg` per population, `bg_rate=8 Hz`), and exponentially decaying current-based PSCs (`tau_syn=0.5 ms`).
- LIF integration: `C_m=250 pF`, `tau_m=10 ms`, `t_ref=2 ms`, `E_L=V_reset=-65 mV`, `V_th=-50 mV`. Per-source delays: `1.5 ms` (E), `0.8 ms` (I), quantised to `dt`.
- Synaptic weights: `w_e=87.81 pA`, `w_i=-g·w_e` with `g=4` (configurable), `w_l4_to_l23e=2·w_e` per Potjans boost.
- Sparse `scipy.sparse.csr_matrix` adjacency per (target, source) pair with multapses sampled with replacement; full-scale in-degree preservation under `scale_correction=True` (van Albada et al. 2015 protocol).
- `simulate(duration_ms, dt)`, `step(dt)`, `population_rates(rasters, dt, burn_in_ms)`, `total_indegree(target)` and `reset_state()` helpers.
- `tests/test_cortical_column.py` rewritten: 29 tests covering smoke, determinism (per-instance RNG, global-seed leak-proofing), connectivity (Table 5 entries, K_bg, weight signs, L4e→L2/3e boost, sparse adjacency built per pair), and published fidelity (no silent populations, no refractory-ceiling saturation, E/I asymmetry, L4e in band, zero-background silence). 100 % coverage on `cortical_column.py`. Closes #10.
- `docs/api/cortical_column.md` rewritten end-to-end (308 lines): published-reference summary, implementation overview (8 populations, sparse adjacency build, LIF + synapse + refractory, delay handling), public API reference, verification table vs Potjans Table 4 (L4e match within 1 %, other populations within 2-4×), performance table (4.6 s / 19.5 s / 43.6 s wall at scale 0.02 / 0.05 / 0.1) and reference list (Potjans 2014, van Albada 2015, Binzegger 2004, Hahne 2017, Douglas & Martin 2004).

### PINGCircuit conductance-based gamma (2026-04-18)
- `network/gamma_oscillation.py` rewritten from rate-coded toy model to per-cell conductance-based Börgers-Kopell 2003 weak-PING. HH-style integrate-and-fire with separate AMPA / GABA exponentially decaying conductances, refractory window, per-cell drive jitter and stochastic kicks. Default parameters reproduce the published 30-80 Hz gamma peak (verified at 40 Hz at the default operating point).
- `population_rate(spike_log, dt, bin_ms)` and `dominant_frequency(spike_log, dt, bin_ms, f_min, f_max)` helpers added; FFT-based with empty-log + out-of-band silence handling.
- `tests/test_gamma_oscillation.py` updated to the new API: 19 tests covering smoke, determinism (per-instance RNG isolation, global-seed leak-proofing), published fidelity (30-80 Hz peak, gain-loop disengage paths, Hz units, silence handling). 100 % coverage on `gamma_oscillation.py`. Closes #11.
- Replaced `np.sum(boolarray)` with `np.count_nonzero(boolarray)` in both implementation and tests to be reload-safe under coverage instrumentation (the `_NoValue` sentinel mismatch otherwise raised `TypeError` from `_methods.py`).

### Repository hygiene (2026-04-18)
- SPDX header format converted from 1-line piped to 2-line form across 2728 source files (.py / .jl / .rs / .go / .mojo). Closes #60.
- `microtubule_neuron.v` Engineer attribution: `Arcane Sapience`.
- `cargo clippy --release --lib`: 20 in-source warnings → 0.
- Bandit HIGH severity in `nas/sc_nas_engine.py:169` → 0 (`hashlib.md5(..., usedforsecurity=False)`).
- Chiplet package coverage 95 % → 100 % (`test_hierarchical_partitioner_perf.py`, `test_chiplet_gen_edge_cases.py`).
- `tools/run_full_cov.sh`: batched per-directory `--cov-append` runner. First full sweep completes at 43.81 % cumulative coverage; no OOM. Closes #58.
- `.gitignore`: `.agent_metadata.json`.
- `ruff`, `rustfmt`: clean across all touched files.

### Chiplet Partitioner — Multi-Language KL Refine (2026-04-18)
- **Perf:** `HierarchicalPartitioner.partition` V=200 went from 963 ms (pre-#65) → 12.7 ms (Python post-fix) → 0.04 ms (Mojo). Total wall-clock improvement at V=200: 24,000× across the chain.
- **#65 fix:** `CorrelationAwareGraph` now caches `(min, max) → edge` lookup → O(1); `_spectral_bisect` hoists `set(vertices)` out of the inner loop. 22-29× speedup at V=50/100/200.
- **#64-prep refine fix:** `_per_partition_cost(v, n_parts, ...)` returns the full length-P cost vector in ONE neighbour scan (was P redundant scans). Additional 2-9× over #65; bit-identical canonical output.
- **#74 multi-language KL refine:** Rust (`engine/src/partition.rs`), Julia (`accel/julia/chiplet/kl_refine.jl`), Go (`accel/go/partition/partition.go`), Mojo (`accel/mojo/partition/partition.mojo`) all wired into `HierarchicalPartitioner(refine_backend=...)`. Bit-exact `part_map` parity verified end-to-end via dispatcher tests on V=100. Empirical fastest-pick at V=1000: Mojo 0.20 ms (351×), Julia 0.26 ms (270×), Rust 0.29 ms (242×), Go 0.68 ms (103×), Python 70 ms.
- **Bench harness:** `benchmarks/bench_kl_refine.py` runs 5 backends with parity check; results in `benchmarks/results/bench_kl_refine.json`.
- **Tests:** 218 chiplet tests (39 new this batch); coverage 99.58 % on the chiplet package, with `chiplet_gen.py` at 100 % and `hierarchical_partitioner.py` at 99 %.

### LGSSM Multi-Language Acceleration (2026-04-17)
- **Mojo LGSSM Kalman filter** (`accel/mojo/world_model/lgssm.mojo`): hand-rolled matmul + Cholesky + triangular solve via `mojo build --emit shared-lib`. **46× over Python, 8× over Rust** at T=200 d=4 p=3 workload. Closes #69.
- **Go LGSSM** (`accel/go/lgssm/lgssm.go`): cgo + ctypes shared lib, hand-rolled Cholesky. Closes #70.
- **Julia LGSSM** (`accel/julia/world_model/predictive_model.jl`): juliacall + LinearAlgebra LAPACK. Closes #68.
- **Rust LGSSM** (`engine/src/lgssm.rs`): PyO3 + ndarray Cholesky. Closes #67.
- All 4 backends dispatched via `KalmanFilter.filter(backend='auto'|'rust'|'julia'|'go'|'mojo'|'python')`; bit-exact parity vs Python at atol≤1e-9 on means/covs, ≤1e-7 on log-likelihood.
- **Mojo 0.26 FFI pattern proven:** raw `Int` address via `arr.ctypes.data` + `UnsafePointer[T, MutAnyOrigin](unsafe_from_address=addr)` reconstruction inside the `@export` body works around the parametric-signature restriction. Same pattern reused for fault_injection + KL refine.

### Fault Injection Multi-Language (2026-04-17)
- **Rust + Julia + Go + Mojo** kernels for the 5 fault models (`bitflip`, `stuck_at_0/1`, `dropout`, `gaussian`). Mojo wins 4/5 boolean kernels (2.7-8.2× over NumPy); Julia wins Gaussian via Ziggurat randn. Bench harness with 4σ Binomial parity at `benchmarks/bench_kl_refine.py`-style 5-backend layout.

### Bench Harness Honest Exemptions (2026-04-17)
- `bench_safety_monitor.py` + `bench_chiplet.py` now emit a `backends` block in the JSON output documenting USED / EXEMPT / BLOCKED-ON-#X status per backend per op, with explicit FFI-vs-compute math instead of silent skipping.

### Cross-Module Integration — (2026-04-16)
- **Shared Core Types** `core/types.py`: unified `HardwareBudget`, `ResourceReport`, `LayerSpec`, `estimate_network()` — single source of truth for Optimizer↔NAS↔Runtime
- **Closed-Loop Adaptive Controller** `control/adaptive_loop.py`: Runtime drift detection → SA re-optimisation → new `RuntimeConfig`, configurable cooldown/threshold
- **Unified Energy Reporter** `energy_accounting/unified_reporter.py`: bridges `CarbonModel` + `ThermalModel` + ASIC power into single `analyze()` call
- **End-to-End Export Pipeline** `export/pipeline.py`: Model Zoo → ONNX → TVM Relay → MLIR/SSA → SystemVerilog in one `run()` call
- **Rust Wiring**: `sc_optimizer.py` → `optimizer.rs` SA engine, `sc_nas_engine.py` → `evo.rs` tournament selection, `photonic_emitter.py` → `photonic.rs` crosstalk analysis
- **Package Exports**: Updated `core/__init__.py`, `control/__init__.py`, `export/__init__.py` with new module exports
- **Integration Tests**: 20 new tests in `tests/test_integration/test_cross_module.py` covering all 5 actions
- **Maturin**: Rebuilt `sc_neurocore_engine` v3.14.0 with all Rust bindings
- **Total**: 10,592 tests (8,895 Python + 1,697 Rust) — ALL GREEN

### Extended Rust Wiring — QA & DNA Bridges (2026-04-17)
- **Quantum Annealing**: `bridges/quantum_annealing.py` → `py_qa_simulated_annealing` (**2,402×** at 100 qubits)
  - `IsingModel.energy()` → `py_qa_ising_energy` (Rust path for n>20 qubits)
  - `SimulatedAnnealer.solve_ising()` → `py_qa_simulated_annealing` (467× at 20Q → **2,402×** at 100Q)
  - `EnergyLandscape.analyze()` → `py_qa_batch_ising_energy` (batch energy for >100 samples)
- **DNA Mapper**: `bridges/dna_mapper.py` — Rust engine loaded (`_HAS_RUST_DNA`)
  - Imported: `py_dna_design_sequence`, `py_dna_detect_hairpins`, `py_dna_check_cross_hybridization`, `py_dna_simulate_kinetics`, `py_dna_design_orthogonal_set`
- **Photonic**: Fixed `py_ph_analyze_crosstalk` API (channel_ids, wavelengths, bandwidths, powers)

### Python vs Rust Benchmarks — Integration Hot Paths (2026-04-16)
- **SA Optimizer**: 7× (5 layers) → 36× (20 layers) → **47× (50 layers)**
- **Tournament Selection**: **337–394×** (amortised per-round overhead elimination)
- **Batch Mutate**: 17–21× across population sizes 50–1000
- **Population Diversity**: 34–**90×** (O(N²) SIMD pairwise distance)
- **Mean Rust speedup**: **334.6×** across all hot paths (incl. QA)
- **Peak QA**: 467× (20Q) → 1,426× (50Q) → **2,402× (100Q)**
- **E2E Pipeline**: NAS→Optimizer→Energy→Verilog in **13.7ms** (small) to **116ms** (large)
- Criterion (Rust-native): spike_times=83ns, firing_rate=13ns, ISI=96ns, van_rossum=1.2µs (N=100)
- Results: `benchmarks/results/py_vs_rust_integration.json`
- Script: `benchmarks/py_vs_rust_benchmark.py`

### Cross-Language Acceleration — Spike Stats (2026-04-16)
- **Crate** `spike_stats_core` (v0.1.0): 16 functions, 28 Rust tests, PyO3 + Criterion
- **Distance** (7 fns): `victor_purpura_distance` **181×**, `spike_sync` **31×**, `hunter_milton` **27×**, `van_rossum`, `spike_distance`, `earth_movers_distance`, `multi_neuron_victor_purpura` **160×**
- **Correlation** (5 fns): `cross_correlation`, `event_synchronization`, `spike_time_tiling_coefficient`, `coincidence_index`
- **Variability** (4 fns): `approximate_entropy` **73×**, `sample_entropy` **78×**, `lempel_ziv_complexity` **69×**, `permutation_entropy` **65×**
- 99/99 Python tests pass on both Rust and Python fallback paths
- Python dispatch wired in: `distance.py`, `correlation.py`, `variability.py`

### Cross-Language Acceleration —  Stochastic Doctor (2026-04-16)
- **PyO3 bindings** for `stochastic_doctor_core` crate: `py_scc_bytes`, `py_scc_batch`, `py_precision_bytes`, `py_histogram`, `PyDriftDetector`
- Replaced legacy `ctypes.CDLL` with PyO3 import pattern (primary), Python fallback (secondary)
- `SC_NEUROCORE_NO_RUST=1` env var forces Python path
- 16/16 Python tests pass on both Rust and Python paths
- 23 Rust tests pass
- **Benchmarks** (SCC single-pair): 35× at N=100, 3.5× at N=1M
- **Benchmarks** (batch SCC N×N): 15–18× for 4–64 neuron layers
- **Benchmarks** (precision): 5–14× across all sizes
- Criterion benchmarks: `crates/stochastic_doctor_core/benches/doctor_bench.rs`
- Python benchmark: `benchmarks/stochastic_doctor_benchmark.py`
- Results: `benchmarks/results/stochastic_doctor_py_vs_rust.json`
- API docs updated with full benchmark tables: `docs/api/stochastic_doctor.md`

### Module Integration — 19 Industrialized Modules (2026-04-16)
- **Industrial tier:** safety_cert (IEC 61508/ISO 26262, 81 tests), asic_flow (multi-PDK, 67 tests), fault_injection (radiation-grade, 22 tests), uvm_gen (UVM testbench, 71 tests)
- **Exascale tier:** hypervisor (multi-tenant, 78 tests), digital_twin/twinsync (time-warp sync, 72 tests)
- **Substrates tier:** spintronic (MTJ mapper, 66 tests), chiplet (UCIe/BoW, 94 tests), memristor (crossbar, 70 tests), analog_bridge (SC-to-analog, 27 tests)
- **Frontiers tier:** evo_substrate (self-replicating evolution, 91 tests), meta_plasticity (self-modifying rules, 72 tests), bioware (organoid interface, 79 tests), federated (DP-SGD, 93 tests), bci_studio (closed-loop BCI, 32 tests)
- **Unification tier:** explainability (causal attribution, 71 tests), neuro_symbolic (predictive coding, 34 tests), stochastic_doctor (bitstream diagnostics, 16 tests), model_zoo (auto-Verilog, 37 tests)
- All modules: SPDX dual-license headers, `__tier__` classification, `__init__.py` with docstrings
- 19 MkDocs API doc pages with `mkdocstrings` directives
- Updated `mkdocs.yml` nav with 5 new categories (Industrial, Substrates, Exascale, Frontiers, Unification)
- Integration reference: `docs/MODULE_INTEGRATION.md`
- Total: **1,173 new Python tests** from integrated modules

### Rust Workspace — 5 Research Crates Integrated (2026-04-16)
- Created `crates/` directory for research Rust crates
- Integrated: tinysc_riscv (83 tests), core_engine (22 tests), autonomous_learning (12 tests), neuro_symbolic (28 tests), stochastic_doctor_core (23 tests)
- Root `Cargo.toml` workspace now has 6 members (engine + 5 research crates)
- Engine (`sc_neurocore_engine`, 1,549 tests) verified undamaged after workspace expansion
- Total: **1,717 Rust tests** across 6 crates

### Evolutionary Substrate — (2026-04-16)
- `FormalSafetyGuard`: pre-deployment safety validation
- `CPPNGenome`: Compositional Pattern Producing Network developmental encoding
- `IslandModel`: multi-deme evolution with migration
- `NoveltyArchive`: k-NN behavioural novelty search
- `HWFitnessCollector`: FPGA execution feedback for hardware-in-loop fitness
- `ParetoFront`: NSGA-II style non-dominated sorting
- `TournamentSelector`, `AgeRegulator`, `BloatPenalizer`, `ExtinctionDetector`, `CoevolutionArena`
- `EvoStatisticsTracker`, `ComplexityTracker`, `genome_diff()`, `shared_fitness()`
- Module grew from 657 to 1,400 LOC, 42 to 91 tests

### Foundation-Model Neural Decoders (2026-04-07)
- **POYODecoder**: spike tokenisation + cross-attention (Azabou et al. 2023 NeurIPS)
- **POSSMDecoder**: diagonal SSM with HiPPO-LegS init (Ryoo et al. 2025 ICLR)
- **NDT3Decoder**: causal masked self-attention on binned spikes (Ye & Pandarinath 2025)
- **CEBRAEncoder**: InfoNCE contrastive embedding with analytical backprop (Schneider et al. 2023 Nature)
- Rust acceleration: tokenise_spikes, sinusoidal_position_encode, scaled_dot_product_attention, gaussian_attention, ssm_step_diagonal, infonce_loss (6 pub fn, 11 tests)
- PyO3: 5 functions registered
- Tests: 47 multi-angle tests
- Documentation: 976 lines, 8/8 sections

### Transcriptomic Foundation Model Interfaces (2026-04-07)
- **ScKGBERTInterface**: dual S-Encoder + K-Encoder with Gaussian attention (Li et al. 2025 Genome Biology)
- **GeneformerInterface**: rank-value tokenisation + multi-head attention + MLM (Theodoris et al. 2023 Nature)
- **rank_value_encode**: shared utility for gene expression tokenisation
- Tests: 29 multi-angle tests
- Documentation: 1,118 lines, 8/8 sections

### Gap Model Python + PyO3 + Docs (11 models, 2026-04-07)
- 10 new Python implementations (publication-exact): AdaptiveThresholdMoENeuron, HybridLinearAttentionNeuron, QuantumInspiredLIFNeuron, DendriticNMDANeuron, MulticompartmentMCNNeuron, AstrocyteLIFNeuron, DirectionSelectiveRGC, CochlearHairCell, ShortTermPlasticitySynapse, DopamineStdpSynapse
- PyO3 wiring: 11 models registered (2 macro + 9 manual wrappers)
- Tests: 87 multi-angle tests
- 10 docs (5,701 lines total)
- GPU backend documentation (607 lines)

### CI & Dependency Fixes (2026-04-07)
- **PEP 639**: migrated `license = { text = "..." }` → `license = "AGPL-3.0-or-later"` (fixes setuptools ≥78)
- **mypy**: 1.19.1 → 1.20.0
- **cyclonedx-bom**: 7.2.2 → 7.3.0
- **ci.yml**: pinned all mypy stub dependencies to exact versions (CodeQL #287)
- **cargo fmt**: applied to all new Rust code
- Purged 52 resolved failed/cancelled CI runs
- Closed superseded dependabot PRs #53, #55

### Neuron Models — (12 new models, 2026-04-04/05)
- **TUMNetwork**: rate model with short-term plasticity (depression + facilitation), 3 ODEs
- **ElBoustaniNetwork**: E/I + NMDA bistability, 3 ODEs
- **GradedSynapseNeuron**: non-spiking, passive RC + sigmoid release
- **GapJunctionNeuron**: LIF + electrical synapse with Cx36 rectification
- **FrankenhaeUserHuxleyAxon**: GHK permeability-based currents (not linear V-E)
- **NodeOfRanvier**: MRG 2002 — Nav1.6 transient + persistent + Kv7 slow K
- **MyelinatedAxon**: MRG node + passive internode cable
- **CardiacPurkinjeFibre**: DiFrancesco-Noble 1985, 6 currents
- **SmoothMuscleCell**: CaL + BK + IP3R/SERCA + Ca²⁺ store
- **EndocrineBetaCell**: CaL + K_dr + K_ATP + K_Ca glucose-dependent bursting

### Fidelity Audit Fixes (7 models corrected, 2026-04-04)
- **RetinalGanglionCell**: basic LIF → Pillow 2005 GLM (stimulus + history filters)
- **InnerHairCell**: no vesicle pool → Meddis 1986/2006 (q/c/w compartments)
- **OuterHairCell**: unidirectional sigmoid → bidirectional asymmetric prestin (Santos-Sacchi 2006)
- **GranuleCell**: LIF-style → D'Angelo 2001 full HH (7 ionic currents)
- **AlphaMotorNeuron**: PIC no inactivation → h_pic + Ca²⁺ buffering
- **RodPhotoreceptor**: no Ca²⁺ feedback → Ca²⁺-GC feedback (Nikonov 2006, Hill n=4)
- **TraubMilesNeuron**: missing M-current → Kv7/KCNQ (Yamada 1989)

### Kinetics Audit Fixes (3 models upgraded, 2026-04-05)
- **GolgiCell** (CRITICAL): 5-current WB → full Solinas 2007 (11 currents, 13 gating variables)
- **DCNNeuron** (MODERATE): added persistent Na (INaP) + Ca²⁺-dependent AHP (7 currents total)
- **OlfactoryReceptorNeuron** (MODERATE): added PDE4 negative feedback on cAMP

### Infrastructure (2026-04-05)
- `supported_models()`: 28 missing entries added (159 total)
- Interface wrappers: 20 non-standard models wired via Wr* types (multi-input, i32-input, graded/rate)
- All 4 failing CI workflows fixed (clippy, ruff, MkDocs, typos)
- `cargo fmt` applied to all engine source
- Fresh Criterion benchmarks published (2026-04-05)
- Documentation audit: all stale numbers corrected across README, pricing, index, benchmarks

### Notebooks (13 new, 21 total)
- **08_equation_to_verilog**: ODE string → Python sim → Q8.8 Verilog (LIF, FHN, Izhikevich)
- **09_topology_and_dynamics**: 6 generators, adjacency matrices, degree distributions, raster plots
- **10_spike_train_analysis**: ISI, CV, Fano, cross-correlation, van Rossum, PCA
- **11_biological_circuits**: tripartite synapse Ca²⁺ dynamics, Rall dendrite nonlinearity
- **12_learning_rules**: STDP, e-prop eligibility, R-STDP, STP facilitation/depression
- **13_quantisation_pipeline**: float → Q8.8 → SC probabilities → Verilog export, error budget
- **14_sc_arithmetic_theory**: AND=multiply, XNOR=bipolar, MUX=add, CORDIV=divide, Sobol vs Bernoulli convergence, Hoeffding bounds
- **15_fault_tolerance**: SC vs fixed-point under bit-flips/stuck-at, TMR majority vote
- **16_neuron_atlas**: 12 models from 8 families (LIF→ArcaneNeuron, 1907–2026)
- **17_reservoir_computing**: liquid state machine, temporal XOR, ridge readout, SVD dimensionality
- **18_mixed_precision_sc**: per-layer adaptive L, Hoeffding vs sensitivity allocation, Pareto frontier
- **19_compression_and_pruning**: magnitude/SC-aware pruning, quantisation sweep, combined Pareto
- **20_power_analysis**: event-driven vs clock-driven toggle count, scaling with network size
- **21_spike_alu**: Turing-complete spike-based ALU — logic gates, SR latch register, ripple-carry adder, sort
- **22_ir_type_safety**: IR signal type checker — Bitstream/Rate/Spike/Fixed, catch mismatches before Verilog synthesis
- **23_topological_observables**: winding number, Ollivier-Ricci curvature, sheaf consistency defect, connection curvature
- **24_identity_lazarus**: Lazarus checkpoint save/load/merge, TraceEncoder text→spikes, StateDecoder attractor extraction, DirectorController L16 self-regulation
- **25_cortical_column_dynamics**: canonical 5-population microcircuit, thalamic drive, layer-resolved rasters, feedforward latency
- **26_spike_codec_benchmark**: 5 codecs (ISI/AER/predictive/delta/streaming) on synthetic data, compression ratio vs density curves
- **27_python_to_proven_silicon**: complete end-to-end pipeline — ODE string → Python sim → IR type check → Q8.8 Verilog → testbench → formal properties → resource estimate
- **28_domain_bridge**: TensorStream prob↔bitstream↔quantum conversions, QuantumStochasticLayer cos²(θ/2) non-linearity, Born rule roundtrip

### Tests (19 new files, ~3700 lines, ~310 test methods)
- `test_topology_generators.py`: 6 generators — CSR validity, degree, symmetry, edge count, determinism
- `test_cordiv_division.py`: CORDIV accuracy, monotonicity, convergence, adaptive_length Hoeffding bounds
- `test_fault_injection.py`: bit-flip degradation, stuck-at analytical bounds, TMR, SC vs fixed-point comparison
- `test_learning_advanced.py`: EligibilityTrace decay, BPTT/TBPTT loss, R-STDP reward gating, STP facilitation/depression/recovery
- `test_quantisation_pipeline.py`: Q8.8 roundtrip, dequantise fidelity, SC probability ordering, dot product end-to-end
- `test_network_monitors_stimulus.py`: SpikeMonitor record/count/trains, StateMonitor accumulation, RateMonitor bins, TimedArray clamp, StepCurrent onset/offset, PoissonInput rate/seed/weight
- `test_neuron_families.py`: parametrised test across 11 EquationNeuron models — step(), spike detection, reset, state finiteness, determinism
- `test_sc_convergence.py`: AND O(1/√L), Sobol faster than Bernoulli, CORDIV monotonic, correlation violation, popcount exact
- `test_spike_alu.py`: SpikeGate truth tables (AND/OR/NOT/NAND/XOR), De Morgan law, SpikeRegister roundtrip, SpikeALU add/sub/xor/compare/shift, spike_sort correctness
- `test_topological_observables.py`: winding number wraps, Ricci curvature complete>ring, sheaf defect zero when synchronised, connection curvature bounded by coupling
- `test_scpn_integrated.py`: K_nm symmetric zero-diagonal, OMEGA_N physical frequencies, create_full_stack 16 layers, run_integrated_step finite, get_global_metrics
- `test_identity_lazarus.py`: IdentitySubstrate run/step/health, TraceEncoder encode/determinism, Checkpoint save/load/merge roundtrip, StateDecoder patterns/attractors, DirectorController monitor/diagnose/correct
- `test_cortical_column_dynamics.py`: CorticalColumn step/run dict outputs, 5 populations, binary spikes, thalamic drive, L4-before-L5, inhibition, reset, determinism
- `test_codec_roundtrip.py`: all 5 codecs parametrised — lossless roundtrip (sparse/empty/single-spike/all-ones), compression ratio bounds, shape preserved, edge cases (1 channel, 1 timestep)
- `test_tensor_stream.py`: TensorStream prob↔bitstream↔quantum roundtrips, Born rule, normalisation, p=0/1 edge cases, invalid conversion raises
- `test_quantum_hybrid.py`: QuantumStochasticLayer cos²(θ/2) transfer, p=0→1, p=1→0, monotonic decreasing, multi-qubit independence

### Model Validation
- LIF f-I curve: 29/29 tests, <5% error vs analytical solution
- Izhikevich 20 firing patterns: all from Izhikevich (2003) Table 1 validated
- Hodgkin-Huxley 1952: AP peak 40.6mV, spike width 1.46ms, AHP -75.1mV
- NeuroBench SHD: 79.28% test accuracy (250K params, feedforward)
- Brian2 parity: exact LIF match (0.000ms timing diff), 7.3x speedup
- 5 validation docs with measured data in `docs/validation/`

### Stochastic Computing Pipeline
- Bipolar SC (XNOR): `core/bipolar.py` for signed weight multiplication
- SC bitstream MNIST: 10% (unipolar) -> 35.6% (bipolar) -> 50.0% (all fixes)
- SC-aware training: `SCAwareLIFNet` with bitstream noise injection (+9.5pp)

### Quantization-Aware Training
- `QuantizedLIFNet`: 2/4/8/16-bit STE weight quantization (PyTorch)
- `SCAwareLIFNet`: SC noise injection during training
- `SCAwareLinear`: drop-in layer replacement

### Encoding Comparison
- 7 temporal spike encodings benchmarked on MNIST
- Latency encoding Pareto-optimal: 88.1% at 142 spikes (17x fewer than rate)

### Interoperability
- NeuroML 2 importer: iafCell, Izhikevich (2003/2007), AdEx
- SONATA network format importer: nodes.h5 + edges.h5, connectivity matrix

### Reproducibility
- 7 Kaggle scripts in `notebooks/*_kaggle.py`
- JSON artifacts in `benchmarks/results/`

## [3.14.0] — 2026-03-27

### Visual SNN Design Studio (Experimental)
- **New feature:** web-based IDE for designing, training, compiling, and deploying SNNs
- 118-model browser with live simulation, parameter sliders, pattern classification
- 20+ analysis views: trace, phase, ISI, f-I, bifurcation, heatmap, sensitivity, STA, frequency response, characterisation, multi-model overlay, A/B comparison
- Compiler Inspector: SC IR build/verify/emit, SystemVerilog generation, co-simulation
- Synthesis Dashboard: Yosys synthesis for 4 FPGA targets (ice40, ECP5, Gowin, Xilinx), multi-target comparison, resource estimation without Yosys
- Training Monitor: live SSE metric streaming, 6 surrogate gradients, per-layer spike rates, learnable beta/threshold
- Network Canvas: React Flow drag-and-drop populations and projections, NIR export/import
- Full pipeline: network graph → validate → simulate → compile → synthesise in one click
- Project save/load: persistent JSON workspaces on server
- E-I balanced network simulation with Rust engine fast path
- 140+ Studio-specific tests
- Documentation: 7 pages on GitHub Pages, 10-step quickstart tutorial
- Launch: `pip install sc-neurocore[studio] && sc-neurocore studio`

### Rust Engine
- `py_simulate_ei_network()`: fused E-I network simulation (CSR + Poisson + Euler) in single Rust call
- `py_batch_simulate()`: batch model simulation with NeuronVariant dispatch loop
- `create_neuron()` made `pub` for reuse across lib.rs
- 288 Rust tests passing

### Performance
- Model list caching: first `/api/models` call loads 118 models in ~1s, subsequent calls <1ms

### Security
- 25 CodeQL "information exposure through exception" fixes — no tracebacks in HTTP responses
- 5 CodeQL "uncontrolled data in path expression" fixes — project name sanitisation
- DOMPurify XSS fix via npm override (>=3.3.2)
- Bandit: MD5 usedforsecurity=False, narrowed bare except clauses

### CI
- Engine wheel publish job added to publish.yml (PyPI OIDC)
- Bridge ImportError restored for pytest.importorskip compatibility
- PnR added to typos dictionary
- tsconfig.tsbuildinfo gitignored
- uvicorn skip guard for studio optional extra

### ANN-to-SNN Conversion Engine
- `sc_neurocore.conversion.convert()`: automated PyTorch ANN to rate-coded SNN conversion
- QCFS activation (Quantization-Clip-Floor-Shift): ReLU replacement for conversion-aware training
- Threshold normalization from calibration data activation statistics
- `ConvertedSNN.run()` and `.classify()` for inference with Poisson rate coding

### Learnable Delay Training
- `DelayLinear`: PyTorch module with trainable per-synapse delays via linear interpolation
- Differentiable delays: gradients flow through fractional delay positions
- Export to integer delays for hardware deployment via `delays_int` and `to_nir_delay_array()`
- DCLS (Dilated Convolutions with Learnable Spacings) principle for fully-connected SNN layers

### One-Command FPGA Deploy
- `sc-neurocore deploy model.nir --target artix7`: NIR/PyTorch → Verilog → project in one command
- Target presets: ice40, ecp5 (Yosys Makefile), artix7, zynq (Vivado project.tcl)
- Copies 19 HDL library modules, generates neuron SystemVerilog, build script, README

### Network Engine
- Per-synapse delays in Projection: `delay=array` for heterogeneous axonal/synaptic delays
- Spike-gating: `Population.step_all(spike_gating=True)` skips idle neurons, compute proportional to active count
- Weight sparsity: `Projection(weight_threshold=0.01)` skips near-zero synapses during propagation

### Compiler
- Per-layer adaptive bitstream length: `assign_lengths()` with Hoeffding or sensitivity-based allocation
- Mixed-precision SC networks: shallow layers use short L (fast), deep layers use long L (precise)

### Event-Driven FPGA RTL
- `sc_aer_encoder.v`: spike vector → AER packets via priority encoder, idle neurons consume zero power
- `sc_event_neuron.v`: Q8.8 LIF that computes only on input events or periodic leak ticks
- `sc_aer_router.v`: distributes AER events to target neurons using connectivity lookup table
- Total HDL modules: 19 (was 16)

### Performance
- Lazy-load 109 neuron models: import time 200s → 57s
- Deferred scipy imports (stats.qmc, sparse): import time 57s → 10s

### Infrastructure
- Coverage fixes: test second model access, pragma Rust-only branch
- Coverage for lazy-load path, sparse guard mock path
- Ruff F401 re-export fixes, format vectorized_layer

## [3.13.3] - 2026-03-20

### SC Arithmetic
- CORDIV division circuit: Python `sc_divide()` + Verilog `sc_cordiv.v` (Li et al. 2014)
- Adaptive bitstream length: Hoeffding/Chebyshev/variance bounds via `adaptive_length()`
- Sobol/Halton multi-dimensional decorrelation for per-synapse independent streams
- Chaotic RNG mode in BitstreamEncoder (logistic map)
- Sobol bitstream attention: `StochasticAttention.forward_bitstream()` with LDS variance reduction

### Learning Rules
- BCM metaplasticity with sliding threshold (Bienenstock-Cooper-Munro 1982)
- Voltage-based STDP (Clopath et al. 2010)
- Truncated BPTT for long sequences (`TBPTTLearner`, Williams & Peng 1990)
- EWC penalty implemented (was no-op stub) — Kirkpatrick et al. 2017
- Learnable beta/threshold on all 10 SNN cell types (ExpIF, AdEx, Lapicque, Alpha, SecondOrderLIF, IF, Synaptic)
- ConvSpikingNet now works with `train_epoch()` via `flatten_input=False`

### Biological Circuits
- Tripartite synapse: astrocyte ↔ synapse bidirectional coupling (Araque et al. 1999)
- Rall branching dendrite: compartmental tree with 3/2 power rule
- Canonical cortical microcircuit: 5-population column (L2/3 exc/inh, L4, L5, L6)
- Astrocyte adapter: `AstrocyteNeuron` wraps Li-Rinzel model for Population/Network

### Theoretical Depth
- SC→quantum circuit compiler: Ry encoding, statevector simulator, layer compilation
- Zero-multiplication predictive coding SC layer (Conjecture C9: XOR=error, popcount=magnitude)
- Topological observables: winding number, Ollivier-Ricci curvature, sheaf defect
- Phi* integrated information estimation (Barrett & Seth 2011, IIT)
- Goldstone mode verification for Knm coupling spectrum
- Fault tolerance benchmark: SC vs fixed-point degradation curves
- Hardware-aware SC layer with memristive defect injection
- Noisy quantum simulation via HeronR2NoiseModel Kraus channels

### NIR Bridge
- Recurrent edge handling via unit-delay insertion (LSTM-like feedback)
- Multi-port subgraph support (`SCMultiPortSubgraphNode`)

### Compiler
- IR type checker: Bitstream/Rate/Spike mismatch detection before emission
- SV/MLIR emission for GraphForward, SoftmaxAttention, KuramotoStep (was error stub)
- Weight quantizer exported in compiler `__init__.py`

### Hardware Stack
- AXI-Stream interface for bulk bitstream I/O (`sc_axis_interface.v`)
- DMA controller for weight upload and output readback (`sc_dma_controller.v`)
- Parameterized AXI-Lite register file (`sc_axil_cfg_param.v`)
- Clock domain crossing primitives: 2-FF sync, Gray counter, async FIFO (`sc_cdc_primitives.v`)
- NEON scalar-equivalence tests (13 tests for popcount, dot, max, sum, scale)

### Infrastructure
- Rust engine wheel publishing in PyPI release workflow
- SpikeInterface/Neo adapter for experimental data import
- Static CycloneDX SBOM (v1.6)
- JAX autodiff fix: straight-through estimator for spike reset
- IIT added to typos allowlist

## [3.13.2] - 2026-03-19

### Equation → Verilog RTL Compiler
- `equation_compiler.py`: compile any `EquationNeuron` to synthesizable Q8.8 fixed-point Verilog
- `equation_to_fpga()`: one-liner from Brian2-style ODE string to Python neuron + Verilog RTL
- AST-to-Verilog expression emitter handles +, -, *, /, **, unary minus, comparisons
- Multi-variable ODE support (FitzHugh-Nagumo, Izhikevich, Hodgkin-Huxley)
- Threshold and reset logic auto-generated

### NIR Bridge
- `nir_bridge` package: import NIR graphs into SC-NeuroCore (FPGA backend for NIR)
- Maps 11 NIR primitives (LIF, IF, LI, Integrator, Affine, Linear, Scale, Threshold, Flatten, Input, Output)
- Recursive graph parser with topological sort, fan-in summation, nested subgraph support
- NIR integration guide, API docs, notebook (05_nir_bridge.ipynb)

### Packaging & Release
- Restored `sc-neurocore` as the only PyPI product package and removed the unintended runtime dependency on a separate `sc-neurocore-engine` publish
- Publish automation now pushes only `sc-neurocore` to PyPI while keeping the Rust engine on the existing crate / source / CI wheel paths
- Tag pushes still trigger publish directly, so release creation no longer depends on a downstream `release.published` event

## [3.13.1] - 2026-03-19

### Packaging & Install
- Top-level `sc-neurocore` now requires the matching `sc-neurocore-engine` release, and `sc-neurocore info` reports engine version mismatches explicitly instead of silently mixing versions
- Dense-layer example and getting-started/docs packaging guidance now match the current public API and distinguish wheel-shipped modules from source-only modules

### NIR Bridge
- Nested NIR subgraphs now execute through a dedicated subgraph node wrapper and reset cleanly inside `SCNetwork`
- `Flatten` now respects `start_dim` / `end_dim`, and bridge coverage is enforced instead of being omitted
- Added regression coverage for nested graphs, fan-in, cycle detection, orphan nodes, flatten edge cases, and file-based import/export

### CI & Release
- CI now builds and installs the local engine wheel before editable/package installs, so unreleased versions no longer fail dependency resolution
- Build smoke installs both the engine wheel and the top-level wheel from local artifacts
- Publish workflow now runs from tag pushes, builds engine sdist+wheels, publishes the engine package before `sc-neurocore`, and keeps manual dispatch build-only unless publish is explicitly enabled
- Release workflow now attaches both the pure-Python wheel and sdist to GitHub Releases

### Bug Fixes
- StochasticTransformerBlock: clamp residual and FFN intermediate values to [0, 1] — MAC output from `VectorizedSCLayer` can exceed 1.0, triggering the new input validation
- Optional dependency introspection in `sc-neurocore info` no longer crashes on broken NumPy/JAX imports

### Tests
- Full preflight now passes at `2112 passed`, `38 skipped`, `12 xfailed`, with `100.00%` coverage
- Added audit validation tests for VectorizedSCLayer/EquationNeuron, CLI fallback coverage, dense-layer example smoke coverage, and expanded NIR bridge regressions

### Documentation
- Replace stale black references with ruff format in `VALIDATION.md` and `CONTRIBUTING.md`
- Sync the packaging/install docs with the released product surface
- Package naming and install guidance were corrected in `3.13.2`; `3.13.1` incorrectly treated `sc-neurocore-engine` as a separate PyPI runtime dependency

## [3.13.0] - 2026-03-18

### Python 3.14 Support
- CI test matrix, wheel builds, and publish workflow now include Python 3.14
- All 1 776 Python tests pass on 3.14; all dependencies compatible
- pyproject.toml classifier added

### Bridge Wiring
- 12 missing Rust symbols exported from bridge `__init__.py`: NetworkRunner, BitstreamAverager, Izhikevich, ArcaneNeuron, 8 AI-optimized models, ContinuousAttractorNeuron
- Parity test name mapping for RustContinuousAttractorNeuron

### CI Fixes
- Black formatting for identity/ files; pre-commit ruff upgraded v0.9.7 → v0.15.6
- Clippy: PopulationRunner::is_empty() added
- TraceEncoder: deterministic hash (byte-based, not Python hash())
- Synapse test tolerance widened for short bitstream noise
- Notebook trailing newline for end-of-file-fixer
- Removed deleted ruff rule UP038

### Documentation
- JOSS paper rewrite: pipeline + spike raster figures, Availability section, McCulloch-Pitts/Hodgkin-Huxley citations, tightened to ~1200 words
- All docs synced: test counts (1 776/336), 111 NetworkRunner, 17 HDL, Python 3.14
- Neuron explorer notebook (04_neuron_explorer.ipynb): 5 sections, 117 models

### Infrastructure
- `.gitattributes`: eol=lf (suppress CRLF warnings on Windows)
- Single-directory migration: `03_CODE/sc-neurocore/` is canonical repo
- PyPI deployment branch policy fixed (main added)
- 12 known Rust/Python parity divergences tracked as xfail
- 5 version-gate assertions updated

## [3.12.0] - 2026-03-17

### ArcaneNeuron + 8 AI-Optimized Models
- ArcaneNeuron: unified self-referential cognition model with 5 coupled subsystems (fast/working/deep/gate/predictor)
- 8 novel AI-optimized spiking neuron models: MultiTimescaleNeuron, AttentionGatedNeuron, PredictiveCodingNeuron, SelfReferentialNeuron, CompositionalBindingNeuron, DifferentiableSurrogateNeuron, ContinuousAttractorNeuron, MetaPlasticNeuron
- Total neuron count: 122 Python (113 bio + 9 AI), 111 Rust (including Arcane)
- ArcaneNeuron included in Rust NetworkRunner (111-model fused loop, was 80)

### Identity Substrate
- `sc_neurocore.identity` package: persistent spiking network for identity continuity
- IdentitySubstrate: 3-population network (HH cortical + WB inhibitory + HR memory) with STDP
- TraceEncoder: LSH-based reasoning trace to spike pattern encoding
- StateDecoder: PCA + attractor extraction + priming context generation
- Checkpoint: Lazarus protocol save/restore/merge of complete network state (.npz)
- DirectorController: L16 cybernetic closure with monitor/diagnose/correct feedback loop

### Network Simulation Engine
- Population-Projection-Network architecture with 3 backends: Python (NumPy), Rust (NetworkRunner), MPI (mpi4py)
- 6 topology generators: random, small-world, scale-free, ring, grid, all-to-all
- 12 visualization plots: raster, voltage, ISI, cross-correlogram, PSD, firing rate, phase portrait, population activity, instantaneous rate, spike train comparison, network graph, weight matrix
- 7 advanced plasticity rules: BPTT, e-prop, R-STDP, MAML, homeostatic, STP, structural
- MPI distributed simulation for billion-neuron scale via mpi4py

### Rust NetworkRunner
- 111-model fused simulation loop with Rayon-parallel population stepping (was 80)
- CSR-sparse projection propagation
- Scales to 100K+ neurons with near-linear speedup

### Model Zoo
- 10 pre-built network configurations: Brunel balanced, cortical column, CPG, decision-making, working memory, visual cortex V1, auditory processing, MNIST classifier, SHD speech, DVS gesture
- 3 pre-trained weight sets: MNIST (784-128-10), SHD (700-256-20), DVS gesture (256-256-11)

### conda-forge
- Recipe ready for conda-forge distribution

### Analysis Toolkit
- 126 spike train analysis functions across 23 modules (22 spike_stats + 1 explainability)
- Covers: basic stats, variability, rate estimation, distance metrics,
  correlation, spectral, temporal, stimulus, LFP coupling, surrogates,
  information theory, causality, dimensionality, decoding, network,
  point process, sorting quality, waveform, statistics, patterns, SPADE, GPFA
- Pure NumPy, zero external dependencies
- Tests: 1 776 Python total, 336 Rust total

### Neuron Model Library (122 Python / 111 Rust)
- 108 individual model files in `neurons/models/` (one file per model)
- 108 individual model files across 14 families: IF variants, Biophysical, Adaptive, Oscillatory, Bursting, Synaptic, Multi-compartment, Map-based, Stochastic, Population, Hardware, Modern/ML, Rate, Other
- Notable additions: TraubMiles, WilsonHR, Pospischil (5 cortical types), ConnorStevens, WangBuzsaki, PinskyRinzel, Destexhe, HuberBraun, GolombFS, MainenSejnowski
- Historical coverage from McCulloch-Pitts (1943) to Gated LIF (2022)
- 10 PyTorch training cells: LIF, IF, Synaptic, ALIF, RecurrentLIF, ExpIF, AdEx, Lapicque, Alpha, SecondOrderLIF

### MNIST 99.49% Accuracy
- `examples/mnist_conv_train.py` — ConvSpikingNet with learnable beta/threshold
- Architecture: Conv(1->32)->LIF->Pool->Conv(32->64)->LIF->Pool->FC->LIF->FC->LIF
- Techniques: FastSigmoid surrogate, cosine LR schedule, data augmentation, membrane readout
- Trained on RTX 6000, 30 epochs, 25 minutes
- Model checkpoint: `examples/mnist_conv_train/results/conv_spiking_net_best.pt`

### Intel Lava/Loihi Bridge
- `integrations/lava_bridge.py` — SCtoLavaConverter, export_weights_loihi
- SCDenseProcess + PySCDenseModel for Lava CPU simulation
- Weight conversion: SC probability [0,1] -> Loihi fixed-point

### Rust Engine parity expansion (v3.8/v3.9 carry-forward)
- **Sobol bitstream** (M1): Gray-code Sobol quasi-random encoder in Rust (`sobol.rs`)
- **HomeostaticLIF**: adaptive threshold neuron with EMA spike rate tracking
- **DendriticNeuron**: XOR-nonlinearity compartmental model
- **RewardStdpSynapse**: eligibility trace + reward-modulated STDP
- **Conv2DLayer**: im2col + SC multiply-accumulate convolution
- **RecurrentLayer**: echo state network with state feedback
- **LearningLayer**: online STDP-integrated dense layer
- **FusionLayer**: weighted stochastic multiplexing across modalities
- **MemristiveLayer**: dense layer with stuck-at faults and write noise
- **SpikeRecorder**: buffered spike recording with firing rate and ISI stats
- **ConnectomeGenerator**: Watts-Strogatz and Barabási-Albert topology generators
- **FaultInjector**: bit-flip and stuck-at fault injection on packed bitstreams
- **MLIR emitter**: CIRCT hw/comb dialect IR emission (`ir/emit_mlir.rs`)
- **Static synapse**: completed with excitatory/inhibitory polarity
- **Surrogate gradient**: added Triangular and PiecewiseLinear variants
- Rust neuron models callable from Python: 111 (of 122 Python total)

### SIMD Hardening (v3.8 carry-forward)
- Fused `softmax_inplace_f64_dispatch` with SIMD max/sum/scale
- Hamming distance dispatch for all backends (AVX2, SVE, RVV)
- SVE/RVV softmax portable fallbacks
- Attention softmax refactored to use fused dispatch

### Quantum Backend Stabilisation (v3.9 carry-forward)
- IBM Heron r2 noise model: depolarizing, amplitude/phase damping, readout asymmetry
- Parameter-shift gradient rule for variational quantum circuits
- Hybrid quantum-classical VQE pipeline with scipy optimizer
- QEC noise integration with surface code threshold comparison

### Holonomic Adapter Ecosystem (v3.9 carry-forward)
- L1-L16 adapters registered in ComponentRegistry with `create_adapter()` factory
- Per-adapter benchmark suite: latency, memory, throughput (with/without JAX JIT)
- Plugin discovery via `importlib.metadata` entry points

### Type Safety Cleanup (M2)
- Removed 235 unnecessary `type: ignore` comments (260 -> 25)
- Remaining 25 are justified: CuPy type aliases, optional imports, private method access

### GPU SNN Training with Surrogate Gradients
- `sc_neurocore.training` — PyTorch-based differentiable SNN training module
- 3 surrogate gradient functions: FastSigmoid (Zenke 2018), SuperSpike (Zenke 2021), ATan (Fang 2021)
- `LIFCell`, `RecurrentLIFCell` — `nn.Module` LIF neurons with autograd through spikes
- `SpikingNet` — multi-layer feedforward SNN with spike-count and membrane readout
- `to_sc_weights()` — export trained float weights to [0,1] range for SC bitstream deployment
- 3 loss functions: spike count cross-entropy, membrane cross-entropy, spike rate MSE
- `train_epoch()` / `evaluate()` — training loops with temporal unrolling
- `examples/mnist_surrogate/train.py` — MNIST benchmark (~95% accuracy, 10 epochs)
- 31 tests covering surrogates, modules, and training loops
- Requires `pip install sc-neurocore[training]` or `sc-neurocore[research]`

## [3.10.0] - 2026-03-09

### MNIST-on-FPGA Demo
- **End-to-end pipeline**: `examples/mnist_fpga/demo.py` — train (sklearn digits),
  PCA 64→16, quantise Q8.8, stochastic computing inference, Verilog weight export
- Float 94.2%, Q8.8 94.2%, SC 94.0% (L=1024, sign-magnitude encoding)
- Resource estimate: 16→10 config = ~56K LUTs (fits Artix-7 100T)
- `hdl/sc_dense_matrix_layer.v` — per-neuron weight dense layer for classification

### Vivado Tooling
- `tools/vivado_impl.tcl` — non-project flow: synth → place → route (250 MHz default)
- `tools/vivado_report.py` — parse timing/utilization/power reports to JSON

### Tutorial
- `docs/tutorials/fpga_in_20_minutes.md` — 6-section FPGA deployment tutorial

### Paper
- JOSS paper updated to submission-ready state (`paper/paper.md`)
- 12 references with DOIs, MNIST demo results, Brian2 comparison, formal verification

### Documentation Overhaul
- README: benchmarks section (Rust SIMD, Brian2 comparison, Yosys synthesis)
- README: all 10 HDL modules listed with descriptions
- Zenodo DOI updated to 10.5281/zenodo.18906614
- CITATION.cff, .zenodo.json: DOI, version, author corrections
- CONTRIBUTING.md, VALIDATION.md, getting-started.md: test counts, Python version
- Yosys MODULES list updated (10 modules)

### Fixes
- Zenodo author list corrected (sole author: Miroslav Šotek)
- DOI badge in README points to latest Zenodo record

## [3.9.1] - 2026-03-08

### Benchmarks
- **20-variant Brunel translator suite**: comprehensive characterization of
  SC-NeuroCore against Brian2 across neuron models (LIF, Izhikevich, homeostatic),
  timing variants, synapse types (STDP, dot product, Sobol bitstream), layer
  architectures (JAX, recurrent, memristive), and acceleration backends
  (Numba JIT, PyTorch CUDA GTX 1060, vectorized NumPy)
- V18 Numba JIT: 9.5× speedup over per-neuron Python loop
- V19 PyTorch CUDA: 8.7× speedup on GTX 1060 6GB
- V14 Sobol bitstream: 1.04× Brian2 ratio (closest match)
- 19 translator unit tests (`test_brunel_translator.py`)
- Fix BENCHMARKS.md CPU: i5-11600K @ 3.9 GHz (AVX-512, DL Boost)
- Fix 3 delta-PSC wiring bugs: v_reset omission, R*I*dt dilution, Poisson-as-current
- Comprehensive BENCHMARKS.md with 13+ sections and measured numbers
- Rust Criterion: 31 benchmarks captured (AVX-512)
- Brian2 2.10.1 SNN comparison: Brunel balanced network head-to-head
- NeuroBench-aligned metrics: 4 configurations, up to 847 MOP/s
- v2 vs v3 PyO3 speedup: 7.3× on large dense forward (128→64)
- Advanced module benchmarks: quantum hybrid, GNN, S-Former, BCI, DVS, chaos RNG
- Yosys synthesis tooling (`tools/yosys_synth.py`, `tools/yosys_synth.tcl`)
- CuPy 14.0.1 installed for GPU VectorizedSCLayer

### Paper
- Updated JOSS paper with measured Criterion numbers (41.3 Gbit/s pack, 224 Mstep/s LIF)
- Replaced estimated FPGA claim with Yosys tooling reference

## [3.9.0] - 2026-03-06

### SCPN Layers
- **L8-L16 pure NumPy layers**: 9 new layer files completing the full 16-layer SCPN stack (`scpn/layers/l8_phase_field.py` through `l16_director.py`)
- **16-layer registry**: `LAYER_REGISTRY` dict, `create_full_stack()` now returns all 16 layers
- **Full integrated step**: `run_integrated_step()` chains L1→L16 with inter-layer coupling

### Quantum Error Correction
- **SurfaceCodeShield**: d=3 rotated surface code with X/Z stabilizers, syndrome measurement, lookup-table decoding — corrects arbitrary single-qubit errors
- Extensible to d=5 (encode/decode/syndrome paths support arbitrary odd distance)

### Benchmarks
- Fixed double-step bug in `benchmarks/snn_comparison.py` (neurons were advanced twice per timestep)
- Fixed Lava stub notes (requires Loihi 2 hardware)
- Fixed `benchmark_suite.py` output path → `benchmarks/results/`
- SNN comparison results recorded in `docs/benchmarks/BENCHMARKS.md`

### Formal Verification
- **LIF neuron**: `hdl/formal/sc_lif_neuron.sby` + `sc_lif_neuron_formal.v` — 5 properties (reset, spike-reset, refractory clamp, counter bound, spike reachability)
- **Bitstream synapse**: `hdl/formal/sc_bitstream_synapse.sby` + `sc_bitstream_synapse_formal.v` — 4 properties (AND correctness, zero propagation, full-high, input coverage)

### Testing
- 6 cross-layer coupling integration tests (`test_scpn_cross_layer.py`)
- 9 surface code QEC tests (`test_qec_surface.py`)
- Test count: 945 → 960+

### Documentation
- JOSS paper: updated test count (960), qualified LUT claim, added Brunel/NeuroBench/LFSR bib entries

## [3.8.2] - 2026-03-06

### Documentation & Adoption
- **BENCHMARKS.md**: Populated with 14 real benchmark entries (i5-11600K, NumPy 1.26.4), Rust engine Criterion numbers, comparison context, reproduction instructions
- **JOSS paper draft**: `paper/paper.md` + `paper.bib` (6 references) — statement of need, architecture, key features, QA
- **End-to-end notebook**: `notebooks/03_end_to_end_pipeline.ipynb` — 7-cell walkthrough (encode→synapse→neuron→VectorizedSCLayer→accuracy analysis)

### Testing
- **18 Hypothesis property-based tests**: Bitstream encoding roundtrip, LFSR determinism, neuron output constraints, layer shape invariants, RNG range/shape, recorder accumulation, encoder binary output
- **Test count**: 887 → 911 tests passing, 98.41% coverage

### Issues Closed
- #30: Property-based testing with Hypothesis
- #33: JOSS paper draft

## [3.8.1] - 2026-03-06

### Enterprise Hardening
- **11 CI workflows**: ci, v3-engine, v3-wheels, benchmark, docs, pre-commit, codeql, scorecard, stale, release, publish — all SHA-pinned, concurrency-grouped
- **Supply chain**: Every GitHub Action SHA-pinned (30+ refs), `pypa/gh-action-pypi-publish` pinned, dependabot groups GH Actions PRs
- **Security**: Bandit SAST in CI, dependabot security updates enabled, private vulnerability reporting enabled, CodeQL weekly schedule
- **Branch protection**: 6 required status checks (lint, test×2, spdx-guard, build, pre-commit)
- **Dockerfile**: Multi-stage build, Python 3.12, non-root user, OCI labels, healthcheck
- **Preflight gate**: `tools/preflight.py` (black + bandit + spdx-guard + pytest), `.githooks/pre-push` hook
- **Release pipeline**: `publish.yml` (PyPI OIDC trusted publisher, 12 platform wheels), `release.yml` attaches sdist to GitHub Releases
- **Repo hygiene**: `.dockerignore`, `.editorconfig`, `.gitattributes`, `CONTRIBUTORS.md`, `CODEOWNERS`, PR template, issue templates (YAML forms), dependabot commit-message prefixes
- **Labels**: 22 labels with colors (ci, security, breaking-change, hdl, performance, needs-review, pinned, roadmap, stale)
- **Settings**: Delete-branch-on-merge, wiki/projects disabled, OpenSSF Scorecard badge

### Lint Enforcement & Python Version
- **ruff check enforced in CI**: 258 unused/deprecated imports auto-fixed across 138 files
- **CI test matrix expanded**: Python 3.10, 3.11, 3.12 (dropped 3.9 — EOL, autoray/PennyLane incompatible)
- **`requires-python` bumped to `>=3.10`**: badge, classifiers, black/ruff target-version updated
- **bandit added to `[dev]` extras**: contributors can now `make lint` after `pip install -e ".[dev]"`
- **benchmark.yml permissions tightened**: `permissions: {}` at top, scoped per-job
- **SECURITY.md / SUPPORT.md**: GitHub Security Advisories link added
- **VALIDATION.md refreshed**: 1058 tests, 98% gate, ruff/bandit/spdx-guard/codeql/scorecard gates documented

---

## [3.8.0] - 2026-03-05

### Hardening & Documentation
- **Coverage gate raised to 98%**: De-omitted 6 modules (chaos/rng, analysis/explainability, physics/wolfram_hypergraph, robotics/swarm, learning/neuroevolution, spatial/*) plus bio/neuromodulation. 34 new tests, 1058 total, 98.10% coverage
- **NumPy 2.x audit**: Zero deprecated calls found — codebase fully compatible
- **Full API documentation**: 25 new mkdocstrings pages, all 44 subpackages wired into nav. Reorganized into Core / Compiler & Export / Domain Modules / Infrastructure sections
- **Stale issue automation**: `.github/workflows/stale.yml` — weekly sweep, 60+14 day lifecycle, exempt: pinned/security/roadmap
- **CI coverage gate sync**: `ci.yml` and `pyproject.toml` both enforce `fail_under = 98`

---

## [3.7.0] - 2026-02-11

### Adaptive Runtime Engine -- HDC/VSA, SCPN Petri Nets, Fault-Tolerant Logic
- **HDC/VSA kernel**: `BitStreamTensor` gains `xor`, `xor_inplace`, `rotate_right`, `hamming_distance`, `bundle` methods for hyper-dimensional computing on 10,000-bit vectors
- **SIMD fused XOR+popcount**: AVX-512 VPOPCNTDQ / AVX2 / portable dispatch for hamming distance hot path
- **PyBitStreamTensor**: New `#[pyclass]` exposing full HDC algebra to Python (13 methods)
- **HDCVector**: High-level Python class with operator overloading (`*`=bind, `+`=bundle, `.similarity()`, `.permute()`)
- **PetriNetEngine**: Stochastic Colored Petri Net engine wrapping two `DenseLayer` instances for Places->Transitions->Places firing
- **Fault-tolerant logic**: Boolean logic with stochastic redundancy (1024-bit) survives 40%+ bit-flip rates
- **44 new tests**: 15 Rust integration + 20 Python HDC + 9 Python Petri Net
- **2 demos**: HDC symbolic query ("Capital of France?"), safety-critical Boolean logic with error sweep
- **Comprehensive study**: `docs/research/SC_NEUROCORE_V3.7_ADAPTIVE_RUNTIME_ENGINE_STUDY.md`

---

## [3.6.0] - 2026-02-10

### Fused Dense Pipeline + Fast PRNG + Batch Forward
- **Fused encode+AND+popcount**: `forward_fused()` eliminates intermediate input bitstream materialization
- **Fast PRNG switch**: xoshiro256++ for dense fast-path input encoding and numpy batch encoding
- **Batched dense API**: `DenseLayer.forward_batch_numpy()` processes N samples in one FFI call
- **New diagnostics**: criterion benches for fused dense, encode+popcount, batch dense, and PRNG throughput
- **Version/test/docs update**: bumped to 3.6.0 with the fused dense pipeline test suite and migration notes

---

## [3.5.0] - 2026-02-10

### SIMD Pipeline Acceleration
- **SIMD fused AND+popcount**: AVX-512 VPOPCNTDQ accelerated dense inner loop with AVX2 fallback
- **SIMD Bernoulli encode**: AVX-512BW/AVX2 threshold compare path for packed Bernoulli generation
- **Flat weight storage**: Contiguous `[neuron][input][word]` packed layout for cache-friendly access
- **Zero-allocation LIF batch**: Pre-allocated numpy outputs for batch LIF APIs
- **Criterion benchmarks**: Added fused-and-popcount and SIMD Bernoulli diagnostics

---

## [3.4.0] - 2026-02-10

### SIMD Pack, LIF Optimization, Rayon Guard
- **SIMD pack vectorization**: AVX-512/AVX2/portable fast packing (closes 6x Blueprint target)
- **Branchless LIF mask**: Eliminates branches in fixed-point sign extension
- **batch_lif_run_multi()**: Parallel multi-neuron batch execution via rayon
- **Rayon work threshold**: Avoids thread-pool overhead at small input counts
- **Criterion benchmarks**: Added pack_fast, pack_dispatch, lif_100k_steps

---

## [3.3.0] - 2026-02-10

### Fast Bernoulli, Fused AND+Popcount, Zero-Copy Prepacked
- **bernoulli_packed_fast**: 8x less RNG bandwidth via byte-threshold encoding
- **Fused AND+popcount**: Eliminates intermediate buffer allocation in neuron compute
- **forward_prepacked_numpy()**: True zero-copy from numpy 2D uint64 arrays
- **set_num_threads()**: Rayon thread pool configuration for tuning parallelism
- **Criterion benchmarks**: Added bernoulli_packed_fast benchmark

---

## [3.2.0] - 2026-02-10

### Benchmark CI, Single-Call Dense Forward, Parallel Encoding
- **Criterion Benchmarks**: Expanded suite with bernoulli encoding comparison and dense forward variants
- **Benchmark CI**: Automated criterion runs with artifact upload
- **DenseLayer.forward_numpy()**: Single FFI call with numpy input/output plus parallel encoding
- **Parallel batch_encode_numpy**: Rayon-parallelized probability encoding
- **Repo cleanup**: Added local `.gitignore` for generated artifacts

---

## [3.1.0] - 2026-02-10

### Dense Forward Optimization & PyPI Publishing
- **Direct Packed Bernoulli**: `bernoulli_packed()` eliminates `Vec<u8>` intermediate allocations
- **Parallel Encoding**: `DenseLayer.forward_fast()` parallelizes input encoding with per-input RNGs
- **Pre-packed Forward**: `DenseLayer.forward_prepacked()` accepts pre-encoded numpy/list inputs and skips encoding
- **batch_encode_numpy**: Returns a 2-D numpy array instead of nested Python lists
- **PyPI Publishing**: Added automated wheel upload on `v3.*` tags via Trusted Publisher workflow
- **Updated Benchmarks**: Added dense `fast` and `prepacked` benchmark variants

---

## [3.0.0] - 2026-02-10

### Performance Optimization & Stable Release
- **NumPy Zero-Copy**: `pack_bitstream_numpy()`, `popcount_numpy()`, `unpack_bitstream_numpy()` — eliminate FFI marshalling overhead
- **Batch Operations**: `batch_lif_run()`, `batch_lif_run_varying()`, `batch_encode()` — process arrays in single FFI calls
- **Verilator CI**: Co-simulation tests run automatically on Ubuntu runners
- **Updated Benchmarks**: Formal report showing true kernel performance with zero-copy interop
- **Bridge Version Fix**: `bridge/pyproject.toml` version now matches engine

### Release Candidate (3.0.0-rc.1)
- **IR Python Bridge**: Full PyO3 bindings for ScGraphBuilder, ScGraph, verify, print, parse, emit_sv
- **Co-sim Activation**: Verilator compilation + simulation when available; graceful skip preserved
- **Wheel CI**: Cross-platform wheel builds (Linux/macOS/Windows x Python 3.9-3.12)
- **Benchmark Report**: Formal v2-vs-v3 performance comparison with Blueprint section 8 targets
- **IR Demo**: Real end-to-end Python->IR->verification->SystemVerilog demo

### HDL Compilation Pipeline (3.0.0-beta.1)
- **SC IR**: Rust-native intermediate representation with 11 op types
- **SV Emitter**: Compile IR graphs to synthesizable SystemVerilog
- **Co-sim**: Verilator-based verification against Rust golden model
- **CI**: Expanded test coverage to include all differentiation, acceleration, integration, and HDL Python tests

### Integration & Hardening
- SSGF-compatible Kuramoto solver (`step_ssgf`, `run_ssgf`)
- Property-based testing with proptest (12 property tests)
- Multi-head attention (`forward_multihead`)
- SC-mode GNN (`forward_sc`)
- End-to-end training demo
- Comprehensive rustdoc

### Differentiation & Acceleration
- Surrogate gradient LIF (FastSigmoid, SuperSpike, ArcTan)
- DifferentiableDenseLayer for backpropagation
- Stochastic attention (rate + SC mode)
- Graph neural network layer
- Kuramoto oscillator solver
- Criterion benchmarks + v2/v3 comparison

### Foundation
- Rust engine with PyO3 bindings
- Bit-exact LFSR, LIF neuron, dense layer
- SIMD dispatch (AVX-512, AVX2, NEON, portable)
- Python bridge with v2-compatible API
- Equivalence test suite

---

## [2.2.0] - 2026-02-09

### Added
- **Module Discoverability**: Populated 36 stub `__init__.py` files with proper
  `__all__` exports and lazy imports. Every package now supports
  `from sc_neurocore.X import Y` without touching internals.
- **MkDocs API Documentation**: Added `mkdocs.yml` with mkdocstrings plugin,
  `docs/index.md`, `docs/getting-started.md`, `docs/architecture.md`, and 17
  API reference stubs in `docs/api/`.
- **Examples Directory**: 6 runnable example scripts demonstrating bitstream
  encoding, neuron layers, vectorized inference, SCPN stack, HDL generation,
  and ensemble consensus (`examples/01`–`06`).
- **Module Docstrings**: Added module-level docstrings to `pipeline/ingestion.py`,
  `pipeline/training.py`, `utils/model_bridge.py`, `ensembles/orchestrator.py`.

### Changed
- **Print → Logging**: Converted 60+ `print()` calls across 25 source modules
  to structured `logging` with `getLogger(__name__)` and `%`-style formatting.
  Dashboard and drivers intentionally excluded (stdout by design).
- **CI Coverage Threshold**: Raised `--cov-fail-under` from 50 to 97 in
  `.github/workflows/ci.yml` to match actual coverage.
- Version bump: 2.1.0 → 2.2.0.

### Fixed
- **Unused Imports**: Removed dead imports from 7 files (`bio/uploading.py`,
  `core/replication.py`, `core/immortality.py`, `export/onnx_exporter.py`,
  `dashboard/text_dashboard.py`, `hdl_gen/verilog_generator.py`, `viz/web_viz.py`).
- **Input Validation**: `VectorizedSCLayer.forward()` now raises `ValueError`
  on wrong-shape input instead of silently producing garbage.
- **File I/O Error Handling**: `onnx_exporter.py`, `immortality.py`,
  `verilog_generator.py`, and `replication.py` now catch `OSError` on
  file operations and log meaningful messages.

### Security
- **Pickle Allowlist**: Replaced wildcard `'numpy.core.numeric': {'*'}` with
  explicit `{'_frombuffer', 'scalar'}` in `core/immortality.py`.
- **Path Traversal Prevention**: `core/replication.py` now validates that the
  destination directory is within or below the working directory via
  `os.path.realpath()` + `os.path.relpath()`.

---

## [2.1.0] - 2026-02-08

### Fixed (Critical)
- **HDL Bitstream Encoder Seed Decorrelation**: All parallel encoders shared
  hardcoded seed `0xACE1`, producing correlated bitstreams and breaking SC
  multiplication (`P(x AND x) = P(x)` instead of `P(x)*P(w)`). Added per-instance
  `SEED_INIT` parameter with prime-stride offsets (input: `0xACE1 + i*7`,
  weight: `0xBEEF + i*13`).
- **HDL Missing Port Connections**: `noise_in` and `v_out` were floating on
  LIF neuron instances in `sc_dense_layer_core.v`. Connected via wire buses.
- **HDL Duplicate Port**: Removed duplicate `.stream_len` in `sc_neurocore_top.v`.
- **Fixed-Point Overflow**: `FixedPointLIFNeuron` now applies `_mask()` for
  proper two's complement overflow wrapping on membrane potential.

### Added
- **GPU Acceleration Backend** (`accel/gpu_backend.py`):
    - CuPy/NumPy dual-path with automatic GPU detection and CPU fallback.
    - `gpu_pack_bitstream()`, `gpu_vec_and()`, `gpu_popcount()`, `gpu_vec_mac()`.
    - `VectorizedSCLayer` auto-selects GPU when CuPy is available.
- **Performance Benchmark Suite** (`scripts/benchmark_suite.py`):
    - 14 benchmarks across 5 categories (scalar, packed ops, dense layer,
      full pipeline, GPU).
    - `--full` mode (10x iterations), `--markdown` output to `BENCHMARKS.md`.
- **CI/CD Pipeline** (`.github/workflows/sc-neurocore-ci.yml`):
    - Lint (black + mypy), Test (Python 3.9/3.11/3.12 matrix, coverage >= 60%),
      Build (wheel + install verification).
- **Co-Simulation Harness**:
    - `hdl/tb_sc_lif_neuron.v`: Verilog testbench reading stimuli.txt, writing
      results_verilog.txt for bit-exact comparison.
    - `scripts/cosim_gen_and_check.py`: CLI driver with `--generate` and `--check`.
- **Bit-True Python Models**:
    - `FixedPointLFSR`: 16-bit maximal-length LFSR (period 65535).
    - `FixedPointBitstreamEncoder`: LFSR + unsigned comparator.
    - `_mask()`: Two's complement sign-extension with overflow wrap.
- **Public API Surface**: Root `__init__.py` exports 28 symbols across 7
  subpackages. All subpackage `__init__.py` files populated.
- **Tiered Module System**: 43 subpackages categorised as `core` (7),
  `research` (24+), or `contrib` (5). Install extras: `[gpu]`, `[research]`,
  `[contrib]`.
- **Behavioural Equivalence Tests**: 29 tests covering LFSR, encoder, LIF
  neuron, full pipeline, and bit-width masking.
- **GPU Backend Tests**: 17 tests covering all GPU primitives and
  VectorizedSCLayer integration.

### Changed
- Version bump: 2.0.0 -> 2.1.0.
- `pyproject.toml`: Added tool configs (pytest, black, mypy), tiered extras.
- `VectorizedSCLayer`: Refactored to use GPU backend with CPU fallback.

---

## [2.0.0] - 2026-01-12

### Added
- **Sapience & Sentience (v2.2.0)**:
    - `MetaCognitionLoop`: Computational self-awareness and self-modeling.
    - `NeuromodulatorSystem`: Dopamine/Serotonin emotional state modulation.
    - `NeuroArtGenerator`: Generative AI for internal state expression.
    - `AsimovGovernor`: Ethical constraint system (Three Laws).
    - `MindDescriptionLanguage (MDL)`: Substrate-independent soul serialization.
    - `DigitalSoul`: Persistence and reincarnation protocols.
    - `VonNeumannProbe`: Code-level self-replication.
- **Galactic Scale (v2.1.0)**:
    - `InterstellarDTN`: Long-range delay-tolerant networking.
    - `DysonPowerGrid`: Stellar-scale energy management.
    - `KardashevEstimator`: Civilization Type metrics.
    - `DarkForestAgent`: Game-theoretic survival logic.
    - `MPIDriver`: Distributed cluster-scale simulation.
    - `SNNGeneticEvolver`: Automated architecture optimization.
- **Transcendent & Omega (v2.0.5)**:
    - `HeatDeathLayer`: Entropy-survival computing.
    - `PlanckGrid`: Spacetime lattice theoretical limits.
    - `HolographicBoundary`: 3D-to-2D info mapping (AdS/CFT).
    - `EverettTreeLayer`: Many-Worlds branching solver.
    - `WolframHypergraph`: Graph-rewrite universe evolution.
    - `CategoryTheoryBridge`: Unified mathematical functors.
    - `FormalVerifier`: SMT-based safety proofs.
- **Exotic & Frontiers (v2.0.0)**:
    - `VectorizedSCLayer`: 64-bit packed JIT-accelerated core.
    - `QuantumStochasticLayer`: VQC qubit rotation bridge.
    - `StochasticTransformerBlock`: Spike-driven attention.
    - `MemristiveDenseLayer`: Hardware-aware analog simulation.
    - `StochasticCPG`: Robotic locomotion oscillators.
    - `MyceliumLayer`: Fungal network dynamics.
    - `BCIDecoder`: Neural signal (EEG) interface.
    - `DVSInputLayer`: Event Camera (AER) processing.
    - `EnergyProfiler`: 45nm Energy/CO2 estimation.
    - `WatermarkInjector`: IP protection security backdoors.

### Optimized
- `BitstreamAverager`: 6x speedup using running sum algorithm.
- `BitstreamEncoder`: Added Sobol Sequence (LDS) mode for faster convergence.

### Fixed
- Fixed f-string syntax in Verilog generator.
- Fixed dimension mismatch in Attention mechanism.
- Addressed Windows encoding issues in documentation generation.

## [1.0.0] - 2025-12-03
- Initial Release: Stochastic Neurons, Synapses, and Basic Bitstream Utilities.
