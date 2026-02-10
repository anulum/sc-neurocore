# SC-NeuroCore v3.0 — Phase 4 Code Review Report

**Reviewer**: Claude (Opus 4.6)
**Date**: 2026-02-10
**Phase**: 4 — HDL Compilation Pipeline
**Agent Under Review**: Codex (GPT-5)
**Handover Document**: `V3_PHASE4_CODEX_HANDOVER.md`
**Session Log**: `SESSION_LOG_2026-02-10_V3_MIGRATION_PHASE4.md`

---

## 1. Compliance Matrix

| Packet | Required Deliverables | Delivered | Status |
|--------|----------------------|-----------|--------|
| **N-0** CI Polish | Expand trigger paths, add Phase 3 tests to CI, accuracy metric in training demo | All 3 fixes delivered | PASS |
| **N** SC IR | 7 files in `engine/src/ir/` — graph.rs, builder.rs, verify.rs, printer.rs, parser.rs, emit_sv.rs, mod.rs | 7 files delivered (1,724 lines total) | PASS |
| **O** SV Emitter | `emit_sv.rs` mapping IR ops to HDL modules | Delivered within ir/ module (365 lines) | PASS |
| **P** Co-Sim | 4 Python files in `cosim/` with graceful Verilator skip | 4 files delivered (conftest.py + 3 test files) | PASS |
| **Q** WASM | Optional packet | Not delivered (as permitted by spec) | N/A |
| **R** Beta Release | Version bump, CHANGELOG, docs, example | All delivered | PASS |

### File Inventory Check

| Spec Requirement | Expected Count | Actual | Match |
|-----------------|----------------|--------|-------|
| New Rust source files | 7 | 7 | YES |
| New Rust test files | 2 | 2 | YES |
| New Python co-sim files | 4 | 4 | YES |
| Modified CI workflow | 1 | 1 | YES |
| Modified Rust files | 2 (lib.rs, Cargo.toml) | 2 | YES |
| Modified docs | 2 (v3_migration.md, CHANGELOG_V3.md) | 2 | YES |
| New example | 1 (02_ir_compile_demo.py) | 1 | YES |
| **Total new** | **14** | **14** | **YES** |
| **Total modified** | **6** | **6** | **YES** |

---

## 2. Packet-by-Packet Review

### Packet N-0: CI Polish — PASS

| Fix | Verified |
|-----|----------|
| CI trigger paths expanded to `tests/**`, `cosim/**`, `examples/**` | YES |
| Phase 3 tests added to CI step (kuramoto_ssgf, multihead_attention, gnn_sc_mode) | YES |
| Training demo updated with binary accuracy metric | YES |

### Packet N: SC Compute Graph IR — PASS

**Type System** (graph.rs):
- `ScType` enum: 7 variants (Bitstream, FixedPoint, Rate, UInt, SInt, Bool, Vec) — matches spec
- `ValueId(u32)`: SSA value reference with Copy semantics — correct
- `ScConst` enum: 5 constant variants (F64, I64, U64, F64Vec, I64Vec) — complete
- `LifParams` defaults: data_width=16, fraction=8, v_threshold=256, refractory_period=2 — matches `hdl/sc_lif_neuron.v`
- `DenseParams` defaults: n_inputs=3, n_neurons=7, stream_length=1024 — matches `hdl/sc_dense_layer_core.v`
- `ScOp` enum: 11 variants — matches spec exactly

**Operation Set**:

| Op | Spec'd | Delivered | Semantics |
|----|--------|-----------|-----------|
| sc.input | YES | YES | Named typed input port |
| sc.output | YES | YES | Named output forwarding |
| sc.constant | YES | YES | Typed constant embedding |
| sc.encode | YES | YES | Bernoulli bitstream encoding (length + seed) |
| sc.and | YES | YES | Bitwise AND (SC multiply) |
| sc.popcount | YES | YES | Hamming weight extraction |
| sc.lif_step | YES | YES | Fixed-point LIF with full LifParams |
| sc.dense_forward | YES | YES | Full pipeline with DenseParams |
| sc.scale | YES | YES | Scalar multiplication |
| sc.offset | YES | YES | Scalar addition |
| sc.div_const | YES | YES | Integer division |

**Builder** (builder.rs, 134 lines):
- Fluent API with `ScGraphBuilder::new()` → method chains → `build()`
- All 11 op types have builder methods — complete
- `build()` consumes builder (ownership transfer) — correct

**Verification** (verify.rs, 118 lines):
- 3 passes: SSA uniqueness, operand-before-use, cycle detection (DFS)
- Error accumulation (collects all violations, not early-exit) — good practice
- All 3 passes are sound; cycle detection is belt-and-suspenders (DAG property already implied by operand-before-use) but adds defense-in-depth

**Text Format** (printer.rs + parser.rs, 708 lines combined):
- Stable text serialization format: `sc.graph @name { ... }`
- Round-trip test passes (graph → text → parse → text → compare)
- Hex seed support (0xACE1 format) — correct

**Minor Issue**: Printer emits `bitstream` type without length annotation; parser reconstructs as `Bitstream { length: 0 }`. Round-trip still passes because the printer/parser agree on the convention, but length information is not preserved through text format. This is a documentation issue, not a correctness bug — the length is carried by the Encode op's parameters, not the type annotation.

### Packet O: SystemVerilog Emitter — PASS

**emit_sv.rs** (365 lines):

| IR Op | HDL Module Instantiated | Port Mapping |
|-------|------------------------|--------------|
| Encode | `sc_bitstream_encoder` | x_value, t_index, bit_out + SEED_INIT |
| BitwiseAnd | `sc_bitstream_synapse` | pre_bit, w_bit, post_bit |
| LifStep | `sc_lif_neuron` | leak_k, gain_k, I_t, noise_in + V_THRESHOLD, REFRACTORY_PERIOD |
| DenseForward | `sc_dense_layer_core` | x_input_fp, weight_fp, cfg_leak, cfg_gain + N_INPUTS, N_NEURONS |
| Scale/Offset/DivConst | Inline assign | Combinational arithmetic |
| Popcount | Wire cast | 64-bit extraction |

**Quality**:
- Emits `` `timescale 1ns / 1ps `` directive — correct for synthesis
- Emits auto-generation header comment — good practice
- All HDL parameters correctly forwarded from IR `LifParams`/`DenseParams`
- Wire naming: consistent `v%d` scheme with special cases for spike outputs
- Module instantiation format matches standard Verilog port-by-name syntax

**Minor Issues**:
- `find_value_width()` and `value_to_wire()` are O(n) linear searches per call — acceptable for small graphs but could be cached for large designs
- Rate type hardcoded to 16-bit width — not parameterized from IR metadata. Acceptable given current HDL modules are all 16-bit Q8.8

### Packet P: Co-Simulation Harness — PASS (with caveats)

**Structure** (4 files, 189 lines):

| File | Tests | Purpose |
|------|-------|---------|
| conftest.py | 0 (fixtures) | Verilator detection, build dir, compile helper |
| test_lif_cosim.py | 2 | LIF step + refractory verification |
| test_encoder_cosim.py | 2 | LFSR cycle + probability convergence |
| test_synapse_cosim.py | 1 | AND probability verification |

**Graceful Skip**: All co-sim tests skip cleanly when Verilator is unavailable — this was a mandatory requirement and is correctly implemented.

**Caveat**: Co-sim tests currently validate against the Rust golden model only, not against actual Verilator HDL simulation. The tests are structured to accept Verilator output when available but currently run only the Rust side. This is explicitly acknowledged in the session log ("Verilator not installed") and is acceptable for Phase 4 scope — the harness scaffolding is ready for when Verilator becomes available.

### Packet R: Beta Release Preparation — PASS

| Deliverable | Status | Detail |
|-------------|--------|--------|
| Version bump in Cargo.toml | YES | `3.0.0-beta.1` |
| Version bump in lib.rs | YES | `__version__` = `3.0.0-beta.1` |
| CHANGELOG_V3.md | YES | Phase 1-4 summary entries, dated 2026-02-10 |
| docs/v3_migration.md update | YES | Phase 4 section added |
| examples/02_ir_compile_demo.py | YES | Dense layer IR → SV demo |

---

## 3. Quality Gates

### Codex-Reported Results

| Gate | Command | Result |
|------|---------|--------|
| Format | `cargo fmt -- --check` | PASS |
| Lint | `cargo clippy --all-targets -- -D warnings` | PASS |
| Rust tests | `cargo test --tests` | **53 passed** (38 existing + 10 IR + 5 SV emitter) |
| Docs | `cargo doc --no-deps` | PASS |
| Python build | `maturin develop --release` | PASS |
| Python tests | `pytest` (core suites) | **46 passed** |
| Co-sim tests | `pytest cosim/` | **5 skipped** (no Verilator) |
| Training demo | `01_sc_training_demo.py` | PASS (decreasing loss + accuracy) |
| IR compile demo | `02_ir_compile_demo.py` | PASS (writes generated_dense.sv) |

### Test Count Progression

| Phase | Rust Tests | Python Tests | Total |
|-------|-----------|-------------|-------|
| Phase 1 | 12 | 20 | 32 |
| Phase 2 | 23 | 36 | 59 |
| Phase 3 | 38 | 46 | 84 |
| **Phase 4** | **53** | **46 + 5 skip** | **99 + 5 skip** |

**Growth**: +15 Rust tests, +5 Python co-sim tests (skip-safe). Core Python tests unchanged at 46 (no new Python bridge features in Phase 4).

---

## 4. Sacred File Integrity

| Check | Method | Result |
|-------|--------|--------|
| `src/sc_neurocore/` source files | `git diff` | **UNTOUCHED** (only `.pyc` cache files) |
| `pyproject.toml` | `git diff` | **UNTOUCHED** |
| `.github/workflows/ci.yml` (v2 CI) | `git diff` | **UNTOUCHED** |

The v2 sacred tree remains completely unmodified. Only `.pyc` bytecode cache files show changes (runtime artifacts, not source modifications).

---

## 5. Architectural Quality Assessment

### Strengths

1. **Pure Rust IR with zero external dependencies** — The IR module adds no new crate dependencies to Cargo.toml. It's implemented entirely in standard Rust with the existing crate infrastructure. This keeps the build fast and the dependency surface small.

2. **Clean module boundary** — The IR is in its own `engine/src/ir/` subtree with a clean `pub mod ir;` in lib.rs. It doesn't touch any existing modules (bitstream, neuron, layer, attention, graph, scpn).

3. **Correct SSA semantics** — Every operation produces exactly one `ValueId`, operand-before-use is enforced, and cycle detection provides defense-in-depth.

4. **HDL module alignment** — The emitter correctly instantiates `sc_bitstream_encoder`, `sc_bitstream_synapse`, `sc_lif_neuron`, and `sc_dense_layer_core` with the proper parameterization (SEED_INIT, V_THRESHOLD, REFRACTORY_PERIOD, N_INPUTS, N_NEURONS, etc.) matching the existing HDL files.

5. **Forward-compatible design** — The text format enables future MLIR interop (import/export), and the co-sim harness is ready for Verilator when installed.

### Minor Issues (non-blocking)

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | Bitstream length not preserved in text format round-trip | LOW | Length carried by Encode op params, not type annotation |
| 2 | `find_value_width()` in emit_sv.rs is O(n) per call | LOW | Acceptable for current graph sizes; cache if scaling |
| 3 | Rate type width hardcoded to 16-bit in emitter | LOW | Matches all current HDL modules (Q8.8 fixed-point) |
| 4 | Co-sim tests only validate Rust golden model | MEDIUM | Expected — Verilator not available in current environment |
| 5 | No negative SV emitter tests (invalid IR → error) | LOW | Verification layer catches invalid IR before emission |
| 6 | `ScOp::operands()` returns Vec allocation | LOW | Performance micro-optimization; not on hot path |

None of these issues block acceptance. Items 1-3 are design choices documented in the IR. Item 4 is an environment constraint. Items 5-6 are nice-to-haves for future phases.

---

## 6. Verdict

### ACCEPTED

Phase 4 is **fully compliant** with the handover specification. All mandatory packets (N-0, N, O, P, R) are delivered with correct implementations. The optional packet Q (WASM) was properly deferred as permitted. Quality gates (format, lint, tests, docs) all pass. Sacred files are untouched. The IR is architecturally sound and the SV emitter correctly maps to existing HDL modules.

**Cumulative v3 engine state after Phase 4**:
- **Version**: 3.0.0-beta.1
- **Rust modules**: 9 (bitstream, encoder, neuron, layer, attention, graph, grad, scpn, ir)
- **Rust tests**: 53 (unit + integration + property + IR + SV emitter)
- **Python tests**: 46 (equivalence + extension + SSGF + attention + GNN)
- **Co-sim tests**: 5 (skip-safe)
- **HDL targets**: 4 (encoder, synapse, LIF, dense layer)
- **Sacred file integrity**: MAINTAINED

---

## 7. Phase 5 Readiness

With Phase 4 complete, the v3 engine now has:
- Phases 1-2: Core SC primitives + differentiation + acceleration
- Phase 3: SSGF integration + property testing + multi-head attention + GNN
- Phase 4: IR compilation pipeline + SV emitter + co-sim scaffolding

The remaining Blueprint items for potential Phase 5 include:
- **§6 Deployment**: WASM target (deferred from Q), Python wheel publishing, npm package
- **§7 Documentation**: Full rustdoc, migration guide completion, benchmark report
- **Co-sim activation**: Install Verilator and run full HDL verification
- **FPGA synthesis trials**: Use the SV emitter output with Vivado/Quartus
- **RC release**: Bump to 3.0.0-rc.1 after co-sim verification passes
