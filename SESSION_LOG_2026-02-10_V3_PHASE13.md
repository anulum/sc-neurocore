# Session Log: SC-NeuroCore v3 Phase 13 — Integrity Hotfix

**Session ID**: SC-NEUROCORE-2026-02-10-V3-PHASE13
**Date**: 2026-02-10
**Agent**: Claude Opus 4.6
**Blueprint Sources**: Phase 13 handover specification (user-provided)
**Semantics Mode**: Strict blueprint semantics

---

## Objective

Execute Phase 13 (Integrity Hotfix) comprising four packets:

- **DA**: Fix SystemVerilog popcount emission in `emit_sv.rs`
- **DB**: Optimize weight packing (`bernoulli_packed` → `bernoulli_packed_simd`)
- **DC**: Verify CI workflow existence
- **DD**: Cross-platform Verilator path fix in `cosim/conftest.py`

---

## Files Modified

- `engine/src/ir/emit_sv.rs` — Packet DA
- `engine/src/layer.rs` — Packet DB
- `cosim/conftest.py` — Packet DD

---

## Implementation Summary

### Packet DA: Fix SystemVerilog Popcount Emission

**Bug**: The `ScOp::Popcount` emitter in `emit_sv.rs` was generating:
```systemverilog
wire [63:0] v5;
assign v5 = {63'd0, input_wire};
```
This is a zero-extension (always 0 or 1), NOT a popcount.

**Fix**:
1. Changed wire declaration from `wire [63:0]` to `logic [63:0]` (required for `always_comb` target in SystemVerilog)
2. Replaced zero-extension `assign` with proper combinatorial for-loop:
```systemverilog
logic [63:0] v5;
// Combinatorial popcount for v5
always_comb begin
    v5 = 64'd0;
    for (integer _pc_i = 0; _pc_i < 64; _pc_i = _pc_i + 1)
        v5 = v5 + {63'd0, input_wire[_pc_i]};
end
```

### Packet DB: Optimize Weight Packing with SIMD

**Change**: In `DenseLayer::refresh_packed_weights()`, replaced `bernoulli_packed` (scalar f64 comparison, 8 bytes RNG per bit) with `bernoulli_packed_simd` (byte-threshold SIMD comparison, 1 byte RNG per bit).

**Files changed**:
- `engine/src/layer.rs:107` — function call in `refresh_packed_weights()`
- `engine/src/layer.rs:510` — matching update in `flat_weight_roundtrip` test

Both functions have identical signatures and return types. The SIMD variant uses the same byte-threshold encoding path as `forward_fast`, ensuring weight encoding is now consistent with the input encoding path.

### Packet DC: CI Workflow Verification

**Finding**: `.github/workflows/v3-engine.yml` already exists with comprehensive multi-platform CI coverage including 6 jobs:
1. `rust-lint` — fmt + clippy
2. `rust-test` — cargo test
3. `equivalence` — 3 OS × 2 Python versions matrix
4. `cosim` — Verilator co-simulation on ubuntu-latest
5. `v2-compat` — Sacred file integrity check
6. `benchmarks` — Criterion benchmarks

**Action**: No changes needed.

### Packet DD: Cross-Platform conftest.py Path Fix

**Issues fixed**:
1. **Hardcoded Windows path**: `GIT_USR_BIN = pathlib.Path(r"C:\Progra~1\Git\usr\bin")` replaced with dynamic `_find_sh_dir()` function that uses `shutil.which("sh")` as primary resolution with Windows fallback candidates
2. **Platform-specific venv directories**: `"Scripts"` → `os.name == "nt" ? "Scripts" : "bin"`, `"Lib"` → `os.name == "nt" ? "Lib" : "lib"`
3. **Platform-specific executable suffix**: `.exe` → `f"{_EXE_SUFFIX}"` based on `os.name`
4. **Shell detection**: `GIT_SH` replaced with dynamic lookup from `_find_sh_dir()` result

---

## Verification Evidence

### Gate 1: Rust (fmt + clippy + test)

```powershell
cd 03_CODE/sc-neurocore/engine
cargo fmt -- --check        # PASS (no formatting issues)
cargo clippy --all-targets -- -D warnings  # PASS (0 warnings)
cargo test --tests          # PASS (74 tests)
```

Test breakdown:
- 20 unit tests (lib.rs)
- 5 equivalence tests
- 4 prop_bitstream tests
- 3 prop_kuramoto tests
- 3 prop_layer tests
- 2 prop_neuron tests
- 5 emit_sv tests
- 10 IR tests
- 3 IR bridge tests
- 4 kuramoto tests
- 4 kuramoto_ssgf tests
- 11 surrogate tests

### Gate 2: Build

```powershell
maturin develop --release
```

Result: `sc_neurocore_engine-3.6.0` installed successfully.

### Gate 3: Python tests (full suite)

```powershell
PYTHONPATH='src;bridge' python -m pytest tests/ -v --tb=short
```

Result: **173 passed in 23.80s**

### Gate 4: Co-simulation

```powershell
PYTHONPATH='src;bridge' python -m pytest cosim/ -v -rs --tb=short
```

Result: **8 passed in 41.28s**

---

## Sacred File Integrity

Sacred files untouched:
- `src/sc_neurocore/` — no modifications
- Repository-root `pyproject.toml` — no modifications
- `.github/workflows/ci.yml` — no modifications

---

## Notes

- Version remains 3.6.0 (hotfix, no version bump per specification)
- The popcount fix (DA) is critical for correct SystemVerilog synthesis — the previous code would always produce 0 or 1 instead of counting set bits
- The SIMD weight packing (DB) aligns weight encoding with the SIMD input encoding path, reducing encoding time by ~8x for weight refresh
- Cross-platform conftest.py (DD) enables co-sim CI on Linux without manual path configuration
