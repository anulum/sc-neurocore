CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
Contact us: www.anulum.li  protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AFFERO GENERAL PUBLIC LICENSE v3
Commercial Licensing: Available

# Session Log: SC-NeuroCore Documentation and Toolchain Stabilization

**Date:** 2026-02-11  
**Scope:** `03_CODE/sc-neurocore` only  
**Purpose:** close documentation gaps, align version assertions, and restore reproducible local Rust/PyO3 validation.

## Completed Work

1. Installed local Rust toolchain under workspace:
- `cargo 1.93.0`
- `rustc 1.93.0`

2. Confirmed `maturin` availability in `.venv` (`1.11.5`).

3. Fixed build blocker:
- Root cause: Windows `ReadOnly` attribute on default `target/` path caused `autocfg` writable-dir check failure.
- Mitigation: use writable target dir `CARGO_TARGET_DIR=.cargo_target_rw`, plus workspace-local `TMP/TEMP`.

4. Rebuilt native extension:
- Command path: `python -m maturin develop --release -m engine/Cargo.toml`
- Result: successful install in editable mode.

5. Synced runtime binary for bridge path testing:
- Copied `.venv/Lib/site-packages/sc_neurocore_engine/sc_neurocore_engine.cp312-win_amd64.pyd`
- Destination: `bridge/sc_neurocore_engine/sc_neurocore_engine.cp312-win_amd64.pyd`

6. Documentation updates:
- Added legal notice block to onboarding docs.
- Added explicit performance path routing guidance.
- Added comprehensive discrepancy remediation plan document.

7. Version/reference updates:
- `bridge/pyproject.toml`: Phase stream set to `3.6.0`
- Phase version assertions in tests (`test_phase8.py` through `test_phase12.py`) set to `3.6.0`
- `docs/BENCHMARK_REPORT.md` Phase metadata set to `3.6.0` with baseline caveats added.

## Validation Evidence

1. Import/feature sanity:
- `sc_neurocore_engine.__version__` observed as `3.7.0` in the local runtime binary during rebuild validation.
- `BitStreamTensor` export present -> `True`

2. Tests:
- Command: `PYTHONPATH=src;bridge python -m pytest tests/test_phase8.py tests/test_phase9.py tests/test_phase10.py tests/test_phase11.py tests/test_phase12.py -q`
- Result: `73 passed in 1.29s`

## Files Added

1. `docs/SC_NEUROCORE_DISCREPANCY_REMEDIATION_PLAN_2026-02-11.md`
2. `SESSION_LOG_2026-02-11_SC_NEUROCORE_DOCS_AND_TOOLCHAIN.md`

## Files Updated

1. `README.md`
2. `docs/getting-started.md`
3. `docs/BENCHMARK_REPORT.md`
4. `bridge/pyproject.toml`
5. `tests/test_phase8.py`
6. `tests/test_phase9.py`
7. `tests/test_phase10.py`
8. `tests/test_phase11.py`
9. `tests/test_phase12.py`

## Correction (2026-02-11)

Per follow-up decision, Phase stream references were rolled back from `3.7.0` to `3.6.0` in:

1. `bridge/pyproject.toml`
2. `docs/BENCHMARK_REPORT.md`
3. `tests/test_phase8.py`
4. `tests/test_phase9.py`
5. `tests/test_phase10.py`
6. `tests/test_phase11.py`
7. `tests/test_phase12.py`

This keeps Polymorphic Engine (`v3.7.x`) separated from Phase documentation/tests metadata.

## Runtime Pin Update (2026-02-11)

Runtime surfaces were pinned to `3.6.0` for Phase consistency:

1. `engine/Cargo.toml` (`version = "3.6.0"`)
2. `engine/src/lib.rs` (`m.add("__version__", "3.6.0")`)
3. `bridge/sc_neurocore_engine/__init__.py` docstring (`v3.6`)

Rebuild and verification:

1. Rebuilt with `maturin develop --release` using workspace-local Rust toolchain.
2. Synced rebuilt `.pyd` into `bridge/sc_neurocore_engine/`.
3. Verified runtime:
- `sc_neurocore_engine.__version__ == "3.6.0"`
4. Test gate:
- `pytest tests/test_phase8.py tests/test_phase9.py tests/test_phase10.py tests/test_phase11.py tests/test_phase12.py -q`
- Result: `73 passed`.
