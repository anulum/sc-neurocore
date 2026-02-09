# SC-NeuroCore v3 Migration Guide

## Status

Phase 1 scaffolding is in place:

- Rust engine crate in `engine/`
- Python bridge package in `bridge/sc_neurocore_engine/`
- v2-vs-v3 equivalence tests in `tests/equivalence/`

## Build (Local)

```powershell
cd 03_CODE/sc-neurocore/engine
maturin develop --release
```

## Quick Sanity Check

```powershell
python -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__); print(sc_neurocore_engine.simd_tier())"
```

## Equivalence Tests

```powershell
cd 03_CODE/sc-neurocore
$env:PYTHONPATH=\"src;bridge\"
python -m pytest tests/equivalence -v --tb=short
```

## Notes

- v2 package under `src/sc_neurocore/` remains untouched.
- v3 bridge is a drop-in import path for hot kernels and fixed-point neuron APIs.
- Encoder and LIF in v3 currently follow strict blueprint operation ordering
  (step-then-compare encoder, refractory override after threshold evaluation).
