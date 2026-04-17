# `sc_neurocore.fault_injection` — Radiation-grade fault injection

## 1. Scope

The `sc_neurocore.fault_injection` package quantifies how a
SC-NeuroCore deployment degrades under **single-event upsets
(SEUs)** and related transient/persistent faults — the dominant
hardware failure mode for systems exposed to ionising radiation.

It exists to produce the **resilience evidence** that goes into a
safety-certification package (consumed by
`sc_neurocore.safety_cert.EvidenceBag`): given a target radiation
profile (LEO, GEO, deep space, terrestrial), the package injects
a stochastic fault stream into the network state and measures the
resulting output drift, error count, and graceful-degradation
slope.

Use cases:

- **Radiation hardening** for satellite payloads (SC-NeuroCore as
  on-board AI / control logic on LEO smallsats, GEO comm-sats,
  interplanetary probes).
- **Aerospace certification** evidence per IEC 61508 + DO-254
  (FPGA / ASIC compliance for civil aviation).
- **Automotive ASIL D** stress tests per ISO 26262-11
  (semiconductor integrity tests).
- **Medical implant** soft-error tolerance per IEC 60601-1
  (pacemakers, neurostimulators that must survive cosmic-ray
  background over 10-year deployment).

## 2. Public API surface

The package re-exports 6 symbols from a single module:

```python
from sc_neurocore.fault_injection import (
    FaultModel,             # enum: BIT_FLIP / STUCK_AT_0 / STUCK_AT_1 / GAUSSIAN_NOISE / DROPOUT
    RadiationProfile,       # dataclass: name + BER + description
    FaultInjectionResult,   # dataclass: per-injection outcome record
    ResilienceReport,       # dataclass: aggregated benchmark output
    FaultInjector,          # injector: applies one fault model to a tensor
    ResilienceBenchmark,    # harness: runs N injections + computes drift curve
)
```

`__tier__ = "industrial"` — appropriate for industrial-tier
deployments.

## 3. Fault models

Five fault models are supported; each maps to a distinct
underlying physical mechanism documented in the radiation-effects
literature:

| Enum value | Physical cause | Affected element | Persistence |
|---|---|---|---|
| `BIT_FLIP` | SEU (single ion → memory cell flip) | Single bit | Transient (one cycle) |
| `STUCK_AT_0` | Latch-up after ionisation | Single bit | Persistent until reset |
| `STUCK_AT_1` | Latch-up after ionisation | Single bit | Persistent until reset |
| `GAUSSIAN_NOISE` | Aggregate analog noise | Tensor element | Transient |
| `DROPOUT` | Erasure (bit zeroed) | Single element | Transient |

`BIT_FLIP` and the stuck-at faults are bit-level (XOR with a
random mask, or AND/OR clamp). `GAUSSIAN_NOISE` and `DROPOUT`
operate on tensor elements (suitable for SC numerator/denominator
representations where the per-element error is more meaningful
than per-bit).

## 4. Radiation profiles

Four published preset profiles, with empirical bit-error rates
(BER per bit per cycle) drawn from public radiation-effects
data:

| Profile | BER (per bit per cycle) | Environment |
|---|---:|---|
| `RadiationProfile.terrestrial()` | 1 × 10⁻¹⁰ | Sea level — thermal neutron background |
| `RadiationProfile.leo()` | 1 × 10⁻⁷ | Low Earth Orbit — moderate Van Allen belt exposure |
| `RadiationProfile.geo()` | 5 × 10⁻⁶ | Geostationary — prolonged Van Allen belt + solar storms |
| `RadiationProfile.deep_space()` | 1 × 10⁻⁴ | Interplanetary — galactic cosmic rays |

The four constants are **monotone non-decreasing** in BER as the
environment becomes more hostile (verified by
`test_radiation_profile_presets`). The deep-space BER is
~6 orders of magnitude worse than the terrestrial baseline.

Custom profiles are constructed directly:

```python
custom = RadiationProfile(
    name="ISS interior",
    ber=2e-8,                       # measured on-orbit cosmic ray + secondary background
    description="ISS pressurised module, behind aluminium hull",
)
```

## 5. Injection mechanics

`FaultInjector(model=FaultModel.BIT_FLIP, profile=RadiationProfile.leo())`
is the basic injector. Its `inject(tensor, rng)` method:

1. Samples per-bit (or per-element) Bernoulli trials at the
   profile's BER.
2. Applies the fault model to each selected bit/element.
3. Records the indices + before/after values as a
   `FaultInjectionResult` (so post-mortem analysis can
   reconstruct what was hit).

Injection is **non-destructive**: the original tensor is preserved
and a corrupted copy is returned, so the user can compare
side-by-side without losing the ground truth.

`ResilienceBenchmark` wraps `FaultInjector` for a sweep over
multiple BERs (or multiple radiation profiles) and produces a
`ResilienceReport`:

```python
bench = ResilienceBenchmark(
    network=my_network,
    fault_model=FaultModel.BIT_FLIP,
    n_trials=1000,
)
report = bench.run([
    RadiationProfile.terrestrial(),
    RadiationProfile.leo(),
    RadiationProfile.geo(),
    RadiationProfile.deep_space(),
])
print(report.degradation_slope)   # 1/BER decade of accuracy lost
```

## 6. Pipeline wiring

`fault_injection` is **not** part of the simulation hot path; it
is post-simulation analysis. The typical workflow:

1. The user runs a baseline simulation (`Network.run(...)`),
   captures golden output.
2. The user constructs a `ResilienceBenchmark` over the same
   network.
3. The benchmark replays the simulation N_trials times with
   injected faults at each radiation profile.
4. The `ResilienceReport` is fed to
   `sc_neurocore.safety_cert.EvidenceBag` for inclusion in the
   certification package.

There is no Rust / Julia / Go / Mojo path in this package —
fault injection is dominated by NumPy element-wise ops which
already vectorise on the CPU's SIMD lane width. Per the
`feedback_multi_language_accel.md` rule, this is a vectorised
NumPy hot path that does not benefit from a separate language
backend (the inner loop is already a single BLAS-level call).

## 7. Pure-Python performance

Single `inject()` call wall time on Intel i5-11600K, NumPy
2.2.0 (Python 3.12.3) for tensor of shape (1000, 1000) at LEO
BER (1e-7):

| Fault model | Mean | p99 |
|---|---:|---:|
| `BIT_FLIP` | ~0.4 ms | ~0.7 ms |
| `STUCK_AT_0` | ~0.4 ms | ~0.7 ms |
| `STUCK_AT_1` | ~0.4 ms | ~0.7 ms |
| `GAUSSIAN_NOISE` | ~3.2 ms | ~4.5 ms |
| `DROPOUT` | ~1.1 ms | ~1.5 ms |

(Numbers from informal `python -m timeit` runs; not from a
committed benchmark — see followup §10.1.)

`ResilienceBenchmark.run()` for 1000 trials × 4 profiles on the
above tensor takes ~3 s wall (~750 µs per trial amortised).

## 8. Test coverage

Two test files cover this package:

| File | Tests | LOC | What it covers |
|---|---:|---:|---|
| `tests/test_fault_injection/test_fault_injection.py` | 22 | 212 | Antigravity-authored unittest-style classes: `TestRadiationProfiles`, `TestFaultInjectionResult`, `TestFaultInjector`, `TestResilienceBenchmark` |
| `tests/test_fault_injection/test_fault_injection_public_api.py` | 7 | new | Arcane Sapience: package re-exports identity, `__all__` membership, FaultModel enum 5-member completeness, RadiationProfile preset BER ordering and exact constants |

**Total: 29 tests.** Both files run in ~0.5 s combined; no
skips, no failures.

## 9. Audit completeness — 7-point rule

| # | Criterion | Status | Notes |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 6 symbols re-exported via `__init__.py`; verified by `test_fault_injection_public_api.py` |
| 2 | Multi-angle tests | ✅ PASS | 29 tests across 2 files; covers fault models × radiation profiles × benchmark sweep |
| 3 | Acceleration path | N/A | Vectorised NumPy hot path; per `feedback_multi_language_accel.md` no separate language backend needed |
| 4 | Benchmarks | ⚠️ WARN | Informal `timeit` numbers in §7; no committed benchmark script |
| 5 | Performance docs | ✅ PASS | §7 with explicit "informal" caveat |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX 2-line header on `__init__.py` and `fault_injection.py` (`__init__.py` fixed in this batch from 1-line piped form). British English in this doc; source uses standard scientific-Python identifiers (acceptable per docs-vs-code rule). |

Net: **1 WARN, 0 FAIL.**

## 10. Known issues / follow-ups

### 10.1 No committed benchmark (WARN row 4)

Open follow-up: commit `benchmarks/bench_fault_injection.py`
reproducing §7 numbers (5 fault models × 4 profiles × 3
tensor sizes = 60 cells, median-of-5 protocol). Lower priority
because `inject()` is sub-millisecond and the benchmark would
mostly characterise NumPy + RNG performance.

### 10.2 Custom-profile validation

`RadiationProfile(name=..., ber=...)` accepts any float for `ber`.
A future refinement should reject `ber > 0.5` (physically
impossible — at that point the bit is essentially random) and
`ber < 0` (physically impossible — BER is a probability).
Currently the user can construct nonsense profiles and the
injector silently produces nonsense output.

### 10.3 No correlation modelling

Real radiation environments produce **bursts** of correlated
faults (a single ion can flip multiple adjacent bits, called a
multi-bit upset / MBU). The current injector samples
independent Bernoulli per bit. A future
`MBURadiationProfile` should accept a burst-length distribution
and inject correlated bit flips within a configurable
neighbourhood.

### 10.4 No latch-up recovery model

`STUCK_AT_0` / `STUCK_AT_1` are persistent in the model but the
injector does not simulate the operator's reset-and-recover
cycle. A `RecoveryProfile(reset_after=...)` parameter would let
the benchmark distinguish "system hangs forever" from "system
recovers within N cycles".

### 10.5 No bug found in this audit

Audit found:
- `__init__.py` did not re-export the 6 public symbols. Wired.
- 1-line piped SPDX header in `__init__.py`. Fixed.
- Pre-existing `docs/api/fault_injection.md` was a 14-line stub
  with mkdocstrings auto-gen and no curated content. Replaced
  with this page.

No semantic bugs (sign errors, wrong invariants, fabricated
constants) found in `fault_injection.py`. The 22 Antigravity
tests pass; the 7 new public-API tests pass.

## 11. References

- IEC 61508-2 (Ed 2.0): Functional safety — Requirements for
  E/E/PE safety-related systems. Geneva: IEC.
- IEC 62396 (Ed 2.0): Process management for avionics —
  Atmospheric radiation effects. Geneva: IEC.
- ECSS-Q-ST-60-15C: Space product assurance — Radiation hardness
  assurance — EEE components. Noordwijk: ESA.
- JEDEC JESD89A: Measurement and reporting of alpha particle and
  terrestrial cosmic ray-induced soft errors in semiconductor
  devices. Arlington VA: JEDEC.
- Petersen, E. (2011). *Single Event Effects in Aerospace*.
  IEEE Press.

## 12. Audit batch identification

This page was produced as part of the **Antigravity audit, batch
B1, package 2** (per
`docs/internal/antigravity_inventory_2026-04-17.md`). Package 1
was `safety_cert/`; package 3 (`chiplet/`) follows in subsequent
batches.
