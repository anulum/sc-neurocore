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

The package re-exports the injector, Monte-Carlo benchmark, seeded degradation
policy, and resilience-mode orchestrator:

```python
from sc_neurocore.fault_injection import (
    FaultModel,             # enum: BIT_FLIP / STUCK_AT_0 / STUCK_AT_1 / GAUSSIAN_NOISE / DROPOUT
    RadiationProfile,       # dataclass: name + BER stress preset + description
    FaultInjectionResult,   # dataclass: per-injection outcome record
    ResilienceReport,       # dataclass: aggregated benchmark output
    FaultInjector,          # injector: applies one fault model to a tensor
    ResilienceBenchmark,    # harness: runs N injections + computes drift curve
    GracefulDegradationPolicy,
    FaultInjectionResilienceMode,
    ResilienceModeConfig,
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

## 4. Radiation Profiles

Four engineering stress presets are available. They are ordered to represent
increasingly harsh environments, but they are not a substitute for a
mission-specific radiation transport analysis using the actual orbit,
shielding, process node, memory macro, scrubbing cadence, and operating
temperature.

| Profile | BER (per bit per cycle) | Environment |
|---|---:|---|
| `RadiationProfile.terrestrial()` | 1 × 10⁻¹⁰ | Sea level — thermal neutron background |
| `RadiationProfile.leo()` | 1 × 10⁻⁷ | Low Earth Orbit — moderate Van Allen belt exposure |
| `RadiationProfile.geo()` | 5 × 10⁻⁶ | Geostationary — prolonged Van Allen belt + solar storms |
| `RadiationProfile.deep_space()` | 1 × 10⁻⁴ | Interplanetary — galactic cosmic rays |

The presets are deliberately conservative stress points for software
resilience testing. Flight, medical, automotive, or safety certification
evidence must replace them with externally justified rates.

Custom profiles are constructed directly:

```python
custom = RadiationProfile(
    name="ISS interior",
    ber=2e-8,                       # measured on-orbit cosmic ray + secondary background
    description="ISS pressurised module, behind aluminium hull",
)
```

## 5. Injection Mechanics

`FaultInjector(seed=...)` is the basic seeded injector. Its
`inject(bitstream, model, ber)` method:

1. Samples per-bit Bernoulli trials at the supplied BER.
2. Applies the fault model to each selected bit/element.
3. Returns a corrupted copy plus the number of affected bits.

Injection is **non-destructive**: the original tensor is preserved
and a corrupted copy is returned, so the user can compare
side-by-side without losing the ground truth.

`ResilienceBenchmark` wraps `FaultInjector` for a BER sweep and produces a
`ResilienceReport`:

```python
bench = ResilienceBenchmark(seed=42)
report = bench.run(
    fault_model=FaultModel.BIT_FLIP,
    ber=RadiationProfile.leo().ber,
    bitstream_length=4096,
    probability=0.5,
    num_trials=1000,
)
print(report.mean_error)
```

`FaultInjectionResilienceMode` runs seeded trials directly on a layer's
existing binary SC bitstreams and combines the drift statistics with
`GracefulDegradationPolicy`:

```python
mode = FaultInjectionResilienceMode(
    ResilienceModeConfig(
        layer_id="encoder.l0",
        radiation_profile=RadiationProfile.leo(),
        num_trials=256,
        seed=17,
    )
)
report = mode.run(bitstreams)  # shape: (neurons, bits), values 0/1
print(report.recommended_action.value)
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

Multi-language kernels are wired into the bench harness at
`benchmarks/bench_fault_injection.py`. All 5 backends (Python,
Rust via PyO3, Julia via `juliacall`, Go via `ctypes` + c-shared,
Mojo via `ctypes` + `mojo build --emit shared-lib`) expose the
same 5 fault-model entry points with the same in/out contract
as the pure-Python `FaultInjector.inject`.

Kernel sources:

| Backend | Entry point | Source |
|---|---|---|
| Python | `FaultInjector.inject` | `src/sc_neurocore/fault_injection/fault_injection.py` |
| Rust | `py_inject_{model}_u8` | `engine/src/fault.rs` + `engine/src/lib.rs` |
| Julia | `FaultInjectionAccel.inject_{model}` | `src/sc_neurocore/accel/julia/fault_injection/fault_injection.jl` |
| Go | `inject_{model}_c` | `src/sc_neurocore/accel/go/fault_injection/fault.go` |
| Mojo | `inject_{model}_c` | `src/sc_neurocore/accel/mojo/fault_injection/fault.mojo` |

RNG parity is **statistical, not bitwise** — each backend uses
a different PRNG (NumPy PCG64 / Rust Xoshiro256++ / Julia
Xoshiro / Go ChaCha8 / Mojo SplitMix64-style LCG). The bench
harness verifies that fault counts lie within 4σ of
Binomial(n, ber) on a 1 Mbit stream.

## 7. Multi-backend performance

Reproducible via the committed benchmark:

```bash
python benchmarks/bench_fault_injection.py \
    --json benchmarks/results/bench_fault_injection.json
```

Per-call wall time on a **1 Mbit boolean bitstream** at BER
1e-3 (raised from LEO 1e-7 so the fault count per call is
non-zero and stable; Gaussian σ=0.5). 5 repeats per cell,
median reported. Hardware: Linux 6.17 x86_64, NumPy 2.2.6,
Python 3.12.3. Captured run in
`benchmarks/results/bench_fault_injection.json`.

| Fault model | Python | Rust | Julia | Go | Mojo | Fastest |
|---|---:|---:|---:|---:|---:|:---|
| `BIT_FLIP` | 3.52 ms | 1.51 ms | 2.74 ms | 6.64 ms | **1.28 ms** | Mojo 2.7× |
| `STUCK_AT_0` | 7.98 ms | 1.58 ms | 2.54 ms | 5.08 ms | **1.05 ms** | Mojo 7.6× |
| `STUCK_AT_1` | 3.78 ms | 1.44 ms | 2.00 ms | 6.10 ms | **0.98 ms** | Mojo 3.9× |
| `DROPOUT` | 7.86 ms | 1.36 ms | 2.58 ms | 5.10 ms | **0.96 ms** | Mojo 8.2× |
| `GAUSSIAN_NOISE` | 22.12 ms | 4.48 ms | **3.88 ms** | 11.94 ms | 19.54 ms | Julia 5.7× |

Mojo wins 4/5 boolean kernels (2.7–8.2× over NumPy) — the LLVM
backend behind `mojo build --emit shared-lib` produces tight
per-byte loops that beat both Rust's `Xoshiro256++` and NumPy's
batch path. Rust is consistently 2nd. Julia takes Gaussian
because its `randn` is implemented via Ziggurat in optimised
Julia, while the current Mojo kernel uses a naïve Box-Muller
that allocates two uniforms per sample and computes
`sqrt(log)`. Go trails on `math/rand/v2.ChaCha8` (the safety
margin of ChaCha8 vs Xoshiro is the tradeoff).

Backends (from JSON output, fastest-first ordering per the
multi-language fallback rule):

| Backend | Status | Reason |
|---|---|---|
| mojo | USED — fastest on 4/5 boolean ops | via `mojo build --emit shared-lib` + ctypes; raw-Int-addr workaround for @export parametric restriction |
| rust | USED | via PyO3 byte-level kernels; fastest on no individual op but never worse than 2nd |
| julia | USED — fastest on Gaussian | via juliacall + Xoshiro; Ziggurat randn beats Mojo's naïve Box-Muller |
| go | USED | via ctypes + ChaCha8 c-shared lib |
| python | USED | baseline (NumPy PCG64); the floor of the chain |

`GAUSSIAN_NOISE` is ~2–3× slower than boolean models across
all backends because it runs a normal draw, a clamp, and a
threshold instead of a single Bernoulli test.

The actual API is `inject(bitstream, model, ber)` — a 1D
boolean array, NOT a 2D tensor. Earlier drafts of this page
incorrectly described per-bit vs per-element semantics; the
implementation is uniformly per-element on a flat bitstream.

`ResilienceBenchmark.run()` is **not yet benchmarked** —
follow-up #61 tracks adding it.

## 8. Test coverage

Two test files cover this package:

| File | Tests | LOC | What it covers |
|---|---:|---:|---|
| `tests/test_fault_injection/test_fault_injection.py` | 22 | 212 | Unittest-style classes covering baseline utility fault-injection behaviour |
| `tests/test_fault_injection/test_fault_injection_public_api.py` | 7 | new | Package re-export identity, `__all__` membership, enum/preset surface contracts |
| `tests/test_fault_injection_module.py` | 59 | new | Production contracts for `RadiationProfile`, `FaultInjectionResult`, `ResilienceReport`, `FaultInjector.inject`, deterministic position injection, benchmark run/sweep guards, finite aggregate checks, and zero-BER no-op guarantees |
| `tests/test_fault_injection/test_resilience_mode.py` | 7 | updated | Config/report/trial contracts, deterministic seeded behaviour, aggregate bound checks, and fail-closed input validation for resilience mode |
| `tests/test_fault_injection/test_resilience_policy.py` | 21 | updated | Policy configuration/evaluate helper contracts, seeded observation/plan invariants, and bounded recommendation semantics |

**Total: 116 tests** across injector, benchmark, resilience mode, policy, and
public API contract surfaces.

## 9. Audit completeness — 7-point rule

| # | Criterion | Status | Notes |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 6 symbols re-exported via `__init__.py`; verified by `test_fault_injection_public_api.py` |
| 2 | Multi-angle tests | ✅ PASS | 29 tests across 2 files; covers fault models × radiation profiles × benchmark sweep |
| 3 | Acceleration path | ✅ PASS | All 5 backends wired (Python + Rust + Julia + Go + Mojo). Mojo fastest on 4/5 ops (closes #69 for fault_injection) |
| 4 | Benchmarks | ✅ PASS | `benchmarks/bench_fault_injection.py` committed; JSON in `benchmarks/results/` |
| 5 | Performance docs | ✅ PASS | §7 with measured numbers from the benchmark |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX 2-line header on `__init__.py` and `fault_injection.py` (`__init__.py` fixed in this batch from 1-line piped form). British English in this doc; source uses standard scientific-Python identifiers (acceptable per docs-vs-code rule). |

Net: **0 WARN, 0 FAIL.**

## 10. Known issues / follow-ups

### 10.1 No committed benchmark (WARN row 4)

Open follow-up: commit `benchmarks/bench_fault_injection.py`
reproducing §7 numbers (5 fault models × 4 profiles × 3
tensor sizes = 60 cells, median-of-5 protocol). Lower priority
because `inject()` is sub-millisecond and the benchmark would
mostly characterise NumPy + RNG performance.

### 10.2 Custom-profile validation

`RadiationProfile` now validates non-empty `name`, finite
`ber` in `[0, 1]`, and string `description`. `FaultInjector.inject`,
`ResilienceBenchmark.run`, and `ResilienceBenchmark.sweep_ber` now also
fail closed on invalid BER/probability ranges and malformed inputs.

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

Audit found and addressed:
- contract hardening now enforces finite/bounded dataclass fields for
  `RadiationProfile`, `FaultInjectionResult`, and `ResilienceReport`;
- `FaultInjector.inject` rejects malformed arrays, invalid enums, invalid BER,
  and non-binary streams for discrete models;
- benchmark run/sweep surfaces now fail closed on malformed inputs and
  non-finite aggregate-state generation;
- `ResilienceModeConfig`, `ResilienceModeTrialReport`, `ResilienceModeReport`,
  `SeededFaultObservation`, `DegradationPlan`, and `GracefulDegradationPolicy`
  now enforce strict contract bounds, finite metric checks, and deterministic
  recommendation constraints.

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
