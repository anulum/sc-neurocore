# `sc_neurocore.safety_cert` — Functional safety certification

## 1. Scope

The `sc_neurocore.safety_cert` package automates the production of
**safety-certification artefacts** required by the international
functional-safety standards covering programmable electronic
systems for industrial control, automotive ECUs, and FDA Class III
medical devices. It exists to turn SC-NeuroCore's deterministic
bitstreams, formal proofs, and traceable execution paths into the
document set that a notified body (TÜV, UL, BSI) or regulator
(FDA, EMA) needs to grant approval.

It targets the following standards (mapping in
`CrossStandardMapper`):

- **IEC 61508** — generic functional safety for E/E/PE systems
  (Safety Integrity Levels SIL 1–4).
- **ISO 26262** — automotive functional safety, derived from
  IEC 61508 (Automotive Safety Integrity Levels ASIL A–D).
- **IEC 62304** — software lifecycle for medical devices
  (Software Safety Classes A / B / C).
- **FDA 21 CFR Part 820 + IEC 62304** — Class III medical device
  software (FDA Premarket Approval pathway).

The deliverable is a **Certification Package** (`CertificationPackage`):
a hash-stamped bundle of (a) a traceability matrix, (b) FMEDA
records, (c) formal-proof certificates, (d) WCET analysis, (e)
compliance checklists, (f) common-cause-failure (CCF) analysis,
(g) proof-test coverage estimates, (h) hardware-fault-tolerance
(HFT) assessment, and — for software — (i) IEC 62304 software
classification with hazard analysis.

The package is **machine-checkable**: every artefact carries a
SHA-256 stamp computed over the raw input evidence so a downstream
auditor can prove the bundle has not been tampered with after
sign-off.

`sc_neurocore.safety_cert.safety_monitor.SafetyMonitor` is a
software-in-the-loop mirror of the formally-proven hardware
safety monitor (six properties, each backed by a Lean / SymbiYosys
proof). It exists so the certification artefacts can include
**simulation evidence** that the monitor would have caught each
target fault class before silicon is taped out.

## 2. Public API surface

The package re-exports 34 symbols from two modules:

- **`safety_cert.py`** — 32 symbols: 6 enums + 13 dataclasses +
  13 generators / analysers.
- **`safety_monitor.py`** — 2 symbols: `SafetyLimits` (config
  thresholds) + `SafetyMonitor` (the runtime checker).

Top-level imports:

```python
from sc_neurocore.safety_cert import (
    # ─── enums (6)
    SafetyStandard, SILLevel, ASILLevel,
    FailureCategory, HFTLevel, SWClass,
    # ─── dataclasses (13)
    Requirement, FailureMode, FormalProperty, FormalProofCertificate,
    WCETPath, ChecklistItem, CertificationPackage,
    CCFDefence, HFTAssessment, ChangeRecord, EvidenceItem,
    PropertyGap, ReliabilityMetrics,
    # ─── generators / analysers (13)
    TraceabilityMatrix, FMEDA, WCETAnalyzer,
    ComplianceChecklist, CertificationGenerator,
    CCFAnalysis, ProofTestCoverage, IEC62304Assessment,
    ChangeImpactTracker, SafetyManualGenerator, EvidenceBag,
    CrossStandardMapper, FormalPropertyGapDetector,
    # ─── runtime monitor (2)
    SafetyLimits, SafetyMonitor,
)
```

`__tier__ = "industrial"` is the package's self-classification —
it tags this module as appropriate for industrial-tier deployments
where regulator scrutiny is expected.

## 3. Standards mapping

`CrossStandardMapper` provides the bidirectional translation
between standards' integrity-level grading systems. The
relationships are (per IEC 61508-5 informative annex + ISO 26262-9
bridging tables):

| IEC 61508 | ISO 26262 | IEC 62304 | Notes |
|---|---|---|---|
| SIL 1 | ASIL A | Class A | Lowest integrity; non-critical |
| SIL 2 | ASIL B | Class A | Possible single-point failures, low harm |
| SIL 3 | ASIL C | Class B | Severe injury possible |
| SIL 4 | ASIL D | Class C | Life-threatening / mass-casualty |
| — | — | Class C | Death likely without correct operation |

The mapper does **not** claim symmetric translation: ASIL D is
*roughly* SIL 3 in raw probability terms, but ISO 26262 adds
controllability (driver intervention) and exposure (frequency in
typical driving) factors that change the picture. The mapper
flags such asymmetries in its output rather than masking them.

## 4. The six safety-monitor properties

`SafetyMonitor.check()` enforces six invariants that mirror the
formally-proven SystemVerilog `neuro_safe_monitor` module +
`safety_bounds.lean` proofs:

| ID | Property | Trigger condition |
|---|---|---|
| P1 | `monitor_soundness` | `current > max_current` ∨ `voltage > max_voltage` ∨ `coherence < coherence_limit` |
| P2 | `safe_transition` | coherence MUST be monotone non-decreasing across calls |
| P3 | `sc_precision_bound` | `popcount_k ≤ sc_denom` (Q8.8 truncation safety) |
| P4 | `sc_add_preserves_range` | `sc_add_result ≤ sc_denom` (no overflow on SC adder) |
| P5 | `lif_membrane_bounded` | `membrane ≤ lif_v_max` |
| P6 | `correlation_range` | `|scc_numerator| ≤ scc_denominator` |

Violations are tracked in the 6-bit sticky `violation_flags`
register; the monitor halts (`halted = True`) on first violation
and stays halted until `reset()` is called. This mirrors the
hardware behaviour where the monitor drives a system-level reset
line that the operator must explicitly clear.

### 4.1 Default `SafetyLimits`

| Field | Default | Meaning |
|---|---|---|
| `max_current` | `0x7FFF` (32 767) | Q8.8 saturation just below INT16_MAX |
| `max_voltage` | `0xC000` (49 152) | 75% of full-scale, leaves headroom for transients |
| `coherence_limit` | `0x0100` (256) | Lower bound; below = entropy too high |
| `sc_denom` | `0x0100` (256) | SC numeric denominator (Q8.8 unit) |
| `lif_v_max` | `0xC000` | Same as voltage cap |

These are the same defaults as `neuro_safe_monitor.sv` so the
software mirror's behaviour matches the silicon to the bit.

## 5. Determinism + audit trail

Every artefact written by `CertificationGenerator` carries:

- A **SHA-256 stamp** over the canonicalised input (sorted JSON,
  no whitespace) so a tamper-evident chain of custody can be
  reconstructed.
- A **timestamp** in ISO 8601 (UTC) at generation time.
- The **package version** of `sc_neurocore` (so re-generation
  with a future version can be traced to a different binary).

The hash is over the input — not the output — so the same
artefact text can be regenerated and verified bit-for-bit without
storing the output blob alongside.

## 6. Pipeline wiring

`sc_neurocore.safety_cert` is **not** part of the simulation hot
path; it is a post-simulation analysis layer. Typical workflow:

1. The user runs a simulation (`Network.run(...)`), monitoring
   spike + voltage statistics with `SafetyMonitor` instrumentation.
2. The user feeds the run's evidence into `EvidenceBag`.
3. `CertificationGenerator` consumes the `EvidenceBag` and emits
   the `CertificationPackage`.
4. `IEC62304Assessment` (if a medical device) classifies the
   software per the IEC 62304 hazard analysis.
5. `CrossStandardMapper` produces the standards-mapping appendix
   for multi-jurisdiction submission.

No Rust / Julia / Go / Mojo backend is wired for this package
— the `Certification*` classes are document generators (not
compute kernels), and `SafetyMonitor.check()` runs in 519-922 ns
(see §7), where any FFI dispatch path adds 1-10 µs of marshalling
overhead per call — larger than the entire compute time. Per
`feedback_multi_language_accel.md`, the four non-Python backends
are documented EXEMPT (not silently skipped) in the bench JSON's
`backends` block. All four toolchains (Rust PyO3, Julia juliacall,
Go cgo+ctypes, Mojo `--emit shared-lib` + ctypes) ARE installed
and proven on heavier ops in this codebase (LGSSM Kalman, fault
injection); the exemption here is a per-op cost-benefit decision,
not a tool-availability claim.

## 7. Pure-Python performance

Reproducible via the committed benchmark:

```bash
python benchmarks/bench_safety_monitor.py \
    --json benchmarks/results/bench_safety_monitor.json
```

100 000 iterations × 5 repeats per scenario, median + min over
the 5 repeats reported. Hardware: Linux 6.17 x86_64,
Python 3.12.3, NumPy 2.2.0. Captured run in
`benchmarks/results/bench_safety_monitor.json`.

| Scenario | Median | Min |
|---|---:|---:|
| All-defaults check (no violation) | 353 ns | 309 ns |
| Triggered overcurrent (P1) | 486 ns | 449 ns |
| All 6 violations | 776 ns | 720 ns |

`CertificationGenerator.build_package()` is **not yet
benchmarked** — follow-up #61 tracks adding it to
`benchmarks/bench_safety_monitor.py`. The dominant cost is
expected to be SHA-256 over the canonicalised JSON, not
template rendering, but no measurement has been made yet.

## 8. Test coverage

Two test files cover this package:

| File | Tests | Lines | What it covers |
|---|---:|---:|---|
| `tests/test_safety_cert/test_safety_cert.py` | 81 | 23 483 | Antigravity-authored: enum semantics, dataclass round-trips, FMEDA failure-mode tabulation, traceability matrix + checklist coverage, hash determinism, CCF + HFT + IEC 62304 assessments |
| `tests/test_safety_cert/test_safety_cert_public_api.py` | 14 | new | Arcane Sapience: package re-exports identity, `__all__` membership, `SafetyMonitor` instantiation + check + sticky flags + reset, `SafetyStandard` enum membership, `SILLevel` 4-level grading |

**Total: 95 tests.** Both files run in ~12 s combined; no skips,
no failures.

The Antigravity tests already cover the heavy semantic surface
(81 tests = ~3.5 tests per public symbol). The new public-API
tests cover the wiring layer that Antigravity didn't write — the
re-exports themselves, the instantiation smoke, and the
documented-enum-membership checks.

## 9. Audit completeness — 7-point rule

| # | Criterion | Status | Notes |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 34 symbols re-exported via `__init__.py`; verified by `test_safety_cert_public_api.py` |
| 2 | Multi-angle tests | ✅ PASS | 95 tests across 2 files; sticky-flag, reset, enum-grading, hash determinism, CCF, HFT, IEC 62304 |
| 3 | Acceleration path | N/A | Not a compute module (document generator + ~µs runtime check). Per `feedback_multi_language_accel.md` exemption for I/O adapters and visualisation. |
| 4 | Benchmarks | ⚠️ WARN | `benchmarks/bench_safety_monitor.py` committed for `SafetyMonitor.check()` (3 scenarios, JSON in `benchmarks/results/`); `CertificationGenerator.build_package()` not yet benchmarked (followup #61) |
| 5 | Performance docs | ✅ PASS | §7 with measured numbers from the benchmark for `check()`; honest "not yet benchmarked" for `build_package()` |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX 2-line header on `__init__.py`, `safety_cert.py`, `safety_monitor.py` (fixed in this batch — `safety_cert.py` had `# mypy: ignore-errors` and 1-line piped SPDX both removed). British English in this doc; source uses standard scientific-Python identifiers (acceptable per docs-vs-code rule). |

Net: **1 WARN, 0 FAIL.**

## 10. Known issues / follow-ups

### 10.1 No committed benchmark (WARN row 4)

Open follow-up (this audit cycle): commit
`benchmarks/bench_safety_monitor.py` reproducing the §7 numbers.
Lower priority than the QA-bridge benchmark because (a) the
runtime check is sub-microsecond, (b) the document generator is
sub-100 ms, (c) neither is a hot path.

### 10.2 `CrossStandardMapper` asymmetry not formally proven

The §3 mapping table asserts SIL ↔ ASIL ↔ Class equivalence.
IEC 61508-5 informative annex calls this "rough", and the actual
mapping depends on controllability + exposure factors that the
mapper currently doesn't take as inputs. A future refinement
should let the user pass a controllability rating (C1/C2/C3 per
ISO 26262) and an exposure rating (E0/E1/E2/E3/E4) so the
mapping reflects the actual ASIL derivation rather than the
worst-case linear mapping.

### 10.3 `IEC62304Assessment` software class is self-declared

The package classifies its own SC-NeuroCore artefacts as
appropriate for Class C (life-threatening), but a notified body
would not accept a self-classification — the device manufacturer
must perform their own hazard analysis. The class returned by
`IEC62304Assessment` is the SC-NeuroCore *bound* (we are
sufficiently rigorous for Class C), not a substitute for the
device manufacturer's HA.

### 10.4 Hardware monitor co-design not documented here

The `SafetyMonitor` Python class mirrors the SystemVerilog
`neuro_safe_monitor` module + `safety_bounds.lean` formal proof.
The mirror relationship is asserted in the code comments but not
verified in this package — a co-simulation test that runs both
the SV monitor (via cocotb) and the Python mirror on the same
input vectors would close that gap.

### 10.5 Audit findings (no semantic bugs)

Audit found:

- `# mypy: ignore-errors` was gratuitous (mypy reports no issues
  on the file). Removed.
- `__init__.py` did not re-export the public API. Wired.
- 1-line piped SPDX header in 3 files. Fixed to 2-line per
  `feedback_spdx_header_format.md`.
- Pre-existing `docs/api/safety_cert.md` was a 16-line stub with
  fabricated import names (`SafetyRequirement`, `RequirementTracer`,
  `FMEDAAnalyzer`, `FormalPropertyLink`, `ReliabilityPredictor`,
  `CertificationAuditor` — none of which exist in the module).
  Replaced with this page in the same batch.

No semantic bugs (sign errors, off-by-ones, wrong invariants)
were found in `safety_cert.py` or `safety_monitor.py`. The 81
Antigravity tests pass; the 14 new public-API tests pass.

## 11. References

- IEC 61508 (Ed 2.0): Functional safety of electrical/electronic/
  programmable electronic safety-related systems. Geneva: IEC.
- ISO 26262 (Ed 2): Road vehicles — Functional safety. Geneva: ISO.
- IEC 62304 (Ed 1.1): Medical device software — Software life
  cycle processes. Geneva: IEC.
- US FDA: Premarket Approval (PMA) for Class III Medical Devices,
  21 CFR Part 814. Washington DC: FDA.
- TÜV SÜD: SIL Compliance Programme — assessor's guide to
  IEC 61508 evidence packages. Munich.

## 12. Audit batch identification

This page was produced as part of the **Antigravity audit, batch
B1, package 1** (per
`docs/internal/antigravity_inventory_2026-04-17.md`). The package
is one of 18 large untracked sub-packages identified for the
Antigravity audit; the others follow in subsequent batches.
