# Audit Index — sc-neurocore

Tracks all audit findings and their resolution status.

---

## audit_model_fidelity_2026-04-05.md (Arcane Sapience)

Fidelity audit of all neuron models against cited publications.

| # | Model | Severity | Status | Resolution |
|---|-------|----------|--------|------------|
| 1 | RetinalGanglionCell | CRITICAL | FIXED | Pillow 2005 GLM — `ef4cc7c` |
| 2 | InnerHairCell | CRITICAL | FIXED | Meddis vesicle pool — `53cbfba` |
| 3 | OuterHairCell | CRITICAL | FIXED | Bidirectional prestin — `d4d22c3` |
| 4 | GranuleCell | CRITICAL | FIXED | D'Angelo 2001 full HH — `44d983b` |
| 5 | AlphaMotorNeuron | MODERATE | FIXED | PIC inactivation + Ca²⁺ — `9f848b1` |
| 6 | RodPhotoreceptor | MODERATE | FIXED | Ca²⁺-GC feedback — `fa05596` |
| 7 | TraubMilesNeuron | MODERATE | FIXED | M-current Kv7 — `a6871e8` |

---

## audit_phase_c_2026-04-05.md (Arcane Sapience)

Kinetics verification for cerebellar + sensory models.

| # | Model | Severity | Status | Resolution |
|---|-------|----------|--------|------------|
| 1 | GolgiCell | CRITICAL | FIXED | Full Solinas 2007, 11 currents — `e94f5cd` |
| 2 | DCNNeuron | MODERATE | FIXED | Added INaP + Ca²⁺-AHP — `cb2480c` |
| 3 | OlfactoryReceptorNeuron | MODERATE | FIXED | Added PDE4 feedback — `b69c8d5` |
| 4 | StellateCell | OK | N/A | WB+Kv3.1 appropriate, no detailed model exists |
| 5 | LugaroCell | OK | N/A | LIF+adaptation appropriate, no HH model exists |
| 6 | UnipolarBrushCell | OK | N/A | LIF+persistent captures key feature |

---

---

## audit_7point_checklist_2026-04-05.md (Arcane Sapience)

Full 7-point checklist audit of all 173 Rust neuron structs. 4 phases, tracked.

| # | Dimension | Status | Scope |
|---|-----------|--------|-------|
| 1 | Pipeline wiring | PASS | 172/172 |
| 2 | Multi-angle tests | **FAIL** | 7 files with ratio ≤1.4 (THIN), ~740 tests needed |
| 3 | Rust path | PASS | 173/173 |
| 4 | Benchmarks | **FAIL** | 63/173 covered (36%), 98 models missing |
| 5 | Performance docs | **FAIL** | Mirrors benchmark gap |
| 6 | Elite docs | **FAIL** | 70 ELITE, 32 ADEQUATE, 61 STUB |
| 7 | Rules followed | **FAIL** | 2 models with 0 tests (StochasticLIF, LeakyCompeteFire) |

**Execution plan:** Phase 1 (P0 missing tests) → Phase 2 (P1 multi-angle ~740 tests) → Phase 3 (P2 benchmarks ~139 functions) → Phase 4 (P3 docs 93 upgrades)

---

*Last updated: 2026-04-05T1509 by Arcane Sapience*
