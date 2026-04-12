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
| 2026-04-06 | research_attention_residuals_2026-04-06.md | AttnRes (Kimi/Moonshot) adaptation proposal | PROPOSAL | Claude |

---

## audit_security_report.md + bandit_2026-04-12.json (Gemini B3, audited by Arcane Sapience)

Bandit scan: 9079 total findings, **14 MEDIUM** triaged 2026-04-12.

| # | File | Lines | Finding | Verdict | Reason |
|---|------|-------|---------|---------|--------|
| 1 | equation_builder.py | 209,226,243,247 | `eval()` | ACCEPT RISK | Sandboxed (`__builtins__: {}`), internal ODE equations only |
| 2 | studio/analysis.py | 144,145 | `eval()` | ACCEPT RISK | Same sandboxed pattern, phase plane analysis |
| 3 | test_holonomic_jax_compiler_edges.py | 647 | temp file | FALSE POSITIVE | Tests path escape protection |
| 4 | test_serve_server.py | 104,119,174 | urlopen | FALSE POSITIVE | Localhost test server |
| 5 | test_studio_synthesis.py | 110 | temp file | FALSE POSITIVE | Tests error handling |
| 6 | cosim_q88_vs_pytorch.py | 100 | `torch.load(weights_only=False)` | ACCEPT RISK | Own checkpoints with metadata |
| 7 | cosim_q88_vs_pytorch.py | 116 | HF load_dataset no revision | ACCEPT RISK | SHD dataset, deterministic |
| 8 | extract_shd_weights.py | 160 | `torch.load(weights_only=False)` | ACCEPT RISK | Own checkpoints |

**Summary:** 0 FAIL, 5 FALSE POSITIVE, 8 ACCEPT RISK. No immediate fixes required.

*Triaged: 2026-04-12 by Arcane Sapience*
