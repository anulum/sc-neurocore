<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Posner r8c cation-radical EPR/HFC evidence -->

# Posner r8c — Cation-Radical EPR/HFC Evidence (Extracted Constants)

**Scope marker — read first.** Every constant in this note comes from the
**r8c cation-radical EPR/HFC single point** (charge +1, doublet, UKS
B3LYP/def2-TZVP). These are **EPR g-tensor and hyperfine (A-tensor) values for
the open-shell oxidised cluster**. They are **not** neutral-cluster NMR
shielding constants or spin–spin (J) couplings, and they are **not** evidence
for QCP-3 NMR verification. The neutral closed-shell species used for NMR is a
separate run (`r7`, charge 0). Do not promote these numbers into any
neutral-NMR claim.

All values below were produced by `tools/quantum/extract_orca_params.py`
parsing the ORCA output directly; no constant was hand-copied. The machine
artefact is `docs/internal/posner_r8c_cation_radical_epr_params_2026-06-14.json`
(SHA-256 `93c60cc55097f9849eeddde0cd111c5633d16ada1a931b7429bb72a6074c6dfe`).

---

## 1. Provenance

| Field | Value | Source |
|---|---|---|
| Run ID | `ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix` | run directory |
| Host | `god-of-the-math` (ML350, `192.168.1.30`) | `REPRODUCIBILITY_LOG.md` |
| ORCA version | 6.1.1 (RELEASE) | parsed from output |
| Route line | `UKS B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 SP` | parsed from output |
| Hartree–Fock type | UHF | parsed from output |
| Charge / multiplicity | +1 / 2 (doublet) | parsed from output |
| Electrons / basis dimension | 461 / 1290 | parsed from output |
| `%eprnmr` request | `gtensor true`; `P {aiso, adip, aorb}`; `Ca {aiso, adip}`; `printlevel 5` | `REPRODUCIBILITY_LOG.md` |
| MPI ranks | 24 (`HWLOC_COMPONENTS=-gl`, `use_hwthreads_as_cpus=1`) | `REPRODUCIBILITY_LOG.md` |
| Started / finished (UTC) | 2026-06-13T20:51:41 / 2026-06-14T00:52:34 | run metadata files |
| Exit status | 0 | `output/exit_status.txt` |
| Termination | `****ORCA TERMINATED NORMALLY****` | parsed from output |
| Total run time | 0 days 4 hours 0 minutes 52 seconds 740 msec (14452.74 s) | parsed from output |

### Source hashes (SHA-256)

| Role | File | SHA-256 |
|---|---|---|
| ORCA output | `output/posner_cation_radical_epr_r8c.out` | `71ea14b90ec596e59bdb31607ed6a33644ff0c89a6a02efe72628133cfaf8c89` |
| Reproducibility log | `REPRODUCIBILITY_LOG.md` | `7328c0bb6f21d99cc4d521041e9494f16c0a427886fe96e97efa768cd91b0508` |
| ORCA input | `run/posner_cation_radical_epr_r8c.inp` | `c6f58d96400255a048b27dd8976663055947f0fffa27365722ba28a646e5a525` |
| Input geometry | `run/input.xyz` | `3ef73733b0eff3b81f1487cc2fe8c113fd9b6128e7ce28e9d1528dac7671e8e9` |

The `.inp` and `input.xyz` hashes recomputed at source equal the hashes recorded
in `REPRODUCIBILITY_LOG.md`, confirming the parsed output corresponds to the
logged input. The geometry `input.xyz` is the converged neutral `r7` structure
carried over as the cation-radical single-point input.

---

## 2. Energies

| Quantity | Value (Eh) |
|---|---|
| Final single-point energy (cation radical) | -9953.72619277419 |
| EPR/NMR property-module reference energy | -9953.479258238438 |

For context only (separate neutral `r7` run, not part of this evidence):
neutral final single-point energy -9954.022351784299 Eh, dipole magnitude
10.392594719 D. The cation-radical sits ~0.296 Eh (~8.06 eV) above the neutral
single-point energy; this difference mixes the vertical ionisation with the
basis/method and is **not** a converged adiabatic ionisation energy.

---

## 3. Electronic g-tensor

Raw electronic g-matrix (dimensionless):

```
 2.0251815   -0.0060895    0.0127598
-0.0063840    2.0679999   -0.0347135
 0.0131680   -0.0328507    2.0382556
```

| Quantity | Component 1 | Component 2 | Component 3 | Isotropic |
|---|---|---|---|---|
| g (principal) | 2.0114962 | 2.0276218 | 2.0923195 | 2.0438125 |
| Δg (principal) | 0.0091769 | 0.0253025 | 0.0900002 | 0.0414932 |

The g-shift is positive and anisotropic, consistent with an oxygen-centred
oxidised radical rather than a phosphorus-centred one.

---

## 4. Hyperfine coupling — ³¹P (6 nuclei, aiso + adip + aorb)

All values in MHz. `A_iso` is the isotropic total; `A(FC)` is the Fermi-contact
term; `A(Tot)` are the three principal values; `A_orb,iso` is the isotropic
orbital (`A(PC)`) contribution.

| Atom index | A_iso | A(FC) | A(Tot) principal | A_orb,iso |
|---:|---:|---:|---|---:|
| 9  | -31.7412 | -31.8049 | -29.5120 / -31.8702 / -33.8414 | 0.0633 |
| 10 | -0.5423  | -0.5431  | -0.3271 / -0.6109 / -0.6889    | 0.0007 |
| 11 | -2.3100  | -2.3085  | -2.1158 / -2.2739 / -2.5404    | -0.0017 |
| 12 | -7.0536  | -7.0586  | -6.2009 / -7.0523 / -7.9076    | 0.0049 |
| 13 | -63.0283 | -63.2979 | -55.9890 / -61.6486 / -71.4474 | 0.2688 |
| 14 | -8.1512  | -8.1676  | -7.7570 / -7.7881 / -8.9085    | 0.0161 |

³¹P summary: A_iso range [-63.0283, -0.5423] MHz; largest magnitude
|A_iso| = 63.0283 MHz at atom index 13; smallest |A_iso| = 0.5423 MHz at atom
index 10; mean A_iso = -18.8044 MHz. The spin density is strongly localised on
one phosphate (atom 13), with the remaining ³¹P couplings an order of magnitude
smaller.

---

## 5. Hyperfine coupling — ⁴³Ca (9 nuclei, aiso + adip)

All values in MHz. The orbital term was not requested for Ca, so `A_orb` is
absent (`null` in the JSON).

| Atom index | A_iso | A(FC) | A(Tot) principal |
|---:|---:|---:|---|
| 0 | 1.2356 | 1.2356 | 0.8959 / 1.3761 / 1.4348 |
| 1 | 1.8511 | 1.8511 | 1.4826 / 1.9502 / 2.1204 |
| 2 | 0.0479 | 0.0479 | 0.0309 / 0.0465 / 0.0663 |
| 3 | 1.5593 | 1.5593 | 1.4163 / 1.5800 / 1.6816 |
| 4 | 0.6043 | 0.6043 | 0.2604 / 0.7293 / 0.8231 |
| 5 | 0.6987 | 0.6987 | 0.4814 / 0.7954 / 0.8192 |
| 6 | 0.9473 | 0.9473 | 0.8116 / 0.9918 / 1.0385 |
| 7 | 0.1382 | 0.1382 | -0.0401 / 0.2151 / 0.2395 |
| 8 | 0.5491 | 0.5491 | 0.4544 / 0.5679 / 0.6250 |

⁴³Ca summary: A_iso range [0.0479, 1.8511] MHz; largest |A_iso| = 1.8511 MHz at
atom index 1; mean A_iso = 0.8479 MHz. Calcium couplings are small and positive,
as expected for closed-shell-like spectator cations carrying little spin
density.

---

## 6. Relationship to the QCP follow-up plan

| Task | What it needs | Does r8c supply it? |
|---|---|---|
| QCP-3 NMR Parameter Extraction | Neutral-cluster ¹H/³¹P chemical shifts and J-couplings vs experimental ACP | **No.** r8c is an open-shell cation EPR/HFC run, not an NMR shielding/J run, and not on the neutral species. |
| QCP-5 IBM Hardware Integration | HF/dipolar constants mapped to `FisherPosnerQuantumBridge` to unblock IBM Heron | **Partially.** r8c provides verified ³¹P/⁴³Ca hyperfine A-tensors and the g-tensor for the radical, but the bridge mapping and the dipolar (through-space) coupling table are not in scope of this extraction. |

This extraction is therefore a clean, hashed, machine-readable evidence slice
for the cation-radical EPR/HFC constants. It does not, on its own, close QCP-3
or QCP-5.

---

## 7. Reproduction

The parser uses only the Python standard library, so it runs with any Python
3.12 interpreter (here on the ML350 host where the run lives, to keep the
provenance paths canonical):

```bash
python3 tools/quantum/extract_orca_params.py \
  --input  <run>/output/posner_cation_radical_epr_r8c.out \
  --source <run>/REPRODUCIBILITY_LOG.md \
  --source <run>/run/posner_cation_radical_epr_r8c.inp \
  --source <run>/run/input.xyz \
  --output docs/internal/posner_r8c_cation_radical_epr_params_2026-06-14.json
```

The output JSON is deterministic (sorted keys, fixed indentation); re-running
against the same inputs reproduces byte-identical output and the SHA-256 in the
header.
