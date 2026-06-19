# ML350 Posner Research Lane Package

Created: 2026-06-19

This package preserves the curated outputs for the SC-NEUROCORE ML350 Posner
research lane. It intentionally excludes full ORCA scratch trees, restart
matrices, and raw host work directories. Those remain on ML350 under
`/home/anulum/sc-neurocore-orca-runs/`.

## Contents

- `runs/`: accepted ORCA outputs, final geometries where available, inputs,
  timestamps, exit statuses, and provenance scripts/logs copied from ML350.
- `extracts/`: structured JSON parameters extracted from accepted ORCA outputs.
- `notes/`: evidence and handoff notes copied from ignored internal workspace
  notes so the research lane has a tracked public-repo record.
- `SHA256SUMS.txt`: checksums for every file in this package.

## Accepted Completed Outputs

| Run | Job | Status | Scope |
|---|---|---:|---|
| `ml350_r6_seeded_20260507` | `posner_ml350_neutral_opt_20260507_r6_seeded` | superseded diagnostic | Long neutral optimization precursor; retained for continuity, but `r7` is the accepted neutral geometry. |
| `ml350_r6_continuation_20260531` | `posner_ml350_neutral_opt_20260531_r7_continue` | accepted | Converged neutral geometry continuation. |
| `ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix` | `posner_cation_radical_epr_r8c` | accepted | Cation-radical EPR/HFC single point from the `r7` neutral geometry. |
| `posner_followup_20260614T0218Z` | `01_neutral_nmr_r7` | accepted | Neutral dry-cluster NMR constants. |
| `posner_followup_20260614T0218Z` | `02_hydration6_sp` | accepted | Six-water hydration single point from heuristic hydrated candidate. |
| `posner_followup_20260614T0218Z` | `03_dimer_sp` | accepted | First dimer candidate single point. |
| `posner_handoff_extension_20260614T0340Z` | `04_hydration6_nmr` | accepted | Hydrated-cluster NMR evidence. |
| `posner_handoff_extension_20260614T0340Z` | `05_neutral_ir_freq_r7` | accepted | Neutral `r7` harmonic frequency/IR evidence. |
| `posner_handoff_extension_20260614T0340Z` | `06_dimer_far_sp` | accepted | Far-separated dimer reference single point. |
| `posner_tier2_physics_20260614T0718Z` | `07_hydration6_opt` | accepted | Optimized six-water hydrated cluster geometry. |

## Not Yet Accepted

- `posner_tier2_physics_20260614T0718Z/08_dimer_opt` was still running during
  the 2026-06-19 audit and is not promoted here as an accepted result.
- Tier-2 jobs `09_bsse_counterpoise` through `13_dimer_pbe0_sp` had not started
  during the 2026-06-19 audit.
- Failed or superseded cation-radical attempts `r8_fullhost` and `r8b` are not
  part of the accepted lane package; `r8c` is the accepted EPR/HFC run.

## Use Boundaries

- Neutral NMR values and cation-radical EPR/HFC values are different species and
  must not be merged into a single physical claim.
- Dimer single-point and far-dimer outputs are first-pass interaction evidence.
  Binding claims require the pending optimized dimer and BSSE/counterpoise tier.
- Hydration outputs are environment-sensitivity evidence unless explicitly
  marked as optimized geometry evidence.
- Claims must use extracted JSON and package checksums, not manual copying from
  ORCA text logs.
