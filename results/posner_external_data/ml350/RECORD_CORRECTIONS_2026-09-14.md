# Posner ORCA record corrections — 2026-09-14

This file corrects statements in checksummed records that are kept byte-identical so their
`SHA256SUMS.txt` manifests stay valid. Read it together with those records.

## 1. `20260619_posner_research_lane/README.md`

The table "Accepted Completed Outputs" lists `posner_tier2_physics_20260614T0718Z` /
`07_hydration6_opt` as "accepted — Optimized six-water hydrated cluster geometry". That is wrong.
The output ends with ORCA's message that the optimisation did not converge and reached the maximum
number of cycles (120). ORCA terminated normally and the exit status was 0, but the geometry is not a
stationary point. The same applies to `08_dimer_opt`, which the README lists as still running on
2026-06-19; it later stopped at its 160-cycle limit without converging.

Consequences:

- Every quantity computed on the `07` or `08` geometries is not minimum-based. This covers the
  counterpoise binding estimate of −193.28 kcal/mol, the rigid distance scan, the CPCM(Water) and PBE0
  single points, and the frequency runs used to test those geometries.
- The hydrated minimum was later obtained by reoptimising along the dominant imaginary mode and
  confirmed by a frequency run with no imaginary mode. The dimer was reoptimised along its soft
  mode; both displacement branches converged to one structure whose frequency validation is pending.

## 2. `../README.md` (acquisition status)

That document describes the state of May 2026. The r7 neutral continuation it asks to monitor
converged (49 cycles, `THE OPTIMIZATION HAS CONVERGED`, normal termination). A later harmonic
frequency run on the r7 geometry has no imaginary mode, recomputed independently from the Hessian.
The workstation ORCA installation path it names no longer exists. Production runs used the ML350
installation of the same ORCA 6.1.1 build, which is byte-identical to the verified installer archive.

## 3. Structure identity

The converged r7 cluster is a C1 arrangement that differs from the S6 structure used in earlier
Posner spin models (permutation-invariant RMSD 1.49 Å to the published S6 coordinates). Constants
derived from r7 describe that structure, not the S6 prototype.

## Verification

Each correction was read from the ORCA output files on the ML350 archive volume and cross-checked
against the deterministic extraction records kept with the project's internal evidence.
