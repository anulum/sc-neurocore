# Posner External Data Acquisition Status

This directory contains acquisition artefacts, not runtime verification data.

## Purpose

The purpose of this workflow is to replace phenomenological or heuristic
Posner-model constants with auditable external data before any serious IBM
QPU validation. The immediate target is the publication-grade molecular data
layer for the SC-NeuroCore Posner verification model:

- optimized neutral `Ca9(PO4)6` geometry;
- cation-radical doublet hyperfine tensors for the electron-hole radical state;
- orientation-specific 31P-31P nuclear dipolar tensors from the optimized
  geometry;
- calcium hyperfine tensors and electron-site assignment data where available;
- explicit provenance for values that cannot be derived from ORCA alone.

The output of this workflow is not a claim that the biological Posner
hypothesis is true. It is a disciplined parameter-acquisition and validation
pipeline so that the downstream simulator and IBM circuits are driven by
traceable molecular and backend data rather than silent placeholders.

## What Has Been Done

- Installed official ORCA 6.1.1 outside the repository and verified the forum
  SHA1 for the archive.
- Confirmed `/usr/bin/orca` is GNOME Orca and must not be used for chemistry.
- Wrote `results/posner_external_data/orca/orca_env.sh` so acquisition runs use
  the real ORCA binary through `ORCA_QC_BIN`.
- Verified ORCA starts correctly with a serial smoke calculation.
- Corrected generated ORCA inputs from the rejected `Grid5` keyword to
  `DefGrid3`, which ORCA 6.1.1 accepts.
- Corrected the ORCA worker default to `nprocs 6` for the local host instead
  of assuming eight physical slots.
- Added charge/multiplicity validation so the invalid neutral doublet is not
  generated. The all-electron neutral model has 462 electrons; the radical
  workflow uses charge `+1`, multiplicity `2`.
- Added the cation-radical relaxed optimization deck and relaxed-geometry EPR
  template.
- Tightened generated acquisition decks to default to `VeryTightSCF` through an
  explicit `--scf` option.
- Extended ORCA parsing so optimized geometry can produce full tensor
  `nuclear_dipolar_pairs` instead of scalar dipolar placeholders.
- Extended runtime validation so `hf.json` and `extended.json` must contain
  complete numeric tensor fields before simulator or QPU use.
- Added IBM calibration acquisition through `qiskit-ibm-runtime`; dependency is
  installed, but the stored IBM token was rejected by IBM and must be rotated.
- Created an isolated SC-NeuroCore-only Vertex AI ORCA lane. It uses
  `gs://gotm-sc-neurocore/sc-neurocore-posner-orca/...` and does not use
  SCPN-QUANTUM-CONTROL data.
- Stopped and labeled the local `TightSCF` ORCA lane as non-final after it
  failed geometry convergence and entered cycle 21. Those artifacts are only
  exploratory/preconditioning data.
- Submitted Vertex r4 with isolated output, root-safe OpenMPI settings, PRRTE
  oversubscription settings, captured ORCA stdout, and explicit marker checks
  for normal termination and geometry convergence.
- Processed the completed ML350 r6 seeded neutral endpoint into
  `ml350/20260507_r6_seeded/`. ORCA exited with status `0` and printed
  `ORCA TERMINATED NORMALLY`, but the last geometry table still failed the
  convergence gate and did not print `THE OPTIMIZATION HAS CONVERGED`.

## Why These Changes Matter

The Posner verification stack is only scientifically useful if every runtime
constant is either externally derived, directly measured, or explicitly labeled
as unavailable. The earlier state mixed correct circuit work with unresolved
data dependencies. That is not enough for meaningful IBM validation because a
QPU run would only validate the behavior of a chosen parameterization, not the
physical plausibility of the model.

The corrected workflow enforces three gates:

1. ORCA must converge the relevant molecular geometry and print normal
   termination.
2. Parsed molecular outputs must satisfy strict tensor-shape and numeric
   validation.
3. IBM backend calibration must be captured with a valid SC-NeuroCore IBM
   Runtime credential before any QPU submission.

Until all three gates pass, results remain acquisition/preparation artifacts.

## ML350 r6 Seeded Neutral Endpoint

The local ML350 lane completed on 2026-05-30 after 117 geometry cycles:

- Run path:
  `/home/anulum/sc-neurocore-orca-runs/ml350_r6_seeded_20260507`
- Method:
  `B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq`
- Charge/multiplicity: neutral closed shell, `0 1`
- Exit status: `0`
- Final energy: `-9954.015112995519 Eh`
- Total runtime: `22 days 16 hours 52 minutes 5 seconds 565 msec`
- Marker present: `ORCA TERMINATED NORMALLY`
- Marker absent: `THE OPTIMIZATION HAS CONVERGED`

Curated processing output is in `ml350/20260507_r6_seeded/`:

- `neutral_geometry.json`: marker counts, final energy, final geometry
  convergence table, endpoint P-P distances, and geometry-derived
  orientation-specific 31P-31P dipolar tensors.
- `extended.geometry.partial.json`: geometry-only partial extended payload with
  the 15 tensor dipolar pairs and explicit runtime-missing fields.
- `neutral_endpoint.xyz`: final endpoint coordinates copied from the ML350 run.

This endpoint is useful for diagnosis and continuation, but it is not accepted
as runtime molecular data. The original promotion gate remains fail-closed:
neutral geometry requires both `THE OPTIMIZATION HAS CONVERGED` and
`ORCA TERMINATED NORMALLY` with exit status `0`.

## Vertex Run History

The previous molecular acquisition lane was Vertex r6:

- Job:
  `projects/144846334489/locations/europe-west4/customJobs/2516198137865961472`
- Project: `gotm-sc-neurocore`
- Region: `europe-west4`
- Output prefix:
  `gs://gotm-sc-neurocore/sc-neurocore-posner-orca/20260507T032756Z-r6-resync/`
- Method:
  `B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq`
- Charge/multiplicity: neutral closed shell, `0 1`
- Last checked state on 2026-05-09: `JOB_STATE_RUNNING`
- Start time: `2026-05-09T05:09:37Z`
- Explicit timeout: `259200s` (deadline `2026-05-12T05:09:37Z`)
- Latest observed output snapshot: cycle 27 completed, latest final single
  point energy `-9953.816434214637`; neither `ORCA TERMINATED NORMALLY` nor
  `THE OPTIMIZATION HAS CONVERGED` had appeared at the latest check.

That cloud lane was retained only as history after the local ML350 lane became
the preserved source of the r6 seeded endpoint.

## Molecular Data

Prepared ORCA quantum-chemistry input files are in `orca/`.

Current status:
- Quantum-chemistry ORCA 6.1.1 is installed at
  `/home/anulum/.local/opt/orca-qc/orca_6_1_1_linux_x86-64_shared_openmpi418/orca`.
- `/usr/bin/orca` is still the GNOME screen reader, not the ORCA chemistry
  package.
- OpenMPI, xTB, OpenBabel, and the Python chemistry workflow dependencies are
  installed and verified in the SC-NeuroCore environment.
- The ORCA 6.1.1 archive was downloaded from the logged-in ORCA forum session
  and verified against forum SHA1 `98490e09ad999792bd23ed7a06a6799aef01fb5a`.
- A serial ORCA smoke calculation terminated normally with ORCA 6.1.1.
- ORCA 6.1.1 accepts `DefGrid3`; it rejects the older generated `Grid5`
  keyword as an input error, so acquisition inputs now use `DefGrid3`.
- ORCA/OpenMPI refused `nprocs 8` on this 6-physical-core host; acquisition
  inputs now default to `nprocs 6` rather than oversubscribing.
- The generated geometry is an initial guess only.
- `hf.json` and a complete `extended.json` must not be produced until completed
  ORCA outputs have been parsed.
- Live Posner verification now requires `extended.json` to include
  full tensor `nuclear_dipolar_pairs` in the same dimensionless circuit units
  as `hf.json`; the built-in geometry table is only a test/reference fallback.

Publication-grade acquisition workflow:

1. Source the installed ORCA environment:

   ```bash
   source results/posner_external_data/orca/orca_env.sh
   ```

   `ORCA_QC_BIN` points to the real ORCA executable. Do not use `/usr/bin/orca`.
2. Run `00_posner_neutral_opt.inp` to convergence and export the optimised XYZ.
3. Run the cation-radical doublet workflow. The all-electron neutral
   Ca9(PO4)6 model has 462 electrons, so charge 0/multiplicity 2 is invalid.
   The electron-hole radical state is charge +1/multiplicity 2:
   - `01_posner_cation_radical_epr.inp.template`: vertical HFC at the neutral
     geometry.
   - `01_posner_cation_radical_relaxed_opt.inp`: relaxed cation-radical
     geometry.
   - `02_posner_cation_radical_relaxed_epr.inp.template`: relaxed-geometry HFC.
4. Parse the completed HFC output and the geometry used for the selected HFC
   calculation together:

   ```bash
   ./.venv/bin/python tools/acquire_posner_external_data.py parse-orca \
     path/to/orca_epr.out \
     --optimized-xyz path/to/00_posner_neutral_opt.xyz \
     --out-dir results/posner_external_data/parsed
   ```

   This creates `hf.json` and an `extended.partial.json` containing the
   orientation-specific 31P-31P dipolar tensors computed from the optimised
   geometry. Scalar dipolar magnitudes are not accepted for runtime use.
5. Complete `extended.json` with externally derived values for:
   `incorporation_tensors`, `transport_depolarizing_rates`,
   `cage_dephasing_rate`, `ca43_hf_tensors`, and `ca_electron_map`.
6. Validate before any simulator/QPU run:

   ```bash
   ./.venv/bin/python tools/acquire_posner_external_data.py validate-runtime \
     --hf-json results/posner_external_data/parsed/hf.json \
     --extended-json results/posner_external_data/parsed/extended.json
   ```

## Published Comparison Data

Published Posner work contains data that can be used for validation and
comparison, but it must not be silently substituted for SC-NeuroCore runtime
parameters. Any imported numeric table needs provenance, units, source DOI,
extraction notes, and a parser/fixture test before it can become a comparison
fixture.

Usable comparison sources:

- Swift, Van de Walle, and Fisher 2018 is the primary structural baseline for
  the `Ca9(PO4)6` workflow. It provides first-principles structure, vibrational
  spectra, cation-interaction, pair-binding, and nuclear-spin context. Use it
  to compare the final ORCA geometry, point-group assumptions, P-P distance
  matrix, P-O/Ca-O statistics, and vibrational signatures. Do not use the
  current generated coordinate table as final evidence; it remains an initial
  guess until the r6 ORCA markers pass.
- Player and Hore 2018 is a spin-dynamics benchmark, not a geometry source.
  It publishes Posner scalar-coupling assumptions and a 37-minute idealised
  entanglement-lifetime upper bound. Use it to compare downstream simulator
  behaviour once molecular tensors and IBM calibration are available.
- Agarwal, Aiello, Kattnig, and Banerjee 2021 is a required control source
  because it challenges the high-symmetry Posner assumption and reports
  predominantly low-symmetry room-temperature structures. Use it to test
  whether our final neutral geometry remains near the high-symmetry baseline
  or relaxes toward low-symmetry configurations.
- Agarwal, Kattnig, Aiello, and Banerjee 2023 extends that challenge into
  spin dynamics and calcium-phosphate dimer comparisons. Use it as a negative
  or alternative-structure benchmark, especially for entanglement-decay claims.
- The 2025 pure/doped Posner coherence paper is directly relevant to coupling
  constants: it reports ORCA-based J-coupling calculations for pure and
  lithium-doped Posner models. Its public article states that generated or
  analysed data is available from the corresponding author, so it is a
  comparison target but not yet a local fixture. Fetch the tables or author
  data package before citing any exact numeric constants from it in runtime
  validation.

Comparison plan after r6 converges:

1. Extract the final neutral geometry and reject it unless ORCA printed both
   normal-termination and optimisation-convergence markers.
2. Compute a reproducible geometry report: P-P distance matrix, P-O and Ca-O
   distance summaries, centre-of-mass alignment, point-group/symmetry residual,
   and RMSD against the Swift structural baseline where the source data is
   licensed or manually entered with provenance.
3. Compare the geometry against both the high-symmetry Swift baseline and the
   low-symmetry Agarwal ensemble framing. If r6 lands in a low-symmetry basin,
   document that explicitly and do not force it into a high-symmetry model.
4. Run the vertical and relaxed cation-radical EPR decks from the accepted
   neutral geometry and parse full hyperfine tensors.
5. Compare parsed hyperfine, dipolar, and scalar-coupling values against
   published tables only after their units, sign conventions, atom ordering,
   and extraction provenance are recorded.
6. Keep published numbers in comparison fixtures; promote only our validated
   ORCA-derived tensors into `hf.json` and `extended.json` runtime data.

Detailed internal tracking is in
`docs/internal/posner_prior_art_comparison_plan_2026-05-09.md`.

## Follow-Up Tasks After ML350 r6 Endpoint Processing

1. Continue the neutral optimization from the ML350 endpoint `.gbw`/`.xyz`
   rather than restarting from the original generated geometry.
2. Accept the neutral geometry only if both markers are present:

   - `ORCA TERMINATED NORMALLY`
   - `THE OPTIMIZATION HAS CONVERGED`

3. If a continuation converges, archive the final neutral XYZ and use it as the starting
   point for the cation-radical doublet workflow:

   - vertical radical EPR at neutral geometry;
   - relaxed cation-radical optimization;
   - relaxed cation-radical EPR.

4. Parse the selected completed EPR output with the geometry used for that EPR
   calculation, then validate `hf.json` and complete `extended.json`.

5. Rotate or repair the SC-NeuroCore IBM Runtime credential, then acquire an
   IBM backend calibration snapshot.

6. Only after molecular JSON validation and IBM calibration acquisition, run
   simulator parity checks and then decide whether to spend IBM QPU budget.

## Failure Handling

If r6 fails before ORCA starts, inspect the Vertex logs and runner environment.
If ORCA starts but fails before convergence, preserve the full output and decide
whether the failure is an input formulation issue, an optimizer instability, or
only an insufficient wall-time issue. Do not promote intermediate `.xyz`,
`.gbw`, or trajectory frames to runtime data.

If the neutral optimization remains unstable, the scientifically defensible
next route is staged geometry preparation: use a cheaper method only as a
preconditioner, then rerun the final neutral geometry at
`B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq` and derive
runtime constants only from the final high-level ORCA outputs.

Primary sources used for the acquisition protocol:
- Swift, Van de Walle & Fisher, *Phys. Chem. Chem. Phys.* 20, 12373 (2018),
  DOI `10.1039/C7CP07720C`.
- Fisher, *Annals of Physics* 362, 593-602 (2015),
  DOI `10.1016/j.aop.2015.08.020`.

## IBM Calibration Data

Current status:
- `qiskit-ibm-runtime` is installed in the active virtual environment.
- Vault-backed acquisition reached IBM Runtime, but IBM rejected the stored
  `IBM Quantum` API key as not found on 2026-05-05.
- Calibration acquisition is therefore blocked on credential rotation, not code
  or dependency availability.

Use:

```bash
./.venv/bin/python tools/acquire_posner_external_data.py acquire-ibm \
  --backend ibm_fez \
  --credential-vault /media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md \
  --vault-section "IBM Quantum"
```

The resulting calibration JSON belongs under `results/posner_external_data/ibm/`.
