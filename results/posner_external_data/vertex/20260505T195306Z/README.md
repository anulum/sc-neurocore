# SC-NeuroCore Vertex ORCA Lane

Timestamp: 20260505T195306Z

This directory records the SC-NeuroCore-only Vertex AI Posner ORCA verification
lane. It does not use SCPN-QUANTUM-CONTROL data.

## Vertex Job

- Name: `projects/144846334489/locations/europe-west4/customJobs/815737434812710912`
- Display name: `sc-neurocore-posner-orca-20260505t195306z`
- Status: failed before ORCA launch, container PATH did not expose `gcloud`
- Retry name: `projects/144846334489/locations/europe-west4/customJobs/7833471554162786304`
- Retry display name: `sc-neurocore-posner-orca-20260505t195306z-r2`
- Retry status: Vertex reported success, but ORCA `.out` shows startup error:
  OpenMPI refused root execution before SCF. The runner is patched to set
  `OMPI_ALLOW_RUN_AS_ROOT=1`/`OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` and to fail
  nonzero unless ORCA prints normal termination and optimization convergence.
- Second retry display name: `sc-neurocore-posner-orca-20260505t195306z-r3`
- Second retry name: `projects/144846334489/locations/europe-west4/customJobs/467834363598340096`
- Second retry status: cancelled before useful execution to avoid reusing r2
  output object names
- Third retry display name: `sc-neurocore-posner-orca-20260505t195306z-r3-isolated`
- Third retry name: `projects/144846334489/locations/europe-west4/customJobs/3431202918408126464`
- Third retry status: submitted
- Third retry result: failed before SCF because OpenMPI/PRRTE did not expose
  enough slots for `nprocs 6`. Runner patched with both OpenMPI 4
  `OMPI_MCA_rmaps_base_oversubscribe=1` and PRRTE/OpenMPI 5
  `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe`.
- Third retry output prefix:
  `gs://gotm-sc-neurocore/sc-neurocore-posner-orca/20260505T195306Z/output-r3/`
- Fourth retry display name: `sc-neurocore-posner-orca-20260505t195306z-r4-oversub`
- Fourth retry name: `projects/144846334489/locations/europe-west4/customJobs/1147877907331284992`
- Fourth retry status: `JOB_STATE_RUNNING` as of 2026-05-06; submitted with
  isolated output and MPI oversubscribe env
- Fourth retry output prefix:
  `gs://gotm-sc-neurocore/sc-neurocore-posner-orca/20260505T195306Z/output-r4/`
- Project: `gotm-sc-neurocore`
- Region: `europe-west4`
- Machine: `n1-standard-8`
- Timeout: 24 hours
- Input/output prefix: `gs://gotm-sc-neurocore/sc-neurocore-posner-orca/20260505T195306Z/`

## Inputs

- ORCA archive SHA1: `98490e09ad999792bd23ed7a06a6799aef01fb5a`
- Geometry snapshot SHA256:
  `4bd565effeacda194666189ad464973b6e23390cf0b0cddc245f575cd22a83ea`
- Runner SHA256:
  `98a6f8fbbde24c7ca771ea01e32c6ec3b1b30ab58c99a0fc1ffcb228785f9f6e`
- Vertex YAML SHA256:
  `125ed26d0834d32179b0498a92e954e5197b116f2a478cdf2d9d3583830bbd16`
- Vertex r4 YAML SHA256:
  `6b472e54eba3344cf0211fe785b81c8075ff9233dfb1a4c2c1c2da54eb0b7fa1`

## Method

`B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq`

Charge/multiplicity: neutral closed shell, `0 1`.

## Purpose

This Vertex lane is the auditable neutral-geometry acquisition run for the
SC-NeuroCore Posner verification model. It exists because the downstream
simulator and IBM circuits should not use estimated geometry-dependent spin
tensors. The run must either produce a converged neutral `Ca9(PO4)6` geometry
under the stated ORCA method or fail loudly with enough output to diagnose the
next chemistry step.

The run does not validate the biological hypothesis by itself. It is the first
external-data gate needed before cation-radical HFC extraction, runtime JSON
validation, simulator parity checks, and any IBM QPU spend.

## Completion Checklist

When r4 leaves `JOB_STATE_RUNNING`, do all of the following before using any
output:

1. Pull the isolated output prefix:

   ```bash
   gcloud storage cp --recursive \
     gs://gotm-sc-neurocore/sc-neurocore-posner-orca/20260505T195306Z/output-r4 \
     results/posner_external_data/vertex/20260505T195306Z/
   ```

2. Inspect the runner status:

   ```bash
   sed -n '1,80p' \
     results/posner_external_data/vertex/20260505T195306Z/output-r4/output/exit_status.txt
   ```

3. Inspect the ORCA output markers:

   ```bash
   rg -n 'ORCA TERMINATED NORMALLY|THE OPTIMIZATION HAS CONVERGED|ORCA finished by error|ERROR|FINAL SINGLE POINT ENERGY' \
     results/posner_external_data/vertex/20260505T195306Z/output-r4/output/posner_vertex_neutral_opt_20260505T195306Z_r4.out
   ```

4. Accept the neutral geometry only if the ORCA output contains both:

   - `ORCA TERMINATED NORMALLY`
   - `THE OPTIMIZATION HAS CONVERGED`

5. Record the final hashes for the accepted `.out`, `.xyz`, `.gbw`, manifest,
   and pulled output directory.

6. If converged, use the final neutral XYZ as the input for the cation-radical
   doublet sequence:

   - vertical EPR at neutral geometry;
   - relaxed cation-radical optimization;
   - relaxed cation-radical EPR.

7. Parse only completed EPR output into `hf.json` and
   `extended.partial.json`; then complete `extended.json` with transport,
   cage-dephasing, incorporation, calcium, and electron-map inputs.

8. Run `validate-runtime` before any simulator or IBM QPU work.

If the run fails or times out, keep the artifacts, update this README with the
observed failure mode, and do not promote intermediate geometry.

## Local Lane Note

The local `results/posner_external_data/orca/` run was launched before the
generator default was tightened and used `TightSCF`. It reached geometry cycle
21 without convergence and was stopped as non-final. Do not use the local
`TightSCF` artifacts as runtime Posner data. Use a captured `VeryTightSCF`
ORCA output with both `ORCA TERMINATED NORMALLY` and
`THE OPTIMIZATION HAS CONVERGED`.

## Cost Estimate

Vertex AI custom training in europe-west4 lists `n1-standard-8` at about
USD 0.57 per hour, plus small disk/storage/logging charges. A 12-24 hour run
is therefore expected to consume roughly USD 7-14 before credits.
