# SC-NeuroCore Vertex ORCA Lane

Timestamp: 20260506T221817Z

Corrected SC-NeuroCore-only retry for the Posner ORCA neutral geometry lane.
This does not use SCPN-QUANTUM-CONTROL data.

## Corrections

- Preserves MPI root execution overrides for OpenMPI, ORTE, and PRRTE.
- Writes `exit_reason.txt` and a non-zero `exit_status.txt` on SIGTERM/SIGINT
  instead of allowing Vertex cancellation to look like a successful ORCA run.
- Requires both `ORCA TERMINATED NORMALLY` and
  `THE OPTIMIZATION HAS CONVERGED` before returning zero.
- Uses a fresh GCS prefix:
  `gs://gotm-sc-neurocore/sc-neurocore-posner-orca/20260506T221817Z/`

## Job

- Display name: `sc-neurocore-posner-orca-20260506t221817z-r5-corrected`
- Vertex resource: `projects/144846334489/locations/europe-west4/customJobs/4388288207968534528`
- Region: `europe-west4`
- Machine: `n1-standard-16`
- ORCA `%pal nprocs`: 12
- `%maxcore`: 3500 MB
- Timeout: 259200 seconds
- Method: `B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq`
- Initial state: `JOB_STATE_PENDING` at `2026-05-06T22:20:24Z`

## Runner Hash

`8702a1de68f426e4da2d3a48c721e2fd056ccf930eb91d8988509884e2a478f5`

## Acceptance Gate

Do not promote outputs unless the final ORCA `.out` contains both:

- `ORCA TERMINATED NORMALLY`
- `THE OPTIMIZATION HAS CONVERGED`

and `exit_status.txt` is `0`.
