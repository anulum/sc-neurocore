# ML350 Posner r7 Neutral Continuation

This directory records the reproducible launch package for the ML350 r7
continuation of the Posner neutral closed-shell ORCA geometry optimisation.
It is a live compute lane, not accepted runtime molecular data.

## Source Endpoint

- Source run root: `/home/anulum/sc-neurocore-orca-runs/ml350_r6_seeded_20260507`
- Source output: `run/posner_ml350_neutral_opt_20260507_r6_seeded.out`
- Source endpoint XYZ: `run/posner_ml350_neutral_opt_20260507_r6_seeded.xyz`
- Source endpoint GBW: `run/posner_ml350_neutral_opt_20260507_r6_seeded.gbw`
- Source status: normal ORCA termination with exit status `0`, but no
  `THE OPTIMIZATION HAS CONVERGED` marker.

## Continuation Run

- Run root: `/home/anulum/sc-neurocore-orca-runs/ml350_r6_continuation_20260531`
- Job name: `posner_ml350_neutral_opt_20260531_r7_continue`
- tmux session: `scn_orca_r7_ml350`
- Compute lock: active SC-NeuroCore ORCA lock under `~/compute-queue/`
- Method: `B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq MOREAD`
- Charge/multiplicity: `0 1`
- Resource limit: single ORCA worker, `%maxcore 12000`
- Geometry continuation limit: `%geom MaxIter 300 end`

The run was launched from the endpoint `.xyz` and `.gbw`, not from the initial
generated geometry. `source_sha256.txt` records the exact endpoint coordinate
and wavefunction hashes used at launch.

## Status Probe

Use this on ML350:

```bash
/home/anulum/sc-neurocore-orca-runs/ml350_r6_continuation_20260531/status_probe.sh
```

Acceptance remains fail-closed. The neutral geometry can be promoted only after
all of the following are true in the r7 output:

- exit status is `0`;
- `ORCA TERMINATED NORMALLY` is present;
- `THE OPTIMIZATION HAS CONVERGED` is present;
- the final geometry convergence table reports `YES` for energy change, RMS
gradient, MAX gradient, RMS step, and MAX step.

Only after that should the cation-radical EPR/HFC workflow be launched.
