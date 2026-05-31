# ML350 Posner Neutral ORCA Endpoint

This package is curated evidence from the ML350 neutral closed-shell ORCA run.
It is not runtime hyperfine data and must not be used as `hf.json`.

- ORCA output: `/home/anulum/sc-neurocore-orca-runs/ml350_r6_seeded_20260507/run/posner_ml350_neutral_opt_20260507_r6_seeded.out`
- Exit status: `0`
- Accepted neutral geometry: `False`
- Last optimization cycle: `117`
- Final energy: `-9954.01511299552` Eh
- Normal termination marker: `True`
- Geometry convergence marker: `False`

The original promotion gate remains fail-closed: neutral geometry is accepted
only when both `THE OPTIMIZATION HAS CONVERGED` and
`ORCA TERMINATED NORMALLY` are present with exit status 0.
