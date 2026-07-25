# Measured Behaviour Facet

Every model in the catalogue carries a **measured behaviour facet**: a small set
of `behavior_tags` describing how it actually fired when probed across a sweep of
constant-current drives. The tags are produced by simulation, never declared by
hand, and the catalogue may only carry tags from a fixed vocabulary — so the
facet describes observed behaviour rather than an author's claim.

The facet is independent of the citation tier. A model can be fully behaviour-
tagged while remaining Tier 1 for want of a confirmable DOI, and a Tier 3 model
that cannot be driven by a constant current carries no behaviour tags at all.

## What the tags mean

Each tag has an exact, observable definition. A tag is asserted only when the
behaviour was seen at some drive in the sweep.

| Tag | Observed condition |
|---|---|
| `excitable` | fired at least one spike under some tested drive |
| `quiescent` | fired no spikes at any tested drive over the probe window |
| `tonic` | fired regularly (low ISI variability) under some sustained drive |
| `adapting` | spike-frequency adaptation — inter-spike intervals lengthening over time |
| `bursting` | a burst–pause structure (bimodal ISI) under some drive |
| `irregular` | moderate ISI variability under some drive |
| `chaotic` | high ISI variability under some drive |
| `phasic` | only one or two spikes despite sustained drive at some level |
| `rate-coded` | firing rate rose monotonically with drive (an increasing f–I curve) |
| `stochastic` | the spike train was not reproducible between two identical runs |

`excitable` and `quiescent` are mutually exclusive, and a `quiescent` model never
carries a firing-pattern tag.

## How the measurement is taken

The probe (`sc_neurocore.studio.behavior_probe`) sweeps each model across a
scale-robust ladder of constant currents:

```
0.0, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0
```

The ladder spans several decades on purpose. The catalogue mixes unit
conventions — physical millivolt/nanoamp models, normalised dimensionless maps,
and integer fixed-point hardware models — so no single current is meaningful
across all of them. A geometric sweep from rest to strong drive exposes each
model's behavioural envelope regardless of its scale, where a single operating
point would miss it (an adaptive-exponential model, for instance, only fires
tonically under the strong-drive rail).

Each model is simulated over a 200 ms window. The firing pattern at each current
is classified, and the tags are derived from the envelope of classifications
plus the shape of the firing-rate curve.

### Reproducibility and stochastic models

Every observation is taken **twice** from an identical seed. A firing-pattern
tag is asserted only from an observation that reproduced exactly. A model whose
spike train differs between the two runs — one with an internal, unseeded random
source — is flagged `stochastic` and carries only the sign-robust excitability
verdict; its fine pattern tags are withheld because they would flip from run to
run. The derived tags are therefore reproducible by construction.

### Models that cannot be driven

The probe is resilient per `(model, current)`. A model that cannot be driven by a
constant current — one whose `step` needs an extra synaptic input, or one with an
internal stability guard that rejects the operating point — records the failure
for that drive and still yields a profile. A model that fails at every drive is
honestly left with no behaviour tags rather than a fabricated one.

## In the Studio

The model browser shows a **behaviour** filter row built from the facet counts
(`GET /api/models/facets` → `behaviors`). Selecting a tag restricts the list to
models measured to show that behaviour. Unlike the on-demand **Scan** button —
which classifies every model live at a single current — the behaviour facet is
part of each descriptor and is available immediately, with no scan to run.

## Keeping the facet honest

The facet is guarded by two cheap gates and one reproducibility gate:

- Every descriptor's `behavior_tags` must be drawn from the controlled
  vocabulary (`tests/test_behavior_taxonomy.py`).
- Every descriptor's committed tags must equal the recorded measurement in
  `behavior_evidence.json` (`tests/test_behavior_tags.py`), compared without
  re-running a simulation.
- The probe's own reproducibility is proven on a sample of deterministic models
  (`tests/test_behavior_probe_models.py`).

The recorded measurement is regenerated with:

```
python tools/populate_behavior_tags.py            # probe + write tags + manifest
python tools/populate_behavior_tags.py --check     # re-probe and verify only
```

This is the slow offline act — it simulates every model twice per drive. The
committed manifest is what the fast gate checks, so continuous integration never
re-runs the full sweep.
