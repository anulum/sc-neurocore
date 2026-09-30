# Parameter Fitting

`sc_neurocore.fitting` fits the parameters of a Universal DSL model — a
catalogue model's canonical schema, or a [candidate's](candidate-models.md)
model — to recorded responses, and states how far the fit can be trusted. The
Studio serves it at `POST /api/fits`.

## The problem

A fit names, explicitly:

- the **model** and the **observed state variable** (for example `v`);
- a **domain** for every fitted parameter: finite bounds, and `log` scale for a
  parameter whose plausible values span decades;
- **fixed** values for any other parameter to set, left otherwise at the
  model's own values;
- a **cohort** of recordings — each a current per step and the observed
  variable after each step — split into a **training** set and a **hold-out**
  set;
- a **seed**.

The split is part of the problem. The objective sees the training recordings
only; a recording whose data appears in both sets, or two recordings with the
same name, is refused. The model runs under its own declared profile through
the Universal DSL.

## The fit

The objective is the mean squared residual of the observed variable over the
training recordings. It is minimised by seeded differential evolution in each
parameter's search space (logarithmic for `log` domains), then polished
locally. The best loss of every generation is kept as the optimiser's history.

A **failed trial** — a state that stopped being finite, or a mean squared
residual of 10¹⁵⁰ or more — is counted and reported, never replaced by a
default. A fit that found no finite trial does not report convergence.

The fitted parameters are then run on the hold-out recordings, and their
error is reported per recording; a recording the fitted model cannot follow
finitely is reported as diverged.

## Identifiability and uncertainty

At the optimum the residual Jacobian is taken by central differences in search
space. The eigenvalues of `JᵀJ` say which parameter combinations the data
constrain: a direction whose eigenvalue is below 10⁻⁸ of the largest is
reported as **unconstrained**, with the combination it moves. The bundled
leaky integrate-and-fire schema shows why this matters: its dynamics see
resistance and capacitance only as `R / C`, so fitting both reports one
unconstrained direction along which `log R` and `log C` move together.

When every direction is constrained, standard errors follow from the
Gauss–Newton asymptotic covariance `s² (JᵀJ)⁻¹`, with `s² = RSS / (n − p)`,
mapped through the logarithm by the delta method for `log` parameters, and the
correlation matrix is reported. When any direction is unconstrained, **no**
standard error or correlation is reported: a covariance of a singular problem
would state a certainty the data do not support.

On the benchmark in the test suite — the leaky integrate-and-fire schema
driven below threshold by three current steps with 0.2 mV Gaussian noise, two
steps for training and one held out — the fit recovers the resting
potential, the membrane time constant and the resistance within four stated
standard errors, and the hold-out error is about the noise level.

Ordinary Gaussian standard errors are also withheld when the optimum is at a
search-domain boundary or the training data have no residual degrees of
freedom (`n <= p`). Local data identifiability is still reported separately;
it does not by itself justify an uncertainty estimate.

A nonfinite residual Gram matrix also produces a numerical identifiability
refusal and no covariance, rather than exporting NaN error bars.

## Replay

The result carries the whole problem — model, domains, fixed values, every
recording, the seed and the optimiser size — the software versions, and one
`result_sha256`. `POST /api/fits/replay` runs it again and reports whether the
new digest equals the exported one. The optimiser is seeded and runs in one
process, so a replay on the same versions reproduces the digest.

## In the Studio

The **Fitting** view fits either the selected catalogue model's canonical schema or
the workspace's candidate draft. Parameters to fit are typed by name with
their bounds and scale — the names are the schema's, which can differ from the
catalogue class's constructor arguments, and an unknown name is refused with
the reason. Fixed values are given as `name=value` lines. Recordings are CSV
files with a current and an observed value per line (a header line is
allowed); each is assigned to the training or the hold-out set, and a file
with a line that is not two numbers is refused with that line. The result
lists each fitted value with its standard error, or *not stated* when a
parameter combination is unconstrained, names that combination, gives the
hold-out error per recording and the optimiser's generations, trials and
failed trials, and can be exported and replayed.

`POST /api/fits` takes the problem as JSON: `catalogue_model` **or** `schema`,
`observable`, `domains`, `fixed`, `train`, `holdout`, `seed`, `generations`
(1–500) and `population` (4–100). A fit runs synchronously, so its size is
bounded before it starts: the number of model steps it can take is estimated
as `population × parameters × (generations + 2) × training samples`, and a fit
estimated above 3 000 000 steps is refused with HTTP 422, `fit_too_large` and
the estimate. An invalid problem is refused with `invalid_fit` and the reason.
A replay, background job or cohort document with a missing field or a wrong
JSON type is refused with HTTP 422 and one fixed sentence ("a required field
is missing or has the wrong JSON type"); responses never carry exception text.
With route policies enforced, both routes require an authenticated principal.

## What a fit does not establish

- A good hold-out error on the recordings given; a different protocol can
  still separate models the cohort cannot.
- Identifiability beyond the local, linearised analysis at the optimum; a
  second optimum elsewhere in the domain is not excluded.
- Anything about hardware: a fitted parameter set has no fixed-point, RTL or
  co-simulation evidence of its own.
- Hardware accuracy/latency/resource/energy evidence from fitting alone.
  The separate receipt comparison below requires external measurements.

## Background jobs and acquisition groups

The Fit panel submits `POST /api/fits/jobs` and observes the returned
`/api/fits/jobs/{job_id}` route. Larger fits run in the existing bounded process
worker, with cancellation through `POST /api/fits/jobs/{job_id}/cancel`, durable
`experiment.json`, `result.json` and `history.jsonl` artifacts, and explicit
failed, cancelled or timed-out status. Reopening the panel in the same browser
session recovers the job. With route policies enabled, only its authenticated
owner can read or cancel it; another caller receives 404.

In isolated storage mode, the registered `laboratory.run` task derives its
owner from the authenticated requester at both the API and storage authority.
The authority independently checks ownership and laboratory admission before
returning a record or cancelling a job. A denied cancellation never signals
the API's local supervisor. Existing service tasks retain their service owners.
Same-UID integration tests exercise the authority, launcher and worker; they
do not establish Linux privilege separation.

The background admission estimate is limited to 200 million model steps. This
estimate is not a wall-time guarantee: local polishing can take additional
iterations. The process supervisor's time and resource limits remain decisive.
The synchronous compatibility routes retain the 3 million-step estimate;
replays now pass the same optimiser-size admission as new fits.
`POST /api/fits/replay/jobs` runs larger replays in the process worker.

Recordings can declare `group`, the subject, acquisition or simulation
replicate they belong to. A group cannot cross the training/hold-out split,
even when its recordings differ. Imported CSV files initially use their file
names as groups; edit those groups to match the actual acquisition custody.
The UI shows the executed search domains and every optimiser generation's
best loss, independently of later form edits.

## Parameter constraints

Fits and cohort sweeps accept named, bounded linear combinations in **parameter
value space**, including for log-searched parameters:

```json
{"name": "R below C", "coefficients": {"R": 1, "C": -1}, "low": -1, "high": -0.3}
```

This requires `-1 <= R - C <= -0.3`. All names must be declared model
parameters. Bounds and coefficients must be finite. Bounds define a nonempty
interval; equalities and arbitrary expression constraints are unsupported and
refused. Fits use SciPy's constrained differential evolution without local
polishing; rejected constraint checks are counted separately from objective
trials (a proposal may be checked more than once), and a search with no feasible
optimum does not claim convergence or a finite training loss.

Identifiability remains a training-data diagnosis. Constrained fits withhold
the unconstrained Gauss–Newton covariance: that approximation is not a
constrained uncertainty estimator. Their result explains why standard errors
are absent. Grouped or constrained problems use `sc-neurocore.fit.v2`;
legacy ungrouped, unconstrained `sc-neurocore.fit.v1` exports remain readable.

## Shared-sample experiment cohorts

The **Experiment cohorts** section imports `sc-neurocore.cohort.v1` JSON,
submits `POST /api/cohorts/jobs`, displays every declared trial and exports the
complete result. `POST /api/cohorts/replay` re-executes the full admitted
cohort and checks its digest. It uses the same owner-bound job status and
cancellation routes as fitting.

Generate a complete synthetic protocol with the public library example:

```bash
python examples/studio_cohort.py --output experiment-cohort.json
python examples/studio_cohort.py --run --output cohort-result.json
```

The file is sufficient for another researcher to import and execute. No local
paths, implicit noise generator state or hidden recordings are required.

Each cohort states:

- A name, seed, noise provenance, positive `dt`, time unit and input unit.
  The seed documents sample generation; execution uses the exported samples
  directly and never resamples noise.
- Named acquisition groups and explicit training/hold-out assignments. Groups
  and identical recording content cannot cross the split.
- Current and additive noise arrays shared exactly by every model/trial.
  Each recording starts a fresh neuron under the model's declared profile.
- Complete DSL schemas, fixed parameters, constraints and explicit sweep
  values. Domains are `real` or `integer`; fractional integer values,
  duplicates, unknown fields and silent numeric coercions are refused.
- A metric per model: state `trace_rmse` in that state's declared physical
  unit, binary `event_disagreement` in fractions, or absolute
  `spike_count_error` in events. Trace observations name state variables;
  event metrics require recorded binary events at every step. Trace-only
  experiments may omit events.

All models must share the declared timebase. The input unit is the experiment
operator's declaration for the DSL's `I` input; the laboratory does not infer
physical units for a schema that does not declare them. Schema/profile and
sample digests accompany the results.

The full Cartesian grid is admitted before execution: at most 4,096 trials
and 5 million model steps. Oversized cohorts are refused, never shortened.
Constraint-rejected and divergent trials remain visible. Each model's selected
trial minimises the arithmetic mean of its per-recording **training** metrics;
held-out values or failures cannot affect this selection. Different metric
contracts are shown separately and are not ranked against each other.

Execution admits and freezes a private JSON snapshot before search or sweeps,
so caller-side schema edits and progress callbacks cannot alter the exported
experiment halfway through a run.

The result binds the complete protocol, trials, selections, software versions,
shared sample digests and noise provenance. Cohort digests normalise integral
float values (`1.0` and `1`) so numerical identity survives browser JSON
round trips; integer sample values outside JSON's exact safe integer range are
refused. A replay reports a mismatch when recomputation differs.

## Measurement custody and Pareto comparisons

Simulation does not measure hardware latency, resources or energy. A cohort
without external measurements states this explicitly and has no hardware
frontier.

`POST /api/cohorts/measurements` accepts a complete cohort result and at least
two `sc-neurocore.measurement.v1` receipts. The UI can import these receipts.
Each has `source_kind: "physical"`, `cohort_sha256`, `trial_sha256`, finite
nonnegative `latency_ms`, `resources`, `energy_j`, and a complete `contract`:
`target`, `device_revision`, `harness_sha256`, `workload_sha256`, `warmup`,
`transport`, positive integer `repeats`, `aggregation`, `resource_unit`,
`instrument`, and `calibration_sha256`. The workload digest must equal the
complete shared-sample cohort digest. `receipt_sha256` binds the receipt
without its own digest field using the public
`sc_neurocore.fitting.cohort.cohort_sha256` helper.

Contracts, resource units and scientific metric contracts must match exactly;
duplicate, edited, synthetic, failed-trial or unrelated receipts yield no
frontier and a reason. A structurally malformed result or receipt (a missing
field, a wrong JSON type, no held-out samples) is refused with one fixed
sentence; the response never carries exception text. Comparable rows minimise held-out error, measured
latency, measured resources and measured energy; all supplied rows remain
visible, with nondominated rows marked. Energy is never derived from operation
counts.

These are **operator-supplied measurements**. The software checks document
custody and declared comparability; it does not independently validate an
instrument or calibration. The comparison displays that boundary. Acquiring
physical receipts requires the device owner and operator; the example and test
protocols establish no hardware measurement claim.
