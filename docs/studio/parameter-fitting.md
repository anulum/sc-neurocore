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

## Replay

The result carries the whole problem — model, domains, fixed values, every
recording, the seed and the optimiser size — the software versions, and one
`result_sha256`. `POST /api/fits/replay` runs it again and reports whether the
new digest equals the exported one. The optimiser is seeded and runs in one
process, so a replay on the same versions reproduces the digest.

## In the Studio

The **Fit** tab fits either the selected catalogue model's canonical schema or
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
With route policies enforced, both routes require an authenticated principal.

## What a fit does not establish

- A good hold-out error on the recordings given; a different protocol can
  still separate models the cohort cannot.
- Identifiability beyond the local, linearised analysis at the optimum; a
  second optimum elsewhere in the domain is not excluded.
- Anything about hardware: a fitted parameter set has no fixed-point, RTL or
  co-simulation evidence of its own.
- Accuracy against latency, resources or energy: no such trade-off is shown,
  because none is measured comparably here.
