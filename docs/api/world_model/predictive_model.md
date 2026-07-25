# Linear Gaussian predictive model

`sc_neurocore.world_model.predictive_model` exposes filtering, smoothing,
learning, and planning-compatible forecasts for a controlled linear Gaussian
state-space model (LGSSM).

## Model

For latent state \(x_t \in \mathbb{R}^d\), observation
\(y_t \in \mathbb{R}^p\), and control \(u_t \in \mathbb{R}^m\):

\[
x_{t+1} = A x_t + B u_t + w_t, \qquad w_t \sim \mathcal{N}(0, Q),
\]

\[
y_t = C x_t + D u_t + v_t, \qquad v_t \sim \mathcal{N}(0, R),
\]

with prior \(x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)\).

`LinearGaussianSSM` copies every parameter into a finite, C-contiguous
`float64` array. It validates all dimensions, requires `Q` to be symmetric
positive semidefinite, and requires `R` and `Sigma_0` to be symmetric positive
definite. A positive diagonal alone is not accepted as a covariance proof.

## Public API and compatibility

```python
from sc_neurocore.world_model.predictive_model import (
    EMLearner,
    FilterResult,
    KalmanFilter,
    LinearGaussianSSM,
    PredictiveWorldModel,
    RTSSmoother,
    SmoothResult,
)
```

The import path, constructor parameter names, public class identities, and
pickle references remain `sc_neurocore.world_model.predictive_model`. The
implementation is partitioned by responsibility behind that facade:

| Responsibility | Owner |
|---|---|
| parameter and result contracts | `_lgssm_types.py` |
| native discovery, selection, and FFI marshalling | `_lgssm_backends.py` |
| Python and native forward-filter dispatch | `_lgssm_filter.py` |
| RTS backward recursion | `_lgssm_smoothing.py` |
| controlled EM learning | `_lgssm_em.py` |
| planning-compatible state forecasts | `_predictive_world_model.py` |

The private import graph is one-way and acyclic. The facade contains no
algorithm implementation.

## Forward filtering

`KalmanFilter.filter(observations, controls=None, backend="auto")` returns:

- filtered state means and covariances;
- one-step predicted means and covariances before each observation; and
- the sequence log-likelihood.

Observations must have shape `(T, p)` with `T > 0`. Controls are required for a
model with `m > 0` and must have shape `(T, m)`. Every input and returned moment
must be finite. Result covariance stacks are checked in one vectorised
symmetry/eigenvalue pass before being exposed to the caller.

The Python update factors each innovation covariance once with Cholesky. It
uses triangular solves for the innovation quadratic form and Kalman gain; it
does not form a matrix inverse. The covariance update uses Joseph form:

\[
P_{t|t} = (I-K_t C)P_{t|t-1}(I-K_t C)^T + K_t R K_t^T.
\]

A non-positive-definite innovation covariance fails closed with
`numpy.linalg.LinAlgError`.

## Native forward-filter chain

The maintained forward filter has five execution paths:

| Backend | Boundary | Numerical source |
|---|---|---|
| Python | NumPy | `_lgssm_filter.py` |
| Mojo | C ABI via `ctypes` | `accel/mojo/world_model/lgssm.mojo` |
| Go | C shared library via `ctypes` | `accel/go/lgssm/lgssm.go` |
| Rust | PyO3 | `engine/src/lgssm.rs` |
| Julia | `juliacall` | `accel/julia/world_model/predictive_model.jl` |

An explicit unavailable backend raises `RuntimeError`; it never silently
changes language. `backend="auto"` follows the stable availability- and
initialisation-aware order:

```text
Mojo -> Go -> Rust -> Julia -> Python
```

The source-bound controlled workload described below rejects any unexplained
ordering with a material warm-timing inversion greater than 10%. Rust is the
one declared exception: it precedes Julia because it is loaded during package
import, while probing Julia may initialise a separate runtime. Adjacent timings
inside the loaded-host noise band retain their stable order. The artifact
records both post-import probe cost and the exact warm median ranking. Python is
always the final maintained fallback. All native results pass through the same
`FilterResult` validation boundary as Python results.

The Mojo kernel solves each row of the \(d \times p\) gain workspace
independently; it never indexes that workspace as though it were
\(p \times p\). A dedicated \(d = 1, p = 3\) test exercises the
observation-wider-than-state case through all five backends and verifies
complete moment and likelihood parity.

## RTS smoothing

`RTSSmoother.smooth(filter_result)` performs the Rauch-Tung-Striebel backward
recursion with positive-definite solves. It returns full-sequence posterior
moments and lag-one covariance blocks oriented as:

\[
\operatorname{Cov}[x_t, x_{t+1} \mid y_{0:T-1}].
\]

That orientation is part of the public result contract. The EM transition
update explicitly transposes each lag block when it needs
(E[x_{t+1}x_t^T]). A multivariate exact batch-conditioning test verifies the
means, covariance blocks, and lag orientation.

RTS smoothing is currently a Python/NumPy responsibility. The native chain
accelerates only the forward filter.

## Controlled expectation-maximisation

`EMLearner.fit` updates `A`, `C`, `Q`, `R`, `mu_0`, and `Sigma_0`. `B` and `D`
are treated as known parameters and are preserved exactly.

Controls still participate in the M-step. The transition statistics subtract
`B @ u_t`, and the observation statistics subtract `D @ u_t`, before solving
for `A` and `C`. Omitting those terms biases a controlled fit even when `B` and
`D` themselves are fixed.

All normal-equation right solves use Cholesky factors rather than explicit
inverses. Learned covariance matrices are symmetrised and projected only as
needed to restore the documented positive-semidefinite or positive-definite
contract. Every M-step candidate, including the candidate from the final
allowed iteration, is filtered and checked before it becomes the returned
model. A material likelihood decrease raises `RuntimeError`; convergence uses
the configured absolute tolerance. `log_likelihood_history` starts with the
initial model and records each evaluated candidate.

The default EM forward-filter backend is Python, making the complete learning
workload deterministic and explicit. A caller may select another maintained
forward backend for the E-step, but the smoother and M-step remain Python.

Latent-state bases are not uniquely identifiable: an invertible change of
basis can alter `A` and `C` while preserving the observation distribution.
Accordingly, recovery tests judge held-out likelihood and posterior moments,
not raw parameter equality alone.

## Planning-compatible forecasts

`PredictiveWorldModel` preserves the historical planning surface:

```python
import numpy as np

from sc_neurocore.world_model import PredictiveWorldModel

world_model = PredictiveWorldModel(state_dim=4, action_dim=2, seed=42)
mean, covariance = world_model.predict_next_state_with_cov(
    current_state=np.zeros(4),
    current_cov=np.eye(4),
    action=np.array([0.25, -0.10]),
)
```

`predict_next_state` returns (A x_t + B u_t).
`predict_next_state_with_cov` additionally returns
(A P_t A^T + Q). `forecast` and `forecast_with_cov` return independent arrays
for every step, so mutating one returned state cannot alter another.

## Benchmark evidence

`benchmarks/bench_predictive_model.py` runs one controlled workload through all
five forward backends and fails if array or likelihood parity exceeds the
declared tolerances. It warms every backend before timing, then rotates the
starting backend across interleaved sampling rounds so changing host load does
not systematically favour a later block. The same run separately measures the
Python-only RTS and EM workloads; unsupported native RTS/EM rows are not
represented as skips.

The committed artifact is
`benchmarks/results/bench_predictive_model.json`. It records:

- every timing sample, median, minimum, maximum, post-import probe cost, exact
  warm measured rank, stable dispatch policy, and interleaving policy;
- parity deltas against the Python result;
- source SHA-256 hashes for the facade, responsibility modules, native
  implementations, bridge, engine registration, and benchmark harness;
- binary hashes for the selected Rust, Go, and Mojo runtimes;
- language versions, CPU model, affinity, frequency governor, and load averages;
- the exact invocation and explicit heavy-job/isolation disclosure.

Reproduce the local evidence with:

```bash
taskset -c 2 env PYTHONPATH=src .venv/bin/python \
  benchmarks/bench_predictive_model.py \
  --backends python rust julia go mojo \
  --steps 200 \
  --repeats 25 \
  --em-iterations 10 \
  --other-heavy-jobs-running yes \
  --other-heavy-jobs-note "shared workstation with concurrent repository work" \
  --isolation-note "single-CPU affinity; no exclusive-core reservation" \
  --json benchmarks/results/bench_predictive_model.json
```

The measurements are loaded-host local-regression evidence. CPU affinity does
not imply an exclusive core, so the artifact does not support a promotion-grade
cross-host performance claim.

### Latest committed local run

The 2026-07-14 source-bound run used 200 time steps, 25 interleaved samples per
backend, four latent states, three observations, and two controls. It ran on
logical CPU 2 of an Intel i5-11600K under the `powersave` governor while other
heavy repository work was active. The core was affinity-pinned but not
reserved. These numbers are regression evidence for this host only.

| Backend | Probe (ms) | Median (ms) | Min (ms) | Max (ms) | Python / median | Max array delta | Likelihood delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python | 0.005959 | 13.847860 | 12.020334 | 15.800676 | 1.000000 | 0 | 0 |
| Rust | 0.003617 | 2.806096 | 2.519172 | 5.343638 | 4.934920 | 6.22e-15 | 1.71e-13 |
| Julia | 6259.491127 | 2.140998 | 1.976630 | 136.903510 | 6.467946 | 5.33e-15 | 1.14e-13 |
| Go | 1.802784 | 1.827930 | 1.572094 | 3.094846 | 7.575706 | 5.33e-15 | 2.27e-13 |
| Mojo | 67.868924 | 1.553183 | 1.382646 | 2.595167 | 8.915794 | 4.00e-15 | 8.52e-08 |

The exact warm ranking was Mojo, Go, Julia, Rust, Python. The stable dispatch
remains Mojo, Go, Rust, Julia, Python: the artifact records Rust-before-Julia
as the sole declared warm-order exception because Rust is already imported and
the Julia probe incurred 6.26 seconds of runtime initialisation. All parity
deltas remain inside the committed `1e-9` array and `1e-7` likelihood limits.

Python-only RTS smoothing measured 6.222437 ms median
(5.728943–9.343452 ms). Ten EM iterations measured 234.127058 ms median
(217.413576–265.916984 ms). The artifact records every raw sample, so timing
dispersion remains visible rather than being discarded.

## Verification surfaces

- `test_linear_gaussian_ssm_parameters.py`: parameter and covariance contracts.
- `test_linear_gaussian_ssm_random.py`: random-model dimension and stability contracts.
- `test_linear_gaussian_filter_result.py`: filtering-result contracts.
- `test_linear_gaussian_smooth_result.py`: smoothing-result contracts.
- `test_kalman_filter.py`: analytic updates, controls, covariance stability,
  validation, and installed-backend parity.
- `test_rts_smoother.py`: exact multivariate batch conditioning and lag
  orientation.
- `test_em_learner.py`: controlled sufficient statistics, monotonicity,
  held-out likelihood, convergence, and fail-closed behavior.
- `test_predictive_world_model.py`: planning-facing mean/covariance forecasts.
- `test_predictive_model_backends.py`: loader and FFI boundaries.
- `test_predictive_model_architecture.py`: facade identity, pickle compatibility,
  responsibility ownership, import DAG, structured solves, and module bounds.
- `test_predictive_model_benchmark.py`: artifact provenance, all-backend parity,
  loaded-host disclosure, and the reduced real CLI.

Focused tests cover every executable line in the seven Python responsibility
modules. The only uncovered branch arcs are the exits of static `Protocol`
declarations; no coverage exclusions or test skips are used.

## API reference

::: sc_neurocore.world_model.predictive_model
    options:
      show_root_heading: true
      members_order: source

## References

- Kalman, R. E. (1960), “A New Approach to Linear Filtering and Prediction
  Problems.”
- Rauch, H. E., Tung, F., and Striebel, C. T. (1965), “Maximum Likelihood
  Estimates of Linear Dynamic Systems.”
- Shumway, R. H., and Stoffer, D. S. (1982), “An Approach to Time Series
  Smoothing and Forecasting Using the EM Algorithm.”
- Bishop, C. M. (2006), *Pattern Recognition and Machine Learning*, section
  13.3.
