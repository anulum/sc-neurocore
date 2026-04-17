# `sc_neurocore.world_model.predictive_model` — Linear Gaussian SSM

## 1. Scope

This module implements a **Linear Gaussian State-Space Model
(LGSSM)** with the standard inference + learning trinity:
**Kalman filter** (forward), **Rauch-Tung-Striebel (RTS)
smoother** (backward), and **Expectation-Maximisation (EM)**
parameter learner. It serves as the probabilistic predictive
substrate for `sc_neurocore.world_model.planner.SCPlanner` and
for any downstream consumer that needs to predict, smooth, or
fit linear-Gaussian dynamics over noisy observations.

Implementation references:

- Kalman, R.E. (1960). *A New Approach to Linear Filtering and
  Prediction Problems.* J. Basic Engineering 82(1): 35-45.
- Rauch, H.E., Tung, F. & Striebel, C.T. (1965). *Maximum
  likelihood estimates of linear dynamic systems.* AIAA J 3(8):
  1445-1450. (the RTS smoother)
- Shumway, R.H. & Stoffer, D.S. (1982). *An approach to time
  series smoothing and forecasting using the EM algorithm.*
  J Time Series Analysis 3(4): 253-264.
- Bishop, C.M. (2006). *Pattern Recognition and Machine
  Learning*, Springer. §13.3 (linear dynamical systems).
- Murphy, K.P. (2023). *Probabilistic Machine Learning:
  Advanced Topics*, MIT Press. §29 (state-space models).

The previous implementation was a **deterministic linear
matmul** on a randomly-initialised matrix that called itself
"stochastic" and admitted "(simplified)" in a comment. It was
replaced 2026-04-17 per
`feedback_sophisticated_from_start.md` — a placeholder
masquerading as a world model is unacceptable.

## 2. Model

Latent state ``x_t ∈ R^d``, observation ``y_t ∈ R^p``,
control ``u_t ∈ R^m``:

```
x_{t+1} = A · x_t + B · u_t + w_t,    w_t ~ N(0, Q)
y_t     = C · x_t + D · u_t + v_t,    v_t ~ N(0, R)
```

with prior ``x_0 ~ N(μ_0, Σ_0)``. All parameters
``{A, B, C, D, Q, R, μ_0, Σ_0}`` are estimable from data via
the EM algorithm (Shumway & Stoffer 1982).

## 3. Public API

```python
from sc_neurocore.world_model.predictive_model import (
    LinearGaussianSSM,    # the model parameters (dataclass)
    KalmanFilter,         # forward inference
    RTSSmoother,          # backward smoothing
    EMLearner,            # parameter estimation via EM
    PredictiveWorldModel, # legacy wrapper (still re-exported)
    FilterResult,         # KalmanFilter output
    SmoothResult,         # RTSSmoother output
)
```

`LinearGaussianSSM.random(state_dim, obs_dim, control_dim)`
constructs a random stable LGSSM (spectral radius < 1) for
smoke tests and EM initialisation.

## 4. Algorithm details

### 4.1 Kalman filter (forward pass)

Per step `t`:

1. **Predict**: `x_pred = A x_filt + B u`,
   `P_pred = A P_filt A^T + Q`.
2. **Update** (Joseph form for numerical stability):
   - innovation `e_t = y_t - C x_pred - D u_t`,
   - innovation covariance `S = C P_pred C^T + R`,
   - Kalman gain `K = P_pred C^T S^{-1}`,
   - filtered mean `x_filt = x_pred + K e_t`,
   - filtered covariance `P_filt = (I - K C) P_pred (I - K C)^T + K R K^T`.
3. **Log-likelihood contribution**: `-½ (p log 2π + log |S| + e_t^T S^{-1} e_t)`.

### 4.2 RTS smoother (backward pass)

Per step `t = T-2 ... 0`:

1. **RTS gain**: `J_t = P_filt(t) A^T P_pred(t+1)^{-1}`.
2. **Smoothed mean**: `x_smooth(t) = x_filt(t) + J_t (x_smooth(t+1) - x_pred(t+1))`.
3. **Smoothed covariance**: `P_smooth(t) = P_filt(t) + J_t (P_smooth(t+1) - P_pred(t+1)) J_t^T`.
4. **Lag-1 cross-covariance**: `C_smooth(t,t+1) = J_t P_smooth(t+1)`. (Required by EM.)

The smoother is invariant at `t = T-1` (no future to incorporate)
and reduces uncertainty for all earlier steps (verified by
`test_rts_smoother_reduces_uncertainty`).

### 4.3 EM learner

Per iteration:

- **E-step**: Kalman filter + RTS smoother → posterior moments.
- **M-step** (closed-form, A and C only — B/D held fixed):
  - `A_new = (Σ E[x_{t+1} x_t^T])(Σ E[x_t x_t^T])^{-1}` (sum
    over t = 0..T-2).
  - `Q_new = (1/(T-1)) Σ (E[x_{t+1} x_{t+1}^T] - A E[x_{t+1} x_t^T]^T - ...)`
    (collapsed RHS).
  - `C_new = (Σ y_t E[x_t]^T)(Σ E[x_t x_t^T])^{-1}` (sum
    over t = 0..T-1).
  - `R_new = (1/T) Σ ((y_t - C E[x_t])(y_t - C E[x_t])^T + C P_t C^T)`.
  - `μ_0, Σ_0` ← smoothed first state.

EM theory (Dempster et al. 1977) guarantees the log-likelihood
is monotone non-decreasing across iterations under exact
arithmetic; the Python implementation respects this to within
a few units of float64 round-off
(test_em_log_likelihood_monotone_non_decreasing).

## 5. Identifiability

Linear Gaussian SSMs have a **well-known sign + scale
ambiguity** (Bishop 2006 §13.3.4): the pair `(A, C)` and
`(αA, C/α)` are observationally equivalent for any `α > 0`.
Direct parameter recovery is therefore brittle — what IS
identifiable is the observation likelihood. Tests verify
recovery via held-out log-likelihood, not raw parameter
agreement.

## 6. Pipeline wiring

`PredictiveWorldModel` is consumed by:

- `sc_neurocore.world_model.planner.SCPlanner` — calls
  `predict_next_state` / `forecast` to evaluate candidate
  action sequences.
- `tests/test_interfaces_generative_worldmodel.py` — exercises
  the legacy API with `state_dim=4, action_dim=2`.

Re-exported via `sc_neurocore.world_model.__init__`:

```python
from sc_neurocore.world_model import PredictiveWorldModel
```

## 7. Multi-language acceleration chain

Per `feedback_multi_language_accel.md` (Rust + Julia + Go +
Mojo + Python fallback). Current state:

| Backend | Status | Source |
|---|---|---|
| **python** (NumPy `linalg.solve`) | ✅ implemented | this module |
| **rust** (PyO3 + `ndarray` Cholesky) | ✅ implemented | `engine/src/lgssm.rs` |
| **julia** (juliacall + `LinearAlgebra` LAPACK) | ✅ implemented | `src/sc_neurocore/accel/julia/world_model/predictive_model.jl` |
| **go** (cgo + ctypes shared library) | ✅ implemented | `src/sc_neurocore/accel/go/lgssm/lgssm.go` |
| **mojo** (Mojo SIMD via subprocess) | ⏳ followup | task #69 (kernel still placeholder) |

The dispatcher
(`KalmanFilter.filter(backend='auto'|'rust'|'julia'|'go'|'python')`)
picks Rust > Julia > Go > Python in priority order under
`'auto'`, then explicit override for the rest. All four
implemented backends return identical (means, covariances,
log-likelihood) results to atol=1e-9 on the parity tests
(`test_four_backend_parity_when_all_available`).

The benchmark
`benchmarks/bench_predictive_model.py` runs the workload on
every available backend and records `unavailable_reason` for the
ones not yet wired. The `accel/julia/world_model/predictive_model.jl`
file that previously existed was a non-functional placeholder
(Python syntax inside a Julia `module`) and was deleted in this
commit; its replacement is the work scoped under #68.

## 8. Performance

Reproducible via:

```bash
python benchmarks/bench_predictive_model.py \
    --json benchmarks/results/bench_predictive_model.json
```

Workload: 4-D state, 3-D obs, T=200 sequence sampled from the
true model. Median + min over 5 repeats. Hardware: Linux 6.17
x86_64, NumPy 2.2.0, Python 3.12.3.

| Workload | Backend | Median | Min | Speedup vs Python |
|---|---|---:|---:|---:|
| Forward Kalman filter | python | 19.04 ms | 9.14 ms | 1.0× |
| Forward Kalman filter | rust | 2.54 ms | 1.80 ms | 7.5× |
| Forward Kalman filter | julia | 1.66 ms | 1.58 ms | 11.5× |
| Forward Kalman filter | **go** | **0.83 ms** | **0.80 ms** | **22.9×** |
| RTS smoother | python | ~10 ms | — | 1.0× |
| EM (10 iters) | python | ~120 ms | — | 1.0× |

All four implemented backends produce **identical
log-likelihood** (-288.0601) to ≤ 1e-9 absolute tolerance
(verified by `test_four_backend_parity_when_all_available`);
the same holds for filtered means and covariances.

**Go is the fastest** on this (T=200, d=4, p=3) workload —
cgo has near-zero call overhead (raw C ABI) and the Go
compiler emits decent SIMD-friendly code for the inner matrix
loops. Julia is second (LAPACK Cholesky beats hand-rolled),
Rust third (similar approach to Go but with an extra PyO3
marshalling layer), Python last.

The auto dispatcher still prefers Rust > Julia > Go because
Rust + Julia are deeper integrations (PyO3 / juliacall types
vs ctypes raw memory) — for very small workloads call setup
matters more than raw compute. For large T or d (≥ 10 000),
re-benchmark on your own data and switch the explicit
`backend='go'` if it wins.

The RTS smoother and EM learner currently dispatch only to the
Python path. Extending all four accel backends to RTS + EM is
deferred — the marginal value is low because RTS smoothing is
already sub-10 ms even in pure Python.

Captured run in
`benchmarks/results/bench_predictive_model.json`.

## 9. Tests

- `tests/test_world_model/test_predictive_model.py` — 20 cases:
  shape validation, PSD invariance, log-likelihood
  monotonicity, RTS smoother covariance reduction, low-noise
  tracking, high-noise prior reliance, EM held-out
  log-likelihood improvement, legacy wrapper compatibility.
- `tests/test_planner.py` — updated: dropped the 3 tests that
  enforced the placeholder `transition_matrix` design;
  added `test_predict_next_state_obeys_ssm_dynamics`
  asserting `output == A·x + B·u`.
- `tests/test_interfaces_generative_worldmodel.py` — pre-existing
  4 tests for the legacy API; pass unchanged after the rewrite
  (API preserved for backwards compatibility).

Run: `pytest tests/test_world_model/ tests/test_planner.py
tests/test_interfaces_generative_worldmodel.py::TestPredictiveWorldModel
tests/test_interfaces_generative_worldmodel.py::TestSCPlanner --no-cov`
→ **31 passed in 7.30 s**.

## 10. Audit completeness — 7-point rule

| # | Criterion | Status | Notes |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | `world_model/__init__` re-exports preserved; SCPlanner consumer still passes |
| 2 | Multi-angle tests | ✅ PASS | 31 tests across 3 files; PSD invariance, EM monotonicity, identifiability caveat |
| 3 | Acceleration path | ⚠️ WARN | python + **rust** + **julia** + **go** (Kalman filter only, all 4 verified parity to atol=1e-9); mojo tracked as #69 |
| 4 | Benchmarks | ✅ PASS | `benchmarks/bench_predictive_model.py` committed; multi-backend harness handles unavailable backends gracefully |
| 5 | Performance docs | ✅ PASS | §8 with measured numbers |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX 2-line header. mypy clean. No `# noqa`. No `# mypy: ignore-errors`. Citation list cites 5 published references. |

Net: **1 WARN, 0 FAIL.** The WARN is the missing Rust/Julia/
Mojo/Go backends — tracked, not deferred indefinitely.

## 11. Known issues / followups

### 11.1 Multi-language chain incomplete (WARN row 3)

Tasks #67 (Rust), #68 (Julia), #69 (Mojo), #70 (Go) track the
proper backend implementations. The current `bench_predictive_model.py`
harness is ready to ingest them as soon as they land —
`backends` dict in the script need only flip the `available`
flag.

### 11.2 EM does not estimate B and D

The M-step in `EMLearner` updates only `A`, `C`, `Q`, `R`,
`μ_0`, `Σ_0`. Joint estimation of `B` and `D` requires
augmenting the sufficient statistics with control terms; this
is documented in Shumway & Stoffer (1982) Appendix A but not
implemented here. Open follow-up: extend EMLearner to optimise
B and D when controls are present.

### 11.3 Identifiability test is held-out-LL not parameter recovery

By design — see §5. Direct parameter recovery is brittle due
to the LGSSM sign+scale ambiguity. The held-out log-likelihood
test is the proper identifiability check.

## 12. Audit batch identification

This page was produced as part of the **Antigravity audit**
(#66 / #62) — third complete audit cycle (after `chiplet_gen.simulate_thermal`
and `physics/heat.py`). One commit per task per
`feedback_per_task_full_workflow.md`.
