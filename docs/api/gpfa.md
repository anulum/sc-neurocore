# GPFA -- deterministic init and polyglot EM

Gaussian Process Factor Analysis extracts smooth, low-dimensional latent
trajectories from binned single-trial population spike counts via
Expectation-Maximisation with squared-exponential Gaussian-process priors on
each latent dimension.

> Yu, Cunningham, Santhanam, Ryu, Shenoy, Sahani (2009). *Gaussian-process
> factor analysis for low-dimensional single-trial analysis of neural
> population activity.* J. Neurophysiol. 102:614--635.

The implementation lives in `sc_neurocore.analysis.spike_stats.gpfa`. It pairs a
**deterministic initialisation** with a **five-language EM chain** (NumPy, Rust,
Julia, Go, Mojo) whose outputs agree to floating-point round-off.

## Deterministic initialisation

The original procedure seeds the loading matrix `C` from a random draw, so two
runs -- and any two language backends -- diverge from the first iteration.
`gpfa_pca_init` replaces that with a reproducible PCA seed:

- `C` is the top `n_latents` left singular vectors of the column-centred data
  `Y`, scaled by their singular values `S / sqrt(n_bins)`.
- A fixed sign convention makes each column's largest-magnitude entry positive,
  which removes the sign ambiguity of the SVD across BLAS/LAPACK builds.
- `d` is the per-neuron mean, `R` the per-neuron variance (`+ 1e-4`), and the GP
  timescales `tau` are set to `2 x bin_ms`.

```python
C, d, R, tau = gpfa_pca_init(Y, n_latents=3, bin_ms=20.0)
```

Because the seed is identical everywhere, the EM result is reproducible across
runs and across backends. The public `gpfa(...)` still accepts a `seed`
argument for API compatibility, but it no longer affects the result.

## The EM contract

Every backend binds the same flat contract, which is what makes cross-language
parity checkable:

```
gpfa_em(Y, C0, d0, R0, tau, max_iter, tol)
    -> (trajectories, C, d, R, log_likelihoods)
```

The linear algebra is **Cholesky-based and structured**. GPFA's systems are all
symmetric positive-definite -- the GP prior `K`, the posterior precision and the
second-moment matrix -- so Cholesky is the numerically stable, ~2x cheaper choice
over a general LU/Gauss-Jordan elimination, and a log-determinant falls out of
the factor for free (`2 Σ log L_ii`).

- The **E-step** assembles the `n_state × n_state` posterior precision
  `M = blkdiag(K_j⁻¹) + AᵀR⁻¹A` (`n_state = n_latents · n_bins`), Cholesky-factors
  it once, and reuses the factor for both the posterior mean and covariance.
- The **M-step** updates `C`, `d` and the noise diagonal `R` from those
  sufficient statistics (a Cholesky solve against the second-moment matrix).
- The **log-likelihood** is the exact marginal Gaussian likelihood, but it never
  forms the dense `(n_neurons·n_bins)²` covariance. The Woodbury identity and the
  matrix-determinant lemma route both terms through the same `n_state × n_state`
  precision `M`:

  ```text
  yᵀ Σ⁻¹ y = yᵀ R⁻¹ y − (AᵀR⁻¹y)ᵀ M⁻¹ (AᵀR⁻¹y)
  log|Σ|   = log|M| + log|K| + log|R_big|
  ```

  This is the structured estimator of Yu et al. (2009): the exact marginal
  likelihood of the regularised model (each GP kernel carries a consistent `1e-6`
  jitter), not an approximation. Because the work scales with `n_state` rather
  than `n_obs = n_neurons · n_bins`, it is both more precise and dramatically
  cheaper at scale -- the dense form would build, for example, an 8000×8000
  (≈0.5 GB) matrix where the structured form factors a 1600×1600 one.

`tau` is held fixed, so the GP kernels are constant across iterations.

A note on the Toeplitz kernel: the squared-exponential `K_j` is symmetric
Toeplitz, but for the smooth (highly ill-conditioned) kernel a Cholesky with
jitter is more numerically robust than a Levinson-Durbin Toeplitz solver, and the
`K_j` factorisation is not the bottleneck -- the `M` factorisation dominates -- so
the precise choice is Cholesky throughout.

## Backends and selection

| Backend | Cholesky path | FFI |
|---------|---------------|-----|
| Python (reference) | SciPy `cho_factor`/`cho_solve` (LAPACK) | -- |
| Rust | `nalgebra` Cholesky (LAPACK-grade) | PyO3 |
| Julia | native `cholesky(Symmetric(·))` (LAPACK) | juliacall |
| Go | in-place Cholesky | cgo c-shared, typed `double*` |
| Mojo | in-place Cholesky | `@export`, raw `int64` addresses |

`gpfa(..., backend="auto")` selects the **measured-fastest** path. With the
structured estimator the compiled backends are no longer bottlenecked on a large
general solve, so the Rust backend -- `nalgebra` Cholesky with no Python dispatch
overhead -- is the fastest path and `auto` prefers it when the engine is present,
falling back to the NumPy reference otherwise. Julia, Go and Mojo run on request.

```python
result = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=50)   # auto -> Rust
py     = gpfa(trains, n_latents=3, bin_ms=20.0, backend="python")
```

## Parity

The deterministic seed gives every backend an identical starting point, so the
only differences are float64 round-off in the linear algebra. The chain is **not
bit-exact** -- floating-point EM and differing Cholesky orders forbid that -- but
the agreement is tight and the iteration counts match. Measured maximum absolute
differences versus the NumPy reference (workload below):

| Backend | trajectories | C | log-likelihood |
|---------|-------------|---|----------------|
| Rust | 1.7e-10 | 9.8e-11 | 4.8e-09 |
| Julia | 3.7e-11 | 4.3e-11 | 3.6e-09 |
| Go | 2.2e-10 | 8.3e-12 | 1.0e-09 |
| Mojo | 1.9e-10 | 7.1e-12 | 4.5e-08 |

The test suite (`tests/test_gpfa.py`) gates each backend's parity class on the
backend being built, so an environment missing (say) Julia or Mojo skips only
those classes. The structured likelihood is additionally checked against a dense
brute-force marginal covariance (agreement ~1e-11).

## Benchmark

`benchmarks/bench_gpfa.py` times every available backend from the shared
deterministic init and writes `benchmarks/results/bench_gpfa.json`. Run it on
shielded cores:

```bash
taskset -c 10-11 python benchmarks/bench_gpfa.py \
    --json benchmarks/results/bench_gpfa.json
```

Reference run -- 11th Gen Intel Core i5-11600K, runtime-cpuset shield on cores
10--11; workload of 8 neurons x 600 samples binned to 30 bins, 3 latents, EM
cap 30 (converged in 3 iterations):

| Backend | Median call | vs NumPy |
|---------|-------------|----------|
| Rust (`nalgebra`) | 2.02 ms | 1.43x |
| Python (NumPy/LAPACK) | 2.88 ms | 1.00x |
| Mojo | 3.33 ms | 0.87x |
| Go | 5.65 ms | 0.51x |
| Julia | 2181 ms | 0.001x |

The structured estimator makes the Rust backend the fastest path -- it does the
same LAPACK-grade Cholesky as NumPy but without the per-call Python dispatch
overhead. The hand-written in-place Cholesky of Go and Mojo is competitive with
NumPy on this small workload and keeps the backends dependency-free. The Julia
figure is dominated by per-call juliacall marshalling and allocation, not the
numerics (its parity is the tightest of the four).

## Rebuilding the compiled backends

```bash
# Rust (PyO3, depends on nalgebra) -- build the engine, then refresh the bridge .so
CARGO_TARGET_DIR=/media/anulum/GOTM/_scratch/sc-neurocore-cargo-target \
    maturin develop --release -m engine/Cargo.toml
cp /media/anulum/GOTM/_scratch/sc-neurocore-cargo-target/release/libsc_neurocore_engine.so \
    bridge/sc_neurocore_engine/sc_neurocore_engine.cpython-312-x86_64-linux-gnu.so

# Go (cgo c-shared)
cd src/sc_neurocore/accel/go/gpfa && \
    CGO_ENABLED=1 go build -buildmode=c-shared -o libgpfa.so gpfa.go

# Mojo (@export shared library)
cd src/sc_neurocore/accel/mojo/kernels && \
    mojo build --emit shared-lib -o libgpfa.so gpfa.mojo
```

Julia needs no build step; `juliacall` includes
`accel/julia/analysis/gpfa.jl` on first use.
