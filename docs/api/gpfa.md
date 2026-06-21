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

- The E-step builds the block-structured posterior precision over all latent
  dimensions and time points and solves it for the posterior mean and the summed
  second moment.
- The M-step updates `C`, `d` and the noise diagonal `R` from those sufficient
  statistics.
- Convergence is tested on the **exact marginal Gaussian log-likelihood**
  `-0.5 (yᵀ Σ⁻¹ y + log|Σ| + n log 2π)` with `Σ = A K Aᵀ + (I_T ⊗ R)`, evaluated
  with an LU-based `slogdet`. (The Rust path originally used an approximate
  residual-sum likelihood; it now matches the exact form.)
- `tau` is held fixed, so the GP kernels are constant across iterations.

## Backends and selection

| Backend | Entry point | FFI |
|---------|-------------|-----|
| Python (reference) | `gpfa_em` | -- (NumPy / LAPACK) |
| Rust | `sc_neurocore_engine.py_gpfa_em` | PyO3 |
| Julia | `GpfaAccel.gpfa_em` | juliacall |
| Go | `libgpfa.so::gpfa_em_c` | cgo c-shared, typed `double*` |
| Mojo | `libgpfa.so::gpfa_em_c` | `@export`, raw `int64` addresses |

`gpfa(..., backend="auto")` selects the **measured-fastest** path. GPFA's inner
loop is dense linear algebra -- Gaussian elimination, a marginal-covariance
`slogdet` and linear solves -- and the NumPy reference dispatches those to
LAPACK, which outpaces the hand-written Gauss-Jordan kernels of the compiled
backends. `auto` therefore runs the NumPy path; the compiled backends are
available by name (`backend="rust"`, `"julia"`, `"go"`, `"mojo"`) for parity
verification and for embedding where a tuned NumPy/LAPACK stack is absent.

```python
result = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=50)   # auto -> NumPy
ru     = gpfa(trains, n_latents=3, bin_ms=20.0, backend="rust")  # explicit Rust
```

## Parity

The deterministic seed gives every backend an identical starting point, so the
only differences are float64 round-off in the linear algebra. The chain is **not
bit-exact** -- floating-point EM and differing BLAS/elimination orders forbid
that -- but the agreement is tight and the iteration counts match. Measured
maximum absolute differences versus the NumPy reference (workload below):

| Backend | trajectories | C | log-likelihood |
|---------|-------------|---|----------------|
| Rust | 8.6e-11 | 2.2e-11 | 5.7e-13 |
| Julia | 3.5e-11 | 2.3e-12 | 8.5e-13 |
| Go | 1.0e-10 | 1.5e-11 | 2.7e-12 |
| Mojo | 1.2e-10 | 1.9e-11 | 3.5e-08 |

The test suite (`tests/test_gpfa.py`) gates each backend's parity class on the
backend being built, so an environment missing (say) Julia or Mojo skips only
those classes.

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
| Python (NumPy/LAPACK) | 7.76 ms | 1.00x |
| Mojo | 47.9 ms | 0.16x |
| Rust | 45.3 ms | 0.17x |
| Go | 82.6 ms | 0.09x |
| Julia | 5556 ms | 0.001x |

These numbers are why `auto` resolves to NumPy for GPFA: LAPACK's blocked,
vectorised solvers dominate the naive `O(n³)` kernels, and the gap widens with
problem size. The Julia figure is dominated by per-call juliacall marshalling
and allocation, not the numerics (its parity is the tightest of the four). The
compiled backends earn their place as parity oracles and as portable fallbacks,
not as the default fast path for this particular kernel.

## Rebuilding the compiled backends

```bash
# Rust (PyO3) -- rebuild the engine wheel, then refresh the bridge .so
CARGO_TARGET_DIR=/dev/shm/sc-neurocore-cargo-target maturin develop --release \
    -m engine/Cargo.toml

# Go (cgo c-shared)
cd src/sc_neurocore/accel/go/gpfa && \
    CGO_ENABLED=1 go build -buildmode=c-shared -o libgpfa.so gpfa.go

# Mojo (@export shared library)
cd src/sc_neurocore/accel/mojo/kernels && \
    mojo build --emit shared-lib -o libgpfa.so gpfa.mojo
```

Julia needs no build step; `juliacall` includes
`accel/julia/analysis/gpfa.jl` on first use.
