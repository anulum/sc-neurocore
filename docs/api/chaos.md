# Chaos — Chaotic RNG for Stochastic Computing

Deterministic-chaos random number generators for stochastic computing bitstream encoding. Provides alternatives to linear PRNGs (LFSR, Mersenne Twister) with desirable statistical properties for SC arithmetic.

## Why Chaotic RNG for Stochastic Computing?

Stochastic computing encodes values as the probability of a 1-bit in a random bitstream. The quality of the random source directly affects arithmetic accuracy. Linear PRNGs have short-range correlations that bias SC multiplication (AND gates). Chaotic maps produce sequences with:

- **Broadband spectrum** — no periodic structure to alias with SC gate frequencies
- **Low short-range autocorrelation** — adjacent bits are nearly independent
- **Deterministic reproducibility** — same seed → same bitstream on hardware
- **Minimal state** — one float (logistic map) vs 624 words (MT19937)

## Available Generators

### ChaoticRNG — Logistic Map

The logistic map `x_{n+1} = r * x_n * (1 - x_n)` at `r=4.0` is fully chaotic with Lyapunov exponent `ln(2) ≈ 0.693`. The invariant density is `Beta(0.5, 0.5)` on (0, 1) — values cluster near 0 and 1. The `generate_bitstream()` method applies the inverse CDF `(2/π) * arcsin(√x)` to uniformize before thresholding.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `r` | 4.0 | Bifurcation parameter. Must be in (3.57, 4.0] for chaos. |
| `x` | 0.37 | Initial condition in (0, 1). Avoid 0, 0.5, 1 (fixed/periodic points). |
| `burn_in` | 100 | Steps to discard before first output. |

**Analysis methods:**

- `lyapunov_exponent(n_steps)` — Estimate maximal Lyapunov exponent via derivative averaging. At r=4.0 the theoretical value is ln(2) ≈ 0.6931.
- `shannon_entropy(n_samples, n_bins)` — Estimate Shannon entropy in bits. At r=4.0, values follow Beta(0.5, 0.5) with entropy ~log2(n_bins) - 0.27 bits below uniform.
- `autocorrelation(n_samples, max_lag)` — Compute autocorrelation up to `max_lag`. A good chaotic RNG shows near-zero autocorrelation for all lags > 0.

### TentMapRNG — Piecewise Linear Alternative

The tent map `x_{n+1} = μ * min(x_n, 1 - x_n)` is topologically conjugate to the logistic map at r=4 but has **uniform** invariant density on (0, 1) — no CDF correction needed for SC bitstreams.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `mu` | 1.9999 | Slope parameter. Must be in (1, 2]. Default slightly below 2.0 to avoid float64 degeneracy. |
| `x` | 0.37 | Initial condition in (0, 1). |

**Reference:** Phatak & Rao, "Logistic map as a random number generator", Physical Review E 51(4), 1995.

## Usage

```python
from sc_neurocore.chaos import ChaoticRNG, TentMapRNG

# Logistic map — generate SC bitstream
rng = ChaoticRNG(r=4.0, x=0.37)
bitstream = rng.generate_bitstream(p=0.7, length=10000)
print(f"P(1) = {bitstream.mean():.3f}")  # ≈ 0.700

# Quality check
print(f"Lyapunov: {rng.lyapunov_exponent():.4f}")  # ≈ 0.6931
print(f"Entropy: {rng.shannon_entropy():.2f} bits")

# Tent map — uniform output, no CDF correction
tent = TentMapRNG(mu=1.9999, x=0.37)
samples = tent.random(10000)
print(f"Mean: {samples.mean():.3f}")  # ≈ 0.500

# Vectorized parallel maps for bulk generation
bulk = rng.random_vectorized(size=100000, n_maps=8)
```

::: sc_neurocore.chaos.rng
    options:
      show_root_heading: true
      members:
        - ChaoticRNG
        - TentMapRNG
