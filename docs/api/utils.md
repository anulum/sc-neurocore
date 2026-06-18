# Utilities

Core utility modules for bitstream encoding/decoding, random number
generation, and signal analysis.

## Bitstream Encoding

::: sc_neurocore.utils.bitstreams.generate_bernoulli_bitstream

::: sc_neurocore.utils.bitstreams.generate_sobol_bitstream

::: sc_neurocore.utils.bitstreams.bitstream_to_probability

::: sc_neurocore.utils.bitstreams.value_to_unipolar_prob

::: sc_neurocore.utils.bitstreams.unipolar_prob_to_value

::: sc_neurocore.utils.bitstreams.BitstreamEncoder

### BitstreamEncoder in chaotic mode

`BitstreamEncoder(mode="chaotic")` uses `sc_neurocore.chaos.rng.ChaoticRNG`
for `encode()`.

- Inputs are first mapped into unipolar probability.
- Output is produced with the same bitstream interface as Bernoulli/Sobol.
- The encode/decode API and output semantics remain unchanged.
- Equal seeds initialise the same logistic-map state; different seeds initialise
  different states.

For chaotic-rng test coverage and implementation details:
- `src/sc_neurocore/utils/bitstreams.py`
- `src/sc_neurocore/chaos/rng.py`
- `tests/test_chaotic_encoder.py`
- [`docs/api/chaos.md`](chaos.md)

::: sc_neurocore.utils.bitstreams.BitstreamAverager

### Bipolar Encoding

::: sc_neurocore.utils.bitstreams.generate_bipolar_bitstream

::: sc_neurocore.utils.bitstreams.bipolar_to_value

### SC Division (CORDIV, Li et al. 2014)

::: sc_neurocore.utils.bitstreams.sc_divide

#### `sc_divide` contract

- Inputs must have the same shape (`numerator.shape == denominator.shape`).
- Output is a binary stream (`uint8` values in `{0, 1}`) with the same shape.
- Per-bit rule:
  - `x[t] = 1` -> `z[t] = 1`
  - `x[t] = 0` and `y[t] = 1` -> `z[t] = 0`
  - `x[t] = 0` and `y[t] = 0` -> `z[t] = z[t-1]` (hold)

See the hardware state-machine contract for the implementation-backed version:
[`docs/hardware/sc_cordiv.md`](../hardware/sc_cordiv.md).

CORDIV estimates `P(numerator=1) / P(denominator=1)` when the numerator
probability is not larger than the denominator probability. It is a sequential
stochastic circuit, so correlated streams and short streams can bias the
estimate; use this function as a bitstream contract, not as a floating-point
division replacement.

### Adaptive Bitstream Length

Compute minimum bitstream length for target precision via Hoeffding,
Chebyshev, or variance bounds.

::: sc_neurocore.utils.bitstreams.adaptive_length

#### `adaptive_length` contract

- Supported methods: `hoeffding`, `chebyshev`, `variance`.
- Output is at least `min_length` and at most `max_length`.
- Returned length is rounded up to a power of two (Sobol compatibility).
- Invalid parameters (`epsilon <= 0`, unknown methods, `confidence >= 1`) raise
  `ValueError`.

The method controls the precision/speed tradeoff:

| Method | Use when |
|--------|----------|
| `hoeffding` | Distribution-free confidence bound for Bernoulli streams. |
| `chebyshev` | Variance-aware confidence bound where `p` matters. |
| `variance` | Quick sizing from `p(1-p)/epsilon^2` without confidence input. |

Validation:
- `tests/test_adaptive_length.py`
- `tests/test_cordiv_division.py` (empirical check of division with generated stream lengths)

## LDS Decorrelation (Sobol/Halton)

Multi-dimensional low-discrepancy sequences for per-synapse decorrelation.

::: sc_neurocore.utils.lds_decorrelation.generate_decorrelated_bitstreams

::: sc_neurocore.utils.lds_decorrelation.star_discrepancy_estimate

## Random Number Generation

::: sc_neurocore.utils.rng.RNG

## Adaptive Utilities

::: sc_neurocore.utils.adaptive

## Connectome Generation

::: sc_neurocore.utils.connectomes

## Decorrelators

::: sc_neurocore.utils.decorrelators

## Fault Injection

::: sc_neurocore.utils.fault_injection

## FSM Activations

::: sc_neurocore.utils.fsm_activations

## Model Bridge

::: sc_neurocore.utils.model_bridge
