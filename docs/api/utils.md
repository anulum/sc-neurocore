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

::: sc_neurocore.utils.bitstreams.BitstreamAverager

### Bipolar Encoding

::: sc_neurocore.utils.bitstreams.generate_bipolar_bitstream

::: sc_neurocore.utils.bitstreams.bipolar_to_value

### SC Division (CORDIV, Li et al. 2014)

::: sc_neurocore.utils.bitstreams.sc_divide

### Adaptive Bitstream Length

Compute minimum bitstream length for target precision via Hoeffding,
Chebyshev, or variance bounds.

::: sc_neurocore.utils.bitstreams.adaptive_length

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
