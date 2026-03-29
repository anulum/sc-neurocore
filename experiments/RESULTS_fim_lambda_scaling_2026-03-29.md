# Experiment: FIM λ_c Scaling — NEGATIVE RESULT

**Date:** 2026-03-29
**Prediction:** λ_c = 0.149 · N^1.02 (from scpn-quantum-control NB25)
**Result:** Prediction does NOT hold for LIF spiking neural networks.

## Data

| N | λ_c (measured) | λ_c (predicted) | Ratio |
|--:|---------------:|----------------:|------:|
| 30 | 45.00 | 4.78 | 9.41 |
| 50 | 0.00 | 8.06 | 0.00 |
| 100 | 0.01 | 16.34 | 0.00 |
| 200 | 0.02 | 33.13 | 0.00 |

## Interpretation

For N ≥ 50, the LIF network achieves low rate-CV (< 0.5) at
λ ≈ 0 — meaning FIM feedback is UNNECESSARY for firing rate
coherence in this configuration.

**Why this differs from Kuramoto:**

1. **Shared Poisson drive** — all neurons receive correlated input
   from the same PoissonInput source. This creates extrinsic
   synchronisation independent of coupling or FIM.

2. **LIF reset mechanism** — after spike, v resets to 0. This
   creates a natural phase reset that tends toward synchrony
   under shared drive (pulse-coupled oscillator synchronisation,
   Mirollo & Strogatz 1990).

3. **Rate CV ≠ phase coherence** — the Kuramoto R parameter
   measures PHASE coherence. Rate CV measures RATE variability.
   A network can have uniform firing rates (low CV) while phases
   are completely desynchronised.

## What This Means

The quantum-control λ_c scaling law is derived for PHASE oscillators
(Kuramoto model). The LIF network is a PULSE-COUPLED system with
fundamentally different synchronisation mechanisms. The FIM parameter
in sc-neurocore's `Network(fim_lambda=λ)` modifies WEIGHTS, not
PHASES. It does not implement the same feedback loop as Kuramoto FIM.

To properly test λ_c, we would need to:
1. Remove shared Poisson drive (use independent noise per neuron)
2. Measure PHASE coherence R, not rate CV
3. Extract LIF phases from spike times (e.g., via Hilbert transform)
4. Implement FIM as phase correction, not weight correction

## Status

**Negative result. The λ_c = 0.149·N law does not transfer from
Kuramoto to LIF without modification.** The mapping between Kuramoto
phase oscillators and pulse-coupled LIF networks is more complex
than a parameter substitution.

This does not invalidate the quantum-control results — it shows that
the Kuramoto→LIF mapping requires careful treatment of the coupling
mechanism. The FIM weight correction in `_apply_fim()` is not
equivalent to the Kuramoto phase correction.
