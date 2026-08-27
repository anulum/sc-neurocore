# SCResettingParallelSpikingNeuron

**Module:** `sc_neurocore.neurons.models.sc_resetting_psn`
**Provenance:** SC-NeuroCore project recurrence (count-neutral preserved
identity; no publication-exact claim)
**Family:** Hardware / neuromorphic (windowed integrator with
spike-triggered reset)
**State variables:** `buffer` (circular, size `kernel_size`), `_ptr`
(write pointer)

---

## Identity

Historical repository model formerly published as
`ParallelSpikingNeuron`. It is structurally distinct from the
Fang et al. (2023) sliding PSN in two ways:

1. **Spike-triggered buffer reset.** The whole input buffer is zeroed
   whenever the neuron fires. The PSN family has no reset — removing
   the reset is that paper's core premise — so this recurrence cannot
   carry the publication-exact identity.
2. **Circular kernel pairing.** A replaced (non-uniform) `kernel` is
   dotted against circular buffer slots, not time-ordered inputs, so
   the weight-to-input pairing rotates with the write pointer. For the
   default uniform kernel the pairing is order-free and this
   difference vanishes.

Finite-input trajectories are preserved bit for bit from the
pre-2026-08-27 implementation; the frozen anchors live in
`tests/test_model_sc_resetting_psn.py`. The class consumes no
source-catalogue slot (count-neutral).

---

## Equations (as preserved)

$$\text{score}[t] = \sum_{i=0}^{n-1} \text{kernel}_i \cdot \text{buffer}_i,
\qquad n = \min(t+1, k)$$

Spike when $\text{score} \ge V_{th}$, then $\text{buffer} \leftarrow 0$.
During warm-up the score divides by the full `kernel_size`, which for
the default uniform kernel matches zero-padded pre-history.

With the default uniform kernel and constant input $I = V_{th}$ the
neuron fires every $\lceil V_{th} \cdot k / I \rceil$ steps: the buffer
fills, the mean reaches threshold, the buffer clears, and the cycle
repeats.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_size` | 8 | Circular buffer and kernel length |
| `v_threshold` | 1.0 | Score threshold for spike emission |
| `kernel` | uniform 1/k | Replaceable weight array |

Validation added over the historical code: construction and every step
reject a non-positive or non-integer `kernel_size`, a non-finite or
reshaped kernel or buffer, and a non-finite input with a typed
`ValueError`; rejection is atomic. Valid trajectories are unchanged.

---

## Backend inventory

Python-only boundary: the preserved recurrence is intentionally not
mirrored in the engine, Go, Julia, Mojo, or silicon lanes. The
canonical [`ParallelSpikingNeuron`](psn.md) carries the cross-language
closure.
