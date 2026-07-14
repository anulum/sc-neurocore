# Threshold-linear rate transfer

- **Class:** `ThresholdLinearRateNeuron`
- **Module:** `sc_neurocore.neurons.models.threshold_linear_rate`
- **Source scope:** Gerstner, Kistler, Naud, and Paninski (2014), Eq. 18.23
- **Book DOI:** [10.1017/CBO9781107447615](https://doi.org/10.1017/CBO9781107447615)

`ThresholdLinearRateNeuron` evaluates a rectified, piecewise-linear continuous
rate:

\[
r = g\,[I-\theta]_+ = g\max(0, I-\theta).
\]

The online *Neuronal Dynamics* text defines the underlying piecewise-linear
gain as \(F(h)=[h]_+\) in
[Section 18.2, Eq. 18.23](https://neuronaldynamics.epfl.ch/online/Ch18.S2.html).
SC-NeuroCore exposes a finite threshold \(\theta\) and non-negative gain
\(g\), which translate and scale that declared transfer.

## Model boundary

This is an algebraic gain function, not a differential equation. Each
successful call overwrites `r` with the output for the current input. The
cached `r` is useful for inspection and network integration, but it does not
carry history into the next evaluation.

- No `dt`, time constant, numerical integrator, or hidden temporal state is
  part of the contract.
- A positive `r` is a continuous rate, not a binary spike event.
- The transfer is unbounded above when both input and gain are unbounded; all
  maintained runtimes reject non-finite results before mutating visible state.

## Parameters and state

| Name | Default | Constraint | Meaning |
|---|---:|---|---|
| `r` | `0.0` | finite, `>= 0` | cached latest continuous output |
| `theta` | `0.0` | finite | onset threshold in input units |
| `gain` | `1.0` | finite, `>= 0` | slope above threshold |

## Python use

```python
from sc_neurocore.neurons.models.threshold_linear_rate import (
    ThresholdLinearRateNeuron,
)

neuron = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)

assert neuron.step(1.0) == 0.0
assert neuron.step(1.5) == 0.0
assert neuron.step(3.0) == 3.0

trace = neuron.simulate(64, current=3.0, backend="auto")
assert trace.shape == (64,)
assert neuron.r == 3.0
```

`simulate` accepts `auto`, `python`, `rust`, `julia`, `go`, or `mojo`.
Explicitly requested unavailable runtimes raise instead of substituting
Python. A successful batch commits only its final validated output; an empty
batch preserves the initial cache.

## Reset

```python
neuron = ThresholdLinearRateNeuron(r=4.0, theta=-0.4, gain=2.5)
neuron.reset()
assert (neuron.r, neuron.theta, neuron.gain) == (0.0, -0.4, 2.5)
```

Reset clears only the cached output.

## Executable backends

| Runtime | Maintained surface | Contract |
|---|---|---|
| Python | model plus public dispatcher | configurable scalar and atomic batch |
| Rust engine | modular PyO3 binding | configurable scalar and batch |
| Rust safety | independently compiled module | validated scalar transfer |
| Julia | `ThresholdLinearRateAccel.simulate_trace` | configurable batch |
| Go | service plus generated C-shared ABI | configurable atomic batch |
| Mojo | exported shared-library C ABI | two-pass atomic batch |

All five public dispatcher lanes reproduce the complete enrolled float64 rate
trace bit-for-bit. The generated Go and Mojo destinations include a final-rate
slot, and invalid contracts leave the caller buffer unchanged.

## Validation and evidence

The focused test cohort covers:

- below-threshold, equality, and above-threshold branches;
- memorylessness and configuration-preserving reset;
- constructor, mutable-state, input, and overflow rejection;
- schema-map parity with the hand model;
- real Rust engine, standalone Rust safety, Julia, Go C-shared, and Mojo
  shared-library execution;
- exact five-runtime traces, empty batches, unavailable runtimes, and C-ABI
  failure atomicity;
- source- and binary-bound local benchmark evidence.

See [Threshold-linear rate source fidelity](../../validation/threshold_linear_rate_source_fidelity.md)
for the evidence matrix and reproduction commands.

## Hardware boundary

The subtract/compare/multiply transfer is a plausible fixed-point hardware
candidate, but this unit makes no RTL claim. Quantisation, registered or folded
RTL, formal equivalence, synthesis, timing, and device measurements require a
separate silicon-enrolment artifact.

## Reference

W. Gerstner, W. M. Kistler, R. Naud, and L. Paninski,
*Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition*,
Cambridge University Press, 2014.
[doi:10.1017/CBO9781107447615](https://doi.org/10.1017/CBO9781107447615).
