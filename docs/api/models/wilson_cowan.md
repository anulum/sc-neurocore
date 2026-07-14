# Wilson-Cowan E/I population rates

- **Class:** `WilsonCowanUnit`
- **Module:** `sc_neurocore.neurons.models.wilson_cowan`
- **Source family:** Wilson and Cowan (1972), coupled excitatory/inhibitory population equations
- **DOI:** [10.1016/S0006-3495(72)86068-5](https://doi.org/10.1016/S0006-3495(72)86068-5)

`WilsonCowanUnit` integrates two coupled continuous population activities:

\[
\tau_e\frac{dE}{dt}=-E+S(w_{ee}E-w_{ei}I+I_{ext}),
\]

\[
\tau_i\frac{dI}{dt}=-I+S(w_{ie}E-w_{ii}I),
\]

with the shifted logistic

\[
S(x)=\operatorname{logistic}(a(x-\theta))-
     \operatorname{logistic}(-a\theta).
\]

The subtraction makes \(S(0)=0\). Both state variables advance together with
one fixed-step fourth-order Runge-Kutta update.

## Scientific boundary

The primary article establishes the coupled E/I population framework and
derives population equations containing response functions plus
availability/refractory factors. SC-NeuroCore maintains a declared normalised
reduction: it omits those availability factors and provides external drive only
to the excitatory equation. It therefore belongs to the Wilson-Cowan family but
is not a verbatim transcription of every factor in the paper's final equations.

`e` and `i` are continuous population activities. A non-zero value is not a
binary spike, and this model exposes no spike counter or reset threshold.

## Parameters and state

| Name | Default | Constraint | Meaning |
|---|---:|---|---|
| `e` | `0.1` | finite `[-beta, 1]` state envelope | excitatory activity |
| `i` | `0.05` | finite `[-beta, 1]` state envelope | inhibitory activity |
| `w_ee` | `10.0` | finite, `>= 0` | recurrent E-to-E weight |
| `w_ei` | `6.0` | finite, `>= 0` | I-to-E weight |
| `w_ie` | `10.0` | finite, `>= 0` | E-to-I weight |
| `w_ii` | `1.0` | finite, `>= 0` | recurrent I-to-I weight |
| `tau_e` | `1.0` | finite, `> 0` | excitatory time constant |
| `tau_i` | `2.0` | finite, `> 0` | inhibitory time constant |
| `a` | `1.2` | finite, `> 0` | sigmoid steepness |
| `theta` | `4.0` | finite | sigmoid midpoint |
| `dt` | `0.1` | finite, `> 0` | RK4 step size |

## Python use

```python
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

unit = WilsonCowanUnit()
e_trace, i_trace = unit.simulate(1_000, current=1.5, backend="auto")

assert e_trace.shape == i_trace.shape == (1_000,)
assert (unit.e, unit.i) == (e_trace[-1], i_trace[-1])
```

`simulate` accepts `auto`, `python`, `rust`, `julia`, `go`, or `mojo`.
Explicitly requested unavailable runtimes raise instead of silently
substituting Python. A successful batch commits the final E/I pair only after
the complete result passes shape, finiteness, range, and final-state checks. An
empty batch preserves both rates.

## Reset

```python
unit = WilsonCowanUnit(w_ee=12.0, tau_i=3.0, theta=3.5, dt=0.05)
unit.step(5.0)
unit.reset()

assert (unit.e, unit.i) == (0.1, 0.05)
assert (unit.w_ee, unit.tau_i, unit.theta, unit.dt) == (12.0, 3.0, 3.5, 0.05)
```

Reset changes only the dynamic E/I state.

## Executable backends

| Runtime | Maintained surface | Contract |
|---|---|---|
| Python | scalar model plus public atomic batch | configurable RK4 reference |
| Rust engine | modular PyO3 batch | validated, failure-atomic E/I traces |
| Rust safety | independently compiled module | validated scalar RK4 state |
| Julia | `WilsonCowanAccel.simulate_wilson_cowan!` | configurable atomic batch |
| Go | generated C-shared ABI | configurable atomic batch |
| Mojo | exported shared-library C ABI | two-pass atomic batch |

Rust, Julia, and Go reproduce the complete enrolled E/I trajectory within
`1e-9` of Python. Mojo remains within `1e-8` over the 100,000-step horizon;
its smaller per-operation libm/FMA differences accumulate but remain bounded.
Go, Julia, Mojo, and Rust failures leave caller-visible state or output buffers
unchanged.

## Validation and evidence

The focused cohort covers:

- constructor and mutable-state validation;
- shifted-sigmoid baseline, range, and saturation;
- sustained oscillation in an enrolled recurrent-feedback regime;
- candidate-first RK4 separation from Euler;
- complete configured Python/Rust/Julia/Go/Mojo trajectories;
- empty batches, unavailable runtimes, malformed results, and atomic failures;
- configuration-preserving reset;
- paired TOML/JSON declarative schemas;
- source- and native-binary-bound local benchmark evidence.

See [Wilson-Cowan source fidelity](../../validation/wilson_cowan_source_fidelity.md)
for the evidence matrix and reproduction commands.

## Hardware boundary

This unit makes no fixed-point schema-execution, RTL, formal-equivalence,
synthesis, timing, or device claim. Its exponential transfer and continuous
coupled state require a separate quantisation and silicon-enrolment study.

## Reference

H. R. Wilson and J. D. Cowan, “Excitatory and Inhibitory Interactions in
Localized Populations of Model Neurons,” *Biophysical Journal*, vol. 12,
pp. 1–24, 1972.
[Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC1484078/).
