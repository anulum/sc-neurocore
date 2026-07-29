<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->

# Amari neural field

`AmariNeuralField` is SC-NeuroCore's finite periodic-grid specialization of
Amari's homogeneous single-layer lateral-inhibition field. It follows equation
(3) of Amari (1977), uses the paper's level output, and reports a continuous
population rate rather than a biological single-neuron spike.

- Python: `sc_neurocore.neurons.models.amari_field.AmariNeuralField`
- Dispatcher: `sc_neurocore.accel.amari_field.simulate_amari_field`
- Source: [Amari 1977, DOI 10.1007/BF00337259](https://doi.org/10.1007/BF00337259)

## Maintained equation

For site `i` on a periodic uniform grid,

$$
\tau\frac{du_i}{dt} = -u_i
  + \Delta x\sum_j w(d(i,j))H(u_j) + I_i,
$$

where

$$
w(d)=A\exp(-a d)-B\exp(-b d),\qquad
H(u)=\begin{cases}1&u>0\\0&u\le 0.\end{cases}
$$

`d(i,j)` is the shortest distance around the periodic ring. One `step` uses a
simultaneous explicit-Euler update and returns the fraction of sites whose new
state is positive.

The defaults are `A=1.5`, `a=2.0`, `B=0.75`, and `b=1.0`. They give a positive
central weight and a negative distal surround. The earlier `a=1`, `b=2`
combination did not provide lateral inhibition at long range and is no longer
the maintained default. The implementation evaluates the interaction matrix
directly; it does not claim an FFT algorithm.

## Public contract

```python
import numpy as np

from sc_neurocore.neurons.models.amari_field import AmariNeuralField

field = AmariNeuralField(n=8)
rate = field.step(np.linspace(-0.1, 0.1, 8))

currents = np.zeros((32, 8), dtype=np.float64)
receipt = field.simulate(currents, backend="python")
assert receipt["states"].shape == (32, 8)
assert receipt["mean_rates"].shape == (32,)
assert receipt["final_state"].shape == (8,)
```

`step(current)` accepts either one finite scalar, broadcast across the field,
or a finite vector of shape `(n,)`. `simulate(currents)` accepts homogeneous
drive with shape `(steps,)` or spatial drive with shape `(steps, n)`. Invalid
configuration, input, or candidate state fails before committing a partial
state update. `reset()` zeros the field while preserving configuration.

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `n` | `64` | number of periodic field sites, at least two |
| `tau` | `10.0` | positive field time constant |
| `a_exc` | `1.5` | local-excitation amplitude `A` |
| `a_width` | `2.0` | excitation inverse width `a` |
| `b_inh` | `0.75` | distal-inhibition amplitude `B` |
| `b_width` | `1.0` | inhibition inverse width `b` |
| `dx` | `0.5` | positive spatial interval |
| `dt` | `0.5` | positive Euler timestep |
| `u` | zeros | optional finite initial vector of shape `(n,)` |

## Execution lanes and documentation

The same complete vector-state and mean-rate receipt is executable through
Python, the modular Rust/PyO3 engine, Julia, Go C-shared, and Mojo shared-library
lanes. Native failure does not silently fall back to Python. The independent
Rust safety mirror is compiled and tested separately.

Public native surfaces carry their language-native documentation:

- Rust engine and safety types use Rustdoc;
- the Go service and C ABI use GoDoc comments;
- Julia constructors and stepping functions use Julia docstrings;
- Mojo kernels and C ABI entry points document shapes and update semantics;
- RTL comments declare fixed-point formats, latency, reset, and saturation.

## Evidence boundary

- An independently evaluated 8-site, 32-step receipt checks the paper equation,
  Heaviside output, kernel, and simultaneous update.
- Paired TOML and JSON schemas execute the same four-site specialization.
- Five-runtime tests compare every state and active-site fraction. Rust, Julia,
  and Go are bounded by `2e-10`; Mojo is bounded by `5e-10`; rates are exact.
- The committed benchmark measures 20,000 steps on 16 sites, binds source and
  native-binary hashes, and is local regression evidence only.
- Q16.16 RTL exposes all four enrolled states and the Q16.16 active-site rate.
  Co-simulation is bit-exact to its integer oracle and remains within `0.0025`
  of the float source state.
- Yosys synthesis passes. The depth-12 SymbiYosys/Z3 job proves its declared
  reset, event-silence, rate-bound, and saturation-safety properties by
  induction.

This evidence does not establish continuous-space convergence, axonal delays,
the paper's two-layer oscillatory system, a biological spike identity, timing,
PPA, device validation, or formal equivalence to the binary64 implementation.
Localized activity and other qualitative regimes depend on parameters and
inputs; the defaults alone are not presented as proof of a persistent bump.
