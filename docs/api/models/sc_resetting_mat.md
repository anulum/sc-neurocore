# SC resetting MAT neuron

**Class:** `sc_neurocore.neurons.models.sc_resetting_mat.SCResettingMATNeuron`
**Identity:** retained SC-NeuroCore project recurrence; no publication attribution

## Why this model exists

The former `MATNeuron` implementation was not the non-resetting MAT* model in
Kobayashi et al. (2009). It used a candidate-first classical RK4 step, a voltage
reset, and project-selected threshold increments. The recurrence remains useful
and already had downstream compatibility evidence, so it is preserved rather
than deleted or silently changed. Its explicit `SC...` name prevents it from
being mistaken for the paper model.

## Recurrence

Between events, the three states obey

$$
\frac{dV}{dt}=\frac{-(V-V_{rest})+RI}{\tau_m},\qquad
\frac{d\theta_1}{dt}=-\frac{\theta_1}{\tau_1},\qquad
\frac{d\theta_2}{dt}=-\frac{\theta_2}{\tau_2}.
$$

All three candidates are advanced together with one classical RK4 step. If

$$V_{candidate}\ge V_{threshold,base}+\theta_{1,candidate}+\theta_{2,candidate},$$

the committed state is

$$V\leftarrow V_{reset},\quad
\theta_1\leftarrow\theta_{1,candidate}+h_1,\quad
\theta_2\leftarrow\theta_{2,candidate}+h_2.$$

Defaults are `V_rest=V_reset=-70 mV`, `V_threshold_base=-50 mV`,
`tau_m=tau_1=10 ms`, `tau_2=200 ms`, `h1=5 mV`, `h2=3 mV`, `R=1`, and
`dt=1 ms`.

```python
from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron

neuron = SCResettingMATNeuron()
events = [neuron.step(50.0) for _ in range(256)]
```

## Compatibility anchor and runtimes

The committed 256-step receipt uses 32 zero samples, 96 samples at 50, then 64
repetitions of `[20, 60]`. It records 13 events, final state
`(-70, 5.262135955944077, 21.149478444493045)`, and binary64 trace SHA-256
`b64411c28f4ab24e87fb52a115fd9379793412350af57a933806f1b6c32af259`.

Python, Rust engine, Rust safety, Julia, Go shared library, and Mojo shared
library implement the complete recurrence. The five accelerated paths preserve
the event vector exactly and complete states within `2e-12`. The 200,000-step
loaded-host benchmark is `benchmarks/results/bench_sc_resetting_mat.json`; it is
local regression evidence only.

## Hardware and catalogue boundary

The signed Q32.32 RTL matches an independent integer oracle and the 13-event
Python vector over the enrolled trace. It synthesizes in repository-local Yosys,
its optimized netlist is cycle-exact over that sequence, and its depth-12 CVC5
bounded-safety job passes. No timing, PPA, device, or universal binary64
equivalence claim is made.

This project-defined identity does not increment the literature-model
polyglot-fidelity count. It is not an alias for
[`MATNeuron`](mat.md), and neither class silently selects the other.
