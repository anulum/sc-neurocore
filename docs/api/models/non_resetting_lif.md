# Kobayashi MAT(1) non-resetting neuron

**Class:** `sc_neurocore.neurons.models.non_resetting_lif.NonResettingLIFNeuron`
**Source:** Kobayashi, Tsubo, and Shinomoto (2009),
[*Made-to-order spiking neuron model equipped with a multi-timescale adaptive threshold*](https://doi.org/10.3389/neuro.10.009.2009)

## Identity

`NonResettingLIFNeuron` is the one-timescale MAT(1) member of the Kobayashi
family. It is not the standard fire-and-reset generalized integrate-and-fire
model described by Jolivet et al. Voltage evolves continuously through a spike;
the event raises one threshold-history state and starts a 2 ms absolute
refractory interval.

The former SC-NeuroCore exact-relaxation recurrence remains available under the
explicit project identity
[`SCNonResettingAdaptiveLIFNeuron`](sc_non_resetting_adaptive_lif.md).

## Maintained equations

The relative-to-rest membrane follows source Equation 1:

$$
\tau_m \frac{dV}{dt}=-V+RI.
$$

One maintained step uses the paper's 0.001 ms forward-Euler discretisation:

$$
V_{n+1}=V_n+\frac{\Delta t}{\tau_m}(-V_n+RI_n).
$$

MAT(1) has one exponentially decaying spike-history contribution:

$$
\theta(t)=\omega+\sum_k \alpha e^{-(t-t_k)/\tau_\theta}.
$$

The public `theta` state stores only the history sum; `threshold` returns
`omega + theta`. A sampled level event is eligible after the refractory timer
decays to zero. It increments `theta` by `alpha` and reloads the timer without
resetting or freezing voltage.

## Default specialization

| Parameter | Default | Meaning |
|---|---:|---|
| `omega` | 19 mV | baseline threshold |
| `tau_m` | 5 ms | membrane time constant |
| `tau_theta` | 50 ms | source-selected MAT(1) threshold timescale |
| `alpha` | 37 mV | documented threshold-history increment |
| `resistance` | 50 MOhm | current-to-voltage factor |
| `refractory_period` | 2 ms | absolute event-suppression interval |
| `dt` | 0.001 ms | source simulation timestep |

The paper selects 50 ms as the optimal MAT(1) timescale but fits the threshold
baseline and amplitude per neuron. These defaults are therefore a documented
numerical specialization, not a universal biological-cell calibration.

```python
from sc_neurocore.neurons.models.non_resetting_lif import NonResettingLIFNeuron

neuron = NonResettingLIFNeuron()
events = [neuron.step(0.7) for _ in range(5000)]
print(sum(events), neuron.v, neuron.threshold)
```

## Executable runtimes

| Lane | Surface | Enrolled parity |
|---|---|---|
| Python | `NonResettingLIFNeuron.step` | reference |
| Rust engine | class plus `py_non_resetting_lif_simulate` | complete trace within `2e-12` |
| Rust safety | `accel/rust/safety/non_resetting_lif.rs` | complete trace within `2e-12` |
| Julia | `NonResettingLifAccel` | complete trace within `2e-12` |
| Go | `accel/go/non_resetting_lif/libnon_resetting_lif.so` | complete trace within `2e-12` |
| Mojo | `accel/mojo/non_resetting_lif/libnon_resetting_lif.so` | complete trace within `2e-12` |

`sc_neurocore.accel.non_resetting_lif.simulate_non_resetting_lif` exposes the
complete batch contract. Unknown or unavailable explicitly requested backends
fail; they are not silently replaced.

## Reproducibility and hardware boundary

The independent 10,272-step direct-equation receipt records one event at index
3945, final state
`[27.965935062410335, 32.60279147075955, 0.0]`, and trace SHA-256
`2ac13e42322a3ac6b4059f29190f0936409c9d4bf28f1837e4bee97add2069c6`.
Paired TOML/JSON schemas reproduce the same discrete contract.

The hand-written signed Q32.32 RTL is cycle-exact to an independent integer
oracle and preserves the enrolled Python event vector. Yosys synthesis passes,
the optimized netlist is sequence-exact on the checked drive, and a depth-12
CVC5 job proves bounded reset/state/refractory/event safety. These are H1
results: universal binary64 equivalence, timing, PPA, and device evidence are
not claimed.

The source/binary-bound 200,000-step five-runtime benchmark records four exact
events at 0.7 nA. It is loaded-host local regression evidence, not a production
speed or hardware-performance claim.

See [dual-identity source and runtime evidence](../../validation/non_resetting_lif_source_fidelity.md).
