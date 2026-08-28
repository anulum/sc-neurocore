# SCUpwardCrossingRulkovMapNeuron

`SCUpwardCrossingRulkovMapNeuron` preserves the event convention historically
exposed by SC-NeuroCore's Rulkov implementation. It is a count-neutral project
identity: its state recurrence is the Rulkov (2002) three-branch map, while its
configurable upward-crossing event is an SC-NeuroCore observation rule and is
not attributed to that paper.

- Python module:
  `sc_neurocore.neurons.models.sc_upward_crossing_rulkov_map`
- Family: discrete map
- State: `x`, `y`
- Project parameter: `x_threshold`
- Source-model count contribution: none

## Equations and state recurrence

The state update is identical to `RulkovMapNeuron`:

$$
x_{n+1} =
\begin{cases}
\dfrac{\alpha}{1-x_n} + y_n + I_n, & x_n \le 0, \\
\alpha+y_n+I_n, & 0 < x_n < \alpha+y_n+I_n, \\
-1, & x_n \ge \alpha+y_n+I_n,
\end{cases}
$$

$$
y_{n+1}=y_n-\mu(x_n+1)+\mu\sigma.
$$

## Retained event convention

After evaluating the simultaneous candidate, the retained event is

$$
s_n^{SC}=\mathbf{1}\left[x_{n+1}\ge\theta
\land x_n<\theta\right],
$$

where $\theta$ is `x_threshold` and defaults to zero. The event observes a
rising crossing and does not change either state coordinate.

For the default state and `I=2.0`, the first three event vectors make the
identity boundary explicit:

| Identity | First three events | State after three iterations |
|---|---|---|
| `RulkovMapNeuron` | `[0, 0, 1]` | same shared recurrence state |
| `SCUpwardCrossingRulkovMapNeuron` | `[1, 0, 0]` | same shared recurrence state |

## Python API

```python
from sc_neurocore.neurons.models import SCUpwardCrossingRulkovMapNeuron

neuron = SCUpwardCrossingRulkovMapNeuron(x_threshold=0.25)
event = neuron.step(current=0.5)
trace, event_count = neuron.simulate(2_000_000, 0.5, backend="rust")
```

The same `python`, `rust`, `julia`, `go`, `mojo`, and `auto` backends are
available. Thresholds, parameters, state, current, and candidate values must be
finite; `alpha` and `mu` must be positive. Batch failures are atomic.

## Independent and runtime verification evidence

The independent 512-step project receipt uses the same mixed drive as the
source identity and records:

| Evidence field | Value |
|---|---:|
| Upward-crossing events | 22 |
| First event index | 2 |
| Final `x` | -2.0005062893863803 |
| Final `y` | -3.795415634023709 |
| `<ddB>` row-stream SHA-256 | `b50ca7c0fdfa3cd17c1ad5676951921f5df6dc18a298e3909ff065fa37b6f68e` |

The state totals match the source identity because only the observation
surface differs. The complete receipt is
`src/sc_neurocore/neurons/reference_receipts/sc_upward_crossing_rulkov_project.json`.

The source-bound 2,000,000-step `I=0.5` benchmark records 34 events in every
runtime. Rust, Julia, and Go are binary64-exact to Python; Mojo's maximum
absolute state difference is `1.7763568394002505e-15`. The measured non-isolated
receipt is `benchmarks/results/bench_sc_upward_crossing_rulkov_map.json`.

## RTL, synthesis, and formal boundary

The paired schemas retain `x_threshold=0` as an explicit rising-crossing
contract. Hand Python, TOML, JSON, and Q16.16 RTL have exact event vectors for
30 iterations at `I=0`, `0.5`, and `1.5`; maximum enrolled state errors remain
below `0.006` for `x` and `0.001` for `y`.

The generated core is
`hdl/formal/catalogue/sc_upward_crossing_rulkov_map.v`. Yosys
`synth_xilinx` succeeds with 26,692 cells and two `DSP48E1` cells; the raw report
is
`hdl/reports/yosys_sc_upward_crossing_rulkov_map_q1616_2026-08-28.json`. Its
SymbiYosys/Z3 depth-4 job proves the bounded reset-output property. Timing, PPA,
placed-device, board, silicon, long-window fixed-point identity, and universal
equivalence remain unclaimed.
