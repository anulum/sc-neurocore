# RulkovMapNeuron

`RulkovMapNeuron` implements the three-branch fast/slow map in Rulkov (2002),
Equations 1–2. It is a discrete map: one call to `step()` performs one
simultaneous iteration, with no ODE solver or `dt` parameter.

- Python module: `sc_neurocore.neurons.models.rulkov_map`
- Source: N. F. Rulkov, *Modeling of spiking-bursting neural behavior using
  two-dimensional map*, Physical Review E 65, 041922 (2002)
- DOI: [10.1103/PhysRevE.65.041922](https://doi.org/10.1103/PhysRevE.65.041922)
- Family: discrete map
- State: fast coordinate `x`, slow coordinate `y`

## Recurrence

SC-NeuroCore binds the scalar caller input `current` to the paper's fast
control $\beta_n$ and retains `sigma` as the static slow control:

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

Both next-state coordinates are evaluated from the same pre-update state.
`current` is therefore not added after every branch; it participates in the
first two branches and in the branch boundary exactly as $\beta_n$ does.

## Source event convention

Rulkov identifies a spike with appearance in the rightmost interval. The
public event is therefore evaluated from the pre-update state:

$$
s_n = \mathbf{1}\left[x_n>0 \land
x_n\ge\alpha+y_n+I_n\right].
$$

The event marks the iteration that executes the hard-reset branch and commits
$x_{n+1}=-1$. It is not an upward crossing of `x=0` and there is no
`x_threshold` parameter on `RulkovMapNeuron`.

The historical SC-NeuroCore upward-crossing behavior remains available as the
separate count-neutral
[`SCUpwardCrossingRulkovMapNeuron`](sc_upward_crossing_rulkov_map.md). The two
classes share the state recurrence and differ only in their observation event.

## Parameters and repository profile

| Field | Default | Contract |
|---|---:|---|
| `x` | -1.0 | finite initial fast state |
| `y` | -3.0 | finite initial slow state |
| `alpha` | 4.0 | finite and positive fast-map control |
| `sigma` | -1.6 | finite slow-map control |
| `mu` | 0.001 | finite and positive slow timescale |

The default state and `sigma=-1.6` are the repository's quiescent operating
profile. They are not presented as a unique parameter set from a source figure.

## Python API

```python
from sc_neurocore.neurons.models import RulkovMapNeuron

neuron = RulkovMapNeuron()
event = neuron.step(current=0.5)
trace, event_count = neuron.simulate(2_000_000, 0.5, backend="auto")
```

`simulate()` returns the post-update `x` trace and total reset-branch event
count, then commits the final `(x, y)` to the instance. The batch is
failure-atomic: invalid parameters, state, input, step count, backend, or a
non-finite candidate raises before instance state is changed.

Available backends are `python`, `rust`, `julia`, `go`, and `mojo`; `auto`
selects an available compiled backend and retains Python as the floor.

## Independent and cross-runtime evidence

The committed 512-step receipt uses the repeated mixed drive
`[0.0, 0.25, 1.5, -0.1]`. An independent literal implementation of the source
equations records:

| Evidence field | Value |
|---|---:|
| Reset-branch events | 22 |
| First event index | 7 |
| Final `x` | -2.0005062893863803 |
| Final `y` | -3.795415634023709 |
| `<ddB>` row-stream SHA-256 | `83ffbbdf6fe825fd6cb1833078ed2aa3836ffbad876ca38af95780914a16c3db` |

The receipt is
`src/sc_neurocore/neurons/reference_receipts/rulkov_2002.json`; its test derives
the recurrence independently and also checks the public model.

The source-bound 2,000,000-step benchmark at constant `I=0.5` executes every
runtime. Python, Rust, Julia, and Go produce identical binary64 traces; Mojo's
maximum absolute difference is `1.7763568394002505e-15`. All five lanes record
34 source events. The measured receipt is
`benchmarks/results/bench_rulkov_map.json`. It is non-isolated local-regression
evidence, not a release-performance claim.

## RTL, synthesis, and formal boundary

The TOML and JSON schemas preserve the simultaneous recurrence and source
rightmost-branch event. Hand Python, both schema loaders, and generated Q16.16
RTL are checked for 30 iterations at `I=0`, `0.5`, and `1.5`:

| Current | Event count | Maximum enrolled `x` error | Maximum enrolled `y` error |
|---:|---:|---:|---:|
| 0.0 | 0 | 0.0011 | 0.0003 |
| 0.5 | 7 | 0.0055 | 0.0006 |
| 1.5 | 10 | 0.0010 | 0.0008 |

The `I=1.5` window visits each of the rational, plateau, and reset branches ten
times. Event vectors are exact at all three enrolled drives. Bounded trajectories
are the valid fixed-point metric for this sensitive map; no long-window Q16.16
trajectory identity is claimed.

The generated core is `hdl/formal/catalogue/sc_rulkov_map.v`. Yosys
`synth_xilinx` succeeds with 26,480 cells and two `DSP48E1` cells; the raw report
is `hdl/reports/yosys_rulkov_map_q1616_2026-08-28.json`. Its SymbiYosys/Z3 job
passes bounded reset-output safety at depth 4. These are H2 compile,
co-simulation, synthesis, and bounded-safety results. Timing, PPA, placed-device,
board, silicon, and universal real-number equivalence remain unclaimed.
