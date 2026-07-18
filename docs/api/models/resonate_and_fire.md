<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Source-bound resonate-and-fire contract -->

# Izhikevich resonate-and-fire neuron

- **Class:** `ResonateAndFireNeuron`
- **Module:** `sc_neurocore.neurons.models.resonate_and_fire`
- **Source:** Izhikevich (2001), equations (1–3)
- **DOI:** [10.1016/S0893-6080(01)00078-8](https://doi.org/10.1016/S0893-6080(01)00078-8)

The source defines a complex linear resonator, identifies its real coordinate
as current-like and its imaginary coordinate as voltage-like, thresholds the
voltage coordinate, and installs `z=i` after a spike. SC-NeuroCore preserves
those identities. It does not use a radius threshold and does not reset to the
origin.

## Equations and exact maintained step

With \(z=x+i y\) and real piecewise-constant input \(I\),

\[
\frac{dz}{dt}=(b+i\omega)z+I,
\]

or equivalently

\[
\dot x=bx-\omega y+I,
\qquad
\dot y=\omega x+by.
\]

For one interval `dt`, define

\[
x_{ss}=\frac{-bI}{b^2+\omega^2},
\qquad
y_{ss}=\frac{\omega I}{b^2+\omega^2}.
\]

The maintained candidate is the exact constant-input flow

\[
x'=x_{ss}+e^{b\,dt}
\left[(x-x_{ss})\cos(\omega dt)-(y-y_{ss})\sin(\omega dt)\right],
\]

\[
y'=y_{ss}+e^{b\,dt}
\left[(x-x_{ss})\sin(\omega dt)+(y-y_{ss})\cos(\omega dt)\right].
\]

The exact flow removes explicit-Euler stability error for this linear
subthreshold system. `dt=0.01` remains a sampling interval and event-detection
contract; it is not asserted to be a parameter selected by the paper.

## Spike event and reset

The source voltage threshold is generalized from \(y=1\) to
`y=threshold`. The discrete API emits one event on a sampled upward crossing:

```text
old_y < threshold <= candidate_y
```

On that event the post-step state is `(x, y) = (0, threshold)`, the generalized
form of the source reset `z=i`. No within-step root localization is claimed.
The sampled crossing rule also means constant-current spike count is not a
globally monotone rate code: after reset, sufficiently large drive can leave
the next sampled `y` above threshold, preventing a new upward crossing until
the trajectory falls below and crosses again.

## Parameters and state

| Name | Default | Constraint | Meaning |
|---|---:|---|---|
| `x` | `0.0` | finite | current-like real coordinate |
| `y` | `0.0` | finite | voltage-like imaginary coordinate |
| `b` | `-1.0` | finite | radial damping/growth coefficient |
| `omega` | `10.0` | finite, `> 0` | angular resonance frequency |
| `threshold` | `1.0` | finite, `> 0` | threshold on `y` |
| `dt` | `0.01` | finite, `> 0` | piecewise-constant sample interval |

`b=-1`, `omega=10`, threshold `y=1`, and reset `z=i` are the values and
contracts used in the source discussion. `reset()` restores `(x, y)=(0, 0)`
while preserving configuration.

## Scalar and batch use

```python
import numpy as np

from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron

unit = ResonateAndFireNeuron()
spike = unit.step(current=5.0)

index = np.arange(128, dtype=np.float64)
drive = 4.5 + 1.4 * np.sin(index * 0.037)
result = unit.simulate(drive, backend="auto")

assert result["x"].shape == (128,)
assert result["y"].shape == (128,)
assert result["spikes"].shape == (128,)
assert result["spike_count"] == int(result["spikes"].sum())
assert unit.y == result["y_final"]
```

The batch returns complete post-update `x`, `y`, and binary `spikes` traces,
both final-state receipts, and an integral `spike_count`. Explicitly selecting
an unavailable backend fails; it does not silently substitute Python. A model
instance mutates only after the complete backend result passes shape,
finiteness, binary-event, reset, final-state, and event-count checks. Empty
batches preserve both states.

## Executable runtimes

| Runtime | Maintained surface | Enrolled contract |
|---|---|---|
| Python | scalar reference and atomic batch | source reference |
| Rust engine | PyO3 scalar class and configured batch | complete result within `1e-12` |
| Rust safety | standalone `rustc` scalar module | equation, validation, reset, atomicity |
| Julia | JuliaCall batch | complete result within `1e-12` |
| Go | C-shared batch ABI | complete result within `1e-12` |
| Mojo | shared-library C ABI | complete result within `1e-10` |

The non-default parity operating point starts at `(x, y)=(0.13, -0.27)`,
changes every public parameter, drives 128 varied samples, and compares all
three trajectories, both final states, and the event count.

## Reproducibility and benchmark evidence

The descriptor’s 256-step varied-input reference interleaves `x`, `y`, and
`spikes` as little-endian float64. Its SHA-256 is
`d91c7ef4a469f7b50498943419288b12329b17ba8380bcc876d4963480ae7130`.
The independent DOI trace separately checks a source-default subthreshold
trajectory without importing production equation helpers.

`benchmarks/results/bench_resonate_and_fire.json` records the same configured
batch across all five public runtimes, exact source and binary hashes, raw
timing samples, affinity, tool versions, complete-trajectory digests, and
bounded parity deltas. These are local single-logical-CPU, non-exclusive
regression timings—not a production, cross-host, or hardware ranking.

## Python-to-Verilog and formal boundary

The paired TOML and JSON schemas encode the same exact map and sampled
crossing. Generated Q32.32 RTL is co-simulated at a deliberately enrolled
grid-exact operating point: `b=0`, `omega=8`, and `dt=1/64`, so the compiler’s
transcendental lookup receives `omega*dt=0.125` exactly. Across varied input,
the full event vector matches and both state errors remain below the declared
Q32.32 envelope. A separate alternating-drive stress trace checks signed
quantization.

This is not a claim that the lookup-table RTL reproduces arbitrary default
parameters exactly. The depth-4 SymbiYosys job proves only that asynchronous
reset clears the public spike flag. It does not assert event silence,
equation equivalence, synthesis, timing closure, resource use, or PPA. The
model therefore remains at H1.

## Scope boundary

- The scalar pipeline accepts real piecewise-constant input. Complex pulse
  coefficients allowed by the paper are outside this API.
- Sampled threshold detection does not localize continuous crossing time.
- The exact linear flow is scientific fidelity; fixed-point transcendental
  lookup error is separately bounded by co-simulation.
- No globally monotone firing-rate/current relationship is claimed.

See [source and runtime fidelity](../../validation/resonate_and_fire_source_fidelity.md)
for the executable evidence matrix and reproduction commands.

## Reference

E. M. Izhikevich, “Resonate-and-fire neurons,” *Neural Networks*, vol. 14,
nos. 6–7, pp. 883–894, 2001.
