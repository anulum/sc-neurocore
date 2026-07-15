<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Jansen–Rit neural-mass model contract -->

# Jansen–Rit cortical-column neural mass

- **Class:** `JansenRitUnit`
- **Module:** `sc_neurocore.neurons.models.jansen_rit`
- **Source:** Jansen and Rit (1995), equation (6)
- **DOI:** [10.1007/BF00199471](https://doi.org/10.1007/BF00199471)

`JansenRitUnit` advances the six first-order states of the three-population
Jansen–Rit cortical-column neural mass. It returns the continuous pyramidal
potential difference `y1 - y2` in millivolts, not a binary spike.

## Equations

The population firing-rate response is

\[
S(v)=\frac{2e_0}{1+\exp[r(v_0-v)]}.
\]

With \(C_1=C\), \(C_2=0.8C\), and \(C_3=C_4=0.25C\), the maintained
first-order form of equation (6) is

\[
\dot y_0=y_3,
\qquad
\dot y_3=AaS(y_1-y_2)-2ay_3-a^2y_0,
\]

\[
\dot y_1=y_4,
\qquad
\dot y_4=Aa\left[p+C_2S(C_1y_0)\right]-2ay_4-a^2y_1,
\]

\[
\dot y_2=y_5,
\qquad
\dot y_5=BbC_4S(C_3y_0)-2by_5-b^2y_2.
\]

The \(C_2\) gain multiplies the excitatory sigmoid output; it is not part of
the sigmoid argument. All six explicit-Euler candidates are evaluated from the
same pre-update state, validated, and committed together. The returned EEG
proxy is the post-update value \(y_1-y_2\).

## Source and solver boundary

The paper defines continuous equations and does not prescribe explicit Euler.
The maintained default `dt=0.0001 s` follows the pinned Brian2 single-column
implementation used to bind the source trace. That example writes `e0=5` as
the complete sigmoid numerator; SC-NeuroCore writes the equivalent
`2 * e0` with `e0=2.5` as the half-maximum rate.

The enrolled trace uses a deterministic sinusoidal external drive to exercise
the recurrence. It is a replay and cross-runtime parity protocol, not a
reproduction of a paper figure or of Brian2's random-input example.

## Parameters and state

| Name | Default | Constraint | Meaning |
|---|---:|---|---|
| `y0`, `y1`, `y2` | `0.0 mV` | finite | postsynaptic-potential states |
| `y3`, `y4`, `y5` | `0.0 mV/s` | finite | corresponding first derivatives |
| `a_exc` | `3.25 mV` | finite, `> 0` | excitatory gain \(A\) |
| `b_exc` | `22.0 mV` | finite, `> 0` | inhibitory gain \(B\) |
| `a_rate` | `100.0 s⁻¹` | finite, `> 0` | excitatory inverse time constant \(a\) |
| `b_rate` | `50.0 s⁻¹` | finite, `> 0` | inhibitory inverse time constant \(b\) |
| `c` | `135.0` | finite, `>= 0` | base connectivity \(C_1\) |
| `e0` | `2.5 s⁻¹` | finite, `> 0` | half the maximum firing rate |
| `v0` | `6.0 mV` | finite | sigmoid midpoint |
| `r` | `0.56 mV⁻¹` | finite, `> 0` | sigmoid slope |
| `dt` | `0.0001 s` | finite, `> 0` | explicit-Euler step |

## Scalar and batch use

```python
import numpy as np

from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit

unit = JansenRitUnit()
eeg = unit.step(p_ext=220.0)

index = np.arange(128, dtype=np.float64)
drive = 220.0 + 80.0 * np.sin(index * 0.037)
result = unit.simulate(drive, backend="auto")

assert result["eeg"].shape == (128,)
assert np.array_equal(result["eeg"], result["y1"] - result["y2"])
assert unit.y5 == result["y5_final"]
```

The batch returns all six post-update state traces, the EEG trace, and six
final-state receipts. An explicitly requested unavailable runtime raises; it
does not silently substitute Python. A model instance mutates only after the
complete result passes shared shape, finiteness, EEG-identity, and final-state
consistency checks. Empty batches preserve every state.

## Executable runtimes

| Runtime | Maintained surface | Enrolled contract |
|---|---|---|
| Python | scalar reference and atomic batch | source reference |
| Rust engine | modular PyO3 scalar and batch | seven traces/six final states within `1e-11` |
| Rust safety | independently compiled scalar module | equation wiring, validation, atomic mutation, reset |
| Julia | JuliaCall batch | seven traces/six final states within `1e-11` |
| Go | C-shared ABI | seven traces/six final states within `1e-11` |
| Mojo | exported shared-library C ABI | seven traces/six final states within `1e-8` |

## Benchmark evidence

The committed `benchmarks/results/bench_jansen_rit.json` records five
50,000-step samples per public runtime after a 1,000-step warm-up. On the
recorded, non-exclusive CPU-11 run, median cost ranged from `69.451 ns/step`
for Rust to `7,732.583 ns/step` for the always-available Python path. The
artefact also binds the exact loaded native binaries, raw samples, affinity,
host load, complete-trajectory digest, and parity deltas. These timings are
local regression evidence, not a production, cross-host, hardware, or
universal-ranking claim. See
[Benchmark Comparison](../../benchmarks/comparison.md) for the complete table
and run conditions.

## Reference and validation evidence

The committed 256-step source trace independently re-derives equation (6),
including the \(C_1/C_2/C_3/C_4\) placement. Its canonical interleaved
little-endian float64 SHA-256 is
`84e9273e381d543a5fb32a510a4c82ea977c32cdd267e88b790296e6e8364933`.
The artefact pins Brian2 commit
`1bfa1a9275bd9672b49f4bf61ffbaf6f7cb55fc9` and the publication DOI.

The cross-runtime contract starts from non-zero states, changes every public
parameter, supplies varied drive for 128 steps, and compares all seven traces
and all six final states. See
[Jansen–Rit source fidelity](../../validation/jansen_rit_source_fidelity.md)
for the evidence matrix and reproduction commands.

## Fixed-point and hardware boundary

Paired TOML and JSON schemas reproduce the hand model. Generated Q32.32
Verilog compiles and preserves the enrolled 64-step trajectory within the
declared potential, derivative, and EEG envelopes. This is H1 co-simulation
evidence only. No synthesis, timing closure, formal equivalence, device
resource, or PPA result is claimed.

## Reference

B. H. Jansen and V. G. Rit, “Electroencephalogram and visual evoked potential
generation in a mathematical model of coupled cortical columns,” *Biological
Cybernetics*, vol. 73, pp. 357–366, 1995. The source-bound implementation is
the pinned Brian2
[`Jansen_Rit_1995_single_column.py`](https://github.com/brian-team/brian2/blob/1bfa1a9275bd9672b49f4bf61ffbaf6f7cb55fc9/examples/frompapers/Jansen_Rit_1995_single_column.py).
