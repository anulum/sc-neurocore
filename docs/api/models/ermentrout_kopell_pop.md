<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Montbrió–Pazó–Roxin population-model contract -->

# Montbrió–Pazó–Roxin exact QIF-network mean field

- **Compatibility class:** `ErmentroutKopellPopulation`
- **Module:** `sc_neurocore.neurons.models.ermentrout_kopell_pop`
- **Source:** Montbrió, Pazó, and Roxin (2015), equations (12a–b)
- **DOI:** [10.1103/PhysRevX.5.021028](https://doi.org/10.1103/PhysRevX.5.021028)

The public class name predates the present provenance repair. Its maintained
dynamics are the exact thermodynamic-limit firing-rate equations of
Montbrió, Pazó, and Roxin, not the Ermentrout–Kopell 1986 single-cell theta
equation. One instance represents an entire QIF-network population and returns
the continuous firing rate (r), not a binary spike.

## Equations

The paper prints dimensionless equations (12a–b) in source rate \(R\) and
source time \(t'\):

\[
\frac{dR}{dt'}=\frac{\Delta}{\pi}+2Rv,
\qquad
\frac{dv}{dt'}=v^2+\bar\eta+JR+I(t')-(\pi R)^2.
\]

SC-NeuroCore restores a physical population rate and time scale using
\(R=\tau r\) and \(t'=t/\tau\). Because
\(dR/dt'=\tau^2\,dr/dt\) and \(dv/dt'=\tau\,dv/dt\), the maintained flow is

\[
\frac{dr}{dt}=\frac{\Delta}{\pi\tau^2}+\frac{2rv}{\tau},
\]

\[
\frac{dv}{dt}=\frac{v^2+\bar\eta+I(t)+J\tau r-(\pi\tau r)^2}{\tau}.
\]

Here `r` is population firing rate, `v` is mean membrane potential,
`Delta` is the half-width of the Lorentzian excitability distribution,
`eta_bar` is its centre, and `J` is recurrent coupling. Both Euler
candidates are evaluated from the same pre-update state, validated, and
committed together. The τ-expanded form is an exact change of variables, not
the equation form printed in the paper and not a numerical approximation.

## Source and solver boundary

The paper defines continuous ordinary differential equations. The maintained
`dt=0.01` simultaneous explicit-Euler step is a deterministic implementation
contract, not a solver mandated by the publication. The enrolled sinusoidal
drive exercises the recurrence reproducibly; it is not a reproduction
of a published figure.

The exact mean-field derivation assumes an infinite, all-to-all QIF network
with a Lorentzian excitability distribution. Finite-size fluctuations,
transmission delay, adaptation, stochastic input generation, and sparse
connectivity are outside this two-state kernel.

## Parameters and state

| Name | Default | Constraint | Meaning |
|---|---:|---|---|
| `r` | `0.1` | finite, `>= 0` | population firing rate |
| `v` | `-2.0` | finite | population mean membrane potential |
| `tau` | `1.0` | finite, `> 0` | membrane time scale |
| `delta` | `1.0` | finite, `>= 0` | Lorentzian half-width (Delta) |
| `eta_bar` | `-5.0` | finite | Lorentzian centre (`eta_bar`) |
| `j` | `15.0` | finite | recurrent coupling (J) |
| `dt` | `0.01` | finite, `> 0` | explicit-Euler step |

## Scalar and batch use

```python
import numpy as np

from sc_neurocore.neurons.models.ermentrout_kopell_pop import (
    ErmentroutKopellPopulation,
)

unit = ErmentroutKopellPopulation()
rate = unit.step(ext_input=1.5)

index = np.arange(128, dtype=np.float64)
drive = 1.5 + 0.5 * np.sin(index * 0.037)
result = unit.simulate(drive, backend="auto")

assert result["r"].shape == (128,)
assert result["v"].shape == (128,)
assert unit.v == result["v_final"]
```

The batch returns both complete post-update traces and two final-state
receipts. An explicitly requested unavailable runtime raises; it does not
silently substitute Python. A model instance mutates only after the complete
native result passes shared shape, finiteness, non-negative-rate, and final-
state consistency checks. Empty batches preserve both states.

## Executable runtimes

| Runtime | Maintained surface | Enrolled contract |
|---|---|---|
| Python | scalar reference and atomic batch | source reference |
| Rust engine | modular PyO3 scalar and batch | two traces/two final states within `1e-12` |
| Rust safety | independently compiled scalar module | equation wiring, validation, atomic mutation, reset |
| Julia | JuliaCall batch | two traces/two final states within `1e-12` |
| Go | C-shared ABI | two traces/two final states within `1e-12` |
| Mojo | exported shared-library C ABI | two traces/two final states within `1e-10` |

## Benchmark evidence

The committed `benchmarks/results/bench_ermentrout_kopell_pop.json` records
five 50,000-step samples per public runtime after a 1,000-step warm-up. It
binds the exact loaded native binaries, raw timing samples, affinity, host
load, complete-trajectory digest, and parity deltas. These timings are local,
single-logical-CPU, non-exclusive regression evidence—not a production,
cross-host, hardware, or universal-ranking claim. See
[Benchmark Comparison](../../benchmarks/comparison.md) for the recorded table.

## Reference and validation evidence

The committed 256-step source trace independently re-derives equations
(12a–b). Its canonical interleaved little-endian float64 SHA-256 is
`6bf2d0b8611cba646e5b575ed3253090e1e02955c75c01efac13e603578edae5`.
The cross-runtime contract starts from non-zero states, changes every public
parameter, supplies varied drive for 128 steps, and compares both complete
trajectories and final states. See
[MPR source fidelity](../../validation/ermentrout_kopell_pop_source_fidelity.md)
for the evidence matrix and reproduction commands.

## Fixed-point and hardware boundary

Paired TOML and JSON schemas reproduce the hand model. Generated Q32.32
Verilog compiles and preserves the enrolled 64-step trajectory inside the
declared rate and voltage envelopes. A depth-4 catalogue job proves only that
asynchronous reset clears the generated spike flag and the event-silent
population output remains zero after initialization. This remains H1
co-simulation and bounded-safety evidence: no synthesis, timing closure,
formal equivalence, device resource, or PPA result is claimed.

## Reference

E. Montbrió, D. Pazó, and A. Roxin, “Macroscopic Description for Networks of
Spiking Neurons,” *Physical Review X*, vol. 5, 021028, 2015.
