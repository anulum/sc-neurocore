<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->

# Brunel-Wang pyramidal cell

`BrunelWangNeuron` is the excitatory pyramidal-cell membrane specialization
from Brunel and Wang (2001), Methods 2.2–2.3. It combines a leaky
integrate-and-fire membrane with external and recurrent AMPA, voltage-dependent
NMDA, recurrent GABA, an absolute refractory period, and explicit midpoint RK2.

- Python: `sc_neurocore.neurons.models.brunel_wang.BrunelWangNeuron`
- Dispatcher: `sc_neurocore.accel.brunel_wang.simulate_brunel_wang`
- Source: [Brunel and Wang 2001, DOI 10.1023/A:1011204814320](https://doi.org/10.1023/A:1011204814320)

## Maintained equation

For four pre-aggregated, non-negative channel gates,

$$
\frac{dV}{dt}=-\frac{V-V_L}{\tau_m}
+\frac{I_{AMPA,ext}+I_{AMPA,rec}+I_{NMDA}+I_{GABA}}{C_m},
$$

with

$$I_{AMPA}=-g_{AMPA}(V-V_{AMPA})s_{AMPA},$$

$$
I_{NMDA}=-\frac{g_{NMDA}(V-V_{NMDA})s_{NMDA}}
{1+[Mg^{2+}]\exp(-0.062V)/3.57},
$$

and

$$I_{GABA}=-g_{GABA}(V-V_{GABA})s_{GABA}.$$

One public step holds all gates fixed and evaluates the membrane derivative at
the initial voltage and the RK2 midpoint. A candidate at or above threshold
emits one event, resets to `v_reset`, and starts `tau_ref`; refractory steps
clamp the membrane to reset while decrementing the timer.

## Public contract

```python
import numpy as np

from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron

cell = BrunelWangNeuron()
event = cell.step(0.05, 0.17, 0.12, 0.05)

index = np.arange(256, dtype=float)
receipt = cell.simulate(
    0.053 + 0.018 * np.sin(0.071 * index),
    0.17 + 0.05 * np.cos(0.053 * index),
    0.12 + 0.04 * np.sin(0.037 * index + 0.2),
    0.05 + 0.02 * np.cos(0.089 * index),
    backend="python",
)
assert receipt["voltages"].shape == (256,)
assert receipt["refractory"].shape == (256,)
assert receipt["events"].shape == (256,)
```

Invalid configuration, gates, or numerical candidates fail before the dynamic
state commits. `reset()` restores voltage and refractory state while preserving
configuration. Empty batches preserve both dynamic states.

## Source defaults

| Parameter | Pyramidal default | Meaning |
|---|---:|---|
| `v_rest` | `-70 mV` | leak reversal |
| `v_threshold` | `-50 mV` | event threshold |
| `v_reset` | `-55 mV` | post-event reset |
| `C_m` | `0.5 nF` | membrane capacitance |
| `tau_m` | `20 ms` | `C_m/g_m`, with `g_m=25 nS` |
| `tau_ref` | `2 ms` | absolute refractory period |
| `g_ampa_ext` | `2.08 nS` | external AMPA conductance |
| `g_ampa_rec` | `0.104 nS` | recurrent AMPA conductance |
| `g_nmda` | `0.327 nS` | recurrent NMDA conductance |
| `g_gaba` | `1.25 nS` | recurrent GABA conductance |
| `tau_ampa` | `2 ms` | source channel decay metadata |
| `tau_nmda_rise/decay` | `2/100 ms` | source NMDA kinetics metadata |
| `tau_gaba` | `10 ms` | source GABA decay metadata |
| `mg_conc` | `1 mM` | extracellular magnesium |
| `dt` | `0.1 ms` | midpoint-RK2 step |

The `tau_*` synaptic constants are retained as source/network metadata. This
single-cell object receives already-summed gates; it does not pretend to own or
integrate presynaptic spike trains or channel states.

## Execution lanes and documentation

Python, the modular Rust/PyO3 engine, independent Rust safety mirror, Julia, Go
C-shared, and Mojo shared-library implementations expose the same configurable
four-gate batch contract. Native failure never silently substitutes Python.

Public native surfaces carry language-native documentation: Rustdoc on engine
and safety types; GoDoc on service/C ABI; Julia docstrings; Mojo module and ABI
comments; and RTL comments for fixed-point format, latency, reset, and outputs.

## Evidence boundary

- An independent 256-step primary-equation oracle pins the complete voltage,
  refractory, and event trace: ten events at steps 12, 36, 64, 89, 113, 139,
  166, 190, 214, and 240.
- Paired TOML and JSON schemas execute the same seven-edge map contract.
- Five-runtime tests compare configured complete traces and final state. Rust,
  Julia, and Go are bounded by `2e-12`; Mojo by `2e-10`; events are exact.
- The committed single-CPU-affinity benchmark measures 200,000 steps, binds
  maintained sources and loaded native binaries, and is local regression
  evidence only.
- Hand Q16.16 RTL matches its integer oracle at the enrolled trace, preserves
  the event vector, synthesizes in Yosys, and carries a depth-4 bounded safety
  job.

This does not establish the paper's network construction, Poisson input,
synaptic-state kinetics, persistent working-memory activity, reaction latency,
mean-field behavior, timing, PPA, device validation, or formal equivalence to
binary64. The separate `BrunelNetwork` model is retained unchanged and is not
an alias of this cell.
