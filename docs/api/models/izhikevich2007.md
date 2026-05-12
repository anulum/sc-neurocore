<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore model documentation -->

# Izhikevich2007Neuron

**Module:** `sc_neurocore.neurons.models.izhikevich2007`
**Reference:** Izhikevich 2007, *Dynamical Systems in Neuroscience*
**Family:** Biophysical quadratic integrate-and-fire
**State variables:** `v` (mV), `u` (pA)

## Equations

The model follows the NeuroML 2 `izhikevich2007Cell` parameterisation:

$$C \frac{dv}{dt} = k(v - v_r)(v - v_t) - u + I$$

$$\frac{du}{dt} = a(b(v - v_r) - u)$$

Spike and reset:

$$v \geq v_{peak} \Rightarrow v \leftarrow c,\quad u \leftarrow u + d$$

Units use the NeuroML base convention used by the importer: `C` in pF,
`k` in nS/mV, voltages in mV, `a` in 1/ms, `b` in nS, and current terms in pA.

## Parameters

| Parameter | Default | Unit | Description |
|-----------|--------:|------|-------------|
| `C` | 100.0 | pF | Membrane capacitance |
| `k` | 0.7 | nS/mV | Quadratic gain |
| `vr` | -60.0 | mV | Resting membrane voltage |
| `vt` | -40.0 | mV | Instantaneous threshold voltage |
| `vpeak` | 35.0 | mV | Spike detection voltage |
| `a` | 0.03 | 1/ms | Recovery time-scale |
| `b` | -2.0 | nS | Recovery-voltage coupling |
| `c` | -50.0 | mV | Reset voltage |
| `d` | 100.0 | pA | Recovery reset increment |
| `v0` | `vr` | mV | Initial membrane voltage |
| `dt` | 0.1 | ms | Integration step |
| `integrator` | `rk4` | - | `rk4` or `euler` |

## NeuroML Import

`<izhikevichCell>` remains mapped to `SCIzhikevichNeuron`, the dimensionless
2003 formulation. `<izhikevich2007Cell>` maps to `Izhikevich2007Neuron`, so
the biophysical 2007 parameters are preserved instead of being converted into
the 2003 parameter set.

## API

::: sc_neurocore.neurons.models.izhikevich2007.Izhikevich2007Neuron
