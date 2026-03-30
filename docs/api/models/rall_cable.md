# RallCableNeuron

**Module:** `sc_neurocore.neurons.models.rall_cable`
**Reference:** Rall 1962
**Family:** Multi-compartment (passive cable)
**State variables:** `v` (N-element array)

## Equations

$$C \frac{dV_i}{dt} = -g_L(V_i - E_L) + g_a(V_{i-1} - 2V_i + V_{i+1}) + I_i$$

Current injected at distal compartment (N-1). Spike detected at soma (0).

## Behaviour

- **Passive attenuation:** Signal decreases from distal to soma. Stronger coupling
  (g_ratio) reduces attenuation.
- **Fewer compartments → easier spiking:** n_comp=2 fires readily; n_comp=5 at
  default g_ratio=0.5 cannot reach somatic threshold.
- **Population incompatible:** Array-valued `v` breaks Population._sync_voltages.
- **Somatic reset:** Only soma (v[0]) resets on spike; other compartments unaffected.

## Test Coverage — 23 tests

Isolation (5), propagation (3), spiking (4), parameters (8), network incompatibility (1), analysis (2).
