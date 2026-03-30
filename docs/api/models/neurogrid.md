# NeuroGridNeuron

**Module:** `sc_neurocore.neurons.models.neurogrid`
**Reference:** Boahen 2014
**Family:** Hardware (analog neuromorphic, 2-compartment)
**State variables:** `v_s` (soma), `v_d` (dendrite)

## Equations

Dendrite: $\tau_d \frac{dV_d}{dt} = -(V_d-V_r) + I - g_c(V_d-V_s)$

Soma (EIF): $\tau_s \frac{dV_s}{dt} = -(V_s-V_r) + \Delta_T\exp((V_s-V_\theta)/\Delta_T) + g_c(V_d-V_s)$

Spike: $V_s \geq V_{peak}$, reset $V_s \to V_{reset}$.

## Test Coverage: 11 tests
construction, step, subthreshold, spikes, 2 compartments, dendritic integration, stability, reset, deterministic, Population, spike_count.
