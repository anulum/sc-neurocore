# NonlinearLIFNeuron

**Module:** `sc_neurocore.neurons.models.nlif`
**Reference:** Touboul & Brette 2008
**Family:** Integrate-and-fire (nonlinear, 2D)
**State variables:** `v` (voltage), `w` (adaptation)

## Equations

$$C \frac{dV}{dt} = a(V-V_r)(V-V_c) - w + I$$
$$\tau_w \frac{dw}{dt} = b(V-V_r) - w$$

Spike: $V \geq V_\theta$, hard reset $V \to V_{reset}$.

Cubic nonlinearity: V above V_crit produces positive feedback (runaway).

## Test Coverage: 13 tests
construction, step, subthreshold, spikes, cubic nonlinearity, w adaptation, rate increase, stability, reset, deterministic, Population, network spikes, spike_count.
