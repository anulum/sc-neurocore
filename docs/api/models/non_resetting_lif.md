# NonResettingLIFNeuron

**Module:** `sc_neurocore.neurons.models.non_resetting_lif`
**Reference:** Kobayashi et al. 2009, Jolivet et al. 2004
**Family:** Integrate-and-fire (non-resetting, adaptive threshold)
**State variables:** `v` (voltage), `theta` (dynamic threshold)

## Equations

$$\tau_m \frac{dV}{dt} = -(V-V_r) + R \cdot I$$
$$\tau_\theta \frac{d\theta}{dt} = -(\theta - \theta_r)$$

Spike: $V \geq \theta$, then $\theta \leftarrow \theta + \Delta_\theta$. **V does NOT reset.**

## Behaviour

- **No voltage reset:** Unlike standard LIF, voltage continues its trajectory.
  Only the threshold jumps up on spike, preventing immediate re-firing.
- **Self-limiting:** Threshold rises with each spike, decays back to theta_rest.
- **aMAT variant:** Related to the MAT family (Kobayashi 2009).

## Test Coverage: 12 tests
construction, step, subthreshold, spikes, no voltage reset, theta increase, theta decay, stability, reset, deterministic, Population, spike_count.
