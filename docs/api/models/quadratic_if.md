# QuadraticIFNeuron

**Module:** `sc_neurocore.neurons.models.quadratic_if`
**Reference:** Ermentrout & Kopell 1986, Latham et al. 2000
**Family:** Integrate-and-fire (Type-I canonical)
**State variables:** `v`

## Equations

$$\frac{dV}{dt} = V^2 + I$$

Reset: $V \geq V_{peak} \Rightarrow V \leftarrow V_{reset}$.

## Behaviour

- **Saddle-node bifurcation at I=0:** I<0 → stable fixed point. I>0 → periodic spiking.
- **Type-I excitability:** Firing rate rises continuously from zero at I=0⁺.
- **Sub-linear f–I:** Rate increases sub-linearly with current (modified sqrt scaling).
- **Constant ISI:** Deterministic, CV(ISI) < 0.02 at steady state.

## Test Coverage — 23 tests

Bifurcation (4), f–I (2), ISI (2), edge cases (4), isolation (5), network (2), analysis (2), dt stability (3).
