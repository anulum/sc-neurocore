# SC non-resetting adaptive LIF

**Class:** `sc_neurocore.neurons.models.sc_non_resetting_adaptive_lif.SCNonResettingAdaptiveLIFNeuron`
**Source:** SC-NeuroCore project recurrence; no publication attribution

## Identity and recurrence

This class preserves the exact behavior formerly exposed as
`NonResettingLIFNeuron`. It is an SC project model, not Kobayashi MAT(1), a
Jolivet generalized integrate-and-fire model, or Brette's adaptive exponential
integrate-and-fire model.

For constant current during one sample, voltage and threshold relax exactly:

$$
V_{n+1}=V_\infty+(V_n-V_\infty)e^{-\Delta t/\tau_m},
\qquad V_\infty=V_{rest}+R_m I_n,
$$

$$
\theta^-_{n+1}=\theta_{rest}+(\theta_n-\theta_{rest})e^{-\Delta t/\tau_\theta}.
$$

If `V[n+1] >= theta-[n+1]`, an event is emitted and
`theta[n+1] = theta-[n+1] + delta_theta`. Voltage is never reset and no
refractory gate is present.

```python
from sc_neurocore.neurons.models.sc_non_resetting_adaptive_lif import (
    SCNonResettingAdaptiveLIFNeuron,
)

neuron = SCNonResettingAdaptiveLIFNeuron()
events = [neuron.step(20.0) for _ in range(200_000)]
print(sum(events), neuron.v, neuron.theta)
```

## Configuration and failure contracts

The constructor accepts `v`, `theta`, `v_rest`, `theta_rest`, `delta_theta`,
`tau_m`, `tau_theta`, `r_m`, and `dt`. All must be finite; `delta_theta` and
`r_m` are non-negative, while both time constants and `dt` are positive.
Finite voltages have no additional bound and `dt` may exceed either time
constant. Reset validates the complete resting candidate before updating
voltage or threshold, and valid configuration can recover invalid dynamic state.

The native Python batch requires an aligned, contiguous, one-dimensional
`float64` current array; readonly inputs are accepted. Its independent owning
outputs are `float64` voltage and threshold arrays and `int32` events. Invalid
configuration is rejected even for an empty batch. Allocation failures raise
`MemoryError`; refused steps leave both dynamic values unchanged.

Julia's constant-current `simulate(n_steps; current, dt)` also validates current
and timestep for zero samples. Valid empty calls return an empty voltage trace
and zero events; nonempty calls preserve the vector simulation's full trajectory.

Go and Mojo C batches return `1` for a negative count or any null buffer, `2`
for invalid configuration or a refused transition, and `0` on success. A valid
empty batch writes only its initial final state. A late transition refusal
retains the accepted trace prefix and leaves later entries and both final-state
buffers untouched. Mojo uses the scalar system `libm` exponential to preserve
the recurrence over the entire accepted timestep range.

## Evidence boundary

Python, the modular Rust engine and PyO3 batch surface, independent Rust safety,
Julia, Go, and Mojo preserve the complete configured trajectory. Rust, Julia,
and Go are byte-identical to Python over the recorded 200,000-step benchmark;
Mojo remains within `2.92e-13`, with the same 577 events.

The five-runtime benchmark uses five full repetitions at 200,000 steps. Its
record binds the actual source and native-library hashes; timings are local
regression measurements without CPU isolation or production speed claims.

The frozen pre-split 256-step receipt records five events, final state
`[-32.61772042832371, -27.97424372241646]`, and trace SHA-256
`7dd9f76fd1d819bc462460112cfb5906b137935db466bfd60e206f1b4303ae25`.
Paired schemas reproduce the recurrence.

The signed Q32.32 RTL is bit-exact to its integer oracle and preserves the five
software events on the enrolled drive. It passes Yosys synthesis, checked
post-optimization sequence equivalence, and depth-12 CVC5 bounded safety. No
literature-model count, universal equivalence, device timing, PPA, or physical
silicon claim follows from this project compatibility surface.

See [dual-identity source and runtime evidence](../../validation/non_resetting_lif_source_fidelity.md).
