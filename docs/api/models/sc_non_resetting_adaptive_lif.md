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

## Evidence boundary

Python, the modular Rust engine and PyO3 batch surface, independent Rust safety,
Julia, Go, and Mojo preserve the complete configured trajectory. Rust, Julia,
and Go are byte-identical to Python over the committed 200,000-step benchmark;
Mojo remains within `2.92e-13`, with the same 577 events.

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
