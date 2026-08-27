# LarterBreakspearNeuron

`LarterBreakspearNeuron` implements the three-state cortical neural mass used by
Breakspear, Terry, and Friston (2003), DOI
[`10.1088/0954-898X/14/4/305`](https://doi.org/10.1088/0954-898X/14/4/305).
The maintained parameter profile and equation disambiguation follow the
Larter–Breakspear implementation in The Virtual Brain.

The states are mean excitatory voltage `v`, the open potassium-channel fraction
`w`, and mean inhibitory voltage `z`. Calcium, sodium, potassium, excitatory
firing-rate, and inhibitory firing-rate gates use the source hyperbolic-tangent
activation functions. `z` feeds back into the voltage balance through
`-a_ie*z*QZ(z)`; removing that term changes the model identity.

`step(coupling)` accepts an external excitatory population firing rate and
returns the continuous voltage. It does not emit a binary spike and has no
threshold reset. The implementation uses simultaneous fixed-step classical RK4
at the default `dt=0.01`; this is a declared repository numerical
specialization rather than an author-prescribed solver.

```python
from sc_neurocore.neurons.models import LarterBreakspearNeuron

mass = LarterBreakspearNeuron()
voltage = mass.step(coupling=0.0)
state = mass.v, mass.w, mass.z
```

Python, production Rust/PyO3, Rust safety, Go, Julia, and executable Mojo carry
the same equations and source defaults. One-step parity is checked within
`2e-12`; a frozen 512-step mixed-drive receipt binds the full three-state
trajectory. The source-hashed benchmark records real timings for all five
runtime lanes without making a comparative production-speed claim.

The previous project recurrence is not deleted. It remains separately available
as [`SCDecoupledAdaptationIonMassNeuron`](sc_decoupled_adaptation_ion_mass.md),
without Larter–Breakspear attribution and without incrementing the
155-literature-model catalogue.

No RTL, formal-equivalence, synthesis, timing, PPA, device, board, or silicon
claim is made. The nonlinear population-rate and RK4 contract is outside the
faithful expressiveness of the currently enrolled scalar RTL schema path.
