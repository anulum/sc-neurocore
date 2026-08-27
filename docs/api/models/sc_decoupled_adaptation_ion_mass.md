# SCDecoupledAdaptationIonMassNeuron

`SCDecoupledAdaptationIonMassNeuron` preserves the three-state project
recurrence that was formerly exposed under the Larter–Breakspear name. It is a
count-neutral compatibility identity and makes no external-paper attribution.

The model retains the historical defaults, tanh ion gates, decoupled adaptation
equation `dz/dt=b*(v+0.5-z)`, and fixed-step RK4 behavior. Python, production
Rust/PyO3, Rust safety, Go, Julia, and executable Mojo reproduce the frozen
one-step state and the 512-step mixed-drive project receipt.

Use [`LarterBreakspearNeuron`](larter_breakspear.md) for the literature-bound
cortical neural mass with excitatory and inhibitory firing-rate feedback.

No literature, biological-validation, RTL, formal-equivalence, synthesis,
timing, PPA, device, board, or silicon claim is made for this retained identity.
