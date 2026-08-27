# SC WB plus NMDA magnesium-block recurrence

`SCWBNMDAMagnesiumBlockNeuron` preserves the complete behavior formerly
published under `NMDANeuron`: a Wang–Buzsáki-style fast-spiking membrane, the
Jahr–Stevens magnesium-block factor, an input-driven saturating NMDA target,
asymmetric gate relaxation, and a threshold reset.

The Wang–Buzsáki and Jahr–Stevens publications support component equations.
They do not define this combined current-to-gate recurrence. The class is
therefore explicitly an SC-NeuroCore project identity and carries no external
paper DOI.

The state is `v`, `h`, `n`, and `s_nmda`. A `0.5 ms` macro-step updates the
project gate once, then advances the membrane through 50 Euler substeps. The
historical one-step anchor at `current=5` is

```text
event = 0
v = -63.15566378039578
h = 0.6480311943997441
n = 0.237221887163776
s_nmda = 0.025
```

Python, production Rust/PyO3, standalone safety Rust, Go, Julia, and executable
Mojo preserve this recurrence. A separate descriptor, paired schema,
independent project receipt, behavior evidence, and benchmark custody keep it
distinct from the Wang 1999 source identity.

The hardware profile is a signed-Q32.32 FSM. A `start` pulse latches one applied
current, the FSM executes the same 50 Euler substeps over 50 clocks, and
`ready` commits the complete macro-step state and event. Voltage-dependent
rates and magnesium block use linear interpolation between 5 mV LUT samples;
the project current-to-gate transform uses 0.5 nA interpolated samples.
It is bit-exact to an independent integer oracle. Against binary64, the
64-step `I=0` vector is event-exact with errors below `0.08 mV`, `0.004`, and
`0.0012` for `v`, `h`, and `n`; the 32-step `I=5` vector preserves events at
indices `6, 12, 18, 24, 30`, with errors below `4.6 mV`, `0.037`, `0.016`, and
`2e-9` for `v`, `h`, `n`, and `s_nmda`. Yosys coarse synthesis and depth-4
CVC5 handshake/output safety are enrolled.

Configurable-parameter RTL, binary64 formal equivalence, timing, PPA, device,
technology gate mapping, and silicon validation remain outside the claim.
