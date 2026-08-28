# McKean source-fidelity boundary

The catalogue-counted `McKeanNeuron` is pinned to the space-clamped equations
published by Tonnelier (2003), equations (1.3)-(1.6), following McKean (1970):
`dv/dt=-lambda*v+mu*H(v-a)-w+I` and `dw/dt=b*v`. The executable
specialization declares `H(0)=1`, simultaneous classical RK4, an observational
event on sampled upward crossing of `v=a`, and no reset.

`SCTriangularMcKeanNeuron` separately preserves the former project recurrence:
three continuous voltage branches, `dw/dt=epsilon*(v-gamma*w)`, and an
independent `v_peak` event. It is count-neutral and carries no McKean-paper
attribution.

Both identities execute in Python, Rust, Julia, Go, and Mojo and carry paired
schemas, independent evidence, NetworkRunner dispatch, source/binary-bound local
benchmarking, native-language documentation, signed-Q32.32 co-simulation, and
Yosys coarse synthesis.
The retained SC trace is locked over all 3,000 rows by SHA-256
`993226d5bf608aaf83f14e1e82a6b9df8278ccbc3326089551fd7bf2f19a8fca`.
The RTL jobs prove bounded reset behavior only. They do not establish universal
binary64 equivalence, timing, PPA, or device evidence.

Spatial diffusion, traveling waves, and any network construction are outside
the scalar unit. In particular, the 2,560-cell ring, connectivity footprint,
Poisson drive, persistent bump, distractor resistance, and network statistics
belong to the separately named `SC-COMPTE-WM-NETWORK` modification.
