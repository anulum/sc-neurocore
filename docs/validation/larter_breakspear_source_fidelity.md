# Larter-Breakspear source-fidelity validation

The literature-labelled model is bound to Breakspear, Terry, and Friston
(2003), DOI `10.1088/0954-898X/14/4/305`, with the maintained TVB source used to
disambiguate the complete parameter profile. Its defining features are the
`QV`/`QZ` population firing-rate functions, NMDA modulation of calcium current,
local-versus-external excitation balance, inhibitory feedback into `dV`, and
inhibitory population drive in `dZ`.

The former implementation omitted those population terms and treated `z` as a
decoupled adaptation state. That recurrence is preserved, not deleted, as the
count-neutral `SCDecoupledAdaptationIonMassNeuron` without paper attribution.

Validation includes an independent equation oracle, simultaneous RK4 checks,
failure atomicity, paired TOML/JSON schemas, separate 512-step source and project
receipts, production Rust/PyO3 registration, one-step Python/Rust/Go/Julia/Mojo
parity, and a source-hashed 20,000-step five-runtime benchmark for both
identities. The continuous state is the stable observable; no spike count is
invented for this population model.

Fixed-step RK4 at `dt=0.01` is an explicit implementation specialization. The
evidence does not claim the authors prescribed that solver, that the scalar mass
reproduces a full network experiment, or that RTL, formal equivalence,
synthesis, timing, PPA, board, device, or silicon evidence exists.
