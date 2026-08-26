# Butera Model 1 source-fidelity validation

The literature-labelled model is bound to Butera, Rinzel, and Smith (1999),
DOI `10.1152/jn.1999.82.1.382`, Model 1 equations 1–7. The decisive repair is
the paper's `C=21 pF` whole-cell capacitance: every intrinsic, tonic, and applied
current participates in one current balance before division by `C`.

The previous implementation omitted that divisor. It is preserved under the
count-neutral `SCUnitCapacitanceRespiratoryNeuron` identity rather than deleted
or left falsely attributed. The source class additionally distinguishes applied
current in pA from the optional tonic conductance/reversal pair.

Validation consists of:

- an independent equation and RK4 implementation that checks source defaults,
  capacitance scaling, tonic current, no-reset event semantics, and failure
  atomicity;
- a frozen 1,024-step mixed-drive state/event receipt with four non-trivial
  threshold crossings;
- native one-step and long-run event parity in Rust engine/safety, Go, Julia,
  and executable Mojo 1.0;
- paired TOML/JSON source schemas and measured behavior evidence;
- a source-hashed 200,000-step five-runtime benchmark with 954 exact events.

The count-neutral SC identity closes separately with public registration,
measured behavior, paired schemas, an independent 1,024-step project-spec
receipt, native Rust safety and Mojo surfaces, one-step five-runtime state
parity, and a source-bound 20,000-step benchmark. Python, Rust, Go, and Julia
record five events while Mojo records four; the committed evidence therefore
claims an explicit one-event cross-`libm` envelope rather than false long-run
bit identity.

RK4 at `0.1 ms` is a declared repository numerical specialization of the
continuous source system. The validation does not claim an author-prescribed
integrator, full respiratory network behavior, biological validation, RTL,
formal equivalence, timing, PPA, board/HIL, device results, or silicon.
