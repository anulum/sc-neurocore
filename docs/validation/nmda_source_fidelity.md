# NMDA source-fidelity validation

The validation boundary separates two identities:

- `NMDANeuron`: Wang 1999 pyramidal LIF plus two-stage NMDA autapse,
  specialized to midpoint RK2 at `0.05 ms` and sampled threshold events.
- `SCWBNMDAMagnesiumBlockNeuron`: the preserved historical project recurrence,
  without whole-model publication attribution.

The source gate binds the first identity to Wang (1999) Eqs. 1, 2, 4, and 5
and the Jahr–Stevens magnesium-block factor. The independent Python oracle
reimplements the derivatives and midpoint step without calling production
helpers. A 512-step mixed-drive receipt covers event increments, NMDA
saturation, refractory evolution, and complete dynamic state. The retained
identity has its own independently encoded receipt and frozen historical
anchor.

Executed parity covers Python, production Rust/PyO3, standalone safety Rust,
Go, Julia, and Mojo. The one-step complete-state tolerance is `2e-12`; the
source and retained receipts additionally pin event counts, final states, and
little-endian binary64 trace hashes. Invalid non-finite input is checked as an
atomic failure on the stateful surfaces.

The local benchmark executes 20,000 steps for both identities in all five
runtime families and hashes every implementation source. Its timings are
loaded-host regression observations only, not production speed claims.

The source-default hardware lane is a signed-Q16.16 midpoint-RK2 recurrence
with a 5 mV linearly interpolated magnesium-block LUT. A separate integer
oracle is bit-exact to RTL for all 512 receipt steps. Against the binary64
source it preserves all four event indices, with maximum absolute errors below
`0.012 mV` (`v`), `0.0004` (`x_nmda`), `0.0006` (`s_nmda`), `0.0024`
(`ca`), and `0.00013 ms` (refractory time). Yosys synthesis and a depth-4 CVC5
bounded reset/state-safety job are enrolled independently.

The retained SC identity has a separate signed-Q32.32 50-cycle FSM. It is
bit-exact to its integer oracle on 64 quiet and 32 driven macro-steps. The
`I=5` vector preserves events at indices `6, 12, 18, 24, 30`; declared state
errors remain below `4.6 mV`, `0.037`, `0.016`, and `2e-9` for `v`, `h`, `n`,
and `s_nmda`. Yosys coarse synthesis and a separate depth-4 CVC5
handshake/output safety job are enrolled.

Excluded claims: full Wang network behavior, AMPA/GABA population dynamics,
interpolated spike timing, exact Figure 3 rate reproduction, configurable-
parameter RTL, binary64 formal equivalence, technology gate mapping for the SC
profile, timing, PPA, device deployment, and silicon validation.
