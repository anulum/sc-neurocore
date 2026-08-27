<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Connor-Stevens source and implementation fidelity

`ConnorStevensNeuron` retains the established catalogue name and the
experimental A-current lineage of Connor and Stevens (1971), DOI
`10.1113/jphysiol.1971.sp009366`. Its exact maintained conductances, batteries,
and analytic gate functions are the standard five-branch parameterization in
Connor, Walter, and McKown (1977), DOI
`10.1016/S0006-3495(77)85598-7`, Appendix A.

## Source identity and numerical specialization

The 1977 appendix defines the modified Hodgkin-Huxley sodium branch, delayed
potassium branch with `g_K=20` and `E_K=-72`, transient A branch with
`g_A=47.7` and `E_A=-75`, the maintained A-gate constants, capacitance 1,
leak conductance 0.3, and adjusted `E_L=-17`. Those values and rate functions
are present on every maintained runtime surface.

The source describes the differential system and numerical simulations. The
repository's candidate-first classical RK4, 100 `dt=0.01` substeps per public
one-millisecond macro step, singularity guards, finite candidate envelope, and
sampled upward `v>=0` event convention are explicit repository numerical and
safety specializations. The model has no artificial reset.

## Runtime evidence

Python, production Rust/PyO3 and NetworkRunner, standalone safety Rust, Go,
Julia, and executable Mojo carry the same six-state recurrence. The source-bound
packet at `benchmarks/results/bench_connor_stevens_mojo.json` executes Python,
production Rust, Julia, Go, and Mojo for 100 macro steps at current 20 over 11
repeats. All five lanes preserve nine events. Rust, Julia, and Go keep the
complete voltage trace and final state within `1e-9` of Python; Mojo stays
within `2e-6` across the transcendental C-ABI path.

The packet is single-logical-CPU but non-isolated loaded-host regression
evidence. It explicitly makes no comparative production-speed or hardware
measurement claim. Safety Rust, Go, Julia, NetworkRunner, and the C ABI retain
independent failure-atomicity or parity tests.

The compact receipt at
`src/sc_neurocore/neurons/reference_receipts/connor_stevens_1977.json` replays
100 macro steps at current 20 and records all nine event indices, the complete
six-variable final state, and a binary digest over every state/event tuple. The
independent feature receipt `connor_stevens_driven_spiking_doi.json` separately
re-derives the 1977 recurrence over a distinct current-100 protocol.

## Hardware boundary

The committed `sc_connor_stevens.v` is exactly the current equation compiler's
signed-Q16.16 lowering. Over 20 macro steps at current 100, the hand model and
schema event counts are exact and Icarus/VVP keeps the committed RTL within one
event. The residual is the declared lookup-table quantization boundary of the
stiff six-state transcendental recurrence. The object passes Yosys coarse
synthesis and `sc_connor_stevens.sby` passes a depth-4 bounded public-port
reset-safety check.

This is the H2 terminal boundary for the present unit. It is not unrestricted
fixed-point/real-number equivalence, a proof of the numerical LUT error bound,
timing closure, PPA, target-device validation, FPGA/board execution, physical
silicon, or biological validation beyond the cited sources and enrolled
protocols.
