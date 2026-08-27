<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Hodgkin-Huxley source and implementation fidelity

The direct source is Hodgkin and Huxley (1952), “A quantitative description
of membrane current and its application to conduction and excitation in
nerve,” DOI 10.1113/jphysiol.1952.sp004764.

## Source coordinate

The paper measures voltage from the resting potential and chooses
\(C_m=1\), \(\bar g_{Na}=120\), \(\bar g_K=36\), \(g_L=0.3\), and batteries
\(+115,-12,-10.613\) mV. The maintained modern form uses
\(V_\text{modern}=V_\text{paper}-65\) mV, producing the familiar
\(E_{Na}=50\), \(E_K=-77\), and rounded \(E_L=-54.4\) mV values. The
shifted alpha/beta functions are the same coordinate transformation.

The source defines the conductance system and its hand-computed numerical
solution. Repository substep ordering, fixed-step solvers, singularity guards,
finite physical envelope, sampled upward-crossing event, and macro-step API are
explicit implementation and safety specialisations.

## Two maintained numerical profiles

The default production profile advances 100 gate-first explicit-Euler
substeps of 0.01 ms per public one-millisecond macro step. Python, production
Rust/PyO3 and NetworkRunner, standalone safety Rust, Go, Julia, and executable
Mojo preserve this profile. The five measured benchmark lanes produce nine
events at current 20 and compare complete voltage traces and final
\((v,m,h,n)\) state.

The paired TOML/JSON schemas intentionally represent the separate simultaneous
classical-RK4 profile over the same equations and substep schedule. The
independent DOI feature trace re-derives that recurrence. The schema/compiler
profile must not be reported as the default Euler production recurrence.

The default-profile receipt at
src/sc_neurocore/neurons/reference_receipts/hodgkin_huxley_1952.json records
all nine event indices, all four final state values, and a binary digest over
every macro-step state/event tuple. The source-hashed five-runtime packet is
benchmarks/results/bench_hodgkin_huxley_mojo.json.

## Hardware boundary

The committed sc_hodgkin_huxley.v is exactly the current equation compiler's
signed-Q16.16 lowering of the RK4 schema profile. Over 20 macro steps at
current 15, the hand RK4 model and paired schemas are event-exact and
Icarus/VVP keeps the lookup-table RTL within one event. The object passes
Yosys coarse synthesis and a depth-4 bounded public-port reset-safety proof
under the enrolled current/reset protocol.

This is the H2 terminal boundary for the present unit. It is not unrestricted
fixed-point/real-number equivalence, a proof of the lookup-table error bound,
timing closure, PPA, target-device validation, FPGA/board execution, physical
silicon, spatial cable conduction, or reproduction of every original figure.

## Durable verification anchors

- Default runtime and receipt:
  tests/test_bench_hodgkin_huxley_mojo.py
- Python dynamics, integrators, failure atomicity, and pipeline:
  tests/test_hodgkin_huxley_integrator_paths.py and the focused
  tests/test_model_hodgkin_huxley_* files
- Independent RK4 source feature trace:
  tests/test_reference_hodgkin_huxley.py
- Committed RTL identity, co-simulation, and synthesis:
  tests/test_cosim_hodgkin_huxley_catalogue.py
- Formal job:
  hdl/formal/catalogue/sc_hodgkin_huxley.sby
