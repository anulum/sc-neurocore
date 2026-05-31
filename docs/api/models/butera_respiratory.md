<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Butera respiratory neuron model documentation -->

# Butera Respiratory Neuron

`ButeraRespiratoryNeuron` models a pre-Bötzinger-complex respiratory bursting
cell with transient sodium, persistent sodium, delayed rectifier potassium, and
leak currents.  The public state is the membrane voltage `v`, potassium gate
`n`, and persistent-sodium inactivation gate `h_nap`.

## Dynamical Contract

The maintained Python reference now advances the coupled three-state conductance
ODE with candidate-first fourth-order Runge-Kutta integration.  The Rust engine,
Go service, Julia mirror, and standalone Rust safety surface use the same RK4
state contract.

At each step the implementation evaluates:

```text
I_Na   = g_Na  m_Na(v)^3 (1 - n)      (v - E_Na)
I_NaP  = g_NaP m_NaP(v) h_NaP         (v - E_Na)
I_K    = g_K   n^4                    (v - E_K)
I_L    = g_L                          (v - E_L)
dv/dt  = -I_Na - I_NaP - I_K - I_L + I_ext
dn/dt  = (n_inf(v) - n) / tau_n(v)
dh/dt  = (h_inf(v) - h) / tau_h(v)
```

Intermediate RK4 stages are finite-checked and projected into the documented
physiological envelope before rate evaluation:

```text
-200 mV <= v <= 100 mV
0 <= n <= 1
0 <= h_nap <= 1
```

The final RK4 candidate is accepted only if it is finite, then clamped into the
same envelope.  This preserves the bounded-state behaviour expected by existing
network simulations while removing the raw forward-Euler drift that previously
made high-current runs integration-order dependent.

## Fail-Closed Semantics

The Python reference raises `ValueError` or `FloatingPointError` before state
mutation when parameters, current, runtime state, derivatives, or RK4 candidates
are non-finite or outside the accepted physical contract.  The Go, Julia, and
Rust safety mirrors expose non-throwing sentinel/error paths for the same
invalid domains and preserve the pre-step state.

Spike emission remains a continuous threshold-crossing contract: a step returns
`1` only when the accepted voltage crosses `v_threshold` from below, otherwise
it returns `0`.

## Verification Evidence

Validated on 2026-05-31:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_butera_respiratory.py -q
go test ./services -run ButeraRespiratory -count=1
cargo test --manifest-path engine/Cargo.toml butera_ --lib
rustc --test src/sc_neurocore/accel/rust/safety/butera_respiratory.rs -o /tmp/sc_neurocore_butera_safety_test && /tmp/sc_neurocore_butera_safety_test
julia --startup-file=no -e 'include("src/sc_neurocore/accel/julia/neurons/butera_respiratory.jl"); using .ButeraRespiratoryAccel; trace, spikes = simulate(100; I_ext=50.0); n_bad = ButeraRespiratoryNeuronState(); n_bad.v = NaN; if length(trace) != 100 || !(spikes isa Integer) || step!(n_bad, 50.0) != -1; error("Butera Julia contract failed"); end'
```

The dedicated Python module file reports `26 passed`.  The Rust engine reports
six Butera-focused tests passing, including the invalid-state preservation path.

## Benchmark Evidence

Fresh local artefacts were generated on 2026-05-31 after the RK4 hardening:

| Surface | Artefact | Result |
|---|---|---:|
| Python reference | `benchmarks/results/local_i5_11600k_python_2026-05-31_butera_respiratory.json` | 34,311 steps/s at `I_ext=50.0` over 50,000 measured steps |
| Rust engine Criterion | `benchmarks/results/local_i5_11600k_criterion_2026-05-31_butera_respiratory.json` | 341.62 µs per 1,000 RK4 steps |

The Rust benchmark row in `docs/api/rust-benchmarks.md` uses the Criterion median
from the saved artefact.
