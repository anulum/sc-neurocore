# IbarzTanakaMapNeuron

`IbarzTanakaMapNeuron` implements the four-branch fast/slow map analysed by
Ibarz, Tanaka, Sanjuan, and Aihara (2007). It is a simultaneous discrete map,
not the rational/linear hybrid formerly stored under this class name.

**Python:** `sc_neurocore.neurons.models.ibarz_tanaka_map.IbarzTanakaMapNeuron`

**Rust engine:** `engine/src/neurons/maps.rs::IbarzTanakaMapNeuron`

**Reference:** B. Ibarz, G. Tanaka, M. A. F. Sanjuan, and K. Aihara,
*Sensitivity versus resonance in two-dimensional spiking-bursting neuron
models*, Physical Review E 75, 041902 (2007),
[DOI 10.1103/PhysRevE.75.041902](https://doi.org/10.1103/PhysRevE.75.041902).

## Source recurrence

For the paper's external input (I_v), let (h_n=I_v+u_n). Equations 2–3 give

\[
v_{n+1}=
\begin{cases}
-\alpha^2/4-\alpha+h_n,
    & v_n < -1-\alpha/2,\\
\alpha v_n+(v_n+1)^2+h_n,
    & -1-\alpha/2 \le v_n \le 0,\\
1+h_n,
    & 0 < v_n < 1+h_n,\\
-1,
    & v_n \ge 1+h_n,
\end{cases}
\]

and

\[
u_{n+1}=u_n-\mu(v_n+1-\sigma).
\]

Both candidates use the pre-step `(v, u)` state. `step()` and `simulate()`
therefore commit the two states simultaneously. The Python `current` argument
is (I_v).

## Event convention

An event is execution of the fourth source branch:

\[
e_n=\mathbf{1}[v_n\ge 1+I_v+u_n].
\]

This is a level decision on the pre-step state and produces the source's fixed
`v_next = -1` return. There is no separate threshold, configurable reset, or
`beta` parameter.

## Defaults

| Field | Default | Meaning |
|---|---:|---|
| `v` | `-1.0` | Initial fast state |
| `u` | `-0.1` | Initial slow state, matching the paper's illustrated map placement |
| `alpha` | `1.0` | Fast-map geometry parameter |
| `mu` | `0.001` | Positive slow timescale |
| `sigma` | `0.1` | Slow-nullcline offset |

Construction, `step()`, and `simulate()` reject non-finite state, parameters,
or input. `alpha` and `mu` must be positive. Candidate validation happens
before mutation, so a rejected step preserves both states.

## Python API

```python
from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron

neuron = IbarzTanakaMapNeuron()
trace, events = neuron.simulate(1_000, current=0.2, backend="auto")

assert events == 33
assert neuron.v == -1.017019564883986
assert neuron.u == -0.199138284637132
```

`backend` accepts `"auto"`, `"rust"`, `"julia"`, `"go"`, `"mojo"`, and
`"python"`. The returned float64 trace stores post-step `v`; the instance
retains final `(v, u)`.

## Reproducibility and behaviour evidence

- At `I=0`, 1,000 iterations produce 9 reset events; the first is step 395.
- At `I=0.2`, 1,000 iterations produce 33 events and the little-endian float64
  trace SHA-256 is
  `68000d6955ffcaedffa3a851f70e8f118156312ab224638defb408ae0b3002ed`.
- At `I=1`, 1,000 iterations produce 195 events.
- The committed behaviour sweep records `adapting`, `excitable`, `irregular`,
  `rate-coded`, and `tonic` tags. Those are measured catalogue facets, not
  traits inferred from the paper title.

The DOI-backed `ibarz_tanaka_map_2007_doi` contract independently re-derives
the four branches and slow update without importing the hand model or schema
expressions.

## Compiled parity

Python, the Rust engine, the independent Rust safety crate, Julia, Go, and Mojo
implement the same five-float state/parameter ABI. Across the committed
1,000-step `I=0/0.2/1.0` envelope, Rust, Julia, and Go reproduce every Python
trace bit and final state. Mojo preserves the complete event vector and stays
below the measured `1.5e-8` absolute trace/state bound; at the benchmark's
`I=0.2` point its maximum difference is `6.883e-15`.

The map is numerically sensitive. FMA-level differences can change the branch
sequence over much longer horizons, so SC-NeuroCore does not claim indefinite
Mojo trajectory or event identity. Rust, Julia, and Go remain bit-exact because
their operation ordering matches the Python reference.

## Controlled benchmark

`benchmarks/results/bench_ibarz_tanaka_map.json` records 1,000 iterations at
`I=0.2`, median of 21 calls, on logical CPU 10 of an Intel i5-11600K. The
process was affinity-pinned, but the host had no kernel-isolated CPU set and
reported high concurrent load; these are local regression timings, not portable
latency claims.

| Backend | Median call | Speed-up vs Python | Maximum trace difference | Events |
|---|---:|---:|---:|---:|
| Rust | 0.079827 ms | 11.28× | `0` | 33 |
| Mojo | 0.148605 ms | 6.06× | `6.883e-15` | 33 |
| Go | 0.242630 ms | 3.71× | `0` | 33 |
| Julia | 0.249030 ms | 3.62× | `0` | 33 |
| Python | 0.900612 ms | 1.00× | `0` | 33 |

The artefact includes runtime versions, affinity, governor, load, final states,
and SHA-256 digests for every maintained kernel and ABI source. Its evidence
gate is `ibarz-tanaka-map-five-backend-controlled-regression`.

## Schema, RTL, and formal boundary

The paired TOML and JSON schemas reproduce the hand recurrence exactly. A
30-step `I=0.2` co-simulation covers all four branches: 23 parabolic, 4 plateau,
3 reset, and the initial constant branch. Hand, TOML, and JSON trajectories are
identical. Generated Q16.16 RTL preserves the full reset-event vector with
maximum errors below `0.003` for `v` and `0.0001` for `u`.

Q16.16 is required because `mu=0.001` rounds to zero in Q8.8. The S5/H1
descriptor therefore emits a Q16.16 catalogue core and port-only harness. The
depth-4 SymbiYosys/Z3 job `sc_ibarz_tanaka_rulkov_map.sby` proves the bounded
reset-spike safety property; the 30-step co-simulation remains the behavioural
fidelity evidence.

## Focused verification

- `tests/test_model_ibarz_tanaka_descriptor.py`: published defaults and
  descriptor topology.
- `tests/test_model_ibarz_tanaka_dynamics.py`: source branches, simultaneous
  state, event semantics, protocol counts, and finite operating envelope.
- `tests/test_model_ibarz_tanaka_reproducibility.py`: trace goldens,
  repeated-step parity, and population integration.
- `tests/test_model_ibarz_tanaka_validation.py`: validation, atomic failure,
  request bounds, and reset semantics.
- `tests/test_ibarz_tanaka_backends.py`: five-backend parity, final state,
  zero-step contract, and explicit fail-closed dispatch.
- `tests/test_reference_ibarz_tanaka_map.py`: independent DOI contract.
- `tests/test_cosim_ibarz_tanaka_map.py`: hand/TOML/JSON/Q16.16 branch and
  event-vector evidence.
- `src/sc_neurocore/accel/rust/safety/ibarz_tanaka_map.rs`: independent Rust
  branch and golden-count tests.
