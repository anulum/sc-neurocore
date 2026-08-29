# IbarzTanakaMapNeuron

`IbarzTanakaMapNeuron` is the compatibility identity for the four-branch
fast/slow map introduced by Shilnikov and Rulkov (2004), using the parameter
profile analysed by Ibarz, Tanaka, Sanjuan, and Aihara (2007). It is a
simultaneous discrete map, not the rational/linear hybrid formerly stored under
this class name.

**Python:** `sc_neurocore.neurons.models.ibarz_tanaka_map.IbarzTanakaMapNeuron`

**Rust engine:** `engine/src/neurons/ibarz_tanaka_map.rs::IbarzTanakaMapNeuron`

**Equation source:** A. L. Shilnikov and N. F. Rulkov, *Subthreshold
oscillations in a map-based neuron model*, Physics Letters A 328, 177–184
(2004),
[DOI 10.1016/j.physleta.2004.05.062](https://doi.org/10.1016/j.physleta.2004.05.062).

**Analysis profile:** B. Ibarz, G. Tanaka, M. A. F. Sanjuan, and K. Aihara,
*Sensitivity versus resonance in two-dimensional spiking-bursting neuron
models*, Physical Review E 75, 041902 (2007),
[DOI 10.1103/PhysRevE.75.041902](https://doi.org/10.1103/PhysRevE.75.041902).

## Source recurrence

For the analysis paper's external input (I_v), let (h_n=I_v+u_n). Its
Equations 2–3 restate the Shilnikov–Rulkov recurrence:

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
e_n=\mathbf{1}[v_n>0\;\land\;v_n\ge 1+I_v+u_n].
\]

This is a level decision on the pre-step state and produces the source's fixed
`v_next = -1` return. The `v_n>0` term preserves the piecewise branch order: a
low value of the upper guard cannot override the earlier `v_n<=0` parabolic
branch. There is no separate threshold or configurable reset; the public
`current` argument carries the source `beta`/analysis `I_v` input.

## Defaults

| Field | Default | Meaning |
|---|---:|---|
| `v` | `-1.0` | Initial fast state |
| `u` | `-0.1` | Initial slow state, matching the paper's illustrated map placement |
| `alpha` | `1.0` | Fast-map geometry parameter |
| `mu` | `0.001` | Positive slow timescale |
| `sigma` | `0.1` | Slow-nullcline offset |

Construction, `step()`, and `simulate()` reject non-finite state, parameters,
or input. `alpha` and `mu` must be positive. Python, production Rust, safety
Rust, Julia, Go, and Mojo validate a complete batch before committing caller
state or output; negative C-ABI statuses and malformed result packets fail
closed.

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
expressions. The paired-source receipt
`src/sc_neurocore/neurons/reference_receipts/ibarz_tanaka_shilnikov_rulkov.json`
binds every post-step `(v,u)` pair and reset decision over 1,000 iterations:
state digest `9fef084b…03cb`, event digest `03426e93…68f`, nine events, first
event index 394, and the exact final state.

## Compiled parity

Python, the Rust engine, the independent Rust safety crate, Julia, Go, and Mojo
implement the same five-float state/parameter order. Across the committed
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
`I=0.2`, median of 21 calls, on logical CPU 2 of an Intel i5-11600K. The
process was affinity-pinned, but the host had no kernel-isolated CPU set and
reported high concurrent load; these are local regression timings, not portable
latency claims.

| Backend | Median call | Speed-up vs Python | Maximum trace difference | Events |
|---|---:|---:|---:|---:|
| Rust | 0.083427 ms | 4.34× | `0` | 33 |
| Mojo | 0.162366 ms | 2.23× | `6.883e-15` | 33 |
| Go | 0.182359 ms | 1.99× | `0` | 33 |
| Julia | 0.226963 ms | 1.60× | `0` | 33 |
| Python | 0.362416 ms | 1.00× | `0` | 33 |

The artefact includes runtime versions, affinity, governor, load, final-state
parity, complete fast-trace and output-packet SHA-256 digests, and hashes for 14
maintained implementation, ABI, schema, descriptor, and receipt sources. Its
evidence gate is `ibarz-tanaka-map-five-backend-controlled-regression`.

## Schema, RTL, and formal boundary

The paired TOML and JSON schemas reproduce the hand recurrence exactly. A
4-step `I=-0.5` protocol covers the parabolic and constant branches; a separate
30-step `I=0.2` protocol covers 23 parabolic, 4 plateau, and 3 reset branches.
Together they visit every source branch. Hand, TOML, and JSON trajectories are
identical, and generated Q16.16 RTL preserves both complete event vectors with
maximum errors below `0.003` for `v` and `0.0001` for `u`.

Q16.16 is required because `mu=0.001` rounds to zero in Q8.8. The S5/H2
descriptor therefore emits a Q16.16 catalogue core. The depth-4
SymbiYosys/Z3 job `sc_ibarz_tanaka_rulkov_map.sby` proves exact public event
equivalence to the ordered fourth-branch guard and proves that every event
commits `v=-1`. Coarse synthesis is tracked in
`hdl/reports/yosys_ibarz_tanaka_rulkov_map_q1616_2026-08-29.json` (33 cells,
zero residual processes). Timing, PPA, device/board, physical silicon, and
universal Python-to-RTL equivalence remain unclaimed.

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
