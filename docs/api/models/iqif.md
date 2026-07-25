# Integer Quadratic Integrate-and-Fire (IQIF)

- **Python:** `sc_neurocore.neurons.models.iqif.IntegerQIFNeuron`
- **Rust:** `sc_neurocore_engine.IntegerQIFNeuron` and `py_iqif_simulate`
- **Reference:** Wu et al. (2021),
  [doi:10.1109/AICAS51828.2021.9458572](https://doi.org/10.1109/AICAS51828.2021.9458572)
- **Pinned implementation:**
  `twetto/iq-neuron@a8752eba49dba9ba43a64be74090b91a51044b2f`

The maintained model is the signed-integer soma published with the 2021 IQIF
paper. Despite the historical name, the pinned digital recurrence is
piecewise linear: it replaces a literal quadratic multiplier with two
restoring-force slopes and an arithmetic right shift.

## Recurrence

The branch point is

$$
v_b = \operatorname{trunc}_0
\left(\frac{b v_{\mathrm{threshold}} + a v_{\mathrm{rest}}}{a+b}\right),
$$

where `trunc₀` means integer truncation toward zero, matching C++. The
pre-step restoring force is

$$
f(v) =
\begin{cases}
a(v_{\mathrm{rest}}-v), & v < v_b,\\
b(v-v_{\mathrm{threshold}}), & v \ge v_b.
\end{cases}
$$

`a` and `b` are non-negative Q0.3 numerators. One tick computes

$$
v_c = v + (f(v) \mathbin{\mathtt{>>>}} 3) + I.
$$

If `v_c > v_max`, the neuron emits one event and commits `v_reset`.
Otherwise it commits `max(v_min, v_c)`. Equality with `v_max` does not spike.
All public scalar fields and the applied current must be signed 32-bit
integers; intermediate arithmetic uses signed 64-bit storage.

## Source tutorial defaults

| Field | Default | Meaning |
|---|---:|---|
| `v` | 128 | live membrane state |
| `v_rest` | 128 | rest state and public `reset()` target |
| `v_threshold` | 200 | upper force reference, not the event boundary |
| `v_reset` | 128 | hard-reset target after an event |
| `a` | 1 | lower-branch Q0.3 numerator |
| `b` | 1 | upper-branch Q0.3 numerator |
| `v_max` | 255 | strict upper event boundary |
| `v_min` | 0 | inclusive lower clamp |

These defaults give `v_b = 164`. With constant current 10, the first 15
post-step states are:

```text
138 146 153 159 165 170 176 183 190 198 207 217 229 242 128
```

The last state is the first event/reset. The 400-tick tutorial therefore has
26 events, first one-based event 15, period 15, final state 198, minimum 128,
maximum 242, and mean 179.76.

## Python API

```python
from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron

neuron = IntegerQIFNeuron()
trace, events = neuron.simulate(400, current=10, backend="python")

assert events == 26
assert neuron.v == 198
assert trace[:15].tolist() == [
    138, 146, 153, 159, 165, 170, 176, 183,
    190, 198, 207, 217, 229, 242, 128,
]
```

`backend` accepts `python`, `rust`, `julia`, `go`, `mojo`, or `auto`.
Explicit unavailable backends fail instead of silently falling through.
`auto` consults the source-hashed same-host benchmark and retains Python as
the final always-available floor. A successful batch commits only the returned
final state; invalid input or malformed native output leaves the instance
unchanged.

## Maintained execution surfaces

| Surface | Contract | Evidence |
|---|---|---|
| Python hand model | exact signed-int32 validation, C++ branch division, Q0.3 shift, strict event, lower clamp | 400-state source trace exact |
| Rust engine/PyO3 | full eight-field constructor plus integer batch | state, count, and final value exact |
| Rust safety | independent recurrence compiled directly with `rustc` | 8 focused tests pass |
| Julia | full state/parameter batch | exact trajectory |
| Go | service plus reproducible C-shared ABI | exact trajectory and generated header |
| Mojo | two-pass atomic C ABI | exact trajectory |
| TOML/JSON schema | map semantics with derived branch point | all 400 ticks exact |
| Registered/folded RTL | signed Q32.0 datapaths | all 400 ticks and 26 events exact |
| SymbiYosys | port-only depth-4 Z3 bounded safety | `PASS` |

The source implementation reference lives at
`src/sc_neurocore/neurons/reference_trace_data/iqif_a8752eb_tutorial.json`.
It pins the paper DOI, source commit, source-file hashes, oracle-text hash, and
zero-tolerance features. See
[IQIF source-to-silicon fidelity](../../validation/iqif_source_fidelity.md)
for the complete evidence and reproduction boundary.

## Benchmark boundary

The committed 200,000-step run records 13,333 events, final state 165, and the
same little-endian int64 trajectory SHA-256
`b5c84ffb7167e23d9ba3a1e4290aa93326649bd65087781e491a237ab347a4f4`
for Python, Rust, Julia, Go, and Mojo. The run uses one logical-CPU affinity but
does not claim exclusive CPU isolation, hardware measurement, or a production
speedup.

## Scope and limitations

- The paper's Eq. 2 and the pinned source soma are implemented. Eq. 3 is a
  network-level exponential synaptic-current generator; callers supply the
  resulting integer current to the soma.
- The pinned tutorial fixes optional implementation noise to zero. This class
  does not hide a separate random generator inside the soma.
- Q32.0 RTL preserves this enrolled integer regime. The bounded formal job is
  not a proof of unbounded arithmetic equivalence, synthesis timing, FPGA
  placement, power, or deployed hardware behavior.
- `v_threshold` selects the upper force branch reference. `v_max` is the
  strict event boundary; conflating them changes the model.

## Reproduction

```bash
PYTHONPATH=bridge:src:. .venv/bin/python -m pytest -q \
  tests/test_model_iqif_source_dynamics.py \
  tests/test_model_iqif_validation.py \
  tests/test_model_iqif_batch.py \
  tests/test_iqif_schema_dsl.py \
  tests/test_reference_iqif.py \
  tests/test_iqif_backend_loading.py \
  tests/test_iqif_backends.py \
  tests/test_cosim_iqif.py \
  tests/test_bench_iqif_evidence_and_hashes.py \
  tests/test_bench_iqif_gates_and_rejects.py \
  tests/test_bench_iqif_probes_and_metadata.py

taskset -c 4 env PYTHONPATH=bridge:src:. .venv/bin/python \
  benchmarks/bench_model_iqif.py \
  --json benchmarks/results/local_python_2026-07-14_iqif.json

cd hdl/formal/catalogue
sby -f sc_integerqifneuron.sby
```
