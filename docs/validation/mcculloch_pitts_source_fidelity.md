# McCulloch-Pitts source-to-silicon fidelity evidence

This page records the primary-source rule, independent truth corpus,
five-language parity, stateless schema execution, Python-to-Verilog
co-simulation, bounded formal job, and source-bound benchmark used to promote
`McCullochPittsNeuron` to the polyglot-complete catalogue.

## Primary source binding

The scientific source is McCulloch and Pitts (1943), *A Logical Calculus of
the Ideas Immanent in Nervous Activity*,
[doi:10.1007/BF02478259](https://doi.org/10.1007/BF02478259). The source rule
provides three relevant single-cell invariants:

1. nervous activity is all-or-none;
2. excitation requires a fixed number of active excitatory afferents within
   one synaptic delay;
3. any active inhibitory afferent prevents excitation absolutely.

The maintained model exposes these invariants directly. A real-valued weighted
sum with negative inhibitory weights is a later abstraction and is not used as
a substitute. The one-synaptic-delay statement defines network scheduling; it
does not justify invented membrane or delay state inside the neuron.

## Independent reference

`tests/test_reference_mcculloch_pitts.py` implements the logical rule without
importing production model code. It canonicalizes the eight primary truth rows
as compact JSON and pins SHA-256:

```text
2aebd2a5ed6ea8a9409b7452d603441d91e2c8b610da8446668452492ee73db4
```

The corpus covers:

| `theta` | Excitatory count | Inhibition | Output |
|---:|---:|:---:|---:|
| 1 | 0 | false | 0 |
| 1 | 1 | false | 1 |
| 1 | 0 | true | 0 |
| 1 | 1 | true | 0 |
| 2 | 0 | false | 0 |
| 2 | 1 | false | 0 |
| 2 | 2 | false | 1 |
| 2 | 2 | true | 0 |

The committed artifact
`src/sc_neurocore/neurons/reference_trace_data/mcculloch_pitts_1943_truth_table.json`
binds the DOI, official paper page, primary reprint, equation statement, schema,
standard reference runner and zero-tolerance features. The hand model, TOML
schema and JSON schema reproduce the enrolled output exactly.

## Executable evidence matrix

| Surface | Executed contract | Result |
|---|---|---|
| Python | strict int32 count/threshold, Boolean inhibition, absolute veto, varying batch | source rows and boundaries exact |
| Rust engine | stateless same-name PyO3 class and full batch function | trace and count exact |
| Rust safety | standalone module compiled with `rustc --test` | 9/9 tests pass |
| Julia | complete integer count/flag batch | trace and count exact |
| Go | validated service and generated C-shared ABI | trace and count exact |
| Mojo | validation pass followed by atomic output pass | trace and count exact |
| TOML/JSON DSL | empty state set and strict level threshold | source rows exact |
| Registered RTL | state-owning timing shell over signed Q32.0 comparator | encoded truth vector exact |
| Folded RTL | combinational signed Q32.0 datapath | encoded truth vector exact |
| SymbiYosys | depth-4 Z3 reset-spike safety | `PASS` |

The batch PyO3 implementation lives in
`engine/src/bindings/mcculloch_pitts.rs` and is registered through the existing
`pyo3_neurons` registry. Relative to the accepted parent, Model 33 leaves
`engine/src/lib.rs` unchanged at 7,116 lines and 212 `#[pyfunction]` entries.
An architecture regression test prevents this binding from returning to the
crate root.

Boundary tests separately prove maximum signed count acceptance, threshold
equality, zero excitation, complete-veto dominance, strict Boolean flags,
zero-row batches, malformed native-output rejection, explicit-backend failure,
and atomic Go/Mojo rejection before destination writes.

## Python-to-Verilog contract

The paired schemas contain no state equations. They encode only

```text
spike = I >= theta
```

on a signed Q32.0 port. Public `encode_hardware_input` maps an uninhibited
non-negative count to itself and maps any active inhibitory afferent to `-1`.
Since `theta >= 1`, the sentinel can never satisfy the threshold. The enrolled
Icarus test drives zero, subthreshold, equality, suprathreshold, maximum-int32
and inhibited rows through both production forms; Python, registered RTL and
folded RTL return the identical binary vector.

The registered module holds only its output timing register. It does not imply
biophysical state. The folded module is purely combinational, allowing a
population engine to own network scheduling explicitly.

## Controlled five-backend benchmark

`benchmarks/bench_model_mcculloch_pitts.py` performs a 1,000-row warm-up and
seven 200,000-row calls through each public dispatcher. Counts cycle from zero
through fifteen, the last count is maximum signed int32, and every eleventh row
activates inhibition. It fails if a backend is missing, an event differs, a
count differs, the process is unpinned without acknowledgement, or the
standalone Rust safety module fails.

The committed artifact records raw samples, exact input and output hashes,
source hashes, loaded Rust/Go/Mojo binary hashes and sizes, runtime versions,
affinity, governor and load averages. The run uses one logical CPU without
claiming exclusive isolation. Its timing fields are local regression evidence,
not a production speed or hardware claim.

All five lanes emit 102,273 events and the identical binary trace SHA-256
`52a05b62f801b9a9856ccac9f6d79f2821d564239b85fd06d454d1d44e28aee4`.
End-to-end public-dispatch medians are 234.741/306.158/328.712/625.298/821.117
ms for Rust/Go/Python/Mojo/Julia, including common Python input validation.
The CPU-4 `powersave` run began at load averages 59.52/52.70/47.26 and makes
no exclusive-isolation or portable ranking claim.

## Descriptor and formal boundary

`McCullochPittsNeuron.toml` records the complete authorship, DOI, parameter and
stateless contracts, five exact backends, reference digest, science S5 truth
evidence and silicon H1 evidence.

The catalogue emitter produces `sc_mccullochpittsneuron.v`, its port-only
formal harness and `sc_mccullochpittsneuron.sby`. The depth-4 BMC proves the
declared reset-spike safety property. It does not prove unbounded equivalence,
absence of overflow for arbitrary invalid external encodings, synthesis
timing, placement, power or physical-device behavior.

## Scope boundary

- The source logical rule is implemented. Perceptron learning, differentiable
  surrogates and arbitrary real weights are separate models.
- Network topology, synaptic propagation and the one-delay scheduler remain
  caller responsibilities.
- `-1` is an explicit transport sentinel, not a claim that every negative
  hardware word has a biological interpretation.
- The model is a deterministic simulation and research RTL primitive, not a
  medical, safety-certified or deployed neuromorphic device.

## Reproduction

```bash
PYTHONPATH=bridge:src:. .venv/bin/python -m pytest -q \
  tests/test_model_mcculloch_pitts.py \
  tests/test_mcculloch_pitts_schema_dsl.py \
  tests/test_reference_mcculloch_pitts.py \
  tests/test_mcculloch_pitts_backend_loading.py \
  tests/test_mcculloch_pitts_backends.py \
  tests/test_cosim_mcculloch_pitts.py \
  tests/test_bench_mcculloch_pitts.py

taskset -c 4 env PYTHONPATH=bridge:src:. .venv/bin/python \
  benchmarks/bench_model_mcculloch_pitts.py \
  --json benchmarks/results/local_python_2026-07-14_mcculloch_pitts.json

cd hdl/formal/catalogue
sby -f sc_mccullochpittsneuron.sby
```
