# McCulloch-Pitts formal neuron

- **Python:** `sc_neurocore.neurons.models.mcculloch_pitts.McCullochPittsNeuron`
- **Rust:** `sc_neurocore_engine.McCullochPittsNeuron` and
  `py_mcculloch_pitts_evaluate_batch`
- **Reference:** McCulloch and Pitts (1943),
  [doi:10.1007/BF02478259](https://doi.org/10.1007/BF02478259)

The maintained model is the all-or-none formal neuron from McCulloch and
Pitts' 1943 logical calculus. It is not a general real-weighted threshold
unit: the source rule counts active excitatory afferents and gives any active
inhibitory afferent an absolute veto.

## Rule and timing boundary

For a positive fixed threshold $\theta$, let $n_E(t-1)$ be the number of
active excitatory afferents and let $h_I(t-1)$ indicate whether any inhibitory
afferent is active. The activity one synaptic delay later is

$$
y(t) =
\begin{cases}
1, & h_I(t-1)=0 \text{ and } n_E(t-1) \geq \theta,\\
0, & \text{otherwise.}
\end{cases}
$$

Activity is binary. The synaptic delay belongs to network scheduling; the cell
does not carry a membrane, accumulator, or hidden delay state. `reset()` only
revalidates the fixed threshold.

With `theta=1`, the uninhibited rule is logical OR over excitatory afferents.
With `theta=2`, two active afferents implement AND. In both cases inhibition
forces zero even at the maximum accepted excitatory count.

## Python API

```python
from sc_neurocore.neurons.models.mcculloch_pitts import McCullochPittsNeuron

logical_and = McCullochPittsNeuron(theta=2)

assert logical_and.step(1) == 0
assert logical_and.step(2) == 1
assert logical_and.step(2, inhibitory_active=True) == 0

events, event_count = logical_and.simulate(
    [0, 1, 2, 3],
    [False, False, False, True],
    backend="auto",
)
assert events.tolist() == [0, 0, 1, 0]
assert event_count == 1
```

`theta` is an integer in `[1, 2**31-1]`; excitatory counts are integers in
`[0, 2**31-1]`; inhibition flags are exact Booleans. Integer-valued Python or
NumPy floats remain accepted at scalar boundaries used by the generic network
runner. Boolean counts, fractional values, non-finite values, out-of-range
values, malformed arrays, and non-Boolean flags fail before dispatch.

`backend` accepts `python`, `rust`, `julia`, `go`, `mojo`, or `auto`.
Explicit unavailable backends fail rather than silently using Python. `auto`
consults the source-hashed same-host benchmark and retains Python as its final
always-available floor. Every native result is revalidated as a contiguous
binary trace with a matching integer event count.

## Maintained execution surfaces

| Surface | Contract | Evidence |
|---|---|---|
| Python hand model | positive threshold, non-negative count, Boolean inhibitor, absolute veto | primary-source truth rows exact |
| Rust engine/PyO3 | same-name stateless class plus varying-input batch | OR, AND, veto, empty and int32 boundaries exact |
| Rust safety | separately compiled rule and atomic batch | 9 focused tests pass |
| Julia | exact integer state-free batch | varying-input trace exact |
| Go | service plus reproducible C-shared ABI | trace/count exact and generated header pinned |
| Mojo | two-pass atomic C ABI | trace/count exact, including zero-row ABI |
| TOML/JSON schema | stateless level threshold | source rows exact without fabricated state |
| Registered/folded RTL | signed Q32.0 comparator datapaths | complete encoded truth vector exact |
| SymbiYosys | port-only depth-4 Z3 reset safety | `PASS` |

The independent reference artifact is
`src/sc_neurocore/neurons/reference_trace_data/mcculloch_pitts_1943_truth_table.json`.
It pins eight source rows and their canonical SHA-256
`2aebd2a5ed6ea8a9409b7452d603441d91e2c8b610da8446668452492ee73db4`.
See [McCulloch-Pitts source-to-silicon fidelity](../../validation/mcculloch_pitts_source_fidelity.md)
for the complete evidence boundary.

## Q32.0 hardware encoding

The public API keeps excitation and inhibition separate. The single signed RTL
input folds them only at the hardware boundary:

- `I_t` in `[0, 2**31-1]` is the active excitatory-afferent count;
- `I_t = -1` is the sole absolute-inhibition sentinel.

Because `theta` is strictly positive, the generated condition `I_t >= theta`
implements both parts of the rule exactly. Other negative values are invalid
public encodings; the generated comparator does not turn the RTL module into a
general negative-weight neuron.

## Benchmark boundary

`benchmarks/bench_model_mcculloch_pitts.py` measures a deterministic
200,000-row count/veto pattern through all five public dispatchers. The
artifact binds the exact event vector and count, workload hashes, all source
hashes, and the loaded Rust/Go/Mojo binary hashes. Its single-logical-CPU
affinity is explicitly non-exclusive: timings are local regression evidence,
not a production speed claim or hardware measurement.

The recorded end-to-end medians (including common Python-side input
validation) are 234.741/306.158/328.712/625.298/821.117 ms for
Rust/Go/Python/Mojo/Julia. Every lane emits 102,273 events with zero mismatch
and exact trace SHA-256
`52a05b62f801b9a9856ccac9f6d79f2821d564239b85fd06d454d1d44e28aee4`.
The CPU-4 run used the `powersave` governor under high host load and does not
claim exclusive isolation.

## Scope and limitations

- The maintained object is the 1943 logical neuron, not a perceptron training
  rule, differentiable activation, negative-weight abstraction, or membrane
  ODE.
- Afferent connectivity, propagation delay, and recurrent scheduling remain
  network responsibilities. No fake state is inserted to simulate them.
- Q32.0 co-simulation proves the enrolled count/sentinel protocol. The bounded
  formal job does not prove unbounded equivalence, synthesis timing, FPGA
  placement, power, or physical-device behavior.
- The implementation is a deterministic research primitive, not a medical or
  safety-certified device.

## Reproduction

```bash
PYTHONPATH=bridge:src:. .venv/bin/python -m pytest -q \
  tests/test_model_mcculloch_pitts_logic.py \
  tests/test_model_mcculloch_pitts_validation.py \
  tests/test_model_mcculloch_pitts_hardware_encoding.py \
  tests/test_model_mcculloch_pitts_batch_dispatch.py \
  tests/test_model_mcculloch_pitts_network.py \
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
