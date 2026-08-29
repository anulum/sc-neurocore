# ChialvoMapNeuron

`ChialvoMapNeuron` implements the simultaneous two-dimensional discrete map in
Chialvo (1995), [DOI 10.1016/0960-0779(93)E0056-H](https://doi.org/10.1016/0960-0779(93)E0056-H).
It is a map, not an ODE: one `step()` call is one complete recurrence and no
Euler timestep is applied.

## Source recurrence

The paper's Eq. 1 is:

\[
x_{n+1}=x_n^2\exp(y_n-x_n)+k
\]

\[
y_{n+1}=a y_n-b x_n+c
\]

The paper permits \(k\) to be a constant bias or a time-dependent additive
perturbation. The maintained API separates those roles:

\[
x_{n+1}=x_n^2\exp(y_n-x_n)+k+I_n
\]

where `k` is the configured constant and `current` supplies \(I_n\). Both
coordinates use the old \((x_n,y_n)\) state and commit simultaneously.

| Surface | Status |
|---|---|
| `a=0.89`, `b=0.6`, `c=0.28`, `k=0.04` | Source-paper parameter set |
| `x=0`, `y=0` initial state | Maintained default |
| `current` additive input | Time-dependent part of the paper's permitted additive perturbation |
| `x_threshold=1.0` upward crossing | Maintained observation convention; not a paper equation or parameter |
| Exponential argument clipped to `[-500, 500]` | Maintained float64 overflow guard; inactive in the enrolled source regimes |

The event returned by the software API is therefore:

\[
x_n < x_{threshold} \le x_{n+1}.
\]

It observes the map and does not reset either coordinate.

## Python API

```python
from sc_neurocore.neurons.models.chialvo_map import ChialvoMapNeuron

neuron = ChialvoMapNeuron()
event = neuron.step(current=0.0)

trace, events = neuron.simulate(
    n_steps=1_000,
    current=0.0,
    backend="auto",
)

x_trace, y_trace, events = neuron.simulate_complete(
    n_steps=1_000,
    current=0.0,
    backend="auto",
)
```

`trace[t]` is the committed fast coordinate after iteration `t`. `simulate()`
advances the object to the final `(x, y)` state and returns the maintained
upward-crossing count. `reset()` restores only `x` and `y`; configured
parameters are preserved.

`simulate_complete()` is the evidence-oriented surface: it returns aligned
post-step traces for both dynamical coordinates plus the maintained event
count. Every backend validates the complete batch before committing object
state or caller-owned C-ABI output buffers.

Invalid state, parameters, current, and non-finite candidate values fail before
a corrupt candidate is committed.

## Acceleration and dispatch

The model exposes the same checked, complete two-state recurrence through:

- the Rust engine batch function `py_chialvo_map_simulate`;
- the Rust safety kernel in `accel/rust/safety/chialvo_map.rs`;
- the Go service and C shared-library boundary;
- the Julia `ChialvoMapAccel.simulate_trace` function;
- the Mojo C shared-library boundary; and
- the Python reference floor.

Explicit backend requests fail if that backend is not built. `backend="auto"`
consults the committed host-matched benchmark record through
`accel.backend_selection.select_backend_order`, tries the measured order, and
keeps Python as the final floor. It does not silently turn an explicit compiled
request into Python.

Build the local C ABI artefacts with:

```bash
cd src/sc_neurocore/accel/go/neurons/chialvo_map
go build -buildmode=c-shared -o libchialvo.so chialvo_map.go

cd ../../../mojo/neurons
mojo build --emit shared-lib -o libchialvo.so chialvo_map.mojo
```

The Rust batch function is built with the optional engine wheel. Julia is
loaded through `juliacall` from `accel/julia/neurons/chialvo_map.jl`.

## Floating-point parity

`exp` and optimised multiply/add evaluation differ slightly across language
runtimes. Chialvo's recurrence can amplify those differences, so long-trace
bit identity is not claimed.

`tests/test_chialvo_map_backends_backend_parity.py` and
`tests/chialvo_map_backends_support.py` enforce two complementary contracts:

1. 1,000 independently sampled one-step updates over the enrolled state/input
   box agree with Python within `5e-15` for Rust, Julia, and Go and `5e-11` for
   Mojo, with identical events.
2. Over 1,000 iterations at `I=-0.05, 0, 0.01, 0.05, 0.1, 1.0`, all four
   compiled lanes reproduce the Python event counts. The trace envelope is
   `5e-14` for Rust, Julia, and Go and `2e-9` for Mojo.

The declared 1,000-step event counts are `0/26/30/0/0/1` in that current order.

## Recorded benchmark

`benchmarks/results/bench_chialvo_map.json` was produced by the committed
`benchmarks/bench_chialvo_map.py`. The run
used 500,000 iterations, five repeats, and one-logical-CPU affinity on CPU 4.
Load, governor, runtime versions, CPU model, complete two-state and event
digests, source hashes, parity, and final states are stored in the JSON rather
than inferred later.

| Backend | Median call | Speed-up vs Python | Maximum trace difference | Events |
|---|---:|---:|---:|---:|
| Rust | 7.896 ms | 458.07x | `5.195e-12` | 12,935 |
| Julia | 14.446 ms | 250.39x | `1.736e-12` | 12,935 |
| Mojo | 25.116 ms | 144.02x | `6.839e-7` | 12,935 |
| Go | 31.117 ms | 116.24x | `3.542e-12` | 12,935 |
| Python | 3,617.089 ms | 1.00x | `0` | 12,935 |

These values describe that recorded host and workload. They are not portable
latency promises.

## Schema and Q16.16 co-simulation

The paired `chialvo_map.toml` and `chialvo_map.json` schemas use
`method="map"` and reproduce the hand float64 states and events exactly at
`I=-0.05, 0, 0.01, 0.1, 1.0` over 100 iterations.

Generated Q16.16 RTL preserves the corresponding event counts
`0/2/3/0/1`. At the stable `I=-0.05, 0.1, 1.0` points, maximum absolute errors
remain below `0.055` for `x` and `0.093` for `y`. At the oscillatory `I=0` and
`I=0.01` points the exponential LUT phase-shifts four and six event positions,
respectively, while retaining the total event counts. Event timing and full
oscillatory trajectory identity are explicitly outside the claim.

The descriptor reaches science S5 and silicon H2 on that bounded evidence.
The tracked Q8.8 core synthesizes to 117 coarse Yosys cells, including three
asynchronous-reset flip-flops and four multipliers. The generated
`sc_chialvo_map.sby` job passes depth-4 Z3 bounded model checking of exact reset
state and the maintained public upward-crossing event relation. Neither result
is presented as timing, PPA, device, physical-silicon, or universal float64
equivalence evidence.

## Independent reference

`chialvo_map_doi.json` records a 100-iteration, zero-input feature contract.
`tests/test_reference_chialvo_map.py` independently re-derives the simultaneous
source recurrence with `math.exp`; it does not call the hand model or copy
schema-runner output. The protocol records two maintained upward crossings,
the first at iteration 33, plus final/minimum/maximum/mean features for both
coordinates. The companion `reference_receipts/chialvo_1995.json` binds every
post-step `(x,y)` pair and every event over 1,000 iterations: 26 events, first
at zero-based index 32, state digest `ccd95d80…52ff4`, and event digest
`d880d810…51e0`.

## Focused verification

```bash
python -m pytest tests/test_model_chialvo_map_chialvo_network.py tests/test_model_chialvo_map_chialvo_isolation.py tests/test_model_chialvo_map_chialvo_analysis.py -p no:cov -q
python -m pytest tests/test_chialvo_map_backends_backend_parity.py tests/test_chialvo_map_backends_rejects_and_reset.py tests/test_chialvo_map_backends_fail_closed_loaders.py tests/test_chialvo_map_backends_auto_dispatch.py -p no:cov -q
python -m pytest tests/test_cosim_chialvo_map.py -p no:cov -q
python -m pytest tests/test_reference_chialvo_map.py -p no:cov -q
cargo test --manifest-path engine/Cargo.toml --lib chialvo_map
go -C src/sc_neurocore/accel/go test ./services -run Chialvo
cd hdl/formal/catalogue
sby -f sc_chialvo_map.sby
```
