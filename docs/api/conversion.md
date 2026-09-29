# ANN-to-SNN Conversion

Convert trained PyTorch ANNs to rate-coded spiking neural networks.

## Contract

Training and ANN extraction require PyTorch. The exported `ConvertedSNN` runtime
uses NumPy, optionally dispatches replay to configured Rust, Go or Julia, and can run without PyTorch; `convert` and ReLU replacement report a
missing dependency when called without Torch, and resolving `QCFSActivation`
requires Torch.

- `convert(model, calibration_data=None, T=None, percentile=99.9,
  max_working_bytes=268435456)` captures an independent inference graph and
  lowers supported dense operations in actual forward order. Shared modules
  retain each invocation, unused registrations do not define topology, and
  consecutive affine maps are composed without inserting IF nonlinearities.
- Each ReLU invocation, including function and tensor-method forms, receives its
  own calibrated scale. Without calibration, ReLU uses unit scale and ReLU6
  retains its saturation scale of six. ReLU IF neurons start from rest.
- Each QCFS invocation retains its learned threshold and a half-threshold IF
  preload. Mixed ReLU/QCFS networks keep separate scales and preloads. `T=None`
  adopts a common trained QCFS budget, or 16 without QCFS; differing trained
  budgets require an explicit simulation `T`. Matching `T` does not guarantee
  zero conversion loss: validate source/target loss for the declared encoding.
- `ConvertedSNN.run(x)` rate-codes NumPy input with a fixed RNG seed and returns
  final-layer spike counts or signed integrated linear output for one vector or
  a batch. Dense runtime weights must be two-dimensional; convolution execution
  is not part of this runtime profile. `initial_membrane_fraction`
  supplies a global default preload; `layer_membrane_fractions` overrides it for
  individual stages (`0.0` ReLU rest, `0.5` QCFS shift). A linear readout starts
  from zero regardless of these fractions.
- `ConvertedSNN.rates(x)` divides the response by `T` and restores the source
  activation scale. `classify(x)` selects the first greatest response index.
- `QCFSActivation` replaces ReLU during conversion-aware training by clipping
  activations to `[0, theta]` and quantising them to `T + 1` spike-rate levels
  with a straight-through gradient. `T` is an integer in `[1, 2**32 - 1]`, the
  step domain every native counterpart shares. Only interior elements carry a
  gradient, so a saturated infinite or NaN neighbour never turns a batch's
  threshold gradient into NaN; upper saturation contributes one and lower
  saturation zero, also for infinite inputs.
- `replace_relu_with_qcfs(model, T=8, theta=1.0, learn_theta=True)` substitutes
  every `ReLU`/`ReLU6` in a model (recursing through submodules) with a
  `QCFSActivation`, preparing the network for QCFS conversion-aware fine-tuning.

The dense target admits Linear, module/function/method ReLU and ReLU6, QCFS,
inference identity/dropout and batch-preserving module Flatten operations.
Residual arithmetic, unrepresented operators, multiple outputs and
input-dependent Python control flow fail compatibility admission. Source dtype
rounding and affine-composition reduction order remain part of measured
source/target loss. Source graph capture uses a model copy. Tracing, source
calibration and `calibrate_activation_thresholds` restore the Python, NumPy and
PyTorch CPU generators and the generator of every device of the current
accelerator type, also when user code raises; custom user forward code can
still perform other external effects. These sections are serialised within the
process, so concurrent conversions cannot restore one another's states; code
that draws from the global generators concurrently outside a conversion is not
isolated. Accelerator custody was verified on a CUDA device. Source copying, inserted identity/fused matrices and source
calibration buffers have separate metadata checks against the supplied budget.

## Replay and buffer budgets

`ConvertedSNN.replay(frames, initial_state=None, trace=False,
binary_inputs=True, max_working_bytes=268435456, backend="auto")` consumes explicit
`(steps, batch, input_neurons)` frames. Binary drive requires exact zero/one
values; `binary_inputs=False` accepts finite currents in `[0, 1]`. The result
owns its output, final states and optional complete state/event traces.
Supplying previous final states continues the trajectory without modifying
caller buffers. Spiking output counts events in this call; linear output retains
the cumulative readout integral.

The `dense-if-f64-sequential-v1` profile accumulates input columns in ascending
order with separate float64 multiply and add operations. Bias is added every
step. IF events fire at equality, propagate within the same timestep and reset
by subtracting one threshold. Each neuron emits at most one event per step.
The final linear layer integrates signed current without firing or reset.

Constructor snapshots and `replay`, `run`, `rates`, and `classify` accept an
operator-selected `max_working_bytes`; each call defaults to 256 MiB. Invalid
budgets raise `ValueError`; reservations above the budget raise `MemoryError`
before the relevant allocation: coefficient storage is checked before snapshots,
and the complete replay reservation before frame or state copies. The reservation is
`8 * (2P + 2F + 2S + 2O + H + 5M + I)` bytes, where P counts coefficients,
F all frames, S all membranes, O final outputs, H requested state/event traces,
M the largest layer and I one input frame. The bound covers numeric temporaries;
it excludes caller-owned storage, interpreter and allocator overhead, and is
not a process-RSS guarantee. Array-like coercion of Python containers precedes
metadata admission; existing array metadata is inspected without materializing
broadcast views. `run` generates frames in blocks of at most 64 timesteps.
An empty batch performs no timestep iteration, including at `T=2**53`.
Native encoded runs longer than 64 steps additionally reserve
`8 * batch * (sum(layer_widths) + final_width)` for the preceding opaque
result owner retained during the next block. This reservation is admitted
before encoding begins. Explicit replay treats supplied initial-state storage
as caller-owned storage.

Shared registered ReLU/ReLU6 children remain shared when
`replace_relu_with_qcfs` prepares a source for fine-tuning. Every registration
alias points to one QCFS activation and one learned threshold, including aliases
through separate parents. The outer model and each replaced activation retain
their training modes. The root module itself is returned unchanged.

## Python native replay

All four runtime methods accept `backend="auto"`, `"numpy"`, `"rust"`, `"go"`, `"mojo"`, or `"julia"`.
Auto tries `SC_NEUROCORE_IF_RUST_LIB`, then `SC_NEUROCORE_IF_GO_LIB`,
then `SC_NEUROCORE_IF_MOJO_LIB`, then Julia when `SC_NEUROCORE_IF_JULIA_ENABLED=1`, and otherwise selects NumPy.
Without a comparison this order is a configuration preference. To use measured native
ordering, set `SC_NEUROCORE_IF_BENCHMARK` to the complete JSON emitted by
`benchmarks/bench_ann_to_snn_replay.py` in a source checkout, or by the
`sc-neurocore-if-benchmark` command an installed wheel provides. Both bind the source
bytes of the imported `sc_neurocore` package and of the comparison scripts that ran, so
a report captured from byte-identical sources is admitted in either location. Auto verifies
the same CPU, Python/NumPy versions, uninstrumented measurement, current source bytes
and every configured native artifact. All five providers must have the same complete
20-workload response digests; workload medians and the aggregate are recomputed from
positive raw samples. Stale, malformed or incompatible explicit reports raise
`RuntimeError`; a different CPU retains the static order. The measured native order
never displaces the NumPy floor. Explicit backend selection bypasses the report.
This records a local warm public-call ranking, not trained accuracy or energy.
Explicit NumPy always uses the reference runtime.
Explicit native selection requires its corresponding configuration. A configured library that cannot load
or has incompatible entry points raises `RuntimeError`; native errors propagate
without silent fallback. Selection is checked even for empty encoded batches.

The wheel ships the dense IF Rust, Go, Mojo and Julia sources, so the native libraries
can also be built from an installed package: replace `src/sc_neurocore` below with the
installed package directory. Build Go with `-buildvcs=false` there, because an installed
package is not a module checkout.

Build the actual dependency-free Rust ownership ABI from the source checkout:

```bash
cargo build --offline --release \
  --manifest-path src/sc_neurocore/accel/rust/safety/if_native/Cargo.toml \
  --target-dir /your/build/directory
export SC_NEUROCORE_IF_RUST_LIB=/your/build/directory/release/libsc_neurocore_if_replay.so
```

```python
from sc_neurocore.conversion import ConvertedSNN

snn = ConvertedSNN([[[1.0]]], [None], [1.0], T=129)
result = snn.replay([[[1.0]]], trace=True, backend="rust")
counts = snn.run([1.0], backend="rust")
```

Rust replays the same ordered float64 profile, with complete owned state and
event traces. NumPy views retain their native result owner and supplying
library until the last view expires; slicing also retains that lifetime.
Final states and traces occupy independent buffers. Invalid domain, numeric
reservation refusal, and arithmetic overflow map to `ValueError`, `MemoryError`,
and `FloatingPointError`. Libraries are loaded from explicit configuration;
calling the runtime does not build or download them. Go exposes the same ABI-one
result contract through its C shared library. Julia uses managed JuliaCall
entry points; Mojo provides the same C ownership ABI through an explicitly built library.
Normal garbage collection releases an owner after its last view expires.
Forced finalization at interpreter exit is disabled, preserving retained views
during other exit hooks; process teardown reclaims any remaining storage.

## Python Julia replay

Prepare an installed Julia executable and an existing locked Julia project
whose `Project.toml` pins `PythonCall` to `=VERSION`, exactly matching the
installed Python `juliacall` version. Its `Manifest.toml` must resolve that same
version. The adapter checks these files and refuses drift; it does not install
packages or resolve the project.
Conflicting `python -X` overrides for Julia's executable, project, worker count,
signal handling or initialization are refused before JuliaCall imports.
An existing runtime must actually have one default-pool worker and
managed signal handling enabled; changing environment variables after startup
cannot make an incompatible runtime eligible.

```bash
export SC_NEUROCORE_IF_JULIA_ENABLED=1
export PYTHON_JULIACALL_EXE=/absolute/path/to/julia
export PYTHON_JULIACALL_PROJECT=/absolute/path/to/locked/project
export PYTHON_JULIACALL_THREADS=1
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PKG_OFFLINE=true
```

Initialize the configured runtime before importing PyTorch or using conversion
factories. Resolving `ConvertedSNN` itself only loads the NumPy runtime.

```python
from sc_neurocore.conversion.if_julia import load_julia
load_julia()

from sc_neurocore.conversion import ConvertedSNN
snn = ConvertedSNN([[[1.0]]], [None], [1.0], T=129)
result = snn.replay([[[1.0]]], trace=True, backend="julia")
counts = snn.run([1.0], backend="julia")
```

The packaged Julia module uses the same owned replay kernel and ABI-one layouts.
Its borrowed `LayerParameters` entry admits the complete reservation before
copying each layer once. Results stay rooted in Julia until the last Python
view expires; a C-allocated metadata slot identifies each result. Complete
numeric traces are exposed without copying them into another output buffer.
Managed calls reject runtime shutdown and avoid entering Julia after its exit
hook. Raw C callbacks require a registered Julia runtime thread and the same
live-pointer and exactly-once-release contract as the other native providers.

The adapter retains JuliaCall's GIL serialization. JuliaCall documents managed
Python-thread support as experimental and recommends explicit signal handling:
[JuliaCall threading](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).
With this signal setting, Ctrl-C does not raise Python `KeyboardInterrupt`.
The dedicated public integration corpus exercises both installed Julia runtime
families, full state/event parity, continued states, retained slices across
Python and Julia garbage collection, 16 worker calls and process exit refusal.
The direct callback corpus independently exercises native layouts and refusal
slot custody. These checks do not establish support for every host platform.


## Verification

The public conversion files are covered by the scoped NumPy-docstring policy:

- `src/sc_neurocore/conversion/__init__.py`
- `src/sc_neurocore/conversion/ann_to_snn.py`
- `src/sc_neurocore/conversion/qcfs.py`
- `src/sc_neurocore/conversion/calibration.py`
- `src/sc_neurocore/conversion/converted_snn.py`
- `src/sc_neurocore/conversion/if_parameters.py`
- `src/sc_neurocore/conversion/if_replay.py`
- `src/sc_neurocore/conversion/if_encoding.py`
- `src/sc_neurocore/conversion/if_resources.py`
- `src/sc_neurocore/conversion/if_inputs.py`
- `src/sc_neurocore/conversion/if_dispatch.py`
- `src/sc_neurocore/conversion/if_native.py`
- `src/sc_neurocore/conversion/if_native_types.py`
- `src/sc_neurocore/conversion/if_julia.py`
- `src/sc_neurocore/conversion/if_julia_types.py`
- `src/sc_neurocore/conversion/if_julia_configuration.py`
- `src/sc_neurocore/conversion/model_trace.py`
- `src/sc_neurocore/conversion/source_graph.py`
- `src/sc_neurocore/conversion/source_calibration.py`

Focused production tests live in the split `tests/test_conversion_*.py` suites.
They exercise real PyTorch modules, the
threshold-balancing and QCFS conversion routes, `ConvertedSNN.run`,
`ConvertedSNN.classify`, the membrane shift, the ReLU→QCFS substitution helper,
QCFS range and gradient behaviour, and the layer-extraction contract.

Additional `tests/test_conversion_*.py` suites exercise full state/event
trajectories against independent scalar arithmetic, state continuation,
parameter custody, signed readouts, calibration mode/hook/RNG preservation,
actual execution without Torch and exact buffer-budget boundaries. Source graph
tests additionally cover actual forward order, repeated modules, functional
activation calibration, affine composition, mixed-route preloads, source model
and CPU RNG custody, bfloat16 observations and genuine compatibility refusals.
`tests/test_conversion_random_custody.py` converts models whose forward draws
from every global generator, in two threads at once and through a failing
forward, and requires every generator state to be unchanged.

## Julia explicit-frame replay

The native `AnnToSnnAccel` module exposes owned `DenseLayer`, `ConvertedSNN`,
`ReplayResult`, `replay` and zero-based `classify`. Its coefficient, frame, state
and trace vectors use row-major ordering, matching the Python replay profile.
Source graph tracing and stochastic input encoding belong to the frontend.

```julia
include("src/sc_neurocore/accel/julia/conversion/ann_to_snn.jl")
using .AnnToSnnAccel

layer = DenseLayer(1, 1, [1.0], [0.25], 1.0; initial_fraction=0.5)
model = ConvertedSNN([layer]; output_mode=:spikes)
result = replay(model, [0.5, 0.5], (2, 1); trace=true, binary_inputs=false)
continued = replay(model, [0.0], (1, 1); initial_state=result.final_state)
```

The shape tuple is `(steps, batch)`. Bias applies every timestep. The last stage
can instead use `output_mode=:linear` for cumulative signed integration.
`max_working_bytes` defaults to 256 MiB; admission uses the same numeric-buffer
formula as Python. Domain/shape refusals raise `ArgumentError`, finite arithmetic
overflow raises `OverflowError`, and budget refusal raises `OutOfMemoryError`.
Parameters and returned arrays are independently owned; public coefficient
mutations are checked again at replay. Direct Julia calls do not automatically
select a Python backend.

`tests/test_accel_julia_if_replay.py` compares every output, state and event bit
against Python for both installed Julia runtime families, checks actual chunked
continuation and requires inferred concrete replay returns. Julia line coverage
and Python branch coverage describe different measurements.

## Go explicit-frame replay

The `github.com/anulum/sc-neurocore/accel/conversion` package exposes owned
`DenseLayer`, `ConvertedSNN`, `ReplayOptions` and `ReplayResult` values. Constructors
require a numeric byte budget, positive finite thresholds and finite per-layer
preloads. Coefficients are private and copied independently. `Replay` takes flat
row-major frames and explicit step/batch counts; `Classify` returns zero-based
first-maximum labels. The resource formula and response semantics match Python.

```go
layer, err := conversion.NewDenseLayer(1, 1, []float64{1}, []float64{0.25}, 1, 0.5, 256<<20)
if err != nil { return err }
model, err := conversion.NewConvertedSNN([]conversion.DenseLayer{layer}, conversion.Spikes, 256<<20)
if err != nil { return err }
result, err := model.Replay([]float64{0.5, 0.5}, 2, 1, conversion.ReplayOptions{
    Trace: true, BinaryInputs: false, MaxWorkingBytes: 256 << 20,
})
if err != nil { return err }
continued, err := model.Replay([]float64{0}, 1, 1, conversion.ReplayOptions{
    InitialState: result.FinalState, MaxWorkingBytes: 256 << 20,
})
if err != nil { return err }
fmt.Println(continued.Output)
```

Use `conversion.Linear` for cumulative signed readout. Refusals return nil results
and `ErrInvalidInput`, `ErrOverflow` or `ErrResourceLimit`. Separate explicit
float64 rounding follows the [Go floating-point specification](https://go.dev/ref/spec#Floating_point_operators),
so multiply/add fusion cannot discard the profile's required intermediate rounding.
Concurrent replays use independent states and traces.
`NewConvertedSNNFromParameters` admits the whole borrowed coefficient stack
before copying each layer once, without retaining caller storage. The C ABI
uses this constructor to avoid a redundant coefficient snapshot.

Build the Go ownership boundary from the source checkout:

```bash
cd src/sc_neurocore/accel/go
go build -buildmode=c-shared -o /your/build/directory/if-go.so ./conversion/cshared
export SC_NEUROCORE_IF_GO_LIB=/your/build/directory/if-go.so
```

`ConvertedSNN.replay`, `run`, `rates`, and `classify` accept `backend="go"`.
The Go owner retains its result through a `runtime/cgo.Handle` stored in an
opaque C-allocated metadata slot. Numeric result vectors are pinned with
[`runtime.Pinner`](https://pkg.go.dev/runtime#Pinner) while any Python view
retains the owner. The final release unpins the vectors, deletes the handle
and frees the metadata slot. No complete trace copy crosses the boundary;
empty vectors use a null data pointer with zero length. The unsafe C caller
contract still requires live handles, exactly-once free, and unchanged live
input arrays throughout a call. The Python adapter manages these lifetimes.

`tests/test_accel_go_if_abi.py` builds with `GOEXPERIMENT=cgocheck2` and
exercises complete C buffers and refusal-slot custody.
`tests/test_conversion_go_native.py` exercises all public Python methods,
continuation, retained array views and concurrent results. Its native race
profile runs the same corpus in a separate Linux x86-64 loader process with
`LD_PREFER_MAP_32BIT_EXEC=1`; the host-default instrumented library failed
ThreadSanitizer shadow allocation. Both ordinary and race-instrumented
profiles retain strict cgo pointer checks.

`tests/test_accel_go_if_replay.py` compares full output/state/event bits and actual
chunked continuation through exported Go methods. Native package tests exercise
admission, ownership, signed classification and concurrent replay with the race
detector. Go statement coverage is distinct from Python branch coverage.

## Python Mojo replay

Build the shared ownership boundary using Mojo 1.0.0 with contraction disabled:

```bash
mojo build --fp-mode contract=off --diagnose-missing-doc-strings --Werror \
  -I src/sc_neurocore/accel/mojo/kernels --emit shared-lib \
  -o /your/build/directory/if-mojo.so \
  src/sc_neurocore/accel/mojo/kernels/ann_to_snn_native.mojo
export SC_NEUROCORE_IF_MOJO_LIB=/your/build/directory/if-mojo.so
```

`ConvertedSNN.replay`, `run`, `rates`, and `classify` accept `backend="mojo"`.
The boundary admits all descriptor extents and the whole numeric reservation
before copying caller doubles once. Consuming constructors and the shared
`ann_to_snn_compute.mojo` kernel transfer those owners without additional
coefficient/frame copies. The native allocation retains every result vector
until the Python adapter releases its last view; slices retain the same owner.
Empty buffers have a null address and zero length. The process runtime is
initialized before replay or buffer access and retained for process lifetime.

The C request/buffer/release contract targets 64-bit Linux. Raw C callers must
provide complete live aligned arrays unchanged during each call, exclusive
writable destination slots, live result handles and exactly-once release
without outstanding or concurrent borrows. Address checks cannot establish OS
accessibility. Refusals leave destination slots unchanged. The Python adapter
manages result lifetimes and maps domain, resource and arithmetic refusals to
the same exceptions as the other native backends.

`tests/test_accel_mojo_if_abi.py` runs the shared complete C-buffer and refusal
contract. `tests/test_conversion_mojo_native.py` exercises all public runtime
methods, encoded continuation, retained slices and concurrent independent
results. These acceptance tests establish behavior; native quantitative
coverage and comparison benchmarks remain separate gates.

## Mojo explicit-frame replay

`ann_to_snn_parameters.mojo` provides owned `DenseLayer` and `ConvertedSNN` values;
`ann_to_snn.mojo` exposes `ReplayResult`, `replay` and zero-based `classify`.
Flat vectors follow the same row-major coefficient, frame, state and trace order
as Python. Compile this numerical profile with `--fp-mode contract=off`; the
default compiler mode can fuse operations across statements.

```mojo
from ann_to_snn import replay
from ann_to_snn_parameters import DenseLayer, ConvertedSNN

var weights: List[Float64] = [1.0]
var bias: List[Float64] = [0.25]
var layers = List[DenseLayer]()
layers.append(DenseLayer(1, 1, weights, bias, 1.0, 0.5))
var model = ConvertedSNN(layers)
var frames: List[Float64] = [0.5, 0.5]
var result = replay(model, frames, 2, 1, trace=True, binary_inputs=False)
var next_frames: List[Float64] = [0.0]
var continued = replay(model, next_frames, 1, 1,
    initial_state=result.final_state, use_initial_state=True)
print(continued.output[0])
```

Use `ConvertedSNN(layers, linear=True)` for cumulative signed output. Bias applies
every timestep; IF thresholds are inclusive with single-event subtractive reset.
`use_initial_state=True` explicitly selects supplied states, including empty
arrays for zero-batch replay. Public parameter edits are snapshotted and checked
again. `max_working_bytes` defaults to 256 MiB and uses the shared numeric-buffer
reservation. Refusals raise `Error` with `IF invalid input`, `IF resource limit`
or `IF overflow`; no partial result or changed caller state is returned.

`tests/test_accel_mojo_if_replay.py` compiles real native calls and compares all
output/state/event bits with Python, including actual chunk continuation.
`tests/test_accel_mojo_if_admission.py` exercises constructor/replay refusals,
exact memory limits, overflow and independent ownership. Missing-docstring and
warning diagnostics are checked separately on each production module, because
checking only an importing caller does not check every imported docstring.
Direct native use does not automatically change Python backend selection.

## Converter

::: sc_neurocore.conversion.ann_to_snn
    options:
      show_root_heading: true
      members:
        - convert
        - ConvertedSNN
        - replace_relu_with_qcfs

## QCFS Activation

::: sc_neurocore.conversion.qcfs
    options:
      show_root_heading: true
      members:
        - QCFSActivation

## Torch-free QCFS evaluation

`qcfs_forward(x, steps=8, theta=1.0, backend="auto")` and
`qcfs_backward(x, upstream, steps=8, theta=1.0, backend="auto")` evaluate the
QCFS lattice and its straight-through derivatives in float64 without PyTorch.
They follow the operation order of `QCFSActivation` and its autograd, so the
values, the input derivative and each element's threshold derivative equal the
float64 activation bit for bit, including NaN payloads and the sign of zero.
The threshold array holds the gradient a one-element batch receives; its sum is
a shared threshold's gradient up to summation order. `upstream` must have the
shape of `x`; boolean, complex, object and text inputs are refused.

`backend` accepts `"auto"`, `"numpy"`, `"rust"`, `"go"`, `"mojo"` or `"julia"`,
and every runtime returns the NumPy bits. Rust, Go and Mojo load the libraries
named by `SC_NEUROCORE_QCFS_RUST_LIB`, `SC_NEUROCORE_QCFS_GO_LIB` and
`SC_NEUROCORE_QCFS_MOJO_LIB`; Julia runs in the configured JuliaCall runtime
(the same `PYTHON_JULIACALL_*` settings as Julia replay) when
`SC_NEUROCORE_QCFS_JULIA_ENABLED=1`. Auto tries Mojo, Rust, Go, then Julia over
what is configured and otherwise uses NumPy: the order of the local comparison
below, a preference rather than a guarantee. An explicit runtime without its
configuration raises `RuntimeError` instead of falling back.

Each library exports `sc_qcfs_abi_version`, `sc_qcfs_forward` and
`sc_qcfs_backward` over float64 arrays, validates the grid, threshold and every
span before writing, and returns `0` or `-1` with outputs unchanged. Build them
from a source checkout, or from an installed package by replacing
`src/sc_neurocore` with the package directory:

```bash
cargo build --offline --release \
  --manifest-path src/sc_neurocore/accel/rust/safety/qcfs_native/Cargo.toml \
  --target-dir /your/build/directory
(cd src/sc_neurocore/accel/go && \
  go build -buildvcs=false -buildmode=c-shared -o /your/qcfs-go.so ./conversion/qcfscshared)
mojo build --fp-mode contract=off -I src/sc_neurocore/accel/mojo/kernels \
  --emit shared-lib -o /your/qcfs-mojo.so src/sc_neurocore/accel/mojo/kernels/qcfs.mojo
export SC_NEUROCORE_QCFS_RUST_LIB=/your/build/directory/release/libsc_neurocore_qcfs.so
```

`benchmarks/bench_qcfs_runtimes.py --output report.json --cpu 0` times public
forward and backward calls for 1, 64, 4096 and 262144 elements in all five
runtimes, each in a fresh process pinned to one CPU. Every response must equal
the NumPy bits before its sample counts, and the report binds the QCFS source
and library bytes; it is published atomically only after all five pass. Host
load is recorded, not isolated.

::: sc_neurocore.conversion.qcfs_dispatch
    options:
      show_root_heading: true
      members:
        - qcfs_forward
        - qcfs_backward
        - select_qcfs_native

## Measured conversion loss

`measure_conversion_loss(model, snn, inputs, labels, *, input_mode="constant",
seed=42, batch_size=256, max_working_bytes=..., backend="auto")` classifies the
same labelled samples with a PyTorch source and the network converted from it,
and returns a `ConversionLossReport` of what both did. Nothing is estimated:

| Field | Meaning |
|---|---|
| `source_accuracy`, `converted_accuracy` | fraction of samples each network labels correctly |
| `accuracy_drop` | `source_accuracy - converted_accuracy`; positive is a loss |
| `agreement` | fraction of samples on which both predict the same class |
| `rate_mean_abs_error`, `rate_max_abs_error` | decoded SNN rates against the source outputs |
| `timesteps`, `input_mode`, `seed`, `batch_size` | how the converted network was driven |
| `backend` | the replay runtime every converted batch ran on |
| `source_sha256`, `converted_sha256`, `data_sha256` | digests of the source parameters and buffers, the converted coefficients and semantics, and the inputs with their shape and labels |

`inputs` are `(samples, *source_shape)` values in `[0, 1]`; the converted
network receives each sample flattened in C order. The source runs under
`no_grad` in inference mode on its first parameter's device and dtype, and
every module's training flag is restored afterwards. Poisson batch `k` uses seed
`(seed + k) mod 2**32`. `backend="auto"` is resolved once, by
`resolve_replay_backend`, before the first batch, so the report names the
runtime that executed rather than the one requested. `to_public_dict()` gives
the JSON form the Studio seals.

::: sc_neurocore.conversion.loss_report
    options:
      show_root_heading: true
      members:
        - measure_conversion_loss
        - ConversionLossReport

## Target fixed-point calibration

`calibrate_for_target(snn, profile, inputs, labels=None, *, batch_size=256,
max_working_bytes=..., backend="auto")` fits a converted network into one
hardware profile's fixed-point format and measures what that costs on the given
samples, driven as constant currents. Integrate-and-fire dynamics with
subtractive reset are unchanged when one layer's weights, bias, threshold and
membrane are scaled together, so each layer gets the finest power-of-two scale
at which its coefficients, threshold and measured membrane peak fit the format;
its coefficients are then rounded half to even onto the format's grid.

The rounded network is replayed on the same samples and compared with the
unrounded one: agreement, the largest decoded-output difference and, with
labels, both accuracies. Each layer reports its scale, threshold code, measured
peak and headroom, rounding errors, weights rounded to zero, whether its preload
lies on the grid and every sample-step at which the rounded network's membrane
left the range. `compatible` is false, with named `refusals`, for a threshold
below one grid step, negative coefficients in an unsigned format or any measured
overflow.

A layer driven by spikes accumulates exact multiples of the grid step, so its
replay equals integer fixed-point accumulation when `exact_accumulation` is
true (format at most 53 bits wide, preloads on the grid). The first layer's
analog products are not rounded as the target would round them, the profile's
overflow and rounding modes are recorded rather than emulated, and only the
numeric format is checked: neuron counts, fan-in, core mapping, routing,
latency and energy are not. A range measured on calibration samples can be
exceeded by other inputs.

::: sc_neurocore.conversion.target_report
    options:
      show_root_heading: true
      members:
        - calibrate_for_target
        - TargetReport
        - LayerCalibration

## Reproducible runtime comparisons

After explicitly building and configuring all four native providers as above,
run the complete public-call comparison on one allowed Linux CPU:

```bash
PYTHONPATH=src python benchmarks/bench_ann_to_snn_replay.py \
  --output /your/evidence/dense-if-comparison.json --cpu 0 --samples 7 --warmup 3
```

From an installed wheel, the same bundled comparison runs as:

```bash
sc-neurocore-if-benchmark \
  --output /your/evidence/dense-if-comparison.json --cpu 0 --samples 7 --warmup 3
```

The command runs NumPy, Rust, Go, Mojo and Julia sequentially in fresh equally
pinned processes. Twenty fixed seeded workloads cover single/dense stacks,
full state/event traces, both output modes, empty timesteps and 129-step
constant/Poisson encoders. Every response bit must match NumPy before a sample
is accepted. Each record binds shaped input/output digests, raw timings, current
source bytes and installed native artifact bytes. The report is published
atomically only after all five runtimes pass and sources/artifacts remain
unchanged; worker stdout/stderr are retained even when a comparison fails.

Latency includes public-call admission, encoding where applicable, native
transport and collection of the owned result vectors. Final returned-vector
release, interpreter startup and builds are excluded. The first selected
provider call follows the NumPy reference and includes lazy native startup;
warm samples follow the declared exact-workload repetitions. The aggregate is
an equal-weight geometric mean of the twenty warm workload medians. Single-CPU
affinity does not isolate host load, and results are specific to that capture.
These seeded weights are untrained runtime workloads. The comparison is
separate from trained source/target loss acceptance and has no energy estimate
or physical instrument/calibration claim.
