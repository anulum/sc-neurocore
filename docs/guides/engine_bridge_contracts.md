<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
# SC-NeuroCore — Stable Engine Bridge Contracts

Stable engine consumers should use explicit wrapper modules under
`bridge/sc_neurocore_engine/`, not the top-level `sc_neurocore_engine`
namespace as a generic export bag.

Native distributions exclude `__pycache__`, `.pyc` and `.pyo` files. Interpreter
caches are created locally from the installed Python source when required.
Wheel CI checks the archive before uploading it and rejects cached bytecode;
installed contract checks retain the originating source and extension digests.

Batch NumPy encoding, fixed-point batch simulation, network population stepping
and Kuramoto baseline/SSGF stepping share the existing global Rayon pool.
Worker creation failures raise `MemoryError` with
`parallel execution resources cannot be allocated`; restoring resources permits
another initialization attempt in the same process. `set_num_threads(0)` leaves
lazy default selection unchanged. A successful explicit thread configuration
remains fixed, and another configuration attempt retains its `ValueError`.
Kuramoto validates the timestep and SSGF matrices before starting workers. A
worker refusal leaves its phases unchanged; restoring resources permits the
same operation to be retried. Zero-step baseline and SSGF runs validate their
inputs and return empty results without initialising the pool.
Network stepping validates the population index and current-vector length
before worker admission. A worker refusal occurs before copying currents,
resetting populations or propagating projections. Zero-step runs and empty
populations return their normal empty results without initialising the pool.
Attention validates matrix and head dimensions before worker admission,
including the K/V row count for multi-head calls. A zero-row query returns its
normal empty result after validation without starting workers; stochastic mode
still rejects a zero bitstream length. Linear, softmax and stochastic methods
retain their numerical kernels and permit retry on the same object after a
worker-resource refusal.
Graph rate-mode aggregation admits workers after checking feature shape for
dense and CSR storage. A zero-node graph stays lazy. Its stochastic forward
path remains serial and does not initialise the global Rayon pool.
Cortical CSR injection checks every used row offset, storage bound and column
index in every block before changing the output. Short row pointers, negative
or decreasing offsets, out-of-range columns and mismatched block lists raise
`ValueError` before worker admission. Unused trailing CSR storage remains
accepted. Empty outputs stay lazy; an empty multi-block sum retains its
original `y += 0.0` floating-point behavior without starting workers.
Dense layer construction validates its positive bitstream length, input count
and neuron count, then admits the shared pool before packing random weights.
Worker refusal raises the same recoverable `MemoryError` and permits a fresh
construction with the original seed. Its existing scalar and parallel forward
thresholds are unchanged; a successfully constructed layer already has an
initialised shared pool.
The Python `GpuLifBatch` and `GpuIzhikevichBatch` runners validate nonempty
output products, shader indices, device buffer capacity and workgroup limits
before allocating or dispatching. Unrepresentable or oversized output storage
raises `MemoryError`; shader-index and dispatch-limit refusals raise `ValueError`.
Input-length errors retain precedence. Empty shapes, accepted numerical kernels
and same-instance retries retain their established behaviour.
`GpuLifBatch` applies the CPU fixed-point configuration validator before
output admission, including for empty outputs. Bit widths outside `[1, 16]`,
fractional precision at or above the bit width and negative refractory periods
raise the same `ValueError` messages as the CPU batch interface.
Optimizer search requires at least one layer and a scoring weight for every
layer. Empty searches and short weight lists raise `ValueError` before candidate
construction, including when `max_iter=0`; they cannot abort the interpreter.
Trailing weights retain their existing score-normalization semantics. The
public Rust search returns `None` for these malformed dimensions or an empty
candidate list. Accepted seeded searches retain their configurations and scores.
Pareto extraction checks that power and score columns cover every LUT entry
before worker admission, retaining unused trailing values. DNA pair scoring
starts workers only for at least two strands. Ising energy batches, annealing
reads and gauge generation admit workers for nonempty work; their empty results
stay lazy. Active annealing refuses a coupling with exactly one endpoint outside
the qubit range before sampling, while wholly out-of-range ignored edges and
zero-read results retain their existing behavior.
Differentiable dense construction parses the surrogate first, validates positive
layer dimensions, then admits workers before allocating weights. Successful
construction fixes the shared pool for forward, backward and weight refresh;
invalid backward or update shapes retain the existing cache and weights.
Photonic routing checks storage for every used off-diagonal entry before worker
admission; the unused final diagonal and trailing storage are not required.
Zero- and one-node routes stay lazy. WDM, power and geometric-pair calculations
admit workers only for nonempty effective input. WDM and power retain zipped
column-prefix semantics, while geometric pair columns must have equal lengths.
Empty inputs, MZI operations and uniform-bank analysis do not start workers.
Evolutionary mutation, fitness, crossover, diversity and novelty admit workers
before touching native output or random draws. Empty populations, unpaired
crossover inputs and fewer than two genomes for diversity retain lazy results.
Ragged gene vectors retain their common-prefix arithmetic and crossover lengths.
Tournament selection remains serial; a positive selection count with an empty
fitness population raises `ValueError`. Zero selections remain valid, and zero
or one tournament round retains the same original single-candidate sampling.

The native project carries the narrow Rayon registry startup patch in
`bridge/vendor/rayon-core`, with upstream source hashes and patch ownership in
`bridge/rayon_source_patch.toml`. Its Cargo path override preserves the locked
upstream package identity and dependency graph for advisory scanning. Native
source distributions include the override and its source and licence files.

Current maintained wrapper surfaces:

- `sc_neurocore_engine.network`
- `sc_neurocore_engine.studio`
- `sc_neurocore_engine.world_model`
- `sc_neurocore_engine.photonics`
- `sc_neurocore_engine.dna`
- `sc_neurocore_engine.quantum`

Why:

- explicit import points are easier to test
- capability failure becomes local and diagnosable
- consumers stop depending on unrelated re-export churn in
  `bridge/sc_neurocore_engine/__init__.py`
- mypy can be paid down module by module instead of treating the entire
  top-level bridge as one dynamic surface

Rule:

- runtime code should import the narrow wrapper it actually needs
- top-level re-exports remain compatibility surface, not the preferred
  implementation contract

## Neural-decoder arrays and workers

The five native neural-decoder functions retain their registered `py_` names
and flat outputs. Tokenisation accepts contiguous one-dimensional `int32`
trains; position encoding, attention and InfoNCE accept contiguous
one-dimensional `float64` buffers, including readonly and aliased inputs.
Noncontiguous arrays raise `ValueError` naming the input. Attention and InfoNCE
check every used buffer prefix before starting workers; unused tails remain
accepted. Unrepresentable matrix dimensions raise `MemoryError` with
`requested matrix is too large`.

Empty token trains, empty or zero-width position encodings, zero-query or
zero-width attention, and zero-row or zero-width InfoNCE return empty arrays
or zero loss without starting workers. Attention with no keys retains its
zero output. Nonempty computations use the same global Rayon pool and its
recoverable worker-resource refusal. Refusals leave caller arrays unchanged;
restoring resources permits a retry with the same inputs. Numerical kernels,
stable token ordering and supplied timestep, sigma and temperature values
remain unchanged.

## Population and statistical decoding

Population decoding accepts contiguous `int32` trains and `float64` preferred
directions in radians. It uses complete windows from the shortest train;
missing directions retain zero radians. Zero windows and no complete bins
return empty output without starting workers.

MAP and maximum-likelihood decoding accept contiguous `float64` counts and
flat row-major tuning rates. Counts must contain the used neuron prefix and
rates the full stimulus-by-neuron prefix. Extra tails remain accepted. Empty
priors remain uniform; short nonempty priors retain the missing-entry floor.
Zero dimensions return zero without starting workers.

Fisher and Gaussian naive Bayes accept contiguous `float64` features and
points with contiguous `int64` labels. Used prefixes are checked before
computation. Fisher returns a sole used class without reading feature prefixes
or starting workers. Multi-class Fisher checks scatter capacity and admits
workers. Naive Bayes retains serial execution, including under worker-startup
pressure. Refusals leave caller arrays unchanged. Restoration permits the same
parallel operation to be retried with the original data and numerical kernels.

## Causality

Causality functions accept contiguous `int32` trains and positive bin sizes.
Pairwise and conditional Granger retain serial execution and common-bin
truncation. Zero order and insufficient common bins return zero. Spectral
Granger, PDC and directed transfer fit VAR using the first train's binned
length. During a fit, order must be positive and every other train must contain
that used prefix; longer tails remain accepted. Short first-train data retain
the original serial fallback. Zero frequency count still fits VAR; empty
train lists return empty output and retain their frequency metadata.
All used matrix capacities are checked before computation or worker admission.
Refusals leave inputs unchanged and worker-resource restoration permits retry.

## Covariance and spike-pattern matrices

Both covariance functions retain shape `(1, 0)` for no trains and shape
`(1, 1)` for one empty train. Short trains contribute one sum bin; multiple
trains use their common complete-bin prefix. A single train stays serial.
Positive bin size is required when a train is present. Multiple rows admit
the shared workers after checking input layout and allocating final output.

Directionality retains serial inclusive time filtering and its original
nearest-before/after rule. Train ordering preserves the upper-pair statistic
and its negation below the diagonal. Empty and single-train order matrices
stay lazy. Cubic cumulants retain the full sample mean, valid-lag denominator
and unused timestep parameter. Zero lag width and empty samples return zero
matrices without workers. Final covariance/order/cumulant NumPy output
allocation is checked before worker admission; failure preserves inputs and
lazy pool configuration. Restoring resources permits the same call to recover.

## Sliding-window bitstream estimator

`BitstreamAverager` retains a positive sliding window and estimates the mean
over observed samples until that window fills. Reset discards observations
without changing the window size. Python and the native class retain the same
partial-window, wrapping and reset traces for binary samples.
The Python estimator converts NumPy integer samples and buffered bytes to
Python integers for the running sum, so large windows do not overflow `uint8`.

A zero window raises `ValueError` before an estimator is created. The Python
reference uses its `SCEncodingError` subclass of `ValueError`. Native buffer
capacity or allocation failure raises `MemoryError` with
`Bitstream averager buffer cannot be allocated`; Python translates NumPy's
allocation failure to the same message. A failed construction leaves existing
estimators usable.

Native `push` retains unsigned-byte extraction. Bytes other than zero or one
raise `ValueError` with `Bit must be 0 or 1.` before changing buffered samples,
the running sum or the next write position. The Python reference retains its
`SCEncodingError` for the same binary-sample refusal. Rust callers can use
`neuron::BitstreamAverager::try_new` and `try_push` for fallible admission;
the existing `new` and `push` convenience methods retain their signatures
and panic on refused inputs or construction allocation failure.

## Bitstream packing

`pack_bitstream_numpy` borrows a contiguous one-dimensional `uint8` array,
including read-only input, and returns an independent, owning, writable,
contiguous `uint64` NumPy array. An empty input produces shape `(0,)`.
Every nonzero byte sets one bit, least significant bit first; unused high
bits in the final word are zero. Other ranks and dtypes retain their typed
refusal, and a noncontiguous input raises `ValueError` before output allocation.

The result is allocated through NumPy's checked allocator before packing.
Allocation failure propagates `MemoryError` without modifying input. AVX-512
and AVX2 selection and full-word instructions remain in use; a partial word
uses the portable byte-packing loop without allocating a temporary vector.
The existing SVE packing implementation is a portable fallback.

Rust callers can use `bitstream::pack_fast_into` or `simd::pack_dispatch_into`
to fill an existing buffer without allocating intermediate storage. The
buffer must contain exactly `ceil(input_length / 64)` words. A mismatch
returns an error before mutation; success overwrites every word. Existing
vector-returning Rust APIs retain their signatures and allocator behavior.
`pack_bitstream` retains a flat Python word list for one-dimensional sequences
and a list of word lists for nested sequences, including ragged and empty rows.
Its unsigned-byte extraction still accepts byte buffers and index-based Python
sequences. Byte-buffer subclasses retain PyO3's raw-buffer conversion; mutable
bytearrays are copied under a Python critical section before packing.
Input copies reserve native storage fallibly. Output lists and each Python
word are allocated through checked Python constructors, without intermediate
packed vectors. A capacity or allocation refusal propagates `MemoryError`;
other unsupported inputs retain `ValueError` with
`Expected a 1-D or 2-D array of uint8 bits.` The caller's input is unchanged.

Generic input storage grows from successfully converted sequence items.
An inaccurate or unavailable `__len__` hint does not reserve storage or replace
the actual contents. Probing a flat input as possible nested rows therefore
does not allocate a row container before its first element is admitted.

## Packed Bernoulli batch encoding

Both batch encoders accept aligned, contiguous one-dimensional `float64`
probabilities, including read-only arrays. Zero bit length and zero rows are
valid. Output rows contain `ceil(length / 64)` words with zero unused high bits.
Other ranks and dtypes retain `TypeError`; unreadable layout retains `ValueError`
with the `Cannot read probs` prefix. Unrepresentable total word storage raises
`MemoryError` with `encoded bitstreams cannot be allocated` before sampling.

`batch_encode` returns Python word lists and uses one sequential ChaCha8 stream
across all rows, with one `float64` draw per bit. Its native word reservation and
Python list/integer output constructors are fallible.

`batch_encode_numpy` returns an independent, owning, writable, contiguous NumPy
`uint64` matrix. It allocates the final matrix through NumPy before sampling and
fills disjoint rows in parallel, without row vectors or a flattening copy. Each
row retains Xoshiro's wrapping `seed + row_index` seed and byte-threshold SIMD
comparison. The 1024-byte batches, full-word remainders and partial-tail draws
remain unchanged. Scalar and NumPy encoders have distinct seeded traces;
clamped endpoints and NaN retain each encoder's sampling behavior.

The measured seeded fixtures exercise both entry points at bit-word and SIMD
batch boundaries, including wrapping seeds and nonfinite probabilities. Rust
callers can fill existing buffers with `bernoulli_packed_into` and
`bernoulli_packed_simd_into`; a wrong word count changes neither output nor RNG
state. Existing vector-returning Rust interfaces retain their signatures.

Installed-wheel CI runs the batch contracts against the installed extension
before adding the checkout's test imports. It checks the wrapper and extension
paths and hashes again after the run, without loading the checkout's root test
bootstrap. The wheel-test dependency lock retains the supported Python version
markers and uses the same dependency pins as the development lock.

## Bitstream unpacking

`unpack_bitstream` retains bytes for a flat word sequence and a list of bytes
for nested sequences. A bare empty sequence retains its empty-list result;
a nonempty flat sequence with zero requested bits returns empty bytes.
`unpack_bitstream_numpy` accepts contiguous one-dimensional `uint64` input
and returns an independent, owning, writable `uint8` NumPy array, including
an empty array for zero length. Its input can be read-only.

Nested `original_shape=(batch, length)` checks the batch count and selects the
per-row length. Without it, the per-row length is `original_length // batch`;
an empty batch remains empty. Extra packed words and unused high bits retain
their existing ignored-tail behavior. Insufficient storage raises `ValueError`
with `packed words do not cover original_length` before output is published.
An output dimension exceeding `numpy.intp` raises `MemoryError` with
`unpacked bitstream cannot be allocated`. The NumPy reference follows the
same row-length and storage rules, including empty batches.

Packed input copies grow native storage fallibly from converted words while
retaining Python's sequence protocol and unsigned-word conversion. Length hints
do not reserve storage for speculative nested rows. Capacity or allocation refusal
raises `MemoryError`. Flat output is allocated directly as Python bytes;
typed output uses NumPy's checked allocator. A refused unpack leaves its input
unchanged and permits subsequent valid calls in the same interpreter.

## Packed HDC vector inputs

`BitStreamTensor` and the `HDCVector` facade accept positive logical lengths.
`from_packed(data, length)` requires exactly `ceil(length / 64)` unsigned
64-bit words and zero unused high bits in the last word. Word zero stores the
first 64 logical bits, least significant bit first. The constructor copies
input words; the `data` property also returns a detached copy. Malformed storage
raises `ValueError` before admitting a vector. It is neither truncated nor padded.

Packed-word and bundle inputs retain Python's sequence protocol, including
tuples, arrays, buffer views and custom sequences. Storage grows fallibly from
the actual converted elements; length hints neither reserve storage nor replace
those elements. An impossible reservation for the actual input raises
`MemoryError` instead of aborting the interpreter. Strings and non-sequences
retain their conversion errors, as do unsigned-word overflow and invalid tensor
elements. Each converted item is appended only after its storage is reserved;
the completed bundle-reference array is also reserved before it is populated.

XOR, bundling, rotation and the detached `data` copy reserve their native output
storage fallibly too. Failure raises `MemoryError`; rotation commits its packed
words only after both the unpacked and repacked allocations succeed. These
refusals preserve the input vectors and permit a later valid operation in the
same interpreter. Actual Linux process-limit contracts exercise input growth,
bundle references, packed outputs, both rotation buffers and Python list output.
They do not qualify allocation behaviour on other operating systems.

XOR, in-place XOR and normalized Hamming distance require equal logical lengths.
Bundling requires at least one vector and equal lengths for every vector.
These refusals raise `ValueError` before reading another vector's packed words
or modifying an input. Valid partial-word vectors retain zero padding through
XOR, strict-majority bundling and cyclic rotation. Their Hamming distance lies
in `[0, 1]`, with padding contributing no bits. `HDCVector` operators propagate
the same native errors.

Random construction reserves packed storage before consuming its seeded RNG.
If that reservation fails, it raises `MemoryError`; successful draws retain the
existing Xoshiro sequence. The Rust `try_bernoulli_packed` producer exposes the
same sampling with a reservation `Result`, including valid empty kernel output.
The existing native `bernoulli_packed` return type remains unchanged. Native
`BitStreamTensor::from_words` and public native struct fields remain caller-owned
storage contracts; these Python admission guarantees do not validate arbitrary
Rust struct literals.

## Default-neuron runtime contracts

The wheel matrix executes `tests/test_default_neuron_engine_binding.py` from
an external consumer directory against the installed extension. The gate covers
every current `py_neuron_default!` producer and checks its native class identity,
zero-argument construction, scalar conversion, detached state dictionary,
TypeError atomicity, temporal stepping and reset contract. Class references
round-trip through every pickle protocol. Instances deliberately refuse pickle,
copy and deepcopy; the refusal must preserve native state. ArcaneNeuron retains
its deep state through reset, as declared by its model.

The source-to-class table is checked against the current binding producers, so
adding a producer also requires adding its actual runtime contract. This shared
binding gate does not establish model equations or cross-language numerical
parity; those require the model's own reference and backend tests.

## Configured adaptive-threshold batches

The default-only native `AdaptiveThresholdIFNeuron` class uses the core's
fallible step. Non-finite scalar current raises `ValueError` with
`current must be finite`; a finite-entry relaxation overflow raises
`FloatingPointError` with `exact relaxation candidate must be finite`.
Both match the Python model and preserve the preceding state. Finite valid
stepping, detached state dictionaries, reset, global pickle identity and
instance-persistence refusal retain their existing contracts. The configured
batch has its separately documented complete-parameter interface below.

`py_adaptive_threshold_if_simulate` accepts nine finite configuration scalars
and a contiguous, aligned, native-endian one-dimensional `float64` current
array. Read-only arrays are accepted without changing their storage. Wrong
dtype, rank or byte order raises `TypeError`; strided and misaligned storage
raises `TypeError` before simulation. The configured batch remains distinct
from the default-only `AdaptiveThresholdIFNeuron` constructor.

All three state/event traces cover the complete input, with final voltage,
threshold and spike count returned separately. Empty input preserves the two
initial states. Invalid finite configuration or non-finite drive raises
`ValueError`; a non-finite numerical candidate raises `FloatingPointError`.
Trace storage is reserved before stepping. Capacity or memory exhaustion
raises `MemoryError` with `adaptive-threshold trace buffers cannot be allocated`
and leaves caller input unchanged. A subsequent valid batch remains usable.
The native Rust `simulate` reports the same resource failure through
`AdaptiveThresholdIFError::TraceAllocation`.

`tests/test_adaptive_threshold_if_engine_binding.py` compares every result
field with the Python exact-relaxation implementation and exercises the
installed native allocation refusal and recovery under real Linux pressure.

The configured Go C ABI reports failed temporary trace allocation as status
`5`, without writing any caller trace or final receipt. Its Python facade maps
that status to the same authored `MemoryError`. The temporary storage uses a
plain C allocator wrapper; cgo's special `C.malloc` would abort on exhaustion.
Every allocated prefix is freed on failure or successful completion. Julia's
real `OutOfMemoryError` is translated to the same Python resource error by its
facade. Its temporary traces remain private until successful completion.
The Mojo C batch instead validates the whole recurrence before writing its
caller-owned buffers and has no internal trace allocation to translate.

## Configurable Brunel-Wang contracts

`tests/test_brunel_wang_engine_binding.py` compares the installed native class
with the Python Brunel-Wang model across all 17 exposed constructor fields.
The native `c_m` argument corresponds to Python's `C_m`; initial
`ref_remaining` is supplied as public dynamic state on the Python reference.
Positional and keyword construction must produce the same trace. The gate
compares events, voltage and refractory time, including continuation after
reset with the configured parameters retained.

The same 21 configurations compare complete 256-step native traces with the
public Python, Rust, Julia, Go and Mojo batch routes. Events must agree exactly;
voltage and refractory state use the existing Rust/backend absolute tolerances.
The main CI compatibility matrix builds these backends and collects this module
through its normal test batches.

All four aggregate synaptic inputs must reject non-finite or negative values
before changing state, including during refractory time. A non-finite RK2
candidate raises the native `ValueError` and leaves the next valid step usable.
Class references round-trip through every pickle protocol; configured instances
deliberately refuse pickle, copy and deepcopy without changing state. These
contracts cover the configured PyO3 class and maintained software backends.
The model's co-simulation tests provide the separate RTL comparison.

## Configurable EnergyLIF contracts

`tests/test_energy_lif_engine_binding.py` exercises all 16 native constructor
fields, positional/keyword equivalence, two-state dictionaries, batch arrays,
reset continuation and serialization refusal. Twenty configurations compare
the real Python model with the native class and batch. The same configurations
exercise all five public backend routes in
`tests/test_energy_lif_backend_configuration.py`, retaining exact events and
the existing `2e-12` state tolerance.

`epsilon_0` must be positive; `alpha * epsilon_0` must be finite, positive and
at most five; `e_0` must lie in the enrolled voltage envelope. Invalid
configuration is refused before any batch, including an empty input. Reset
validates the equilibrium target before committing it. Direct Go/Mojo C ABI
tests verify that configuration refusal leaves caller-owned trace and final
buffers unchanged. These software contracts complement the model's immutable
source receipt and pinned RTL co-simulation.

The native `py_energy_lif_simulate` batch returns three independently owned,
writable, aligned, contiguous NumPy arrays: `float64` voltage and energy and
`int32` events. It validates the input layout before allocating output storage.
Output allocation failures raise Python `MemoryError`; partially allocated
outputs are released and the caller's current array is unchanged. A subsequent
batch or scalar call remains usable. The Linux allocation-pressure contracts in
`tests/test_energy_lif_engine_binding_allocation.py` exercise refusal at each
of the three allocations with a process address-space limit and restore that
limit before testing recovery.

`tests/test_energy_lif_engine_binding_inputs.py` retains exact scalar extraction
errors for every constructor and batch parameter. It also checks reversed,
broadcast and misaligned arrays, nonnative endianness, dtype/rank refusals,
contiguous offset views, empty input and readonly input at the actual extension.

The shared neuron benchmark records the loaded Julia runtime under
`tool_versions.julia` and the compiler embedded in the measured Go library
under `tool_versions.go`. `julia_cli` and `go_cli` retain the separate command
line tool versions. A newer installed tool does not identify an older binary's
producer. These workstation results retain their local regression classification;
they do not establish isolated performance or measured hardware energy.

## Configurable source MAT(1) contracts

The source MAT(1) `NonResettingLIFNeuron` exposes ten constructor fields and
three detached dynamic state values. Its configured class and batch traces,
including reset continuation, are compared with the Python source model in
`tests/test_non_resetting_lif_engine_binding_configuration.py`. The same sixteen
profiles exercise all five public backend routes in
`tests/test_non_resetting_lif_backend_configuration.py`. Global class identity
and exact instance pickle/copy refusal are pinned separately in
`tests/test_non_resetting_lif_engine_binding_serialization.py`.

`py_non_resetting_lif_simulate` checks configuration and input layout before
allocating its three `float64` state arrays and `int32` event array. All four
outputs own writable, contiguous NumPy storage. Real Linux allocation-pressure
contracts in `tests/test_non_resetting_lif_engine_binding_allocation.py` exercise
failure at each output allocation, partial-output release, unchanged inputs and
subsequent scalar/batch recovery. Scalar extraction and supported/refused NumPy
views are covered by `tests/test_non_resetting_lif_engine_binding_inputs.py`.

## Configurable source APSDM contracts

The native `SigmaDeltaNeuron` exposes five constructor fields and two detached
dynamic states. Twelve complete profiles compare positional and keyword
construction, full batch traces, empty input and reset continuation with the
Python source model in `tests/test_sigma_delta_engine_binding_configuration.py`.
The same profiles exercise the five actual public runtime routes in
`tests/test_sigma_delta_backend_configuration.py`. Native class identity and
instance pickle/copy refusal are pinned in the dedicated serialization module.

`py_sigma_delta_simulate` validates configuration and input layout before its
two `float64` state allocations and `int32` event allocation. The three outputs
own independent writable NumPy storage. The dedicated allocation module
exercises actual Linux address-space limits at every output allocation and
checks layout refusal under pressure, partial-output release, unchanged inputs
and subsequent class/batch recovery. The input module pins real scalar
extraction, accepted contiguous views and exact dtype/rank/layout refusal.
These are bounded installed-wheel contracts, with platform installation and
complete ABI-inventory qualification retaining their separate evidence gates.

## Configurable retained bipolar accumulator contracts

`SCSigmaDeltaAccumulatorNeuron` retains the project-defined signed recurrence,
with one event per sample and excess residual carried forward. Its two native
constructor fields are covered by thirteen profiles in the dedicated
configuration module, including both threshold equalities, signed zero,
maximum finite residuals/threshold and the minimum positive threshold.
Complete class/batch/reset traces and empty-input finals are compared exactly
with Python. The same profiles exercise all five actual public backend routes,
with exact residual and signed-event traces and preserved empty-state zero sign.
The native model's finite-state contract is retained without an extra state cap.

The direct batch checks configuration and layout before allocating its
`float64` residual and `int32` event arrays. Both arrays own independent writable
NumPy storage. Dedicated allocation tests exercise both real Linux allocation
failures, layout refusal under pressure, partial release and subsequent retry.
The input and serialization modules separately pin scalar conversion, NumPy
layout/dtype/rank refusal, class identity and instance pickle/copy refusal.
These contracts preserve the distinct SC compatibility identity; they do not
promote it to the sampled APSDM source model or qualify other platforms.

## Capturing the installed interface

With the wheel consumer's Python environment active, record the repository path
and run the tool from a consumer directory outside the source checkout:

```bash
repository_dir="$(git rev-parse --show-toplevel)"
consumer_dir="$(mktemp -d)"
cd "$consumer_dir"
python "$repository_dir/tools/engine_abi_inventory.py" capture \
  --require-installed --output engine-after.json
```

The capture records both the facade and compiled-extension namespaces: exported
names, aliases, class members, callable signatures, module and qualified names,
and whether each global reference resolves to the same live object. Measured
module paths and SHA-256 hashes identify the loaded code. The installed guard
rejects a checkout facade before importing it and verifies the loaded extension
also originates inside this interpreter's site-packages.
Both `purelib` and `platlib` installation roots are accepted, including Python
schemes that use a separate `lib64` directory for platform extensions.

Compare a previously captured interface with the candidate using the same Python,
NumPy and engine feature profile:

```bash
python "$repository_dir/tools/engine_abi_inventory.py" compare \
  --before engine-before.json --after engine-after.json
```

Comparison checks every interface field, including added names and changed
aliases or signatures. Installation paths and binary hashes remain provenance
and may differ. Exit status is `0` for a successful capture or matching interface,
`1` for interface drift, and `2` for invalid input or capture failure. Damaged
schemas, missing advertised exports, duplicate exports and non-finite JSON data
are rejected. Every symbol must retain its recorded identity fields; callable
records require a signature or an observed introspection failure. Classes retain
their member records, and globals retain reference-resolution and unique alias
metadata. Comparing two equally damaged captures cannot replace these fields
with missing observations. Duplicate JSON object keys are refused at every
capture layer, including provenance, before either comparison input is admitted;
an earlier value must not disappear during decoding. These checks validate the recorded metadata shape;
the original capture's byte identity and the completeness of its export cohort
still require separately retained producer evidence.

Capture and comparison do not execute pickle data. Dedicated runtime tests check
actual global pickle roundtrips; each binding's tests cover supported instance
state or its explicit refusal. Interface equality also needs separate behavioral
evidence for shape, dtype, contiguity, exceptions, numerical results and error
atomicity. A metadata capture alone cannot establish those contracts.

The wheel test matrix captures this interface after installation from a consumer
directory and retains `engine-abi-<os>-py<version>` artifacts. These measured
captures identify each tested installation; a stored capture is a compatibility
baseline only after its source, wheel and runtime profile have been qualified.

`tests/fixtures/engine_abi_default.json` retains the complete measured metadata
for the Linux x86-64, CPython 3.12, NumPy 2.2.3 release wheel with default engine
features. Its provenance identifies the source, wheel, capture tool and loaded
modules without including private installation paths. The matching wheel job
compares its installed capture with this reference. Other matrix profiles retain
their own captures; this reference does not certify their inherited Python
protocol members, optional exports or model behavior.

## Model compatibility and migration

The core native `FixedPointLif` class and `batch_lif_run`,
`batch_lif_run_multi`, and `batch_lif_run_varying` use signed `int16` state
and inputs. They require `data_width` in `[1, 16]`, `fraction` in
`[0, data_width)`, and a nonnegative refractory period. Invalid configurations
raise `ValueError` before stepping or returning empty outputs. Batch input
layout and length errors retain their existing precedence. The pure Python
`FixedPointLIFNeuron` has a separate width profile; its wider configurations do
not establish native support.

Batch outputs remain newly owned, writable, contiguous `int32` spike arrays
and `int16` voltage arrays. Dimensions must fit `numpy.intp`. NumPy size and
memory allocation failures propagate their Python exceptions, allowing the
consumer to catch the error and continue using the engine.

Rust callers can use `neuron::FixedPointLif::try_new` for configuration errors.
The existing `new` constructor retains its return type and panics for invalid
configurations, as documented.

`BrunelNetwork()` is the mean-field population model, with `step`, `get_state`
and `reset` methods. The fixed-point CSR simulator is separately exported as
`FixedPointBrunelNetwork`; its constructor takes connectivity arrays and LIF
parameters, and `run(n_steps)` returns a one-dimensional `uint32` array of spike
counts. The scaling benchmark uses this explicit fixed-point class. The two
classes have distinct native names and global serialization identities.

`FixedPointBrunelNetwork` instances refuse pickle with `TypeError` for every
protocol supported by Python. A refused serialization leaves the network's
next spike-count trace unchanged; the class itself still round-trips as a
global reference.

Simulation allocates an owned, writable, contiguous NumPy spike-count array
and reserves the synaptic-current buffer before advancing neurons or drawing
external events. A capacity or allocation failure raises `MemoryError` and
preserves the next seeded trace. Editing returned counts does not change the
network. Rust callers use
`BrunelNetwork::try_run` for the same fallible admission; the existing `run`
convenience method retains its `Vec<u32>` return type and documented panic on
allocation failure. Zero-step calls return an empty array without allocating a
synaptic-current buffer.

The constructor uses fallible allocation for its owned CSR copies, neurons
and previous-spike storage. Allocation refusal raises `MemoryError` without
changing the caller's arrays; a subsequent small construction remains usable.
The Linux profiles in
`tests/test_brunel_fixed_point_engine_binding_allocation.py` exercise real
address-space limits for CSR copies, neuron storage, output and synaptic working
storage, restoring the limit and comparing seeded continuation after refusal.
The working-storage profile retains real matching-size buffers to occupy
allocator storage freed by its warm-up calls before applying the limit.
These profiles do not qualify allocation pressure on another operating system.

The fixed-point constructor copies contiguous one-dimensional CSR arrays:
`w_indptr` and `w_indices` have dtype `int64`, and `w_data` has dtype `int16`.
Read-only arrays are accepted. Offsets start at zero, are nondecreasing and end
at the weight count; column indices lie in `[0, n_neurons)`. Repeated columns
and self-connections are accepted. Invalid CSR values raise `ValueError` before
simulation. The spike-count dtype limits `n_neurons` to the `uint32` range.

Its native LIF state uses signed `int16` values, so `data_width` is in `[1, 16]`
and `fraction` is in `[0, data_width)`. The refractory period is nonnegative.
`ext_lambda` must be finite and nonnegative; zero gives silent external drive,
and positive means must fit `rand_distr::Poisson::<f64>::MAX_LAMBDA`. Means below
30 retain the original seeded Knuth draws. Larger means use rejection sampling
to avoid exponential underflow. Synaptic and external current accumulation wraps
before fixed-point masking, including in debug builds. An empty network and a
zero-step run return typed empty or zero counts without fabricating spikes.

The shared `FixedPointLif` kernel keeps `v_rest - v` at twice the configured
width through leak multiplication and fractional scaling. It narrows the scaled
increment and final voltage to the configured state width. A 16-bit state can
therefore have a signed difference outside the `int16` range without losing its
high bits before scaling. Scalar, constant, varying and parallel batch entry
points use this kernel. Python and RTL comparisons must configure identical
thresholds, reset values and refractory periods; their defaults can differ.

Compare decomposition changes separately from later model corrections. Moving a
binding into its own Rust module should preserve its existing callable and state
contracts. A later correction to a model's equations can change those contracts
even when the exported class name remains the same.

Several original project dynamics have explicit names alongside the canonical
source models. Select the profile required by your saved configuration:

| Canonical engine class | Retained project profile |
| --- | --- |
| `BendaHerzNeuron` | `SCStochasticRateAdaptationNeuron` |
| `McKeanNeuron` | `SCTriangularMcKeanNeuron` |
| `MATNeuron` | `SCResettingMATNeuron` |
| `NonResettingLIFNeuron` | `SCNonResettingAdaptiveLIFNeuron` |
| `EnergyLIFNeuron` | `SCNormalizedEnergyLIFNeuron` |
| `SigmaDeltaNeuron` | `SCSigmaDeltaAccumulatorNeuron` |
| `TwoCompartmentLIFNeuron` | `SCExponentialTwoCompartmentLIF` |
| `GLIFNeuron` | `SCFourStateGLIFNeuron` |

Canonical `BendaHerzNeuron()` is deterministic and has no `seed` constructor
parameter; the stochastic profile accepts a seed. Canonical McKean batches take
a contiguous one-dimensional `float64` current array and return voltage,
recovery, event and final-state fields. The retained triangular batch keeps its
constant-current, step-count and tuple interface.

Method calls also need migration. `BrunelWangNeuron.step` accepts four aggregate
gate inputs and its state is a voltage/refractory tuple. `CompteWMNeuron.step`
uses `recurrent_event`, `external_event` and `inhibitory_event` keywords in place
of `spike_in`. Canonical `TwoCompartmentLIFNeuron.step` takes one `i_ext` input;
use the exponential project profile for the former two-current interface.

Inspect the installed callable's signature before migrating a positional batch
call. Canonical GLIF exposes six dynamic states, Wilson–Hindmarsh requires a
capacitance argument, Rulkov no longer takes `x_threshold`, and Mihalas–Niebur
uses decay-rate, retention and current-jump parameters. Their retained project
simulators have explicit `py_sc_..._simulate` names. Passing old positional
arguments to the canonical simulator can change their meaning or fail.

## Configured retained normalized energy-LIF

`SCNormalizedEnergyLIFNeuron` is the retained two-state normalized-energy
exact-flow profile. Its eleven constructor fields include initial voltage and
energy, resting/reset/threshold voltage, both time constants, event depletion,
equilibrium energy, resistance and step size. It retains a level-triggered event
with the strict energy gate `epsilon > 0.1`, clamped event depletion and
constant-current coupled exponential flow.

Current, resting and reset voltages belong to `[-200, 100]`. Parameters are
finite, energy belongs to `[0, epsilon_0]`, equilibrium energy and depletion are
nonnegative, time constants/resistance/step size are positive, step size does
not exceed either time constant, and threshold exceeds resting/reset voltage.
The complete configured resting state must be valid. Python reset checks a
candidate after configuration edits; valid reset can repair invalid dynamic
state, while an invalid configuration commits neither voltage nor energy.
Rust exposes checked `try_reset`; the native Python method retains its
None-returning successful reset.

The batch returns independent owning float64 voltage/energy arrays and int32
events; normalized five-runtime dispatch retains int64 events. Configuration
and readable contiguous input are checked before allocation. Empty batches keep
exact initial finals. Go/Mojo caller-buffer functions and Julia simulation also
refuse invalid empty-batch configuration before final-state writes. Native class
global pickle identity and instance pickle/copy refusal are separate contracts.

Dedicated configured contracts cover 25 profiles and 125 actual runtime routes,
including equal and nearly equal time constants, depleted/gated energy, every
field, voltage endpoints, custom configuration and reset continuation. The
existing `2e-12` floating envelope and exact event comparisons remain unchanged.
Mojo computes the small exponential difference with `expm1` and decay with
`1 + expm1(-dt/tau)` over the validated `0 < dt/tau <= 1` range, including the
maximum permitted timestep. Three real Linux output-allocation pressure cases
and reversed-layout pressure exercise exception propagation, released partial
outputs, readonly-input preservation and successful runtime retry. Other
platform allocator profiles require their own qualification.

## Configured retained non-resetting adaptive LIF

`SCNonResettingAdaptiveLIFNeuron` retains the project's exact voltage and
adaptive-threshold relaxation, with a level-triggered event and no voltage
reset. Its nine constructor fields are `v`, `theta`, `v_rest`, `theta_rest`,
`delta_theta`, `tau_m`, `tau_theta`, `r_m` and `dt`. They must be finite;
`delta_theta` and `r_m` are nonnegative, and time constants and step size are
positive. Finite voltages have no additional bound, and step size may exceed
either time constant.

Reset validates the complete resting candidate before committing either
dynamic value. A valid configuration can recover invalid dynamic state.
Checked Rust `try_reset` exposes refusal; successful native Python reset
returns `None`. Julia's constant-current overload also checks current and
step size for empty input.

The native batch accepts aligned, contiguous one-dimensional `float64`
currents, including readonly arrays. It returns independent owning `float64`
voltage/threshold arrays and `int32` events. Empty batches retain exact initial
finals and reject invalid configuration. Actual Linux allocation-pressure
tests cover both state allocations, event allocation, released partial
outputs, input preservation and successful retry. Reversed input retains its
`TypeError` refusal before allocation. These observations do not qualify
allocator behaviour on other operating systems.

Configured class/batch/reset, input layout, serialization refusal and actual
five-runtime traces have dedicated contracts. Events remain exact and the
floating envelope stays `2e-12`, including accepted timesteps larger than the
time constants. See the [retained model reference](../api/models/sc_non_resetting_adaptive_lif.md)
for the recurrence and the Go/Mojo caller-buffer failure contract.

## Cortical CSR output borrowing

`py_parallel_csr_spmv_add` and `py_parallel_csr_multi_spmv_add` accept
one-dimensional `int32` row offsets/column indices and `float64` data,
input vectors and output. Inputs may be readonly. A nonempty output must
be writable and contiguous, and must not overlap any input array. An
unavailable input borrow or overlapping/readonly output raises `ValueError`
before any output changes or parallel worker startup.

An empty output is a no-op, including when it is the same empty array as
an input. Multi-block lists must still have matching lengths. With no
blocks and a nonempty output, the existing serial addition of zero is
retained, including its conversion of negative zero to positive zero.

Every used CSR row is checked before execution. Unused trailing indices
and data do not enter the calculation. Worker allocation failure leaves
the output and caller inputs unchanged so the same call can be retried.

Attention and dense graph matrix inputs retain their existing refusal of
zero-row matrices; these inputs do not become successful empty operations.
Configuration and shape refusals precede parallel worker startup.

An inventory must retain these differences. Qualify each Python/NumPy, operating
system and engine-feature profile used for comparison; inherited Python protocol
members and optional native exports can vary with that profile. Matching names
or successful global-reference serialization does not establish matching model
dynamics or support for serializing a live neuron instance.
