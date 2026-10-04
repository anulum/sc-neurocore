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

## Packed HDC vector inputs

`BitStreamTensor` and the `HDCVector` facade accept positive logical lengths.
`from_packed(data, length)` requires exactly `ceil(length / 64)` unsigned
64-bit words and zero unused high bits in the last word. Word zero stores the
first 64 logical bits, least significant bit first. The constructor copies
input words; the `data` property also returns a detached copy. Malformed storage
raises `ValueError` before admitting a vector. It is neither truncated nor padded.

Packed-word and bundle inputs retain Python's sequence protocol, including
tuples, arrays, buffer views and custom sequences. Their length hints reserve
storage fallibly; iteration still determines the actual elements. Missing or
inaccurate hints do not replace those elements. An impossible reservation raises
`MemoryError` instead of aborting the interpreter. Strings and non-sequences
retain their conversion errors, as do unsigned-word overflow and invalid tensor
elements. Sequence conversion and bundle-reference storage both reserve before
writing their complete inputs.

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

An inventory must retain these differences. Qualify each Python/NumPy, operating
system and engine-feature profile used for comparison; inherited Python protocol
members and optional native exports can vary with that profile. Matching names
or successful global-reference serialization does not establish matching model
dynamics or support for serializing a live neuron instance.
