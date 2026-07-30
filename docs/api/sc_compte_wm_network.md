<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# SC Compte working-memory network

`SC-COMPTE-WM-NETWORK` is the retained SC-NeuroCore network-level successor to
the source-bounded [`CompteWMNeuron`](models/compte_wm.md). Its public Python
specification and executor are `sc_neurocore.network.SCCompteWMNetworkSpec`
and `sc_neurocore.network.SCCompteWMNetwork`.

This is deliberately an **SC project model**. It uses the architecture and
control parameters reported by Compte, Brunel, Goldman-Rakic, and Wang,
*Cerebral Cortex* 10(9), 910–923 (2000), DOI
`10.1093/cercor/10.9.910`, while freezing reproducibility choices that the
paper does not define as a portable executable contract. It is neither another
neuron nor the legacy 500-cell `working_memory_circuit` approximation.

## Frozen v1 surface

- 2,048 pyramidal cells plus 512 inhibitory interneurons on uniform
  preferred-cue rings;
- independent 1,800 Hz per-cell external Poisson drive through AMPA;
- a counter-addressed SplitMix64/inverse-CDF input stream whose seed, stream,
  step, and cell mapping is stable across batching and intended for direct
  native-language ports;
- control conductances `G_EE=0.381 nS`, `G_EI=0.292 nS`,
  `G_IE=1.336 nS`, and `G_II=1.024 nS`;
- a unit-mean E-to-E footprint with `J_plus=1.62` and `sigma=18 degrees`;
- optional tuned E-to-I connectivity with `J_plus=1.25` and
  `sigma=18 degrees`;
- source cell constants and AMPA/NMDA/GABAA kinetics at `dt=0.02 ms`;
- no recurrent E-to-E or I-to-I autapses in the SC v1 execution contract;
- named control and modulated sets, where the latter scales recurrent NMDA by
  1.2 and recurrent GABAA by 1.4; and
- a deterministic SC compact cue profile plus explicit circular population
  statistics.

The implementation computes shortest circular distances, exact discrete
unit-mean connectivity footprints, cue currents, signed distractor
displacements, population firing rates, bump angle, resultant length, and
circular width. The vectorized executor advances all 2,560 cells with coupled
midpoint RK2 channel flow, circular E-to-E convolution through a real FFT,
optional structured E-to-I convolution, uniform inhibitory projections,
sampled threshold/reset/refractory behaviour, explicit event overrides, and
atomic candidate validation. Every step receipts external counts plus input
and state digests; every run receipts input, spike, final-state, and bounded
window statistics. Invalid sizes, non-finite values, non-positive parameters,
partial event overrides, out-of-run stimuli, empty target grids, and
spike-count shape mismatches fail closed.

## Source and SC choices

The paper supplies the biological architecture, control conductances, channel
equations, timestep, population sizes, and Poisson rate. It does not supply a
portable pseudorandom stream, a cross-language aggregate-input mapping, or an
unambiguous autapse convention. The counter stream, inverse-CDF sampler,
per-cell aggregate counts, no-autapse rule, compact raised-cosine cue, sampled
threshold detector, digest encoding, and reduction order are therefore
explicit SC project choices. The source used a larger 4,096+1,024 network for
its reported distractor experiment; the frozen SC v1 identity remains the
requested 2,048+512 ring, so future distractor evidence must be described as
SC-network evidence rather than a reproduction of that larger figure.

The preserved scalar `CompteWMNeuron` remains a separate original model. A
focused executable parity test isolates one network pyramidal cell and proves
its external-AMPA midpoint step agrees with that original model. Another test
compares the FFT ring path against an independently reduced dense target sum.

## Native Rust lane

`engine::sc_compte_wm_network` is the documented modular Rust runtime for the
same fixed state transition. It owns the complete 2,560-cell state, preplans
its `rustfft` circular reductions, supports control/modulated and optional
structured E-to-I modes, accepts explicit event arrays for co-simulation, and
fails atomically on invalid state or input. Its counter-Poisson fixture has the
same active cells as Python, its isolated external-AMPA step agrees with the
preserved Rust scalar cell, and a non-trivial recurrent-NMDA FFT step agrees
with the Python dense-oracle fixture within `3e-13 mV`.

The separately compiled dependency-free Rust safety file validates all nine
state arrays, reproduces the counter stream, and supplies an O(N²) dense
no-autapse E-to-E oracle without sharing the production FFT. Public structs,
fields, constants, functions, and methods have rustdoc. A normal no-default-
features Cargo documentation build succeeds; promoting crate-wide
`RUSTDOCFLAGS=-D warnings` remains blocked by pre-existing broken links in
unrelated legacy modules and is not represented as a clean project-wide gate.

## Native Julia lane

`accel/julia/sc_compte_wm_network/SCCompteWMNetwork.jl` is the complete Julia
runtime for the same separately named network. It owns all nine state arrays,
uses the declared `FFTW.jl` dependency for circular E-to-E and optional tuned
E-to-I reductions, implements the counter-addressed Poisson streams, supports
explicit event-count co-simulation, and returns step/window/run receipts. Its
native suite binds Julia to the Python/Rust counter fixture, the preserved
Julia scalar Compte cell, and the non-trivial recurrent dense-oracle anchor;
it also exercises deterministic seed separation, full-population stimulus and
refractory behavior, atomic invalid-input rejection, and native docstrings.

The dedicated `Project.toml` and `Manifest.toml` freeze the Julia dependency
surface. Dependencies and compiled caches are instantiated with
`JULIA_DEPOT_PATH=.venv/julia_depot`, keeping them inside the repository venv
boundary. The committed Julia benchmark receipt measures three fresh 1,000-
step FFT runs and hashes its project, manifest, runtime, and benchmark source.
The full 1,000-step input digest, spike digest, and population spike counts are
exactly equal to the committed Python receipt; binary64 state digests are not
claimed bit-identical across FFT libraries. It is local regression evidence
only: it makes no production-throughput, persistent-bump,
distractor-resistance, hardware, or all-runtime claim.

## Native Go lane

`accel/go/sc_compte_wm_network` is the complete Go runtime for the same fixed
SC network. It owns all state and receipt arrays, supports control/modulated
and optional structured E-to-I modes, implements explicit-event
co-simulation, stimuli, activity windows, and atomic failure, and preserves
the source-bounded Go scalar Compte cell in `services` as a separate model.
Every exported package/type/function/method surface has GoDoc.

The production circular reduction uses an in-tree iterative radix-2 complex
FFT and therefore adds no third-party module dependency. Go toolchain/module
and build caches are routed through `.venv/go`. Native tests bind the counter
fixture, isolated scalar-cell impulse, non-trivial recurrent anchor,
deterministic seed behavior, full-population current/refractory behavior, and
atomic invalid-input boundary. The source-bound three-repeat 1,000-step Go
receipt exactly matches Python and Julia input digests, spike digests, and
population spike counts. Binary64 state hashes are not claimed bit-identical
across FFT implementations, and the timing is local regression evidence only.

## Native Mojo lane

`accel/mojo/sc_compte_wm_network/sc_compte_wm_network.mojo` exports the complete
fixed-size transition through a stable C ABI. Mojo builds both unit-mean
footprint spectra, samples the portable counter-Poisson streams, applies its
dependency-free radix-2 FFT reductions, advances all nine state arrays through
midpoint RK2, and commits threshold/reset/refractory events atomically. The
Python custody facade supplies contiguous caller-owned storage, protocol
currents, window statistics, and canonical receipts; it never substitutes the
Python network recurrence. Native comments document every ABI address, scalar,
status, ownership, and invalid-output boundary.

The committed shared library is built with the repository-local
`.venv/bin/mojo` 0.26.2 toolchain and the portable `x86-64-v3` target. The
source/binary-bound three-repeat 1,000-step receipt exactly matches the
Python, Rust, Julia, and Go input digest, spike digest, and population spike
counts. Its binary64 final-state digest is runtime-specific because FFT
reduction orders differ. The timing is local regression evidence only and
does not establish persistent-bump behavior, distractor resistance, hardware
performance, or production throughput.

## Public backend dispatch

`sc_neurocore.network.run_sc_compte_wm_network` exposes the five complete
runtime routes under the explicit backend names `python`, `rust`, `julia`,
`go`, and `mojo`. `sc_compte_wm_backend_status()` reports their availability
and execution mode before a run. Python and Mojo execute in process; Rust,
Julia, and Go use documented repository-native JSON adapters. Their toolchains,
package stores, and build caches resolve through `.venv`. Native v1 dispatch
accepts the frozen constants plus `seed`, `structured_ei`, `modulated`, and
`allow_recurrent_autapses`; unrepresented configuration changes fail before
launch.

Selection is deliberately fail closed. There is no `auto` mode and no silent
fallback: requesting a missing, timed-out, nonzero-exit, wrong-identity, or
malformed native route raises `SCCompteWMBackendUnavailable`. The returned
`SCCompteWMBackendRun` identifies the backend, reports the runtime's measured
execution interval, and carries the common `SCCompteWMRunReceipt`.

The consolidated source/binary-bound benchmark invokes all five routes for
three 1,000-step repetitions. Every runtime exactly agrees on the canonical
input digest, spike digest, and `1` excitatory / `27` inhibitory spike counts.
Final binary64 state hashes remain per-runtime custody rather than an asserted
bit identity across FFT reductions. The recorded medians are local loaded-host
regression measurements; Julia's command route includes JIT compilation in
its reported interval. They are not production, hardware, or behavior claims.

## Claim boundary

The Python executor and committed benchmark are deterministic simulator
receipts, not behavioral validation. The benchmark exercises 1,000 full
network steps and binds its result to source hashes, but records local loaded-
host regression timing only. Persistent-bump formation, delay stability,
random drift, response reset, distractor resistance, and silicon behavior
remain open until demonstrated by separately committed ensemble and backend
evidence. The current native claim is explicit Python/Rust/Julia/Go/Mojo
dispatch with exact event and short-trace custody. The network therefore does not
increment the 49/155 neuron-model fidelity count.

## Example

```python
from sc_neurocore.network import (
    SCCompteWMNetwork,
    SCCompteWMNetworkSpec,
    SCCompteWMStimulus,
    run_sc_compte_wm_network,
)

spec = SCCompteWMNetworkSpec(modulated=True)
angles = spec.preferred_angles_deg("excitatory")
cue_pa = spec.cue_current_pa(180.0, angles)
ee_footprint = spec.connectivity_footprint("ee", 180.0, angles)

assert spec.n_cells == 2560
assert cue_pa.max() == 200.0
assert abs(ee_footprint.mean() - 1.0) < 1e-12

network = SCCompteWMNetwork(spec)
cue = SCCompteWMStimulus(0.0, 250.0, 200.0, center_deg=180.0)
receipt = network.run(250.0, stimuli=(cue,))
assert receipt.steps == 12_500
assert len(receipt.final_state_sha256) == 64

native = run_sc_compte_wm_network(
    0.1,
    backend="rust",
    statistics_window_ms=0.1,
)
assert native.backend == "rust"
assert native.receipt.steps == 5
```
