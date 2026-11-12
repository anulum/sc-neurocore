<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Project — SC-NeuroCore acceleration mirror authority guide
-->

# Acceleration Mirror Authority

SC-NeuroCore contains multiple acceleration trees under `src/sc_neurocore/accel/`.
They do not all carry the same authority.

## Rule

Treat an acceleration file as authoritative only if at least one maintained
Python execution path loads it directly and tests cover that path.

The polyglot acceleration tree is research-only unless it satisfies that rule.
It is not shipped as the production runtime, and it is not required by
`pip install sc-neurocore`.

Everything else is one of:

- research-only source
- partial port
- transcript mirror
- historical scaffold

## Codec implementations and retained sources

The Python packet APIs live in `sc_neurocore.spike_codec`. The production Rust
engine provides waveform channel coders, stateful waveform-stream and spike-event
coding, and EMA/LFSR predictive kernels. These components do not establish that
every Python packet container has a complete Rust implementation. Use the
[codec catalogue](../api/spike_codec.md) for component availability and the
[wire-format reference](../reference/spike_codec_wire_formats.md) for packet
boundaries.

The internal `sc-neurocore-safety` crate disables registry publication with
`publish = false`. Its neuron APIs remain available, but it does not export
the placeholder modules `aer_codec`, `codec`, `delta_codec`, `entropy`,
`predictive_codec`, `streaming_codec` or `waveform_codec`. Their `.rs` files remain
in `src/sc_neurocore/accel/rust/safety/` as unsupported source and are not compiled
into the crate. Use the production engine for real Rust codec components.

The following source mirrors also remain unsupported as codec backends:

| Language | Retained source area | Qualification boundary |
|---|---|---|
| Go | `accel/go/services/{registry,streaming_codec,waveform_codec}/` | Empty factory/codec functions; waveform configuration validation alone does not encode or decode packets. |
| Julia | `accel/julia/spike_codec/*.jl` | Algorithm transcripts, including entropy and registry helpers; no maintained codec loader or executed packet parity. |
| Mojo | `accel/mojo/kernels/*codec.mojo` and `entropy.mojo` | Algorithm transcripts returning scalar placeholders; no maintained codec loader or executed packet parity. |

Paths in the table resolve under `src/sc_neurocore/`. Constants, validator
definitions and static source-marker tests do not qualify a working codec.
Only a real maintained implementation, exercised through its public loader with
packet parity and measured benchmarks, can establish codec support.

## Current authoritative Julia entrypoints

The maintained Julia surface is currently limited to files explicitly loaded by
Python code or referenced by exercised parity paths:

- `src/sc_neurocore/accel/julia/world_model/predictive_model.jl`
- `src/sc_neurocore/accel/julia/chiplet/kl_refine.jl`
- `src/sc_neurocore/accel/julia/_native/learning_bridge.jl`
- `src/sc_neurocore/accel/julia/fault_injection/fault_injection.jl`

Package-level reference:

- `sc_neurocore.accel.julia.AUTHORITATIVE_JULIA_ENTRYPOINTS`

## Autonomous-learning polyglot chain

Autonomous learning is a maintained C-FFI chain rather than an algorithm
transcript mirror:

- authority: `crates/autonomous_learning/src/`
- Python facade: `src/sc_neurocore/_native/learning_bridge.py` and its focused
  `learning_*` implementation modules
- Go adapter: `src/sc_neurocore/accel/go/autonomous_learning/learning_bridge.go`
- Julia adapter: `src/sc_neurocore/accel/julia/_native/learning_bridge.jl`

Python and Julia accept the exact artifact through
`SC_NEUROCORE_LIB_PATH`; Go links and loads that artifact from its parent
directory. Live parity tests exercise invalid-domain rejection and compare
learned weights across the three language adapters. The source-bound benchmark
also executes Torch, scalar Rust, batched Rust, and Rayon state paths.

The deleted `src/sc_neurocore/accel/rust/safety/learning_bridge.rs` file was a
non-dispatched transcript and is not a backend. Do not recreate a second Rust
implementation in the safety-mirror crate. Add rule behavior and ABI changes to
the authoritative `crates/autonomous_learning` crate, then update every
maintained adapter and parity test in the same change.

## Current authoritative Mojo entrypoints

The maintained Mojo surface is currently limited to Python loaders and compiled
library paths that are actually consumed from Python:

- `src/sc_neurocore/accel/mojo/runner.py`
- `src/sc_neurocore/accel/mojo/world_model/lgssm.mojo`
- `src/sc_neurocore/accel/mojo/fault_injection/fault.mojo`
- `src/sc_neurocore/accel/mojo/wong_wang/__init__.py`
- `src/sc_neurocore/accel/mojo/wilson_cowan/__init__.py`

Package-level reference:

- `sc_neurocore.accel.mojo.AUTHORITATIVE_MOJO_ENTRYPOINTS`

## Non-authoritative mirror zones

These areas must not be treated as the source of truth unless a maintained
Python loader and tests are added later:

- `src/sc_neurocore/accel/julia/studio/*.jl`
- `src/sc_neurocore/accel/julia/analysis/*.jl`
- `src/sc_neurocore/accel/julia/analysis_spike_stats/*.jl`
- large parts of `src/sc_neurocore/accel/mojo/kernels/*.mojo`

Known transcript-style examples:

- `src/sc_neurocore/accel/julia/studio/compiler.jl`
- `src/sc_neurocore/accel/mojo/kernels/app.mojo`

## Practical workflow

When fixing behaviour:

1. change the authoritative Python source or maintained compiled backend first
2. update tests
3. update docs
4. only then refresh any mirrors if they are still worth keeping

Do not reverse that order. Updating a mirror first creates drift and false
confidence.
