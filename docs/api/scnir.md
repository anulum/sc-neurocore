<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SC-NIR API documentation -->

# SC-NIR API

SC-NIR is the SC-NeuroCore metadata layer for stochastic-computing semantics
that plain NIR does not encode. It records bitstream length, stochastic
encoding, stream signal kind, fixed-point precision, deterministic transform
metadata, random-source metadata, stream correlation constraints, and explicit
hierarchy instance boundaries before a model reaches hardware compilation or
experiment handoff.

The schema is intentionally strict. Unknown fields, missing fields, duplicate
stream identifiers, invalid random-source metadata, dangling correlation
references, and hierarchy ports that do not match an existing stream are
rejected rather than ignored.

## JSON Schema

The reference schema is tracked at:

```text
schemas/scnir/scnir.schema.json
```

Current schema version:

```text
sc-neurocore.scnir.v0.6
```

Each stream entry must provide:

| Field | Purpose |
|---|---|
| `stream_id` | Stable identifier used by correlation constraints |
| `layer` | Producer layer or graph node name |
| `bitstream_length` | Positive integer SC stream length |
| `encoding` | `unipolar`, `bipolar`, low-discrepancy, replay, LFSR, or hardware-source encoding |
| `signal_kind` | Logical stream role: `spike`, `analogue_state`, or `weight` |
| `delay_steps` | Explicit unit-delay count as a scalar integer or exact source-width integer vector; feed-forward streams use `0` |
| `transforms` | Ordered deterministic transforms applied to the stream, currently threshold comparators |
| `precision` | Signedness, total bits, fractional bits, accumulator bits, rounding, overflow |
| `source` | LFSR, Sobol, Halton, replay, or hardware source metadata |
| `correlation_constraints` | Pairwise policy metadata between streams |

The top-level `hierarchy` array records bounded nested-hardware contracts. Each
entry must provide a stable `instance_id`, synthesisable `module_name`, and at
least one port. Each port records `port_name`, `direction`, `stream_id`,
`signal_kind`, and `bit_width`; the referenced stream must exist and its
`signal_kind` must match the port. Single-input/single-output nested NIR graphs
that are inlined for flat HDL lowering preserve their original boundary as a
hierarchy instance, with ports generated from the SC-NIR streams produced by
the inlined contents.

## CLI

Validate a document:

```bash
sc-neurocore scnir validate model.scnir.json
```

Upgrade a supported document to the current canonical schema:

```bash
sc-neurocore scnir upgrade model.scnir.json --output upgraded.scnir.json
```

The `sc-neurocore.scnir.v0.1` upgrade path adds explicit `delay_steps=0` to
legacy streams, every pre-`v0.3` upgrade adds `signal_kind` inferred from the
stream identifier, and every pre-`v0.4` upgrade adds an explicit empty
`transforms` list. Version `v0.5` keeps scalar delay metadata compatible and
adds per-source delay vectors for heterogeneous NIR `Delay` lowering. Version
`v0.6` adds top-level hierarchy instance and port metadata; legacy documents
upgrade with an empty hierarchy list. The typed validator and deterministic JSON
writer then produce `sc-neurocore.scnir.v0.6`. Unknown schema versions fail
closed so migration support must be added deliberately when the schema evolves.

Export SC-NIR metadata from a NIR graph:

```bash
sc-neurocore scnir export model.nir --output model.scnir.json --T 1024
```

Exit code `0` means the document passed the SC-NIR validator. Exit code `1`
means validation, upgrade, export, or input loading failed.

## FPGA Compilation Integration

`compile_network_to_fpga(...)` constructs SC-NIR metadata for the lowered
`NeuronGraph` before emitting top-level RTL. The returned
`NetworkCompilationResult` exposes `scnir_document`,
`scnir_source_modules`, and `scnir_source_manifest`. The generated top module
includes deterministic handoff localparams:

```verilog
localparam integer SCNIR_BITSTREAM_LENGTH = 1024;
localparam integer SCNIR_STREAM_COUNT = 2;
localparam integer SCNIR_SOURCE_MODULE_COUNT = 2;
```

`scnir_source_modules` is keyed by emitted Verilog module name and contains the
standalone LFSR-16 or Sobol-16 source RTL generated from each SC-NIR stream.
The `compile-nir` CLI writes the full validated document as
`scnir_document.json` in the output directory so dense exported-network runs
can be reproduced from the same stream metadata that drove source generation.
`scnir_source_manifest` records the stream identifier, module name, source
family, seed, bitstream length, encoding, signal kind, recurrent delay steps,
precision, transform metadata, and source-specific metadata used for each
module. CLI manifests also record the selected interconnect, Q-format, total
neuron count, total synapse count, SC-NIR stream count,
`scnir_signal_kinds` counts, and `scnir_signal_routes` so AER/event-driven
and mixed analogue/spiking output
directories carry machine-readable compile evidence. FPGA compilation
marks non-spiking LI/CubaLI/integrator population streams as
`analogue_state`, so mixed analogue/spiking NIR graphs expose voltage-state
handoff metadata instead of being mislabeled as spike streams; combined mixed
AER graphs record analogue-state streams as direct MAC routes and spike streams
as weighted event routes. FPGA
compilation currently materialises LFSR-16 and Sobol-16 source families
because both expose the standard `threshold[15:0]`/`bit_out` contract;
unsupported source families fail closed instead of being emitted through an
incompatible HDL interface.

## NIR Primitive Compatibility

The executable compatibility matrix is exposed by
`scnir_compatibility_matrix()` and checked against the parser's declared
`NODE_MAP` support. It separates parser execution from SC-NIR/FPGA handoff so
documentation cannot claim hardware closure for primitives that are currently
parser-only or only closed under a bounded shape/port contract.

| NIR primitive | SC-NIR / FPGA level | Stream metadata | HDL handoff |
|---|---|---|---|
| `Input`, `Output` | boundary | none | external input/output buses |
| `LIF`, `IF`, `CubaLIF` | metadata and HDL | `signal_kind=spike`, `encoding=unipolar` | canonical ODE module plus direct or AER interconnect |
| `LI`, `CubaLI`, `I` | metadata and HDL | `signal_kind=analogue_state`, `encoding=bipolar` | canonical ODE or integrator state-update module with direct analogue-state MAC routing |
| `Affine`, `Linear` | metadata and HDL | `signal_kind=weight`, `encoding=bipolar`; recurrent or explicit delayed streams carry `delay_steps` | weight ROM plus direct or weighted-event interconnect |
| `Scale` | metadata and HDL when adjacent to `Affine`/`Linear` | folded into the downstream weight stream as connection gain | folded fixed-point gain in direct/AER weight terms |
| `Flatten` | metadata and HDL when exact shape metadata preserves element count adjacent to `Affine`/`Linear` | folded into the downstream weight stream as `shape_preserving_flatten` | fixed-point weight indexing with exact flattened width checks |
| `Threshold` | metadata and HDL when adjacent to `Affine`/`Linear` with scalar or exact-width thresholds | weight stream carries a `threshold` transform with `source` or `destination` position | fixed-point comparator before weighted-event contribution or destination current |
| `Delay` | metadata and HDL for scalar or exact source-width source-side delays feeding `Affine`/`Linear` population connections | downstream weight stream carries scalar `delay_steps>=0` or vector `delay_steps=[...]` | direct-interconnect register chains with per-source delay taps for spike and analogue-state sources |
| `Conv1d` | metadata and HDL when `input_shape` is explicit and output is flattened into a destination population | lowered to a `signal_kind=weight` stream as `convolution_lowered_weight` | dense Toeplitz-style fixed-point MAC terms through the weight path |
| `Conv2d` | metadata and HDL when exact spatial input shape is explicit and output is flattened into a destination population | lowered to a `signal_kind=weight` stream as `convolution_lowered_weight` | dense 2D convolution fixed-point MAC terms through the weight path |
| `SumPool2d`, `AvgPool2d` | metadata and HDL when exact CHW shape metadata is present and output is flattened into a destination population | lowered to a `signal_kind=weight` stream as `pool2d_lowered_weight` | dense pooling fixed-point MAC terms through the weight path |
| nested `NIRGraph` | metadata and HDL for single-input/single-output subgraphs inlined into the parent graph; multi-port hierarchy fails closed | namespaced stream IDs from the inlined subgraph contents plus a top-level hierarchy instance whose ports reference those streams | namespaced inline fixed-point terms; standalone hierarchical submodule handoff remains open |

Use `validate_scnir_compatibility_matrix()` in tests or release checks to fail
when parser support changes without a corresponding compatibility row. Use
`build_scnir_compatibility_audit()` for release evidence bundles that need the
validated matrix, support-level counts, and the exact evidence file set in one
versioned JSON object.

## Python API

```python
from sc_neurocore.ir import (
    SCNIR_HDL_HANDOFF_MANIFEST_VERSION,
    SCNIR_PREVIOUS_SCHEMA_VERSION,
    SCNIR_SCHEMA_VERSION,
    SCNIR_COMPATIBILITY_AUDIT_VERSION,
    SCNIRCompatibilityRow,
    SCNIRConversionConfig,
    SCNIRDocument,
    SCNIRHierarchyInstance,
    SCNIRHierarchyPort,
    SCNIRPrecision,
    SCNIRSignalKind,
    SCNIRSource,
    SCNIRStream,
    build_scnir_compatibility_audit,
    build_scnir_source_bundle,
    build_scnir_from_neuron_graph,
    export_scnir_from_nir,
    load_scnir,
    scnir_compatibility_matrix,
    scnir_compatibility_matrix_dicts,
    validate_scnir_compatibility_matrix,
    validate_scnir_dict,
    write_scnir,
    upgrade_scnir_dict,
)
```

::: sc_neurocore.ir.scnir_schema
    options:
      show_root_heading: true
      members:
        - SCNIR_SCHEMA_VERSION
        - SCNIR_PREVIOUS_SCHEMA_VERSION
        - SCNIRValidationError
        - SCNIRPrecision
        - SCNIRSource
        - SCNIRSignalKind
        - SCNIRCorrelationConstraint
        - SCNIRHierarchyPort
        - SCNIRHierarchyInstance
        - SCNIRStream
        - SCNIRDocument
        - validate_scnir_dict
        - scnir_from_dict
        - scnir_to_dict
        - load_scnir
        - write_scnir
        - upgrade_scnir_dict

::: sc_neurocore.ir.scnir_convert
    options:
      show_root_heading: true
      members:
        - SCNIRConversionConfig
        - build_scnir_from_neuron_graph
        - export_scnir_from_nir

::: sc_neurocore.ir.scnir_compatibility
    options:
      show_root_heading: true
      members:
        - SCNIRCompatibilityRow
        - SCNIR_COMPATIBILITY_AUDIT_VERSION
        - scnir_compatibility_matrix
        - scnir_compatibility_matrix_dicts
        - build_scnir_compatibility_audit
        - validate_scnir_compatibility_matrix

::: sc_neurocore.ir.scnir_hdl
    options:
      show_root_heading: true
      members:
        - SCNIRHDLSourceManifestEntry
        - SCNIRHDLSourceBundle
        - build_scnir_source_bundle

::: sc_neurocore.ir.scnir_handoff_audit
    options:
      show_root_heading: true
      members:
        - SCNIR_HDL_HANDOFF_MANIFEST_VERSION
        - SCNIRHDLHandoffAuditReport
        - audit_scnir_hdl_handoff
        - write_scnir_hdl_handoff_audit
