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
metadata, random-source metadata, and stream correlation constraints before a
model reaches hardware compilation or experiment handoff.

The schema is intentionally strict. Unknown fields, missing fields, duplicate
stream identifiers, invalid random-source metadata, and dangling correlation
references are rejected rather than ignored.

## JSON Schema

The reference schema is tracked at:

```text
schemas/scnir/scnir.schema.json
```

Current schema version:

```text
sc-neurocore.scnir.v0.4
```

Each stream entry must provide:

| Field | Purpose |
|---|---|
| `stream_id` | Stable identifier used by correlation constraints |
| `layer` | Producer layer or graph node name |
| `bitstream_length` | Positive integer SC stream length |
| `encoding` | `unipolar`, `bipolar`, low-discrepancy, replay, LFSR, or hardware-source encoding |
| `signal_kind` | Logical stream role: `spike`, `analogue_state`, or `weight` |
| `delay_steps` | Explicit unit-delay count for recurrent streams; feed-forward streams use `0` |
| `transforms` | Ordered deterministic transforms applied to the stream, currently threshold comparators |
| `precision` | Signedness, total bits, fractional bits, accumulator bits, rounding, overflow |
| `source` | LFSR, Sobol, Halton, replay, or hardware source metadata |
| `correlation_constraints` | Pairwise policy metadata between streams |

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
`transforms` list. The typed validator and deterministic JSON writer then
produce `sc-neurocore.scnir.v0.4`. Unknown schema versions fail closed so
migration support must be added deliberately when the schema evolves.

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
parser-only.

| NIR primitive | SC-NIR / FPGA level | Stream metadata | HDL handoff |
|---|---|---|---|
| `Input`, `Output` | boundary | none | external input/output buses |
| `LIF`, `IF`, `CubaLIF` | metadata and HDL | `signal_kind=spike`, `encoding=unipolar` | canonical ODE module plus direct or AER interconnect |
| `LI`, `CubaLI`, `I` | metadata and HDL | `signal_kind=analogue_state`, `encoding=bipolar` | canonical ODE or integrator state-update module with direct analogue-state MAC routing |
| `Affine`, `Linear` | metadata and HDL | `signal_kind=weight`, `encoding=bipolar`; recurrent or explicit delayed streams carry `delay_steps` | weight ROM plus direct or weighted-event interconnect |
| `Scale` | metadata and HDL when adjacent to `Affine`/`Linear` | folded into the downstream weight stream as connection gain | folded fixed-point gain in direct/AER weight terms |
| `Flatten` | metadata and HDL when exact shape metadata preserves element count adjacent to `Affine`/`Linear` | folded into the downstream weight stream as `shape_preserving_flatten` | fixed-point weight indexing with exact flattened width checks |
| `Threshold` | metadata and HDL when adjacent to `Affine`/`Linear` with scalar or exact-width thresholds | weight stream carries a `threshold` transform with `source` or `destination` position | fixed-point comparator before weighted-event contribution or destination current |
| `Delay` | metadata and HDL for homogeneous source-side delays feeding `Affine`/`Linear` population connections | downstream weight stream carries `delay_steps>=0` | direct-interconnect register chain for spike and analogue-state sources |
| `Conv1d` | metadata and HDL when `input_shape` is explicit and output is flattened into a destination population | lowered to a `signal_kind=weight` stream as `convolution_lowered_weight` | dense Toeplitz-style fixed-point MAC terms through the weight path |
| `Conv2d` | metadata and HDL when exact spatial input shape is explicit and output is flattened into a destination population | lowered to a `signal_kind=weight` stream as `convolution_lowered_weight` | dense 2D convolution fixed-point MAC terms through the weight path |
| `SumPool2d`, `AvgPool2d` | metadata and HDL when exact CHW shape metadata is present and output is flattened into a destination population | lowered to a `signal_kind=weight` stream as `pool2d_lowered_weight` | dense pooling fixed-point MAC terms through the weight path |
| nested `NIRGraph` | parser only; SC-NIR/FPGA lowering fails closed | none | nested execution exists; hierarchical SC-NIR/HDL handoff remains open |

Use `validate_scnir_compatibility_matrix()` in tests or release checks to fail
when parser support changes without a corresponding compatibility row.

## Python API

```python
from sc_neurocore.ir import (
    SCNIR_PREVIOUS_SCHEMA_VERSION,
    SCNIR_SCHEMA_VERSION,
    SCNIRCompatibilityRow,
    SCNIRConversionConfig,
    SCNIRDocument,
    SCNIRPrecision,
    SCNIRSignalKind,
    SCNIRSource,
    SCNIRStream,
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
        - scnir_compatibility_matrix
        - scnir_compatibility_matrix_dicts
        - validate_scnir_compatibility_matrix

::: sc_neurocore.ir.scnir_hdl
    options:
      show_root_heading: true
      members:
        - SCNIRHDLSourceManifestEntry
        - SCNIRHDLSourceBundle
        - build_scnir_source_bundle
