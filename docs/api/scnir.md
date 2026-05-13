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
encoding, fixed-point precision, random-source metadata, and stream correlation
constraints before a model reaches hardware compilation or experiment
handoff.

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
sc-neurocore.scnir.v0.1
```

Each stream entry must provide:

| Field | Purpose |
|---|---|
| `stream_id` | Stable identifier used by correlation constraints |
| `layer` | Producer layer or graph node name |
| `bitstream_length` | Positive integer SC stream length |
| `encoding` | `unipolar`, `bipolar`, low-discrepancy, replay, LFSR, or hardware-source encoding |
| `precision` | Signedness, total bits, fractional bits, accumulator bits, rounding, overflow |
| `source` | LFSR, Sobol, Halton, replay, or hardware source metadata |
| `correlation_constraints` | Pairwise policy metadata between streams |

## CLI

Validate a document:

```bash
sc-neurocore scnir validate model.scnir.json
```

Export SC-NIR metadata from a NIR graph:

```bash
sc-neurocore scnir export model.nir --output model.scnir.json --T 1024
```

Exit code `0` means the document passed the SC-NIR validator. Exit code `1`
means validation/export failed or the input could not be read.

## FPGA Compilation Integration

`compile_network_to_fpga(...)` constructs SC-NIR metadata for the lowered
`NeuronGraph` before emitting top-level RTL. The returned
`NetworkCompilationResult` exposes `scnir_document`, and the generated top
module includes deterministic handoff localparams:

```verilog
localparam integer SCNIR_BITSTREAM_LENGTH = 1024;
localparam integer SCNIR_STREAM_COUNT = 2;
```

These localparams provide the stable boundary for follow-on HDL source-generator
work, where LFSR/Sobol source instances will consume the same SC-NIR metadata
directly.

## Python API

```python
from sc_neurocore.ir import (
    SCNIRConversionConfig,
    SCNIRDocument,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
    build_scnir_from_neuron_graph,
    export_scnir_from_nir,
    load_scnir,
    validate_scnir_dict,
    write_scnir,
)
```

::: sc_neurocore.ir.scnir_schema
    options:
      show_root_heading: true
      members:
        - SCNIR_SCHEMA_VERSION
        - SCNIRValidationError
        - SCNIRPrecision
        - SCNIRSource
        - SCNIRCorrelationConstraint
        - SCNIRStream
        - SCNIRDocument
        - validate_scnir_dict
        - scnir_from_dict
        - scnir_to_dict
        - load_scnir
        - write_scnir

::: sc_neurocore.ir.scnir_convert
    options:
      show_root_heading: true
      members:
        - SCNIRConversionConfig
        - build_scnir_from_neuron_graph
        - export_scnir_from_nir
