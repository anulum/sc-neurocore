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

Exit code `0` means the document passed the SC-NIR validator. Exit code `1`
means it failed validation or could not be read.

## Python API

```python
from sc_neurocore.ir import (
    SCNIRDocument,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
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
