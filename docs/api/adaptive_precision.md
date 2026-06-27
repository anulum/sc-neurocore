# Adaptive Precision

Per-layer adaptive bitstream length and per-synapse bit-width planning for
mixed-precision SC networks.

- `assign_lengths` — auto-select bitstream length per layer from
  Hoeffding or sensitivity bounds.
- `assign_synapse_precisions` — choose integer bit widths and stochastic
  bitstream lengths per synapse.
- `auto_tune_synapse_precisions` — emit the public precision-plan manifest for a
  percent error target.
- `write_precision_formal_evidence_bundle` — write bounded SVA/SBY/JSON evidence
  scaffolding for a precision plan.

```python
from sc_neurocore.compiler.adaptive_precision import assign_lengths
```

`sc_neurocore.compiler.adaptive_precision` is in the scoped public-docstring
policy. Its dedicated adaptive-precision tests are strict typed and cover
layer-length assignment, sensitivity analysis, per-synapse planning, manifest
stability, and formal-evidence bundle writing at 100% isolated facade coverage.
The touched surface is a Python compiler facade; no polyglot or benchmark
counterpart changes were required for this docstring-policy slice.

## 2026-04-30 per-synapse precision plan

The adaptive precision module now includes a conservative per-synapse planner
for the roadmap auto-adaptive precision optimiser. It assigns integer
`bit_width`, SC `bitstream_length`, sensitivity, quantisation-error bound,
stochastic-error bound, and total bound for each synapse:

```python
import numpy as np

from sc_neurocore.compiler.adaptive_precision import (
    assign_synapse_precisions,
    precision_plan_manifest,
)

weights = [np.array([[0.1, 0.8], [0.0, 0.4]])]
plan = assign_synapse_precisions(weights, target_error=0.05)
manifest = precision_plan_manifest(plan)
```

This is a deterministic planning surface, not a training-result claim. Bounds
are intentionally conservative: quantisation is bounded by half an integer
step scaled by sensitivity, and stochastic sampling uses the existing Hoeffding
bitstream-length helper. Custom sensitivity maps can be supplied after an
external sensitivity-analysis pass.

## Adaptive runtime precision BFP metadata

`compile_adaptive_precision(...)` accepts fixed Q-format strings such as
`Q8.8`/`Q16.16` and block-floating strings such as `BFP16E3X32`.  When a
block-floating precision is supplied with `lp_parameter_count` or
`hp_parameter_count`, the generated manifest records the exact
`block_exponent_layout`: flattened row-major parameter count, block size,
exponent-vector length, and final partial-block size.  Invalid negative
parameter counts fail before RTL emission.

This metadata is a compiler contract for downstream emitters.  The generated
adaptive wrapper still emits fixed mantissa-width datapaths; shared exponents
remain explicit metadata until the target-specific BFP datapath is selected.
Every adaptive manifest carries `adaptive_precision_emitter.v1`, explicit
`emitted_datapath_width`, `emitted_datapath_fraction`,
`exponent_stream_width`, and `exponent_vector_width` fields. Fixed Q-format
paths set the exponent widths to zero and reject accidental
`*_parameter_count` inputs so a block-exponent layout cannot be silently
dropped before HDL/Rust emitter handoff.

::: sc_neurocore.compiler.adaptive_precision
    options:
      show_root_heading: true
