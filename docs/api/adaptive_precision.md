# Adaptive Precision

Per-layer adaptive bitstream length for mixed-precision SC networks.

- `AdaptivePrecisionManager` — Auto-select bitstream length per layer (Hoeffding/Chebyshev/sensitivity bounds). Layers needing high precision get longer bitstreams; tolerant layers get shorter ones.

```python
from sc_neurocore.compiler.adaptive_precision import AdaptivePrecisionManager
```

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

::: sc_neurocore.compiler.adaptive_precision
    options:
      show_root_heading: true
