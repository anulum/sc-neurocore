# Explainability

Bitstream-level explainability and causal attribution for SC decisions.
Anchored to formal verification properties.

## Replay Guardrails

`ExplainabilityEngine.explain_spike` and `ExplainabilityEngine.replay_bitstream`
validate deterministic replay inputs before allocating streams or mutating
provenance. `threshold_q16` must be an integer in `[0, 65535]`, replay lengths
must be positive integers, and `spike_threshold_count` must be between `0` and
the replayed bitstream length. Rejected inputs raise `ValueError` before any
decision node or provenance step is recorded.

## Quick Start

```python
from sc_neurocore.explainability.explainability import (
    ExplainabilityEngine, CausalAttributor, FormalPropertyLink,
)
```

::: sc_neurocore.explainability.explainability
