# Multi-Timescale

Multi-timescale SNN: per-synapse time constants and multi-clock scheduling.

- `MultiTimescaleNetwork` — Different layers/populations operate at different timescales (fast sensory: 5ms, working memory: 200ms, deep context: 10s). Scheduler coordinates updates across timescales.

Biological brains have timescales spanning 5 orders of magnitude. This module enables that in simulation.

```python
from sc_neurocore.temporal_hierarchy import MultiTimescaleNetwork
```

See [Tutorial 69: Multi-Timescale](../tutorials/69_multi_timescale.md).

::: sc_neurocore.temporal_hierarchy
    options:
      show_root_heading: true
