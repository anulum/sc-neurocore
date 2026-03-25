# World Model

Predictive state transition models for spike-domain planning.

- `PredictiveWorldModel` — Learns `state_next = f(state, action)` in probability domain. Linear transition matrix, clip to [0,1]. Stub for future learnable world models.
- `SCPlanner` — Action selection via predictive rollouts using the world model.

```python
from sc_neurocore.world_model import PredictiveWorldModel, SCPlanner
```

::: sc_neurocore.world_model
    options:
      show_root_heading: true
