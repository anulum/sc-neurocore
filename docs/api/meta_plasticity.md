# Meta-Plasticity

Self-modifying plasticity rules. A threshold controller and an evolver
change STDP, short-term, homeostatic and bitstream parameters from
performance metrics the caller supplies.

## Quick Start

```python
from sc_neurocore.meta_plasticity.meta_plasticity import (
    MetaPlasticityEngine, PlasticityRuleSet, RuleEvolver,
    SleepPhase,
)
```

::: sc_neurocore.meta_plasticity.meta_plasticity
