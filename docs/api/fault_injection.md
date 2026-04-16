# Fault Injection

Radiation-grade fault injection for resilience testing. Models SEU, MBU,
and stuck-at faults with configurable radiation profiles (LEO, GEO, terrestrial).

## Quick Start

```python
from sc_neurocore.fault_injection.fault_injection import (
    FaultInjector, FaultModel, RadiationProfile, ResilienceBenchmark,
)
```

::: sc_neurocore.fault_injection.fault_injection
