# Hypervisor

Multi-tenant neuromorphic hypervisor. Manages isolated workloads on shared
FPGA resources with resource quotas, priority preemption, and health monitoring.

## Quick Start

```python
from sc_neurocore.hypervisor.hypervisor import (
    Hypervisor, Tenant, ResourceAllocator, HealthMonitor,
)
```

::: sc_neurocore.hypervisor.hypervisor
