# Hypervisor

Multi-tenant neuromorphic hypervisor. Manages isolated workloads on shared
FPGA resources with resource quotas, priority preemption, and health monitoring.

## Quick Start

```python
from sc_neurocore.hypervisor.hypervisor import (
    HWRegion, Hypervisor, HypervisorConfig, QoSPolicy, Tenant,
)
```

::: sc_neurocore.hypervisor.hypervisor
