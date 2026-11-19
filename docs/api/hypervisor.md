# Hypervisor

Multi-tenant neuromorphic hypervisor. Manages isolated workloads on shared
FPGA resources with resource quotas, priority preemption, and health monitoring.

## Quick Start

```python
from sc_neurocore.hypervisor.hypervisor import (
    HWRegion, Hypervisor, HypervisorConfig, QoSPolicy, Tenant,
)
```

## Isolation

- A tenant holds at most one region: `allocate` for a tenant that already holds
  one returns that region's id and takes nothing more.
- `Hypervisor.preempt(victim_id, preemptor_id, cycle)` hands the victim's region
  to a preemptor that holds none, saves the victim's state, and moves the
  firewall rule with the region, so the victim can no longer reach those
  addresses. It returns None and changes nothing otherwise.
- The rule moves through `PreemptionManager.preempt(..., firewall=...)`: every
  victim rule that shares any address with the region is removed whole, and
  the preemptor gets one rule for the region. A victim rule that reaches across
  the region's edge loses its part outside the region too; such a rule already
  breaks `verify_isolation`, and removing it denies access rather than grants it.
- `verify_isolation(firewall, regions)` reports rules of two tenants that share
  an address and, against the region model, overlapping or empty regions and
  any rule that does not lie in a region its tenant holds.

::: sc_neurocore.hypervisor.hypervisor
