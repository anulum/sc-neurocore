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

## Capacity, metering and priority

- A region fits a tenant when it has at least the neurons and the synapses of
  the tenant's QoS (`HWRegion.fits`); allocation and migration both use it.
  `migrate` refuses a busy target, a target too small and the tenant's own
  region, before anything moves.
- `BandwidthMeter.throughput` is spikes per cycle over `window_cycles`: the
  spikes of the samples after the window's base, over the cycles from the base
  to the last sample; 0.0 with one sample or no span. `throughput_mbps` and
  `exceeds_quota` take `bits_per_spike` and `clock_hz` from the caller; the
  units and the limit refuse a bool.
- `BandwidthMeter.record` refuses a cycle that is not later than the
  tenant's last, with `ValueError` and no change to the history: the window
  and the span run along increasing cycles. Because they increase, no later
  window reads a sample before the current window's base, so `record` drops
  those samples. A tenant's history stays within one window and its base, and
  every rate equals the whole history's. Set `window_cycles` before
  recording; enlarging it later cannot recover dropped samples.
- Under the priority policy, the realtime tenants share one half of the cycles
  equally; the others keep an equal share each, in priority order.

::: sc_neurocore.hypervisor.hypervisor
