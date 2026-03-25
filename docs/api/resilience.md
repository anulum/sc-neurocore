# Fault Resilience

Systematic fault injection and resilience analysis for SNN deployments.

- `FaultInjector` — Inject stuck-at faults, bit-flips, neuron death, synapse dropout. Measure accuracy degradation curves.
- SC vs fixed-point comparison: SC inherent noise tolerance gives 2-10x better resilience.

```python
from sc_neurocore.resilience import FaultInjector
```

See [Tutorial 63: Fault Resilience](../tutorials/63_fault_resilience.md).

::: sc_neurocore.resilience
    options:
      show_root_heading: true
