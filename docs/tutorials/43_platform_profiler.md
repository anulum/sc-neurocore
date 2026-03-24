<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 43: Cross-Platform Performance Profiling

Which platform should you run your SNN on? The profiler compares CPU,
Rust, and multiple FPGA targets, showing latency, throughput, power,
and energy — sorted by efficiency.

## 1. Compare All Platforms

```python
from sc_neurocore.profiler import compare
from sc_neurocore.profiler.platform_profiler import format_table

results = compare(layer_sizes=[(16, 8), (8, 4)], bitstream_length=256)
print(format_table(results))
```

## 2. Specific Platforms

```python
results = compare(
    layer_sizes=[(64, 32), (32, 10)],
    platforms=["python", "fpga_artix7"],
)
for r in results:
    print(f"{r.platform}: {r.energy_per_inf_nj:.2f} nJ, {r.latency_ms:.3f} ms")
```

## 3. Interpret Results

FPGA wins on energy by orders of magnitude for small networks.
CPU wins on flexibility and development speed. Rust bridges the gap.

## Further Reading

- [Tutorial 41: Energy Estimation](41_energy_estimation.md)
- [API: Profiler](../api/profiler.md)
