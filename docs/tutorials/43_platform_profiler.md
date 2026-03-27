<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 43: Cross-Platform Performance Profiling

Which platform should you run your SNN on? The profiler compares
Python (NumPy), Rust engine, and multiple FPGA targets in one call,
producing a ranked table of latency, throughput, power, and energy
per inference.

## The Decision Problem

You've trained an SNN. Now where do you deploy it?

| Platform | Latency | Power | Dev Time | Best For |
|----------|---------|-------|----------|----------|
| Python (NumPy) | ~10 ms | ~15 W (CPU) | Minutes | Prototyping |
| Rust engine | ~0.1 ms | ~15 W (CPU) | Minutes | Production server |
| FPGA (ice40) | ~0.02 ms | ~5 mW | Hours | Ultra-low power edge |
| FPGA (Artix-7) | ~0.005 ms | ~50 mW | Hours | High throughput |

The profiler measures all of these for your specific network and
produces a recommendation.

## Compare All Platforms

```python
from sc_neurocore.profiler import compare
from sc_neurocore.profiler.platform_profiler import format_table

results = compare(
    layer_sizes=[(784, 128), (128, 10)],
    bitstream_length=256,
)
print(format_table(results))
# Platform        Latency     Throughput    Power     Energy/Inf
# ─────────────────────────────────────────────────────────────
# fpga_ice40      0.021 ms    47,619 inf/s  4.6 mW   98 nJ
# fpga_ecp5       0.021 ms    47,619 inf/s  5.8 mW   121 nJ
# rust            0.082 ms    12,195 inf/s  15.0 W    1,230,000 nJ
# fpga_artix7     0.021 ms    47,619 inf/s  7.2 mW   151 nJ
# python          12.9 ms     78 inf/s      15.0 W    193,500,000 nJ
```

The table is sorted by energy efficiency. FPGA wins by ~12,000× over
Python and ~12× over Rust for energy per inference.

## Specific Platforms

Compare only the platforms you're considering:

```python
results = compare(
    layer_sizes=[(64, 32), (32, 10)],
    platforms=["python", "rust", "fpga_artix7"],
    bitstream_length=128,
)
for r in results:
    print(f"{r.platform:15s}: {r.energy_per_inf_nj:>10.2f} nJ, "
          f"{r.latency_ms:>8.3f} ms, {r.throughput_per_s:>10,} inf/s")
```

## Scaling Analysis

How does each platform scale with network size?

```python
import matplotlib
# (or just print the table)

for n_hidden in [32, 64, 128, 256, 512]:
    layers = [(784, n_hidden), (n_hidden, 10)]
    results = compare(layers, platforms=["rust", "fpga_ice40"])
    rust = next(r for r in results if r.platform == "rust")
    fpga = next(r for r in results if r.platform == "fpga_ice40")
    print(f"Hidden={n_hidden:>3d}: Rust={rust.latency_ms:.3f}ms, "
          f"FPGA={fpga.latency_ms:.3f}ms, "
          f"FPGA energy advantage={rust.energy_per_inf_nj / fpga.energy_per_inf_nj:.0f}×")
```

At small network sizes (hidden≤64), FPGA advantage is ~10,000×.
At large sizes (hidden≥256), FPGA may not fit on iCE40 — switch to
ECP5 or Artix-7.

## Interpreting Results

**Choose FPGA when:**
- Energy budget < 1 mW (battery, implant, sensor node)
- Latency requirement < 100 µs (real-time control)
- Network fits on target device

**Choose Rust when:**
- Network is too large for FPGA
- Need flexibility (change network without resynthesis)
- Server deployment with high throughput

**Choose Python when:**
- Prototyping and experimentation
- Integration with other Python libraries
- Training (FPGA can't train)

## Integration with Studio

The Studio's Synthesis Dashboard shows FPGA resource usage. The
energy estimate is available via the **Estimate** button (no Yosys
needed). For a full platform comparison, run the profiler from Python.

## References

- Sze et al. (2017). "Efficient Processing of Deep Neural Networks:
  A Tutorial and Survey." Proceedings of the IEEE 105(12):2295-2329.
- Horowitz (2014). "Computing's Energy Problem (and what we can do
  about it)." ISSCC 2014 Keynote.
