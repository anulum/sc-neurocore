<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 41: Pre-Silicon Energy Estimation

Predict FPGA resource usage, power consumption, and energy per inference
in under one second — before running any synthesis tool.

## Why This Matters

Every SNN paper claims energy efficiency, but most use estimates or omit
hardware numbers entirely. SC-NeuroCore's energy estimator gives you
concrete numbers for your specific network on your specific FPGA target,
instantly. No Vivado. No synthesis. Just answers.

## 1. Basic Estimation

```python
from sc_neurocore.energy import estimate

report = estimate(
    layer_sizes=[(16, 8), (8, 4)],
    target="ice40",
    bitstream_length=256,
)
print(report.summary())
```

## 2. Compare FPGA Targets

```python
for target in ["ice40", "ecp5", "artix7", "zynq"]:
    r = estimate([(16, 8), (8, 4)], target=target)
    print(f"{target}: {r.total_luts:,} LUTs, {r.utilization_pct:.0f}%, "
          f"{r.total_dynamic_power_mw:.2f} mW")
```

## 3. Event-Driven vs Clock-Driven

```python
clock = estimate([(16, 8)], target="ice40", event_driven=False)
event = estimate([(16, 8)], target="ice40", event_driven=True)
savings = (1 - event.total_dynamic_power_mw / clock.total_dynamic_power_mw) * 100
print(f"Power reduction: {savings:.0f}%")
```

## 4. Does It Fit?

```python
report = estimate([(784, 128), (128, 10)], target="ice40")
if report.fits_on_target:
    print("Deploy!")
else:
    print(f"Too large: {report.utilization_pct:.0f}% utilization")
```

## Further Reading

- [Tutorial 43: Platform Profiler](43_platform_profiler.md)
- [Tutorial 44: Model Compression](44_model_compression.md)
- [API: Energy](../api/energy.md)
