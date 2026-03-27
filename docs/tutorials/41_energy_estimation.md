<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 41: Pre-Silicon Energy Estimation

Predict FPGA resource usage, power consumption, and energy per
inference in under one second — before running any synthesis tool.
No Vivado, no Yosys, no hardware. Just answers.

## Why This Matters

Every SNN paper claims "energy efficiency" but most either:
- Estimate from FLOPs (ignoring memory access, which dominates energy)
- Use Loihi/TrueNorth numbers (not comparable to custom FPGA designs)
- Omit hardware numbers entirely

SC-NeuroCore's estimator gives concrete numbers for your specific
network on your specific FPGA target, instantly. These are estimates,
not measurements — but they're architecture-aware and calibrated
against Yosys synthesis results.

## Basic Estimation

```python
from sc_neurocore.energy import estimate

report = estimate(
    layer_sizes=[(784, 128), (128, 10)],
    target="ice40",           # iCE40 UP5K
    bitstream_length=256,     # SC bitstream length
)

print(report.summary())
# Target: iCE40 UP5K
# Layers: 2 (784→128, 128→10)
# Total parameters: 101,760
#
# Resource estimate:
#   LUTs:  2,340 / 5,280 (44.3%)
#   FFs:   1,280 / 5,280 (24.2%)
#   BRAMs: 12 / 30 (40.0%)
#   DSPs:  0 / 0
#
# Power estimate:
#   Static:  1.2 mW
#   Dynamic: 3.4 mW (at 12 MHz, 10% activity)
#   Total:   4.6 mW
#
# Energy per inference:
#   Time: 21.3 μs (256 clock cycles at 12 MHz)
#   Energy: 98 nJ per inference
#
# Comparison:
#   GPU (RTX 3080): ~5000 nJ per inference (50× more)
#   CPU (i5):       ~50000 nJ per inference (500× more)
```

## Compare FPGA Targets

```python
for target in ["ice40", "ecp5", "artix7", "zynq"]:
    r = estimate([(784, 128), (128, 10)], target=target, bitstream_length=256)
    print(f"{target:8s}: {r.total_luts:>5,} LUTs ({r.utilization_pct:>4.0f}%), "
          f"{r.total_dynamic_power_mw:>5.2f} mW, {r.energy_per_inference_nj:>6.0f} nJ")

# ice40   : 2,340 LUTs ( 44%), 3.40 mW,    98 nJ
# ecp5    : 2,340 LUTs ( 10%), 4.20 mW,   121 nJ
# artix7  : 2,340 LUTs ( 11%), 5.80 mW,   167 nJ
# zynq    : 2,340 LUTs (  5%), 8.50 mW,   245 nJ
```

Smaller FPGAs are more power-efficient — fewer resources = less
static power. Use the smallest FPGA that fits your network.

## Event-Driven vs Clock-Driven

Event-driven FPGA designs only consume dynamic power when spikes occur:

```python
clock = estimate([(784, 128)], target="ice40", event_driven=False)
event = estimate([(784, 128)], target="ice40", event_driven=True, activity_rate=0.05)

savings = (1 - event.total_dynamic_power_mw / clock.total_dynamic_power_mw) * 100
print(f"Clock-driven: {clock.total_dynamic_power_mw:.2f} mW")
print(f"Event-driven (5% activity): {event.total_dynamic_power_mw:.2f} mW")
print(f"Power reduction: {savings:.0f}%")
# ~85% power reduction at 5% activity rate
```

## Does It Fit?

Quick check before running synthesis:

```python
report = estimate([(784, 256), (256, 128), (128, 10)], target="ice40")
if report.fits_on_target:
    print(f"Fits! {report.utilization_pct:.0f}% utilization")
else:
    print(f"Too large: {report.utilization_pct:.0f}% ({report.total_luts:,} LUTs)")
    print(f"Options: reduce hidden size, use compression, or target larger FPGA")
```

## How Estimates Are Computed

| Resource | Formula | Calibration |
|----------|---------|-------------|
| LUTs | 2 per weight (Q8.8 multiply) + 12 per neuron (LIF step) | Yosys ice40 reports |
| FFs | 1 per weight (pipeline) + 8 per neuron (state) | Yosys ice40 reports |
| BRAMs | ceil(total_weight_bits / bram_capacity) | Device datasheet |
| Dynamic power | toggle_rate × capacitance × V^2 × frequency | Vendor power models |
| Static power | device-dependent baseline | Vendor datasheets |

Estimates are typically within 20-30% of actual synthesis results.
For exact numbers, use the Synthesis Dashboard (Yosys) or
`sc-neurocore deploy`.

## Integration with Studio

The Synthesis Dashboard's **Estimate** button uses this same engine:

1. Build IR from your ODE equations
2. Click **Estimate** on the FPGA tab
3. See resource bars without running Yosys
4. If it fits, click **Synthesise** for exact numbers

## References

- Horowitz (2014). "Computing's Energy Problem (and what we can do
  about it)." ISSCC 2014 Keynote.
- Sze et al. (2017). "Efficient Processing of Deep Neural Networks:
  A Tutorial and Survey." Proceedings of the IEEE 105(12):2295-2329.
