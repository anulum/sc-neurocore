# sc-neurocore Hardware Guide

This guide covers practical hardware workflows for sc-neurocore. It focuses on FPGA prototyping but also notes considerations for ASIC or mixed-signal implementations. The Python simulator remains the reference model; hardware should be validated against it using statistical comparisons.

---

## 1. Hardware architecture summary

Stochastic computing maps well to digital hardware because core operations use simple logic:

- Unipolar multiplication uses AND gates.
- Weighted addition can be approximated with MUX logic.
- Integration is a popcount or accumulator over a window.
- Thresholding and reset logic are simple comparators and state machines.

This makes SC designs highly parallel and energy efficient, but precision is controlled by bitstream length rather than word width.

---

## 2. FPGA deployment workflow

A typical workflow is:

1. Validate a network in Python.
2. Export or implement HDL that matches the validated configuration.
3. Integrate with a top-level design that includes data input/output and control signals.
4. Generate a bitstream and deploy to a target board.
5. Run test vectors and compare distributions to the Python output.

Keep early prototypes small. Use short bitstreams to confirm correctness before scaling.

---

## 3. Verilog generator usage

The Verilog generator is intended as a scaffold for dense layer networks. It emits a simple top-level module with a fixed 8-bit input and output bus. Use it for quick experiments and then extend it for production designs.

Recommended steps:

- Create a generator instance and add dense layers.
- Generate the Verilog string and save to a file.
- Wrap the generated module with any additional I/O logic.

If you need width parameterization or custom layer types, extend the generator rather than editing the emitted Verilog manually.

---

## 4. SPICE generation for memristive crossbars

The SPICE generator outputs netlists for memristive crossbars. It maps weights in [0,1] to conductances between G_off and G_on. Use this to evaluate analog behavior or device variability.

Guidelines:

- Validate the G_on and G_off values for your target technology.
- Use consistent scaling between software weights and device conductance.
- Include realistic load and sensing circuits for meaningful output measurements.

---

## 5. Bitstream length and timing

Bitstream length translates directly to latency. For a 250 MHz clock:

- 256 bits -> about 1.0 microsecond
- 1024 bits -> about 4.1 microseconds
- 4096 bits -> about 16.4 microseconds

For early hardware validation, 256 or 512 bits are usually enough. For high precision experiments, increase to 1024 or 4096.

---

## 6. RNG strategy on hardware

Correlation between bitstreams is a common source of error. In hardware, RNG quality and independence are critical.

Options:

- LFSR arrays with unique taps per stream.
- Low-discrepancy sequences (Sobol) for deterministic experiments.
- Noise-derived RNGs if available on the target platform.

Always verify correlation empirically. If correlation is high, introduce scrambling or independent RNG instances.

---

## 7. Validation methodology

Two validation modes are recommended:

1. Bit-true validation: compare bitstream outputs for short runs.
2. Statistical validation: compare output mean and variance for long runs.

Statistical validation is more robust for SC systems and is the preferred method for larger tests.

---

## 8. Hardware checklist

Before deploying:

- Confirm input ranges map to [0,1] or your encoding range.
- Confirm bitstream length matches the expected precision.
- Verify RNG independence across inputs and weights.
- Run at least one deterministic test with fixed seeds.
- Record results in BENCHMARKS.md.

---

## 9. Common pitfalls

- Saturation from unnormalized inputs.
- Correlated RNG streams causing biased multiplication.
- Mismatched bitstream lengths between layers.
- Forgetting to reset state between runs.

---

## 10. Event-Driven Architecture

SC-NeuroCore includes event-driven Verilog modules for power-efficient
SNN execution:

| Module | Purpose |
|--------|---------|
| `sc_aer_encoder.v` | Converts spike vector to AER (Address-Event Representation) packets. Only active neurons generate events. |
| `sc_event_neuron.v` | Q8.8 LIF that computes only on input events or periodic leak ticks. Idle neurons consume zero switching power. |
| `sc_aer_router.v` | Distributes AER events to target neurons using BRAM connectivity lookup. Sparse fanout serialized. |
| `sc_aer_priority_queue.v` | Strict-priority AER queue for downstream backpressure, FIFO ties within priority, sticky drop traps, and critical-event deadline traps. |

For a 1000-neuron network firing at 10 Hz with 1 MHz clock:
- **Clock-driven**: 1000 neurons × 1M cycles/s = 1 billion operations/s
- **Event-driven**: 10,000 events/s × ~32 fanout = 320K operations/s (**3000x fewer**)

Use event-driven modules when:
- Network firing rates are sparse (<50 Hz average)
- Power budget is critical (edge deployment)
- FPGA resource constraints limit dense matrix operations

Use clock-driven modules when:
- Bit-exact Python-Verilog co-simulation is needed
- Network is dense (most neurons active every cycle)
- Deterministic timing is required for formal verification

For live control or safety-critical fanout, enable the router priority path and
connect `out_event_ready`.  See [AER Priority Queue and Backpressure
Contract](aer_priority_queue.md) for the full port, trap, formal, and benchmark
contract.

## 11. One-Command Deployment

```bash
sc-neurocore deploy model.nir --target artix7 -o build/
```

This generates a complete project with all 19 HDL modules, a generated
neuron SystemVerilog file, and target-specific build scripts. See
Tutorial 40 for details.

## 12. Next steps

- Use BENCHMARKS.md to record performance and accuracy data.
- Update TECHNICAL_MANUAL.md if hardware flows change.
- Keep HDL outputs aligned with the Python reference.
