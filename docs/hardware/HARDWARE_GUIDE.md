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
| `sc_aer_encoder.v` | Captures a coalesced spike mask and emits each lowest-index pending address once, with its service-cycle timestamp. |
| `sc_event_neuron.v` | Q8.8 LIF with membrane integration on input events and decay on periodic leak ticks. |
| `sc_aer_router.v` | Distributes AER events to target neurons using BRAM connectivity lookup. Sparse fanout serialized. |
| `sc_aer_priority_queue.v` | Strict-priority AER queue for downstream backpressure, FIFO ties within priority, sticky drop traps, and critical-event deadline traps. |

The clocked encoder emits one address per clock while pending bits remain.
Repeated spikes on an already pending bit coalesce; clearing the served bit
takes precedence over a simultaneous spike on that bit. Its `busy` output
reports captured pending bits. Connect its event outputs to a queue that can
accept this cadence; the encoder has no output-ready input. See the dated
[AER synthesis and simulation observations](SYNTHESIS_RESULTS.md) for the
current source, resource counts and cycle measurements.

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

## 12. SHD AXI-Lite input commits

`hdl/sc_shd_axi_wrapper.v` connects the fixed-point SHD network
(`sc_shd_top`, 140 → 128 → 128 → 20) to a 32-bit AXI4-Lite bus.
Its input protocol is:

1. Write `T_ORIG` at `0x04` and the three Q16.16 scales at `0x08`–`0x10`.
   Valid input lengths are 1–65503; the upper bound leaves room in the
   16-bit logical counter for padding and pipeline drain.
2. Write bit 0 of `CTRL` (`0x00`) to reset and arm an inference. Configuration
   is captured for this inference; subsequent configuration writes stage the
   next one. Starting while busy, or with an invalid length, returns SLVERR.
3. For each of the `T_ORIG` time bins, stage the five words at `0x14`–`0x24`.
   The first four hold channels 0–127; the low twelve bits of the fifth hold
   channels 128–139. AXI byte strobes preserve the other bytes.
4. Write any value, with at least one byte strobe enabled, to `SPIKE_COMMIT`
   (`0x28`). This snapshots the whole 140-bit vector and advances one input
   step. The first commit starts the core with sample 0 already present.
5. After the last input, the wrapper automatically runs thirty zero-padding
   steps and three pipeline-drain clocks. Poll `CTRL` bit 2 for completion,
   then read the twenty signed-int32 class sums at `0x40`–`0x8c`.

`CTRL` bit 1 is busy, including the wait for the first commit. Bit 2 remains
set until the next accepted start or reset. Idle commits and commits during
or after the automatic tail have no effect. Existing drivers which send
thirty extra padding commits remain compatible. Unaligned writes and writes
to reserved or read-only addresses return SLVERR without mutation.

All network state holds between input commits: delay pointers and storage,
membranes, dense output registers, pipeline state and accumulators. The core
and its cells use a synchronous `step_en` with `USE_STEP_ENABLE=1`; the clock
itself is unchanged. Their default `USE_STEP_ENABLE=0` retains the existing
one-vector-per-clock direct-core interface. Asynchronous reset remains active
when logical time is stopped.

A time-multiplexed implementation may take several physical clocks per
logical timestep. Such a replacement needs an explicit input-acceptance and
timestep-completion handshake: holding `step_en` low must not stop its
internal computation. The wrapper above currently connects the parallel
core; compatibility with a different backend needs separate verification.

Run the AXI regression in a new output directory:

It requires Icarus Verilog 12 or later, Verilator with timing and coverage
support, and a C++ compiler. The same test runs with both simulators; the
existing direct-core testbench checks the free-running interface as well.

```bash
mkdir -p build
bash hdl/test_shd_axi.sh build/shd-axi
```

The test sends real AXI transactions with independent AW/W arrival, response
backpressure, partial-byte writes, uncommitted input changes, repeated
inferences and reset. It compares all twenty class sums for three input
sequences against `hdl/fixtures/shd_axi_expected.hex`, generated with the
maintained `tools/cosim_shd_top_verilog.py::python_output_v_sum` reference.
`coverage.info` retains RTL line coverage, and CI requires at least 95% for
both the AXI wrapper and the network logical-time controller. This transport
threshold does not claim complete coverage of the cells\' arithmetic or LUTs.
The deterministic fixture uses:

- Weight at flattened index `i`, with `n` inputs:
  `((13*i + 7*(i//n)) % 17) - 8`; the three Q16.16 scales are 4096, 4096, 2048.
- Delay for channel `c`: `15 - ((7*c) % 31)`.
- Input lengths 6, 1, 3. Samples 0 and 1 spike when
  `(7*c + 11*t + 13*sample) % 17` is below 6 and 11 respectively; sample 2 is zero.

This regression exercises the AXI transport in simulation. Before using a
trained network on a board, provide a verified weight/delay initialisation
path: the current SHD cells guard their `$readmemh` calls with
`ifndef SYNTHESIS`, and the wrapper exposes no model-data loading registers.
The shipped Vivado script and constraints target PYNQ-Z2; select and verify
the actual board part, constraints and clock setup for another board.

## 13. Next steps

- Use BENCHMARKS.md to record performance and accuracy data.
- Update TECHNICAL_MANUAL.md if hardware flows change.
- Keep HDL outputs aligned with the Python reference.
