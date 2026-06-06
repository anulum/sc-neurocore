# DCLS Q8.8 RTL Contract

SC-NeuroCore now has a synthesizable scalar DCLS path for delay-coded
learnable-spike kernels.  The implementation targets the deterministic
integer-delay contract needed by the MIF core lane before a full SHD checkpoint
trace is replayed through FPGA timing.

## Contract

The layer consumes one spike stream and samples it through `N_TAPS` axonal
delay lines.  Each delayed tap is multiplied by a Q8.8 learnable weight and a
Q8.8 triangular tent gate:

```text
delay_q88 = tap_index << 8
distance_q88 = abs(delay_q88 - centre_q88)
gate_q88 = max(0, sigma_q88 - distance_q88) << 8 / sigma_q88
contribution_q16_16 = weight_q88 * gate_q88
```

The accumulator is Q16.16 and the emitted output is saturated back to Q8.8.
`sigma_q88 <= 0` is invalid and raises `invalid_sigma` in RTL or a fail-closed
error in the Rust/Python references.

## Implemented surfaces

| Surface | File | Role |
| --- | --- | --- |
| Rust reference | `engine/src/scpn/dcls.rs` | Bit-true DCLS Q8.8 arithmetic, error boundaries, saturation telemetry |
| IR graph | `engine/src/ir/graph.rs` | `DclsLayer` operation and `DclsParams` |
| SystemVerilog emitter | `engine/src/ir/emit_sv.rs` | Emits `sc_dcls_layer_core` with packed tap offsets |
| Axonal delay RTL | `hdl/sc_dcls_axonal_delay.v` | DCLS-specific delay-line module preserving legacy behaviour |
| Tent kernel RTL | `hdl/sc_dcls_tent_kernel.v` | Q8.8 tent weighting and Q16.16 accumulation |
| Layer core RTL | `hdl/sc_dcls_layer_core.v` | Composes delay lines and tent kernel |
| Formal harness | `hdl/formal/sc_dcls_layer_core.sby` | Non-negative monotonic-input safety and valid liveness |
| Cosim reference | `tools/cosim_dcls_q88_vs_pytorch.py` | Python/PyTorch deterministic parity |

## SystemVerilog module

```verilog
sc_dcls_layer_core #(
    .N_TAPS(16),
    .DATA_WIDTH(16),
    .FRACTION(8),
    .ACC_WIDTH(32),
    .DELAY_DEPTH(31),
    .PTR_WIDTH(5)
) dcls (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .spike_in(spike_in),
    .tap_offsets(tap_offsets),
    .tap_weights_q88(tap_weights_q88),
    .centre_q88(centre_q88),
    .sigma_q88(sigma_q88),
    .out_valid(out_valid),
    .weighted_sum_q88(weighted_sum_q88),
    .accumulator_q16_16(accumulator_q16_16),
    .overflow(overflow),
    .invalid_sigma(invalid_sigma)
);
```

## Validation

The committed local evidence covers deterministic DCLS arithmetic and RTL
elaboration, not board-level throughput:

| Evidence | Command |
| --- | --- |
| Rust reference and emitter tests | `cargo test dcls --lib` from `engine/` |
| Python/PyTorch parity | `.venv/bin/python tools/cosim_dcls_q88_vs_pytorch.py --json benchmarks/results/local_python_2026-06-04_dcls_cosim.json` |
| RTL elaboration | `yosys -p "read_verilog -sv hdl/sc_dcls_axonal_delay.v hdl/sc_dcls_tent_kernel.v hdl/sc_dcls_layer_core.v; hierarchy -check -top sc_dcls_layer_core; proc; opt; stat"` |
| Bounded formal check | `sby -f hdl/formal/sc_dcls_layer_core.sby` |
| Python/SystemVerilog benchmark evidence | `benchmarks/bench_dcls_q88_rtl.py` |
| Rust benchmark evidence | `cargo run --manifest-path engine/Cargo.toml --release --example bench_dcls_q88` |

The 2026-06-04 benchmark artefacts are local contract/regression evidence:
Python/PyTorch/SystemVerilog `6349.497 ns/sample`, Rust `40.184 ns/sample`,
SymbiYosys/cvc5 bounded formal pass in `1.533 s`, and Yosys generic synthesis
estimate of `106003` cells.  Any throughput claim must be rerun on reserved
isolated cores with CPU affinity, host load, governor, frequency, and tool
versions recorded in the raw JSON.

## Open hardware evidence

`tests/test_dcls_synth_zu3eg.py` is gated by `MIF_VIVADO_CI=1`.  The ZU3EG
WNS/utilisation report is not claimed until the Vivado 2024.2 self-hosted
runner archives a passing timing summary.
