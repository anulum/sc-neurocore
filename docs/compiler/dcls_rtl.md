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

## Multi-language reference kernel

The same Q8.8 tent arithmetic is exposed as a wired, importable batch kernel in
`sc_neurocore.scpn.dcls_tent_kernel`. It contracts many output channels in one
call, each with a per-channel learnable `(centre, sigma)` tent — the shape an
emitted DCLS layer evaluates per spike frame. Because every operation is exact
integer arithmetic (truncating gate division, arithmetic-shift output), all five
backends return identical raw arrays; the parity tolerance is exactly zero, with
no last-ULP allowance.

### Kernel sources

| Backend | File | Build |
| --- | --- | --- |
| Python primary | `src/sc_neurocore/scpn/dcls_tent_kernel.py` | — (floor reference) |
| Rust | `engine/src/scpn/dcls.rs` + `py_dcls_max_forward_batch_q88` | `maturin develop --release` |
| Julia | `src/sc_neurocore/accel/julia/scpn/dcls.jl` | `juliacall` (lazy include) |
| Go | `src/sc_neurocore/accel/go/dcls_tent/dcls_tent.go` | `go build -buildmode=c-shared` |
| Mojo | `src/sc_neurocore/accel/mojo/dcls_tent/dcls_tent.mojo` | `mojo build --emit shared-lib` |

`dcls_max_forward_batch(...)` dispatches fastest-first (Rust → Mojo → Julia → Go
→ Python) and accepts an explicit `backend=` override; `available_backends()`
reports which compiled artefacts are present. Inputs are a row-major
`n_channels * n_taps` spike (`uint8`) and weight (`int16`) buffer plus per-channel
`int16` `centres`/`sigmas`. The result carries `outputs_q88` (`int16`),
`accumulators_q16_16` (`int32`), `overflow` (`bool`), `active_tap_counts`
(`int64`) and `max_gates_q88` (`int16`).

### Cross-language parity and throughput

Measured on an 11th Gen Intel Core i5-11600K at 3.90 GHz, CPU affinity pinned to
cores 10-11, workload 4096 channels × 64 taps (262 144 elements), via
`benchmarks/bench_dcls_tent_kernel.py`
(`benchmarks/results/bench_dcls_tent_kernel.json`):

| Backend | Channels/s | Per-call (ms) | Speedup vs Python | Parity |
| --- | --- | --- | --- | --- |
| Rust | 3 119 904 | 1.313 | 85.41× | bit-exact (Δ = 0) |
| Julia | 2 946 718 | 1.390 | 80.67× | bit-exact (Δ = 0) |
| Mojo | 2 673 594 | 1.532 | 73.19× | bit-exact (Δ = 0) |
| Go | 2 486 490 | 1.647 | 68.07× | bit-exact (Δ = 0) |
| Python | 36 527 | 112.136 | reference | reference |

The host carried a load average of ≈17.9 during this run, so the absolute
throughput figures are functional and regression evidence only and must be
re-measured on a reserved, quiet host before any production speedup claim. The
zero parity delta is independent of host load. The JSON artefact records the
command, affinity, cpuset shield, host load before/after, governor sample and
toolchain versions.

### Tests

| File | Verifies |
| --- | --- |
| `tests/test_dcls_tent_kernel.py` | Gate, single/batch contraction, saturation, validation, dispatch and fallback |
| `tests/test_dcls_tent_kernel_parity.py` | Bit-exact parity of every built backend against the Python floor across deterministic, all-silent, saturating and large random workloads |
| `engine/src/scpn/dcls.rs` (`#[cfg(test)]`) | Rust single + batch arithmetic, error boundaries and saturation |

## Open hardware evidence

`tests/test_dcls_synth_zu3eg.py` is gated by `MIF_VIVADO_CI=1`.  The ZU3EG
WNS/utilisation report is not claimed until the Vivado 2024.2 self-hosted
runner archives a passing timing summary.
