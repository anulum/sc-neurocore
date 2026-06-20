# Mixed-precision Q8.8 × Q16.16 dense MAC

`sc_neurocore.compiler.mixed_dense_kernel` is a wired, batched integer reference
for the mixed-precision dense operator: Q8.8 weights contract Q16.16 input codes
into a Q16.16 saturating accumulator. It is the integer branch of the
mixed-precision pipeline — the per-tensor scale folded so the accumulator divisor
equals the Q8.8 weight scale, which is the production Zynq/UltraScale+ contract —
and matches the engine reference in `engine/src/ir/qformat.rs`.

## Contract

For each batch row `b` and output channel `o`:

```text
raw      = Σ_i weight_q88[o, i] * input_q1616[b, i]      (signed 64-bit)
scaled   = raw >> 8                                       (arithmetic shift = floor ÷ 256)
output   = clamp(scaled, -2**31, 2**31 - 1)               (Q16.16 saturation)
overflow = scaled outside the Q16.16 range
underflow= raw ≠ 0 and scaled == 0 and not overflow
```

The arithmetic right shift equals floor division by the power-of-two weight scale,
so the result matches NumPy `floor_divide` for negative accumulators as well as
positive. Because the whole path is exact integer arithmetic, the Python floor and
the Rust, Julia, Go and Mojo backends agree bit-for-bit; the parity tolerance is
exactly zero.

**Accumulation contract.** The kernel accumulates in signed 64-bit integers. The
caller keeps `max|weight| * max|input| * n_inputs` within `int64`; the reference
checks this conservative bound up front and fails closed rather than wrapping.

## Kernel sources

| Backend | File | Build |
| --- | --- | --- |
| Python primary | `src/sc_neurocore/compiler/mixed_dense_kernel.py` | — (floor reference) |
| Rust | `engine/src/ir/qformat.rs` (`mixed_dense_forward_batch_q88_q1616`) + `py_mixed_dense_forward_batch_q88_q1616` | `maturin develop --release` |
| Julia | `src/sc_neurocore/accel/julia/mixed_dense/mixed_dense.jl` | `juliacall` (lazy include) |
| Go | `src/sc_neurocore/accel/go/mixed_dense/mixed_dense.go` | `go build -buildmode=c-shared` |
| Mojo | `src/sc_neurocore/accel/mojo/mixed_dense/mixed_dense.mojo` | `mojo build --emit shared-lib` |

`mixed_dense_forward_batch(...)` dispatches through `backend="auto"` (fastest-first
fallback) or an explicit `backend=` name; `available_backends()` reports which
compiled artefacts are present. `weights_q88` is a row-major `n_outputs * n_inputs`
buffer, `inputs_q1616` a row-major `n_batch * n_inputs` buffer; the result carries
`(n_batch, n_outputs)` `outputs_q1616` (`int32`), `overflow` and `underflow`
(`bool`).

## Cross-language parity and throughput

Measured on an 11th Gen Intel Core i5-11600K at 3.90 GHz, CPU affinity pinned to
cores 10-11, workload 256 outputs × 256 inputs × 64-batch (1 048 576
multiply-accumulates), via `benchmarks/bench_mixed_dense_kernel.py`
(`benchmarks/results/bench_mixed_dense_kernel.json`):

| Backend | MAC/s | Per-call (ms) | Speedup vs Python | Parity |
| --- | --- | --- | --- | --- |
| Julia | 1.95 × 10⁹ | 2.153 | 1.08× | bit-exact (Δ = 0) |
| Python (NumPy) | 1.80 × 10⁹ | 2.326 | reference | reference |
| Mojo | 1.52 × 10⁹ | 2.756 | 0.84× | bit-exact (Δ = 0) |
| Rust | 1.40 × 10⁹ | 2.989 | 0.78× | bit-exact (Δ = 0) |
| Go | 0.90 × 10⁹ | 4.655 | 0.50× | bit-exact (Δ = 0) |

For this dense matmul the NumPy integer `@` (a blocked, compiled C loop) and the
JIT-compiled Julia loop are competitive with or faster than the scalar C-ABI FFI
backends, which use straightforward triple loops without SIMD or cache blocking.
The accelerator backends are kept because they are bit-exact and remove the
Python call overhead in embedded dispatch paths, but the JSON artefact carries the
real per-backend ranking so a data-driven dispatcher does not assume an
accelerator is fastest here. The host carried a load average of ≈12.4 during this
run, so the absolute throughput figures are functional and regression evidence
only and must be re-measured on a reserved, quiet host before any production
speedup claim. The zero parity delta is independent of host load.

## Tests

| File | Verifies |
| --- | --- |
| `tests/test_mixed_dense_kernel.py` | Contraction, signed floor division, saturation, underflow, validation, accumulation bound, dispatch and fallback |
| `tests/test_mixed_dense_kernel_parity.py` | Bit-exact parity of every built backend against the Python floor across deterministic, all-zero, saturating, underflow and large random workloads |
| `engine/src/ir/qformat.rs` (`#[cfg(test)]`) | Rust batch arithmetic, signed floor division, overflow/underflow flags and shape validation |
