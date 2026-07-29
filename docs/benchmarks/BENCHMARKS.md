# SC-NeuroCore Benchmarks

Performance measurements for SC-NeuroCore. Historical core engine rows come from v3.13.3, FPGA synthesis additions from v3.14.0, and documentation/release evidence is current for v3.15.8. All Python numbers are CPU-only unless a GPU environment is named. Rust numbers use Criterion and must be read with their recorded SIMD/environment context.

## Evidence boundary

This page is a curated benchmark report, not an authority for untracked local measurements or marketing estimates. Every quoted performance, power, utilisation, timing, or
hardware-efficiency number must remain traceable to a committed raw artefact:
JSON or CSV under `benchmarks/results/`, a named tool report under `hdl/` or
`build/`, or a companion paper artefact with command and environment
provenance.

When a newer local run exists but its raw output is not committed, cite it only
as local exploratory evidence. Do not promote it to README, roadmap, release,
or paper claims until the raw artefact and environment record are committed.

The 2026-06-04 local Python/Rust precision benchmark artefacts were captured on
a workstation under concurrent load and without an exclusive isolated CPU core
set.  Treat those medians as contract/regression context, not final throughput
claims.  Any production performance claim must be rerun on reserved isolated
cores, with CPU affinity, host-load, governor, and frequency evidence recorded
in the raw artefact.

The 2026-06-05 refreshed mixed-dense, block-floating, and precision-envelope
artefacts were executed with process affinity pinned to CPUs `8-9`.  The raw
JSON records host load, affinity, CPU governor, and sampled frequency context,
but the workstation still did not expose kernel-reserved isolated cores to this
user session.  Treat the refreshed medians as local regression evidence and the
proof fields as the authoritative contract evidence.

The 2026-07-14 live-control AXI4-Lite/PCIe-MMIO rerun was executed with
process affinity pinned to CPUs `8-9`, and the raw artefact records that the
process affinity matched the requested benchmark cpuset.  The workstation did
not expose kernel-reserved isolated cores to this user session, so these numbers
remain local regression evidence rather than production throughput claims.

The module-level pytest throughput checks use load-tolerant smoke thresholds
unless `SC_NEUROCORE_STRICT_THROUGHPUT=1` is set. The shared default smoke floor
is one percent of the historical strict rate, capped at 100 operations per
second; finite positive progress is always required. Use strict mode only for
isolated-core benchmark captures and commit the raw artefact before treating the
numbers as release evidence.

The 2026-06-04 AER priority queue benchmark follows the same boundary.  It was
run under a temporary runtime cpuset shield: system/user slices were moved off
the benchmark cores, the benchmark ran in its own `benchmark.slice`, and the raw
artefact records the process affinity plus cgroup effective CPU set.  This is
stronger than `taskset` pinning but still distinct from boot-time
`isolcpus`/`nohz_full` kernel isolation.  The artefact is
Python/SystemVerilog contract evidence for strict-priority ordering, FIFO ties,
backpressure drops, critical-deadline traps, and Yosys RTL elaboration; it is
not an FPGA throughput or latency measurement.

> **v3.14.0 additions:** SHD FPGA synthesis on Zynq XC7Z020 — 1 317 LUT
> (2.5%), 848 FF (0.8%), WNS +4.048 ns at 100 MHz. See
> `hdl/reports/vivado_util_xc7z020_100mhz.rpt` for full Vivado report.

---

## 1. Environment

| Field | Value |
|-------|-------|
| Date | 2026-03-15 |
| Git tag | v3.13.3 |
| OS | Windows 11 Pro 10.0.26200 |
| CPU | Intel Core i5-11600K (6C/12T, 3.9 GHz base, AVX-512, DL Boost) |
| RAM | 32 GB DDR4-3200 |
| Python | 3.12.5 |
| NumPy | 1.26.4 |
| Rust | 1.86.0 (stable) |
| SIMD tier | avx512-vpopcntdq |
| GPU (NVIDIA) | GeForce GTX 1060 6GB — Pascal sm_61, PyTorch 2.6.0+cu124 |
| GPU (AMD) | Radeon RX 6600 XT — no ROCm on Windows, torch-directml incompatible |
| CuPy | unavailable — CUDA Toolkit 13.1 dropped Pascal (sm_61) support |

Current developer environments may run newer PyTorch wheels whose compiled CUDA
architecture list no longer includes Pascal `sm_61`. In that case
`sc_neurocore.training.auto_device()` falls back to CPU instead of selecting an
unsupported CUDA device. Historical PyTorch CUDA benchmark rows remain tied to
the PyTorch/CUDA build named in their evidence table and should not be reused
for newer wheels without a fresh benchmark artefact.

---

## 2. Scalar Primitives (Python)

| Operation | Iterations | Latency (µs) | Throughput |
|-----------|-----------|---------------|------------|
| LFSR step (16-bit) | 1,000,000 | 0.8 | 1.33 Mstep/s |
| Bitstream encoder step | 1,000,000 | 0.9 | 1.10 Mstep/s |
| LIF neuron step (Q8.8) | 1,000,000 | 0.9 | 1.07 Mstep/s |

---

## 3. Packed Bitstream Operations (Python/NumPy)

| Operation | Size | Iterations | Latency (µs) | Throughput |
|-----------|------|-----------|---------------|------------|
| pack_bitstream 1-D | 1,024 | 10,000 | 8.7 | 0.12 Gbit/s |
| pack_bitstream 1-D | 65,536 | 2,000 | 123.1 | 0.53 Gbit/s |
| pack_bitstream 2-D | 64×1,024 | 2,000 | 121.6 | 0.54 Gbit/s |
| vec_and | 1,024 words | 50,000 | 1.6 | 41.0 Gbit/s |
| vec_popcount SWAR | 1,024 words | 50,000 | 30.2 | 2.17 Gbit/s |

---

## 4. Dense Layer Forward Pass (Python)

| Configuration | Iterations | Latency (µs) | Throughput |
|---------------|-----------|---------------|------------|
| 16×8, L=256 | 500 | 352.7 | 0.09 GOP/s (SC) |
| 64×32, L=1,024 | 100 | 2,405.8 | 0.87 GOP/s (SC) |

### AER Priority Queue Backpressure Contract (2026-06-04)

This benchmark covers the NEU-C.4 event-control contract: AER fanout packets
with lower numeric priority must overtake best-effort packets while preserving
FIFO order within equal priority classes.  The committed artefact also records
finite-capacity backpressure, sticky drop traps, sticky critical-deadline traps,
CPU affinity, cgroup effective CPU set, host load, CPU governor, and Yosys
elaboration time.

| Path | Workload | Result | Raw evidence |
|------|----------|--------|--------------|
| Python reference model | 4,096 deterministic events x 100 repeats | 4.138 us/event under runtime cpuset shield `10-11`; `priority_violations=0`, `fifo_tie_violations=0` | `benchmarks/results/local_python_2026-06-04_aer_priority_queue.json` |
| SystemVerilog `sc_aer_priority_queue` | Yosys synthesis/elaboration | `yosys.exit_code=0`, 6,364 cells in the artefact | `benchmarks/results/local_python_2026-06-04_aer_priority_queue.json` |

No Rust, Julia, Go, or Mojo counterpart exists for this HDL-only queue surface
as of 2026-06-04.  Cross-language comparison therefore means Python reference
contract versus SystemVerilog RTL elaboration/simulation for this task.

### Live-control AXI4-Lite / PCIe-MMIO Register Window (2026-07-14)

This benchmark covers the live-parameter update contract for hot-swappable
weights and Kuramoto coupling parameters.  Both protocols use the same
CRC32-gated shadow-bank core: AXI4-Lite exposes the core directly, while the
PCIe path emits a PCIe-MMIO register-window adapter that expects upstream PCIe
hard IP to present decoded single-clock MMIO strobes.

| Path | Workload | Result | Raw evidence |
|------|----------|--------|--------------|
| Python update-sequence builder | 20,000 deterministic staged writes x 7 repeats | AXI4-Lite median `15.466 us/sequence`; PCIe-MMIO median `16.588 us/sequence` under process affinity `8-9` | `benchmarks/results/local_python_2026-06-04_live_control_updates.json` |
| Python static RTL regeneration | 7 batches of 7 generations per protocol | AXI4-Lite median `96.172 us/source`; PCIe-MMIO median `146.004 us/source` | `benchmarks/results/local_python_2026-06-04_live_control_updates.json` |
| SystemVerilog AXI4-Lite core | Generated trap-capture simulation | `trap_capture.passed=true`; staged overflow and underflow traps latched without mutating active coefficients | `benchmarks/results/local_python_2026-06-04_live_control_updates.json` |
| SystemVerilog PCIe-MMIO wrapper | Generated commit simulation | `pcie_mmio_commit_capture.passed=true`; partial write strobes raise sticky `partial_write`, stale CRC32 guard raises sticky `checksum_mismatch`, invalid bank selection and invalid active readback raise sticky `invalid_selection`, read-only bank writes raise sticky `read_only_bank`, and retargeting selection registers after shadow load cannot redirect the committed bank | `benchmarks/results/local_python_2026-06-04_live_control_updates.json` |

The 2026-07-14 capture used process affinity `8-9` but no kernel-reserved
isolated cores. One-minute host load was `10.132` before and `9.801` after the
run, and the CPU governor was `powersave`. These timings are local diagnostic
regression evidence only; the successful RTL simulations and deterministic
checksums are the functional evidence.

No Rust, Julia, Go, or Mojo counterpart exists for this HDL bus-adapter surface
as of 2026-07-14. Cross-language comparison therefore means Python control
contract generation versus SystemVerilog RTL simulation for AXI4-Lite and
PCIe-MMIO.

### Mixed Q8.8/Q16.16 Dense Contract (2026-06-04)

This benchmark covers the deterministic dense mixed-precision contract: stored
Q8.8 weights, Q16.16 inputs and outputs, signed arithmetic product scaling, and
explicit saturation/overflow handling.  The Python path is the deployment
reference and manifest writer; the Rust path is the low-latency integer mirror.
The Python path is run with `QFormatMixed(scale_per_tensor=False)` for this
benchmark so the Python, Rust, and HDL surfaces share the same raw Q8.8/Q16.16
arithmetic contract instead of Python-only per-tensor rescaling.

| Path | Workload | Median | Raw evidence |
|------|----------|-------:|--------------|
| Python `CompiledMixedDense.forward_accumulator_codes` | 64×32 dense, 2,000 calls × 7 repeats | 51.934 µs/call | `benchmarks/results/local_python_2026-06-04_mixed_dense.json` |
| Python `CompiledMixedDense.forward_with_overflow` | Same deterministic matrix/vector | 51.104 µs/call | `benchmarks/results/local_python_2026-06-04_mixed_dense.json` |
| NumPy float64 dot baseline | Same deterministic matrix/vector | 1.656 µs/call | `benchmarks/results/local_python_2026-06-04_mixed_dense.json` |
| Rust `mixed_dense_q88_q1616` | 64×32 dense, 20,000 calls × 7 repeats | 2.659 µs/call | `benchmarks/results/local_rust_2026-06-04_mixed_dense.json` |
| HDL `sc_mixed_precision_dense` Yosys RTLIL stat | Default 64×32 parameters | 12,708 cells, 2,048 multipliers | `hdl/reports/yosys_mixed_precision_dense_2026-06-04.json` |

The Python mixed path reconstructed the float64 dot product with maximum
absolute error `0.0` on the committed deterministic workload.
The Python and Rust artefacts both recorded safe-workload overflow count `0`
and saturating-probe overflow count `32`, matching the lane-level HDL
`overflow_vector` contract.  The same artefacts now record conservative
precision-envelope telemetry: Python and Rust safe max absolute bound `531400`,
and saturating-probe max absolute bound
`17454214414336`; the HDL exports the matching per-output `abs_bounds_q1616`
vector.  The refreshed Python and Rust artefacts also prove the signed
symmetric fixed-point width contract: the safe workload requires `21` signed
total bits, `5` Q16.16 integer bits, and has `11` bits of headroom; the
saturating probe requires `45` signed total bits and `29` Q16.16 integer bits,
has `-13` bits of headroom, and records `saturation_required=true`.

### Block-Floating Dense Contract (2026-06-04)

This benchmark covers dense `BFP16E3X32` weights with Q16.16 inputs and
saturated Q16.16 outputs.  The shared exponent path preserves larger dynamic
range per block than canonical Q8.8 weights, at the cost of dynamic shifts in
the HDL datapath.

| Path | Workload | Median | Raw evidence |
|------|----------|-------:|--------------|
| Python `CompiledBlockFloatingDense.forward_accumulator_codes` | 64×32 dense, 2,000 calls × 7 repeats | 38.760 µs/call | `benchmarks/results/local_python_2026-06-04_block_floating_dense.json` |
| Python `CompiledBlockFloatingDense.forward_with_overflow` | Same deterministic matrix/vector | 42.356 µs/call | `benchmarks/results/local_python_2026-06-04_block_floating_dense.json` |
| NumPy float64 dot baseline | Same deterministic matrix/vector | 1.029 µs/call | `benchmarks/results/local_python_2026-06-04_block_floating_dense.json` |
| Rust `block_floating_dense_q16` | 64×32 dense, 20,000 calls × 7 repeats | 11.471 µs/call | `benchmarks/results/local_rust_2026-06-04_block_floating_dense.json` |
| HDL `sc_block_floating_dense` Yosys RTLIL stat | Parameterised 2×2, `BLOCK_SIZE=2` elaboration copy | 96 cells, 4 multipliers | `hdl/reports/yosys_block_floating_dense_2026-06-04.json` |

The deterministic block-floating workload recorded maximum absolute error
`0.22306060791015625` versus float64 dot.  This reflects BFP16E3 block-scale
quantisation on the committed synthetic workload, not runtime nondeterminism.
The Python and Rust artefacts both recorded safe-workload overflow count `0`
and saturating-probe overflow count `32`, matching the lane-level HDL
`overflow_vector` contract.  Both languages now compare the same deterministic
BFP contract: mantissa checksum `-15`, exponent checksum `0`, exponent code
range `[0, 0]`, safe max absolute bound `610816`, and saturating-probe max
absolute bound `1125865547104256`.  The refreshed proof fields record safe
width `21` signed total bits, `5` Q16.16 integer bits, and `11` bits of
headroom; the saturating probe records `51` signed total bits, `35` Q16.16
integer bits, `-19` bits of headroom, and `saturation_required=true`.  The
64×32 payload records
`parameter_count=2048` and `block_exponent_count=64` in both the Python
manifest and Rust artefact.  The HDL exports per-output
`abs_bounds_q1616`; the
full-size 64×32 block-floating Yosys frontend path is documented as toolchain
debt because Yosys 0.33 elaborates the default procedural loops during
`read_verilog` before `chparam` can reduce the dimensions.

The same Python and Rust artefacts now include a seeded `BFP16E3X2` exponent
edge sweep.  Both languages agree on safe exponent codes `[0, 7, 0, 7]`, safe
Q16.16 output codes `[1056736, -1069024]`, safe overflow and underflow counts
`0`, conservative safe bound `1069024`, and headroom `2146414623`.  The
max-exponent saturation probe records exponent code `[7]`, saturated output
code `[2147483647]`, overflow count `1`, underflow count `0`, and conservative
bound `2251662376828928`.  The safe edge sweep requires `22` signed total
bits and `6` Q16.16 integer bits with `10` bits of headroom; the max-exponent
saturation probe requires `52` signed total bits and `36` Q16.16 integer bits
with `-20` bits of headroom, proving that max shared-exponent payloads trap
rather than silently wrapping.

### Precision Trap Reports (2026-06-04)

This benchmark covers the saturation telemetry path for mixed Q8.8/Q16.16 and
block-floating dense outputs.  The workload intentionally saturates all 32
output channels so the trap report has to preserve an exact overflow count
rather than only a collapsed Boolean.

| Path | Workload | Median | Raw evidence |
|------|----------|-------:|--------------|
| Python mixed `precision_trap_report` | 64×32 dense, 2,000 calls × 7 repeats | 44.744 µs/call | `benchmarks/results/local_python_2026-06-04_precision_traps.json` |
| Python BFP `precision_trap_report` | 64×32 dense, 2,000 calls × 7 repeats | 45.906 µs/call | `benchmarks/results/local_python_2026-06-04_precision_traps.json` |
| Rust mixed `PrecisionTrapReport` | 64×32 dense, 20,000 calls × 7 repeats | 2.506 µs/call | `benchmarks/results/local_rust_2026-06-04_precision_traps.json` |
| Rust BFP `PrecisionTrapReport` | 64×32 dense, 20,000 calls × 7 repeats | 8.777 µs/call | `benchmarks/results/local_rust_2026-06-04_precision_traps.json` |
| HDL `sc_precision_overflow_trap` Yosys stat | Default `TRAP_WIDTH=1` | 3 cells, 8 wire bits | `hdl/reports/yosys_precision_overflow_trap_2026-06-04.json` |

The committed Python and Rust trap workloads both report
`mixed_overflow_count=32` and `bfp_overflow_count=32`, matching the number of
output channels.  The HDL trap primitive synthesises to one `$adff`, one
`$mux`, and one `$or` cell at the default width.
The 2026-06-05 rerun also records matched sub-LSB underflow probes:
`mixed_underflow_count=32` and `bfp_underflow_count=32` in both Python and
Rust, while the saturating overflow workloads retain `underflow_count=0`.

### Precision Envelope Reports (2026-06-04)

This benchmark covers the conservative predeployment envelope path for mixed
Q8.8/Q16.16 and block-floating dense outputs.  The Python and Rust workloads
use matched fixed-point codes and report the same maximum absolute bounds:
`132850` for the mixed dense workload and `78032768` for the block-floating
workload.

| Path | Workload | Median | Raw evidence |
|------|----------|-------:|--------------|
| Python mixed `precision_envelope_report` | 64×32 dense, 2,000 calls × 7 repeats | 87.578 µs/call | `benchmarks/results/local_python_2026-06-04_precision_envelopes.json` |
| Python BFP `precision_envelope_report` | 64×32 dense, 2,000 calls × 7 repeats | 90.475 µs/call | `benchmarks/results/local_python_2026-06-04_precision_envelopes.json` |
| Rust mixed `PrecisionEnvelopeReport` | 64×32 dense, 20,000 calls × 7 repeats | 2.991 µs/call | `benchmarks/results/local_rust_2026-06-04_precision_envelopes.json` |
| Rust BFP `PrecisionEnvelopeReport` | 64×32 dense, 20,000 calls × 7 repeats | 9.874 µs/call | `benchmarks/results/local_rust_2026-06-04_precision_envelopes.json` |
| HDL `sc_precision_envelope_guard` Yosys stat | Default `N_OUTPUTS=32` | 67 cells, 1,701 wire bits | `hdl/reports/yosys_precision_envelope_guard_2026-06-04.json` |

Both Python and Rust envelope reports returned
`conservative_overflow_free=true` for the committed safe workload.  The HDL
guard synthesises to two `$adff`, thirty-two `$gt`, thirty-two `$mux`, and one
`$reduce_or` cell at the default width.
The raw artefacts now include `observed_underflow_free=true` for the safe
workload and matched underflow probes with `underflow_count=32` for mixed and
BFP dense paths in both Python and Rust.  The refreshed manifests expose
`proof_kind=signed_symmetric_fixed_point_width`: mixed dense requires `19`
signed total bits, `3` Q16.16 integer bits, and has `13` bits of headroom,
while block-floating dense requires `28` signed total bits, `12` Q16.16
integer bits, and has `4` bits of headroom.

---

## 5. Full Pipeline (Python)

encode → AND synapse → popcount → LIF neuron

| Configuration | Iterations | Latency (µs) | Throughput |
|---------------|-----------|---------------|------------|
| 4 synapses, 256 steps | 200 | 1,830 | 139.9 Kstep/s |
| 16 synapses, 256 steps | 50 | 8,679 | 29.5 Kstep/s |

---

## 6. GPU Backend

### 6a. Local CPU fallback (NumPy, no CuPy)

| Operation | Iterations | Latency (µs) | Throughput |
|-----------|-----------|---------------|------------|
| gpu_pack_bitstream (65,536) | 2,000 | 375.9 | 0.17 Gbit/s |
| gpu_vec_mac (64×32×16w) | 1,000 | 736.4 | 2.85 GOP/s |

### 6b. Cloud GPU — NVIDIA RTX A6000 (48 GB, CUDA 12.6)

Environment: JarvisLabs A6000, Xeon Silver 4216 (64 vCPU), PyTorch 2.6.0+cu124.
1000 ms simulation, 3 runs, AI regime (conn_prob=0.1).

| Neurons | Synapses | Wall (s) | Rate (Hz) | Syn events/s | Peak RSS |
|--------:|:--------:|:--------:|:---------:|:------------:|:--------:|
| 1,000 | 100K | 1.55 | 99.0 | 3.2 M | 12 MB |
| 2,000 | 400K | 1.80 | 85.5 | 9.5 M | 24 MB |
| 5,000 | 2.5M | 2.74 | 63.6 | 29.0 M | 104 MB |
| 20,000 | 40M | 8.80 | 26.1 | 59.2 M | 775 MB |
| 50,000 | 250M | 35.4 | 14.7 | 51.9 M | 4,793 MB |

Source: `benchmarks/results/jarvislabs_a6000/gpu_large_scale.json`,
`benchmarks/results/jarvislabs_a6000/scaling_4regime.json`.

---

## 7. Rust Engine — Criterion Results

All benchmarks run with `cargo bench --manifest-path engine/Cargo.toml` on
AVX-512 hardware. Times are Criterion medians.

> **Updated 2026-04-05** with UpCloud EPYC 9575F measurements. Previous values
> (25.4 µs, 446 µs) were from an unidentified earlier run.

### Bitstream Packing (1M bits = 1,048,576 bits)

| Variant | Time | Throughput | vs. Python |
|---------|------|------------|------------|
| `pack` (scalar) | 897 µs | 1.17 Gbit/s | 2.2× |
| `pack_fast` (u64 chunks) | 286 µs | 3.67 Gbit/s | 7× |
| `pack_dispatch` (AVX-512) | 8.85 µs | **113 Gbit/s** | **46.6×** |

### Popcount (16,384 u64 words = 1M bits)

| Variant | Time | Throughput | vs. Python |
|---------|------|------------|------------|
| `popcount_portable` | 12.1 µs | 86.6 Mword/s | 2.5× |
| `popcount_simd` (AVX-512) | 2.86 µs | **366 Mword/s** | **10.6×** |

### Fused AND+Popcount (16 words)

| Variant | Time |
|---------|------|
| Scalar (iter + count_ones) | 19.1 ns |
| SIMD dispatch (AVX-512) | 9.58 ns |

### Encoder / Neuron

| Operation | Time | Throughput |
|-----------|------|------------|
| LFSR encoder (64K steps) | 131 µs | 500 Mstep/s |
| LIF neuron (10K steps) | 47.9 µs | 209 Mstep/s |
| LIF neuron (100K steps) | 219 µs | 456 Mstep/s |

### Bernoulli Encoding (1,024-bit packed streams)

| Variant | Time | Notes |
|---------|------|-------|
| `bernoulli_stream` (unrolled) | 3.99 µs | generate bits then pack |
| `bernoulli_stream` + pack | 4.79 µs | two-pass |
| `bernoulli_packed` (ChaCha8) | 4.14 µs | direct packed generation |
| `bernoulli_packed_fast` (ChaCha8) | 1.72 µs | optimized threshold loop |
| `bernoulli_packed_simd` (ChaCha8) | 779 ns | SIMD comparison |
| `bernoulli_packed_simd` (Xoshiro) | **398 ns** | fastest: SIMD + fast PRNG |
| `encode_and_popcount` (Xoshiro) | **285 ns** | fused encode+AND+popcount |

### Dense Layer (64 inputs → 32 neurons, L=1024)

| Variant | Time | vs. Python |
|---------|------|------------|
| `forward` (baseline) | 1.22 ms | 2.0× |
| `forward_fast` (packed) | 337 µs | **7.1×** |
| `forward_fused` (encode+AND+pop) | 1.67 ms | 1.4× |
| `forward_prepacked` (pre-encoded) | 54.9 µs | **43.8×** |
| `forward_batch` (100 samples) | 13.7 ms | 17.6× per sample |

### PRNG Fill (1,024 bytes)

| Generator | Time | Throughput |
|-----------|------|------------|
| ChaCha8 | 320 ns | 3.13 GB/s |
| Xoshiro256++ | 191 ns | 5.24 GB/s |

### Domain-Specific

| Operation | Time |
|-----------|------|
| Kuramoto solver (100 osc, 1000 steps) | 199 ms |
| Stochastic attention (10×16 → 20×32) | 138 µs |
| Graph layer (20 nodes, 8 features) | 253 µs |

---

## 8. v2 (Python) vs v3 (Rust) Speedup

SIMD tier: avx512-vpopcntdq. The v3 engine wraps Rust via PyO3.

| Operation | v2 (ms) | v3 (ms) | Speedup |
|-----------|---------|---------|---------|
| pack_bitstream (1M bits) | 8.26 | 52.66 | 0.2× |
| popcount (1M bits) | 0.12 | 0.44 | 0.3× |
| LIF neuron (10K steps) | 8.58 | 4.48 | 1.9× |
| Dense forward (16→8, L=1024) | 0.49 | 0.18 | 2.7× |
| Dense forward (64→32, L=1024) | 4.48 | 4.05 | 1.1× |
| Dense forward (128→64, L=1024) | 8.02 | 1.10 | **7.3×** |
| Attention (10×16 → 20×32) | 0.03 | 0.28 | 0.1× |
| Attention (50×32 → 100×64) | 0.10 | 2.19 | 0.0× |

**Geometric mean speedup: 0.5×** — across all operations, the Rust FFI path is
slower than pure Python on average. PyO3 call overhead (argument marshalling,
GIL release/acquire) adds ~50–200 µs per invocation, which dominates when the
payload is small.

The Rust engine amortises FFI cost above ~64K bits per call. On payloads >1M
bits the SIMD kernel wins decisively (Dense 128→64 at 7.3×). For small
networks (<64 neurons), pure Python is faster. The Rust engine targets
large-payload inference (>=128 neurons, L>=1024).

The pure-Rust Criterion numbers in Section 7 show true engine throughput
without FFI overhead.

---

## 9. NeuroBench-Aligned Metrics

Aligned with the NeuroBench methodology (Yik et al., 2023; arXiv:2304.04640).

| Model | Neurons | SynOps | Act. Sparsity | Latency (µs) | Throughput (MOP/s) | Memory (B) |
|-------|--------:|-------:|--------------:|-------------:|-------------------:|-----------:|
| SCDenseLayer(8×4, L=256) | 4 | 409,600 | 0.00 | 1,293 | 6.3 | 256 |
| SCDenseLayer(16×8, L=512) | 8 | 1,966,080 | 0.00 | 2,446 | 26.8 | 1,024 |
| VectorizedSCLayer(16×8, L=512) | 8 | 3,276,800 | 0.00 | 348 | 188.1 | 1,024 |
| VectorizedSCLayer(64×32, L=1024) | 32 | 41,943,040 | 0.00 | 2,476 | **847.0** | 16,384 |

Activation sparsity is 0.00 because SC outputs are graded probabilities, not
binary spikes — every neuron produces a non-zero output on every step.

---

## 10. SNN Comparison: Brunel Balanced Network

### 4-Variant Translator Benchmark

1000 neurons (800E/200I), conn_prob=0.1, adapted params (weight_exc=5.0 mV,
external_rate=200 Hz), 1000 ms simulation. Delta-PSC semantics: synaptic
events applied as instantaneous voltage jumps (v += w), matching Brian2's
`on_pre="v_post += w"`.

| Variant | Spikes | Rate (Hz) | Brian2 Ratio | Wall (s) |
|---------|-------:|----------:|-------------:|---------:|
| Brian2 reference | 1,057,908 | 1057.9 | 1.00 | 1.11 |
| V1 StochasticLIF | 1,725,955 | 1726.0 | 1.63 | 30.23 |
| V2 RateMatched | N/A | 0.049 (prob) | — | 51.81 |
| V3 FixedPoint Q8.8 | 1,722,195 | 1722.2 | 1.63 | 15.41 |
| V4 Hybrid SC+LIF | 1,888,351 | 1888.4 | 1.78 | 46.23 |

#### Variant descriptions

- **V1 StochasticLIF**: Bug-fixed delta-PSC wiring. Previous benchmark passed
  input through `R * I * dt` (diluted by dt=0.1) and omitted `v_reset`. Fixed:
  synaptic events as `neuron.v += weight`, Poisson drive as voltage kicks,
  `v_reset=10.0` passed correctly.
- **V2 RateMatched**: VectorizedSCLayer in probability domain. Weights mapped
  to `p = w / v_threshold`. 100-neuron subset, bitstream_length=1024. Not
  spike-comparable; mean output probability = 0.0488.
- **V3 FixedPoint Q8.8**: Hardware-faithful FixedPointLIFNeuron. Params mapped
  to Q8.8 integers (scale=256). Rate 1.63x Brian2 (higher due to different noise model).
- **V4 Hybrid SC+LIF**: BitstreamSynapse AND gates → popcount → voltage →
  StochasticLIFNeuron. Higher rate due to stochastic amplification in the
  bitstream encoding.

#### Historical note (v3.9.0, resolved)

Prior to v3.10.0, three wiring bugs prevented the Brunel network
from firing. All three were fixed in v3.10.0:
1. `v_reset` never passed (defaulted to 0.0 instead of 10.0)
2. Delta-PSC diluted through `R * I * dt` instead of direct `v += w`
3. Poisson drive fed as steady current instead of voltage kicks

### 10b. 20-Variant Brunel Translator Suite

Adapted Brunel parameters (weight_exc=5.0, ext_rate=200 Hz), 1000 neurons,
1000 ms simulation. Brian2 2.10.1 reference: 748,777 spikes, 748.8 Hz.

| # | Variant | Spikes | Rate (Hz) | Brian2 Ratio | Wall (s) | Note |
|---|---------|-------:|----------:|-------------:|---------:|------|
| — | Brian2 reference | 748,777 | 748.8 | 1.00 | 1.60 | |
| V1 | StochasticLIF | 1,725,955 | 1726.0 | 2.31 | 49.33 | delta-PSC baseline |
| V2 | RateMatched | — | 0.0488 (prob) | — | 80.98 | probability domain |
| V3 | FixedPoint Q8.8 | 1,722,195 | 1722.2 | 2.30 | 20.79 | hardware-faithful |
| V4 | Hybrid SC+LIF | 1,571,994 | 1572.0 | 2.10 | 42.46 | bitstream synapse |
| V5 | Izhikevich | 15,331 | 15.3 | 0.02 | 11.49 | burst dynamics |
| V6 | Homeostatic LIF | 1,727,113 | 1727.1 | 2.31 | 39.36 | adaptive threshold |
| V7 | Noisy LIF | 1,714,361 | 1714.4 | 2.29 | 44.62 | noise_std=1.0 |
| V8 | Refractory LIF | 114,317 | 114.3 | 0.15 | 26.56 | 5-step refractory |
| V9 | Post-kick LIF | 1,671,636 | 1671.6 | 2.23 | 36.16 | Brian2 timing |
| V10 | Exact-leak LIF | 1,713,399 | 1713.4 | 2.29 | 32.78 | exp(-dt/tau) |
| V11 | Q16.12 FixedPoint | 464,644 | 464.6 | 0.62 | 18.19 | 32-bit, 12 frac |
| V12 | STDP LIF | 1,689,552 | 1689.6 | 2.26 | 758.21 | 2000 STDP synapses |
| V13 | DotProduct LIF | 497,647 | 9952.9 | — | 261.40 | n=50, bl=256 |
| V14 | Sobol bitstream | 780,390 | 780.4 | 1.04 | 220.46 | low-discrepancy |
| V15 | JAX vectorized | — | — | — | — | skipped: JAX not installed |
| V16 | Recurrent reservoir | — | 0.9997 (prob) | — | 16.21 | probability domain |
| V17 | Memristive defects | — | 48.7 | — | 51.62 | stuck=1%, var=5% |
| V18 | Numba JIT | 1,685,521 | 1685.5 | 2.25 | **5.20** | **9.5× vs V1** |
| V19 | PyTorch CUDA | 1,725,955 | 1726.0 | 2.31 | **5.70** | GTX 1060 6GB |
| V20 | Vectorized NumPy | 1,725,955 | 1726.0 | 2.31 | **10.27** | batch update |
| V21 | Sparse Numba (CSR) | 1,685,521 | 1685.5 | 2.25 | **0.49** | 10% connectivity |

#### Acceleration comparison (1000 neurons, 1000 ms)

| Backend | Wall (s) | Speedup vs V1 |
|---------|-------:|------:|
| V1 per-neuron Python | 49.33 | 1.0× |
| V20 vectorized NumPy | 10.27 | 4.8× |
| V18 Numba JIT | 5.20 | 9.5× |
| V19 PyTorch CUDA (GTX 1060) | 5.70 | 8.7× |
| V21 Sparse Numba (CSR) | 0.49 | **100.7×** |
| Brian2 (Cython) | 1.60 | 30.8× |

#### 10K neuron scaling

| Backend | Wall (s) | Memory |
|---------|-------:|--------|
| V18 Numba JIT (dense) | 15.3 | 800 MB (N²×8) |
| V21 Sparse Numba (CSR) | 22.6 | 80 MB (10% nnz) |
| Brian2 (C++ codegen) | 9.6 | sparse (internal) |

At 10K, Brian2's compiled C++ sparse codegen wins. V21 CSR reduces memory 10×
but scattered index access prevents SIMD vectorization. The Rust SIMD CSR
engine (planned) targets this gap.

#### Variant notes

- **V5 Izhikevich**: Low spike rate expected — Izhikevich dynamics (quadratic
  nonlinearity, v range -65 to +30) respond differently to delta-PSC drive.
  Tonic baseline current of 5.0 added for sub-threshold depolarization.
- **V8 Refractory**: 5-step (0.5 ms) dead time reduces max firing rate to
  ~2000 Hz, cutting observed rate by 15×.
- **V11 Q16.12**: Higher precision fixed-point produces fewer spikes than Q8.8
  due to more accurate leak computation (less rounding-induced depolarization).
- **V12 STDP**: Online weight learning with 2000 STDP synapses. 15× slower
  due to per-synapse process_step() calls.
- **V14 Sobol**: Low-discrepancy bitstream achieves 1.04× Brian2 ratio — closest
  match to reference among all spiking variants.
- **V18/V19/V20**: Acceleration variants show 5–10× speedup over per-neuron
  Python loop. Numba JIT and PyTorch CUDA achieve similar wall times on this
  workload (1000 neurons); GPU advantage grows with N.
- **V21 Sparse Numba**: scipy.sparse CSR connectivity. At 1K (10% connectivity):
  100× faster than V1, 3× faster than V18 dense. At 10K: 1.5× slower than V18
  due to scattered CSR index access, but uses 10× less memory (80 MB vs 800 MB).

---

## 11. Advanced Module Performance

| Module | Configuration | Latency (100 runs) | Per-run | Key metric |
|--------|---------------|--------------------:|--------:|------------|
| Quantum-Classical Hybrid | 64 qubits, L=1024 | 76.8 ms | 0.77 ms | cos²(θ/2) error < 0.03 |
| Event-Based GNN | 100 nodes, 5% density | 6.6 ms | 0.07 ms | 17× sparse reduction |
| Stochastic Transformer | d=64, 4 heads, L=512 | 1,691 ms | 16.9 ms | 196× energy vs FP32 MAC |
| BCI Decoder | 64 ch, 1s signal | 53.1 ms | 0.53 ms | Native bitstream encoding |
| DVS Input Layer | 128×128, 1000 events | 1,953 ms | 19.5 ms | 492× data reduction |
| Chaotic RNG | 100K samples | 13.5 ms | — | 7.42 Msample/s |
| Predictive World Model | 32-dim state, 50-step | 34.5 ms | 0.34 ms | 1000× sample efficiency |

---

## 12. FPGA Resource Utilization

Synthesis tooling (`tools/yosys_synth.py`) targets Xilinx 7-series via Yosys
`synth_xilinx`. Yosys is not installed on this machine; run when available:

```bash
python tools/yosys_synth.py --json benchmarks/results/yosys_synth.json --markdown
```

CI runs this command with `--allow-skips` so timeout-limited hosted runners
still publish `benchmarks/results/yosys_synth.json` as evidence. A skipped
module is not a hardware timing claim; it records that the bounded CI runner
could not complete that module inside the configured synthesis timeout. Local
or release-grade FPGA evidence should rerun without `--allow-skips` on an
isolated synthesis host.

Target modules: `sc_bitstream_encoder`, `sc_lif_neuron`, `sc_bitstream_synapse`,
`sc_dotproduct_to_current`, `sc_firing_rate_bank`, `sc_dense_layer_core`,
`sc_neurocore_top`.

Estimated: `sc_bitstream_encoder` < 100 LUTs (pending Yosys validation).

---

## 13. Bitstream Length Scaling (32x16 Dense)

Fixed network: 32 inputs, 16 neurons. Mean of 5 runs per length.
Expected: roughly linear scaling (2x L = 2x time).

| L | Mean Time (ms) | Throughput (Mbit/s) |
|--:|---------------:|--------------------:|
| 128 | 0.43 | 151 |
| 256 | 0.66 | 197 |
| 512 | 1.05 | 250 |
| 1024 | 1.14 | 459 |
| 2048 | 1.36 | 773 |
| 4096 | 4.22 | 497 |

Scaling is sub-linear up to L=2048 due to NumPy vectorization amortizing
fixed overhead. At L=4096, packed array allocation begins to dominate.

---

## 14. Memory Footprint (L=1024)

Peak allocation measured via `tracemalloc` (includes layer construction
and one forward pass). Weight matrix size is the float64 weight array only.

| Config | Weight Matrix (MB) | Peak Alloc (MB) | Forward Time (ms) |
|--------|-------------------:|----------------:|-------------------:|
| 32x16 (tiny) | 0.004 | 4.63 | 3.1 |
| 64x32 (small) | 0.016 | 18.33 | 7.5 |
| 128x64 (medium) | 0.062 | 73.13 | 13.2 |
| 256x128 (large) | 0.250 | 292.31 | 26.2 |

Peak allocation scales as O(N_neurons * N_inputs * L / 8) bytes for the
packed bitstream arrays, which dominate the weight matrix by ~1000x.

---

## 15. Reproducing

```bash
# Python benchmark suite (quick ~15s, full ~120s)
python benchmarks/benchmark_suite.py --full --markdown

# Rust Criterion benchmarks (~5 min)
cargo bench --manifest-path engine/Cargo.toml

# v2 vs v3 comparison (requires Rust wheel)
PYTHONPATH=src python benchmarks/bench_v2_vs_v3.py

# NeuroBench-aligned metrics
python benchmarks/neurobench_harness.py --json benchmarks/results/neurobench.json --markdown

# SNN comparison — 20 variants (requires brian2: pip install brian2)
python benchmarks/snn_comparison.py --all --adapted --sim-ms 1000 \
    --json benchmarks/results/snn_translator_20v.json --markdown

# Advanced modules
python benchmarks/benchmark_advanced_modules.py

# FPGA synthesis (requires yosys in PATH)
python tools/yosys_synth.py --json benchmarks/results/yosys_synth.json --markdown

# CI artifact mode: record timeout skips without treating hosted-runner
# synthesis incompletion as a design failure.
python tools/yosys_synth.py --json benchmarks/results/yosys_synth.json --markdown --allow-skips
```

---

## 12. snnTorch Head-to-Head Comparison

**Artifact:** `benchmarks/results/snntorch_vs_sc_microbench.json`

Three-way comparison: SC-NeuroCore (NumPy), SC-NeuroCore (Rust SIMD), snnTorch 0.9.4.

| Test | SC NumPy (us/step) | SC Rust SIMD (us/step) | snnTorch (us/step) |
|------|-------------------|----------------------|-------------------|
| Single neuron (1000 steps) | **3.7** | — | 876 |
| Dense 100->50 (500 steps) | 2,280 | **1,059** | 1,103 |
| Scale 500->500 (100 steps) | 158,741 | **17,473** | 35,998 |
| Scale 1000->1000 (50 steps) | 602,730 | 28,882 | **9,421** |

**Paradigm difference:** SC-NeuroCore performs bit-true stochastic computation
(uint64 popcount on packed bitstreams, L=256-512 bits per value). snnTorch does
float32 matrix multiply. SC-NeuroCore is hardware-faithful (maps directly to
Verilog RTL); snnTorch is GPU-optimized but not synthesizable.

- At small scale (1-100 neurons), SC-NeuroCore's zero-overhead Python step
  is 237x faster than snnTorch's PyTorch dispatch overhead.
- At medium scale (500 neurons), Rust SIMD engine is 2x faster than snnTorch.
- At large scale (1000+), snnTorch's O(n^2) float matmul beats bitstream
  packing at O(n^2 * L).
- Rust engine provides 9-21x speedup over Python SC at all scales.

```bash
python benchmarks/snntorch_vs_sc_microbench.py --runs 5 --scales 100 500 1000
```

---

## 16. Spike Codec Library (2026-03-25)

Compression ratios for the spike codec library. All codecs lossless.
Measured on (2000 x 64) rasters at various firing rates.

### ISI Codec vs General-Purpose Compressors

Auto entropy selection (varint for sparse, Huffman for dense):

| Firing Rate | ISI (auto) | zlib-9 | lzma | ISI Advantage |
|-------------|-----------|--------|------|---------------|
| 0.1% | **401x** | 359x | 194x | +12% over zlib |
| 1% | **78x** | 65x | 48x | +20% over zlib |
| 5% | **24x** | 19x | 20x | +28% over zlib |
| 10% | **16x** | 12x | 13x | +30% over zlib |
| 30% | **8.8x** | 7.0x | 7.8x | +24% over zlib |

### Context Predictor on Structured Data

Periodic bursting (32ch, 5-spike bursts every 50 steps):

| Predictor | Ratio | Accuracy |
|-----------|-------|----------|
| ISI (no prediction) | 8.6x | — |
| EMA | 8.5x | 90.0% |
| **Context (Markov)** | **25.5x** | 97.8% |

### Realistic SpikeInterface Benchmarks

SpikeInterface ground-truth recordings with physiological ISI distributions:

| Scenario | Channels | Units | Firing Rate | Best Ratio |
|----------|----------|-------|-------------|-----------|
| Neuropixels-like | 96 | 10 | 1-5 Hz | 457x |
| BCI-scale | 256 | 50 | 0.5-3 Hz | 756x |
| High-density | 384 | 100 | 1-10 Hz | 317x |

All above Neuralink 200x target.

### Yosys Synthesis (gate counts)

Generic gate-level synthesis via Yosys 0.63:

| Verilog Module | Cells | Function |
|---------------|------:|----------|
| `sc_bitstream_encoder.v` | 115 | LFSR predictor (bit-true with Python/Rust) |
| `sc_cordiv.v` | 2 | Stochastic division |
| `sc_dotproduct_to_current.v` | 448 | AND accumulation + popcount |
| `sc_aer_encoder.v` | 1,423 | Priority encoder for AER |
| `sc_event_neuron.v` | 2,135 | Event-driven LIF |
| `sc_lif_neuron.v` | 3,134 | Q8.8 fixed-point LIF |

1024-channel codec estimate: ~406K gates, ~0.02 mm^2 at 7nm.

### WaveformCodec: Raw Electrode Compression

End-to-end pipeline: raw 10-bit ADC -> spike detect -> template match -> compress.
Measured on synthetic 1024-channel, 1 second at 20 kHz:

| Metric | Value |
|--------|-------|
| Raw data | 40,960,000 bytes (328 Mbit/s) |
| Compressed (q=4) | 1,703,435 bytes (13.6 Mbit/s) |
| **Compression ratio** | **24x** |
| Spikes detected | 3,087 |
| Templates learned | 16 |
| Bluetooth capacity | 15 Mbit/s |
| **Fits in uplink** | **YES** |

Scaling (4-bit background quantization):

| Channels | Raw Mbit/s | Compressed Mbit/s | Fits BT |
|----------|-----------|-------------------|---------|
| 128 | 26 | 1.0 | YES |
| 256 | 51 | 2.0 | YES |
| 384 | 77 | 3.0 | YES |
| 1024 | 205 | 8.0 | YES |
| 3072 | 614 | 23.9 | NO |

Competitive comparison (raw waveform compression):

| Method | Compression | Notes |
|--------|------------|-------|
| MuSCoRE (2023) | 50-100x | Multi-scale decomposition, academic |
| CREST (2022) | 10-50x | Raw electrode, academic |
| **SC-NeuroCore WaveformCodec** | **24x** | Spike-aware pipeline, open source |
| Delta + arithmetic (standard) | 5-15x | No spike awareness |

---

## Notes

- Python benchmarks run in `--full` mode (10x iterations vs quick).
- Rust benchmarks use Criterion defaults (100 samples, 3s warmup).
- v2 vs v3 comparison shows PyO3 FFI overhead for small payloads; Section 7
  reports true Rust throughput without FFI.
- Brian2 installed with numpy 2.4.2 (its requirement); benchmarks run after
  downgrading to numpy 1.26.4 for sc-neurocore compatibility.

## Timing-aware formal framework (2026-06-04)

Artefact: `benchmarks/results/local_python_2026-06-04_timing_formal_framework.json`.

This benchmark exercises the NEU-C.2 timing formal framework across the active polyglot proof surfaces. The run was executed under runtime core isolation with `host_context.cgroup_effective_cpuset=10-11` and `runtime_cpuset_shield_claimed=true`; the workstation load average during the run was `1.96, 2.40, 3.13`. These timings should not be compared against unloaded baselines unless the same isolated-core condition is reproduced.

| Surface | Operation | Result |
| --- | --- | --- |
| SystemVerilog | Dense-layer timing monitors proved through SymbiYosys/cvc5 | pass, `1.476097` s |
| Python | TimingProperty construction and proof orchestration | 16 properties |
| nuXmv | bounded transition-model emission | 16 models, `0.000016258` s, runtime unavailable locally |
| Kind 2 | Lustre bounded-node emission | 16 models, `0.000012825` s, runtime unavailable locally |

Evidence boundary: this is formal proof and model-emitter evidence, not hardware throughput evidence. `hardware_measurement_claimed=false` remains intentional.

## ADC-to-spike quantiser (2026-06-04)

Artefact: `benchmarks/results/local_python_2026-06-04_adc_to_spike_quantiser.json`.

This benchmark exercises the NEU-C.5 ADC-to-spike sensor-ingress contract across the active Python and SystemVerilog surfaces. The run used runtime core isolation with `host_context.cgroup_effective_cpuset=10-11` and `runtime_cpuset_shield_claimed=true`; load average during the run was `3.71, 3.80, 3.48`.

| Surface | Operation | Result |
| --- | --- | --- |
| Python | Bit-true ADC decimation and deterministic AER rate-code reference | `3704.696` ns/sample over `409600` samples |
| SystemVerilog | SymbiYosys/cvc5 formal proof | pass, `4.579` s |
| SystemVerilog | Yosys generic synthesis estimate | `7675` cells, `11.861` s |

Evidence boundary: this is local contract, formal, and synthesis-estimate evidence. `hardware_measurement_claimed=false` remains intentional until board-level isolated hardware evidence is captured.

## DCLS Q8.8 RTL contract (2026-06-04)

Artefacts: `benchmarks/results/local_python_2026-06-04_dcls_q88.json` and
`benchmarks/results/local_rust_2026-06-04_dcls_q88.json`.

This benchmark exercises the NEU-C.6 DCLS Q8.8 scalar layer contract across
Python, PyTorch, Rust, and SystemVerilog.  The Python/SystemVerilog run used
runtime core isolation with `host_context.cgroup_effective_cpuset=10-11` and
`runtime_cpuset_shield_claimed=true`; load average during the run was
`3.43, 3.66, 3.20`.  The Rust run used the same isolated unit and recorded CPU
affinity `10-11`.

| Surface | Operation | Result |
| --- | --- | --- |
| Python | Bit-true DCLS Q8.8 tent-kernel reference | `6349.497` ns/sample over `409600` samples |
| PyTorch | Quantised deterministic parity reference | 5/5 cases passed, max accumulator diff `0` |
| Rust | Bit-true DCLS Q8.8 reference | `40.184` ns/sample median over `409600` samples x 7 repeats |
| SystemVerilog | SymbiYosys/cvc5 bounded formal check | pass, `1.533` s |
| SystemVerilog | Yosys generic synthesis estimate | `106003` cells, `105.897` s |

Evidence boundary: this is local contract, bounded-formal, and
synthesis-estimate evidence.  `hardware_measurement_claimed=false` remains
intentional.  The Vivado ZU3EG WNS/utilisation contract is gated behind
`MIF_VIVADO_CI=1` and is not claimed until the self-hosted Vivado runner
archives a passing timing summary.

## UltraScale+ target contract (2026-06-04)

This benchmark exercises the NEU-C.1 Zynq UltraScale+ target contract across the
Python Vivado-project generator and Rust SystemVerilog emitter/resource model.
Both runs used runtime core isolation: system/user/init slices were moved off
the benchmark cores, the benchmark ran in `benchmark.slice`, and the raw
artefacts record CPU affinity or cgroup cpuset evidence for CPUs 10-11.

| Surface | Contract | Result |
| --- | --- | --- |
| Python + Vivado Tcl | Manifest validation and deterministic ZU3EG/ZU9EG batch Tcl generation | `122678.065` ns/manifest median over 2 manifests x 2000 iterations x 7 repeats |
| Rust | Target-aware SystemVerilog emission and conservative resource reporting for a 64x32 dense graph | `130835.757` ns/emit median over 2000 iterations x 7 repeats |

The Rust report estimates `2048` DSPs for a one-DSP-per-MAC 64x32 dense graph.
That exceeds the ZU3EG budget of `360`, while the BRAM estimate `2` fits the
budget `216`. This over-budget DSP result is intentional fail-closed evidence:
SC-NeuroCore must not claim that this unfurled graph fits ZU3EG until a folded
or time-multiplexed dense implementation is added and validated.

The generated Tcl and Rust target metadata use the Zynq UltraScale+ `DSP48E2`
primitive baseline. The checked-in XDC files are clock/timing baselines only;
they intentionally avoid `PACKAGE_PIN` and `LOC` constraints until a
board-revision pin manifest is verified.

## UltraScale+ dense folding (2026-06-04)

This benchmark exercises the resource-safe fold plan added after the ZU3EG
target benchmark proved that an unfurled 64x32 dense layer would require 2,048
DSPs. The shared Python and Rust planners use row-group folding: five output
rows are processed per cycle with all 64 input lanes live, using 320 DSPs per
compute cycle and completing the 32 output rows in seven cycles.

| Surface | Contract | Result |
| --- | --- | --- |
| Python + SystemVerilog | Planner parity plus bounded 8x8 HDL elaboration for `sc_dense_folded_q88_core` | `2447.444` ns/plan median over 20000 iterations x 7 repeats; Yosys reports 240 generic cells |
| Rust | `SvTarget::dense_fold_plan(64, 32)` | `6.661` ns/plan median over 20000 iterations x 7 repeats |

Both runs used the runtime cpuset shield on CPUs 10-11. The Yosys evidence is a
bounded parameterised elaboration check, not a Vivado ZU3EG utilisation report.
The folded HDL core implements deterministic Q8.8-weight/Q16.16-MAC dense
execution and is covered by Icarus simulation; it must be selected deliberately
by deployment code and does not silently replace the existing stochastic dense
path.
