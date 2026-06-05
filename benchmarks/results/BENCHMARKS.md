# SC-NeuroCore Performance Benchmarks

Generated on: 2026-03-07 00:40:28
Backend: NumPy (CPU only)

Note: the 2026-06-04 local Python/Rust precision benchmark rows were captured on
a workstation under concurrent load and without exclusive CPU core isolation.
Use them as committed contract/regression evidence only. Production throughput
claims require a rerun on isolated cores with recorded CPU affinity, host-load,
governor, and frequency evidence. The live-control update rerun below is pinned
to CPUs `8-9` by process affinity and records that affinity in the raw artefact;
it is not a kernel-reserved isolated-core claim.

| Benchmark | Backend | Iterations | Avg Latency | Throughput |
|-----------|---------|------------|-------------|------------|
| LFSR step (16-bit) | cpu | 1000000 | 0.8 us | 1.33 Mstep/s |
| Bitstream encoder step | cpu | 1000000 | 0.9 us | 1.10 Mstep/s |
| LIF neuron step (Q8.8) | cpu | 1000000 | 0.9 us | 1.07 Mstep/s |
| pack_bitstream 1-D (1024) | cpu | 10000 | 8.7 us | 0.12 Gbit/s |
| pack_bitstream 1-D (65536) | cpu | 2000 | 123.1 us | 0.53 Gbit/s |
| pack_bitstream 2-D (64x1024) | cpu | 2000 | 121.6 us | 0.54 Gbit/s |
| vec_and (1024 words) | cpu | 50000 | 1.6 us | 40.99 Gbit/s |
| vec_popcount SWAR (1024 words) | cpu | 50000 | 30.2 us | 2.17 Gbit/s |
| Dense forward (16x8, L=256) | cpu | 500 | 352.7 us | 0.09 GOP/s (SC) |
| Dense forward (64x32, L=1024) | cpu | 100 | 2405.8 us | 0.87 GOP/s (SC) |
| Mixed dense Q8.8/Q16.16 (64x32) | Python | 2000 | 31.634 us | max abs error 0, safe_bound=531400 |
| Mixed dense Q8.8/Q16.16 overflow telemetry (64x32) | Python | 2000 | 28.136 us | safe=0, saturating probe=32 |
| Mixed dense Q8.8/Q16.16 (64x32) | Rust | 20000 | 2.245 us | safe=0, saturating probe=32, safe_bound=531400 |
| Mixed dense Q8.8/Q16.16 lane telemetry | HDL/Yosys | N_OUTPUTS=32 | 12,708 cells | `overflow_vector` + `abs_bounds_q1616` registered |
| Block-floating dense BFP16E3X32/Q16.16 (64x32) | Python | 2000 | 31.429 us | max abs error 0.2231, safe_bound=610816 |
| Block-floating dense BFP16E3X32/Q16.16 overflow telemetry (64x32) | Python | 2000 | 30.326 us | safe=0, saturating probe=32 |
| Block-floating dense BFP16E3X32/Q16.16 (64x32) | Rust | 20000 | 10.056 us | safe=0, saturating probe=32, safe_bound=610816 |
| Block-floating dense BFP16E3X32/Q16.16 lane telemetry | HDL/Yosys | 2x2 parameterised report | 96 cells | `overflow_vector` + `abs_bounds_q1616` registered |
| Precision trap report mixed dense (64x32 overflow) | Python | 2000 | 52.810 us | overflow_count=32 |
| Precision trap report mixed dense (64x32 overflow) | Rust | 20000 | 2.325 us | overflow_count=32 |
| Precision trap report block-floating dense (64x32 overflow) | Python | 2000 | 50.173 us | overflow_count=32 |
| Precision trap report block-floating dense (64x32 overflow) | Rust | 20000 | 8.669 us | overflow_count=32 |
| Precision overflow trap latch | HDL/Yosys | TRAP_WIDTH=1 | 3 cells | `$adff`+`$mux`+`$or` |
| Precision envelope report mixed dense (64x32 safe) | Python | 2000 | 82.388 us | max_abs_bound=132850 |
| Precision envelope report mixed dense (64x32 safe) | Rust | 20000 | 2.322 us | max_abs_bound=132850 |
| Precision envelope report block-floating dense (64x32 safe) | Python | 2000 | 79.117 us | max_abs_bound=78032768 |
| Precision envelope report block-floating dense (64x32 safe) | Rust | 20000 | 8.748 us | max_abs_bound=78032768 |
| Precision envelope guard | HDL/Yosys | N_OUTPUTS=32 | 67 cells | `$adff`+`$gt`+`$mux`+`$reduce_or` |
| Live-control parameter update sequence | Python+SystemVerilog | 20000 | 13.128 us AXI4-Lite; 12.366 us PCIe-MMIO | process affinity `8-9`, CRC32 update guard, checksum-mismatch, invalid-selection, and read-only-bank traps, AXI trap simulation passed, PCIe commit simulation passed |
| AER strict-priority queue backpressure | Python+SystemVerilog | 4096 events x 100 repeats | 4.138 us/event | runtime cpuset shield 10-11, priority=0 violations, FIFO=0 violations, drop/deadline traps latched |
| ADC-to-spike quantiser | Python+SystemVerilog | 4096 samples x 100 repeats | 3.705 us/sample | cpuset 10-11, formal pass, Yosys 7675 cells |
| DCLS Q8.8 tent-kernel layer | Python+PyTorch+SystemVerilog | 4096 samples x 100 repeats | 6.349 us/sample | cpuset 10-11, PyTorch parity 5/5, formal pass, Yosys 106003 cells |
| DCLS Q8.8 tent-kernel layer | Rust | 4096 samples x 100 iterations x 7 repeats | 40.184 ns/sample | cpuset 10-11, overflow_count=0, active_tap_total=2808700 |
| UltraScale+ target contract | Python+Vivado Tcl | 2 manifests x 2000 iterations x 7 repeats | 122.678 us/manifest | cpuset 10-11, ZU3EG/ZU9EG Tcl, DSP48E2 baseline |
| UltraScale+ target contract | Rust | 64x32 graph x 2000 iterations x 7 repeats | 130.836 us/emit | cpuset 10-11, DSP estimate 2048 > ZU3EG budget 360, BRAM 2 <= 216 |
| UltraScale+ dense folding | Python+SystemVerilog | 64x32 plan x 20000 iterations x 7 repeats | 2.447 us/plan | cpuset 10-11, 320 DSP/cycle, 7 compute cycles, bounded Yosys 240 cells |
| UltraScale+ dense folding | Rust | 64x32 plan x 20000 iterations x 7 repeats | 6.661 ns/plan | cpuset 10-11, 320 DSP/cycle, 7 compute cycles |
| Full pipeline (4 syn, 256 steps) | cpu | 200 | 1830.0 us | 139.9 Kstep/s |
| Full pipeline (16 syn, 256 steps) | cpu | 50 | 8678.5 us | 29.5 Kstep/s |
| gpu_pack_bitstream (65536) | cpu | 2000 | 375.9 us | 0.17 Gbit/s |
| gpu_vec_mac (64x32x16w) | cpu | 1000 | 736.4 us | 2.85 GOP/s |

## Timing-aware formal framework - 2026-06-04

| Artefact | Cpuset | Surfaces | Key result |
| --- | --- | --- | --- |
| `local_python_2026-06-04_timing_formal_framework.json` | `10-11` | Python, SystemVerilog, nuXmv, Kind 2 | SymbiYosys/cvc5 pass in `1.476097` s; 16 nuXmv models and 16 Kind 2 nodes emitted |

The run records `hardware_measurement_claimed=false` and `runtime_cpuset_shield_claimed=true`. The nuXmv and Kind 2 binaries were not installed locally, so the benchmark validates deterministic model emission for those surfaces and records runtime execution as unavailable.

## ADC-to-spike quantiser - 2026-06-04

| Artefact | Cpuset | Surfaces | Key result |
| --- | --- | --- | --- |
| `local_python_2026-06-04_adc_to_spike_quantiser.json` | `10-11` | Python, SystemVerilog | `3704.696` ns/sample; SymbiYosys/cvc5 pass in `4.579` s; Yosys `7675` cells |

The run records `hardware_measurement_claimed=false` and `runtime_cpuset_shield_claimed=true`. This is local contract evidence, not board-level throughput evidence.

## DCLS Q8.8 RTL contract - 2026-06-04

| Artefact | Cpuset | Surfaces | Key result |
| --- | --- | --- | --- |
| `local_python_2026-06-04_dcls_q88.json` | `10-11` | Python, PyTorch, SystemVerilog | `6349.497` ns/sample; PyTorch parity 5/5; SymbiYosys/cvc5 bounded check in `1.533` s; Yosys `106003` cells |
| `local_rust_2026-06-04_dcls_q88.json` | `10-11` | Rust | `40.184` ns/sample median; `overflow_count=0`; `active_tap_total=2808700` |

The run records `hardware_measurement_claimed=false` and
`runtime_cpuset_shield_claimed=true` for the Python/SystemVerilog artefact.
This is local contract, bounded-formal, and synthesis-estimate evidence, not
Vivado ZU3EG timing evidence.

## UltraScale+ target contract - 2026-06-04

| Artefact | Cpuset | Surfaces | Key result |
| --- | --- | --- | --- |
| `local_python_2026-06-04_ultrascale_plus_target.json` | `10-11` | Python, Vivado Tcl | `122.678` us/manifest median; deterministic ZU3EG/ZU9EG project Tcl; `DSP48E2` primitive baseline |
| `local_rust_2026-06-04_ultrascale_plus_target.json` | `10-11` | Rust | `130.836` us/emit median; 64x32 dense graph estimates `2048` DSPs against the ZU3EG budget of `360`; BRAM estimate `2` fits budget `216` |

Both artefacts record runtime cpuset evidence and `hardware_measurement_claimed=false` where applicable. This is target-contract and resource-budget evidence, not board-level Vivado timing closure. The DSP over-budget result is intentional fail-closed evidence that the 64x32 dense contract requires folding/time-multiplexing or a larger target before it can be claimed as a ZU3EG implementation.

## UltraScale+ dense folding - 2026-06-04

| Artefact | Cpuset | Surfaces | Key result |
| --- | --- | --- | --- |
| `local_python_2026-06-04_ultrascale_dense_folding.json` | `10-11` | Python, SystemVerilog | `2.447` us/plan median; 64x32 plan uses 320 DSPs per compute cycle and completes in 7 cycles; bounded 8x8 Yosys elaboration reports 240 cells |
| `local_rust_2026-06-04_ultrascale_dense_folding.json` | `10-11` | Rust | `6.661` ns/plan median; same 320-DSP, 7-cycle fold contract |

Both artefacts record runtime cpuset evidence. This is deterministic
resource-planning and HDL-elaboration evidence. It does not claim Vivado
board-level timing closure or replace the generic stochastic dense path.

## Live-control AXI4-Lite / PCIe-MMIO register window - 2026-06-04

| Artefact | Cpuset | Surfaces | Key result |
| --- | --- | --- | --- |
| `local_python_2026-06-04_live_control_updates.json` | process affinity `8-9` | Python, SystemVerilog, AXI4-Lite, PCIe-MMIO | AXI4-Lite staged-update sequence median `13127.835` ns; PCIe-MMIO staged-update sequence median `12365.626` ns; AXI trap simulation and PCIe commit simulation both passed |

The PCIe surface is a register-window adapter contract over the same staged
parameter-bank core used by AXI4-Lite. The update guard is
`crc32-ieee-le-4x32`, computed over bank select, entry index, low data word, and
high data word. A stale CRC32 guard raises sticky `checksum_mismatch` trap bit
`0x4`, and an out-of-range bank/entry selection raises sticky
`invalid_selection` trap bit `0x8`. A write to a valid but read-only bank raises
sticky `read_only_bank` trap bit `0x10`, before any shadow load can occur. It does
not claim a full PCIe hard-IP endpoint implementation.
The same simulation retargets the selection registers after a valid shadow load
and verifies that commit still applies the originally accepted bank and entry.
Upstream PCIe hard IP must decode MMIO transactions into the single-clock
strobes exposed by the generated wrapper.
