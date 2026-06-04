# SC-NeuroCore Performance Benchmarks

Generated on: 2026-03-07 00:40:28
Backend: NumPy (CPU only)

Note: the 2026-06-04 local Python/Rust precision benchmark rows were captured on
a workstation under concurrent load and without exclusive CPU core isolation.
Use them as committed contract/regression evidence only.  Production throughput
claims require a rerun on isolated cores with recorded CPU affinity, host-load,
governor, and frequency evidence.

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
| Live-control parameter update sequence | Python+SystemVerilog | 20000 | 19.501 us | static_regeneration=109.204 us, generated trap capture passed |
| AER strict-priority queue backpressure | Python+SystemVerilog | 4096 events x 100 repeats | 4.138 us/event | runtime cpuset shield 10-11, priority=0 violations, FIFO=0 violations, drop/deadline traps latched |
| ADC-to-spike quantiser | Python+SystemVerilog | 4096 samples x 100 repeats | 3.705 us/sample | cpuset 10-11, formal pass, Yosys 7675 cells |
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
