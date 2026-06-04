# SC-NeuroCore Performance Benchmarks

Generated on: 2026-03-07 00:40:28
Backend: NumPy (CPU only)

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
| Mixed dense Q8.8/Q16.16 (64x32) | Python | 2000 | 31.330 us | max abs error 7.63e-05, safe_bound=531401 |
| Mixed dense Q8.8/Q16.16 overflow telemetry (64x32) | Python | 2000 | 39.562 us | safe=0, saturating probe=32 |
| Mixed dense Q8.8/Q16.16 (64x32) | Rust | 20000 | 2.986 us | safe=0, saturating probe=32, safe_bound=531400 |
| Mixed dense Q8.8/Q16.16 lane telemetry | HDL/Yosys | N_OUTPUTS=32 | 12,708 cells | `overflow_vector` + `abs_bounds_q1616` registered |
| Block-floating dense BFP16E3X32/Q16.16 (64x32) | Python | 2000 | 29.453 us | max abs error 0.2231, safe_bound=610816 |
| Block-floating dense BFP16E3X32/Q16.16 overflow telemetry (64x32) | Python | 2000 | 42.582 us | safe=0, saturating probe=32 |
| Block-floating dense BFP16E3X32/Q16.16 (64x32) | Rust | 20000 | 9.116 us | safe=0, saturating probe=32, safe_bound=312131072 |
| Block-floating dense BFP16E3X32/Q16.16 lane telemetry | HDL/Yosys | 2x2 parameterised report | 96 cells | `overflow_vector` + `abs_bounds_q1616` registered |
| Precision trap report mixed dense (64x32 overflow) | Python | 2000 | 340.461 us | overflow_count=32 |
| Precision trap report mixed dense (64x32 overflow) | Rust | 20000 | 3.483 us | overflow_count=32 |
| Precision trap report block-floating dense (64x32 overflow) | Python | 2000 | 161.677 us | overflow_count=32 |
| Precision trap report block-floating dense (64x32 overflow) | Rust | 20000 | 11.277 us | overflow_count=32 |
| Precision overflow trap latch | HDL/Yosys | TRAP_WIDTH=1 | 3 cells | `$adff`+`$mux`+`$or` |
| Precision envelope report mixed dense (64x32 safe) | Python | 2000 | 150.131 us | max_abs_bound=132850 |
| Precision envelope report mixed dense (64x32 safe) | Rust | 20000 | 3.887 us | max_abs_bound=132850 |
| Precision envelope report block-floating dense (64x32 safe) | Python | 2000 | 140.462 us | max_abs_bound=78032768 |
| Precision envelope report block-floating dense (64x32 safe) | Rust | 20000 | 13.510 us | max_abs_bound=78032768 |
| Precision envelope guard | HDL/Yosys | N_OUTPUTS=32 | 67 cells | `$adff`+`$gt`+`$mux`+`$reduce_or` |
| Full pipeline (4 syn, 256 steps) | cpu | 200 | 1830.0 us | 139.9 Kstep/s |
| Full pipeline (16 syn, 256 steps) | cpu | 50 | 8678.5 us | 29.5 Kstep/s |
| gpu_pack_bitstream (65536) | cpu | 2000 | 375.9 us | 0.17 Gbit/s |
| gpu_vec_mac (64x32x16w) | cpu | 1000 | 736.4 us | 2.85 GOP/s |
