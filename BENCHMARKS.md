# SC-NeuroCore Performance Benchmarks

Generated on: 2026-02-08 21:49:42
Backend: NumPy (CPU only)

| Benchmark | Backend | Iterations | Avg Latency | Throughput |
|-----------|---------|------------|-------------|------------|
| LFSR step (16-bit) | cpu | 100000 | 0.4 us | 2.73 Mstep/s |
| Bitstream encoder step | cpu | 100000 | 0.4 us | 2.27 Mstep/s |
| LIF neuron step (Q8.8) | cpu | 100000 | 0.7 us | 1.38 Mstep/s |
| pack_bitstream 1-D (1024) | cpu | 1000 | 14.7 us | 0.07 Gbit/s |
| pack_bitstream 1-D (65536) | cpu | 200 | 159.9 us | 0.41 Gbit/s |
| pack_bitstream 2-D (64x1024) | cpu | 200 | 134.2 us | 0.49 Gbit/s |
| vec_and (1024 words) | cpu | 5000 | 1.1 us | 61.48 Gbit/s |
| vec_popcount SWAR (1024 words) | cpu | 5000 | 36.9 us | 1.78 Gbit/s |
| Dense forward (16x8, L=256) | cpu | 50 | 485.2 us | 0.07 GOP/s (SC) |
| Dense forward (64x32, L=1024) | cpu | 10 | 2609.6 us | 0.80 GOP/s (SC) |
| Full pipeline (4 syn, 256 steps) | cpu | 20 | 1243.0 us | 206.0 Kstep/s |
| Full pipeline (16 syn, 256 steps) | cpu | 5 | 4959.9 us | 51.6 Kstep/s |
| gpu_pack_bitstream (65536) | cpu | 200 | 172.7 us | 0.38 Gbit/s |
| gpu_vec_mac (64x32x16w) | cpu | 100 | 459.8 us | 4.56 GOP/s |
