# Stochastic Doctor

Bitstream-level stochastic correlation analysis and diagnostics engine.
Detects correlation anomalies, precision drift, and produces full
JSON-serializable audit reports per network layer.

## Rust Acceleration

This module includes **optional Rust PyO3 acceleration** via the
`stochastic_doctor_core` crate. When the compiled extension is present,
all hot-path operations dispatch to Rust automatically.

Set `SC_NEUROCORE_NO_RUST=1` to force the pure Python fallback.

### Performance (Python vs Rust PyO3)

Benchmarked on x86_64, Python 3.12, Rust 1.86 (release):

#### SCC — Single Pair

| Stream Length | Python | Rust | Speedup |
|-------------:|-----------:|-----------:|--------:|
| 100 | 12 µs | 0.3 µs | **35×** |
| 1,000 | 29 µs | 0.9 µs | **32×** |
| 10,000 | 42 µs | 4.6 µs | **9×** |
| 100,000 | 128 µs | 40 µs | **3.2×** |
| 1,000,000 | 1,297 µs | 369 µs | **3.5×** |

#### Batch SCC — N×N Pairwise Matrix (stream_len=2048)

| Neurons | Pairs | Python | Rust | Speedup |
|--------:|------:|---------:|---------:|--------:|
| 4 | 6 | 0.10 ms | 0.007 ms | **15×** |
| 8 | 28 | 0.40 ms | 0.025 ms | **16×** |
| 16 | 120 | 1.6 ms | 0.10 ms | **16×** |
| 32 | 496 | 7.0 ms | 0.46 ms | **15×** |
| 64 | 2,016 | 26.8 ms | 1.5 ms | **18×** |

#### Precision Estimation

| Stream Length | Python | Rust | Speedup |
|-------------:|-----------:|-----------:|--------:|
| 100 | 3 µs | 0.2 µs | **14×** |
| 1,000 | 4 µs | 0.3 µs | **12×** |
| 10,000 | 7 µs | 0.8 µs | **9×** |
| 100,000 | 32 µs | 7 µs | **5×** |
| 1,000,000 | 304 µs | 61 µs | **5×** |

### Criterion Benchmarks (Pure Rust)

| Function | Input | Time |
|----------|------:|-----:|
| `scc_bytes` | 100 | 50 ns |
| `scc_bytes` | 1,000 | 397 ns |
| `scc_bytes` | 10,000 | 3.9 µs |
| `scc_bytes` | 100,000 | 48 µs |
| `scc_packed` | 256 bits | 15 ns |
| `scc_packed` | 1,024 bits | 36 ns |
| `scc_packed` | 8,192 bits | 219 ns |
| `scc_packed` | 65,536 bits | 1.7 µs |
| `scc_batch` | 4 streams | 4.6 µs |
| `scc_batch` | 8 streams | 22 µs |
| `scc_batch` | 16 streams | 97 µs |
| `scc_batch` | 32 streams | 398 µs |
| `precision_packed` | 256 bits | 5.3 ns |
| `precision_packed` | 65,536 bits | 519 ns |
| `histogram_u64` | 64 words | 110 ns |
| `histogram_u64` | 4,096 words | 4.6 µs |
| `drift_detector` | 1,000 obs | 4.8 µs |

## Quick Start

```python
from sc_neurocore.stochastic_doctor.diagnostics import (
    StochasticDoctor, DriftDetector, BitstreamAuditReport,
)

import numpy as np

doc = StochasticDoctor(correlation_threshold=0.3, critical_threshold=0.7)

# Audit a layer of bitstreams
bitstreams = np.random.randint(0, 2, size=(8, 2048), dtype=np.uint8)
report = doc.audit_layer("V1_Cortex", bitstreams)

print(report.status)         # OK / WARNING / CRITICAL
print(report.max_correlation)
print(report.to_json())

# Drift monitoring
dd = DriftDetector(alpha=0.1, threshold=0.3)
for scc_value in correlation_stream:
    if dd.observe(scc_value):
        print(f"Drift detected! EMA={dd.ema:.4f}")
```

## Architecture

```
diagnostics.py ──┬── PyO3 (stochastic_doctor_core.so)  ← primary
                 ├── Python/NumPy fallback              ← secondary
                 └── SC_NEUROCORE_NO_RUST=1             ← force Python
```

::: sc_neurocore.stochastic_doctor.diagnostics
