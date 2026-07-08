# ADC-to-spike decimating rate-code encoder

`sc_neurocore.sensors.adc_to_spike_kernel` is a wired, batched integer reference
for the sensor-ingress rate coder: each decimation window of raw ADC samples is
centred and quantised to a Q-format code, sign-aware averaged, and converted into
a deterministic spike rate code. It is the hot per-window compute of the
synthesisable bridge in `hdl/sensors/adc_to_spike_quantiser.v` and the
cycle-stepped golden model in `tools/adc_to_spike_reference.py` (Indiveri 2003
rate coding); the cycle-accurate handshake/drain FSM stays in that reference,
while this kernel is the per-window arithmetic that polyglot backends accelerate.
The public package facade also exports the maintained surface:

```python
from sc_neurocore.sensors import ADCSpikeWindowConfig, adc_to_spike_windows
```

Existing submodule imports remain compatible.

## Contract

For each completed decimation window:

```text
centred  = two's-complement or offset-binary recentred ADC sample
q_sample = Q-format up-shift, or sign-aware round-down, then saturate
total    = Σ q_sample over the window
window   = clamp(trunc(sign_round(total) / decimation))      (toward zero)
spikes   = |window| / threshold                              (floor)
polarity = window < 0
```

Quantisation uses an arithmetic right shift for the round-down (floor for
negatives) and a sign-aware half-offset; the window average rounds half
away-from-zero then truncates toward zero, matching `int(adjusted / decimation)`
in the golden model. The whole path is exact integer arithmetic, so the Python
floor and the Rust, Julia, Go and Mojo backends agree bit-for-bit; the parity
tolerance is exactly zero. A dedicated test also pins the Python floor against the
cycle-stepped golden reference's own `quantise_adc`/`_average_window`.

## Kernel sources

| Backend | File | Build |
| --- | --- | --- |
| Python primary | `src/sc_neurocore/sensors/adc_to_spike_kernel.py` | — (floor reference) |
| Rust | `engine/src/adc_to_spike.rs` + `py_adc_to_spike_windows` | `maturin develop --release` |
| Julia | `src/sc_neurocore/accel/julia/adc_to_spike/adc_to_spike.jl` | `juliacall` (lazy include) |
| Go | `src/sc_neurocore/accel/go/adc_to_spike/adc_to_spike.go` | `go build -buildmode=c-shared` |
| Mojo | `src/sc_neurocore/accel/mojo/adc_to_spike/adc_to_spike.mojo` | `mojo build --emit shared-lib` |

`adc_to_spike_windows(samples, config, backend="auto")` dispatches fastest-first
or takes an explicit `backend=` name; `available_backends()` reports which
artefacts are present. The `ADCSpikeWindowConfig` carries the ADC width, Q-format,
decimation, signed/offset-binary flag and spike threshold. The result carries
`window_values_q` (`int32`), `spike_counts` (`int32`) and `polarities` (`bool`),
one entry per completed window.

## Cross-language parity and throughput

Measured on an 11th Gen Intel Core i5-11600K at 3.90 GHz, CPU affinity pinned to
cores 10-11, workload 65 536 windows of decimation 8 (524 288 ADC samples), via
`benchmarks/bench_adc_to_spike_kernel.py`
(`benchmarks/results/bench_adc_to_spike_kernel.json`):

| Backend | Samples/s | Per-call (ms) | Speedup vs Python | Parity |
| --- | --- | --- | --- | --- |
| Mojo | 5.53 × 10⁸ | 0.948 | 653× | bit-exact (Δ = 0) |
| Rust | 4.45 × 10⁸ | 1.178 | 526× | reference |
| Julia | 2.98 × 10⁸ | 1.758 | 352× | bit-exact (Δ = 0) |
| Go | 0.92 × 10⁸ | 5.692 | 109× | bit-exact (Δ = 0) |
| Python | 8.5 × 10⁵ | 618.965 | reference | reference |

Unlike the dense matmul kernels, this is a per-sample scalar loop with no BLAS
path, so the compiled backends are two to three orders of magnitude faster than
the Python floor; Mojo and Rust lead. The host carried a load average of ≈5.4
during this run, so the absolute throughput figures are functional and regression
evidence only and must be re-measured on a reserved, quiet host before any
production speedup claim. The zero parity delta is independent of host load.

## Tests

| File | Verifies |
| --- | --- |
| `tests/test_adc_to_spike_kernel.py` | Quantisation across the three width regimes, saturation, sign-aware averaging, validation, dispatch and fallback |
| `tests/test_adc_to_spike_kernel_parity.py` | Bit-exact parity of every built backend against the Python floor, and of the Python floor against the cycle-stepped golden reference, across five fixed-point configs |
| `engine/src/adc_to_spike.rs` (`#[cfg(test)]`) | Rust quantise/average/rate-code and config/stream validation |

`src/sc_neurocore/sensors/adc_to_spike_kernel.py` is also included in the scoped
public docstring policy. The policy gate now rejects duplicate file entries so
the reported enforced-file count stays tied to unique public modules.
