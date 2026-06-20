# Public SC inference — `sc_forward`

`sc_neurocore.accel.sc_forward` is the stable public entry point for running a
unipolar stochastic matrix-vector product over **caller-owned, pre-packed weight
bitstreams**. It is the supported replacement for the removed
`get_backend` / `VectorizedSCLayer(W_packed, encoder, backend=...)` integration
that downstream compilers (the SCPN-CONTROL Petri-net `FusionCompiler` fast path)
relied on.

## What it computes

For packed unipolar weights and input probabilities, `sc_forward` encodes the
inputs into bitstreams, ANDs them against the weights, popcounts the result and
divides by the stream length:

```text
out[o] = (Σ_i popcount(W_packed[o, i] & encode(input_probs[i]))) / length
       ≈ (W @ input_probs)[o]      # unbiased estimate, unipolar SC
```

The input encoder is the 16-bit LFSR comparator used by the SC-NeuroCore hardware
path (taps 16, 14, 13, 11; `bit = reg < x_value` then advance). Because it is
deterministic integer arithmetic, the Rust accelerated path and the NumPy fallback
produce **bit-identical** results for a fixed seed — identical to the last bit, not
merely within stochastic tolerance.

## API

```python
from sc_neurocore.accel import sc_forward, get_backend, available_backends

estimate = sc_forward(weights_packed, input_probs, length=4096, backend="auto", seed=0xACE1)
```

| Symbol | Contract |
| --- | --- |
| `weights_packed` | `(n_out, n_in, n_words)` `uint64`, `n_words = ceil(length / 64)` |
| `input_probs` | `(n_in,)` `float64` in `[0, 1]` |
| returns | `(n_out,)` `float64`, the popcount estimate of `weights @ input_probs` |
| `backend` | `"auto"` (fastest available), `"rust"`, `"numpy"`, or a `Backend` handle |
| `get_backend(name="auto")` | returns the active `Backend`; order Rust → Mojo → Julia → Go → NumPy |
| `available_backends()` | `dict[str, bool]` of resolvable backends (NumPy always `True`) |

The caller packs weights once with `sc_neurocore.accel.pack_bitstream` and never
surrenders them to a layer that re-packs internally.

## Accuracy and parity

`sc_forward` is an unbiased estimator: for a single product the estimate lies
within `3·sqrt(p(1−p)/length)` of `p = w·x` with high probability; for a dense row
the error scales with the square root of the fan-in. The accepted contract
(`tests/test_sc_forward.py`) is:

- per-output agreement with the dense float `weights @ input_probs` within the
  stochastic tolerance at `length = 4096`; and
- exact equality between the Rust path and the NumPy fallback for a fixed seed.

## Throughput

Measured on an 11th Gen Intel Core i5-11600K at 3.90 GHz, CPU affinity pinned to
cores 10-11, workload 128 × 128 weights at `length = 4096`, via
`benchmarks/bench_sc_forward.py` (`benchmarks/results/bench_sc_forward.json`):

| Backend | MAC/s | Per-call (ms) | Speedup vs NumPy | Parity |
| --- | --- | --- | --- | --- |
| Rust | 2.39 × 10¹⁰ | 2.810 | 16.5× | bit-identical |
| NumPy | 1.45 × 10⁹ | 46.346 | reference | reference |

The host carried a load average of ≈5.6 during this run, so the absolute
throughput figures are functional and regression evidence only and must be
re-measured on a reserved, quiet host before any production speedup claim. The
bit-identical parity is independent of host load.

## Backward compatibility

`BitstreamEncoder(length=..., seed=...)` is valid again: `x_min`/`x_max` default to
the unipolar probability domain `[0, 1]`, so callers that only set the length and
seed construct a working encoder while explicit ranges remain supported.
