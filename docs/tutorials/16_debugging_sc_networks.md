<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Debugging and Profiling SC Networks

Diagnose common failure modes in stochastic computing networks:
silent accuracy loss, bitstream correlation, weight collapse, and
performance bottlenecks.

**Prerequisites**: `pip install sc-neurocore matplotlib`

## 1. The debugging challenge

SC networks fail silently. A broken conventional DNN produces NaN or
Inf — obvious errors. A broken SC network produces plausible-looking
but wrong firing rates. The bitstream noise masks bugs.

Common failure modes:

| Symptom | Likely cause | Section |
|---------|-------------|---------|
| All neurons fire at ~50% | Weights collapsed to 0.5 | §2 |
| Accuracy drops with more layers | Bitstream correlation | §3 |
| Output is always 0 or always 1 | Degenerate encoding | §4 |
| Training doesn't converge | Learning rate too high for L | §5 |
| Rust engine gives different results | Seed mismatch | §6 |

## 2. Weight collapse detection

Weights drifting to the same value kills network expressivity:

```python
import numpy as np
from sc_neurocore import VectorizedSCLayer

layer = VectorizedSCLayer(n_inputs=50, n_neurons=128, length=256)

def diagnose_weights(weights, name="layer"):
    """Report weight distribution health."""
    mean = weights.mean()
    std = weights.std()
    unique = len(np.unique(np.round(weights, 4)))
    near_zero = (weights < 0.05).sum()
    near_one = (weights > 0.95).sum()
    total = weights.size

    print(f"{name}:")
    print(f"  mean={mean:.4f}  std={std:.4f}  unique={unique}")
    print(f"  near 0: {near_zero}/{total} ({100*near_zero/total:.1f}%)")
    print(f"  near 1: {near_one}/{total} ({100*near_one/total:.1f}%)")

    if std < 0.05:
        print(f"  WARNING: weight collapse (std < 0.05)")
    if near_zero + near_one > 0.5 * total:
        print(f"  WARNING: >50% weights at rails")

diagnose_weights(layer.weights, "Initial")
```

**Fix**: Reduce learning rate. Add weight decay toward 0.5. Clip
weights to [0.05, 0.95] instead of [0.01, 0.99].

## 3. Bitstream correlation diagnosis

SC arithmetic assumes independent bitstreams. Correlated streams
produce biased results:

```python
from sc_neurocore import BitstreamEncoder

# Bad: same seed → perfectly correlated streams
enc_a = BitstreamEncoder(length=256, seed=42)
enc_b = BitstreamEncoder(length=256, seed=42)  # same seed!

bits_a = enc_a.encode(0.7)
bits_b = enc_b.encode(0.3)

# AND of correlated streams ≠ 0.7 * 0.3
product_corr = (bits_a & bits_b).mean()
print(f"Correlated:   AND mean = {product_corr:.3f} (expected 0.21)")

# Good: different seeds → independent streams
enc_c = BitstreamEncoder(length=256, seed=42)
enc_d = BitstreamEncoder(length=256, seed=137)  # different seed

bits_c = enc_c.encode(0.7)
bits_d = enc_d.encode(0.3)

product_indep = (bits_c & bits_d).mean()
print(f"Independent:  AND mean = {product_indep:.3f} (expected 0.21)")
```

**Detection**: Compare observed product vs expected product across
many test values. Systematic bias indicates correlation.

**Fix**: Ensure every encoder/LFSR in the network has a unique seed.
The `VectorizedSCLayer` handles this automatically.

## 4. Degenerate encoding detection

Values at 0.0 or 1.0 produce all-zero or all-one bitstreams —
these carry no information through AND gates:

```python
def check_encoding_range(inputs, name="inputs"):
    """Verify inputs are in valid SC range."""
    zeros = (inputs <= 0.0).sum()
    ones = (inputs >= 1.0).sum()
    near_zero = (inputs < 0.01).sum()
    near_one = (inputs > 0.99).sum()
    total = inputs.size

    print(f"{name}: range [{inputs.min():.4f}, {inputs.max():.4f}]")
    if zeros > 0:
        print(f"  CRITICAL: {zeros} exact zeros (all-zero bitstreams)")
    if ones > 0:
        print(f"  CRITICAL: {ones} exact ones (all-one bitstreams)")
    if near_zero > 0.1 * total:
        print(f"  WARNING: {near_zero} values < 0.01 ({100*near_zero/total:.0f}%)")
    if near_one > 0.1 * total:
        print(f"  WARNING: {near_one} values > 0.99 ({100*near_one/total:.0f}%)")

# Example: raw pixel values often include exact 0s and 1s
test_data = np.random.rand(50)
test_data[0] = 0.0   # bad
test_data[1] = 1.0   # bad
check_encoding_range(test_data, "raw input")

# Fix: clamp to safe range
safe_data = np.clip(test_data, 0.01, 0.99)
check_encoding_range(safe_data, "clamped input")
```

## 5. Learning rate vs bitstream length

SC noise standard deviation is 1/√L. If the learning rate is larger
than the noise floor, training is unstable:

```python
def recommend_lr(bitstream_length, n_inputs):
    """Recommend learning rate based on SC noise floor."""
    # SC noise std ≈ 1/sqrt(L) per output
    # Gradient noise ≈ noise_std * sqrt(n_inputs)
    noise_std = 1.0 / np.sqrt(bitstream_length)
    grad_noise = noise_std * np.sqrt(n_inputs)

    # LR should be smaller than gradient noise / 10
    recommended = grad_noise / 10
    max_safe = grad_noise / 3

    print(f"L={bitstream_length}, N_in={n_inputs}")
    print(f"  SC noise std:     {noise_std:.4f}")
    print(f"  Gradient noise:   {grad_noise:.4f}")
    print(f"  Recommended LR:   {recommended:.4f}")
    print(f"  Max safe LR:      {max_safe:.4f}")
    return recommended

for L in [64, 128, 256, 512]:
    recommend_lr(L, 50)
    print()
```

## 6. Cross-backend verification

Verify Python and Rust produce the same results:

```python
def verify_backends(layer, test_input, tolerance=0.05):
    """Compare Python vs Rust output for the same input."""
    py_out = layer.forward(test_input)

    try:
        from sc_neurocore_engine import DenseLayer as RustDenseLayer
        rust_out = RustDenseLayer(layer.weights, layer.length).forward(test_input)
        max_diff = np.max(np.abs(py_out - rust_out))
        print(f"Python vs Rust max diff: {max_diff:.4f}")
        if max_diff > tolerance:
            divergent = np.where(np.abs(py_out - rust_out) > tolerance)[0]
            print(f"  Divergent neurons: {divergent}")
            print(f"  Python: {py_out[divergent]}")
            print(f"  Rust:   {rust_out[divergent]}")
    except ImportError:
        print("Rust engine not available — skipping backend comparison")

test_input = np.random.uniform(0.1, 0.9, size=50)
verify_backends(layer, test_input)
```

**Note**: Stochastic outputs will differ between runs (different
LFSR sequences). Compare averaged outputs over multiple runs,
not single-run values.

## 7. Layer-by-layer forward pass inspection

Trace values through the network to find where things go wrong:

```python
def trace_forward(layers, x, names=None):
    """Print statistics at each layer."""
    if names is None:
        names = [f"Layer {i}" for i in range(len(layers))]

    print(f"Input: mean={x.mean():.3f} std={x.std():.3f} "
          f"range=[{x.min():.3f}, {x.max():.3f}]")

    for layer, name in zip(layers, names):
        x = layer.forward(np.clip(x, 0.01, 0.99))
        dead = (x < 0.02).sum()
        saturated = (x > 0.98).sum()
        print(f"{name}: mean={x.mean():.3f} std={x.std():.3f} "
              f"range=[{x.min():.3f}, {x.max():.3f}] "
              f"dead={dead} sat={saturated}")
    return x

layer1 = VectorizedSCLayer(n_inputs=50, n_neurons=128, length=256)
layer2 = VectorizedSCLayer(n_inputs=128, n_neurons=64, length=256)
layer3 = VectorizedSCLayer(n_inputs=64, n_neurons=10, length=256)

x = np.random.uniform(0.1, 0.9, size=50)
trace_forward([layer1, layer2, layer3], x)
```

Look for:
- **Mean collapsing to 0.5**: weight collapse
- **Std shrinking to 0**: all outputs identical
- **Many dead/saturated neurons**: encoding problems

## 8. Performance profiling

```python
import time

def profile_layer(layer, n_samples=100):
    """Measure forward pass time."""
    x = np.random.uniform(0.1, 0.9, size=layer.weights.shape[1])

    # Warmup
    for _ in range(5):
        layer.forward(x)

    start = time.perf_counter()
    for _ in range(n_samples):
        layer.forward(x)
    elapsed = time.perf_counter() - start

    samples_per_sec = n_samples / elapsed
    us_per_sample = elapsed / n_samples * 1e6
    print(f"  {layer.weights.shape[1]}→{layer.weights.shape[0]}, L={layer.length}: "
          f"{us_per_sample:.0f} μs/sample, {samples_per_sec:.0f} samples/s")

print("Layer performance:")
for n_in, n_out, L in [(50, 128, 256), (128, 64, 256), (50, 128, 512), (50, 128, 1024)]:
    layer = VectorizedSCLayer(n_inputs=n_in, n_neurons=n_out, length=L)
    profile_layer(layer)
```

Throughput scales as O(n_in × n_out × L / 64) due to packed uint64
operations. If you need more speed, use the Rust engine.

## 9. Diagnostic checklist

Run before every training session:

```python
def preflight_check(layers, test_input):
    """Run all diagnostics on a network."""
    print("=" * 50)
    print("SC Network Preflight Check")
    print("=" * 50)

    # 1. Input range
    check_encoding_range(test_input, "Input")
    print()

    # 2. Weight health per layer
    for i, layer in enumerate(layers):
        diagnose_weights(layer.weights, f"Layer {i}")
    print()

    # 3. Forward trace
    trace_forward(layers, test_input)
    print()

    # 4. LR recommendation
    recommend_lr(layers[0].length, test_input.shape[0])
    print()

    print("Preflight complete.")

preflight_check([layer1, layer2, layer3], x)
```

## What you learned

- SC networks fail silently — always run diagnostics before training
- Weight collapse: std < 0.05 means all weights are nearly identical
- Bitstream correlation: shared LFSR seeds cause biased arithmetic
- Degenerate encoding: values at 0 or 1 produce uninformative bitstreams
- LR must be smaller than SC noise floor (1/√L × √N_in) / 10
- Trace forward passes to find where signal degrades
- Profile with `time.perf_counter()`, compare L and layer sizes

## Next steps

- Add the preflight check to your training script
- Use `BitstreamSpikeRecorder` to log full spike trains for offline analysis
- Compare Rust engine output against Python for your specific network
- Profile memory usage for large networks (>10K neurons)
