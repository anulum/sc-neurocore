# GPU Compute Backend (wgpu)

**Module:** `sc_neurocore_engine.GpuDenseLayer`
**Rust path:** `sc_neurocore_engine::gpu::GpuDenseLayer`
**Feature gate:** `--features gpu`
**Backends:** Vulkan (Linux/Windows), Metal (macOS), DX12 (Windows)
**Tested on:** NVIDIA GTX 1060 6GB, AMD RX 6600 XT 8GB

---

## 1. Mathematical Formalism

### Stochastic computing dense layer

The GPU backend accelerates the SC-NeuroCore DenseLayer, which performs
stochastic computing matrix-vector multiplication using Bernoulli-encoded
bitstreams and AND+popcount accumulation.

**Bernoulli encoding:**

$$B_i[k] \sim \text{Bernoulli}(p_i), \quad k = 1, \ldots, L$$

Each input probability $p_i$ is encoded as a bitstream of length $L$ where each
bit is independently 1 with probability $p_i$.

**Stochastic AND multiplication:**

$$Y_{ij}[k] = W_{ij}[k] \wedge B_i[k]$$

where $W_{ij}$ is the pre-encoded weight bitstream and $B_i$ is the input bitstream.
The AND operation implements stochastic multiplication: $\mathbb{E}[Y] = w_{ij} \cdot p_i$.

**Popcount accumulation:**

$$\text{out}_j = \frac{1}{L} \sum_{i=1}^{N_{in}} \text{popcount}(Y_{ij})$$

The output for neuron $j$ is the normalised sum of bit-counts across all inputs.

### GPU kernel architecture

Two compute shader kernels mirror the CPU hot path:

**Kernel 1 — Encode (`encode.wgsl`):**

Each GPU thread generates one packed `vec2<u32>` word (64 bits) of a Bernoulli bitstream.
The PRNG is Philox 4×32-10, a counter-based RNG that enables massively parallel
independent random streams without state sharing.

Dispatch: `(ceil(n_inputs × words_per_input / 256), n_samples, 1)`

**Kernel 2 — Accumulate (`accumulate.wgsl`):**

Each workgroup processes one (neuron, sample) pair. 256 threads cooperatively AND+popcount
all input words for one neuron, then perform workgroup-local tree reduction via shared memory.

Dispatch: `(n_neurons, n_samples, 1)` — one workgroup per (neuron, sample)

### Philox 4×32-10 PRNG

Counter-based RNG producing 4 random u32 values per round:

```
Input: (counter[4], key[2])
10 rounds of: {
    x0 = mulhi(R0, counter[0]) ^ key[0] ^ counter[3]
    x1 = mullo(R0, counter[0])
    x2 = mulhi(R1, counter[2]) ^ key[1] ^ counter[1]
    x3 = mullo(R1, counter[2])
    counter = (x0, x1, x2, x3)
    key += (PHILOX_W0, PHILOX_W1)
}
```

Constants: $R_0 = \text{0xD2511F53}$, $R_1 = \text{0xCD9E8D57}$,
$W_0 = \text{0x9E3779B9}$ (golden ratio), $W_1 = \text{0xBB67AE85}$.

The PRNG is implemented in WGSL with 16-bit decomposition for `mulhilo` since
WGSL has no native 64-bit integer multiplication.

### u64 emulation via vec2<u32>

WGSL has no native u64 type. Bitstreams are packed as `vec2<u32>` (2 × 32 bits = 64 bits).
AND and popcount operate on both halves independently:

```wgsl
let result = vec2<u32>(a.x & b.x, a.y & b.y);
let count = countOneBits(result.x) + countOneBits(result.y);
```

This incurs ~15-25% overhead vs native u64 popcount but is memory-bound anyway,
so the penalty is negligible at large layer sizes.

---

## 2. Theoretical Context

### Problem statement

SC-NeuroCore's single-neuron Rust engine measures 3.8 ns/step for LIF, but
network-level stochastic computing is O(n²) in the number of neurons × inputs,
dominated by Bernoulli encoding and AND+popcount. At 1K+ neurons, CPU performance
is insufficient for real-time simulation:

| Framework | 1K neurons | 10K neurons |
|-----------|-----------|-------------|
| SC-NeuroCore (CPU) | 11 ms | ~1.1 s |
| Brian2 (CPU) | 2.5 ms | 250 ms |
| snnTorch (GPU) | 0.5 ms | 5 ms |
| GeNN (GPU) | 0.3 ms | 3 ms |

The GPU backend closes this gap by offloading the encoding and accumulation
to massively parallel GPU compute shaders.

### Why wgpu?

| Alternative | Pros | Cons |
|-------------|------|------|
| **CUDA** | Fastest on NVIDIA | NVIDIA-only, no AMD support |
| **ROCm** | AMD support | Linux-only, poor API stability |
| **OpenCL** | Cross-platform | Stagnant, poor tooling |
| **Vulkan Compute** | Cross-platform, low-level | Verbose API |
| **wgpu** | **Cross-platform, Rust-native, safe** | **Slight overhead vs raw Vulkan** |
| Metal | Best on macOS | Apple-only |

wgpu was chosen because:
1. **Cross-platform:** Vulkan (Linux/Windows), Metal (macOS), DX12 (Windows)
2. **Rust-native:** First-class Rust API, no FFI overhead
3. **Production-grade:** Powers Firefox WebGPU implementation
4. **AMD support:** User's primary GPUs are AMD RX 6600 XT (RDNA2)
5. **No CUDA dependency:** Avoids vendor lock-in

### GPU-native vs CPU-upload design

Two design alternatives were considered:

**Option A: Upload CPU random bytes to GPU**
- Simple: generate bitstreams on CPU, upload packed words via PCIe
- Bottleneck: PCIe bandwidth (8-16 GB/s) limits throughput at batch scale
- For 1K inputs × 1024-bit bitstreams × 256 batch = 32 MB/batch → 2 ms upload alone

**Option B: GPU-native RNG (chosen)**
- Philox PRNG generates random bits directly on GPU
- Only probabilities (f32, small) and output (f32, small) cross PCIe
- For 1K inputs: upload 4 KB probabilities, download 4 KB output
- Eliminates PCIe bottleneck entirely

### SC_GPU_ADAPTER environment variable

Multi-GPU systems can select a specific adapter via substring match:

```bash
export SC_GPU_ADAPTER="AMD"    # selects AMD GPU
export SC_GPU_ADAPTER="1060"   # selects GTX 1060
export SC_GPU_ADAPTER="RDNA"   # selects RDNA-based GPU
```

If unset, wgpu's `HighPerformance` heuristic selects the most powerful adapter.

---

## 3. Pipeline Position

```
Python / Rust application
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  GpuDenseLayer                                       │
│                                                     │
│  ┌──────────┐     ┌──────────────────────────────┐  │
│  │ CPU      │     │ GPU (wgpu)                   │  │
│  │ DenseLayer│     │                              │  │
│  │ (fallback)│     │  Encode kernel (Philox RNG)  │  │
│  │          │     │       │                      │  │
│  │          │     │  Accumulate kernel (AND+pop)  │  │
│  │          │     │       │                      │  │
│  │          │     │  Output buffer → readback     │  │
│  └──────────┘     └──────────────────────────────┘  │
│                                                     │
│  Weights: uploaded once, persistent on GPU           │
│  Inputs: uploaded per forward() call (f32 probs)     │
│  Output: downloaded per forward() call (f32 values)  │
└─────────────────────────────────────────────────────┘
```

### Inputs

| Input | Type | Range | Description |
|-------|------|-------|-------------|
| `input_values` | `Vec<f64>` or `list[float]` | $[0, 1]$ | Input probabilities per neuron |
| `seed` | `u64` | Any | Philox PRNG seed (different seed = different bitstreams) |
| `n_samples` | `usize` (batch only) | $\geq 1$ | Batch size for batch forward |

### Outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `Vec<f64>` or `list[float]` | Per-neuron | $[0, n_{inputs}]$ | Stochastic dot product output |

---

## 4. Features

| Feature | Description |
|---------|-------------|
| **Cross-platform GPU** | Vulkan, Metal, DX12 via wgpu |
| **Feature-gated** | `--features gpu`, no compile-time cost without |
| **Philox 4×32-10 PRNG** | GPU-native counter-based RNG |
| **Two-kernel architecture** | Encode + Accumulate, matching CPU hot path |
| **Persistent weights** | Uploaded once, reused across forward calls |
| **Batch mode** | `forward_batch()` processes multiple samples in one dispatch |
| **Single-sample mode** | `forward_fast()` for interactive use |
| **SC_GPU_ADAPTER** | Environment variable for GPU selection |
| **CPU fallback** | GpuDenseLayer wraps a CPU DenseLayer internally |
| **Pre-allocated buffers** | No per-call allocation up to max_batch |
| **PyO3 bindings** | `from sc_neurocore_engine import GpuDenseLayer` |
| **Workgroup reduction** | Shared memory tree reduction for popcount |
| **Adapter detection** | Reports GPU name and availability |

---

## 5. Usage Examples

### Python: basic GPU forward

```python
from sc_neurocore_engine import GpuDenseLayer

# Check GPU availability.
if GpuDenseLayer.is_gpu_available():
    layer = GpuDenseLayer(n_inputs=256, n_neurons=128, length=1024, seed=42)
    print(f"GPU: {layer.gpu_name()}")

    # Single-sample forward.
    inputs = [0.5] * 256
    output = layer.forward_fast(inputs, seed=42)
    print(f"Output shape: {len(output)}, first 5: {output[:5]}")
else:
    print("No GPU available — use CPU DenseLayer instead")
```

### Python: batch forward

```python
import numpy as np

layer = GpuDenseLayer(256, 128, length=1024, seed=42, max_batch=512)

# Batch of 64 samples.
n_samples = 64
inputs_flat = np.random.uniform(0.0, 1.0, size=n_samples * 256).tolist()
outputs = layer.forward_batch(inputs_flat, n_samples=n_samples, seed=42)

# Reshape to (n_samples, n_neurons).
import numpy as np
out_array = np.array(outputs).reshape(n_samples, 128)
print(f"Batch output shape: {out_array.shape}")
```

### Rust: GPU forward

```rust
use sc_neurocore_engine::gpu::GpuDenseLayer;

if let Some(layer) = GpuDenseLayer::try_new(256, 128, 1024, 42, 256) {
    let inputs = vec![0.5f64; 256];
    let output = layer.forward_gpu(&inputs, 42);
    println!("GPU output: {:?}", &output[..5]);
    println!("GPU: {}", layer.gpu_name());
}
```

### Python: GPU Kuramoto oscillators

```python
import numpy as np
from sc_neurocore_engine import GpuKuramoto

n = 512
omega = np.zeros(n)                      # identical natural frequencies
coupling = np.full(n * n, 4.0)           # strong uniform all-to-all coupling
phases = (np.arange(n) * 0.4) % (2 * np.pi)

solver = GpuKuramoto()
final = solver.run(n, omega, coupling, phases, n_steps=500, dt=0.02)

# Kuramoto order parameter R -> ~1 under strong coupling (phase locking).
r = abs(np.mean(np.exp(1j * final)))
print(f"R = {r:.3f} on {solver.gpu_name()}")
```

The GPU kernel reproduces the noise-free baseline of the CPU `KuramotoSolver`.
Because WGSL has no `f64`, the GPU math is `f32`, so results agree with the `f64`
CPU solver within tolerance rather than bit-for-bit (unlike the fixed-point LIF
kernel). One thread per oscillator sums its coupling row over all others, so the
O(N²) all-to-all coupling parallelises — this is where the GPU overtakes the
rayon CPU path (see `cargo bench --bench gpu_bench gpu_kuramoto` for the measured
crossover; simple per-neuron LIF, by contrast, stays CPU-favoured).

### Rust: GPU Kuramoto oscillators

```rust
use sc_neurocore_engine::gpu::GpuKuramoto;

if let Some(solver) = GpuKuramoto::try_new() {
    let n = 512;
    let omega = vec![0.0f32; n];
    let coupling = vec![4.0f32; n * n];
    let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.4).collect();
    let final_phases = solver.run(n, &omega, &coupling, &phases, 500, 0.02);
    println!("final[0..3] = {:?} on {}", &final_phases[..3], solver.gpu_name());
}
```

### Python: GPU Izhikevich neurons

```python
import numpy as np
from sc_neurocore_engine import GpuIzhikevichBatch

n_neurons = 1024
n_steps = 300
currents = np.full(n_neurons, 10.0, dtype=np.float32)   # supra-threshold drive

batch = GpuIzhikevichBatch()
# Regular-spiking preset: a=0.02, b=0.2, c=-65, d=8, dt=1.0, v_peak=30.
spikes, voltages = batch.run(n_neurons, n_steps, currents)

print(f"spikes shape {spikes.shape}, total {int(spikes.sum())} on {batch.gpu_name()}")
print(f"voltage trace of neuron 0: {voltages[0, :5]}")
```

Each GPU thread owns one neuron and loops all `n_steps` internally, applying the
two half-steps of the Euler update (for stability on the `0.04·v²` term) then the
`v ≥ v_peak` threshold with reset `v ← c`, `u ← u + d`. It returns row-major
`[n_neurons × n_steps]` arrays of spikes (`int32`) and voltages (`float32`).
Because WGSL has no `f64`, the GPU math is `f32`, so agreement with the `f64` CPU
`Izhikevich` model is tolerance-based (spike counts and the sub-threshold trace),
not bit-for-bit — unlike the fixed-point LIF kernel.

Like LIF, this is O(N) per-neuron work with no inter-neuron coupling, so the rayon
CPU stays ahead across the measured range (256–65 536 neurons the CPU is ~5–9×
faster; see `cargo bench --bench gpu_bench gpu_izhikevich`). The GPU kernel's value
is not a standalone speed win but keeping the neuron state resident on the GPU
inside a larger pipeline (e.g. coupled to the O(N²) Kuramoto kernel, which *does*
overtake the CPU), avoiding host round-trips.

### Rust: GPU Izhikevich neurons

```rust
use sc_neurocore_engine::gpu::GpuIzhikevichBatch;

if let Some(batch) = GpuIzhikevichBatch::try_new() {
    let n_neurons = 1024;
    let n_steps = 300;
    let currents = vec![10.0f32; n_neurons];
    // (a, b, c, d, dt, v_peak) — regular-spiking preset.
    let result = batch.run(n_neurons, n_steps, &currents, 0.02, 0.2, -65.0, 8.0, 1.0, 30.0);
    let total: i32 = result.spikes.iter().sum();
    println!("total spikes {total} on {}", batch.gpu_name());
}
```

### GPU adapter selection

```bash
# List available adapters.
python3 -c "
from sc_neurocore_engine import GpuDenseLayer
layer = GpuDenseLayer(64, 32)
print(f'Selected: {layer.gpu_name()}')
"

# Force AMD GPU.
SC_GPU_ADAPTER="AMD" python3 -c "
from sc_neurocore_engine import GpuDenseLayer
layer = GpuDenseLayer(64, 32)
print(f'Selected: {layer.gpu_name()}')
"
```

### Build with GPU support

```bash
# Compile Rust engine with GPU feature.
cd engine
cargo build --release --features gpu

# Run GPU tests.
cargo test --features gpu

# Run GPU benchmarks.
cargo bench --features gpu --bench gpu_bench
```

### GPU vs CPU comparison

```python
import time
from sc_neurocore_engine import GpuDenseLayer

sizes = [(64, 32), (256, 128), (512, 256), (1000, 500)]
for n_in, n_out in sizes:
    layer = GpuDenseLayer(n_in, n_out, length=1024, seed=42)
    inputs = [0.5] * n_in

    # Warm up.
    for _ in range(10):
        layer.forward_fast(inputs, seed=42)

    # Measure.
    start = time.perf_counter()
    for _ in range(100):
        layer.forward_fast(inputs, seed=42)
    elapsed = (time.perf_counter() - start) / 100
    print(f"{n_in}x{n_out}: {elapsed*1e6:.0f} us/forward")
```

---

## 6. Technical Reference

### Class: `GpuDenseLayer` (Python)

Exposed via PyO3. Module: `sc_neurocore_engine`.

#### Constructor

```python
GpuDenseLayer(n_inputs, n_neurons, length=1024, seed=24301, max_batch=256)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_inputs` | `int` | Required | Number of input features |
| `n_neurons` | `int` | Required | Number of output neurons |
| `length` | `int` | `1024` | Bitstream length (higher = more accurate, slower) |
| `seed` | `int` | `24301` | Weight initialisation seed |
| `max_batch` | `int` | `256` | Maximum batch size (pre-allocates buffers) |

Raises `RuntimeError` if no GPU is available.

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward_fast` | `(inputs: list[float], seed: int) -> list[float]` | Single-sample forward |
| `forward_batch` | `(inputs_flat: list[float], n_samples: int, seed: int) -> list[float]` | Batch forward (flat row-major) |
| `gpu_name` | `() -> str` | Name of the GPU adapter |
| `is_gpu_available` | `() -> bool` (static) | Check GPU availability |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_inputs` | `int` | Number of input features |
| `n_neurons` | `int` | Number of output neurons |
| `length` | `int` | Bitstream length |

### Rust API

```rust
pub struct GpuDenseLayer {
    pub cpu: DenseLayer,
    ctx: Arc<GpuContext>,
    weight_buf: wgpu::Buffer,
    // ... pre-allocated buffers
}

impl GpuDenseLayer {
    pub fn try_new(n_inputs, n_neurons, length, seed, max_batch) -> Option<Self>
    pub fn forward_gpu(&self, inputs: &[f64], seed: u64) -> Vec<f64>
    pub fn forward_batch_gpu(&self, inputs_flat: &[f64], n_samples: usize, seed: u64) -> Vec<f64>
    pub fn gpu_name(&self) -> &str
}
```

### GpuContext singleton

```rust
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub encode_pipeline: wgpu::ComputePipeline,
    pub accumulate_pipeline: wgpu::ComputePipeline,
    pub adapter_name: String,
}
```

Initialised once via `OnceLock`. Shared across all `GpuDenseLayer` instances.

### WGSL shader files

| File | Workgroup size | Description |
|------|---------------|-------------|
| `philox.wgsl` | — | Philox 4×32-10 RNG functions (included by encode) |
| `encode.wgsl` | 256 | Bernoulli encoding: probability → packed bitstream |
| `accumulate.wgsl` | 256 | AND + popcount + workgroup tree reduction |

### Buffer layout

| Buffer | Size | Persistence | Description |
|--------|------|-------------|-------------|
| `weight_buf` | $N_{in} \times N_{out} \times W$ × 8 bytes | Persistent | Packed weight bitstreams |
| `input_prob_buf` | $N_{in} \times B$ × 4 bytes | Per-call | Input probabilities (f32) |
| `packed_input_buf` | $N_{in} \times W \times B$ × 8 bytes | Per-call | Encoded input bitstreams |
| `output_buf` | $N_{out} \times B$ × 4 bytes | Per-call | Output values (f32) |
| `output_staging` | Same as output | Per-call | MAP_READ staging buffer |

where $W = \lceil L/64 \rceil$ (words per bitstream) and $B$ = batch size.

### Memory limits

The `maxStorageBufferBindingSize` on most GPUs is 128-256 MB. For 1000×1000 layers
with length=1024 (16 words): weight buffer = $1000 \times 1000 \times 16 \times 8$ = 128 MB.
Larger layers may require chunking or reduced bitstream length.

---

## 7. Performance Benchmarks

### Local GPU benchmarks (i5-11600K host)

Measured with Criterion micro-benchmarks (commit 27299c8b):

| Layer size | CPU (i5-11600K) | GTX 1060 (x16 PCIe) | RX 6600 XT (x1 PCIe) |
|------------|----------------|---------------------|----------------------|
| 64×32 | 58 µs | 137 µs (0.4×) | 1.0 ms (0.06×) |
| 512×256 | 5.6 ms | 263 µs (**21.3×**) | 1.65 ms (3.4×) |
| 1000×500 | 11.2 ms | 539 µs (**20.9×**) | 2.8 ms (4.0×) |

### Key findings

1. **GPU crossover at ~128 neurons:** Below this, dispatch overhead (~50 µs) dominates
2. **21× speedup at 512+ neurons on GTX 1060** (x16 PCIe 3.0)
3. **RX 6600 XT limited by PCIe x1 riser:** Only 985 MB/s bandwidth vs 15.75 GB/s for x16
4. **Batch mode amortises dispatch:** 256 samples in one dispatch costs ~2× single-sample

### GTX 1060 vs RX 6600 XT

The RX 6600 XT has more compute units (32 vs 10) and higher bandwidth (256 GB/s vs 192 GB/s)
but is connected via a PCIe x1 mining riser, limiting host-GPU transfer to ~985 MB/s.
With a proper x16 connection, the RX 6600 XT would likely outperform the GTX 1060 by 2-3×.

### Scaling behaviour

| Dimension | Complexity | GPU advantage |
|-----------|-----------|---------------|
| $N < 128$ | O(N²) small | CPU faster (dispatch overhead) |
| $128 \leq N < 512$ | O(N²) medium | GPU ~5-10× faster |
| $N \geq 512$ | O(N²) large | GPU ~20× faster (memory-bound) |
| Batch $B$ | O(B × N²) | GPU ~40× (amortised dispatch) |

### Memory bandwidth analysis

For 512×256 layer with length=1024:
- Weight reads: $512 \times 256 \times 16 \times 8$ = 16.7 MB
- Input reads: $512 \times 16 \times 8$ = 65 KB (negligible)
- Output writes: $256 \times 4$ = 1 KB (negligible)

GTX 1060 memory bandwidth: 192 GB/s → theoretical minimum: 16.7 MB / 192 GB/s = 87 µs.
Measured: 263 µs → ~33% memory bandwidth utilisation (reasonable for compute shaders
with workgroup synchronisation overhead).

---

## 8. Citations

1. **wgpu.** "wgpu: Safe and portable GPU abstraction in Rust."
   https://wgpu.rs/ — Cross-platform GPU compute framework.

2. **Philox PRNG.** Salmon, J. K. et al. "Parallel Random Numbers: As Easy as 1, 2, 3."
   SC '11: Proceedings of the International Conference for High Performance Computing,
   Networking, Storage and Analysis, 2011.
   — Counter-based RNG algorithm used for GPU Bernoulli encoding.

3. **Stochastic computing.** Alaghi, A. & Hayes, J. P. "Survey of Stochastic Computing."
   ACM Transactions on Embedded Computing Systems 12(2s):92:1-92:19, 2013.
   — Theoretical foundation for bitstream-based computation.

4. **WGSL specification.** W3C. "WebGPU Shading Language."
   https://www.w3.org/TR/WGSL/ — Shader language used for compute kernels.

5. **GPU-accelerated SNNs.** Knight, J. C. & Nowotny, T. "GPUs outperform current
   HPC and neuromorphic solutions in terms of speed and energy when simulating a
   highly-connected cortical model." Frontiers in Neuroscience 12:941, 2018.
   — Motivation for GPU acceleration of spiking network simulation.

6. **GeNN.** Yavuz, E. et al. "GeNN: a code generation framework for accelerated
   brain simulations." Scientific Reports 6:18854, 2016.
   — GPU-accelerated SNN framework, comparison target.

---

## Validation

### Test suite results (11 GPU integration tests)

All tests passing with `cargo test --features gpu`:

| Test | What it verifies | Status |
|------|-----------------|--------|
| `gpu_layer_creation` | GpuDenseLayer::try_new succeeds | PASS |
| `gpu_output_shape` | Output length = n_neurons | PASS |
| `gpu_output_non_negative` | All outputs ≥ 0 | PASS |
| `gpu_output_range` | All outputs ≤ n_inputs | PASS |
| `gpu_deterministic` | Same seed → same output | PASS |
| `gpu_cpu_agreement` | GPU ≈ CPU within 5% relative | PASS |
| `gpu_batch` | Batch output shape = n_samples × n_neurons | PASS |
| `gpu_zero_inputs` | All-zero inputs → ~zero output | PASS |
| `gpu_all_ones` | All-one inputs → output ≈ sum(weights) | PASS |
| `gpu_different_seeds` | Different seeds → different output | PASS |
| `gpu_adapter_name` | Non-empty adapter name string | PASS |

### GPU-CPU parity

GPU and CPU outputs agree within 5% relative tolerance (at the scale of n_inputs).
The remaining difference arises from:
- Different PRNG (Philox on GPU vs rand on CPU)
- f32 accumulation on GPU vs f64 on CPU
- Different operation ordering (parallel vs sequential)

These differences are within the inherent stochastic noise of the bitstream computation.

---

## Design Decisions

### Why two kernels instead of one fused kernel?

Separating encode and accumulate allows:
1. **Memory efficiency:** Encoded bitstreams can be reused across neurons
2. **Debugging:** Each stage can be tested independently
3. **Flexibility:** Different encode strategies (Bernoulli, LFSR, Sobol) can be
   swapped without changing the accumulate kernel

A fused kernel would reduce global memory traffic but increase register pressure
and limit workgroup occupancy.

### Why f32 on GPU, f64 on CPU?

RDNA2 (RX 6600 XT) has 1:1 f32:f16 ratio but only 1:16 f64 rate.
Pascal (GTX 1060) has 1:32 f64 rate. Since the Bernoulli encoding quantises to
8-bit effective precision anyway (p mapped to [0, 255] threshold), f32 accumulation
is more than sufficient.

### Why persistent weights?

Weights are uploaded once at construction time and stored in a GPU buffer.
They are reused across all forward calls. This amortises the upload cost
(which can be significant for large layers) and keeps the per-call data transfer
minimal (only input probabilities and output values).

### Why OnceLock singleton for GpuContext?

The wgpu Device and Queue are heavyweight objects. Creating them for each
GpuDenseLayer would waste GPU memory and initialisation time. The singleton
ensures all layers share a single device/queue, and the compiled compute
pipelines are also shared.

### Why max_batch pre-allocation?

GPU buffer allocation is expensive (~100 µs). Pre-allocating for the maximum
expected batch size eliminates per-call allocation overhead. If the actual
batch size is smaller, only the used portion is dispatched.

---

## Known Limitations

1. **No auto-chunking:** Layers exceeding the `maxStorageBufferBindingSize`
   (typically 128-256 MB) will fail at creation time. Manual chunking is needed.

2. **No mixed precision:** All GPU computation is f32. No f16 or int8 modes.

3. **No sparse support:** Only dense layers. Sparse connectivity patterns
   still use the full weight matrix.

4. **No gradient computation:** Forward pass only. Training requires CPU
   fallback with surrogate gradients.

5. **No multi-GPU:** Only one adapter is used (selected by SC_GPU_ADAPTER
   or wgpu heuristic). Multi-GPU data parallelism is not supported.

6. **Dispatch overhead:** ~50 µs per kernel dispatch makes GPU slower than
   CPU for small layers (<128 neurons).

7. **PCIe bandwidth dependency:** Performance depends heavily on PCIe link
   speed. x1 risers (mining rigs) significantly reduce throughput.

8. **No timeline profiling:** No built-in GPU profiling. Use external tools
   (Nsight, RenderDoc) for kernel-level analysis.

---

*SC-NeuroCore v3.14.0 — Stochastic Computing Spiking Neural Network Framework*

*© 2020–2026 Miroslav Šotek / ANULUM. AGPL-3.0-or-later.*
