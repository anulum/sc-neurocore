# Autonomous Learning Engine: "Project Zenith" Core

The Autonomous Learning Engine within SC-NEUROCORE represents a beta-grade research-to-runtime C-FFI pipeline mapping low-level physical Rust biological models into deep learning infrastructures and external hardware persistence. Following the "Zenith" update, this toolchain provides strong execution parity bridging PyTorch optimizations with asynchronous neuromorphic hardware constraints.

## 1. What This Solves

Conventional Spiking Neural Network (SNN) frameworks explicitly force engineers to choose between:
1. **PyTorch Tensor Arrays:** Used by frameworks like *snnTorch* or *SpikingJelly*. These calculate accurate surrogate gradients for training but map poorly to unstructured physical hardware and lack native low-level biological efficiency.
2. **Low-Level Software Physics:** Found in bare-metal Rust/C++ simulations and FPGAs. While hyper-fast and memory efficient, these frameworks discard backward derivatives, walling researchers off from `loss.backward()` ML optimizations.

The **Zenith Architecture** eliminates the traditional trade-off, providing native 10-Million parameter scale surrogate gradient compatibility, raw MTJ (Magnetic Tunnel Junction) float-to-stochastic analog emulation, and Exascale memory persistence unbottlenecked by Python.

---

## 2. Analog Stochasticity (`step_analog`)

### How it Works
In a physical analog Spintronic chip or MTJ array, signals do not cleanly resolve to boolean `True` or `False`. They resolve probabilistically (e.g., this spike fired with 65% certainty). Passing rigid booleans into STDP arrays degrades true physical realization.
The Rust engine exposes:
```rust
pub unsafe extern "C" fn step_rule_layer_analog(...)
```
Instead of converting float probabilities into Boolean threshold spikes inside Python (which forces a heavy numpy casting bottleneck), the Python layer natively maps contiguous `f32` arrays into the Rust FFI.

Inside Rust, `rayon` dispatches these probabilities natively over thread-local pseudo-random number generators (`rand::thread_rng()`). The PRNG rolls physical probabilities on isolated CPU threads for millions of nodes simultaneously, converting floating uncertainty into exact discrete biological realities localized to individual synaptic weights.

### Formal Verification & Deterministic Mode
While probabilistic sampling is highly accurate to biological stochastic resonance, hardware deployment pipelines (like generating Verilog out of equation logic using `SymbiYosys` formal proofs) require strictly comparable exact results without RNG divergence.

By engaging the `set_deterministic_mode(seed: int)` wrapper from `sc_neurocore._native.learning_bridge`, the native Rayon `rand::thread_rng` logic is explicitly bypassed in favor of a strictly reproducible pseudo-random state identical across bit-true tests.

### Performance & Benchmarks
> [!NOTE]
> For exact hardware bounds and verifiable execution data on reference hardware (31 GB / 12-core CPU) reproducing biological convergence, see the full suite at **[Project Zenith: End-to-End Benchmark Report](../benchmarks/ZENITH_BENCHMARKS.md)** (`benchmarks/bench_zenith_e2e.py`).

### Real Time Applications
*   **Emulating Hardware Drift:** Simulating actual chip unreliability and thermal drift before taping out silicon designs.
*   **Stochastic Resonance Models:** Adding calculated noise directly into the brain's internal learning mechanics to overcome deterministic local minimums.

---

## 3. The Supervised-Biological Bridge (`AutogradSTDPLayer`)

### How it Works
STDP (Spike-timing-dependent plasticity) is naturally an unsupervised algorithm. It changes weights locally based strictly on pre/post occurrence. To train these biological systems using Modern Deep Learning, we integrated it seamlessly into the `torch.autograd.Function` pipeline via Surrogate Gradients.

We utilized the **Straight-Through Estimator (STE)** paradigm inside the backward pass. The Unsupervised biological FFI runs purely out-of-place during the `forward()` operation, caching states (`ctx.save_for_backward`). When you fire `loss.backward()` anywhere in the ML chain, the Python Autograd intercepts the gradients and maps them directly through the biological eligibility trace (`grad_weights * a_plus * pre_trace`) backward up to preceding dense / CNN layers.

### Performance & Benchmarks
> [!NOTE]
> For exact throughput measurements of surrogate memory graphs at Exascale boundaries, see the physical trace tests at **[Project Zenith: End-to-End Benchmark Report](../benchmarks/ZENITH_BENCHMARKS.md)** (`benchmarks/bench_zenith_e2e.py`).

### Real Time Applications
*   **Hybrid Neuromorphic ResNets:** Plugging a biological STDP module into the center of a traditional GPU PyTorch model and maintaining end-to-end mathematical optimization.
*   **Supervised BCI Implants:** Retraining brain-computer STDP logic parameters using backpropagation optimization based on human-in-the-loop reward metrics.

---

## 4. Exascale Persistence (`save` / `load`)

### How it Works
Standard Python architectures (like `pickle` or `torch.save()`) must map Python objects into bytes. Doing this across 10 Million specific node combinations (Weight, Threshold, Trace histories) crashes RAM buffers and stalls operations.

We circumvent Python entirely by executing a contiguous C-pointer cast natively bounded in `lib.rs`:
```rust
let mut state_buffer: Vec<f32>; // Pre-allocated scaling
// ... dumps arrays strictly to generic u8 bytes
file.write_all(byte_slice)
```
Python simply calls `layer.save("/path.bin")`, and the file IO writes instantaneously directly out of RAM limits over the OS storage pipeline.

### Performance & Benchmarks
> [!NOTE]
> To review the actual disk-mapping NVMe latency arrays comparing Python Pickle formats explicitly to Exascale boundaries, see **[Project Zenith: End-to-End Benchmark Report](../benchmarks/ZENITH_BENCHMARKS.md)** (`benchmarks/bench_zenith_e2e.py`).

### Real Time Applications
*   **mmap2 Exascale Ready:** A single motherboard cannot hold 1-Trillion biological parameters (Approx 64 Terabytes). By having perfectly constrained binary dumps, the engine establishes the roadmap for reading arrays natively using `memmap`/`mmap2`, allowing future systems to load parameters seamlessly over clusters of multi-petabyte NVMe SSD arrays.
*   **Robotic Reboot Insurance:** If a Boston Dynamics robot loses power mid-mission, Python script startup drops. The native Rust daemon flushes biological STDP memories to `.bin` in milliseconds avoiding catastrophic agent memory loss.
