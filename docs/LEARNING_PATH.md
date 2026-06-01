# SC-NeuroCore Learning Path

A structured progression from first contact to FPGA deployment. Use this page as the main onboarding route: it explains what to read, what to run, and what evidence each stage should produce. Each level builds on the previous. Estimated times assume familiarity with Python and basic linear algebra.

Before starting, read [Product Overview](product_overview.md) for the project scope and [Applications and Market](applications_and_market.md) for practical use cases and commercial boundaries.

---

## Level 0 — What Is Stochastic Computing? (30 min)

**Goal**: Understand why random bitstreams can replace arithmetic.

| Resource | What you learn |
|----------|---------------|
| [SC Fundamentals tutorial](tutorials/01_stochastic_computing_fundamentals.md) | Unipolar encoding, AND-gate multiplication, majority-gate addition |
| [Neuromorphic Primer](research/NEUROMORPHIC_COMPUTING_PRIMER.md) | Where SC fits in the neuromorphic landscape |

**Checkpoint**: You can explain why `P(A AND B) = P(A) * P(B)` for
independent Bernoulli streams.

---

## Level 1 — Core Primitives (1 h)

**Goal**: Encode values, build neurons, connect synapses.

```python
from sc_neurocore import (
    BitstreamEncoder, generate_bernoulli_bitstream, bitstream_to_probability,
    StochasticLIFNeuron, BitstreamSynapse,
)

# Encode a value as a bitstream
enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=1024)
bits = enc.encode(0.7)            # ~70% ones
p_hat = bitstream_to_probability(bits)  # ≈ 0.7

# Spike a LIF neuron
neuron = StochasticLIFNeuron(v_threshold=1.0, noise_std=0.0)
spikes = [neuron.step(1.5) for _ in range(100)]

# Weight a synapse
syn = BitstreamSynapse(w_min=0.0, w_max=1.0, w=0.5, length=1024)
post = syn.apply(bits)  # P(post) ≈ 0.7 * 0.5
```

| Resource | What you learn |
|----------|---------------|
| [Getting Started](guides/getting-started.md) | Install, first neuron, first layer |
| API: [Neurons](api/neurons.md), [Synapses](api/synapses.md), [Utilities](api/utils.md) | Class signatures and parameters |

**Checkpoint**: You can encode a value, feed it through a synapse into
a neuron, and read back the firing rate.

---

## Level 2 — Networks & Layers (2 h)

**Goal**: Compose neurons into layers, run multi-step simulations.

```python
from sc_neurocore import SCDenseLayer, VectorizedSCLayer
import numpy as np

# Dense layer with SC input pipeline
layer = SCDenseLayer(
    n_neurons=8, x_inputs=[0.5, 0.3], weight_values=[0.8, 0.6],
    x_min=0.0, x_max=1.0, w_min=0.0, w_max=1.0,
)
layer.run(T=200)
trains = layer.get_spike_trains()  # (8, 200)

# High-performance vectorized layer (packed bitwise ops)
fast = VectorizedSCLayer(n_inputs=16, n_neurons=32, length=512)
rates = fast.forward(np.random.rand(16))  # (32,) firing rates
```

| Resource | What you learn |
|----------|---------------|
| [Building Your First SNN](tutorials/02_building_your_first_snn.md) | Multi-layer network, spike raster plots |
| [Layer-by-Layer Guide](guides/LAYER_BY_LAYER_GUIDE.md) | Dense, Conv2D, Recurrent, Fusion, Attention |
| API: [Layers](api/layers.md) | All layer classes |

**Checkpoint**: You can build a 2-layer SNN, run it for 500 steps,
and plot the spike raster.

---

## Level 3 — Learning & Plasticity (2 h)

**Goal**: Train networks using STDP, R-STDP, and surrogate gradients.

```python
from sc_neurocore import StochasticSTDPSynapse, RewardModulatedSTDPSynapse

# Hebbian STDP
syn = StochasticSTDPSynapse(w_min=0.0, w_max=1.0, w=0.5, length=64)
for _ in range(200):
    syn.process_step(pre_bit=1, post_bit=1)  # correlated → LTP
print(syn.w)  # > 0.5

# Reward-modulated (three-factor) learning
rsyn = RewardModulatedSTDPSynapse(w_min=0.0, w_max=1.0, w=0.5, length=64)
for _ in range(50):
    rsyn.process_step(pre_bit=1, post_bit=1)
rsyn.apply_reward(1.0)  # delayed reward signal
```

| Resource | What you learn |
|----------|---------------|
| [Surrogate Gradient Training](tutorials/03_surrogate_gradient_training.md) | Gradient-based SNN training with PyTorch |
| [Online Learning with STDP](tutorials/08_online_learning_stdp.md) | STDP, R-STDP, eligibility traces |
| [User Manual](guides/USER_MANUAL.md) | Learning rules, training loops |

**Checkpoint**: You can train a network to classify a simple pattern
using either STDP or surrogate gradients.

---

## Level 4 — Advanced Architectures (2 h)

**Goal**: Use convolutional, recurrent, hyperdimensional, and
attention layers.

| Resource | What you learn |
|----------|---------------|
| [MNIST SC Classification](tutorials/07_mnist_sc_classification.md) | End-to-end SC image classification |
| [Reservoir Computing](tutorials/10_reservoir_computing.md) | Echo state networks with SC recurrent layers |
| [Multi-Scale Networks](tutorials/11_multi_scale_networks.md) | Hierarchical cortical column models |
| [Hyper-Dimensional Computing](tutorials/04_hyperdimensional_computing.md) | HDC encoding, bundling, binding |
| [Advanced Usage Patterns](guides/ADVANCED_USAGE_PATTERNS.md) | Custom neuron models, quantum entropy, analysis tools |
| [Brunel Network Translation](tutorials/06_brunel_network_translation.md) | Translate a classic balanced network to SC |
| API: [HDC](api/hdc.md), [Ensembles](api/ensembles.md) | Hyperdimensional and ensemble modules |

**Checkpoint**: You can classify MNIST with an SC network or
implement a reservoir computing benchmark.

---

## Level 5 — Performance & Acceleration (1 h)

**Goal**: Use the Rust engine, GPU backend, and packed-bitstream
vectorization for production-grade throughput.

| Resource | What you learn |
|----------|---------------|
| [Rust Engine & Performance](tutorials/05_rust_engine_performance.md) | SIMD-accelerated Rust engine, Python↔Rust bridge |
| [Performance Tuning](guides/PERFORMANCE_TUNING.md) | Packed uint64 ops, CuPy GPU path, sparse layers |
| [Rust Engine API](api/rust-engine.md) | Rust crate API reference |
| [Benchmark Report](benchmarks/BENCHMARK_REPORT.md) | Throughput numbers vs Brian2, Norse, snnTorch |

**Checkpoint**: You can run the Rust engine from Python and see
>10x speedup over pure NumPy for a Brunel network.

---

## Level 6 — Hardware Deployment (2 h)

**Goal**: Synthesize SC neuron designs onto an FPGA.

| Resource | What you learn |
|----------|---------------|
| [FPGA in 20 Minutes](tutorials/fpga_in_20_minutes.md) | Yosys synthesis, Verilator simulation, resource utilization |
| [Hardware Co-simulation](tutorials/09_hardware_cosimulation.md) | Python ↔ Verilog cycle-exact verification |
| [Fixed-Point Arithmetic](tutorials/13_fixed_point_design.md) | Q8.8 format, overflow, weight export |
| [Network Export & Deployment](tutorials/14_network_export_deployment.md) | Rust, FPGA, and checkpoint export |
| [Hardware Guide](hardware/HARDWARE_GUIDE.md) | HDL module inventory, timing constraints |
| [FPGA Toolchain Guide](hardware/FPGA_TOOLCHAIN_GUIDE.md) | Yosys + sv2v + nextpnr setup |
| [Hardware Manual](hardware/SC_NEUROCORE_HARDWARE_MANUAL.md) | Pin-level specification for each Verilog module |

**Checkpoint**: You can synthesize `sc_lif_neuron.v` with Yosys,
verify it with Verilator, and read the resource utilization report.

---

## Level 7 — Research & Extension (open-ended)

**Goal**: Extend SC-NeuroCore for your own research.

| Resource | What you learn |
|----------|---------------|
| [Architecture Overview](architecture/architecture.md) | Full system architecture, module dependency graph |
| [Component Inventory](architecture/COMPONENT_INVENTORY.md) | Every module, its role, its test coverage |
| [Technical Manual](guides/TECHNICAL_MANUAL.md) | Internal design decisions, numerical choices |
| [Foundational Whitepaper](research/SC_NEUROCORE_FOUNDATIONAL_WHITEPAPER.md) | Theoretical basis for the SC-SNN approach |
| [Quantum-SC Hybrid Networks](tutorials/15_quantum_sc_hybrid.md) | Quantum circuits as SC front-ends |
| [Neuromorphic Signal Processing](tutorials/12_neuromorphic_signal_processing.md) | SC filterbanks, edge detection |
| [Integration Guide](guides/INTEGRATION_GUIDE.md) | Embedding SC-NeuroCore in larger systems |

**Checkpoint**: You can add a custom neuron model, wire it into the
compiler, and generate HDL for it.

---

## Quick Reference: Which Doc for Which Question?

| Question | Go to |
|----------|-------|
| "How do I install it?" | [Getting Started](guides/getting-started.md) |
| "What does class X do?" | [API Reference](api/API_REFERENCE.md) |
| "How fast is it?" | [Benchmarks](benchmarks/BENCHMARKS.md) |
| "Can it run on an FPGA?" | [FPGA in 20 Minutes](tutorials/fpga_in_20_minutes.md) |
| "How does SC multiplication work?" | [SC Fundamentals](tutorials/01_stochastic_computing_fundamentals.md) |
| "I come from Brian2/NEST" | [SC for Neuroscientists](guides/SC_FOR_NEUROSCIENTISTS.md) |
| "I come from PyTorch/JAX" | [SC for ML Engineers](guides/SC_FOR_ML_ENGINEERS.md) |
| "I design FPGAs/ASICs" | [SC for Hardware Engineers](guides/SC_FOR_HARDWARE_ENGINEERS.md) |
| "What changed in v3?" | [V3 Migration](development/v3_migration.md) |
