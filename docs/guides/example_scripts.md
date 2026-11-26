<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Example scripts
-->

# Example Scripts

This text was part of the repository README until the README became a short landing page.

Runnable scripts in `examples/`:

| Script | Description |
|--------|-------------|
| `01_basic_sc_encoding.py` | Bernoulli & Sobol bitstream encoding/decoding |
| `02_sc_neuron_layer.py` | SCDenseLayer construction, spike trains, and firing-rate summary |
| `03_ir_compile_demo.py` | IR graph building, verification, SystemVerilog emission (v3 Rust engine) |
| `04_vectorized_layer.py` | VectorizedSCLayer throughput benchmarking |
| `05_scpn_stack.py` | 16-layer SCPN (Stochastic Coupled Phase-oscillator Network) stack with inter-layer coupling and global coherence metrics |
| `06_hdl_generation.py` | Verilog top-level generation from a network description |
| `07_ensemble_consensus.py` | Multi-agent ensemble orchestration and voting |
| `08_hdc_symbolic_query.py` | Hyper-Dimensional Computing symbolic memory (v3 Rust engine) |
| `09_safety_critical_logic.py` | Fault-tolerant Boolean logic with stochastic redundancy (v3 Rust engine) |
| `10_benchmark_report.py` | Head-to-head v2/v3 benchmark suite (v3 Rust engine) |
| `11_sc_training_demo.py` | Surrogate-gradient training of an SC dense layer (v3 Rust engine) |
| `12_load_pretrained_model.py` | Load pretrained ConvSpikingNet and classify MNIST digits |
| `zenith_hybrid_resnet.py` | Train a hybrid network with a Torch BCM plasticity layer through autograd, then save and reload a Rust-backend layer state file |
| `jax_training_demo.py` | JAX JIT surrogate-gradient SNN training on synthetic data |
| `mnist_fpga/demo.py` | MNIST classifier: train → quantise Q8.8 → SC simulate → Verilog export |
| `mnist_conv_train.py` | **ConvSpikingNet: 99.49% MNIST** in the committed 2026-03-26 run (learnable beta/threshold, cosine LR; evidence in `benchmarks/results/mnist_conv_accuracy_reproducibility.json`) |
| `mnist_surrogate/train.py` | Surrogate gradient SNN training with the default ATan surrogate (95.9% MNIST for the same network in the committed 10-epoch benchmark run, `benchmarks/results/upcloud_atan_20260310/`) |
| `nir_roundtrip_demo.py` | NIR roundtrip: CubaLIF + recurrent connections, build → import → run → export |
| `norse_nir_roundtrip.py` | Norse → NIR → SC-NeuroCore roundtrip with real Norse weights |
| `snntorch_nir_roundtrip.py` | snnTorch RSynaptic → NIR → SC-NeuroCore roundtrip (CubaLIF + recurrent) |
| `spikingjelly_nir_roundtrip.py` | SpikingJelly → NIR → SC-NeuroCore roundtrip |
| `ann_to_snn_demo.py` | Convert trained PyTorch ANN to rate-coded SNN |
| `delay_training_demo.py` | Train spiking network with learnable per-synapse delays |

```bash
PYTHONPATH=src:bridge python examples/01_basic_sc_encoding.py
```

Examples marked **(v3 Rust engine)** require an available `sc_neurocore_engine`
bridge install. For source-tree runs against local bridge code, use
`PYTHONPATH=src:bridge` or install `bridge/` in the same environment.
