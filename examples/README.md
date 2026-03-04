# Examples

Runnable demos showing core SC-NeuroCore capabilities.

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | `01_basic_sc_encoding.py` | Bernoulli & Sobol bitstream encoding/decoding |
| 02 | `02_sc_neuron_layer.py` | SCDenseLayer construction and forward pass |
| 03 | `03_ir_compile_demo.py` | IR graph building, verification, SystemVerilog emission (v3 Rust engine) |
| 04 | `04_vectorized_layer.py` | VectorizedSCLayer throughput benchmarking |
| 05 | `05_scpn_stack.py` | Full 7-layer SCPN consciousness stack with inter-layer coupling |
| 06 | `06_hdl_generation.py` | Verilog top-level generation from a network description |
| 07 | `07_ensemble_consensus.py` | Multi-agent ensemble orchestration and voting |
| 08 | `08_hdc_symbolic_query.py` | Hyper-Dimensional Computing symbolic memory (v3 Rust engine) |
| 09 | `09_safety_critical_logic.py` | Fault-tolerant Boolean logic with stochastic redundancy (v3 Rust engine) |
| 10 | `10_benchmark_report.py` | Head-to-head v2/v3 benchmark suite (v3 Rust engine) |
| 11 | `11_sc_training_demo.py` | Surrogate-gradient training of an SC dense layer (v3 Rust engine) |

## Running

```bash
cd 03_CODE/sc-neurocore
PYTHONPATH=src:bridge python examples/01_basic_sc_encoding.py
```

Examples marked **(v3 Rust engine)** require the compiled `sc_neurocore_engine` wheel.
All other examples run with the pure-Python `sc_neurocore` package.
