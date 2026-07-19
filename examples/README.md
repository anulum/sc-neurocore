# Examples

Runnable demos showing core SC-NeuroCore capabilities.

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | `01_basic_sc_encoding.py` | Bernoulli & Sobol bitstream encoding/decoding |
| 02 | `02_sc_neuron_layer.py` | SCDenseLayer construction, spike trains, and firing-rate summary |
| 03 | `03_ir_compile_demo.py` | IR graph building, verification, SystemVerilog emission (v3 Rust engine) |
| 04 | `04_vectorized_layer.py` | VectorizedSCLayer throughput benchmarking |
| 05 | `05_scpn_stack.py` | Full 16-layer SCPN stack with inter-layer coupling |
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

| 12 | `mnist_fpga/demo.py` | MNIST-on-FPGA: train → Q8.8 → SC simulate → Verilog export |
| DM-01 | `dm01_spike_raster_gif.py` | Hodgkin–Huxley spike-raster GIF/PNG (local demo, not a claim) |
| DM-02 | `dm02_sc_error_sweep.py` | SC unipolar reconstruction error vs bitstream length (HH proxy) |
| DM-03 | `dm03_mnist_verilog_path.md` | Pointer to `mnist_fpga/demo.py` one-command path |
| DM-04 | `dm04_synthesis_report_reader.py` | Summarise committed Vivado/Yosys reports under `hdl/reports/` |

High-fidelity demo programme (notebooks 41–48 + these scripts): monorepo plan
`PLAN_2026-07-19T2301_notebook_demo_programme_high_fidelity_neurons.md`.

```bash
PYTHONPATH=src python examples/dm01_spike_raster_gif.py
PYTHONPATH=src python examples/dm02_sc_error_sweep.py
PYTHONPATH=src python examples/dm04_synthesis_report_reader.py
```

Generated files under `examples/output/` are local only — do not promote as package evidence.

Examples marked **(v3 Rust engine)** require an available `sc_neurocore_engine`
bridge install. For source-tree runs against local bridge code, use
`PYTHONPATH=src:bridge` or install `bridge/` in the same environment. The
MNIST demo requires `scikit-learn`.
