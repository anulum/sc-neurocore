# Notebook Guide

Complete index of all 29 Jupyter notebooks in `notebooks/`.

## Quickstart

| Notebook | Topic | Prerequisites |
|----------|-------|---------------|
| [quickstart_colab](https://github.com/anulum/sc-neurocore/blob/main/notebooks/quickstart_colab.ipynb) | Single neuron, dense layer, spike raster | `pip install sc-neurocore` |

## Core SC Pipeline (01–07)

| # | Notebook | What you learn | Tutorial cross-ref |
|---|----------|---------------|-------------------|
| 01 | `01_hdc_symbolic_query` | HDC/VSA symbolic memory with 10,000-bit vectors | [Tutorial 04](../tutorials/04_hyperdimensional_computing.md) |
| 02 | `02_fault_tolerant_logic` | Boolean logic with stochastic redundancy (survives 40% error) | [Tutorial 27](../tutorials/27_fault_tolerance.md) |
| 03 | `03_end_to_end_pipeline` | Encode → synapse → popcount → LIF → VectorizedSCLayer | [Tutorial 01](../tutorials/01_stochastic_computing_fundamentals.md) |
| 04 | `04_neuron_explorer` | Interactive exploration of 117+ neuron models | [Tutorial 22](../tutorials/22_neuron_model_selection.md) |
| 05 | `05_nir_bridge` | NIR graph import/export, cross-framework interop | [Tutorial 30](../tutorials/30_nir_bridge.md) |
| 06 | `06_network_engine` | Population-Projection-Network, E/I balance, STDP | [Tutorial 31](../tutorials/31_network_simulation_engine.md) |
| 07 | `07_identity_substrate` | Persistent spiking network, STDP consolidation | [Tutorial 32](../tutorials/32_identity_substrate.md) |

## Feature Demos (08–13)

| # | Notebook | What you learn | Tutorial cross-ref |
|---|----------|---------------|-------------------|
| 08 | `08_equation_to_verilog` | ODE string → Python sim → Q8.8 Verilog RTL (LIF, FHN, Izhikevich) | [Tutorial 33](../tutorials/33_equation_to_verilog.md) |
| 09 | `09_topology_and_dynamics` | 6 connectivity generators, degree distributions, dynamics comparison | [Tutorial 83](../tutorials/83_topology.md) |
| 10 | `10_spike_train_analysis` | ISI, CV, Fano, cross-correlation, van Rossum, PCA (124 functions) | [Tutorial 23](../tutorials/23_spike_train_analysis.md) |
| 11 | `11_biological_circuits` | Tripartite synapse (Ca²⁺), Rall dendrite (3/2 power law) | [Tutorial 24](../tutorials/24_biological_circuits.md) |
| 12 | `12_learning_rules` | STDP, e-prop eligibility, R-STDP, STP facilitation/depression | [Tutorial 28](../tutorials/28_learning_rules.md) |
| 13 | `13_quantisation_pipeline` | Float → Q8.8 → SC probabilities → Verilog export, error budget | [Tutorial 77](../tutorials/77_qat.md) |

## Advanced Topics (14–20)

| # | Notebook | What you learn | Tutorial cross-ref |
|---|----------|---------------|-------------------|
| 14 | `14_sc_arithmetic_theory` | AND=multiply, XNOR=bipolar, MUX=add, CORDIV=divide, Hoeffding bounds | [Tutorial 19](../tutorials/19_sc_theory_deep_dive.md) |
| 15 | `15_fault_tolerance` | SC vs fixed-point under bit-flips/stuck-at, TMR majority vote | [Tutorial 63](../tutorials/63_fault_resilience.md) |
| 16 | `16_neuron_atlas` | 12 models from 8 families (1907–2026), voltage traces | [Tutorial 17](../tutorials/17_custom_neuron_models.md) |
| 17 | `17_reservoir_computing` | Liquid state machine, temporal XOR, ridge readout | [Tutorial 74](../tutorials/74_reservoir.md) |
| 18 | `18_mixed_precision_sc` | Per-layer adaptive bitstream length, Hoeffding vs sensitivity | [Tutorial 44](../tutorials/44_model_compression.md) |
| 19 | `19_compression_and_pruning` | Magnitude/SC-aware pruning, quantisation sweep, Pareto | [Tutorial 44](../tutorials/44_model_compression.md) |
| 20 | `20_power_analysis` | Event-driven vs clock-driven toggle count, scaling with network size | [Tutorial 41](../tutorials/41_energy_estimation.md) |

## Frontier Research (21–28)

These notebooks demonstrate capabilities unique to SC-NeuroCore — features not available in any other SNN framework.

| # | Notebook | What you learn | Uniqueness |
|---|----------|---------------|------------|
| 21 | `21_spike_alu` | Turing-complete spike-based ALU: logic gates, register, adder, sort | First spike ALU in an SNN framework |
| 22 | `22_ir_type_safety` | IR signal type checker: Bitstream/Rate/Spike/Fixed compatibility | First SNN→FPGA type system |
| 23 | `23_topological_observables` | Winding number, Ricci curvature, sheaf defect on coupling graphs | First topological observables on SNN |
| 24 | `24_identity_lazarus` | Lazarus checkpoint save/load/merge, TraceEncoder, DirectorController | First persistent AI identity substrate |
| 25 | `25_cortical_column_dynamics` | 5-population canonical microcircuit, feedforward latency | Biophysically grounded column model |
| 26 | `26_spike_codec_benchmark` | 5 codecs (ISI/AER/predictive/delta/streaming), density curves | Comprehensive codec comparison |
| 27 | `27_python_to_proven_silicon` | Complete ODE→sim→type check→Verilog→testbench→formal→resource | End-to-end verified hardware pipeline |
| 28 | `28_domain_bridge` | TensorStream prob↔bitstream↔quantum, Born rule, cos²(θ/2) | Cross-domain probability bridge |

## Running Notebooks

### Local

```bash
pip install sc-neurocore[dev]
pip install matplotlib jupyter
jupyter notebook notebooks/
```

### Google Colab

Use `quickstart_colab.ipynb` — installs sc-neurocore automatically.

### Prerequisites by Notebook

| Notebooks | Extra dependencies |
|-----------|-------------------|
| 01–16, 20, 25 | `matplotlib` (visualisation) |
| 17 | `matplotlib`, `numpy.linalg` (SVD) |
| 21–22 | None beyond sc-neurocore |
| 23 | `matplotlib` |
| 24 | `tempfile` (checkpoint I/O) |
| 26 | `time` (benchmarking) |
| 27 | None beyond sc-neurocore |
| 28 | `matplotlib` |

## Test Coverage

Every notebook topic has a corresponding test suite in `tests/`:

| Notebook topic | Test file | Tests |
|---------------|-----------|------:|
| Topology (09) | `test_topology_generators.py` | 26 |
| SC arithmetic (14) | `test_sc_convergence.py` | 13 |
| Fault tolerance (15) | `test_fault_injection.py` | 14 |
| Neuron atlas (16) | `test_neuron_families.py` | 57 |
| Learning rules (12) | `test_learning_advanced.py` | 11 |
| Quantisation (13) | `test_quantisation_pipeline.py` | 12 |
| Spike ALU (21) | `test_spike_alu.py` | 27 |
| IR types (22) | `test_ir_type_checker.py` | existing |
| Topology obs (23) | `test_topological_observables.py` | 17 |
| Identity (24) | `test_identity_lazarus.py` | 20 |
| Cortical col (25) | `test_cortical_column_dynamics.py` | 13 |
| Domain bridge (28) | `test_tensor_stream.py`, `test_quantum_hybrid.py` | 24 |
| SCPN (—) | `test_scpn_integrated.py` | 17 |
| CORDIV (14) | `test_cordiv_division.py` | 10 |
| Monitors (—) | `test_network_monitors_stimulus.py` | 19 |

**Total new tests: 340 passing on Python 3.12.**
