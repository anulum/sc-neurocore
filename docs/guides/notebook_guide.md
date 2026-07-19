# Notebook Guide

Complete index of the public Jupyter notebooks in `notebooks/`. The notebooks are organised as onboarding, feature demonstrations, and evidence notebooks. Treat evidence notebooks as reproducible documentation: each one states what it proves and what it does not prove.

## How to use the notebooks

| Reader | Recommended path | Output to keep |
| --- | --- | --- |
| New user | `quickstart_colab` -> `03_end_to_end_pipeline` -> `04_neuron_explorer` | Screenshots or local notes only; do not promote as benchmark evidence. |
| Hardware evaluator | `08_equation_to_verilog` -> `27_python_to_proven_silicon` -> `29_golden_path_evidence` | Generated manifests, RTL, and named report artefacts. |
| Interop reviewer | `05_nir_bridge` -> cross-framework benchmark docs | NIR graph artefacts and raw benchmark JSON. |
| Industrial reviewer | Evidence notebooks `34` through `39` | Evidence-gap tables and readiness manifests. |

Notebook output becomes a public claim only when the raw artefact is committed
and the docs name it. Otherwise it remains explanatory or local exploratory
evidence.

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
| 10 | `10_spike_train_analysis` | ISI, CV, Fano, cross-correlation, van Rossum, PCA (127 functions) | [Tutorial 23](../tutorials/23_spike_train_analysis.md) |
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

## Evidence Notebooks (29–39)

| # | Notebook | What you learn | Evidence boundary |
|---|----------|---------------|-------------------|
| 29 | `29_golden_path_evidence` | Deterministic training -> Q8.8 -> SC simulation -> typed IR -> generated Verilog -> manifest | Local generated evidence only; no physical FPGA, timing, power, QPU, or external-data claim |
| 30 | `30_shd_vertex_deployable_evidence` | Summarise downloaded SHD Vertex deployable-selector artifacts by seed | Available local artifacts only; no claim that all intended seeds are downloaded or externally accepted |
| 31 | `31_balanced_resonate_and_fire_evidence` | Verify BRF Algorithm 1 update, deterministic traces, guardrails, and committed benchmark artifact | Scalar model and benchmark evidence only; no full BRF-RSNN training reproduction or hardware timing claim |
| 32 | `32_posner_ibm_readiness_evidence` | Verify ORCA/IBM readiness gates, runtime JSON refusal paths, and minimum QPU shot budget | Readiness gates only; no ORCA-derived runtime parameter, IBM calibration, or QPU-result claim |
| 33 | `33_hil_digital_twin_evidence` | Generate HIL protocol, drift controller, digital twin, SEU schedule, and telemetry drift manifest | Local generated protocols and synthetic telemetry only; no physical hardware calibration or certification claim |
| 34 | `34_industrial_readiness_evidence` | Evaluate industrial application profiles, mandatory evidence categories, and fail-closed readiness arithmetic | Local readiness profile arithmetic only; no certification, target-hardware, or authority-accepted safety-case claim |
| 35 | `35_bci_closed_loop_evidence` | Process synthetic waveform windows through BCI compression, AER, rate decoding, emulator feedback, telemetry, and HIL manifests | Synthetic waveform and implant-emulator evidence only; no clinical, stimulation, physical implant, or HIL timing claim |
| 36 | `36_fault_resilience_evidence` | Run seeded fault-injection resilience mode, deterministic replay policy, radiation stress presets, and input guardrails | Local seeded fault-injection evidence only; no radiation qualification, mission acceptance, hardware SEU, or certified fault-tolerance claim |
| 37 | `37_neuro_symbolic_self_verification_evidence` | Verify neuro-symbolic inference traces, stable digests, tamper detection, symbol-score ordering, trace-only evidence, and vector guardrails | Internal consistency evidence only; no external semantic-truth, clinical, physical, or deployment-validity claim |
| 38 | `38_formal_snn_verification_standard_evidence` | Exercise the formal SNN verification profile, pass/fail/missing evidence accounting, wrong-kind rejection, optional safety-case evidence, and evidence guardrails | Local evidence-accounting behavior only; no external prover, model-checker, HDL proof, safety certification, or unbounded semantic-correctness claim |
| 39 | `39_self_hosted_hub_evidence` | Generate and validate offline-first hub manifests, model-zoo index, Compose hardening, opt-in benchmark plan, bundle files, and config guardrails | Local bundle generation only; no Docker start, image build, production exposure, network-isolation proof, or operational-security certification claim |

## High-fidelity demo notebooks (41–43)

Programme plan: monorepo
`.coordination/planning/SC-NEUROCORE/PLAN_2026-07-19T2301_notebook_demo_programme_high_fidelity_neurons.md`.

Neurons are taken only from the polyglot-complete set in
[`model_fidelity_status.md`](../api/model_fidelity_status.md).

| # | Notebook | What you learn | Evidence boundary |
|---|----------|---------------|-------------------|
| 41 | `41_one_model_five_views` | Hodgkin–Huxley, Morris–Lecar, AdEx: dynamics, f–I, SC rate pedagogy, didactic Q8.8 voltage quantisation, hardware-path pointers | Local figures only; not polyglot parity, FPGA power/timing, or full RTL co-sim for all three |
| 42 | `42_fault_tolerance_theatre` | Perfect Integrator baseline plus SC unipolar product vs float under controlled BER | Local BER trends only; not radiation qualification or FIT rates |
| 43 | `43_studio_evidence_cart_lab` | AdEx run plus pedagogical SHA-256 ledger aligned with `studio.evidence-cart.v1` field names | Python in-process digests only; not Studio browser cart or server evidence-bundle product export |

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
| 01–16, 20, 25, 41–43 | `matplotlib` (visualisation) |
| 17 | `matplotlib`, `numpy.linalg` (SVD) |
| 21–22 | None beyond sc-neurocore |
| 23 | `matplotlib` |
| 24 | `tempfile` (checkpoint I/O) |
| 26 | `time` (benchmarking) |
| 27 | None beyond sc-neurocore |
| 28 | `matplotlib` |
| 29 | None beyond sc-neurocore |
| 30 | None beyond sc-neurocore |
| 31 | None beyond sc-neurocore |
| 32 | None beyond sc-neurocore |
| 33 | None beyond sc-neurocore |
| 34 | None beyond sc-neurocore |
| 35 | None beyond sc-neurocore |
| 36 | None beyond sc-neurocore |
| 37 | None beyond sc-neurocore |
| 38 | None beyond sc-neurocore |
| 39 | `pyyaml` |

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
| Golden path evidence (29) | `test_notebooks/test_golden_path_notebook.py` | 2 |
| SHD Vertex deployable evidence (30) | `test_notebooks/test_shd_vertex_notebook.py` | 2 |
| BRF evidence (31) | `test_notebooks/test_brf_notebook.py` | 2 |
| Posner/IBM readiness evidence (32) | `test_notebooks/test_posner_ibm_readiness_notebook.py` | 2 |
| HIL/digital twin evidence (33) | `test_notebooks/test_hil_digital_twin_notebook.py` | 2 |
| Industrial readiness evidence (34) | `test_notebooks/test_industrial_readiness_notebook.py` | 2 |
| BCI closed-loop evidence (35) | `test_notebooks/test_bci_closed_loop_notebook.py` | 2 |
| Fault resilience evidence (36) | `test_notebooks/test_fault_resilience_notebook.py` | 2 |
| Neuro-symbolic self-verification evidence (37) | `test_notebooks/test_neuro_symbolic_self_verification_notebook.py` | 2 |
| Formal SNN verification standard evidence (38) | `test_notebooks/test_formal_snn_verification_standard_notebook.py` | 2 |
| Self-hosted hub evidence (39) | `test_notebooks/test_self_hosted_hub_notebook.py` | 2 |
| SCPN (—) | `test_scpn_integrated.py` | 17 |
| CORDIV (14) | `test_cordiv_division.py` | 10 |
| Monitors (—) | `test_network_monitors_stimulus.py` | 19 |

**Total new tests: 340 passing on Python 3.12.**
