# SC-NeuroCore Component Inventory

> **Superseded by [`SYSTEM_MAP.md`](SYSTEM_MAP.md) (2026-06-23).** This inventory
> is stale and inflated: sections 4–5 list speculative/aspirational modules, and
> several classes named here (e.g. `DigitalSoul`, `MetaCognitionLoop`,
> `PhiEvaluator`, `QualiaTuringTest`, `KardashevEstimator`) are **not present in
> `src/`**. Use `SYSTEM_MAP.md` for the verified component set.

**Version:** 3.14.0
**Date:** 2026-04-13

This document lists all components in the `sc-neurocore` framework, organized
by domain. Sections 1--3 and 6--7 are implemented and tested. Sections 4--5
contain modules from the broader Anulum Research / SCPN theoretical framework
that are planned but not yet implemented in this repository.

## 1. Core Neuromorphic
*   **`neurons/`**
    *   `StochasticLIFNeuron`: Leaky Integrate-and-Fire with noise and refractory period.
    *   `FixedPointLIFNeuron`: Bit-true integer model for RTL verification.
    *   `HomeostaticLIFNeuron`: Self-regulating firing threshold.
    *   `StochasticDendriticNeuron`: Multi-compartment neuron for non-linear logic (XOR).
*   **`synapses/`**
    *   `BitstreamSynapse`: Basic stochastic weight multiplication.
    *   `StochasticSTDPSynapse`: Spike-Timing-Dependent Plasticity.
    *   `RewardModulatedSTDPSynapse`: R-STDP for reinforcement learning.
*   **`layers/`**
    *   `SCLearningLayer`: Dense layer with local plasticity.
    *   `SCConv2DLayer`: Stochastic 2D Convolution.
    *   `SCRecurrentLayer`: Reservoir Computing / RNN.
    *   `VectorizedSCLayer`: High-performance 64-bit packed simulation.

## 2. Advanced Computing & AGI
*   **`transformers/`**
    *   `StochasticTransformerBlock`: "S-Former" with spike-based attention.
*   **`world_model/`**
    *   `PredictiveWorldModel`: Stochastic state transition forecasting.
    *   `SCPlanner`: Action selection via predictive modeling.
*   **`spatial/`**
    *   `VoxelGrid`, `PointCloud`: 3D spatial representations.
    *   `SpatialTransformer3D`: Attention-based 3D data processing.
*   **`generative/`**
    *   `SCTextGenerator`: Token-level probabilistic text synthesis.
    *   `SCAudioSynthesizer`: Waveform generation from bitstreams.
    *   `SC3DGenerator`: 3D point cloud and mesh export.
*   **`graphs/`**
    *   `StochasticGraphLayer`: Event-based Graph Neural Network (GNN).
*   **`hdc/`**
    *   `HDCEncoder`, `AssociativeMemory`: Hyperdimensional Computing logic.
*   **`chaos/`**
    *   `ChaoticRNG`: Logistic Map bitstream generator (True Randomness).

## 3. Physical & Biological Integration
*   **`bio/`**
    *   `GeneticRegulatoryLayer`: Gene Regulatory Network (GRN) dynamics.
    *   `DNAEncoder`: Bitstream-to-Nucleotide storage interface.
    *   `ConnectomeEmulator`: Whole-brain emulation framework.
    *   `NeuromodulatorSystem`: Dopamine/Serotonin emotional state dynamics.
*   **`physics/`**
    *   `StochasticHeatSolver`: Random walk PDE solver.
    *   `WolframHypergraph`: Universe evolution via hypergraph rewriting.
*   **`robotics/`**
    *   `StochasticCPG`: Central Pattern Generator for locomotion.
    *   `SwarmCoupling`: Brain-to-Brain synchronization protocol.
*   **`optics/`**
    *   `PhotonicBitstreamLayer`: Laser phase noise interference simulation.

## 4. Quantum & Meta-Computing
*   **`quantum/`**
    *   `QuantumStochasticLayer`: Variational Quantum Circuit (VQC) bridge.
    *   `AnyonBraidLayer`: Topological quantum computing.
*   **`meta/`**
    *   `TimeCrystalLayer`: Discrete Time Crystal (Period Doubling).
    *   `VacuumNoiseSource`: Zero-point energy harvesting.
    *   `OracleLayer`: Hyper-Turing predictive lookahead.
    *   `EventHorizonLayer`: Black Hole information scrambling.
    *   `CTCLayer`: Closed Timelike Curve (Time Travel) consistency solver.
    *   `RecursiveSelfImprover`: Singularity self-rewriting architecture.
    *   `OmegaIntegrator`: Omega Point universal integration.
    *   `AgentDAO`: Decentralized governance and consensus.
    *   `DarkForestAgent`: Fermi Paradox game-theoretic logic.

## 5. Post-Silicon & Speculative
*   **`post_silicon/`**
    *   `ReversibleLayer`: Adiabatic logic (Toffoli gates).
    *   `FemtoSwitch`: Quark Color Charge computing ($SU(3)$).
    *   `CellularComputer`: Synthetic cell molecular collision logic.
    *   `CatomLattice`: Programmable Matter (Claytronics) reconfiguration.
*   **`exotic/`**
    *   `MyceliumLayer`: Fungal network transport dynamics.
    *   `ReactionDiffusionSolver`: Chemical Turing pattern computing.
    *   `MechanicalLatticeLayer`: Stiffness-adaptive material logic.
    *   `RadHardLayer`: Space-hardened TMR (Triple Modular Redundancy).
    *   `DysonSwarmNet`: Matrioshka Brain hierarchical energy harvesting.
    *   `ConstructorCell`: Von Neumann Universal Constructor.
    *   `DysonPowerGrid`: Stellar-scale energy distribution.

## 5b. Implemented Modules (Not Listed Above)
*   **`nir_bridge/`**
    *   `from_nir()`, `to_nir()`: NIR graph import/export (18/18 primitives)
    *   `SCNetwork`: Topological execution engine for imported NIR graphs
    *   `SCSubgraphNode`, `SCMultiPortSubgraphNode`: Nested graph support
    *   Verified interop: Norse, snnTorch, SpikingJelly, Sinabs, Rockpool
*   **`compiler/`**
    *   IR graph builder, parser, verifier
    *   SystemVerilog + MLIR/CIRCT emitters
    *   Weight quantizer, type checker
*   **`identity/`**
    *   `IdentitySubstrate`: 3-population persistent SNN (HH + WB + HR)
    *   `TraceEncoder`: LSH text-to-spike encoding
    *   `StateDecoder`: PCA + attractor extraction
    *   `Checkpoint`: Lazarus protocol (save/restore/merge)
    *   `DirectorController`: L16 cybernetic self-regulation
*   **`network/`**
    *   Population-Projection-Network engine (3 backends: Python, Rust, MPI)
    *   Per-synapse delays, spike-gating, weight sparsity exploitation
    *   6 topology generators, 14 visualization plots
*   **`model_zoo/`**
    *   10 pre-built network configs, 3 pre-trained weight sets (MNIST, SHD, DVS)
*   **`conversion/`**
    *   `convert()`: ANN-to-SNN conversion (PyTorch → rate-coded SNN)
    *   `QCFSActivation`: conversion-aware training activation
*   **`training/`**
    *   Surrogate gradient training cells (LIF, ALIF, RecurrentLIF, EPropALIF)
    *   `DelayLinear`: trainable per-synapse delays with differentiable interpolation
    *   ConvSpikingNet (99.49% MNIST; evidence manifest:
        `benchmarks/results/mnist_conv_accuracy_reproducibility.json`)
*   **`learning/`**
    *   13 plasticity rules: pair/triplet/voltage STDP, BCM, BPTT, TBPTT, EWC, e-prop, R-STDP, MAML, homeostatic, STP, structural
*   **`adapters/`**
    *   Holonomic L1-L16 adapters with JAX acceleration
*   **`scpn/`**
    *   SCPN 16-layer stack implementations
*   **`solvers/`**
    *   ODE solvers (RK4, Euler)
*   **`datasets/`**
    *   Dataset loaders (MNIST, SHD, DVS)
*   **`cli/`**
    *   `sc-neurocore` command-line interface
*   **`spike_codec/`** — Spike Codec Library (6 codecs, unified API)
    *   `codec.py`: `SpikeCodec` — ISI + LEB128 varint baseline (50-200x).
    *   `predictive_codec.py`: `PredictiveSpikeCodec` — EMA predictor + XOR error coding for BCI implants.
    *   `delta_codec.py`: `DeltaSpikeCodec` — inter-channel XOR residuals for correlated probe arrays.
    *   `streaming_codec.py`: `StreamingSpikeCodec` — fixed-latency, independently decodable frames.
    *   `aer_codec.py`: `AERSpikeCodec` — address-event representation for neuromorphic routing.
    *   `waveform_codec.py`: `WaveformCodec` — end-to-end raw 10-bit waveform compression (spike detect + template match + LFP compress, 24x on 1024ch).
    *   `registry.py`: `get_codec()`, `list_codecs()`, `recommend_codec()` — unified lookup and auto-selection.

## 6. Systems, Pipeline & Tools
*   **`core/`**
    *   `TensorStream`: Unified high-performance data pipeline.
    *   `CognitiveOrchestrator`: Central executive with attention/goals.
    *   `DigitalSoul`: Agent state persistence and immortality.
    *   `VonNeumannProbe`: Code-level self-replication.
    *   `MindDescriptionLanguage`: Substrate-independent soul format.
    *   `MetaCognitionLoop`: Computational self-awareness.
*   **`pipeline/`**
    *   `DataIngestor`: Multimodal dataset preparation.
    *   `SCTrainingLoop`: Standard and RL training orchestration.
*   **`ensembles/`**
    *   `EnsembleOrchestrator`: Multi-agent mission coordination.
*   **`accel/`**
    *   `jit_kernels.py`: Numba JIT acceleration for bit-packing and MAC operations.
    *   `MPIDriver`: Distributed cluster-scale simulation.
*   **`export/`**
    *   `SCOnnxExporter`: ONNX-compatible JSON schema export.
*   **`hdl_gen/`**
    *   `VerilogGenerator`: Automated Verilog RTL generation.
    *   `SpiceGenerator`: Analog SPICE netlist export.
*   **`profiling/`**
    *   `EnergyProfiler`: 45nm Energy (J) and CO2e estimation.
*   **`security/`**
    *   `WatermarkInjector`: Backdoor weight injection.
    *   `ZKPVerifier`: Zero-Knowledge Proof commitments.
    *   `DigitalImmuneSystem`: Anomaly detection and threat response.
    *   `AsimovGovernor`: Ethical constraint implementation.
*   **`verification/`**
    *   `FormalVerifier`: Interval arithmetic proofs for safety properties.
    *   `CodeSafetyVerifier`: AST-based analysis for self-modifying code.
*   **`math/`**
    *   `CategoryTheoryBridge`: Functors between Stochastic, Quantum, and Bio domains.
*   **`interfaces/`**
    *   `BCIDecoder`: Neural signal (EEG) interface.
    *   `DVSInputLayer`: Event Camera (AER) interface.
    *   `PlanetarySensorGrid`: Global telemetry aggregation.
    *   `SymbiosisProtocol`: Human-AI thought exchange protocol.
    *   `InterstellarDTN`: Long-range delay-tolerant networking.
    *   `RealWorldInterfaces`: LSL and ROS2 mock bridges.
*   **`viz/`**
    *   `WebVisualizer`: HTML5/Canvas network topology renderer.
    *   `NeuroArtGenerator`: Generative AI for internal state expression.
*   **`analysis/`**
    *   `PhiEvaluator`: IT Consciousness ($\Phi$) measurement.
    *   `Explainability`: Spike-to-concept semantic mapping.
    *   `KardashevEstimator`: Civilization progress metrics.
    *   `QualiaTuringTest`: Subjectivity verification framework.

## 7. Deployment
*   `deploy/`: Dockerfile and Compose configuration for containerization.
