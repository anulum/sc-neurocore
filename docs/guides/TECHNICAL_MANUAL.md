# sc-neurocore Technical Manual
Version: 3.15.21
Last updated: 2026-04-13

This manual is the canonical, long-form reference for the sc-neurocore codebase. It is written for engineers and researchers who need to understand how the stochastic computing pipeline is constructed, how it maps to hardware, and how to extend it safely. It assumes basic familiarity with Python and numerical programming but does not require prior neuromorphic or FPGA experience.

The manual is intentionally verbose. It covers architecture, installation, core concepts, and advanced features, and it explains the reasoning behind key design choices. If you only need a quick start, see QUICKSTART_TUTORIAL.md. If you only need the public surface area, see API_REFERENCE.md.

## Table of contents
1. Architecture overview
2. Installation and setup
3. Core concepts
4. API reference and documentation workflow
5. Hardware integration
6. Advanced features
7. Troubleshooting
8. Appendix: glossary, defaults, and checklists

---

## 1. Architecture overview

### 1.1 Mission and design goals
sc-neurocore is a software and hardware co-design framework for stochastic computing (SC) applied to neural dynamics. The core goal is to provide a faithful, testable, and extensible implementation of stochastic neural models that can be executed in Python, exported to hardware (FPGA or ASIC flows), and integrated with the larger SCPN ecosystem. The code aims to balance three competing requirements:

- Scientific expressiveness. It must model a wide spectrum of neural and physical processes, including standard LIF neurons, advanced synapses with plasticity, and exotic computation models used in SCPN research.
- Engineering efficiency. The code must run fast enough to support iterative experimentation, and the hardware mapping must be realistic enough to be synthesized.
- Operational reliability. The system must provide deterministic configuration paths, robust testing, and clear documentation so it can be used by multiple agents without ambiguity.

The architecture is built around a consistent pipeline: encode inputs into bitstreams, perform SC arithmetic using logic-like operations, feed neurons and recorders, and then return measurements or bitstreams for downstream processing. The same abstractions are reused across many modules so that new layers can be added without rewriting the data flow.

### 1.2 Package layout
The main package lives at `03_CODE/sc-neurocore/src/sc_neurocore`. It is intentionally modular. The most frequently used modules are:

- `neurons/` for neuron models such as StochasticLIFNeuron and advanced variants.
- `synapses/` for SC synapse primitives and plasticity rules.
- `layers/` for higher-level network layers such as dense, convolutional, recurrent, vectorized, and fusion layers.
- `sources/` for input current sources and bitstream generators.
- `recorders/` for spike and metric recording utilities.
- `utils/` for bitstream encoding, random number utilities, and helper logic.
- `accel/` for packed bitstream operations and vectorized acceleration.
- `hdl_gen/` and `hardware/` for export and hardware integration tooling.
- `interfaces/` for real-world signals such as DVS and BCI inputs.
- `learning/` for federated and lifelong learning experiments.
- `graphs/`, `transformers/`, `quantum/`, `optics/`, `bio/`, `exotic/`, and `meta/` for specialized models used in SCPN research.

Documentation lives under `03_CODE/sc-neurocore/docs`. The docs are designed to be complementary: quick start, manuals, advanced usage, and performance tuning exist as separate files so the technical manual can stay focused and readable.

### 1.3 Data flow and execution model
At a high level, most simulations follow this flow:

1. Input normalization: raw inputs are normalized into a numeric range suitable for bitstream encoding (typically [0, 1] for unipolar SC).
2. Bitstream encoding: values are encoded into Bernoulli bitstreams of length N.
3. Synapse and dot product: bitstreams are multiplied (AND) or aggregated (MUX or popcount) to compute a current.
4. Neuron update: neurons integrate the current, apply leak and noise, and optionally emit spikes.
5. Recording and metrics: spikes are recorded and converted to rates, histograms, or other statistics.
6. Downstream layers: outputs may be decoded back into probabilities or passed as bitstreams to subsequent layers.

The system supports both bit-level simulation and probability-level approximations. Some modules, such as SCRecurrentLayer, explicitly choose a soft simulation approach to reduce computational cost. The difference is important when comparing hardware results or precision requirements.

### 1.4 Bitstream pipeline as a first-class abstraction
The bitstream is the core data type. Most operations are defined in terms of the probability of a bit being 1 rather than the exact bit pattern. This has two implications:

- Individual bit patterns are less important than statistical properties.
- Correlation between bitstreams matters. Two bitstreams with identical marginal probabilities can produce very different results if they are correlated.

The codebase uses separate RNG streams (often with offsets from a base seed) to avoid unintended correlations. When designing new layers, always consider how random streams interact. For example, using the same seed for input encoding and weight encoding can introduce correlations that distort multiplication results.

### 1.5 Layer stack and SCPN alignment
sc-neurocore contains a wide catalog of layers to support SCPN research. The 16 SCPN layers are represented by different models or combinations of models. The mapping is not rigid; instead, sc-neurocore provides primitives that can be assembled into a layer-specific pipeline. Key building blocks include:

- Dense layers using shared or per-neuron SC sources.
- Convolutional layers for local spatial processing.
- Recurrent and reservoir layers for temporal dynamics.
- Vectorized layers for high-throughput compute.
- Memristive layers to model hardware non-idealities.
- Fusion layers to combine multimodal inputs.
- Attention and transformer blocks for higher-order integration.

The goal is not to force every SCPN layer into a single class, but to provide a toolbox that can be assembled into layer-specific simulations. This also allows different deployment strategies for software vs hardware runs.

### 1.6 Software vs hardware parity
The Python implementation is designed to be bit-true when required, but some modules intentionally use approximations for performance. If you are validating hardware behavior, prefer modules that operate on explicit bitstreams or packed bitstreams. If you are exploring theory or doing high-level experiments, you can use probability-level approximations to save runtime.

Parity guidelines:

- If a function uses explicit bitstreams and logical operations (AND, MUX), it is closer to hardware behavior.
- If a function uses float math directly (dot product, tanh), treat it as a statistical approximation.
- The vectorized layer provides a middle ground by packing bitstreams into 64-bit words and using bitwise operations to approximate hardware-level multiplication.

### 1.7 Determinism and reproducibility
Reproducibility is essential when tests and papers depend on outcomes. The codebase supports deterministic runs by allowing seeds to be set at different levels:

- Global numpy seed for deterministic module initialization.
- Base seeds passed into layers to generate unique but reproducible RNG streams.
- Dedicated RNG utilities for neurons and synapses.

Best practice: for any experiment that will be compared or published, set a top-level seed and ensure that all subordinate modules derive their seeds from that. Also store bitstream length and encoding parameters in your experiment logs.

### 1.8 Scaling and complexity
Stochastic computing accuracy grows with the square root of bitstream length. That means doubling accuracy requires a 4x increase in length. sc-neurocore lets you tune length per layer to balance speed and accuracy. This matters because different SCPN layers have different tolerance for noise and different timescales. In practice:

- Use shorter bitstreams for exploratory runs and for layers that act as filters or normalizers.
- Use longer bitstreams for layers that require precise coupling or long-range coherence.
- Use packed bitstream operations when simulating large networks on CPU.

---

## 2. Installation and setup

### 2.1 Requirements
The minimum environment is Python 3.10+, numpy, and standard library modules. Additional dependencies are optional and used for specific modules. For example, matplotlib or plotly are required for visualization, and scipy or pandas may be required for some simulation layers. If you intend to run FPGA or PYNQ flows, you will need vendor-specific toolchains and drivers.

### 2.2 Local virtual environment
Use a local virtual environment. Do not install dependencies globally.

Steps:
1. Create the environment:
   `python -m venv .venv`
2. Activate:
   - Windows: `.\.venv\Scripts\activate`
   - Linux/macOS: `source .venv/bin/activate`
3. Install required packages (minimal):
   `pip install numpy`

Optional packages can be installed as needed, but keep the core environment light. This allows tests and CI to remain fast.

### 2.3 Running tests
The test suite uses pytest. Tests are organized by module, with separate directories for core layers, advanced layers, and research models. You can run the full suite or a subset.

Examples:
- `python -m pytest 03_CODE/sc-neurocore/tests`
- `python -m pytest 03_CODE/sc-neurocore/tests/test_layers`

Performance tests are gated by an environment variable. To enable them, set `SC_NEUROCORE_PERF=1` before running pytest. This prevents CI and quick local runs from being slowed down by performance checks.

### 2.4 Project structure and conventions
Key conventions:

- All code is under `src/sc_neurocore` and uses absolute imports from the package root.
- Tests are under `tests/` and use pytest naming conventions.
- Docs live under `docs/` and are written in Markdown.
- Hardware collateral (HDL, SPICE, PYNQ) is placed in dedicated folders to avoid mixing with core Python modules.

### 2.5 Reproducibility checklist
Before a run you intend to compare or publish, check the following:

- Set a global numpy seed.
- Set layer-level base seeds where supported.
- Record bitstream length and encoding ranges.
- Record any randomness in initialization (e.g., weights).
- Record any environment-dependent settings (CPU type, OS, BLAS library).

### 2.6 Configuration patterns
Most classes accept parameters at initialization. For complex experiments, use a small configuration dictionary or dataclass and pass it through. Avoid relying on global state. This makes it easier to share experiments across agents and keep results stable.

### 2.7 Path and OS notes
The repository is Windows-friendly but also works on Linux. When writing scripts, prefer `os.path.join` or pathlib. Avoid hard-coded absolute paths in notebooks or experiments, and always use project-relative references.

---

## 3. Core concepts

This section is intentionally detailed. It explains the foundational ideas behind sc-neurocore and provides practical guidance for working with the code. The goal is to ensure that anyone extending the codebase understands the implicit assumptions that make stochastic computing behave correctly.

### 3.1 Stochastic computing primer
In stochastic computing, values are represented by the probability of a bit being 1 in a sequence. A length-N bitstream encodes a value x in [0,1] by making approximately N*x bits equal to 1. This representation is noisy but resilient, and it is particularly well suited for hardware implementations that can perform massively parallel bitwise operations.

Key points:

- Unipolar encoding maps x in [0,1] to bit probability x.
- Bipolar encoding maps x in [-1,1] to probability (x+1)/2.
- Multiplication of probabilities can be performed with a single AND gate (for unipolar streams).
- Weighted addition can be approximated with MUX logic.

The downside is precision. The standard deviation of the estimate scales as sqrt(x(1-x)/N). That means large N is required for high precision. This is why sc-neurocore treats bitstream length as a first-class tuning parameter.

### 3.2 Correlation and random streams
Correlation is the most common pitfall in SC systems. If two streams are correlated, the AND of those streams does not equal the product of probabilities. This is why sc-neurocore uses distinct RNG streams for different streams. When you add a new layer, follow these rules:

- Use independent RNG streams for input encoding and weight encoding.
- If you reuse streams, do it intentionally and document the reason.
- When combining streams, consider using scrambling or reshuffling if correlation is suspected.

### 3.3 Encoding and decoding
`BitstreamEncoder` provides a simple interface for converting values to bitstreams and back. It supports configurable ranges so that input values can be mapped to [0,1] or another range. The encoder is deterministic when seeded.

Decoding is typically done by averaging the bitstream or by using a rolling average when processing in streaming mode. When decoding, remember that a single bit has very little meaning. Always average over a window of bits.

### 3.4 Neuron models
The core neuron is the stochastic leaky integrate-and-fire (LIF) neuron. It integrates current, applies leak, and produces spikes when the membrane potential reaches a threshold. Noise can be added to simulate biological variability.

Important parameters:

- `tau_mem`: membrane time constant. Larger values integrate more slowly.
- `dt`: time step. Keep consistent with your simulation scale.
- `v_threshold`: spike threshold. Lower values produce more spikes.
- `noise_std`: controls stochasticity. In SC systems, noise can be both a feature and a risk.

Advanced models include dendritic neurons, homeostatic LIF, and fixed point variants. These are used in high-level SCPN layers or experimental modules.

### 3.5 Synapse models and plasticity
Synapses in sc-neurocore are represented by bitstreams as well. `BitstreamSynapse` stores a weight bitstream and applies it to input streams. `BitstreamDotProduct` combines multiple synapses to produce a current estimate.

Plasticity is implemented through STDP variants. `StochasticSTDPSynapse` updates weight probabilities based on pre and post spike timing. Higher-level layers such as `SCLearningLayer` integrate the STDP updates into a full layer, allowing experiments with learning dynamics.

### 3.6 Sources and current generation
`BitstreamCurrentSource` provides a way to generate a scalar current from multiple input streams. It uses encoders, synapses, and dot product operations internally. This is a common entry point for layer implementations that want to use a shared input source across multiple neurons.

### 3.7 Recording and analysis
The `BitstreamSpikeRecorder` records spikes and provides statistics such as firing rate and inter-spike interval histograms. Recorders are simple but critical. Use them to validate that neurons are behaving within expected ranges, and to compare different configurations.

For more complex experiments, use analysis utilities in `analysis/` or build custom recorders that log intermediate values for later inspection.

### 3.8 Layer types
sc-neurocore includes a variety of layer types. The most common ones are:

- `SCDenseLayer`: uses a shared SC current source for 1-D weights, or one SC
  current source per neuron for `(n_neurons, n_inputs)` dense weight matrices.
- `SCLearningLayer`: similar to dense but includes per-neuron synapses with STDP updates.
- `SCConv2DLayer`: convolutional layer using SC multiplication for local patches.
- `SCRecurrentLayer`: recurrent layer that updates a state vector based on inputs and previous state.
- `VectorizedSCLayer`: packed bitstream operations for fast CPU simulation.
- `MemristiveDenseLayer`: vectorized layer with hardware non-idealities such as stuck-at faults and variability.
- `SCFusionLayer`: weighted fusion of multiple modalities using SC-style arithmetic.
- `StochasticAttention` and `StochasticTransformerBlock`: attention and transformer primitives for higher-order integration.
- `StochasticGraphLayer`: graph convolution for relational data.

Each layer has different assumptions about input shapes. Always check the forward method signature and validate shapes in your experiments.

### 3.9 Accelerated bitstream operations
The vectorized operations in `accel/vector_ops.py` pack bitstreams into uint64 arrays. This allows 64 time steps to be processed with a single bitwise operation. `vec_and` and `vec_popcount` are core primitives. These functions are fast and deterministic, but they operate on packed representations, so you must ensure correct packing and unpacking.

### 3.10 Interfaces
The interfaces module contains input adapters:

- `DVSInputLayer` turns event camera events into a surface of probabilities and bitstreams.
- `BCIDecoder` converts continuous neural signals into bitstreams.

These interfaces provide bridge points between real-world data and SC processing. They are intentionally simple and should be extended for production use.

### 3.11 Learning systems
Federated and lifelong learning are implemented as separate modules to keep the core pipeline clean.

- `FederatedAggregator` implements majority vote aggregation and a secure sum protocol for bitstream gradients.
- `EWC_SCLayer` extends the learning layer with a consolidation step that captures important weights for lifelong learning.

These implementations are prototypes. They are intended for experimentation and should be extended if used for real research.

### 3.12 Solvers and optimization
The Ising solver in `solvers/ising.py` provides a simple Metropolis-Hastings update step for energy minimization problems. It is used as a quantum-inspired optimizer. Because it is stochastic, it can be used for exploration of energy landscapes rather than strict deterministic optimization.

### 3.13 HDC and symbolic layers
Hyperdimensional computing (HDC) is used for symbolic binding and associative memory. The `HDCEncoder` provides binding, bundling, and permutation. `AssociativeMemory` provides a clean-up memory using Hamming distance. These modules are useful for representing symbolic patterns or higher-level concepts within SCPN layers.

### 3.14 Quantum, optical, bio, and exotic layers
The specialized layers exist to explore unconventional computation models. Examples:

- `QuantumStochasticLayer` maps bitstream probabilities to simulated qubit rotations.
- `PhotonicBitstreamLayer` simulates bitstream generation from optical interference.
- `GeneticRegulatoryLayer` models protein-level modulation of neuron thresholds.
- `DNAEncoder` maps bitstreams to and from DNA strings.
- `MyceliumLayer`, `ReactionDiffusionSolver`, `MechanicalLatticeLayer`, `AnyonBraidLayer`, and `DysonSwarmNet` model fungal, chemical, mechanical, topological, and dyson swarm computing.
- `TimeCrystalLayer`, `VacuumNoiseSource`, and `OracleLayer` explore meta-computational ideas.

These layers are included because SCPN research is interdisciplinary. They are not all intended for direct deployment but serve as a library of experimental tools.

### 3.15 Testing and validation philosophy
Testing focuses on shape checks, value bounds, determinism with seeds, and edge cases. Because SC systems are inherently stochastic, tests are designed to tolerate randomness by using broad numerical ranges or deterministic seeds. This ensures tests catch regressions without being flaky.

When adding new modules:

- Provide at least 10 tests per new test file.
- Include edge cases, shape checks, and determinism checks.
- Gate performance tests behind an environment variable.

---

## 4. API reference and documentation workflow

### 4.1 API reference
The API reference is generated from docstrings using `scripts/generate_docs.py`. The resulting file is stored at `docs/API_REFERENCE.md`. This ensures that method signatures and short descriptions remain synchronized with code.

When you add new classes or functions, include a clear docstring. The docstring should explain purpose, parameters, and expected shapes. The generator extracts only the first line of each method docstring, so write a concise summary sentence there.

### 4.2 Documentation update flow
Recommended workflow for documentation updates:

1. Update or add docstrings in code.
2. Run the documentation generator:
   `PYTHONPATH=src:. python scripts/generate_docs.py`
3. Review `docs/API_REFERENCE.md` for any missing descriptions.
4. Run `PYTHONPATH=src:. pytest -q tests/test_generate_docs.py tests/test_public_docstring_policy.py` after generator or docstring-policy changes.
5. Update higher-level docs (this manual, EXAMPLES, HARDWARE_GUIDE, BENCHMARKS) as needed.

### 4.3 Navigation policy
MkDocs navigation is intentionally curated. Some public Markdown files are generated references, supplemental guides, or long-form research documents that should remain reachable through links without expanding the primary navigation.

The tracked `docs/navigation_policy.toml` file classifies every public Markdown file that is allowed to remain outside `mkdocs.yml` navigation. After adding or moving public Markdown files, run `PYTHONPATH=src:. pytest -q tests/test_docs_navigation_policy.py`. If the test fails, either add the page to `mkdocs.yml` or update the policy bucket and expected count with the reason.

### 4.4 Public docstring policy
The tracked `docs/docstring_policy.toml` file defines the audited Python files whose public module, class, function, method, and property docstrings are enforced. This gate is intentionally scoped: add files only after their public surface has been reviewed and documented accurately.

Run `PYTHONPATH=src:. pytest -q tests/test_public_docstring_policy.py` after changing docstrings in a policy-listed file. If the test fails for a newly cleaned file, add that file to the policy instead of weakening the gate.

### 4.5 Style guidelines
- Use plain ASCII in documentation to avoid rendering issues.
- Use short code blocks for examples, and explain them in surrounding text.
- Keep file paths project-relative.
- Prefer explicit parameter ranges when describing layer behavior.

---

## 5. Hardware integration

### 5.1 Overview
Hardware integration in sc-neurocore is centered on two paths:

- HDL generation for FPGA or ASIC flows.
- SPICE netlist generation for memristive crossbar modeling.

The Python simulation is the reference. Hardware targets should be validated against it using bit-true or statistically equivalent tests.

### 5.2 Verilog generation
`VerilogGenerator` builds a simple top-level module for networks composed of dense layers. The generator is intentionally limited and is intended as a scaffold for more complex HDL integration. It outputs a module with a fixed 8-bit input and output bus and internal wires linking layers.

If you need more advanced HDL features (e.g., per-layer width, parameterized bit widths, or custom modules), extend the generator rather than editing the emitted Verilog manually.

### 5.3 SPICE generation
`SpiceGenerator` outputs SPICE netlists for memristive crossbars. It maps weight values in [0,1] to conductances between G_off and G_on. This is useful for evaluating analog or mixed-signal behavior of memristive hardware.

When using the SPICE output:

- Validate that the resistance values are within realistic ranges for your target technology.
- Use a consistent scaling between software weights and hardware conductances.
- Include realistic load and measurement models in the SPICE environment.

### 5.4 FPGA and PYNQ deployment
The core hardware target is PYNQ-Z2 for prototyping. A typical deployment path is:

1. Use Python simulation to validate the layer configuration.
2. Generate HDL modules for core layers or map to hand-written HDL.
3. Integrate with a top-level design that includes AXI interfaces for control and data transfer.
4. Deploy on PYNQ and run validation using test vectors produced from the Python simulator.

Key considerations:

- Clock frequency trade-offs: 100 to 250 MHz is typical for stability.
- Memory bandwidth: bitstream data can be large; use streaming interfaces and avoid unnecessary buffering.
- Fixed point vs floating point: most SC operations can be represented with simple integer logic, reducing resource usage.

### 5.5 DVS integration
The DVS input layer provides a pipeline from event data to bitstreams. For hardware, the interface is typically a stream of (x, y, t, p) events. The software layer uses an accumulation surface with exponential decay. When mapping to hardware, consider:

- Using on-chip memory for the surface state.
- Using fixed-point arithmetic for decay and saturation.
- Limiting event rate to prevent overflow.

### 5.6 BCI integration
BCI integration requires normalizing continuous signals into probabilities and then generating bitstreams. In hardware, this can be done with a simple ADC and comparator chain. The software model uses mean amplitude per channel. For real-time systems, you may want to implement a rolling window or adaptive normalization.

### 5.7 Hardware verification
Hardware verification should use two types of tests:

- Bit-true tests: compare exact bitstream outputs for short sequences.
- Statistical tests: compare mean and variance of outputs over long runs.

The statistical tests are often more robust and better reflect SC behavior, but bit-true tests are useful for verifying control logic and state machines.

---

## 6. Advanced features

### 6.1 Federated learning
Federated aggregation is implemented as a majority vote across client gradient bitstreams. This is a simple model that captures the robustness of SC representations. For real federated learning, you would integrate secure aggregation protocols and include client weighting based on data size or reliability.

### 6.2 Lifelong learning
The EWC layer records a snapshot of weights and a simple importance measure (Fisher-like). This is used to limit catastrophic forgetting by discouraging changes to important weights. The current implementation is a scaffold; it shows how consolidation can be represented in an SC context, but it does not implement full EWC penalty logic. Use it as a starting point for research.

### 6.3 Transformers and attention
The transformer block uses SC attention and vectorized dense layers to approximate the transformer pipeline. It is simplified and assumes a single token in practice, but it demonstrates how attention can be represented in the SC domain. For multi-token inputs, you should build a position-wise feed-forward layer and handle shape alignment explicitly.

### 6.4 Graph neural networks
The graph layer provides a simple message-passing mechanism using adjacency matrix multiplication and a tanh nonlinearity. It is useful for modeling relational structure in SCPN layers such as social or noospheric dynamics. If you need more expressiveness, consider adding edge features or attention-based aggregation.

### 6.5 Quantum and optical models
The quantum hybrid layer maps input probabilities to a simulated qubit rotation. This is useful for exploring non-linear mappings that are difficult to achieve with standard SC logic. The photonic layer simulates bitstream generation from optical interference, capturing a hardware scenario where randomness is derived from physical noise.

### 6.6 Bio and genetic modules
Bio layers model protein production, gene regulation, and DNA encoding. These models are intentionally simplified but provide an interface for integrating biological data. Use them for hypothesis exploration, not for precise biological simulation.

### 6.7 Exotic and meta-computation
The exotic and meta modules are highly experimental. They represent conceptual computing paradigms such as time crystals, vacuum fluctuations, and oracle-like hyper-computation. These modules should be used carefully. They are intended to broaden the SCPN research space rather than serve as production components.

### 6.8 Export to ONNX schema
The ONNX exporter writes a JSON representation of a network and stores weights as sidecar numpy files. This provides a bridge to other frameworks, but it is not a full ONNX exporter. Treat it as an interchange format for experiments rather than a deployment-ready exporter.

### 6.9 Performance and profiling
The profiling tools under `profiling/` and the performance tuning guide provide guidance on bitstream length, clock frequency, and resource usage. When optimizing, start with the smallest acceptable bitstream length and increase only if accuracy demands it. Use vectorized layers for CPU experiments and validate accuracy against bit-level results for critical runs.

### 6.10 Compiler intelligence features (Waves 6–11)
SC-NeuroCore includes compiler intelligence features covering formal verification, safety certification, power management, carbon estimation, multi-target deployment, security hardening, supply-chain compliance, platform extensibility, and research hardware paradigms. The live hardware registry currently exposes **194 hardware profiles** across **38 platform classes**. These features are documented in dedicated guides:

- **[Compiler Intelligence Guide](compiler_intelligence.md)** — Reference for compiler-intelligence features with code examples and usage patterns.
- **[Research Platforms Guide](research_platforms.md)** — Deep-dive into the research platform classes (superconducting, spintronic, biological, quantum, thermodynamic, probabilistic, polariton, metamaterial, etc.).
- **[Platform Extensibility Guide](platform_extensibility.md)** — 3 mechanisms: TOML loader, runtime discovery hook, and `from_constraints()` auto-constructor.
- **[Safety Certification Guide](safety_certification.md)** — Automated certification pipeline for DO-254, IEC 61508, and ISO 26262.
- **[Carbon & Sustainability Guide](carbon_sustainability.md)** — Carbon footprint estimation, energy scheduling, and EU regulatory compliance.
- **[Verification & Debug Guide](verification_debug.md)** — 14 verification features from pre-RTL stability through post-silicon debug, HIL calibration, and digital twin monitoring.
- **[Multi-Target Deployment Guide](multi_target_deployment.md)** — Auto-target recommendation, heterogeneous dispatch, UCIe chiplet mapping, and topology optimisation.
- **[Universal Coverage API Reference](universal_coverage_api_reference.md)** — Complete API reference for §52–§59 and 4 platform classes (optical_io, acoustic, fluidic, space_qualified).
- **[Security and Sovereignty API Reference](security_sovereignty_api_reference.md)** — Complete API reference for §60–§67, `from_constraints()`, and 3 platform classes (magnonic, organic_bioelectronic, risc_v_sovereign).
- **[Adaptive Reliability API Reference](adaptive_reliability_api_reference.md)** — Complete API reference for §68–§76 and 4 platform classes (thermodynamic, probabilistic, polariton, metamaterial).

---

## 7. Troubleshooting

This section lists common issues and their likely causes.

### 7.1 Bitstream saturation
Symptom: outputs are always 0 or always 1. Likely causes:

- Input values outside the expected range.
- Improper normalization in encoders.
- Correlated bitstreams producing extreme results.

Fixes: clamp inputs, check encoder ranges, set separate seeds, or increase bitstream length.

### 7.2 Shape mismatches
Symptom: numpy dot errors or broadcasting errors.

Fixes: check that input shapes match layer configuration. For convolution layers, ensure input shape matches `in_channels`. For vectorized layers, ensure `n_inputs` matches the input length. For transformer blocks, use single-token inputs unless you implement a multi-token FFN.

### 7.3 Unstable firing rates
Symptom: firing rates jump unpredictably or saturate.

Fixes: adjust thresholds, increase tau_mem, reduce input current scale, or reduce noise_std. Check that bitstream length is sufficient for the desired accuracy.

### 7.4 Performance bottlenecks
Symptom: simulations slow for large networks.

Fixes: use VectorizedSCLayer, reduce bitstream length, or enable packed operations. For repeated experiments, cache bitstreams where possible.

### 7.5 File export issues
Symptom: HDL or SPICE files not created or missing fields.

Fixes: verify file paths, ensure output directories exist, and confirm that you have write permissions. Use tmp paths for tests.

### 7.6 Hardware mismatch
Symptom: hardware outputs diverge from software results.

Fixes: verify that bitstream lengths and RNG seeds are aligned, use statistical comparisons rather than bit-true comparisons for long runs, and confirm that hardware timing matches software assumptions.

---

## 8. Appendix

### 8.1 Glossary
- SC: Stochastic Computing. A representation where probability encodes values.
- Bitstream: A sequence of bits used to represent a probability.
- LIF: Leaky Integrate-and-Fire neuron.
- STDP: Spike Timing Dependent Plasticity.
- HDC: Hyperdimensional Computing.
- DVS: Dynamic Vision Sensor.
- BCI: Brain-Computer Interface.
- FPGA: Field Programmable Gate Array.

### 8.2 Recommended defaults
These defaults are good starting points for most experiments:

- Bitstream length: 1024
- dt: 1.0 ms
- tau_mem: 20.0
- v_threshold: 1.0
- noise_std: 0.02
- learning_rate: 0.01

### 8.3 Documentation checklist
Use this checklist when adding a new module:

- Add docstrings for all classes and public methods.
- Add tests (10+ per new test file).
- Update API_REFERENCE.md via the generator.
- Update TECHNICAL_MANUAL.md if new architectural concepts are introduced.
- Update EXAMPLES.md if there is a new user-facing demo.

### 8.4 Experiment logging template
When you run a study that might be reused, log at least the following items. This reduces ambiguity and makes it possible to reproduce results exactly, even if the codebase evolves.

- Experiment name and purpose
- Date and time
- sc-neurocore version or git commit
- Random seeds (global and per-layer)
- Bitstream length for each layer
- Input normalization ranges
- Key hyperparameters (thresholds, tau_mem, learning rates)
- Environment details (OS, Python version, numpy version)
- Output artifacts (plots, CSV, JSON, or notebooks)

Even simple experiments benefit from this log. Stochastic systems can produce different results with tiny changes in random seeds or bitstream length, so a complete log saves time and prevents confusion in later analysis.

### 8.5 Calibration checklist
When you move from exploratory runs to calibrations or validation, use this checklist:

- Validate input scaling by encoding a known value and decoding it back.
- Confirm bitstream length yields acceptable variance for your metric.
- Run at least one deterministic test using fixed seeds.
- Compare software approximation layers against bit-true layers on a small case.
- For hardware targets, compare statistical distributions rather than single bit sequences.

---

## 9. Extended design notes

This section is a deeper reference for engineers who need to make architectural decisions or explain design choices in documentation, proposals, or research notes. It is not required reading for basic use, but it provides context that helps when the system is extended beyond its original scope.

### 9.1 Bitstream length planning
Bitstream length is the single most important parameter for accuracy and performance. Because SC estimates rely on random sampling, the variance of the estimate scales inversely with length. In practice, this means you should think of length as a precision knob, not a cosmetic detail.

When planning a simulation, start by defining the precision you need for your key outputs. If you need a metric accurate to within 1 percent, you will likely need lengths in the thousands. If you only need coarse behavior, lengths of a few hundred may be sufficient. Keep in mind that the required length depends on the distribution of values. Values near 0.5 require more samples to estimate accurately than values near 0 or 1.

It is often effective to use different lengths in different layers. For example, use short lengths in early layers that act as filters or normalization, and use longer lengths for layers that drive final decisions or high-level coherence metrics. This matches the idea of allocating precision where it matters most.

If you are working on hardware, remember that length translates to latency. Longer bitstreams mean more clock cycles for each operation. For FPGA systems, consider whether you can pipeline operations or interleave multiple streams to recover throughput.

### 9.2 Correlation mitigation strategies
Correlation is subtle but critical. Two streams that are correlated can produce biased results. This is not always obvious, because the bias may only appear in specific parameter regimes.

Common correlation sources include:

- Reusing the same RNG seed for multiple encoders
- Encoding multiple values with the same generator without reseeding
- Using identical weight and input streams for many neurons

Mitigation strategies:

- Use distinct seeds for each encoder and synapse.
- Derive seeds from a base seed using a large offset per component.
- Scramble or shuffle bitstreams if you must reuse them.
- Use low-discrepancy sequences (Sobol or similar) for deterministic streams and separate them by dimension.

The vectorized layer helps because it generates bitstreams for all weights and inputs in a single call and uses large random arrays. This tends to reduce accidental correlation compared to manual looping with small RNG calls.

### 9.3 Numerical stability and saturation
SC systems can saturate if inputs or weights are out of range. This is especially common when combining multiple layers, because the output of one layer may exceed the expected input range of the next. Saturation can lead to dead networks that always output 0 or 1.

To mitigate saturation:

- Normalize between layers. For example, divide by the maximum possible current or use a sigmoid mapping.
- Clip values to the encoding range before encoding.
- Use monitoring tools to detect when a layer output is saturated for long periods.

In recurrent systems, saturation can also cause fixed points that prevent dynamics. Use feedback scaling or noise injection to prevent lock-in if the goal is dynamic behavior.

### 9.4 Experiment logging and reproducibility, extended
For long-running experiments, include a minimal configuration file or JSON manifest. This should include all parameters required to reproduce the run. You can store it alongside output artifacts. Consider including a hash of the main configuration so that it can be quickly compared across runs.

If you are publishing results, include the following in the paper or appendix:

- sc-neurocore version
- Bitstream lengths and encoding ranges
- Summary of random seeds or RNG strategy
- Any approximations used (soft vs bit-true)

This makes results defensible and helps other teams replicate your work.

### 9.5 Integration with SCPN layers
sc-neurocore is designed to integrate with SCPN. When mapping SCPN layers to sc-neurocore modules, focus on functional behavior rather than one-to-one mapping. The intent is to capture dynamics, not to force a fixed structure.

Suggested mapping patterns:

- Low-level physical layers: use direct bitstream operations and LIF neurons.
- Mid-level cognitive layers: use fusion, recurrent, or graph layers.
- High-level coordination layers: use attention or transformer blocks.

The bridge modules in SCPN can use sc-neurocore as a backend. When doing so, pass parameters explicitly and avoid relying on implicit defaults. This makes integrations more stable over time.

### 9.6 Backward compatibility and API evolution
The codebase evolves quickly. When introducing breaking changes, follow these guidelines:

- Provide a migration note in the relevant documentation file.
- Keep old parameter names as aliases when possible.
- Add a test that verifies the old behavior if it is still supported.
- Update API_REFERENCE.md and this manual.

Backward compatibility is not guaranteed forever, but a short deprecation window reduces friction for collaborators.

### 9.7 Hardware roadmap notes
The current hardware path is oriented toward FPGA prototypes. If you move toward ASIC flows, revisit the following assumptions:

- Bitstream length and latency trade-offs will be different because of fixed clock rates.
- Memory bandwidth constraints are stricter, so streaming pipelines are preferred.
- Correlation handling must be explicit because RNG quality varies across hardware implementations.

When possible, keep a software reference for any hardware module and validate with statistical tests.

### 9.8 Precision validation workflow
Precision validation is a structured way to confirm that a bitstream-based model produces stable estimates. It is useful when you need to compare multiple implementations or when you want to justify a specific bitstream length in a report.

Suggested workflow:

1. Select a small set of representative inputs across the full range, including edge values near 0 and 1 and mid-range values near 0.5.
2. Run the model at a short bitstream length (for example 256), record the mean and variance of the outputs.
3. Increase the bitstream length by a factor of 4 and repeat. The variance should drop by roughly a factor of 2 if the system behaves as expected.
4. Compare the outputs against a non-stochastic baseline where possible. This may be a float computation or an analytical value.
5. Document the length at which further increases no longer change the output distribution meaningfully.

This process produces a defensible justification for the chosen length. It also helps catch hidden correlation issues, because correlated streams often show variance behavior that does not match the expected sqrt scaling.

### 9.9 Large-scale simulation operations
For large simulations, the bottleneck is often memory movement rather than arithmetic. The following operational practices reduce overhead:

- Use vectorized layers when the input size is large.
- Avoid per-step Python loops when a vectorized or packed operation is available.
- Precompute static bitstreams (for example fixed weights) and reuse them instead of regenerating each step.
- Use recorder objects sparingly and only for the metrics you need. Recording every spike for a large network can dominate runtime.
- Break large simulations into segments and checkpoint intermediate outputs. This is especially useful if you are running parameter sweeps or exploring multiple seeds.

When running on hardware, the same principle applies. The data path is often the limiting factor, so aim for steady streaming with minimal buffering.

### 9.10 Change management note
sc-neurocore evolves quickly because it sits at the intersection of research and engineering. When you add a new layer or alter an existing interface, plan for downstream impacts. Update tests and docs first, then update integration scripts, and finally update any higher level pipelines that import the module. If you are unsure whether a change is breaking, assume it is and document the change explicitly. Small notes here prevent hours of confusion later.

End of manual.
End of manual.
