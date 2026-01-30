# SC-NeuroCore API Reference

## Module `accel.jit_kernels`

### Function `jit_pack_bits(bitstream, packed_arr)`
Packs a uint8 bitstream into uint64 array.
bitstream: (N,) uint8 {0, 1}
packed_arr: (N//64,) uint64

### Function `jit_vec_mac(packed_weights, packed_inputs, outputs)`
Vectorized Multiply-Accumulate (MAC).
Simulates: Output[i] = Sum(Weights[i] AND Inputs)
weights: (n_neurons, n_inputs, n_words)
inputs: (n_inputs, n_words)
outputs: (n_neurons,)

---

## Module `accel.mpi_driver`

### Class `MPIDriver`
Distributed SC-NeuroCore Driver using MPI.
Handles partitioning and synchronization of bitstreams across cluster nodes.

- **__init__**()
- **scatter_workload**(global_inputs)
  - Distributes a large input array across nodes.
- **gather_results**(local_results)
  - Collects results from all nodes to Root.
- **barrier**()
  - Synchronize all nodes.

---

## Module `accel.vector_ops`

### Function `pack_bitstream(bitstream)`
Packs a uint8 bitstream (0s and 1s) into uint64 integers.
This allows processing 64 time steps in parallel.

Args:
    bitstream: Shape (N,) or (Batch, N) of uint8 {0,1}
    
Returns:
    packed: Shape (ceil(N/64),) or (Batch, ceil(N/64)) of uint64

### Function `unpack_bitstream(packed, original_length)`
Unpacks uint64 array back to uint8 bitstream.

### Function `vec_and(a_packed, b_packed)`
Bitwise AND on packed arrays. Simulates SC Multiplication.

### Function `vec_popcount(packed)`
Count total set bits (1s) in the packed array.
Used for integration/accumulation.

---

## Module `analysis.consciousness`

### Class `PhiEvaluator`
Evaluates Integrated Information (Phi) for SC Networks.
Based on 'PyPhi' principles but adapted for bitstreams.

- **entropy**(bitstream)
  - Shannon entropy of a bitstream distribution.
- **calculate_phi**(layer_outputs)
  - Approximates Phi for a layer state.

---

## Module `analysis.explainability`

### Class `SpikeToConceptMapper`
XAI Module: Maps spike patterns to semantic concepts.

- **__init__**(concept_map)
- **explain**(spikes)
  - Input: Spike vector (n_neurons,)

---

## Module `analysis.kardashev`

### Class `KardashevEstimator`
Calculates Civilization Type on the Kardashev Scale.

- **calculate_type**(power_watts)
  - K = (log10(P) - 6) / 10
- **estimate_from_compute**(ops_per_second, efficiency_j_per_op)
  - Estimate based on Landauer-limited computing.

---

## Module `analysis.qualia`

### Class `QualiaTuringTest`
Test for subjective experience (or simulation thereof).
Can the agent describe a novel internal state using meaningful metaphors?

- **__init__**(semiotics)
- **administer_test**(state_vector)
  - 1. Generate Art from state (The 'Qualia').

---

## Module `bio.dna_storage`

### Class `DNAEncoder`
Interface for DNA Data Storage.
Maps Bitstreams to Nucleotides (A, C, T, G).

- **encode**(bitstream)
  - Converts uint8 {0,1} bitstream to DNA string.
- **decode**(dna_str)
  - Converts DNA string back to bitstream.

---

## Module `bio.grn`

### Class `GeneticRegulatoryLayer`
Bio-Hybrid Layer.
Neural Activity -> Gene Expression (Protein) -> Neural Param Modulation.

- **__post_init__**()
- **step**(spikes)
  - Update protein levels based on spike activity.
- **get_threshold_modulators**()
  - Protein acts as inhibitor: Higher protein -> Higher threshold.

---

## Module `bio.neuromodulation`

### Class `NeuromodulatorSystem`
Global Emotional/Chemical System.
Modulates neuron parameters based on Dopamine (DA), Serotonin (5HT), Norepinephrine (NE).

- **update_levels**(reward, stress)
  - Adjust chemicals based on environmental feedback.
- **modulate_neuron**(neuron_params)
  - Returns modified parameters for a StochasticLIFNeuron.

---

## Module `bio.uploading`

### Class `ConnectomeEmulator`
Framework for Whole Brain Emulation (Consciousness Uploading).
Simulates massive sparse connectomes.

- **__post_init__**()
- **step**()
  - Executes one clock cycle of the entire brain slice.

---

## Module `chaos.rng`

### Class `ChaoticRNG`
Chaotic Random Number Generator using Logistic Map.
x_{n+1} = r * x_n * (1 - x_n)

Provides 'True' Randomness simulation (Deterministic Chaos)
unlike linear PRNGs.

- **__post_init__**()
- **random**(size)
  - Generate 'size' random floats [0, 1].
- **generate_bitstream**(p, length)
  - Generate bitstream using chaotic source.

---

## Module `core.immortality`

### Class `DigitalSoul`
Handles the persistence and 'immortality' of an SC Agent.
Captures full state (weights, traces, parameters) for restoration.

- **capture_agent**(orchestrator)
  - Extracts state from all modules registered in the orchestrator.
- **save_soul**(filepath)
  - Serializes the soul to a file.
- **load_soul**(cls, filepath)
  - Restores a soul from a file.
- **reincarnate**(orchestrator)
  - Injects the soul data back into an existing orchestrator's modules.

---

## Module `core.mdl_parser`

### Class `MDLSpecification`

### Class `MindDescriptionLanguage`
Parser for Mind Description Language (MDL).
A universal, substrate-independent format for archiving consciousness.

- **encode**(orchestrator, agent_name)
  - Exports the Orchestrator state to YAML MDL.
- **decode**(mdl_string)
  - Parses MDL back to a dictionary (for reconstruction).

---

## Module `core.orchestrator`

### Class `CognitiveOrchestrator`
Central Orchestrator for sc-neurocore Agents.
Connects disparate modules into a functional pipeline.

- **register_module**(name, module_obj)
- **set_attention**(module_name)
  - Focuses resources on a specific module.
- **execute_pipeline**(pipeline, initial_input)
  - Executes a sequence of modules.

---

## Module `core.replication`

### Class `VonNeumannProbe`
Simulates a self-replicating code entity.
Can copy the sc-neurocore source and its own state to a new 'host'.

- **replicate**(destination_dir)
  - Quine-like behavior: Copies the library source to a new location.

---

## Module `core.self_awareness`

### Class `SelfModel`

### Class `MetaCognitionLoop`
Implements Computational Self-Awareness.
Observes the Orchestrator and maintains a dynamic Self-Model.

- **observe**(orchestrator)
  - Introspection step. Reads internal state of the executive.
- **reflect**()
  - Returns a linguistic summary of the self-state.

---

## Module `core.tensor_stream`

### Class `TensorStream`
Unified Data Structure for sc-neurocore.
Handles automatic conversion between domains.

- **from_prob**(cls, probs)
- **to_bitstream**(length)
- **to_prob**()
- **to_quantum**()

---

## Module `dashboard.text_dashboard`

### Class `SCDashboard`
Simple CLI Dashboard for monitoring SC simulation.

- **__init__**(n_neurons)
- **update**(firing_rates, step)
- **_render**(step)

---

## Module `drivers.physical_twin`

### Class `PhysicalTwinBridge`
Bridge for Hardware-In-the-Loop (HIL) Synchronization.
Connects a Python Neuron to a physical PYNQ-Z2/FPGA neuron via TCP/Serial.

- **__init__**(ip, port)
- **sync_step**(sw_v_mem, sw_spike)
  - Sends software state, receives hardware state.

---

## Module `drivers.sc_neurocore_driver`

### Class `RealityHardwareError`
Raised when physical hardware is required but missing.


### Class `SC_NeuroCore_Driver`
Primary driver for the sc-neurocore FPGA overlay on PYNQ-Z2.

This driver enforces 'Reality Checks'. It will NOT run on standard x86 CPUs
unless explicitly in 'EMULATION' mode.

- **__init__**(bitstream_path, mode)
- **_connect_to_fpga**()
  - Attempts to load the PYNQ libraries and flash the bitstream.
- **write_layer_params**(layer_id, params)
  - Writes parameters to a specific layer's AXI-Lite registers.
- **run_step**(input_vector)
  - Executes one integration step on the FPGA.

---

## Module `drivers.verify_hardware_link`

### Function `verify_link()`
---

## Module `ensembles.orchestrator`

### Class `EnsembleOrchestrator`
Manages a collective of SC-NeuroCore Agents.
Implements ensemble consensus and coordinated action.

- **add_agent**(name, agent)
- **run_consensus**(pipeline, initial_input)
  - Runs the same pipeline on all agents and averages results.
- **coordinated_mission**(goal)
  - Assigns sub-tasks to agents based on their capabilities.

---

## Module `eschaton.computronium`

### Class `PlanckGrid`
Simulates a volume of Planck-Level Computronium.
Theoretical maximum density of computation.

- **bekenstein_bound**()
  - Maximum information (Entropy) in bits that can be contained in the sphere enclosing the mass.
- **bremermann_limit**()
  - Maximum processing speed (bits per second).
- **simulate_step**()
  - Simulate one Planck time step of processing.

---

## Module `eschaton.heat_death`

### Class `HeatDeathLayer`
Simulates computation at the Heat Death of the Universe.
Maximizes information processing per unit of remaining free energy.

- **__post_init__**()
- **compute_step**(bitstream)
  - Processes bits only if Free Energy > Landauer Limit.
- **status**()

---

## Module `eschaton.holographic`

### Class `HolographicBoundary`
Simulates the Holographic Principle (AdS/CFT correspondence).
3D Bulk dynamics are equivalent to 2D Boundary dynamics.

- **__post_init__**()
- **encode_to_boundary**(bulk_data)
  - Projects 3D bulk data onto the 2D boundary surface.
- **reconstruct_bulk**()
  - Reconstructs bulk representation from boundary bits.

---

## Module `eschaton.simulation`

### Class `NestedUniverse`
Simulation Hypothesis Engine.
Spawns child universes (simulations) within the parent.

- **spawn_simulation**(overhead)
  - Creates a child universe with a fraction of parent resources.
- **run_recursive_step**()
  - Propagates clock cycles down the simulation stack.

---

## Module `exotic.anyon`

### Class `AnyonBraidLayer`
Simulates Topological Quantum Computing using Fibonacci Anyons.
Information is encoded in the 'braid' of world-lines.

- **__post_init__**()
- **braid**(i)
  - Swaps anyon i and i+1.
- **measure**()
  - Collapses topological state to bitstream probabilities.

---

## Module `exotic.chemical`

### Class `ReactionDiffusionSolver`
Chemical Computing using Gray-Scott Reaction-Diffusion.

- **__post_init__**()
- **laplacian**(M)
- **step**()
- **get_state**()

---

## Module `exotic.constructor`

### Class `ConstructorCell`
Simulates a Universal Constructor (Von Neumann).
A cell capable of replicating itself and evolving structure.

- **replicate**()
  - Creates a copy of itself based on the blueprint.
- **mutate_blueprint**(rate)
  - Evolves instructions.

---

## Module `exotic.dyson_grid`

### Class `DysonPowerGrid`
Manages energy distribution for a Dyson Swarm.

- **__post_init__**()
- **step**(solar_output)
  - Simulate one time step.

---

## Module `exotic.fungal`

### Class `MyceliumLayer`
Fungal Computing Layer.
Simulates a dynamic mycelial network that reinforces active paths.

- **__post_init__**()
- **step**(inputs)
  - inputs: Activity at nodes.

---

## Module `exotic.matrioshka`

### Class `DysonSwarmNet`
Simulates a Matrioshka Brain (Dyson Swarm Computing).
Hierarchical nested shells processing at different 'temperatures'.

- **__post_init__**()
- **process**(input_energy)
  - Energy (data) flows from Inner Shell (Hot) to Outer Shell (Cold).

---

## Module `exotic.mechanical`

### Class `MechanicalLatticeLayer`
Mechanical Neural Network.
Computing with Stiffness (k) and Displacement (x).
F = K x

- **__post_init__**()
- **relax**(inputs, clamped_nodes)
  - Solve equilibrium: Sum(Forces) = 0.
- **train**()
  - Adjust stiffness to minimize stress?

---

## Module `exotic.space`

### Class `RadHardLayer`
Space-Hardened Layer with TMR (Triple Modular Redundancy) logic.
Simulates radiation effects (SEU) and correction.

- **forward**(input_values)
- **_noisy_forward**(input_values)

---

## Module `experiments.advanced_demo`

### Function `run_advanced_demo()`
---

## Module `experiments.agent_synergy_demo`

### Function `run_agent_demo()`
---

## Module `experiments.bitstream_drive`

### Function `run_bitstream_driven_lif(x_input, x_min, x_max, length, neuron_params)`
Drive a StochasticLIFNeuron with a bitstream-encoded input current.

Steps:
1. Encode scalar input current x_input in [x_min, x_max] as a unipolar
   bitstream of length `length`.
2. At each time step t, set:
   I_t = I_high if bitstream[t] == 1 else I_low
   or more simply, treat the bit directly as a scaled current.
3. Run neuron for `length` steps, collect spike bitstream.
4. Estimate:
   - input probability p_in from the input bitstream
   - firing probability p_fire from the spike bitstream

Returns
-------
input_bits : np.ndarray
    Input bitstream (0/1).
spike_bits : np.ndarray
    Output spike bitstream (0/1).
p_in : float
    Estimated input probability.
p_fire : float
    Estimated firing probability.

### Function `demo()`
---

## Module `experiments.blue_sky_demo`

### Function `run_blue_sky_demo()`
---

## Module `experiments.deep_research_demo`

### Function `run_deep_research_demo()`
---

## Module `experiments.demo_param_sweep`

### Function `run_pattern_trials(label, x_inputs, weight_values, n_neurons, T, noise_std, n_trials, base_seed)`
Run multiple trials of SCDenseLayer for a given pattern (x_inputs).
Return matrix of shape (n_trials, n_neurons) with firing rates.

### Function `nearest_centroid_multi(sample, centroids)`
Nearest-centroid classifier over K classes.
centroids[k]: firing-rate centroid for class k.

### Function `demo()`
---

## Module `experiments.demo_pattern_classification`

### Function `run_pattern_trials(label, x_inputs, weight_values, n_neurons, T, n_trials, base_seed)`
Run multiple trials of SCDenseLayer for a given pattern (x_inputs).
Return matrix of shape (n_trials, n_neurons) with firing rates.

### Function `nearest_centroid_classify(sample, centroid_A, centroid_B)`
Simple nearest-centroid classifier in firing-rate space.
Returns label 0 or 1.

### Function `demo()`
---

## Module `experiments.demo_pattern_classification_3class`

### Function `run_pattern_trials(label, x_inputs, weight_values, n_neurons, T, n_trials, base_seed)`
Run multiple trials of SCDenseLayer for a given pattern (x_inputs).
Return matrix of shape (n_trials, n_neurons) with firing rates.

### Function `nearest_centroid_multi(sample, centroids)`
Nearest-centroid classifier over K classes.
centroids[k]: firing-rate centroid for class k.

### Function `demo()`
---

## Module `experiments.demo_pattern_pca`

### Function `compute_pca_2d(X)`
Simple 2D PCA using SVD.

Parameters
----------
X : np.ndarray
    Data matrix of shape (n_samples, n_features)

Returns
-------
X_2d : np.ndarray
    Projection of X into 2D principal component space, shape (n_samples, 2)
mean : np.ndarray
    Mean vector of original data (n_features,)
components : np.ndarray
    PCA components (2, n_features)

### Function `demo_pca_plot()`
---

## Module `experiments.demo_poisson_spikes`

### Function `run_demo()`
---

## Module `experiments.demo_sc_dense_layer`

### Function `demo()`
---

## Module `experiments.demo_sc_pipeline`

### Function `demo()`
---

## Module `experiments.eschaton_demo`

### Function `run_eschaton_demo()`
---

## Module `experiments.exascale_demo`

### Function `run_exascale_demo()`
---

## Module `experiments.experimental_horizons_demo`

### Function `run_horizons_demo()`
---

## Module `experiments.final_frontier_demo`

### Function `run_frontier_demo()`
---

## Module `experiments.galactic_demo`

### Function `run_galactic_demo()`
---

## Module `experiments.governance_demo`

### Function `run_governance_demo()`
---

## Module `experiments.immortal_probe_demo`

### Function `run_immortal_demo()`
---

## Module `experiments.l7_symbolic_coupling`

### Function `gather_symbolic_features()`
### Function `run()`
---

## Module `experiments.learning_demo`

### Function `run_learning_experiment()`
---

## Module `experiments.mega_advancements_demo`

### Function `run_demo()`
---

## Module `experiments.meta_computing_demo`

### Function `run_meta_demo()`
---

## Module `experiments.post_silicon_demo`

### Function `run_post_silicon_demo()`
---

## Module `experiments.quantum_neuromorphic_demo`

### Function `run_demo()`
---

## Module `experiments.sapience_demo`

### Function `run_sapience_demo()`
---

## Module `experiments.sentient_demo`

### Function `run_sentient_demo()`
---

## Module `experiments.spatial_generative_demo`

### Function `run_spatial_gen_demo()`
---

## Module `experiments.system_level_demo`

### Function `run_system_demo()`
---

## Module `experiments.terminal_horizon_demo`

### Function `run_terminal_demo()`
---

## Module `experiments.transcendent_demo`

### Function `run_transcendent_demo()`
---

## Module `experiments.ultimate_frontier_demo`

### Function `run_ultimate_demo()`
---

## Module `experiments.unified_reality_demo`

### Function `run_unified_demo()`
---

## Module `experiments.whitepaper_benchmark`

### Function `run_whitepaper_benchmark()`
---

## Module `export.onnx_exporter`

### Class `SCOnnxExporter`
Exports SC Networks to ONNX-compatible JSON schema (or Protobuf if libs available).
Standard ONNX doesn't support 'StochasticBitstream' types natively.
We map SC layers to integer operations (MatMulInteger) or custom domains.

- **export**(layers, filename)
  - Export layer list to a JSON definition.

---

## Module `generative.audio_synthesis`

### Class `SCAudioSynthesizer`
A stub for SC Audio Synthesis.
Converts bitstreams/probabilities to waveform buffers.

- **synthesize_tone**(frequency, duration_ms, probability)
  - Synthesize a simple sine tone modulated by probability (amplitude).
- **bitstream_to_audio**(bitstream)
  - Roughly convert a bitstream to an audio signal (Filtering).

---

## Module `generative.text_gen`

### Class `SCTextGenerator`
A minimal token-level text generator for SC.
Maps probability distributions over vocabulary to tokens.

- **generate_token**(prob_dist)
  - Input: prob_dist (len(vocab),)
- **generate_sequence**(length)
  - Generate a random sequence of tokens.

---

## Module `generative.three_d_gen`

### Class `SC3DGenerator`
Adapter for generating 3D outputs (Mesh/Point Cloud).

- **export_point_cloud_json**(points, intensities, filename)
  - Exports a point cloud to a simple JSON format.
- **generate_surface_mesh**(voxel_grid)
  - Stub for generating a surface mesh from a voxel grid.

---

## Module `graphs.gnn`

### Class `StochasticGraphLayer`
Event-Based Graph Convolution Layer.
Message Passing happens via Bitstreams.

- **__init__**(adj_matrix, n_features)
- **forward**(node_features)
  - node_features: (N, Features)

---

## Module `hdc.base`

### Class `HDCEncoder`
Hyperdimensional Computing Encoder.
Dimension D usually >= 10,000.

- **generate_random_vector**()
  - Generates a random D-dimensional bipolar vector {-1, 1} or {0, 1}.
- **bind**(v1, v2)
  - XOR Binding operation.
- **bundle**(vectors)
  - Majority Bundling (Superposition).
- **permute**(v, shifts)
  - Cyclic shift (Permutation).

### Class `AssociativeMemory`
Simple HDC Associative Memory (Clean-Up Memory).
Stores (Key, Value) pairs or just prototypes.

- **__post_init__**()
- **store**(label, vector)
- **query**(query_vec)
  - Returns label of closest vector (Hamming Distance).

---

## Module `hdl_gen.spice_generator`

### Class `SpiceGenerator`
Generates SPICE netlists for Memristive Crossbars.

- **generate_crossbar**(weights, filename)
  - weights: (Rows, Cols) - Conductance values [0, 1] mapped to [G_off, G_on].

---

## Module `hdl_gen.verilog_generator`

### Class `VerilogGenerator`
Generates Top-Level Verilog for a defined SC Network.

- **__init__**(module_name)
- **add_layer**(layer_type, name, params)
- **generate**()
  - Emits Verilog code.
- **save_to_file**(path)

---

## Module `interfaces.bci`

### Class `BCIDecoder`
Brain-Computer Interface Decoder.
Converts continuous neural signals (e.g., EEG, LFP) into SC Bitstreams.

- **normalize_signal**(signal)
  - Normalize signal to [0, 1] for probability encoding.
- **encode_to_bitstream**(signal, length)
  - Encodes a [Channels, Time] signal block into [Channels, Bitstream_Length].

---

## Module `interfaces.dvs_input`

### Class `DVSInputLayer`
Interface for Dynamic Vision Sensors (Event Cameras).
Converts AER events (x, y, t, p) into SC Bitstreams.

- **__post_init__**()
- **process_events**(events)
  - Integrate a batch of events.
- **generate_bitstream_frame**(length)
  - Generate a HxWxLength bitstream cube from current surface state.

---

## Module `interfaces.interstellar`

### Class `Packet`

### Class `InterstellarDTN`
Delay-Tolerant Networking (DTN) for Interstellar Communication.
Uses 'Store-and-Forward' architecture.

- **receive**(packet)
  - Store packet in non-volatile memory.
- **step**()
  - Attempt to forward a packet.

---

## Module `interfaces.planetary`

### Class `PlanetarySensorGrid`
Planetary-Scale Computing (Gaia Interface).
Aggregates global telemetry into an SC computational field.

- **aggregate_field**(telemetry_data)
  - Fuses multi-regional telemetry (e.g. Temperature, CO2, Humidity)

---

## Module `interfaces.real_world`

### Class `LSLBridge`
Lab Streaming Layer (LSL) Bridge.
Connects EEG/Physiological streams to sc-neurocore.
(Mock implementation for standalone use).

- **__init__**(stream_name)
- **receive_chunk**(max_samples)
  - Simulates receiving a chunk of samples.

### Class `ROS2Node`
ROS 2 Interface Node.
Publishes motor commands from sc-neurocore to robots.

- **__init__**(node_name)
- **publish_cmd_vel**(linear_x, angular_z)
  - Simulates publishing to /cmd_vel.

---

## Module `interfaces.symbiosis`

### Class `SymbiosisProtocol`
Human-AI Symbiosis Interface.
Translates Semantics <-> Bitstreams.

- **encode_thought**(semantic_vector, urgency)
  - Human -> Machine.
- **decode_sensation**(bitstream)
  - Machine -> Human.

---

## Module `layers.attention`

### Class `StochasticAttention`
Stochastic Computing Attention Block.

Approximates: Output = Softmax(Q * K^T) * V

- **forward**(Q, K, V)
  - Input: 

---

## Module `layers.fusion`

### Class `SCFusionLayer`
Fuses multiple data modalities using Stochastic Multiplexing.

Inputs: Dictionary of feature vectors (e.g., {'audio': [...], 'visual': [...]})
Output: Fused feature vector.

- **__post_init__**()
- **forward**(inputs)
  - inputs: {'modality': np.array([values])}

---

## Module `layers.memristive`

### Class `MemristiveDenseLayer`
Simulates a Dense Layer mapped to a Memristor Crossbar.
Includes hardware non-idealities.

- **__post_init__**()
- **apply_hardware_defects**()
  - Corrupt weights based on physical properties.

---

## Module `layers.recurrent`

### Class `SCRecurrentLayer`
Stochastic Computing Recurrent Neural Network (RNN) / Reservoir Layer.

Inputs: [Batch, Time, Features] or just sequential vector inputs.
Internal State: Neurons connect to themselves (or each other).

- **__post_init__**()
- **step**(input_vector)
  - Process one time step (e.g., one frame of audio).
- **reset**()

---

## Module `layers.sc_conv_layer`

### Class `SCConv2DLayer`
Stochastic Computing 2D Convolutional Layer.

Processes 2D input (e.g., images) using SC bitstreams.

- **__post_init__**()
- **forward**(input_image)
  - input_image: (in_channels, H, W)

---

## Module `layers.sc_dense_layer`

### Class `SCDenseLayer`
Simple stochastic-computing "dense layer" of LIF neurons.

- Each neuron shares the same multi-channel BitstreamCurrentSource
  (same inputs + weights for now, can be diversified later).
- Each neuron has its own stochastic LIF parameters and RNG seed.
- We simulate T time steps and collect spike trains for all neurons.

This is software-only but fully SC-driven at the input/synapse level.

- **__post_init__**()
- **reset**()
- **run**(T)
  - Run the layer for T time steps, updating all neurons.
- **get_spike_trains**()
  - Return spike matrix of shape (n_neurons, T).
- **summary**()
  - Return firing statistics for each neuron.

---

## Module `layers.sc_learning_layer`

### Class `SCLearningLayer`
An SC Dense Layer with integrated STDP learning.
Each neuron has its own unique weights for the input vector.

- **__post_init__**()
- **run_epoch**(input_values)
  - Run one bitstream epoch (length 'length').
- **get_weights**()

---

## Module `layers.vectorized_layer`

### Class `VectorizedSCLayer`
High-Performance SC Layer using packed bitwise operations.
Simulates thousands of neurons efficiently on CPU.

- **__post_init__**()
- **_refresh_packed_weights**()
- **forward**(input_values)
  - Compute output firing rates for the layer.

---

## Module `learning.federated`

### Class `FederatedAggregator`
Privacy-Preserving Federated Learning using SC Bitstreams.

- **aggregate_gradients**(client_gradients)
  - Aggregates gradient bitstreams from multiple clients.
- **secure_sum_protocol**(client_gradients)
  - Simulates a secure aggregation where the server only sees the sum,

---

## Module `learning.lifelong`

### Class `EWC_SCLayer`
Lifelong Learning Layer using Elastic Weight Consolidation (Approx).

- **__post_init__**()
- **consolidate_task**()
  - Call after finishing a task.
- **apply_ewc_penalty**()
  - This would be called during the learning loop.

---

## Module `learning.neuroevolution`

### Class `SNNGeneticEvolver`
Genetic Algorithm for evolving SNN weights/parameters.

- **__init__**(layer_factory, fitness_func)
- **evolve**(generations)
- **_crossover**(p1, p2)
- **_mutate**(ind)

---

## Module `math.category_theory`

### Class `CategoryObject`

### Class `Morphism`
- **__init__**(func, name)
- **__call__**(obj)

### Class `CategoryTheoryBridge`
Functor mapping between distinct computational domains.
Stochastic <-> Quantum <-> Bio

- **stochastic_to_quantum**(bitstream)
  - Map bitstream probability p to quantum amplitude sqrt(p).
- **quantum_to_bio**(state_vector)
  - Map quantum probability |beta|^2 to concentration [0, 10] uM.
- **bio_to_stochastic**(concentration, length)
  - Map concentration to bitstream.
- **get_functor**(source, target)

---

## Module `meta.black_hole`

### Class `EventHorizonLayer`
Simulates information scrambling at a Black Hole Event Horizon.
Maps volume information to surface area bits (Holographic Principle).

- **__post_init__**()
- **scramble**(input_bitstream)
  - Input: (n_inputs, length)

---

## Module `meta.dao`

### Class `Proposal`

### Class `AgentDAO`
Decentralized Autonomous Organization for Agent Governance.
Uses 'Proof of Compute' as voting weight.

- **create_proposal**(action)
- **vote**(proposal_id, approve)
  - Cast vote weighted by credits.
- **finalize_proposal**(proposal_id)
  - Tally votes.

---

## Module `meta.fermi_game`

### Class `DarkForestAgent`
Game Theoretic Agent for the Fermi Paradox (Dark Forest Theory).
Decides whether to Broadcast or Hide.

- **decide**(alien_signal_strength)
  - Input: Strength of detected alien signal [0, 1].

---

## Module `meta.hyper_turing`

### Class `OracleLayer`
Simulates a Hyper-Turing Oracle.
Accesses future stream statistics to solve otherwise uncomputable tasks.

- **solve_halting**(bitstream)
  - Oracle function: Determines if a bitstream will eventually 'settle'
- **predictive_compute**(current_data, future_data)
  - Uses future knowledge to adjust current processing.

---

## Module `meta.omega`

### Class `OmegaIntegrator`
Simulates Omega Point Integration.
Final state where all information is unified.

- **unify**(system_states)
  - Losslessly integrates multiple bitstreams into a single 'God Qubit' state.

---

## Module `meta.singularity`

### Class `RecursiveSelfImprover`
Simulates a Singularity Architecture.
The network analyzes its own weights and generates improvements.

- **improve**(layer)
  - Analyzes weights and applies a 'intelligence explosion' gradient.

---

## Module `meta.time_crystal`

### Class `TimeCrystalLayer`
Simulates a Discrete Time Crystal (DTC).
Exhibits stable sub-harmonic oscillations (Period Doubling).

- **__post_init__**()
- **drive**(flip_pulse)
  - One cycle of the DTC drive:
- **get_bitstream**(cycles)
  - Generates bitstream over multiple drive cycles.

---

## Module `meta.time_travel`

### Class `CTCLayer`
Closed Timelike Curve (Time Travel) Simulation.
Finds a self-consistent state where Output(T) == Input(0).

- **compute_self_consistency**(transform_func)
  - Iterates the feedback loop until the state stabilizes

---

## Module `meta.vacuum`

### Class `VacuumNoiseSource`
Simulates harvesting computation from Vacuum Fluctuations (Zero Point Energy).
Uses the Casimir effect logic to correlate noise streams.

- **generate_virtual_bits**(length)
  - Produces bitstreams derived from simulated quantum fluctuations.

---

## Module `models.zoo`

### Class `SCDigitClassifier`
Pre-configured SC Network for MNIST-like Digit Classification.
Uses: Conv Layer -> Vectorized Dense Layer

- **__init__**()
- **forward**(image)
  - Classify a 28x28 image.

### Class `SCKeywordSpotter`
Audio Keyword Spotter (e.g., "Yes"/"No").
Uses: Recurrent / Dense Layer

- **__init__**(n_keywords)
- **predict**(mfcc_features)

---

## Module `neurons.base`

### Class `BaseNeuron`
Abstract base class for stochastic neuron models.

All neurons should expose:
- step(input_current) -> spike (0 or 1)
- reset_state()
- get_state() -> dict

- **step**(input_current)
  - Advance the neuron by one time step and return a spike (0 or 1).
- **reset_state**()
  - Reset the internal state to default / initial values.
- **get_state**()
  - Return a dict with the internal state (e.g., membrane potential).

---

## Module `neurons.dendritic`

### Class `StochasticDendriticNeuron`
Two-Compartment Neuron (Soma + 2 Dendrites).
Can solve non-linear problems (XOR) singly.

Structure:
Input A -> Dendrite 1
Input B -> Dendrite 2
Dendrite Output = NonLinear(Input)
Soma = Integrate(D1 + D2)

- **step**(input_a, input_b)
  - Inputs are probabilities/currents.

---

## Module `neurons.fixed_point_lif`

### Class `FixedPointLIFNeuron`
Bit-true fixed-point model of the Verilog sc_lif_neuron.

- **__post_init__**()
- **step**(leak_k, gain_k, I_t, noise_in)
  - Executes one clock cycle.

---

## Module `neurons.homeostatic_lif`

### Class `HomeostaticLIFNeuron`
LIF Neuron with Homeostatic Threshold Adaptation.
Self-regulates firing rate to a target setpoint.

- **step**(input_current)
- **get_state**()

---

## Module `neurons.sc_izhikevich`

### Class `SCIzhikevichNeuron`
Stochastic Izhikevich neuron (software-only).

Standard Izhikevich model:
v' = 0.04*v^2 + 5*v + 140 - u + I + noise
u' = a*(b*v - u)

When v >= 30 mV:
spike, then v <- c, u <- u + d

Here we add Gaussian noise to v' each step.

- **__post_init__**()
- **step**(input_current)
- **reset_state**()
- **get_state**()

---

## Module `neurons.stochastic_lif`

### Class `StochasticLIFNeuron`
Discrete-time noisy leaky integrate-and-fire neuron.

dv/dt = -(v - v_rest) / tau_mem + R * I + noise

We work in simple units:
- dt: time step
- tau_mem: membrane time constant
- v_threshold: firing threshold
- v_reset: reset potential
- noise_std: std dev of Gaussian noise added each step

- **__post_init__**()
- **step**(input_current)
- **reset_state**()
- **get_state**()
- **process_bitstream**(input_bits, input_scale)
  - Process a bitstream (array of 0s and 1s) as input current.

---

## Module `optics.photonic_layer`

### Class `PhotonicBitstreamLayer`
Simulates a Photonic Stochastic Computing Layer.
Uses Phase Noise (Laser Interference) to generate bitstreams.

- **simulate_interference**(length)
  - Simulates the interference of two laser beams with phase noise.
- **forward**(input_probs, length)
  - Generates bitstreams where '1' occurs if interference intensity < input_prob.

---

## Module `physics.heat`

### Class `StochasticHeatSolver`
Solves 1D Heat Equation using Stochastic Random Walks (Feynman-Kac).

- **__init__**(length, num_walkers, alpha)
- **step**()
  - Move walkers.
- **get_temperature_profile**()
  - Convert walker density to temperature.

---

## Module `physics.wolfram_hypergraph`

### Class `WolframHypergraph`
Simulates the Wolfram Physics Project Hypergraph.
Universe is a set of relations (Hyperedges).

- **evolve**(steps)
  - Applies a rewrite rule.
- **dimension_estimate**()
  - Estimates the effective dimension of the space graph.

---

## Module `pipeline.ingestion`

### Class `MultimodalDataset`
A container for multimodal training data.

- **get_sample**(idx)

### Class `DataIngestor`
Ingests and normalizes multimodal datasets for SC training.

- **prepare_dataset**(raw_data)
  - Normalizes and packages raw multimodal data.

---

## Module `pipeline.training`

### Class `SCTrainingLoop`
Standard and Reinforcement Learning loops for SC Networks.

- **run_rl_epoch**(agent, env_step_func, input_data, generations)
  - Runs a reinforcement learning epoch.
- **train_multimodal_fusion**(fusion_layer, dataset, epochs)
  - Stub for training weights in a fusion layer.

---

## Module `post_silicon.claytronics`

### Class `CatomLattice`
Programmable Matter Simulation (Claytronics).
Catoms rearrange to form optimal topology.

- **__post_init__**()
- **reconfigure**()
  - Catoms swap positions to group high-load units together (Heat dissipation logic).
- **get_topology**()

---

## Module `post_silicon.femto`

### Class `FemtoSwitch`
Simulates Femto-scale computing using Chromodynamics (Quark Colors).
States: 0 (Red), 1 (Green), 2 (Blue).
Interaction rules based on SU(3) symmetry (simplified).

- **interact**(quark_a, quark_b)
  - Interacts two streams of quarks.
- **bit_to_quark**(bitstream)
  - 0->Red(0), 1->Green(1).

---

## Module `post_silicon.reversible`

### Class `ReversibleLayer`
Simulates Reversible (Adiabatic) Logic.
Uses Toffoli (CCNOT) gates which are universal and reversible.
(a, b, c) -> (a, b, c XOR (a AND b))

- **toffoli_gate**(a, b, c)
  - Applies Toffoli gate to bitstreams.
- **reverse_toffoli**(a, b, c_prime)
  - Reverses the Toffoli gate.
- **forward**(input_a, input_b)
  - Simulates an AND gate reversibly.

---

## Module `post_silicon.synthetic_cell`

### Class `CellularComputer`
Simulates computing inside a Synthetic Cell.
Logic is driven by Brownian motion collisions between molecules and enzymes.

- **step**(inject_a, inject_b)
  - Inject reactants, simulate collisions, release product C.

---

## Module `profiling.energy`

### Class `EnergyMetrics`
- **reset**()
- **estimate_energy**()
- **co2_emission_g**(carbon_intensity_g_per_kwh)

### Function `track_energy(func)`
Decorator to track energy of a layer call (simulated).

---

## Module `quantum.hybrid`

### Class `QuantumStochasticLayer`
Simulates a Quantum-Classical Hybrid Layer.
Input bitstream probability -> Qubit Rotation -> Measurement Probability.

Mapping: p_in -> theta = p_in * pi
P_out = |<0|Ry(theta)|0>|^2 = cos^2(theta/2)
This non-linearity is useful for classification.

- **forward**(input_bitstreams)
  - input_bitstreams: (n_qubits, length)

---

## Module `recorders.spike_recorder`

### Class `BitstreamSpikeRecorder`
Record spikes over time and compute basic statistics.

Stores a 1D bitstream of spikes (0/1) and provides:
- total spikes
- firing rate (Hz) given dt (ms)
- inter-spike interval (ISI) histogram

- **record**(spike)
- **reset**()
- **as_array**()
- **total_spikes**()
- **firing_rate_hz**()
- **isi_histogram**(bins)
  - Compute histogram of inter-spike intervals in ms.

---

## Module `robotics.cpg`

### Class `StochasticCPG`
Central Pattern Generator using two mutually inhibiting neurons.
Generates rhythmic alternating outputs (e.g., Left/Right leg).

- **__post_init__**()
- **step**()

---

## Module `robotics.swarm`

### Class `SwarmCoupling`
Brain-to-Brain coupling for SC Networks (Swarm Intelligence).
Synchronizes two agents via mutual spike injection.

- **synchronize**(agent_a, agent_b)
  - Adjust weights of both agents to align their firing patterns.

---

## Module `scpn.layers.l1_quantum`

### Class `L1_StochasticParameters`
Parameters for the Stochastic L1 Layer.


### Class `L1_QuantumLayer`
Stochastic implementation of the Quantum Cellular Field.

- **__init__**(params)
- **step**(dt, external_field)
  - Advance the layer by one time step. 
- **get_global_metric**()
  - Return the global coherence metric (Phi-like).

---

## Module `security.ethics`

### Class `ActionRequest`

### Class `AsimovGovernor`
Implements the Three Laws of Robotics.
Vetoes actions that violate ethical constraints.

- **check_laws**(action)
  - Returns True if action is allowed, False if vetoed.

---

## Module `security.immune`

### Class `DigitalImmuneSystem`
Artificial Immune System (AIS) for Agent Security.
Detects anomalies (Non-Self) and neutralizes threats.

- **train_self**(normal_state)
  - Learn a 'Self' pattern (Normal behavior).
- **scan**(current_state)
  - Check if current state matches 'Self'.
- **_trigger_response**()

---

## Module `security.watermark`

### Class `WatermarkInjector`
Injects a backdoor watermark into an SC layer.

- **inject_backdoor**(layer, trigger_pattern, target_neuron_idx)
  - Modifies weights of 'target_neuron_idx' so it fires maximally
- **verify_watermark**(layer, trigger_pattern, target_neuron_idx)
  - Returns the activation of the target neuron for the trigger.

---

## Module `security.zkp`

### Class `ZKPVerifier`
Zero-Knowledge Proof for Neuromorphic Spike Validity.
Proves that a spike sequence matches a committed input without revealing input.

- **commit**(bitstream)
  - Creates a cryptographic commitment (hash) of the bitstream.
- **generate_challenge**(commitment)
  - Simulates a random index challenge.
- **verify**(commitment, challenge_idx, revealed_bit, bitstream_slice)
  - Verifies that the revealed bit and slice match the original commitment.

---

## Module `solvers.ising`

### Class `StochasticIsingGraph`
Quantum-Inspired Ising Machine Solver.

Spins S_i in {-1, 1} (mapped to 0, 1 for SC).
Energy E = -Sum(J_ij * S_i * S_j) - Sum(h_i * S_i).
Goal: Find configuration that minimizes E.

- **__post_init__**()
- **step**()
  - Perform one Metropolis-Hastings update step (parallel / cellular automaton style).
- **get_energy**()
  - Calculate global energy.
- **get_config**()

---

## Module `sources.bitstream_current_source`

### Class `BitstreamCurrentSource`
Multi-channel bitstream current source.

- Takes scalar inputs x_i in [x_min, x_max]
- Encodes each into a bitstream via BitstreamEncoder
- Passes them through BitstreamSynapses
- Uses BitstreamDotProduct to compute a scalar current I(t)
  for the neuron.

For now we assume static inputs and weights over the full length,
but you can extend this to time-varying later.

- **__post_init__**()
- **reset**()
- **step**()
  - Return the current I_t at the current time index and advance.
- **full_current_estimate**()
  - Estimate average current over full bitstream duration

---

## Module `spatial.representations`

### Class `VoxelGrid`
A 3D Voxel Grid representation for SC.
Each voxel stores a probability of being 'occupied'.

- **__post_init__**()
- **set_voxel**(x, y, z, prob)
- **get_as_bitstream**(length)
  - Converts the voxel grid to a 4D bitstream (X, Y, Z, Length).

### Class `PointCloud`
A Point Cloud representation.
Each point has (x, y, z) coordinates and an associated probability/intensity.

- **normalize**()

---

## Module `spatial.transformer_3d`

### Class `SpatialTransformer3D`
A transformer block specialized for 3D spatial data.
Processes voxel grids using SC attention.

- **__post_init__**()
- **forward**(voxel_grid)
  - Input: voxel_grid (res, res, res)

---

## Module `synapses.dot_product`

### Class `BitstreamDotProduct`
Compute a bitstream-level dot product using SC synapses.

Given:
- pre_bits: array of shape (n_inputs, length) with {0,1}
- synapses: list of BitstreamSynapse (length = n_inputs)

For each input i:
    post_i_bits = synapse_i.apply(pre_bits[i])

Then we sum probabilities:
    y(t) ~ sum_i w_i * x_i(t)

In 'pure' SC we could implement multi-bit accumulation via stochastic
adders, but for now we:
- decode each post_i_bits to its probability P_i
- compute y_scalar = sum_i P_i
- optionally map y_scalar into a current range [y_min, y_max].

- **__post_init__**()
- **n_inputs**()
- **apply**(pre_matrix, y_min, y_max)
  - Apply all synapses to the pre-synaptic bitstreams and compute

---

## Module `synapses.r_stdp`

### Class `RewardModulatedSTDPSynapse`
Reward-Modulated STDP Synapse.

Instead of updating weights immediately, we update an 'eligibility trace'.
Weights are only updated when a 'reward' signal is applied.

Delta_W = Learning_Rate * Reward * Eligibility

- **process_step**(pre_bit, post_bit)
- **apply_reward**(reward)
  - Global reward signal triggers weight update.

---

## Module `synapses.sc_synapse`

### Class `BitstreamSynapse`
Stochastic-computing synapse using bitstreams.

Each synapse has a weight w in [w_min, w_max].
For 'pure' SC mode, we encode w as a bitstream and AND it with the
pre-synaptic bitstream:

    out_bit[t] = pre_bit[t] & weight_bit[t]

In expectation:
    P(out=1) ≈ P(pre=1) * P(weight=1)
which corresponds to multiplication of the underlying probabilities.

For now we implement:
- encode_weight() -> weight_bitstream
- apply(pre_bits) -> post_bits

- **__post_init__**()
- **encode_weight**(w)
  - Encode scalar weight w into a unipolar bitstream.
- **update_weight**(new_w)
  - Change synaptic weight and recompute its bitstream.
- **apply**(pre_bits)
  - Apply synapse to a pre-synaptic bitstream.
- **effective_weight_probability**()
  - Decode the weight bitstream's probability P(weight_bit=1).

---

## Module `synapses.stochastic_stdp`

### Class `StochasticSTDPSynapse`
Stochastic Synapse with Spike-Timing-Dependent Plasticity (STDP).

Implements a simplified stochastic STDP rule:
- If PRE spikes and POST spikes shortly after -> Potentiation (LTP)
- If POST spikes and PRE spikes shortly after -> Depression (LTD)

Instead of full trace tracking, we use a probabilistic update based on
coincidence windows.

- **__post_init__**()
- **process_step**(pre_bit, post_bit)
  - Process a single time step, updating internal state and weight.
- **_potentiate**()
  - Increase weight probability.
- **_depress**()
  - Decrease weight probability.

---

## Module `transcendent.multiverse`

### Class `EverettNode`

### Class `EverettTreeLayer`
Simulates Many-Worlds Interpretation (MWI) Computing.
Every decision splits the universe.
We post-select the 'World' that solved the problem.

- **solve**(start_val, goal_func, transition_func)
  - Finds a path of choices (0 or 1) that leads to a state satisfying goal_func.

---

## Module `transcendent.noetic`

### Class `Sign`

### Class `SemioticTriad`
Simulates Noetic/Semiotic Computing.
Processing via Meaning Shifts (Metaphor/Metonymy).

- **__init__**()
- **learn_association**(concept, related)
- **interpret**(sign)
  - Shift meaning based on 'Interpretant'.
- **metaphor_distance**(start, end, depth)
  - Distance in meaning space (Noetic distance).

---

## Module `transcendent.spacetime`

### Class `SpinNode`

### Class `SpinNetwork`
Loop Quantum Gravity Spin Network.
Computing via topological evolution of spacetime graph.

- **__post_init__**()
- **pachner_move_1_3**(node_idx)
  - Simulates 1->3 Pachner move (Vertex subdivision).
- **calculate_volume**()
  - Volume of space = Sum of contributions from nodes.

---

## Module `transcendent.vacuum_decay`

### Class `FalseVacuumField`
Simulates Scalar Field Vacuum Decay.
Computing via Phase Transition Bubbles.

- **__post_init__**()
- **nucleate**(x, y)
  - Injects a True Vacuum bubble (Logic Input 1).
- **step**()
  - Propagate bubbles (Phase Transition).
- **measure_energy**()
  - Total energy released (Proportional to True Vacuum area).

---

## Module `transformers.block`

### Class `StochasticTransformerBlock`
Spiking Transformer Block (S-Former).
Structure:
Input -> Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm -> Output

- **__post_init__**()
- **forward**(x)
  - x: (Sequence_Length, d_model) - Probabilities [0,1]

---

## Module `utils.adaptive`

### Class `AdaptiveInference`
Manages Progressive Precision / Early Exit for SC.

- **run_adaptive**(step_func)
  - Runs the SC process step-by-step until convergence or max_length.

---

## Module `utils.bitstreams`

### Class `BitstreamEncoder`
Helper for encoding continuous scalar values into SC bitstreams
using linear unipolar mapping.
Example
-------
encoder = BitstreamEncoder(x_min=0.0, x_max=0.1, length=1024, seed=123)
bitstream = encoder.encode(0.06)  # 60% ones
p_hat = bitstream_to_probability(bitstream)
x_rec = encoder.decode(bitstream)

- **__post_init__**()
- **encode**(x)
- **decode**(bitstream)

### Class `BitstreamAverager`
Utility to accumulate bits over time and estimate probability on the fly.
Can be used, e.g., to estimate a neuron firing probability over a window.

- **__post_init__**()
- **push**(bit)
- **estimate**()
- **reset**()

### Function `generate_bernoulli_bitstream(p, length, rng)`
Generate a Bernoulli bitstream of given length with probability p of '1'.
This is the core SC primitive: a sequence of 0/1 bits where the
proportion of 1s ~ p.
Parameters
----------
p : float
    Probability of 1 (unipolar encoding, 0 <= p <= 1).
length : int
    Number of bits in the stream.
rng : RNG, optional
    RNG instance. If None, a fresh RNG is created.
Returns
-------
np.ndarray
    Array of shape (length,) with dtype=uint8, values in {0,1}.

### Function `generate_sobol_bitstream(p, length, seed)`
Generate a bitstream using a Sobol sequence (Low Discrepancy Sequence).
LDS provides faster convergence than random Bernoulli sequences (O(1/N) vs O(1/sqrt(N))).

Parameters
----------
p : float
    Target probability.
length : int
    Length of the bitstream.
seed : int, optional
    Seed for the Sobol engine.
    
Returns
-------
np.ndarray
    Array of shape (length,) with dtype=uint8, values in {0,1}.

### Function `bitstream_to_probability(bitstream)`
Decode a unipolar bitstream back into a probability estimate.
p_hat = (# of ones) / length

### Function `value_to_unipolar_prob(x, x_min, x_max, clip)`
Map a scalar x from [x_min, x_max] into a unipolar probability [0,1].
Linear mapping:
    p = (x - x_min) / (x_max - x_min)
If clip=True, x is clipped into [x_min, x_max].

### Function `unipolar_prob_to_value(p, x_min, x_max)`
Map a unipolar probability p in [0,1] back to a scalar in [x_min, x_max].
Inverse of value_to_unipolar_prob.

---

## Module `utils.connectomes`

### Class `ConnectomeGenerator`
Generates biologically plausible connectivity matrices.

- **generate_watts_strogatz**(n_neurons, k_neighbors, p_rewire)
  - Watts-Strogatz Small-World Model.
- **generate_scale_free**(n_neurons)
  - Barabasi-Albert Scale-Free Model (Preferential Attachment).

---

## Module `utils.decorrelators`

### Class `Decorrelator`
Base class for bitstream decorrelators.

- **process**(bitstream)

### Class `ShufflingDecorrelator`
Decorrelates a bitstream by randomly shuffling bits within a window.
This preserves the exact bit count (probability) but destroys temporal correlations.

- **__post_init__**()
- **process**(bitstream)

### Class `LFSRRegenDecorrelator`
Regenerates a new bitstream with the same probability estimate
but using a different random source (LFSR-like or just new RNG).

- **__post_init__**()
- **process**(bitstream)

---

## Module `utils.fault_injection`

### Class `FaultInjector`
Simulates hardware faults in Stochastic Computing bitstreams.

- **inject_bit_flips**(bitstream, error_rate)
  - Randomly flips bits with probability 'error_rate'.
- **inject_stuck_at**(bitstream, fault_rate, value)
  - Simulates Stuck-At-0 or Stuck-At-1 faults.

---

## Module `utils.fsm_activations`

### Class `FSMActivation`
Base class for FSM-based stochastic activation functions.

The FSM takes a bitstream input and transitions between states.
The output bit is determined by the current state (e.g., if state > N/2, out=1).
This implements saturating non-linearities like Tanh or Sigmoid efficiently.

- **__post_init__**()
- **step**(bit)
- **process**(bitstream)

### Class `TanhFSM`
Implements a Tanh-like function using a linear FSM.

States: 0 to N-1
Input 0: state -> max(0, state - 1)
Input 1: state -> min(N-1, state + 1)
Output: 1 if state >= N/2 else 0

- **__init__**(states)
- **step**(bit)

### Class `ReLKFSM`
Implements a Rectified Linear (ReLU-like) behavior.
Can be complex in SC, often approximated or used with bipolar coding.
Here we implement a simple saturating counter.

- **__init__**(states)
- **step**(bit)

---

## Module `utils.model_bridge`

### Class `SCBridge`
Bridge between standard DL frameworks (like PyTorch) and SC-NeuroCore.

- **load_from_state_dict**(state_dict, layer_mapping)
  - Load weights from a state_dict (numpy or torch tensors) into SC layers.
- **export_to_numpy**(layers)
  - Export SC weights back to numpy dictionary.

### Function `normalize_weights(weights)`
Normalizes weights to [0, 1] range for unipolar SC.

---

## Module `utils.rng`

### Class `RNG`
Thin wrapper around NumPy RNG to keep a single interface.

Later this can be extended to support:
- hardware TRNGs
- p-bit devices
- reproducible streams per neuron

- **__init__**(seed)
- **normal**(mean, std, size)
- **uniform**(low, high, size)
- **bernoulli**(p, size)
- **random**(size)
- **shuffle**(x)

---

## Module `verification.formal_proofs`

### Class `Interval`
- **__add__**(other)
- **__mul__**(other)
- **__repr__**()

### Class `FormalVerifier`
Simulated SMT Solver using Interval Arithmetic.
Proves properties of Stochastic Functions.

- **verify_probability_bounds**(input_interval, weight_interval)
  - Prove that Output Probability is always in [0, 1].
- **verify_energy_safety**(energy, cost)
  - Prove that operation will not consume more energy than available.

---

## Module `verification.safety`

### Class `CodeSafetyVerifier`
Formal Verification for Self-Modifying Code.
Analyzes AST to prevent catastrophic bugs in auto-generated updates.

- **verify_code_safety**(source_code)
  - Static analysis of source code for dangerous patterns.
- **verify_logic_invariant**(func, input_sample, expected_condition)
  - Dynamic verification (Unit Test on the fly).

---

## Module `viz.neuro_art`

### Class `NeuroArtGenerator`
Generates Art (Images) from Neural State.

- **generate_visual**(state_vector)
  - Maps a 1D state vector to a 2D RGB abstract image.

---

## Module `viz.web_viz`

### Class `WebVisualizer`
Generates a standalone HTML file to visualize the SC Network.

- **generate_html**(layers, filename)

---

## Module `world_model.planner`

### Class `SCPlanner`
A planner that uses a PredictiveWorldModel to select actions.

- **propose_action**(current_state, goal_state, n_candidates)
  - Propose the best action among n_candidates based on predicted outcome.
- **plan_sequence**(current_state, goal_state, horizon)
  - Simple greedy planning for a sequence of actions.

---

## Module `world_model.predictive_model`

### Class `PredictiveWorldModel`
A stochastic predictive world model.
Predicts state_next = f(state_curr, action).

- **__post_init__**()
- **predict_next_state**(current_state, action)
  - Predicts the next state given current state and action.
- **forecast**(initial_state, actions)
  - Forecast multiple steps ahead given a sequence of actions.

---

