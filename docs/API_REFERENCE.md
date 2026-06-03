# SC-NeuroCore API Reference

## Module `_native.array_guards`

### Function `require_c_contiguous(arr, name, dtype)`
Validate array layout before a zero-copy native call.

---

## Module `_native.core_engine_bridge`

### Function `_get_lib()`
Return the loaded Rust core-engine library or raise if absent.

### Function `is_available()`
Return True if the Rust core engine is loaded.

### Function `sc_multiply(a, b)`
SC multiply: AND of two u32 bitstreams.

### Function `sc_mux(a, b, sel)`
SC MUX: (sel & a) | (~sel & b).

### Function `sc_popcount(a)`
Population count of a u32.

### Function `sc_popcount64(a)`
Population count of a u64.

### Function `sc_popcount_packed(data)`
Popcount of packed u64 array (from Python list).

### Function `sc_popcount_packed_np(data)`
Popcount of packed u64 numpy array (zero-copy).

### Function `sc_scc_packed(a, b)`
Stochastic cross-correlation of packed u64 arrays (from Python list).

### Function `sc_scc_packed_np(a, b)`
Stochastic cross-correlation of packed u64 numpy arrays (zero-copy).

### Function `lfsr_encode_packed()`
Encode an LFSR-16 stochastic bitstream as packed little-endian u64 words.

### Function `lfsr_encode_bits()`
Encode an LFSR-16 stochastic bitstream as a uint8 0/1 numpy array.

### Function `_logical_bit_length(word_count, bit_length)`
### Function `_mask_trailing_words(words, bit_length)`
### Function `_validate_lfsr_contract()`
### Function `_python_lfsr_encode_packed()`
### Function `_python_scc_packed(a, b)`
---

## Module `_native.learning_bridge`

### Class `OnlineO1SnapshotFFI`
ctypes representation of the Rust fixed-point online O(1) snapshot.


### Class `RustOnlineO1Synapse`
RAII wrapper for the Rust bounded fixed-point online O(1) learner.

- **__init__**()
- **step**()
  - Advance one timestep and return the bounded fixed-point state.
- **per_synapse_state_bits**()
- **__del__**()

### Class `RustPlasticityRule`
RAII wrapper around a Rust plasticity rule handle.

rule_type: RULE_STDP(0), RULE_BCM(1), RULE_REWARD_STDP(2), RULE_ELIGENT(3)

- **__init__**(rule_type, weight, param_a, param_b)
- **step**(pre_spike, post_spike, dt, reward)
- **step_batched**(pre_spikes, post_spikes, rewards, dt)
  - Process Numpy arrays in a single FFI boundary crossing.
- **weight**()
- **reset**()
- **__del__**()

### Class `RustEligentLearner`
RAII wrapper around a Rust ELIGENT learner handle.

- **__init__**(threshold, target_rate, weight)
- **step**(fired, pre_spike, global_reward, dt)
- **step_batched**(fired_slice, pre_spikes, rewards, dt)
- **__del__**()

### Class `RustRuleLayer`
RAII wrapper around a Rust RuleLayerHandle for parallelized spatial (layer) execution.

- **__init__**(count, rule_type, weight, param_a, param_b)
- **__getstate__**()
- **get_state_dict**()
- **__setstate__**(state)
- **load_state_dict**(state_dict)
- **step**(pre_spikes, post_spikes, rewards, dt)
  - Process spatial Numpy arrays natively across Rayon Rust threads.
- **step_analog**(pre_probs, post_probs, rewards, dt, seed)
  - Process MTJ analogue probability states directly into sampled stochastic spikes using Rayon RNG.
- **get_weights**()
- **save**(path)
  - Serialize the internal traces and weights to a binary file natively.
- **load**(path)
  - Deserialize traces and weights from a binary file directly into memory.
- **reset**()
  - Zero each rule's plasticity traces; learned weights are preserved.
- **__del__**()

### Class `RustWgpuRuleLayer`
RAII wrapper around explicit Rust WGPU execution instances computing via WGSL bindings.

- **__init__**(count, rule_type, weight, param_a, param_b, tau_e, target_sum_weights)
- **step**(pre_spikes, post_spikes, rewards, dt)
- **step_analog**(pre_probs, post_probs, rewards, dt, seed)
  - Native true analog probabilistic emulation via WGSL.
- **get_weights**()
- **get_state_dict**()
- **load_state_dict**(state_dict)
- **reset**()
  - Zero WGSL plasticity traces; weights preserved. See `WgpuRuleLayer::reset`.
- **__del__**()

### Function `_get_lib()`
Return the loaded Rust learning library or raise if absent.

Narrows ``_lib`` from ``CDLL | None`` to ``CDLL`` at every call site so
mypy does not need per-method ``assert _lib is not None`` in each class
method. Kept intentionally cheap: one pointer-equality check per call.

### Function `_load_native_library()`
Load libautonomous_learning if present. Returns True on success, False if absent.

A missing shared library is a legitimate state: the Rust plasticity backend is
optional. Consumers must gate runtime use behind ``is_available()`` — the class
constructors below raise on construction when the library is absent. Raising at
import time couples every ``sc_neurocore`` import to a Rust build artefact and
breaks 398 unrelated tests when the crate has not yet been compiled.

### Function `set_deterministic_mode(seed)`
Force exact random generator seeds for bit-true SC formal verification.

### Function `is_available()`
Return True if the Rust learning engine is loaded.

---

## Module `accel.gpu_backend`

### Class `_BackendProxy`
Runtime-switching array namespace proxy.

- **__getattr__**(name)

### Function `_gpu_enabled()`
### Function `_mark_gpu_runtime_broken(exc)`
### Function `_warn_cpu_fallback()`
### Function `to_device(arr)`
Move a NumPy array to the active backend (GPU copy or no-op).

### Function `to_host(arr)`
Bring an array back to host RAM as a NumPy array.

### Function `_numpy_pack_bitstream(bits)`
### Function `_numpy_popcount(packed)`
### Function `gpu_pack_bitstream(bits)`
Pack uint8 {0,1} array into uint64 words.

Works on both CuPy and NumPy arrays.

Args:
    bits: Shape ``(N,)`` or ``(B, N)`` of uint8.

Returns:
    Packed uint64 array, shape ``(ceil(N/64),)`` or ``(B, ceil(N/64))``.

### Function `gpu_vec_and(a, b)`
Bitwise AND on packed uint64 arrays (SC multiplication).

### Function `gpu_popcount(packed)`
Vectorised SWAR popcount on uint64 arrays — returns per-element counts.

On CuPy this runs as a fused GPU kernel; on NumPy it uses the same
SWAR bit-trick as ``vector_ops.vec_popcount`` but returns an array
instead of a scalar.

### Function `gpu_vec_mac(packed_weights, packed_inputs)`
GPU-accelerated multiply-accumulate for a dense SC layer.

Args:
    packed_weights: ``(n_neurons, n_inputs, n_words)`` uint64
    packed_inputs:  ``(n_inputs, n_words)`` uint64

Returns:
    ``(n_neurons,)`` total bit counts (= SC dot products).

---

## Module `accel.jax_backend`

### Function `to_jax(arr)`
Move a NumPy array to the JAX device.

### Function `to_host(arr)`
Bring a JAX array back to host RAM as a NumPy array.

### Function `jax_pack_bitstream(bits)`
Pack uint8 {0,1} array into uint64 words using JAX.

---

## Module `accel.jit_kernels`

### Function `jit_pack_bits(bitstream, packed_arr)`
Packs a uint8 bitstream into uint64 array.
bitstream: (N,) uint8 {0, 1}
packed_arr: (N//64,) uint64

### Function `jit_vec_mac(packed_weights, packed_inputs, outputs)`
Vectorized Multiply-Accumulate (MAC).
Simulates: Output&#91;i&#93; = Sum(Weights&#91;i&#93; AND Inputs)
weights: (n_neurons, n_inputs, n_words)
inputs: (n_inputs, n_words)
outputs: (n_neurons,)

---

## Module `accel.mojo.runner`

### Class `MojoKernelRunner`
Manages execution and telemetry gathering for the underlying monolithic Mojo suite.

- **__post_init__**()
- **build**()
  - Helper to invoke `mojo build` natively across the active working directory.
- **run_benchmark**(timeout_sec)
  - Runs the entire kernel suite and parses output times natively in MS.
- **popcount**(data)
  - Call the Mojo SIMD kernel directly or fall back to Python.
- **lfsr_encode**(seed, threshold, bits)
  - Call the Mojo LFSR-16 encoder directly or fall back to Python.

---

## Module `accel.mojo_dispatch`

### Function `_detect_mojo()`
Check if Mojo is available via pixi.

### Function `_detect_rust()`
Check if Rust FFI .so is available.

### Function `_py_popcount32(word)`
### Function `_py_popcount_array(arr)`
### Function `_np_popcount64(packed)`
Wilkes-Wheeler-Gill popcount over uint64 array.

### Function `_np_pack_bitstream(bits)`
Pack uint8 {0,1} array into uint64 words.

### Function `_np_sc_and(a, b)`
### Function `_np_sc_xor(a, b)`
### Function `_np_sc_or(a, b)`
### Function `_np_sc_mux(a, b, sel)`
### Function `_np_popcount(packed)`
### Function `_np_scc(a, b, bit_length)`
### Function `pack_bitstream(bits)`
Pack a uint8 {0,1} bitstream into uint64 packed words.

Uses the fastest available backend.

### Function `popcount(packed)`
Count total set bits in a packed uint64 array.

### Function `sc_and(a, b)`
SC multiply (bitwise AND) on packed bitstreams.

### Function `sc_or(a, b)`
SC saturating addition (bitwise OR) on packed bitstreams.

### Function `sc_xor(a, b)`
SC absolute difference (bitwise XOR) on packed bitstreams.

### Function `sc_mux(a, b, sel)`
SC scaled addition (2:1 MUX) on packed bitstreams.

### Function `scc(a, b, bit_length)`
Stochastic correlation coefficient between two packed bitstreams.

### Function `vec_mac(weights, inputs)`
SC multiply-accumulate: popcount(weights AND inputs) per row.

Parameters
----------
weights : np.ndarray
    Shape (N, W) packed uint64 weight matrix.
inputs : np.ndarray
    Shape (W,) packed uint64 input vector.

Returns
-------
np.ndarray
    Shape (N,) int array of popcount results.

### Function `backend_info()`
Return info about available acceleration backends.

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

### Function `unpack_bitstream(packed, original_length, original_shape)`
Unpacks uint64 array back to uint8 bitstream.

Args:
    packed: Packed uint64 array (1D or 2D)
    original_length: Total number of bits to extract
    original_shape: Optional tuple for reshaping output (batch, length)

Returns:
    Unpacked bitstream of shape (original_length,) or original_shape

### Function `vec_and(a_packed, b_packed)`
Bitwise AND on packed arrays. Simulates SC Multiplication.

### Function `vec_xnor(a_packed, b_packed)`
Bitwise XNOR on packed arrays. SC bipolar multiplication: P(A XNOR B) = P(A)*P(B) + (1-P(A))*(1-P(B)).

### Function `vec_not(packed)`
Bitwise NOT on packed arrays. SC complement: P(NOT A) = 1 - P(A).

### Function `vec_mux(select_packed, a_packed, b_packed)`
Bitwise MUX on packed arrays. SC scaled addition: P(out) = P(sel)*P(A) + (1-P(sel))*P(B).

When sel is a Bernoulli(0.5) stream, this computes the average (A+B)/2.

### Function `vec_popcount(packed)`
Count total set bits (1s) in the packed array.
Used for integration/accumulation.

---

## Module `adapters.base`

### Class `BaseStochasticAdapter`
Abstract base class for all domain-specific adapters.

- **encode**(state)
  - Map domain state to stochastic bitstreams.
- **step_jax**(dt, inputs)
  - The JAX-accelerated mathematical kernel for the domain dynamics.
- **decode**(bitstreams)
  - Map stochastic bitstreams back to domain-specific observables.
- **get_metrics**()
  - Return domain-specific metrics (e.g. Coherence, Concentration).

---

## Module `adapters.holonomic._jax_compat`

### Function `make_rng(seed)`
Create a PRNG key (JAX) or seed array (NumPy fallback).

### Function `split_rng(key)`
Split a PRNG key into two children.

### Function `uniform(key, shape, minval, maxval)`
Uniform samples in &#91;minval, maxval).

### Function `normal(key, shape)`
Standard normal samples.

### Function `maybe_jit(fn)`
JIT-compile if JAX available, otherwise identity.

---

## Module `adapters.holonomic.dna_storage`

### Class `DNAEncoder`
Interface for DNA Data Storage.
Maps Bitstreams to Nucleotides (A, C, T, G).

- **encode**(bitstream)
  - Converts uint8 {0,1} bitstream to DNA string.
- **decode**(dna_str)
  - Converts DNA string back to bitstream.

---

## Module `adapters.holonomic.grn`

### Class `GeneticRegulatoryLayer`
Bio-Hybrid Layer.
Neural Activity -> Gene Expression (Protein) -> Neural Param Modulation.

- **__post_init__**()
- **step**(spikes)
  - Update protein levels based on spike activity.
- **get_threshold_modulators**()
  - Protein acts as inhibitor: Higher protein -> Higher threshold.

---

## Module `adapters.holonomic.l10_fire`

### Class `L10_HolonomicParameters`
Parameters derived from Paper 10 and Topological Insulation specs.


### Class `L10_FirewallAdapter`
JAX-traceable adapter for the SCPN Topological Firewall layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps firewall strength to stochastic bitstreams.
- **_firewall_kernel**(strength, intention, noise_inputs, gain, dt)
  - Solves the Firewall / Topological dynamics:
- **step_jax**(dt, inputs)
  - Advances the L10 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Firewall Integrity index.
- **get_metrics**()
  - Returns L10-specific metrics.

---

## Module `adapters.holonomic.l11_noos`

### Class `L11_HolonomicParameters`
Parameters derived from Paper 11 and NTHS specifications.


### Class `L11_NoosphericAdapter`
JAX-traceable adapter for the SCPN Noospheric layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps cultural spins to stochastic bitstreams.
- **_nths_kernel**(spins, field_input, j_avg, h_bias, dt)
  - Solves the NTHS Spin-Glass dynamics:
- **step_jax**(dt, inputs)
  - Advances the L11 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Noospheric Polarization index.
- **get_metrics**()
  - Returns L11-specific metrics like Polarization and Info Density.

---

## Module `adapters.holonomic.l12_gaian`

### Class `L12_HolonomicParameters`
Parameters derived from Paper 12 and MQN specifications.


### Class `L12_GaianAdapter`
JAX-traceable adapter for the SCPN Ecological-Gaian layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps ecological coherence to stochastic bitstreams.
- **_enaqt_kernel**(coherence, flow, j_coupling, noise_gain, dt)
  - Solves the ENAQT transport dynamics:
- **step_jax**(dt, inputs)
  - Advances the L12 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Gaian Synchrony Index.
- **get_metrics**()
  - Returns L12-specific metrics like Coherence and Flow.

---

## Module `adapters.holonomic.l13_source`

### Class `L13_HolonomicParameters`
Parameters derived from Paper 13 and Vacuum Lattice specs.


### Class `L13_SourceAdapter`
JAX-traceable adapter for the SCPN Source-Field layer.

- **__init__**(params, seed)
- **_validate_positive_int**(name, value)
- **_validate_params**(cls, params)
- **encode**(domain_state)
  - Maps vacuum potential to stochastic bitstreams.
- **_vacuum_kernel**(state, coupling, bias, dt)
  - Advances local spin-like vacuum lattice dynamics.
- **_vacuum_lattice_kernel**(state, coupling, bias, scission_rate, feedback_drive, dt)
  - Advances local spin-like vacuum lattice dynamics.
- **_project_feedback**(inputs)
- **step_jax**(dt, inputs)
  - Advances the L13 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Primordial Coherence.
- **get_metrics**()
  - Returns L13-specific metrics.

---

## Module `adapters.holonomic.l14_trans`

### Class `L14_HolonomicParameters`
Parameters derived from Paper 14 and Keystone Tuning specs.


### Class `L14_TransdimensionalAdapter`
JAX-traceable adapter for the SCPN Transdimensional layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps resonance alignment to stochastic bitstreams.
- **_resonance_kernel**(alignment, pta_input, keystone_f, dt)
  - Solves the Inter-brane Resonance dynamics:
- **step_jax**(dt, inputs)
  - Advances the L14 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Brane Alignment index.
- **get_metrics**()
  - Returns L14-specific metrics.

---

## Module `adapters.holonomic.l15_cons`

### Class `L15_HolonomicParameters`
Parameters derived from Paper 15 and executive optimization specs.


### Class `L15_ConsiliumAdapter`
JAX-traceable adapter for the SCPN Consilium layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps executive optimization state to stochastic bitstreams.
- **_umo_kernel**(metric, layer_coherences, target, lr, dt)
  - Solves the UMO / SEC optimization:
- **step_jax**(dt, inputs)
  - Advances the L15 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Global Coherence Index.
- **get_metrics**()
  - Returns L15-specific metrics.

---

## Module `adapters.holonomic.l16_meta`

### Class `L16_HolonomicParameters`
Parameters derived from Paper 16 and Meta-Layer specifications.


### Class `L16_MetaAdapter`
JAX-traceable adapter for the SCPN Cybernetic Closure layer (The Director).

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps director's will to stochastic bitstreams.
- **_director_kernel**(will, gci_input, entropy, threshold, dt)
  - Solves the Recursive Closure dynamics:
- **step_jax**(dt, inputs)
  - Advances the L16 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Cybernetic Will index.
- **get_metrics**()
  - Returns L16-specific metrics.

---

## Module `adapters.holonomic.l1_quantum`

### Class `L1_HolonomicParameters`
Parameters derived from Paper 1 and Monograph 28.


### Class `L1_QuantumAdapter`
JAX-traceable adapter for the SCPN Quantum Biological layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps coherence probabilities to stochastic bitstreams.
- **_ignition_kernel**(coherence, s_pump, s_crit, gamma, f_prot, dt)
  - Solves the Ignition / Metabolic Coherence dynamics:
- **step_jax**(dt, inputs)
  - Advances the L1 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to global coherence metric.
- **get_metrics**()
  - Returns L1-specific metrics like Coherence and Pumping levels.

---

## Module `adapters.holonomic.l2_chem`

### Class `L2_HolonomicParameters`
Parameters derived from Paper 2 and Monograph 28.


### Class `L2_NeurochemicalAdapter`
JAX-traceable adapter for the SCPN Neurochemical layer.

- **__init__**(params, seed)
- **_validate_positive_int**(name, value)
- **_validate_params**(cls, params)
- **encode**(domain_state)
  - Maps neurochemical concentrations to stochastic bitstreams.
- **_iiief_kernel**(phi, velocity, integrated_info, alpha, c_info, dt)
  - Advances the damped finite-difference IIIEF wave equation.
- **step_jax**(dt, inputs)
  - Advances the L2 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to neurochemical concentrations.
- **get_metrics**()
  - Returns L2-specific metrics like Field Potential and Tonus.

---

## Module `adapters.holonomic.l3_gen`

### Class `L3_HolonomicParameters`
Parameters derived from Paper 3 and the CBC Bridge specification.


### Class `L3_GenomicAdapter`
JAX-traceable adapter for the SCPN Genomic/Epigenomic layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps accessibility states to stochastic bitstreams.
- **_cbc_kernel**(v_bio, p_spin, alpha_b, g_op, dt)
  - Solves the CBC Bridge transduction:
- **step_jax**(dt, inputs)
  - Advances the L3 CBC bridge dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to average genomic accessibility.
- **get_metrics**()
  - Returns L3-specific metrics like Spin Polarization and Bio-Potential.

---

## Module `adapters.holonomic.l4_cell`

### Class `L4_HolonomicParameters`
Parameters derived from Paper 4 and T7 Validation protocols.


### Class `L4_CellularAdapter`
JAX-traceable adapter for the SCPN Cellular-Tissue layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps synchronization activity to stochastic bitstreams.
- **_kuramoto_kernel**(phases, omega, k, dt, noise)
  - Solves the Kuramoto-UPDE interaction:
- **step_jax**(dt, inputs)
  - Advances the L4 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Kuramoto order parameter.
- **get_metrics**()
  - Returns L4-specific metrics.

---

## Module `adapters.holonomic.l5_org`

### Class `L5_HolonomicParameters`
Parameters derived from Paper 5 and FEP (Free Energy Principle).


### Class `L5_OrganismalAdapter`
JAX-traceable adapter for the SCPN Organismal-Psychoemotional layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps organismal state to stochastic bitstreams.
- **_autonomic_kernel**(current, target, tau, dt)
  - Euler-integration of autonomic homeostasis.
- **step_jax**(dt, inputs)
  - Advances the L5 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to average valence and HRV coherence.
- **get_metrics**()
  - Returns L5-specific metrics.

---

## Module `adapters.holonomic.l6_plan`

### Class `L6_HolonomicParameters`
Parameters derived from Paper 6 and Gaia-field specifications.


### Class `L6_PlanetaryAdapter`
JAX-traceable adapter for the SCPN Planetary-Biospheric layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps planetary coherence to stochastic bitstreams.
- **_gaia_kernel**(phi, sync_inputs, alpha, freq, q_factor, p_percolation, t, dt)
  - Solves the Planetary Gaia-field dynamics:
- **_validate_params**(params)
- **step_jax**(dt, inputs)
  - Advances the L6 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Global Consciousness Index.
- **get_metrics**()
  - Returns L6-specific metrics like Gaia Potential and Schumann Alignment.

---

## Module `adapters.holonomic.l7_sym`

### Class `L7_HolonomicParameters`
Parameters derived from Paper 7 and Metatron's Cube geometry.


### Class `L7_SymbolicAdapter`
JAX-traceable adapter for the SCPN Geometrical-Symbolic layer.

- **__init__**(params, seed)
- **_init_metatron_matrix**()
  - Initialise a symmetric bounded Metatron routing topology.
- **_metatron_coordinates**(n_nodes)
- **_validate_params**(params)
- **encode**(domain_state)
  - Maps symbolic phases to stochastic bitstreams.
- **_symbolic_kernel**(phases, metatron, inputs, dt)
  - Solves the Symbolic routing dynamics:
- **step_jax**(dt, inputs)
  - Advances the L7 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Symbolic Coherence.
- **get_metrics**()
  - Returns L7-specific metrics like Routing Density.

---

## Module `adapters.holonomic.l8_cosm`

### Class `L8_HolonomicParameters`
Parameters derived from Paper 8 and PTA specifications.


### Class `L8_CosmicAdapter`
JAX-traceable adapter for the SCPN Cosmic Phase-Locking layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps cosmic phases to stochastic bitstreams.
- **_cosmic_kernel**(phases, pulsar_omegas, k_cosmic, dt)
  - Solves the Cosmic Phase-Locking dynamics:
- **step_jax**(dt, inputs)
  - Advances the L8 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Cosmic Alignment.
- **get_metrics**()
  - Returns L8-specific metrics.

---

## Module `adapters.holonomic.l9_mem`

### Class `L9_HolonomicParameters`
Parameters derived from Paper 9 and TSVF specifications.


### Class `L9_MemoryAdapter`
JAX-traceable adapter for the SCPN Existential Memory layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps memory imprints to stochastic bitstreams via TSVF overlap.
- **_tsvf_kernel**(psi, phi, inputs, strength, dt)
  - Updates the forward/backward holographic imprints.
- **step_jax**(dt, inputs)
  - Advances the L9 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Memory Retrieval quality.
- **get_metrics**()
  - Returns L9-specific metrics.

---

## Module `adapters.holonomic.neuromodulation`

### Class `NeuromodulatorSystem`
Global Emotional/Chemical System.
Modulates neuron parameters based on Dopamine (DA), Serotonin (5HT), Norepinephrine (NE).

- **update_levels**(reward, stress)
  - Adjust chemicals based on environmental feedback.
- **modulate_neuron**(neuron_params)
  - Returns modified parameters for a StochasticLIFNeuron.

---

## Module `adapters.neuroml`

### Class `ImportedCell`
Result of importing a NeuroML cell definition.


### Function `_strip_ns(tag)`
Remove XML namespace prefix.

### Function `_parse_unit_value(s)`
Parse NeuroML unit string like '10nS', '-65mV', '100pF' to SI-ish float.

Returns value in base NeuroML units (mV, nS, pF, ms, nA).

### Function `_parse_current_pa(s)`
Parse a NeuroML current string into pA for biophysical IF equations.

### Function `_import_iaf_cell(elem)`
Import <iafCell> or <iafRefCell>.

### Function `_import_iaf_tau_cell(elem)`
Import <iafTauCell> or <iafTauRefCell>.

### Function `_import_izhikevich_cell(elem)`
Import <izhikevichCell> (2003 dimensionless).

### Function `_import_izhikevich2007_cell(elem)`
Import <izhikevich2007Cell> (biophysical units).

Preserve the NeuroML 2 biophysical parameterisation.

### Function `_import_adex_cell(elem)`
Import <adExIaFCell>.

### Function `import_neuroml(path)`
Parse a NeuroML 2 XML file and return imported cell definitions.

Parameters
----------
path : str or Path
    Path to .nml or .xml file.

Returns
-------
list of ImportedCell
    One per cell definition found in the file.

### Function `create_neuron(cell)`
Instantiate an SC-NeuroCore neuron from an ImportedCell.

Returns a neuron object ready for .step() calls.

---

## Module `adapters.sonata`

### Class `SONATANode`
A single node (neuron) from a SONATA population.


### Class `SONATAEdge`
A single edge (synapse) from a SONATA population.


### Class `SONATANetwork`
Parsed SONATA network with nodes and edges.

- **n_nodes**()
- **n_edges**()
- **connectivity_matrix**()
  - Build dense connectivity matrix (n_nodes x n_nodes).

### Function `import_sonata_nodes(path)`
Parse a SONATA nodes HDF5 file.

Expected structure:
  /nodes/<population_name>/node_id
  /nodes/<population_name>/node_type_id
  /nodes/<population_name>/0/model_type  (optional)

### Function `import_sonata_edges(path)`
Parse a SONATA edges HDF5 file.

Expected structure:
  /edges/<population_name>/source_node_id
  /edges/<population_name>/target_node_id
  /edges/<population_name>/edge_type_id
  /edges/<population_name>/0/syn_weight  (optional)
  /edges/<population_name>/0/delay       (optional)

### Function `import_sonata(nodes_path, edges_path)`
Import a complete SONATA network from nodes + edges files.

Parameters
----------
nodes_path : path to nodes.h5
edges_path : path to edges.h5 (optional)

Returns
-------
SONATANetwork with parsed nodes, edges, and connectivity.

---

## Module `adapters.spikeinterface`

### Function `spike_trains_to_bitstreams(spike_times, duration_ms, dt)`
Convert spike times to binary bitstream matrix.

Parameters
----------
spike_times : dict mapping unit_id → array of spike times (ms)
duration_ms : float
    Total recording duration in ms.
dt : float
    Time bin width in ms.

Returns
-------
np.ndarray
    Shape (n_units, n_bins), dtype uint8, binary {0, 1}.

### Function `spike_trains_to_population_input(spike_times, duration_ms, dt)`
Convert spike times to current input array for Population.step_all().

Each spike becomes a current pulse of amplitude 1.0 at the spike time bin.

Parameters
----------
spike_times : dict mapping unit_id → array of spike times (ms)
duration_ms : float
dt : float

Returns
-------
np.ndarray
    Shape (n_timesteps, n_units), suitable for time-stepped simulation.

### Function `firing_rates_to_sc_probs(spike_times, duration_ms, max_rate_hz)`
Convert firing rates to SC probabilities in &#91;0, 1&#93;.

Parameters
----------
spike_times : dict mapping unit_id → array of spike times (ms)
duration_ms : float
max_rate_hz : float
    Rate corresponding to probability 1.0.

Returns
-------
np.ndarray
    Shape (n_units,), probabilities in &#91;0, 1&#93;.

### Function `from_sorting(sorting, dt)`
Convert a SpikeInterface SortingExtractor to bitstream matrix.

Parameters
----------
sorting : spikeinterface.core.BaseSorting
    SpikeInterface sorting result.
dt : float
    Time bin width in ms.

Returns
-------
np.ndarray
    Shape (n_units, n_bins), dtype uint8.

---

## Module `analog_bridge.analog_bridge`

### Class `AnalogSubstrateProfile`
Parameter set for analog/mixed-signal neuromorphic chips.

- **brainscales3**(cls)
- **dynapse2**(cls)

### Class `AEREvent`
Address-Event Representation spike event.


### Class `AnalogBridge`
- **__init__**(g_range, v_range, dac_res, profile)
- **_quantize**(val, v_min, v_max)
  - Returns (DAC_Value, Actual_Analog_Value) after quantization.
- **emit_analog_config**(nodes)

### Class `EventDrivenInterface`
Converts between SC bitstreams and AER event streams.

- **__init__**(clock_period_us)
- **bitstream_to_events**(neuron_id, bitstream)
  - Convert a boolean bitstream to a sequence of AER spike events.
- **events_to_current**(events, duration_us, tau_syn, weight)
  - Convert AER events to time-discretized synaptic current trace.
- **rate_code**(events, window_us)
  - Compute firing rate (Hz) from an event list.

### Class `CalibrationRoutine`
On-chip characterization loop for analog substrate alignment.

- **__init__**(bridge, num_steps)
- **sweep_conductance**()
  - Sweep DAC range and report (dac_value, target_g, actual_g) tuples.
- **max_quantization_error**()
  - Return worst-case quantization error across the conductance range.
- **effective_resolution_bits**()
  - Compute effective number of bits (ENOB) given quantization errors.

---

## Module `analysis.explainability`

### Class `SpikeToConceptMapper`
XAI Module: Maps spike patterns to semantic concepts.

- **__init__**(concept_map)
- **explain**(spikes)
  - Input: Spike vector (n_neurons,)

---

## Module `analysis.phi_estimation`

### Function `phi_star(data, tau)`
Geometric integrated information (Barrett & Seth 2011).

Phi* = MI(past; future) - max_partition sum MI(past_k; future_k)

Approximated via Gaussian assumption: MI = 0.5 * log(det(Sigma_X) * det(Sigma_Y) / det(Sigma_XY)).

Parameters
----------
data : np.ndarray
    Shape (n_channels, n_timesteps). Spike counts or continuous signals.
tau : int
    Time lag for past→future mapping.

Returns
-------
float
    Phi* estimate (bits). Non-negative; 0 = fully reducible.

### Function `_gaussian_mi(x, y)`
Mutual information under Gaussian assumption.

MI(X;Y) = 0.5 * log(det(Cov_X) * det(Cov_Y) / det(Cov_XY))

### Function `phi_from_spike_trains(spikes, bin_size, tau)`
Compute Phi* from binary spike trains.

Parameters
----------
spikes : np.ndarray
    Shape (n_neurons, n_timesteps), binary {0, 1}.
bin_size : int
    Number of timesteps per bin for spike count computation.
tau : int
    Time lag in bins.

Returns
-------
float
    Phi* in bits.

---

## Module `analysis.spike_stats.basic`

### Function `spike_times(binary_train, dt)`
Extract spike times (seconds) from a binary 0/1 array.

### Function `isi(binary_train, dt)`
Inter-spike intervals (seconds) from a binary train.

### Function `firing_rate(binary_train, dt)`
Mean firing rate (Hz).

### Function `spike_count(binary_train)`
Total spike count.

### Function `bin_spike_train(binary_train, bin_size)`
Bin a binary spike train into spike counts per bin.

---

## Module `analysis.spike_stats.causality`

### Function `_var_coefficients(trains_binned, order)`
Fit VAR(order) model. Returns (coefficients &#91;order*d x d&#93;, residual covariance).

### Function `pairwise_granger_causality(source, target, bin_size, order)`
Pairwise Granger causality. Granger 1969.

Tests if past source spike counts reduce prediction error for target.
Returns log-likelihood ratio. Positive = source Granger-causes target.

### Function `conditional_granger_causality(source, target, condition, bin_size, order)`
Conditional Granger causality. Geweke 1984.

Tests if source Granger-causes target controlling for condition.

### Function `spectral_granger_causality(trains, bin_size, order, n_freqs)`
Spectral Granger causality. Geweke 1982.

Returns (n_neurons x n_neurons x n_freqs) array of frequency-domain GC values.

### Function `partial_directed_coherence(trains, bin_size, order, n_freqs)`
Partial directed coherence (PDC). Baccala & Sameshima 2001.

Returns (n_neurons x n_neurons x n_freqs) normalized PDC values.

### Function `directed_transfer_function(trains, bin_size, order, n_freqs)`
Directed transfer function (DTF). Kaminski & Blinowska 1991.

Returns (n_neurons x n_neurons x n_freqs) normalized DTF values.

---

## Module `analysis.spike_stats.correlation`

### Function `cross_correlation(train_a, train_b, max_lag_ms, dt)`
Cross-correlogram between two binary spike trains.

Returns (correlation, lags_ms).

### Function `pairwise_correlation(trains, dt)`
Pairwise Pearson correlation matrix across neurons.

### Function `event_synchronization(train_a, train_b, dt, tau_ms)`
Quian Quiroga et al. 2002 -- event synchronization.

### Function `spike_train_coherence(train_a, train_b, dt)`
Magnitude-squared coherence between two binary spike trains.

Returns (coherence, freqs_hz).

### Function `spike_time_tiling_coefficient(train_a, train_b, dt_param, delta_ms)`
Spike Time Tiling Coefficient (STTC). Cutts & Eglen 2014.

Corrects for firing rate bias unlike simple coincidence measures.

### Function `covariance_matrix(trains, bin_size)`
Spike count covariance matrix across neurons. de la Rocha et al. 2007.

### Function `autocorrelation_time(binary_train, dt, max_lag_ms)`
Autocorrelation time (seconds). Integral of normalized autocorrelation until first zero crossing.

### Function `noise_correlation(trains, bin_size)`
Noise correlation (trial-to-trial variability correlation). Averbeck & Lee 2006.

Uses residuals after subtracting mean across neurons.

### Function `signal_correlation(trains, bin_size)`
Signal correlation (tuning similarity). Pearson correlation of mean responses.

### Function `spike_count_covariance(trains, window)`
Windowed spike count covariance. Kohn & Smith 2005.

### Function `joint_psth(train_a, train_b, bin_size)`
Joint PSTH (JPSTH) matrix. Aertsen et al. 1989.

Returns 2D histogram of binned spike counts (neuron_a x neuron_b).

### Function `coincidence_index(train_a, train_b, dt, delta_ms)`
Coincidence index (kappa). Joris et al. 2006.

Corrects raw coincidence count for expected coincidences from rate.

---

## Module `analysis.spike_stats.decoding`

### Function `population_vector_decode(trains, preferred_directions, window)`
Georgopoulos population vector decoding.

Each neuron i has a preferred direction (angle in radians).
Decoded direction per time bin = weighted sum of preferred directions.
Returns decoded angles per time bin.

### Function `bayesian_decode(spike_counts, tuning_rates, prior)`
Bayesian MAP decoder. Dayan & Abbott 2001.

spike_counts: (n_neurons,) observed counts.
tuning_rates: (n_stimuli, n_neurons) mean rates per stimulus.
prior: (n_stimuli,) prior probabilities. Uniform if None.
Returns: MAP stimulus index.

### Function `maximum_likelihood_decode(spike_counts, tuning_rates)`
Maximum likelihood stimulus decoder. Dayan & Abbott 2001.

Poisson likelihood: argmax_s prod_j (lambda_j^{n_j} * exp(-lambda_j) / n_j!).

### Function `linear_discriminant_decode(train_data, labels, test_point)`
Fisher linear discriminant decoder. Fisher 1936.

train_data: (n_samples, n_features). labels: (n_samples,). test_point: (n_features,).
Returns predicted class label.

### Function `naive_bayes_decode(train_data, labels, test_point)`
Gaussian naive Bayes decoder. Mitchell 1997.

Assumes feature independence. Returns predicted class label.

---

## Module `analysis.spike_stats.dimensionality`

### Function `spike_train_pca(trains, n_components, bin_size)`
PCA on binned spike count matrix (neurons x time_bins).

Returns (projected, explained_variance_ratio).

### Function `demixed_pca(trains_by_condition, n_components, bin_size)`
Demixed PCA. Kobak et al. 2016.

Separates condition-dependent and condition-independent variance.
trains_by_condition: {condition_id: &#91;list of binary trains per neuron&#93;}.
Returns (projected, explained_variance_ratio).

### Function `factor_analysis(trains, n_factors, bin_size, n_iter)`
Factor analysis via EM. Rubin & Thayer 1982.

Returns (loading_matrix &#91;n_neurons x n_factors&#93;, uniquenesses &#91;n_neurons&#93;).

---

## Module `analysis.spike_stats.distance`

### Function `van_rossum_distance(train_a, train_b, dt, tau_ms)`
Van Rossum 2001 -- exponential-kernel spike train distance.

### Function `victor_purpura_distance(times_a, times_b, cost_per_s)`
Victor-Purpura 1996 -- edit distance between spike time arrays.

### Function `isi_distance(train_a, train_b, dt)`
ISI-distance (Kreuz et al. 2007) -- ratio-based ISI comparison.

### Function `spike_distance(times_a, times_b, t_start, t_end)`
SPIKE-distance. Kreuz et al. 2013.

### Function `_local_isi(times, idx)`
Nearest-neighbour ISI at index idx.

### Function `spike_sync(times_a, times_b, t_start, t_end)`
SPIKE-synchronization. Kreuz et al. 2015.

### Function `spike_sync_profile(times_a, times_b, n_bins, t_start, t_end)`
Binned SPIKE-synchronization profile. Kreuz et al. 2015.

### Function `spike_profile(times_a, times_b, n_bins, t_start, t_end)`
Binned SPIKE-distance profile. Kreuz et al. 2013.

### Function `isi_profile(binary_train_a, binary_train_b, dt, n_bins)`
Binned ISI-distance profile. Kreuz et al. 2007.

### Function `adaptive_spike_distance(times_a, times_b, t_start, t_end, cost)`
Adaptive SPIKE-distance with cost parameter interpolating ISI and SPIKE. Kreuz et al. 2013.

cost=0: pure SPIKE-distance. cost=1: ISI-like weighting.

### Function `schreiber_similarity(train_a, train_b, dt, sigma_ms)`
Schreiber et al. 2003 -- spike train similarity via smoothed correlation.

Convolves each train with Gaussian kernel, returns Pearson correlation.

### Function `hunter_milton_similarity(times_a, times_b, dt_max)`
Hunter-Milton 2003 similarity.

### Function `earth_movers_distance(times_a, times_b, t_start, t_end, n_bins)`
Earth mover's distance between spike time distributions. Rubner et al. 1998.

### Function `multi_neuron_victor_purpura(spike_times_list, cost_per_s)`
All-pairs Victor-Purpura distance matrix.

### Function `generalized_victor_purpura(times_a, times_b, cost_func)`
Generalized Victor-Purpura with arbitrary cost function. Victor & Purpura 1997.

cost_func(dt) returns the cost of shifting a spike by dt seconds.
Default: linear cost q*|dt| with q=1000.

### Function `spike_distance_matrix(spike_times_list, metric, t_start, t_end)`
All-pairs spike train distance matrix.

metric: 'spike_distance', 'spike_sync', 'victor_purpura'.

---

## Module `analysis.spike_stats.gpfa`

### Function `_gp_kernel(n_bins, tau, sigma)`
Squared-exponential kernel matrix for *n_bins* time points.

### Function `_gpfa_e_step(Y, C, d, R, K_all)`
Posterior p(x|y) for each latent dimension jointly.

### Function `_gpfa_m_step(Y, x_post, xx_post)`
Update C, d, R from sufficient statistics.

### Function `_gpfa_log_likelihood(Y, C, d, R, K_all)`
Exact marginal Gaussian log likelihood for the GPFA observation model.

### Function `gpfa(trains, n_latents, bin_ms, dt, max_iter, tol, seed)`
Extract smooth latent trajectories from parallel spike trains via EM.

### Function `gpfa_transform(new_trains, params, bin_ms, dt)`
Project new spike trains using learned GPFA parameters.

---

## Module `analysis.spike_stats.information`

### Function `mutual_information(train_a, train_b, bin_size)`
Mutual information between two binned spike trains (bits).

MI = H(A) + H(B) - H(A,B) using binned spike counts.

### Function `transfer_entropy(source, target, bin_size, lag)`
Transfer entropy from source to target spike train (bits).

TE = H(target_future | target_past) - H(target_future | target_past, source_past)

### Function `spike_train_entropy(binary_train, bin_size, word_length)`
Spike train entropy via binary word analysis. Strong et al. 1998.

### Function `noise_entropy(binary_train, n_trials, bin_size, word_length)`
Noise entropy estimate via splitting train into pseudo-trials. de Ruyter van Steveninck et al. 1997.

Splits the train into n_trials segments, computes entropy per segment, averages.

### Function `stimulus_specific_information(spike_counts, stimulus_ids)`
Stimulus-specific information (SSI). Butts 2003.

spike_counts: array of spike counts per trial.
stimulus_ids: corresponding stimulus labels.
Returns SSI in bits.

### Function `kozachenko_leonenko_mi(x, y, k)`
Kozachenko-Leonenko k-NN mutual information estimator. Kraskov et al. 2004.

### Function `time_rescaling_ks_test(times, rate_func, t_start, t_end)`
Time-rescaling KS test for point process goodness-of-fit. Brown et al. 2002.

rate_func(t) -> float: conditional intensity function.
Returns (ks_statistic, passes_at_95pct).

---

## Module `analysis.spike_stats.lfp`

### Function `phase_locking_value(binary_train, lfp_signal)`
Phase locking value (PLV) between spikes and LFP phase.

Extracts instantaneous phase of LFP via Hilbert transform,
then computes PLV = |mean(exp(j*phase_at_spikes))|.

### Function `spike_field_coherence(binary_train, lfp_signal, dt)`
Spike-field coherence (SFC) between binary train and LFP.

Returns (coherence, freqs_hz). SFC = |S_xy|^2 / (S_xx * S_yy).

### Function `spike_phase_histogram(binary_train, lfp_signal, n_bins)`
Histogram of LFP phase at spike times.

Returns (counts, bin_centers_rad) with bins spanning &#91;-pi, pi&#93;.

---

## Module `analysis.spike_stats.network`

### Function `functional_connectivity(trains, max_lag_ms, dt)`
Infer functional connectivity matrix from peak cross-correlation.

Returns NxN matrix where entry (i,j) is max |cross-correlation|
between neuron i and neuron j within +/-max_lag.

### Function `unitary_events(trains, bin_size, alpha)`
Unitary event analysis. Gruen et al. 2002.

Detects bins where coincident spikes exceed chance (Poisson assumption).
Returns list of significant bin indices.

### Function `cell_assembly_detection(trains, bin_size, threshold)`
Cell assembly detection via PCA on binned spike matrix. Lopes-dos-Santos et al. 2013.

Returns list of assemblies (each a list of neuron indices above threshold).

### Function `synfire_chain_detection(trains, dt, max_delay_ms, min_chain_length)`
Synfire chain detection via cross-correlation peak ordering. Abeles 1991.

Returns list of chains (ordered neuron indices with sequential activation).

---

## Module `analysis.spike_stats.neural_decoders`

### Class `POYODecoder`
Population decoder via spike tokenisation and cross-attention.

Azabou et al. "A Unified, Scalable Framework for Neural Population
Decoding." NeurIPS 2023. arXiv:2310.16046.

Architecture: individual spikes are tokenised with learned unit
embeddings + sinusoidal temporal encoding. Learnable latent queries
attend to the spike tokens via cross-attention (PerceiverIO backbone).
Output queries decode from the latent representation.

- **__post_init__**()
- **_unit_embedding**(unit_id)
- **encode**(spike_trains, dt)
  - Encode population activity to latent representation.
- **decode**(latents, output_queries)
  - Cross-attention decode from latents.
- **reset**()
  - Clear cached unit embeddings.

### Class `POSSMDecoder`
Population decoder via spike tokenisation and diagonal state-space model.

Ryoo et al. "Generalizable, Real-Time Neural Decoding with Hybrid
State-Space Models." ICLR 2025. arXiv:2506.05320.

Uses POYO spike tokenisation with a recurrent SSM backbone instead
of full attention, enabling causal online prediction at millisecond
resolution with up to 9x lower inference cost.

Diagonal SSM recurrence (S4D, Gu et al. 2022):
    h_t = A_bar h_{t-1} + B_bar x_t
    y_t = Re(C h_t) + D x_t

Discretisation (zero-order hold):
    A_bar = exp(dt * A)
    B_bar = (A_bar - I) * diag(A)^{-1} * B

- **__post_init__**()
- **discretise**(step_dt)
  - Zero-order hold discretisation.
- **step**(x)
  - Single causal SSM step.
- **encode_causal**(spike_trains, dt)
  - Causal online encoding of spike trains.
- **reset**()
  - Reset hidden state to zero.

### Class `NDT3Decoder`
Autoregressive neural data transformer for motor decoding.

Ye & Pandarinath. "A Generalist Intracortical Motor Decoder."
bioRxiv 2025.02.02.634313.

Bins population spike counts, projects to d_model embeddings,
adds positional encoding, and predicts next-bin output via
causal (masked) self-attention.

- **__post_init__**()
- **bin_and_embed**(spike_trains, dt)
  - Bin spike trains and project to embeddings.
- **predict_next**(embedded)
  - Causal autoregressive prediction of next time bin.
- **decode**(spike_trains, dt)
  - Full decode pipeline: bin → embed → causal attention → output.

### Class `CEBRAEncoder`
Contrastive embedding encoder for neural data.

Schneider, Lee & Mathis. "Learnable latent embeddings for joint
behavioural and neural analysis." Nature 604 (2023).
arXiv:2204.00673.

Uses InfoNCE contrastive loss with time-based or behaviour-based
positive pair sampling to learn low-dimensional embeddings of
neural population activity.

InfoNCE loss (van den Oord et al. 2018):
    L = -log( exp(sim(z_i, z_j^+) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
where sim(a, b) = a · b / (||a|| ||b||)   (cosine similarity)

- **__post_init__**()
- **encode**(x)
  - Encode neural data through 2-layer MLP.
- **cosine_similarity**(a, b)
  - Pairwise cosine similarity matrix.
- **infonce_loss**(anchors, positives)
  - InfoNCE contrastive loss. van den Oord et al. (2018).
- **_forward_and_loss**(anchors, positives)
  - Forward pass with cached intermediates for backprop.
- **_backward**(cache)
  - Analytical backprop through InfoNCE + MLP.
- **fit**(data, n_steps, time_offset)
  - Train encoder with time-contrastive learning.
- **transform**(data)
  - Embed neural data into learned latent space.

### Function `tokenise_spikes(spike_trains, dt)`
Convert binary spike trains to sorted (unit_id, timestamp) tokens.

Used by POYO+ and POSSM (Azabou et al. 2023; Ryoo et al. 2025).

Parameters
----------
spike_trains : list of 1-D binary arrays (one per neuron).
dt : timestep in ms.

Returns
-------
unit_ids : int64 array &#91;n_tokens&#93;.
timestamps : float64 array &#91;n_tokens&#93; in ms.

### Function `sinusoidal_position_encode(timestamps, d_model)`
Sinusoidal position encoding. Vaswani et al. (2017).

PE(t, 2i)   = sin(t / 10000^{2i/d})
PE(t, 2i+1) = cos(t / 10000^{2i/d})

### Function `scaled_dot_product_attention(queries, keys, values)`
Scaled dot-product attention.

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

---

## Module `analysis.spike_stats.patterns`

### Function `spike_directionality(times_a, times_b, t_start, t_end)`
Spike directionality. Kreuz et al. 2015.

Returns asymmetry measure in &#91;-1, 1&#93;. Positive: A leads B.

### Function `spike_train_order(times_list, t_start, t_end)`
Spike train order matrix. Kreuz et al. 2017.

Returns (n x n) matrix of pairwise directionality values.

### Function `cubic_higher_order(binary_train, dt, max_lag)`
Third-order cumulant (bispectrum domain). Nikias & Petropulu 1993.

Returns 2D array C3(tau1, tau2) for lag pairs up to max_lag.

---

## Module `analysis.spike_stats.point_process`

### Function `conditional_intensity(binary_train, dt, window_ms)`
Conditional intensity function estimate (Hz). Brown et al. 2004.

Moving-window MLE of the Poisson rate at each time step.

### Function `isi_hazard_function(binary_train, dt, bins)`
ISI hazard function h(t) = f(t) / S(t). Tuckwell 1988.

Returns (hazard, bin_centers) where hazard is the failure rate at each ISI duration.

### Function `isi_survivor_function(binary_train, dt, bins)`
ISI survivor function S(t) = P(ISI > t). Tuckwell 1988.

Returns (survivor, bin_centers).

### Function `renewal_density(binary_train, dt, bins)`
Renewal density h(t) from ISI distribution. Cox 1962.

Returns (density, bin_centers). Density normalized by mean rate.

---

## Module `analysis.spike_stats.rate`

### Function `instantaneous_rate(binary_train, dt, kernel, sigma_ms)`
Instantaneous firing rate via kernel convolution (Hz).

Kernels: 'gaussian', 'exponential', 'rectangular'.

### Function `population_rate(trains, dt, sigma_ms)`
Population-level instantaneous firing rate (Hz).

Sums all trains then applies Gaussian kernel smoothing.

### Function `psth(trials, bin_ms, dt)`
Peri-stimulus time histogram across trials.

Returns (rates_hz, bin_centers_ms).

---

## Module `analysis.spike_stats.sorting_quality`

### Function `isolation_distance(cluster, noise)`
Isolation distance. Harris et al. 2001.

Mahalanobis distance at which the number of noise points equals cluster size.
cluster: (n_cluster, n_features). noise: (n_noise, n_features).

### Function `l_ratio(cluster, noise)`
L-ratio. Schmitzer-Torbert et al. 2005.

Sum of inverse-chi2 CDF values for noise Mahalanobis distances, normalized by cluster size.
Approximated here as sum(1/mah_dist) / n_cluster.

### Function `silhouette_score(features, labels)`
Mean silhouette score. Rousseeuw 1987.

Measures cluster separation: s_i = (b_i - a_i) / max(a_i, b_i).

### Function `d_prime(cluster_a, cluster_b)`
d-prime (sensitivity index) between two clusters. Green & Swets 1966.

Uses first principal axis for projection.

### Function `isi_violation_rate(binary_train, dt, refractory_ms)`
ISI violation rate: fraction of ISIs below refractory period. Hill et al. 2011.

### Function `presence_ratio(binary_train, n_bins)`
Presence ratio: fraction of time bins containing at least one spike. IBL 2019.

### Function `amplitude_cutoff(amplitudes, bins)`
Amplitude cutoff estimate. Hill et al. 2011.

Fraction of spikes estimated to be missing below the amplitude histogram peak.

### Function `snr(waveforms)`
Signal-to-noise ratio of spike waveforms. Suner et al. 2005.

waveforms: (n_spikes, n_samples). SNR = peak_amplitude / noise_std.

### Function `nn_hit_rate(cluster, noise, k)`
Nearest-neighbor hit rate. Chung et al. 2017.

Fraction of cluster points whose k nearest neighbors are also in the cluster.

### Function `drift_metric(waveforms, timestamps, n_bins)`
Waveform drift metric. IBL 2019.

Measures change in mean waveform amplitude over time.

---

## Module `analysis.spike_stats.spade`

### Function `_find_frequent_itemsets(binary_matrix, min_support, max_size)`
Apriori-style frequent itemset mining on a binary neuron x time matrix.

### Function `_extend_to_spatiotemporal(trains, itemsets, bin_ms, dt, max_lag_bins)`
Extend synchronous itemsets to spatiotemporal patterns with lags.

### Function `spade_detect(trains, bin_ms, dt, min_support, max_pattern_size, n_surrogates, alpha, seed)`
Detect repeated spatiotemporal spike patterns with significance testing.

---

## Module `analysis.spike_stats.spectral`

### Function `power_spectrum(binary_train, dt)`
Power spectral density of a binary spike train.

Returns (psd, freqs_hz).

---

## Module `analysis.spike_stats.statistics`

### Function `significance_bootstrap(statistic_func, train_a, train_b, n_surrogates, seed)`
Bootstrap significance test for a pairwise statistic.

Returns (observed_value, p_value).
statistic_func(a, b) -> float.

---

## Module `analysis.spike_stats.stimulus`

### Function `spike_triggered_average(stimulus, binary_train, window_steps)`
Spike-triggered average (STA) of a stimulus signal.

Returns the average stimulus snippet preceding each spike.

### Function `spike_triggered_covariance(stimulus, binary_train, window_steps)`
Spike-triggered covariance (STC). Schwartz et al. 2006.

Returns covariance matrix of stimulus snippets preceding spikes.

### Function `spatial_information(binary_train, positions, n_bins, dt)`
Spatial information (bits/spike). Skaggs et al. 1993.

positions: 1D array of position values (same length as binary_train).
SI = sum(p_i * r_i/r_mean * log2(r_i/r_mean)).

### Function `place_field_detection(binary_train, positions, n_bins, threshold_std, dt)`
Detect place fields as contiguous bins with rate > mean + threshold_std * std. O'Keefe & Dostrovsky 1971.

Returns list of (field_start, field_end) position values.

### Function `tuning_curve(binary_train, stimulus_values, n_bins, dt)`
Compute tuning curve: mean firing rate vs stimulus value. Dayan & Abbott 2001.

Returns (mean_rates, bin_centers).

---

## Module `analysis.spike_stats.surrogates`

### Function `surrogate_isi_shuffle(binary_train, seed)`
Generate surrogate by shuffling ISIs. Preserves rate + ISI distribution.

### Function `surrogate_dither(binary_train, dither_ms, dt, seed)`
Generate surrogate by jittering each spike time +/-dither_ms.

### Function `surrogate_trial_shuffle(trains, seed)`
Shuffle trial order. Destroys trial-to-trial correlation.

### Function `homogeneous_poisson(rate_hz, duration_s, dt, seed)`
Generate homogeneous Poisson spike train. Heeger 2000.

### Function `inhomogeneous_poisson(rate_func, duration_s, dt, seed)`
Generate inhomogeneous Poisson spike train via thinning. Lewis & Shedler 1979.

rate_func(t) -> float: time-varying rate in Hz.

### Function `gamma_process(rate_hz, shape, duration_s, dt, seed)`
Generate gamma-renewal spike train. Kuffler et al. 1957.

shape=1: Poisson. shape>1: more regular. shape<1: more bursty.

### Function `compound_poisson_process(rate_hz, burst_mean, duration_s, dt, seed)`
Compound Poisson process: Poisson events each producing a burst. Snyder & Miller 1991.

burst_mean: mean number of spikes per event (Poisson distributed).

### Function `surrogate_joint_isi(binary_train, seed)`
Joint-ISI surrogate: preserves ISI distribution and serial ISI correlations. Louis et al. 2010.

Shuffles pairs of consecutive ISIs while preserving their joint statistics.

### Function `surrogate_bin_shuffling(binary_train, bin_size, seed)`
Bin-shuffling surrogate: shuffles spikes within bins. Hatsopoulos et al. 2003.

### Function `surrogate_spike_train_shifting(binary_train, max_shift, seed)`
Circular shifting surrogate: shifts entire train by random offset. Hatsopoulos et al. 2003.

---

## Module `analysis.spike_stats.temporal`

### Function `burst_detection(binary_train, dt, max_isi_ms, min_spikes)`
Detect bursts: consecutive spikes with ISI < max_isi_ms.

Returns list of (start_time_s, end_time_s, spike_count).

### Function `first_spike_latency(binary_train, dt)`
Time to first spike (seconds). Returns nan if no spike.

### Function `response_onset(binary_train, baseline_steps, dt, threshold_sigma)`
Detect response onset as first bin exceeding baseline + threshold_sigma * std.

Returns onset time (seconds), or nan if no response detected.

### Function `change_point_detection(binary_train, bin_size, threshold)`
CUSUM-based change point detection in firing rate. Page 1954.

Returns list of bin indices where significant rate changes occur.

---

## Module `analysis.spike_stats.variability`

### Function `cv_isi(binary_train, dt)`
Coefficient of variation of ISI. CV=1 for Poisson, <1 for regular.

### Function `cv2(binary_train, dt)`
Local coefficient of variation CV2. Holt et al. 1996.

CV2 = mean(2|ISI_{i+1} - ISI_i| / (ISI_{i+1} + ISI_i)).
Less sensitive to firing rate changes than global CV.

### Function `local_variation(binary_train, dt)`
Local variation LV. Shinomoto et al. 2003.

LV = (3/(N-1)) * sum((ISI_i - ISI_{i+1})^2 / (ISI_i + ISI_{i+1})^2).
LV=1 for Poisson, <1 for regular, >1 for bursty.

### Function `lvr(binary_train, dt, refractoriness_ms)`
Revised local variation LvR. Shinomoto et al. 2009.

Corrects LV for refractoriness: LvR = mean(3(1 - 4*ISI_i*ISI_{i+1}/(ISI_i+ISI_{i+1})^2)(1 + 4*R/(ISI_i+ISI_{i+1}))).

### Function `fano_factor(binary_train, window_ms, dt)`
Fano factor: variance/mean of spike counts in sliding windows.

### Function `isi_entropy(binary_train, dt, bins)`
Shannon entropy of the ISI distribution (bits).

Higher entropy = more irregular. Poisson has maximum entropy
for a given rate.

### Function `lempel_ziv_complexity(binary_train)`
Lempel-Ziv 1976 complexity. Normalized by N/log2(N).

### Function `approximate_entropy(binary_train, m, r_factor)`
Approximate entropy (ApEn). Pincus 1991.

### Function `sample_entropy(binary_train, m, r_factor)`
Sample entropy (SampEn). Richman & Moorman 2000.

### Function `permutation_entropy(binary_train, order, delay)`
Bandt-Pompe permutation entropy. Bandt & Pompe 2002.

### Function `hurst_exponent(binary_train, min_window)`
Hurst exponent via detrended fluctuation analysis (DFA). Peng et al. 1994.

H > 0.5: long-range positive correlation. H < 0.5: anti-correlated.

### Function `allan_factor(binary_train, dt, n_scales)`
Allan factor for spike trains. Allan 1966, adapted for point processes.

Returns (af_values, window_sizes_s). AF > 1 indicates fractal clustering.

### Function `rescaled_range(binary_train, min_window)`
Hurst exponent via rescaled range (R/S) analysis. Hurst 1951.

Classic alternative to DFA. Returns H from log-log fit of R/S vs scale.

### Function `complexity_pdf(binary_train, dt, bins)`
ISI probability density function via histogram. Abeles 1982.

### Function `optimal_bin_width(binary_train, dt)`
Shimazaki-Shinomoto 2007 optimal histogram bin width for firing rate.

Minimizes MISE cost C(delta) = (2*mean - var) / (N * delta)^2 over candidate deltas.
Returns optimal bin width in seconds.

### Function `optimal_kernel_bandwidth(binary_train, dt)`
Silverman's rule-of-thumb bandwidth for ISI kernel density. Silverman 1986.

h = 0.9 * min(std, IQR/1.34) * N^{-1/5}.

---

## Module `analysis.spike_stats.waveform`

### Function `waveform_width(waveform, dt)`
Trough-to-peak width (seconds). Bartho et al. 2004.

Measures time from waveform minimum to subsequent maximum.

### Function `waveform_amplitude(waveform)`
Peak-to-trough amplitude. Bartho et al. 2004.

### Function `waveform_repolarization_slope(waveform, dt)`
Repolarization slope: max dV/dt after trough. Bean 2007.

### Function `waveform_recovery_slope(waveform, dt)`
Recovery slope: dV/dt during return to baseline after peak. Bean 2007.

### Function `waveform_halfwidth(waveform, dt)`
Half-width: duration at half-minimum amplitude. Bartho et al. 2004.

### Function `waveform_pt_ratio(waveform)`
Peak-to-trough ratio. Bartho et al. 2004.

Ratio of post-trough peak amplitude to trough amplitude.

---

## Module `arcane_zenith`

### Class `ArcaneZenithCognitiveCore`
A self-improving cognitive primitive combining ArcaneNeuron and Zenith plasticity.

Rather than maintaining static deep-context parameters, the ArcaneZenith module
deploys 4 synchronized Zenith meta-plasticity connections controlling physical limits.
Zenith plasticity weights ∈ &#91;0, 1&#93; are smoothly mapped to safe biological ranges
for each parameter using a sigmoid interpolator.

Example:
    >>> core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
    >>> for i in range(10):
    ...     spike = core.step(current=i % 50)
    >>> print(f"drift={core.neuron.identity_drift:.4f}")

- **__init__**(backend)
- **_map_to_range**(w, min_val, max_val)
  - Smooth sigmoid mapping centered at 0.5 to prevent edge explosions.
- **step**(current)
  - Step the unified physical simulation one tick forward.
- **step_from_bio_rates**(rates)
  - Modulate phenomenological bounds leveraging a multi-channel biological firing rate map.
- **step_from_genome**(genome)
  - Modulate phenomenological bounds leveraging a generated Evo Substrate Genome.
- **reset**()
- **get_state**()
  - Output serialized limits combining Arcane and Zenith structures natively.
- **get_state_dict**()
- **load_state_dict**(state_dict)

### Function `create_arcane_neuron_with_zenith_plasticity(backend)`
Seamless factory configuring a unified ArcaneZenith primitive running entirely connected.

---

## Module `asic_flow.asic_flow`

### Class `PDKType`

### Class `PDKConfig`
Process Design Kit configuration.

- **from_pdk_type**(cls, pdk)
- **is_open_source**()
- **with_pdk_root**(pdk_root)
  - Return a copy with ``$PDK_ROOT`` variables bound to ``pdk_root``.

### Class `ResolvedPDKFiles`
Resolved file paths required by the open-source ASIC flow.

- **required_paths**()
- **optional_paths**()

### Class `PDKResolution`
Outcome of resolving a PDK against the local filesystem.

- **usable_for_synthesis**()
- **usable_for_signoff**()

### Class `OpenSourcePDKResolver`
Resolve Sky130/GF180 file locations without requiring OpenLane at import time.

- **resolve**(pdk, pdk_root, require_existing)
  - Bind ``$PDK_ROOT`` and report missing PDK artefacts.
- **_file_manifest**(pdk)
- **_pdk_root_from_path**(path, marker)

### Class `SCASICOptimisationConfig`
SC-specific synthesis settings for stochastic neuromorphic datapaths.

- **yosys_passes**()

### Class `DesignParams`
ASIC design parameters.

- **clock_period_ns**()
- **die_width_um**()
- **die_height_um**()
- **core_area_mm2**()

### Class `SynthesisGenerator`
Generates Yosys synthesis TCL scripts.

- **generate**(pdk, design)

### Class `FloorplanGenerator`
Generates OpenROAD floorplan TCL scripts.

- **generate**(pdk, design)

### Class `PlaceRouteGenerator`
Generates OpenROAD place-and-route TCL scripts.

- **generate**(pdk, design)

### Class `SDCGenerator`
Generates Synopsys Design Constraints (SDC) for STA.

- **generate**(pdk, design)

### Class `SignoffCheckResult`
Result of one signoff check.


### Class `SignoffGenerator`
Generates signoff scripts and evaluates results.

- **generate_sta_script**(pdk, design)
  - Generate OpenSTA timing analysis script.
- **generate_drc_script**(pdk, design)
  - Generate DRC check script (KLayout-based for open PDKs).
- **generate_lvs_script**(pdk, design)
  - Generate LVS check script.
- **evaluate_timing**(wns, tns, clock_period_ns)
  - Evaluate timing signoff from worst/total negative slack.
- **evaluate_power**(dynamic_mw, leakage_mw, budget_mw)
- **evaluate_area**(cell_count, used_area_um2, die_area_um2)

### Class `GDSIIExporter`
Generates GDSII stream-out scripts.

- **generate**(pdk, design)

### Class `ASICFlowOutput`
Complete output of the ASIC tape-out flow.

- **to_dict**()

### Class `ASICFlowGenerator`
Top-level generator for the complete ASIC tape-out pipeline.

- **generate**(pdk, design)
- **_generate_makefile**(design)

### Class `ASICFlowBundle`
Generated ASIC flow files plus the evidence manifest path.

- **to_dict**()

### Class `DesignEstimate`
Pre-synthesis area/power/timing estimate for an SC module.


### Class `PreSynthEstimator`
Estimates area, power, and timing before synthesis.

Uses empirical models based on SC circuit characteristics:
- Bitstream ops: ~10 gates/bit
- LIF neuron: ~500 gates
- STDP synapse: ~200 gates
- AER router: ~100 gates/port

- **estimate**(cls, n_neurons, n_synapses, bitstream_width, n_aer_ports, pdk)
  - Estimate design metrics from architectural parameters.

### Class `CornerType`

### Class `PVTCorner`
Process-Voltage-Temperature corner definition.

- **label**()

### Class `MultiCornerAnalysis`
Generates multi-corner STA scripts for all PVT corners.

- **generate**(pdk, design, corners)
- **worst_slack**(per_corner_wns)

### Class `CDCCheckGenerator`
Generates clock-domain crossing lint scripts.

- **generate**(design, clock_domains)

### Class `IRDropGenerator`
Generates IR drop analysis scripts for OpenROAD.

- **generate**(pdk, design, toggle_rate)

### Class `IOPin`
Specification for one IO pad.


### Class `IOConstraintGenerator`
Generates IO placement constraint files.

- **generate**(pins, design)
- **auto_assign**(signal_names, sides)
  - Auto-assign pins to die edges round-robin.

### Class `LECGenerator`
Generates Logic Equivalence Checking scripts.

- **generate**(design)

### Class `OCVConfig`
On-Chip Variation derating factors.

- **generate_sdc_fragment**()
- **conservative**(cls)

### Class `DRCViolation`
One DRC rule violation.


### Class `SignoffSummary`
Structured signoff summary with pass/fail per check.

- **drc_clean**()
- **all_pass**()
- **to_dict**()

### Class `PDKValidationResult`
Result of PDK sanity check.


### Class `BlockConfig`
One block in a hierarchical ASIC flow.


### Class `HierarchicalFlow`
Multi-block ASIC flow with per-block synthesis + top integration.

- **add_block**(block)
- **block_names**()
- **generate_block_scripts**(pdk)
- **generate_top_integration**(pdk)

### Class `TapeOutChecklist`
Go/no-go checklist for ASIC tape-out.

- **readiness_score**()
- **is_tape_out_ready**()
- **failing_checks**()
- **from_signoff**(summary)
  - Populate from a signoff summary.

### Function `generate_asic_flow_bundle(output_dir)`
Write a complete ASIC flow deck and evidence manifest in one call.

The helper deliberately does not run Yosys/OpenROAD. It materialises the
scripts, resolves the requested PDK paths, records missing artefacts, and
adds a pre-synthesis estimate so Python API users can inspect the bundle
before launching external EDA tools.

### Function `_normalise_pdk_type(pdk_type)`
### Function `_pdk_to_manifest(pdk)`
### Function `_design_to_manifest(design)`
### Function `_build_asic_flow_manifest()`
### Function `validate_pdk(pdk)`
Check PDK configuration for obvious errors.

### Function `validate_pdk_installation(pdk, pdk_root, require_signoff)`
Check whether the resolved open-source PDK files are present locally.

---

## Module `audio.adaptive_engine`

### Class `SessionPhase`

### Class `_AdaptationRecord`
Single parameter adaptation event.


### Class `AdaptiveSessionReport`
Summary of a completed adaptive audio session.

- **to_dict**()

### Class `AdaptiveAudioEngine`
Closed-loop adaptive audio controller coupling SSGF with EVS.

Parameters
----------
ssgf : SSGFEngine
    The geometry solver producing audio mappings.
evs : EVSEngine
    The entrainment verification scorer.
profile : UserProfile, optional
    User preferences for chronotype-aware adaptation.

- **__init__**(ssgf, evs, profile)
- **_update_phase**()
  - Transition between session phases based on tick count.
- **_evs_trend**()
  - Return recent EVS trend: positive = improving, negative = declining.
- **on_evs_update**(snapshot)
  - Process one EVS update and return adapted audio parameters.
- **_adapt_discovery**(snap, trend)
  - Discovery phase: gentle frequency sweep, widen geometry.
- **_adapt_lock_on**(snap, trend)
  - Lock-On phase: responsive frequency tracking, tighten geometry.
- **_adapt_deepening**(snap, trend)
  - Deepening phase: push toward theurgic coherence.
- **_log_adaptation**(param, old, new, reason)
- **get_session_report**()
  - Generate summary report of the current session.
- **current_phase**()
- **tick**()
- **reset**()
  - Reset session state (does not reset SSGF or EVS).

### Function `_compute_grade(verified_pct)`
Map verified percentage to letter grade.

---

## Module `audio.evs_engine`

### Class `EVSConfig`
Tuneable parameters for the EVS engine.


### Class `EVSSnapshot`
Single-tick EVS observation.

- **to_dict**()

### Class `EVSEngine`
FFT-based Entrainment Verification Score engine.

Workflow
--------
1. ``start_baseline()`` -- begin collecting baseline EEG
2. ``add_sample(voltage)`` -- feed raw EEG samples one at a time
3. After baseline_duration_s, baseline finalises automatically
4. ``set_target(hz)`` -- set the entrainment target frequency
5. ``compute()`` returns ``EVSSnapshot`` every *update_interval_samples*

- **__init__**(cfg)
- **start_baseline**()
  - Begin baseline EEG collection.
- **_finalise_baseline**()
  - Compute baseline band powers from collected samples.
- **add_sample**(voltage)
  - Feed one raw EEG voltage sample.
- **set_target**(hz)
  - Set the entrainment target frequency.
- **_ordered_buf**()
  - Return the ring buffer in time-order.
- **_band_powers**(signal)
  - Compute power in each canonical EEG band via FFT.
- **_peak_frequency**(signal)
  - Dominant frequency in the signal.
- **compute**()
  - Compute current EVS snapshot.
- **baseline_done**()
- **score_history**()
- **reset**()
  - Full reset.

### Function `_hz_to_band(hz)`
Map a frequency in Hz to its canonical EEG band name.

---

## Module `audio.ssgf_engine`

### Class `SSGFConfig`
All tuneable knobs for SSGFEngine.


### Class `SSGFEngine`
Lightweight SSGF geometry-coupled Kuramoto solver.

Maintains a latent vector *z* whose decoded geometry matrix W(t)
feeds back into the micro-cycle, steering oscillators toward
higher global coherence R.  Audio-mapping observables are derived
from the resulting phase dynamics and spectral properties of W.

- **__init__**(cfg)
- **_decode**(z)
  - Decode latent vector into a symmetric, non-negative weight
- **_micro_step**()
  - One Kuramoto + geometry-feedback timestep (vectorised).
- **_spectral**()
  - Compute eigendecomposition of the normalised Laplacian of W.
- **_compute_R**()
  - Kuramoto order parameter R = |<exp(i*theta)>|.
- **_cost**()
  - Composite cost: minimise negative coherence + regularise W.
- **outer_step**()
  - One outer-cycle step: micro-cycle -> spectral -> grad update on z.
- **get_audio_mapping**()
  - Derive CCW audio parameters from current SSGF state.
- **get_state**()
  - Full engine state snapshot.

---

## Module `audio.user_profile`

### Class `Chronotype`
Sleep chronotype model (after Dr. Michael Breus).

Each chronotype has a preferred entrainment frequency range and
optimal session timing.


### Class `UserProfile`
Per-user preference and adaptation model.

Parameters
----------
user_id : str
    Unique user identifier.
chronotype : Chronotype
    Sleep chronotype.
baseline_band_powers : dict
    Resting-state EEG band powers (populated after first baseline).
preferred_cost_weights : dict
    SSGF cost weights tuned to this user.
sensitivity_map : dict
    Per-band sensitivity multipliers (e.g. {"alpha": 1.2}).
session_count : int
    Total completed sessions.
preferred_target_hz : float, optional
    Explicitly set target frequency (overrides chronotype default).

- **__post_init__**()
- **get_best_target_hz**()
  - Return the best entrainment target for this user.
- **update_from_session**(avg_evs, peak_evs, best_target_hz, band_powers)
  - Update profile after a completed session.
- **to_dict**()
- **from_dict**(cls, data)

---

## Module `augmentation.curriculum`

### Class `SpikeCurriculum`
Schedule training difficulty across epochs.

Parameters
----------
total_epochs : int
    Total training epochs.
start_timesteps : int
    Initial sequence length.
end_timesteps : int
    Final sequence length.
start_rate_scale : float
    Initial firing rate multiplier (>1 = amplified = easier).
end_rate_scale : float
    Final firing rate multiplier (1.0 = natural).
start_noise : float
    Initial background noise rate.
end_noise : float
    Final background noise rate.
warmup_fraction : float
    Fraction of epochs for linear warmup (0.0-1.0).

- **_progress**(epoch)
  - Compute curriculum progress in &#91;0, 1&#93;.
- **timesteps**(epoch)
  - Sequence length for this epoch.
- **rate_scale**(epoch)
  - Firing rate multiplier for this epoch.
- **noise_rate**(epoch)
  - Background noise rate for this epoch.
- **apply_to_spikes**(spikes, epoch, seed)
  - Apply curriculum-scheduled transforms to a spike tensor.
- **schedule_summary**()
  - Print the curriculum schedule.

---

## Module `augmentation.spike_augment`

### Class `SpikeAugment`
Composable spike-domain augmentation.

Parameters
----------
jitter_steps : int
    Max temporal jitter in timesteps (spikes shift +/- jitter).
dropout_rate : float
    Probability of dropping each spike (0.0 = none, 1.0 = all).
rate_scale : tuple of float
    (min_scale, max_scale) for random firing rate scaling.
polarity_flip_prob : float
    Probability of flipping spike polarity (for DVS ON/OFF channels).
bg_noise_rate : float
    Background noise spike probability per neuron per step.
hot_pixel_prob : float
    Probability of a neuron becoming a hot pixel (fires every step).
seed : int
    Random seed for reproducibility.

- **__call__**(spikes)
  - Apply all augmentations to a spike tensor.
- **_temporal_jitter**(spikes, rng)
- **_spike_dropout**(spikes, rng)
- **_rate_scaling**(spikes, rng)
- **_polarity_flip**(spikes, rng)
- **_background_noise**(spikes, rng)
- **_hot_pixel**(spikes, rng)

---

## Module `autofit.features`

### Function `extract_spike_times(voltage, threshold, dt)`
Find spike times from a voltage trace via threshold crossing.

### Function `extract_features(voltage, dt, threshold)`
Extract standard electrophysiology features from a voltage trace.

Returns dict with:
    spike_times, spike_count, mean_isi, cv_isi, firing_rate,
    v_rest, v_max, v_min, ap_height, ap_width

---

## Module `autofit.fitter`

### Class `FittedModel`
Result of fitting one model to experimental data.


### Function `_get_model_class(name)`
Resolve model name to class.

### Function `_simulate(model_class, params, current, dt)`
Run a model with given params and current injection.

### Function `_cost_rmse(voltage_target, voltage_model)`
Root mean squared error between two voltage traces.

### Function `_cost_features(target_feats, model_feats)`
Feature-based cost: compare spike count, rate, ISI statistics.

### Function `_fit_single_model(model_class, model_name, voltage_target, current, dt, threshold)`
Fit one model to the target recording.

### Function `fit(voltage, current, dt, threshold, candidates, top_k)`
Fit neuron models to an experimental voltage recording.

Parameters
----------
voltage : ndarray
    Target voltage trace.
current : ndarray
    Injected current trace (same length as voltage).
dt : float
    Timestep in ms.
threshold : float
    Spike detection threshold.
candidates : list of str, optional
    Model names to try. Default: all fittable models.
top_k : int
    Return top K best-fitting models.

Returns
-------
list of FittedModel
    Sorted by combined_score (lower is better).

---

## Module `bci_studio.bci_primitives`

### Class `BCIPrimitiveConfig`
Configuration for the deterministic closed-loop primitive.

- **__post_init__**()

### Class `BCIFrame`
One raw neural signal frame.


### Class `BCIFeedbackCommand`
Feedback command emitted by the primitive.

Packet layout is 24 bytes: `&#91;schema:u16, command:u8, flags:u8,
channel:u16, reserved:u16, amplitude:f32, timestamp_us:u64, score:f32&#93;`.

- **to_packet**()
- **from_packet**(cls, packet)

### Class `BCIClosedLoopTrace`
Audit trace for one processed frame.

- **as_dict**()

### Class `BCIPrimitiveResult`
Result from one closed-loop primitive step.

- **as_legacy_dict**()

### Class `BCIClosedLoopPrimitive`
Deterministic raw-signal to feedback primitive with audit trace.

- **__init__**(config)
- **process_frame**(frame)
- **_next_frame_id**(explicit)
- **_validate_samples**(samples)
- **_extract_spikes**(samples)
- **_score**(channel_counts, n_samples)
- **_build_command**(score, timestamp_us)
- **_adapt**(channel_counts, command, reward)

### Class `BCIClosedLoopEngine`
Backward-compatible wrapper around :class:`BCIClosedLoopPrimitive`.

- **__init__**(channels)
- **weights**()
- **process_bci_frame**(raw_ephys, reward)

---

## Module `bci_studio.bci_studio`

### Class `SessionMetrics`
- **mean_latency_ms**()
- **p95_latency_ms**()
- **spike_rate**()
- **summary**()

### Class `SpikeCodec`
SC-domain lossy compression for neural data streams.

Uses run-length encoding on spike trains with delta-time encoding.

- **encode**(spikes)
  - Compress boolean spike array to RLE byte stream.
- **decode**(data)
  - Decompress RLE byte stream back to spike array.
- **compression_ratio**(original)
  - Return compression ratio (original_bytes / compressed_bytes).

### Class `OnlineLearner`
Local STDP-inspired weight update rule (pure Python fallback).

- **__init__**(num_weights, lr, decay)
- **step**(spikes, reward)
  - Apply reward-modulated STDP update.

### Class `FPGAFeedbackController`
Serializes BCI commands for DMA push to FPGA feedback register.

- **serialize**(command, channel, amplitude, timestamp_us)
  - Pack a feedback command into a 16-byte DMA-aligned struct.
- **deserialize**(data)
  - Unpack a feedback command.

### Class `LatencyProfiler`
Rolling window latency tracker with percentile reporting.

- **__init__**(window_size)
- **record**(latency_ms)
- **mean**()
- **p50**()
- **p95**()
- **p99**()
- **budget_met**()
  - True if p95 latency is under 10 ms BCI hard real-time target.

### Class `BCIStudio`
End-to-end BCI closed-loop orchestrator.

- **__init__**(channels, lr)
- **start_session**()
- **stop_session**()
- **process_frame**(raw_ephys, reward)
  - Process a single BCI frame through the full pipeline.

---

## Module `benchmarks.metrics`

### Class `BenchmarkResult`
NeuroBench-compatible benchmark result.

- **to_neurobench_json**()
  - Export as NeuroBench-compatible JSON.
- **summary**()

### Function `compute_metrics(predictions, targets, spike_counts, weights, timesteps, latency_ms, task, model)`
Compute NeuroBench-compatible metrics from model outputs.

Parameters
----------
predictions : ndarray
    Model predictions (class indices for classification).
targets : ndarray
    Ground truth labels.
spike_counts : ndarray, optional
    Per-sample total spike counts.
weights : list of ndarray, optional
    Weight matrices for parameter counting.
timesteps : int
    Number of simulation timesteps.
latency_ms : float
    Inference latency in milliseconds.
task : str
    Task name for the report.
model : str
    Model name for the report.

Returns
-------
BenchmarkResult

---

## Module `benchmarks.mlperf_sc_report`

### Function `aggregate_mlperf_sc_results(result_paths)`
Aggregate validated MLPerf-SC result files into a deterministic report.

---

## Module `benchmarks.mlperf_sc_runner`

### Function `run_mlperf_sc_fixture()`
Run the deterministic synthetic MLPerf-SC fixture and return result path.

### Function `_synthetic_xor_raw_payload()`
### Function `_fixture_baseline(model)`
### Function `_environment_payload()`
### Function `_write_canonical_json(path, payload)`
### Function `_sha256_file(path)`
---

## Module `benchmarks.mlperf_sc_schema`

### Class `MLPerfSCValidationError`
Raised when an MLPerf-SC result violates the fail-closed schema.


### Class `MLPerfSCRun`
Benchmark run identity and dataset contract.


### Class `MLPerfSCExecution`
Execution target and stochastic-computing mode metadata.


### Class `MLPerfSCArea`
Hardware utilisation metrics when a synthesis or board path exists.


### Class `MLPerfSCMetrics`
Accuracy, latency, energy, power, and area metrics.


### Class `MLPerfSCArtifact`
One evidence artifact referenced by the result.


### Class `MLPerfSCEvidence`
Evidence class, environment manifest, and raw artifact references.


### Class `MLPerfSCResult`
Typed MLPerf-SC benchmark result.


### Function `validate_mlperf_sc_result(payload)`
Validate a decoded MLPerf-SC result and return a typed result.

### Function `mlperf_sc_result_to_dict(result)`
Serialise a typed MLPerf-SC result to the canonical dictionary shape.

### Function `_run_from_mapping(payload)`
### Function `_execution_from_mapping(payload)`
### Function `_metrics_from_mapping(payload)`
### Function `_area_from_mapping(payload)`
### Function `_evidence_from_mapping(payload)`
### Function `_artifact_from_mapping(payload)`
### Function `_validate_relative_artifact_path(path)`
### Function `_validate_evidence_metrics(evidence, metrics)`
### Function `_expect_keys(payload, expected, label)`
### Function `_expect_mapping(value, label)`
### Function `_expect_sequence(value, label)`
### Function `_expect_non_empty_string(value, label)`
### Function `_expect_int(value, label)`
### Function `_expect_optional_non_negative_int(value, label)`
### Function `_expect_float(value, label)`
### Function `_expect_optional_positive_float(value, label)`
---

## Module `benchmarks.online_o1_adaptation`

### Class `_OnlineO1Runner`
- **step**()

### Function `build_online_o1_adaptation_benchmark()`
Return a deterministic Python/Rust adaptation benchmark report.

### Function `write_online_o1_adaptation_benchmark(path)`
Write a canonical benchmark report and return the output path.

### Function `_rust_pairing_protocol(config)`
### Function `_run_pairing_protocol(runner)`
---

## Module `benchmarks.stochastic_backprop`

### Function `build_stochastic_backprop_benchmark()`
Return deterministic evidence for SC-aware backpropagation loss reduction.

### Function `write_stochastic_backprop_benchmark(path)`
Write a canonical stochastic backpropagation benchmark report.

### Function `build_stochastic_backprop_estimator_regression_manifest()`
Return seeded estimator-family variance evidence across bitstream lengths.

### Function `write_stochastic_backprop_estimator_regression_manifest(path)`
Write seeded estimator-family regression evidence to canonical JSON.

### Function `_task_loss(inputs, targets, weight, bias, sc_config)`
### Function `_design_length_options(bitstream_length)`
### Function `_validate_bitstream_length_grid(bitstream_lengths)`
### Function `_joint_design_snapshot(report)`
### Function `_estimator_variance_evidence()`
### Function `_gradient_estimator_stats(estimates)`
### Function `_all_estimator_variances_are_finite_nonnegative(row)`
### Function `_round_nested(value)`
---

## Module `benchmarks.tasks`

### Class `BenchmarkTask`
Definition of a benchmark task.


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

## Module `bio.transcriptomic`

### Class `ScKGBERTInterface`
Knowledge-enhanced foundation model for single-cell transcriptomics.

Li Y, Qiao G, Du H, Gao X, Wang G. "scKGBERT: a knowledge-enhanced
foundation model for single-cell transcriptomics." Genome Biology
26:402 (2025).

Dual-encoder architecture: S-Encoder (sequence) + K-Encoder
(knowledge graph from STRING PPI database).  Uses Gaussian
attention for biomarker identification.

Gaussian attention (Li et al. 2025):
    α_ij = exp(-||q_i - k_j||² / (2σ²)) / Σ_m exp(-||q_i - k_m||² / (2σ²))

This emphasises genes whose query-key distance is small, concentrating
attention on biologically relevant gene–gene interactions.

- **__post_init__**()
- **gaussian_attention**(queries, keys, values)
  - Gaussian attention mechanism. Li et al. (2025).
- **encode_expression**(expression)
  - Encode a single-cell expression profile via S-Encoder.
- **encode_with_knowledge**(expression)
  - Encode via dual S-Encoder + K-Encoder pathway.
- **predict_cell_type**(expression, prototypes, labels)
  - Predict cell type via nearest prototype.
- **gene_importance**(expression)
  - Compute gene importance scores via Gaussian attention weights.

### Class `GeneformerInterface`
Rank-value tokenisation and masked gene prediction.

Theodoris CV et al. "Transfer learning enables predictions in
network biology." Nature 619 (2023).

Core innovation: each cell's transcriptome is represented as a
sequence of gene tokens, ranked by expression scaled by inverse
corpus frequency.  The model is pretrained with masked gene
prediction (analogous to BERT MLM) to learn network dynamics.

- **__post_init__**()
- **tokenise**(expression, global_medians)
  - Rank-value tokenisation. Theodoris et al. (2023).
- **mask_tokens**(token_ids, rng_seed)
  - Randomly mask tokens for MLM pretraining.
- **multi_head_attention**(x)
  - Multi-head self-attention. Vaswani et al. (2017).
- **encode_cell**(expression, global_medians)
  - Extract cell-level embedding from expression profile.
- **predict_masked_genes**(expression, global_medians, rng_seed)
  - Masked gene prediction (MLM objective).
- **gene_network_attention**(expression, global_medians)
  - Extract attention-derived gene–gene interaction matrix.

### Function `rank_value_encode(expression, global_medians)`
Rank-value encoding for single-cell gene expression.

Theodoris et al. (2023): genes are ranked by their expression
in the cell, scaled by inverse frequency across the corpus
(approximated by 1 / global_median).

Parameters
----------
expression : 1-D array &#91;n_genes&#93;, raw counts or normalised expression.
global_medians : 1-D array &#91;n_genes&#93;, median expression per gene across
    the corpus.  If None, uniform weighting is used.

Returns
-------
ranked_indices : int64 array — gene indices sorted by weighted expression
    (highest first), with zero-expression genes excluded.

---

## Module `bioware.bioware`

### Class `MEALayout`
Standard MEA electrode layouts.


### Class `MEAConfig`
Multi-electrode array configuration.

- **from_layout**(cls, layout)

### Class `DetectedSpike`
One detected spike event from MEA data.


### Class `SpikeDetector`
Threshold-based spike detector for MEA voltage traces.

Uses adaptive threshold: threshold = mean ± sigma * noise_estimate
where noise_estimate = median(|x|) / 0.6745 (robust RMS).
Supports configurable refractory period to prevent double-counting.

- **estimate_noise**(voltage_data)
  - Estimate per-channel noise from voltage data.
- **detect**(voltage_data, snippet_ms)
  - Detect spikes in multi-channel voltage data.

### Class `AEREvent`
Address-Event Representation packet.

Compatible with sc_aer_encoder.v format:
{valid, neuron_id, timestamp}


### Class `MEAToAERTranscoder`
Converts MEA spike events to AER events for hardware.

Maps biological electrode channels to AER neuron IDs,
converting real-time timestamps to hardware clock ticks.

- **transcode**(spikes, t_start_s)
  - Convert detected spikes to AER events.
- **_map_channel**(channel)

### Class `AERToSCConverter`
Converts AER event streams to SC bitstreams.

Uses a time-windowed rate code: count events per neuron per window,
then LFSR-encode the resulting firing probabilities.

- **convert**(events)
  - Convert AER events to per-neuron SC bitstreams.
- **_lfsr_encode**(probability, neuron_id)
  - LFSR-16 encoding (bit-compatible with core_engine).

### Class `StimProtocol`
Optogenetic stimulation protocols.


### Class `OptogeneticPulse`
One optical stimulation pulse.


### Class `SCToOptoEncoder`
Encodes SC bitstream output as optogenetic pulse sequences.

Maps SC bitstream density to optical stimulation intensity,
enabling closed-loop feedback from in-silico → biological.
Enforces total power budget for tissue safety.

- **encode**(bitstreams, t_start_ms)
  - Convert SC bitstreams to optogenetic pulses.

### Class `BiologicalSTDP`
Spike-Timing-Dependent Plasticity adapter for bio-hybrid loops.

Bridges biological STDP time constants (∼20 ms) to SC clock
rates (MHz) via a time-scaling factor. Computes ΔW from
pre/post spike timing in biological time, then converts to
Q8.8 weight updates for the SC domain.

- **compute_dw**(dt_ms)
  - Compute weight change from spike timing difference.
- **update_weight**(current_q88, dt_ms)
  - Update Q8.8 weight from spike timing.

### Class `BCMPlasticity`
Bienenstock-Cooper-Munro plasticity adapter.

Implements sliding-threshold BCM rule where the modification
threshold θ tracks the postsynaptic firing rate. Converts
biological firing rates to Q8.8 weight deltas.

- **update_theta**(post_rate_hz, dt_ms)
  - Update the sliding threshold from postsynaptic activity.
- **compute_dw**(pre_rate_hz, post_rate_hz)
  - BCM weight change: ΔW = η * x * y * (y - θ).
- **update_weight**(current_q88, pre_rate, post_rate)

### Class `CultureHealth`
Monitor organoid/culture viability from MEA activity.

- **assess**(spike_counts, duration_s)
  - Assess culture health from spike activity.

### Class `BioHybridFrameResult`
Strictly typed output packet detailing a full closed-loop step.

Behaves both as a dataclass (``result.round``) and, for backward
compatibility with pre-dataclass callers, as a mapping view of its
fields (``result&#91;"round"&#93;``, ``"latency_us" in result``,
``dict(result)``). The mapping surface is read-only.

- **__getitem__**(key)
- **__contains__**(key)
- **keys**()

### Class `BioHybridSession`
Manages a complete bio-hybrid experiment session.

Orchestrates: MEA recording → spike detection → AER transcoding →
SC processing → optogenetic feedback → plasticity update.

- **process_frame**(voltage_data, t_start_s, stim_times_s)
  - Process one MEA data frame through the full pipeline.

### Class `SpikeSorter`
Production-ready spike sorter utilizing PCA feature extraction and K-Means clustering.

Extracts the dominant principal components from the input raw waveforms, and cleanly
separates units. Handles missing datasets explicitly natively. Requires `scikit-learn` to execute correctly.

- **fit**(spikes)
  - Fit PCA and KMeans models sequentially on available waveforms.
- **assign**(spikes)
  - Assign cluster IDs based on PCA feature projections.

### Class `LFPBand`
Frequency band definition for LFP extraction.


### Class `LatencyBudget`
Tracks and enforces closed-loop latency requirements.

- **record**(latency_us)
  - Record a latency measurement. Returns True if within budget.
- **mean_latency_us**()
- **p99_latency_us**()
- **compliance_ratio**()

### Class `PharmModel`
Simulates effect of pharmacological agents on spike rate.

Models excitatory (e.g., bicuculline) or inhibitory (e.g., TTX) agents
as gain factors on firing rate.

- **apply**(t_current_s)
- **effective_gain**(t_current_s)
- **modulate_spikes**(spike_counts, t_current_s)
  - Modulate spike counts by pharmacological gain.
- **modulate_spike_events**(spikes, t_current_s)
  - Apply pharmacological rate gain to spike events.

### Class `WellConfig`
One well in a multi-well MEA plate.

- **label**()

### Class `MultiWellPlate`
Multi-well plate (e.g., 6/24/48/96-well MEA plate).

- **add_well**(well)
- **standard_6_well**(cls, layout)
- **num_wells**()
- **get_well**(well_id)

### Class `NetworkBurst`
Detected network-wide synchronised burst event.


### Class `ArtifactRejector`
Blanks stimulation artifacts from voltage data.

Zeros the voltage trace in a window around each stimulation onset.

- **blank**(voltage_data, stim_times_s, sample_rate_hz)
  - Return voltage data with stimulus artifacts blanked.

### Class `BioAuditEntry`
One audit entry for a bio-hybrid session.


### Class `BioAuditLog`
Regulatory-grade audit log for bio-hybrid experiments.

- **log**(entry)
- **total_rounds**()
- **to_list**()
- **checksum**()
  - SHA-256 of log contents for tamper detection.

### Class `HomeostaticPlasticity`
Intrinsic excitability scaling to maintain target firing rate.

Implements homeostatic plasticity: if a neuron fires too fast,
reduce its excitability (threshold up); too slow, increase it.
Operates on Q8.8 threshold values.

- **update_threshold**(current_q88, observed_rate_hz, dt_ms)
  - Adjust threshold to drive firing rate toward target.

### Function `extract_lfp_power(voltage_data, sample_rate_hz, bands)`
Extract per-channel power in each LFP band.

Uses FFT-based power spectral density estimation.
Returns dict of band_name → per-channel power array.

### Function `_quantile_indices(n_items, target_count)`
### Function `_clone_spike(template)`
### Function `detect_network_bursts(spikes, bin_width_s, threshold_sigma, min_channels)`
Detect network-wide synchronised bursts.

Bins spikes in time, detects bins with activity > threshold_sigma
above the mean, and requires participation from ≥ min_channels.

### Function `decode_bitstream_rate(bitstreams, sc_clock_hz)`
Decode SC bitstreams back to biological firing rates (Hz).

Interprets popcount/length as probability, scales by SC clock
to get equivalent biological firing rate.

### Function `mea_fitness_hook(detected_spikes, target_rate)`
Organism fitness metrics derived from MEA response dynamics.

Designed to plug into the evo_substrate
``ReplicationEngine(metrics_fn=mea_fitness_hook)`` — returns the
``{"accuracy", "energy_mw", "latency_ms"}`` triple the engine scores.

Accuracy is a bounded distance to the target mean per-channel firing
rate when ``duration_s`` is supplied, or to the legacy per-channel
spike count when it is omitted. ``energy_mw`` remains the documented
spike-count proxy (0.5 mW / spike). ``latency_ms`` is either a caller
supplied closed-loop measurement, the first response latency after
``stimulus_time_s``, or the first spike timestamp relative to frame
start.

### Function `_mea_response_latency_ms(detected_spikes)`
---

## Module `bridges.aer_router`

### Class `SpikePacket`
Wire format for an AER spike event (28 bytes big-endian).

- **encode**()
  - Serialize to 28-byte big-endian frame.
- **decode**(cls, data)
  - Deserialize from 28-byte big-endian frame.

### Class `RouteStats`
Per-route delivery statistics.


### Class `AERRouter`
Manages route registration, spike dispatch, and ACK tracking.

This is a pure-Python simulation/client. For high-performance
UDP routing, use the Go server (hil_debugger/interconnect).

- **__init__**()
- **register_route**(neuron_id, addr)
  - Map a neuron ID to a destination address (host:port).
- **unregister_route**(neuron_id)
  - Remove a route for the given neuron ID.
- **route_count**()
  - Return the number of currently registered neuron routes.
- **dispatch_spike**(packet)
  - Dispatch a spike packet to the registered target.
- **ack_received**(seq)
  - Process an ACK for the given sequence number.
- **pending_count**()
  - Return the number of dispatched packets awaiting ACKs.
- **total_sent**()
  - Return the total number of packets accepted for dispatch.
- **total_acked**()
  - Return the total number of ACKs processed by the router.
- **get_stats**(neuron_id)
  - Return a defensive copy of per-route statistics when present.

---

## Module `bridges.dna_mapper`

### Class `GateType`
Supported DNA logic gate types.


### Class `CompilationMethod`
Compilation target for the DNA circuit.


### Class `DNAStrand`
A single-stranded DNA molecule used in a circuit.

Attributes
----------
name : str
    Unique identifier (e.g. ``"gate_0_input_a"``).
sequence : str
    5' → 3' nucleotide sequence (A, C, G, T).
role : str
    Functional role: ``"signal"``, ``"fuel"``, ``"output"``,
    ``"waste"``, ``"toehold"``, ``"translator"``.
concentration_nM : float
    Initial concentration in nanomolar.

- **length**()
- **gc_content**()
- **complement**()
- **max_homopolymer_run**()
- **delta_g_37**()
  - Nearest-neighbour ΔG° at 37 °C (kcal/mol).
- **melting_temperature**(na_conc_M, strand_conc_M)
  - Return nearest-neighbour DNA duplex melting temperature in °C.

### Class `DNAGate`
A logic gate implemented via DNA strand displacement.

Attributes
----------
gate_id : int
    Unique gate index in the circuit.
gate_type : GateType
    Logic operation (AND, OR, NOT, etc.).
input_names : list&#91;str&#93;
    Names of input signal strands.
output_name : str
    Name of the output signal strand.
strands : list&#91;DNAStrand&#93;
    All DNA strands participating in this gate (inputs, fuel,
    translator complexes, output, waste).
threshold : float
    For threshold gates, the activation threshold concentration.
leak_rate : float
    Estimated spurious activation rate (per second).

- **strand_count**()

### Class `DNACircuitDesign`
Complete compiled DNA circuit.

Holds the full strand-level design for a compiled SC network,
including all gates, signal routing, and thermodynamic validation.

Attributes
----------
name : str
    Circuit identifier.
gates : list&#91;DNAGate&#93;
    Ordered list of compiled gates.
input_strands : list&#91;DNAStrand&#93;
    Primary input signal strands.
output_strands : list&#91;DNAStrand&#93;
    Primary output signal strands.
fuel_strands : list&#91;DNAStrand&#93;
    Fuel/helper strands consumed during computation.
method : CompilationMethod
    Compilation target used.
temperature_c : float
    Design temperature in Celsius.
na_concentration_M : float
    Sodium concentration for thermodynamic calculations.

- **total_strands**()
- **total_gates**()
- **total_nucleotides**()
- **validate**()
  - Run design rule checks. Returns list of warnings.

### Class `SequenceDesigner`
Deterministic DNA sequence generator with constraint satisfaction.

Generates sequences that satisfy GC content, homopolymer, and
orthogonality constraints using a seed-based deterministic algorithm.
This ensures reproducible designs without requiring NUPACK.

Parameters
----------
seed : int
    Random seed for reproducible sequence generation.
gc_target : tuple&#91;float, float&#93;
    Acceptable GC content range (default 0.40–0.60).
max_homopolymer : int
    Maximum consecutive identical nucleotides (default 3).

- **__init__**(seed, gc_target, max_homopolymer)
- **generate**(length, name)
  - Generate a sequence satisfying all constraints.
- **generate_complement**(sequence)
  - Return the Watson-Crick complement (3' → 5').
- **generate_toehold**(name)
  - Generate a toehold domain (6 nt).
- **generate_recognition**(name)
  - Generate a recognition domain (15 nt).

### Class `StrandDisplacementCompiler`
Compile SC Boolean gates into toehold-mediated displacement circuits.

Implements the seesaw gate architecture from Qian & Winfree (2011)
adapted for SC-NeuroCore's bitstream operations.

Parameters
----------
designer : SequenceDesigner | None
    Sequence generator. If None, a default is created.
temperature_c : float
    Target operating temperature in Celsius.

- **__init__**(designer, temperature_c)
- **compile_and**(input_a, input_b, output)
  - Compile a 2-input AND gate.
- **compile_or**(input_a, input_b, output)
  - Compile a 2-input OR gate via catalytic hairpin assembly.
- **compile_not**(input_name, output)
  - Compile a NOT gate via strand sequestration.
- **compile_threshold**(input_name, output, threshold)
  - Compile a threshold gate for concentration-dependent activation.
- **compile_mux**(select, input_a, input_b, output)
  - Compile a 2:1 multiplexer (MUX) gate.
- **compile_amplifier**(input_name, output)
  - Compile a catalytic signal amplifier.
- **compile_buffer**(input_name, output)
  - Compile a signal restoration buffer.
- **_estimate_leak_rate**(strand, blocker)
  - Estimate spurious strand displacement rate.
- **_strongest_blocker_delta_g**(strand, blocker)
  - Return the most stable contiguous blocker-binding ΔG° at 37 °C.

### Class `EnzymaticGateCompiler`
Compile SC gates into enzyme-mediated DNA logic circuits.

Uses restriction enzymes and ligases to implement Boolean operations
on DNA substrates. Operates on double-stranded DNA with specific
recognition sites.

Parameters
----------
designer : SequenceDesigner | None
    Sequence generator.

- **__init__**(designer)
- **compile_nand**(input_a, input_b, output)
  - NAND gate via dual restriction enzyme cascade.
- **compile_xor**(input_a, input_b, output)
  - XOR gate via nick-sealing ligase logic.

### Class `NUPACKInterface`
Interface to NUPACK for thermodynamic validation.

Provides minimum free energy (MFE) structure prediction, base-pair
probability computation, and design validation. Falls back to
internal nearest-neighbour estimates, Watson-Crick secondary-structure
dynamic programming, and Boltzmann-style pair probabilities when NUPACK is
not installed.

Parameters
----------
temperature_c : float
    Temperature in Celsius.
na_concentration_M : float
    Sodium concentration in molar.

- **__init__**(temperature_c, na_concentration_M)
- **has_nupack**()
- **compute_mfe**(sequence)
  - Compute minimum free energy and structure.
- **compute_pair_probabilities**(sequence)
  - Compute base-pair probability matrix.
- **validate_design**(design)
  - Validate a full circuit design.

### Class `KineticSimulator`
Mass-action kinetics simulator for DNA strand displacement.

Simulates the time evolution of strand concentrations using
selectable integration (Euler or RK4) with Arrhenius temperature
scaling of rate constants.

Parameters
----------
rate_hybridization : float
    Second-order rate constant for toehold binding (M⁻¹ s⁻¹).
rate_displacement : float
    First-order rate constant for branch migration (s⁻¹).
temperature_c : float
    Temperature in Celsius.
integrator : str
    Integration method: ``"euler"`` or ``"rk4"``.

- **__init__**(rate_hybridization, rate_displacement, temperature_c, integrator)
- **_arrhenius_scale**(k_ref, ea_kcal)
  - Scale rate constant from 37°C to operating temperature via Arrhenius.
- **_compute_k_eff**(gate, input_concentrations)
  - Compute effective rate constant for a gate.
- **simulate**(design, input_concentrations, duration_s, dt)
  - Simulate circuit kinetics.

### Class `BitstreamToDNA`
High-level API for mapping SC bitstreams to DNA circuits.

This is the primary entry point for the DNA computing bridge.
Accepts a description of an SC Boolean network and compiles it
into a complete DNA circuit design.

Parameters
----------
method : str
    Compilation method: ``"displacement"`` (default),
    ``"enzymatic"``, or ``"hybrid"``.
seed : int
    Sequence generation seed for reproducibility.
temperature_c : float
    Design temperature in Celsius.

Examples
--------
>>> compiler = BitstreamToDNA(method="displacement", seed=42)
>>> design = compiler.compile_network(
...     gates=&#91;
...         {"type": "AND", "inputs": &#91;"A", "B"&#93;, "output": "C"},
...         {"type": "NOT", "inputs": &#91;"C"&#93;, "output": "D"},
...     &#93;,
...     input_names=&#91;"A", "B"&#93;,
...     output_names=&#91;"D"&#93;,
... )
>>> print(design.total_gates)
2
>>> print(design.total_strands)
...
>>> export_genbank(design, "nand_circuit.gb")

- **__init__**(method, seed, temperature_c)
- **compile_network**(gates, input_names, output_names, name)
  - Compile an SC Boolean network into a DNA circuit.
- **simulate**(design, input_concentrations, duration_s, dt)
  - Simulate the compiled circuit.
- **validate**(design)
  - Validate design using NUPACK (or fallback).
- **_compile_displacement_gate**(gate_type, inputs, output, spec)
- **_compile_enzymatic_gate**(gate_type, inputs, output, spec)

### Class `GF4ErrorCorrection`
Reed–Solomon-like error correction over GF(4) for DNA sequences.

Maps nucleotides to GF(4) elements: A=0, C=1, G=2, T=3.
Adds parity symbols for error detection and correction of
synthesis/sequencing errors.

Parameters
----------
n_parity : int
    Number of parity nucleotides per block (default 4).
block_size : int
    Data nucleotides per block (default 12).

- **__init__**(n_parity, block_size)
- **encode**(sequence)
  - Add error-correction parity nucleotides to a sequence.
- **decode**(encoded_sequence)
  - Decode and correct errors. Returns (corrected_data, n_corrections).
- **_compute_parity**(symbols)
  - Compute parity symbols over GF(4).

### Class `CrossHybridizationChecker`
Detect unwanted cross-hybridization between circuit strands.

Computes a pairwise alignment score matrix for all strands in a
design and flags pairs with dangerous complementarity.

Parameters
----------
max_complementary_run : int
    Maximum allowed consecutive complementary bases between
    two non-interacting strands (default 8).

- **__init__**(max_complementary_run)
- **check**(design)
  - Check all strand pairs for cross-hybridization.
- **_longest_common_substring**(a, b)
  - Length of the longest common substring.

### Class `NoiseModel`
Monte Carlo noise injection for robustness analysis.

Perturbs strand concentrations, hybridization rates, and
temperature to assess circuit robustness under realistic
experimental conditions.

Parameters
----------
concentration_cv : float
    Coefficient of variation for pipetting noise (default 0.05 = 5%).
temperature_std_c : float
    Temperature uncertainty in °C (default 0.5).
n_trials : int
    Number of Monte Carlo trials (default 50).
seed : int
    Random seed.

- **__init__**(concentration_cv, temperature_std_c, n_trials, seed)
- **sensitivity_analysis**(design, input_concentrations, duration_s)
  - Run Monte Carlo sensitivity analysis.

### Class `TopologicalAnalyzer`
Analyze circuit topology: depth, fan-out, feedback detection.

Builds a directed graph from gate connectivity, then computes:
- Topological sort order (or detects cycles)
- Circuit depth (critical path length)
- Fan-out per signal (number of consumers)
- Feedback loops (cycles in the gate graph)

- **analyze**(design)
  - Run full topological analysis.

### Class `DualRailEncoder`
Dual-rail encoding for fault-tolerant DNA circuits.

Each logical signal is encoded as two physical strands: the
"true" rail and the "complement" rail. Valid states:
    - (high, low)  = logical 1
    - (low, high)  = logical 0
    - (high, high) = fault detected
    - (low, low)   = fault detected

This provides single-fault detection for each signal.

- **encode**(design, compiler)
  - Convert a single-rail circuit to dual-rail.
- **check_faults**(result, threshold_nM)
  - Detect faults in dual-rail simulation results.
- **_complement_gate_type**(gate_type)
  - De Morgan complement gate type.

### Class `ConcentrationOptimizer`
Gradient-free optimization of strand concentrations.

Uses Nelder–Mead simplex to minimize output error across
all truth-table entries, finding optimal working concentrations
for translator, threshold, and fuel strands.

Parameters
----------
n_evaluations : int
    Maximum function evaluations (default 200).
seed : int
    Random seed for initial simplex.

- **__init__**(n_evaluations, seed)
- **optimize**(design, truth_table, duration_s)
  - Optimize concentrations against a truth table.

### Class `SCNetworkBridge`
Bridge between SC-NeuroCore network objects and DNA compilation.

Converts Population/Projection-based SC networks into the gate-spec
format consumed by BitstreamToDNA. Supports automatic gate-type
inference from connection weights.

Parameters
----------
method : str
    Compilation method (``"displacement"`` or ``"enzymatic"``).
seed : int
    Random seed.

- **__init__**(method, seed)
- **from_adjacency**(adjacency, input_indices, output_indices, name)
  - Compile from an adjacency matrix.

### Class `HairpinChecker`
Detect potential hairpin (stem-loop) secondary structures.

Scans each strand for self-complementary regions that could form
intramolecular hairpins, reducing effective concentration and
interfering with gate operation.

Parameters
----------
min_stem_length : int
    Minimum stem length to flag (default 4 bp).
min_loop_length : int
    Minimum loop length for a valid hairpin (default 3 nt).

- **__init__**(min_stem_length, min_loop_length)
- **check_strand**(sequence)
  - Find potential hairpins in a single sequence.
- **check_design**(design)
  - Check all strands in a circuit for hairpins.

### Class `GateOptimizer`
Circuit-level gate optimization.

Performs:
- Dead gate elimination (outputs not consumed by any downstream gate)
- Constant propagation (gates with all-zero or all-max inputs)
- Identity elimination (BUFFER gates with direct pass-through)
- Duplicate detection (identical gate specs)

- **optimize**(gates, output_names)
  - Optimize a gate list before compilation.

### Class `SCPrecisionAnalyzer`
Stochastic computing precision analysis for DNA circuits.

Evaluates the effective bit-width, signal-to-noise ratio, and
output precision achievable by a DNA-encoded SC circuit at given
strand concentrations.

In standard SC, a bitstream of length L encodes precision
log2(L+1) bits. In DNA circuits, the analog concentration range
&#91;0, max_nM&#93; plays the role of L.

- **analyze**(design, input_concentrations, max_conc_nM, duration_s)
  - Analyze SC precision of a compiled circuit.

### Class `DegradationModel`
Time-dependent DNA strand degradation model.

Models first-order exponential decay of strand concentrations
based on nuclease activity, temperature, and strand length.

Parameters
----------
half_life_hr : float
    Base half-life in hours at 37°C (default 24 for ssDNA).
temperature_c : float
    Operating temperature in Celsius.

- **__init__**(half_life_hr, temperature_c)
- **_length_factor**(length)
  - Longer strands degrade faster (more nuclease attack sites).
- **_temp_factor**()
  - Higher temperature accelerates degradation.
- **predict_concentration**(initial_nM, strand_length, time_hr)
  - Predict remaining concentration after time_hr hours.
- **analyze_design**(design, time_hr)
  - Predict degradation across all circuit strands.

### Class `PlateLayout`
Organize oligos into 96-well synthesis plate format.

Maps each unique oligo to a well position (A01–H12), generates
ordering manifests for IDT/Sigma/Eurofins, and computes plate
utilization.

Parameters
----------
n_wells : int
    Wells per plate (default 96).

- **__init__**(n_wells)
- **layout**(design)
  - Generate plate layout for a circuit design.

### Function `_canonical_sequence(sequence)`
### Function `_can_pair(left, right)`
### Function `_hairpin_loop_penalty(loop_nt)`
### Function `_fallback_pair_energy(sequence, i, j)`
### Function `_fallback_secondary_structure(sequence)`
### Function `_fallback_pair_probability_matrix(sequence, temperature_c)`
### Function `export_genbank(design, path)`
Export circuit design to GenBank format.

Creates a multi-record GenBank file with one record per strand,
including annotations for functional domains (toehold, recognition,
clamp).

Parameters
----------
design : DNACircuitDesign
    Compiled circuit.
path : str
    Output file path (e.g. ``"circuit.gb"``).

### Function `export_fasta(design, path)`
Export all strands to FASTA format.

Parameters
----------
design : DNACircuitDesign
    Compiled circuit.
path : str
    Output file path (e.g. ``"circuit.fasta"``).

### Function `export_nupack_input(design, path)`
Export circuit in NUPACK multi-strand input format.

Parameters
----------
design : DNACircuitDesign
    Compiled circuit.
path : str
    Output file path (e.g. ``"circuit.nupack"``).

### Function `export_json(design, path)`
Export circuit design as JSON for visualization/interchange.

Parameters
----------
design : DNACircuitDesign
    Compiled circuit.
path : str
    Output file path (e.g. ``"circuit.json"``).

### Function `estimate_cost(design, price_per_base_usd, fixed_per_oligo_usd, purification)`
Estimate oligo synthesis cost for a circuit design.

Parameters
----------
design : DNACircuitDesign
    Compiled circuit.
price_per_base_usd : float
    Cost per nucleotide (default $0.10 for standard desalt).
fixed_per_oligo_usd : float
    Fixed setup cost per oligonucleotide.
purification : str
    ``"standard"`` (1×), ``"hplc"`` (2.5×), ``"page"`` (3×).

Returns
-------
dict
    Cost breakdown: per-strand costs, total, summary.

### Function `generate_protocol(design, volume_uL, buffer_name)`
Generate a wet-lab protocol for assembling a DNA circuit.

Parameters
----------
design : DNACircuitDesign
    Compiled circuit.
volume_uL : float
    Total reaction volume in µL.
buffer_name : str
    Buffer system name.

Returns
-------
str
    Markdown-formatted protocol.

### Function `visualize_circuit(design)`
Generate a text-based circuit diagram.

Returns an ASCII diagram showing gate connectivity, signal flow,
and strand counts per gate.

Parameters
----------
design : DNACircuitDesign
    Compiled circuit.

Returns
-------
str
    Multi-line ASCII circuit diagram.

### Function `visualize_kinetics(result)`
Generate a text-based time-course chart.

Produces a simple ASCII sparkline for each output trace.

Parameters
----------
result : dict
    Simulation result from ``KineticSimulator.simulate()``.

Returns
-------
str
    Multi-line ASCII sparkline chart.

---

## Module `bridges.local_llm`

### Class `LocalLLMError`
Raised when the local LLM endpoint is unavailable or malformed.


### Class `LocalLLMProvider`
Local LLM endpoint protocol.


### Class `LocalLLMConfig`
Connection and generation settings for a local LLM endpoint.

- **resolved_provider**()
  - Resolve AUTO to a concrete provider using the configured URL.

### Class `LocalLLMResponse`
Structured response from a local LLM endpoint.


### Class `SpikePromptAdapter`
Convert spike activity into compact text suitable for local LLM prompts.

- **summarise_rates**(rates_hz)
  - Summarise per-neuron firing rates as compact ranked text.
- **raster_summary**(raster)
  - Summarise a boolean spike raster into rates and density statistics.

### Class `LocalLLMBridge`
Thin client for local LLM chat endpoints.

- **__init__**(config)
- **_endpoint**()
- **_validate_endpoint**(url)
- **_post_json**(payload)
- **chat**(user_prompt)
  - Send a prompt to the configured local LLM chat endpoint.
- **analyse_spike_raster**(raster)
  - Summarise a spike raster through the local LLM.

---

## Module `bridges.photonic_codesign`

### Class `PhotonicCoDesignConfig`
Configuration for a reproducible stochastic photonic design pass.

- **__post_init__**()

### Class `BitstreamEvidence`
One encoded SC channel and its statistical evidence.

- **to_json**()
  - Return a compact JSON-ready evidence record.

### Class `PhotonicCoDesignReport`
Complete output of a stochastic photonic co-design pass.

- **to_json**()
  - Return a deterministic JSON-ready report.
- **export_json**(path)
  - Write the report to a JSON file.

### Class `StochasticPhotonicCoDesignLoop`
End-to-end stochastic bitstream, photonic NoC, and FDTD loop.

- **__init__**(config)
- **compile**(adjacency)
  - Run the full co-design loop for one SC connectivity matrix.
- **_collect_blockers**()

### Function `_unpack_words(words, bit_length)`
### Function `_transition_count(bits)`
### Function `_density_tolerance(bitstream_length, alpha)`
Two-sided Hoeffding density tolerance for a Bernoulli bitstream.

### Function `derive_probabilities_from_adjacency(adjacency, floor, ceiling)`
Derive per-node SC probabilities from inbound absolute weight mass.

### Function `encode_bitstream_bank(probabilities)`
Encode probabilities into deterministic LFSR-backed SC bitstreams.

### Function `_scc_matrix(bitstreams)`
### Function `_layout_manifest(design)`
Create a PDA handoff manifest without claiming foundry DRC/LVS.

---

## Module `bridges.photonic_noc`

### Class `WaveguideType`
Photonic waveguide type.


### Class `WaveguideSegment`
A single waveguide path segment.

Attributes
----------
source : int
    Source node index.
target : int
    Target node index.
length_um : float
    Physical length in micrometers.
wavelength_nm : float
    Operating wavelength (default 1550 nm).
loss_db : float
    Total insertion loss for this segment.
n_crossings : int
    Number of waveguide crossings.
wg_type : WaveguideType
    Waveguide geometry type.


### Class `MZIGate`
Mach-Zehnder interferometer gate specification.

Models a single MZI stage implementing an SC computing operation
via thermo-optic or electro-optic phase shifting.

Attributes
----------
gate_id : str
    Unique gate identifier.
operation : str
    Gate operation type (AND, OR, NOT, MUL, ADD).
input_ports : list&#91;int&#93;
    Input waveguide port indices.
output_port : int
    Output waveguide port index.
phase_shift_rad : float
    Applied phase shift in radians.
arm_length_um : float
    MZI arm length in micrometers.
insertion_loss_db : float
    Total insertion loss.
extinction_ratio_db : float
    On/off extinction ratio.


### Class `WDMChannel`
Wavelength-division multiplexing channel.

Attributes
----------
channel_id : int
    Channel index.
wavelength_nm : float
    Center wavelength.
bandwidth_nm : float
    Channel bandwidth (default 0.8 nm for DWDM).
signal_name : str
    Associated SC signal name.
power_dbm : float
    Launch power.


### Class `PhotonicCircuitDesign`
Complete photonic NoC design.

Attributes
----------
name : str
    Design name.
waveguides : list&#91;WaveguideSegment&#93;
    All waveguide segments.
mzi_gates : list&#91;MZIGate&#93;
    All MZI computing stages.
wdm_channels : list&#91;WDMChannel&#93;
    WDM channel assignments.
n_nodes : int
    Number of processing element nodes.
routing_table : dict&#91;tuple&#91;int, int&#93;, list&#91;int&#93;&#93;
    Hop-by-hop routing table.
total_area_um2 : float
    Estimated chip area.


### Class `WaveguideRouter`
Route waveguides between SC network nodes.

Uses a mesh topology with shortest-path (Manhattan) routing.

Parameters
----------
pitch_um : float
    Node-to-node pitch in micrometers (default 250).
loss_db_per_cm : float
    Waveguide propagation loss (default 2.0 dB/cm).

- **__init__**(pitch_um, loss_db_per_cm)
- **route**(adjacency, node_labels)
  - Route waveguides for an SC network adjacency matrix.

### Class `MZICompiler`
Compile SC operations into MZI gate cascades.

Maps SC gates to photonic MZI operations:
- AND/MUL → MZI with π/2 phase shift (coherent multiplication)
- OR/ADD → Y-junction combiner
- NOT → MZI with π phase shift (bar state)

Parameters
----------
arm_length_um : float
    Default MZI arm length (default 200 μm).

- **__init__**(arm_length_um)
- **compile_gate**(gate_type, input_ports, output_port, gate_id)
  - Compile a single SC gate to an MZI specification.
- **compile_network**(gates)
  - Compile a list of SC gate specs into MZI cascade.

### Class `WDMAssigner`
Assign WDM channels to SC signal paths.

Parameters
----------
base_wavelength_nm : float
    Starting wavelength (default 1550.0 nm).
channel_spacing_nm : float
    Channel spacing (default 0.8 nm for 100 GHz DWDM).
max_channels : int
    Hard cap on the number of channels the assigner will emit.
    Default 96 follows the ITU-T G.694.1 DWDM C-band grid at
    50 GHz spacing (~0.4 nm). At the default 0.8 nm spacing the
    physical C-band (~1530-1565 nm, ~35 nm) only fits ~44
    channels — the cap protects callers from silently spilling
    into invalid wavelengths. Pass a larger value (or
    ``max_channels=0`` to disable) for multi-band (C+L+S)
    designs.

Raises
------
ValueError
    From :meth:`assign` when ``len(signal_names)`` exceeds
    ``max_channels`` and ``max_channels > 0``.

- **__init__**(base_wavelength_nm, channel_spacing_nm, max_channels)
- **assign**(signal_names, power_dbm)
  - Assign a WDM channel to each signal.

### Class `PowerBudgetAnalyzer`
Optical power budget and OSNR analysis.

Computes end-to-end power budget for each path in the photonic
circuit, flagging paths that exceed the detector sensitivity.

- **analyze**(design, laser_power_dbm, detector_sensitivity_dbm)
  - Run power budget analysis.

### Class `SCToPhotonic`
Top-level compiler: SC network → photonic NoC design.

Parameters
----------
pitch_um : float
    Mesh pitch (default 250 μm).
arm_length_um : float
    MZI arm length (default 200 μm).

- **__init__**(pitch_um, arm_length_um)
- **compile**(adjacency, node_labels, gate_specs, name)
  - Compile SC network into a photonic design.

### Class `ThermalPhaseShifter`
Thermo-optic phase shifter model for MZI tuning.

Parameters
----------
heater_length_um : float
    Heater length (default 100 μm).
dn_dt : float
    Thermo-optic coefficient (default 1.86e-4 K⁻¹ for Si).
thermal_resistance_kw : float
    Heater thermal resistance (default 10 K/mW).

- **__init__**(heater_length_um, dn_dt, thermal_resistance_kw)
- **power_for_phase**(phase_rad, wavelength_nm)
  - Compute electrical power needed for a given phase shift.
- **analyze_design**(design)
  - Compute total power budget for all MZI phase shifters.

### Class `CrosstalkAnalyzer`
Analyze inter-channel crosstalk in WDM systems.

Parameters
----------
adjacent_xt_db : float
    Adjacent-channel crosstalk (default -25 dB).

- **__init__**(adjacent_xt_db)
- **analyze**(channels)
  - Analyze crosstalk between WDM channels.

### Function `export_photonic_json(design, path)`
Export photonic design to JSON.

Parameters
----------
design : PhotonicCircuitDesign
    The design to export.
path : str
    Output file path.

### Function `visualize_photonic(design)`
Generate ASCII visualization of a photonic design.

Returns
-------
str
    Multi-line ASCII representation.

---

## Module `bridges.quantum_annealing`

### Class `ProblemType`
Quantum optimization problem type.


### Class `QubitSpec`
Specification for a single logical qubit.

Attributes
----------
index : int
    Logical qubit index.
label : str
    Human-readable label (e.g. neuron name).
bias : float
    Local field / linear bias (h_i in Ising, Q_ii in QUBO).


### Class `CouplerSpec`
Specification for a qubit-qubit coupling.

Attributes
----------
qubit_a : int
    First qubit index.
qubit_b : int
    Second qubit index.
strength : float
    Coupling strength (J_ij in Ising, Q_ij in QUBO).


### Class `IsingModel`
Ising spin-glass model: H = Σ h_i·s_i + Σ J_ij·s_i·s_j.

Attributes
----------
h : dict&#91;int, float&#93;
    Linear biases (local fields). Key = qubit index.
J : dict&#91;tuple&#91;int, int&#93;, float&#93;
    Quadratic couplings. Key = (i, j) pair, i < j.
offset : float
    Constant energy offset.
qubit_labels : dict&#91;int, str&#93;
    Index → label mapping.
n_qubits : int
    Total logical qubits.
source : str
    Origin description.

- **energy**(spins)
  - Compute Ising energy for a spin configuration.

### Class `QUBOModel`
QUBO model: min x^T Q x.

Attributes
----------
Q : dict&#91;tuple&#91;int, int&#93;, float&#93;
    QUBO matrix entries. Diagonal = linear, off-diagonal = quadratic.
offset : float
    Constant energy offset.
qubit_labels : dict&#91;int, str&#93;
    Index → label mapping.
n_qubits : int
    Total logical qubits.
source : str
    Origin description.

- **energy**(bits)
  - Compute QUBO energy for a binary configuration.
- **to_ising**()
  - Convert QUBO to Ising model.

### Class `SCToIsing`
Compile SC network adjacency matrices into Ising models.

Maps SC populations to qubits and projections to couplings.
Excitatory connections → ferromagnetic (J < 0, favoring alignment).
Inhibitory connections → antiferromagnetic (J > 0, favoring anti-alignment).

Parameters
----------
coupling_scale : float
    Multiplier applied to connection weights (default 1.0).
field_scale : float
    Multiplier for external field from bias (default 0.1).

- **__init__**(coupling_scale, field_scale)
- **compile**(adjacency, node_labels, biases, name)
  - Compile adjacency matrix into an Ising model.

### Class `SCToQUBO`
Compile SC network into QUBO formulation.

Parameters
----------
penalty : float
    Constraint penalty coefficient (default 2.0).

- **__init__**(penalty)
- **compile**(adjacency, node_labels, name)
  - Compile adjacency matrix into a QUBO model.

### Class `SimulatedAnnealer`
Classical simulated annealing solver for Ising/QUBO models.

Implements the Metropolis-Hastings algorithm with exponential
temperature schedule.

Parameters
----------
n_sweeps : int
    Number of Monte Carlo sweeps (default 1000).
beta_start : float
    Initial inverse temperature (default 0.1).
beta_end : float
    Final inverse temperature (default 10.0).
seed : int
    Random seed.

- **__init__**(n_sweeps, beta_start, beta_end, seed)
- **solve_ising**(model, num_reads)
  - Solve an Ising model via simulated annealing.
- **_solve_ising_rust**(model, num_reads)
  - Rust-accelerated SA path.
- **_solve_ising_python**(model, num_reads)
  - Pure-Python SA fallback.
- **solve_qubo**(model, num_reads)
  - Solve a QUBO model via simulated annealing.

### Class `DWaveInterface`
Interface to D-Wave quantum annealer via Ocean SDK.

Wraps ``DWaveSampler`` + ``EmbeddingComposite`` for transparent
minor-embedding. Falls back to simulated annealing if no QPU
is available.

Parameters
----------
chain_strength : float
    Chain strength for embedding (default 2.0).
num_reads : int
    Number of QPU reads (default 1000).
annealing_time_us : float
    Annealing time in microseconds (default 20.0).

- **__init__**(chain_strength, num_reads, annealing_time_us)
- **available**()
  - Whether D-Wave SDK is available.
- **solve_ising**(model)
  - Submit Ising model to D-Wave QPU.

### Class `EnergyLandscape`
Analyze the energy landscape of an Ising model.

Computes energy statistics, degeneracy, spectral gap, and
partition function (for small models).

- **analyze**(model, samples)
  - Run landscape analysis.
- **_enumerate_all**(n)
  - Enumerate all 2^n spin configurations.

### Class `EmbeddingAnalyzer`
Analyze embedding requirements for D-Wave hardware.

Computes logical-to-physical qubit ratios, chain length
statistics, and connectivity requirements.

- **analyze**(model)
  - Analyze embedding requirements.

### Class `HardwareGraph`
D-Wave hardware graph topology model.

Generates adjacency structure for Chimera, Pegasus, and Zephyr
topologies to enable embedding feasibility analysis.

Parameters
----------
topology : str
    One of ``chimera``, ``pegasus``, ``zephyr``.
size : int
    Topology size parameter (M for Chimera M×M×4,
    M for Pegasus P(M), M for Zephyr Z(M)).

- **__init__**(topology, size)
- **n_physical_qubits**()
  - Total physical qubits in this hardware graph.
- **connectivity**()
  - Per-qubit connectivity.
- **can_embed**(model)
  - Check whether a model can be embedded on this hardware.

### Class `ChainBreakResolver`
Post-process D-Wave samples to repair broken chains.

When a logical qubit is embedded as a chain of physical qubits,
some physical qubits in the chain may disagree. This class
resolves disagreements using majority vote or energy minimization.

Parameters
----------
method : str
    Resolution method: ``majority_vote`` or ``minimize_energy``.

- **__init__**(method)
- **resolve**(physical_samples, chains, model)
  - Resolve chain breaks in physical samples.
- **analyze_breaks**(physical_samples, chains)
  - Analyze chain break statistics.

### Class `AnnealingSchedule`
Custom annealing schedule builder for D-Wave.

Supports linear, pause-and-quench, and reverse annealing
protocols.

The schedule is a list of (time_us, s) points where s ∈ &#91;0, 1&#93;
is the anneal fraction (0 = transverse field dominant,
1 = problem Hamiltonian dominant).

- **__init__**()
- **linear**(duration_us)
  - Standard linear anneal from s=0 to s=1.
- **pause_and_quench**(ramp_time_us, pause_at_s, pause_duration_us, quench_time_us)
  - Pause-and-quench: ramp to s, hold, then quench to s=1.
- **reverse**(initial_s, reverse_to_s, ramp_time_us, hold_time_us, forward_time_us)
  - Reverse annealing: start at s=1, go back, then forward.
- **points**()
  - Schedule points as &#91;(time_us, s), ...&#93;.
- **total_time_us**()
  - Total annealing time in microseconds.
- **to_dict**()
  - Export schedule as dict for D-Wave API.

### Class `GaugeTransform`
Random gauge transformations for improved sampling.

Applies random spin-flip transformations (g_i ∈ {+1, -1}) to the
Ising model: h'_i = g_i · h_i, J'_ij = g_i · g_j · J_ij.
This breaks systematic QPU biases without changing the energy
landscape.

Parameters
----------
n_gauges : int
    Number of gauge transforms to apply (default 10).
seed : int
    Random seed.

- **__init__**(n_gauges, seed)
- **transform**(model)
  - Generate gauge-transformed copies of the model.
- **untransform_sample**(sample, gauge)
  - Undo gauge transform on a sample.

### Class `SCBitstreamQUBO`
SC-specific QUBO formulations for bitstream optimization.

Provides problem-specific encodings for common SC optimization
tasks:
- **Weight optimization**: Find binary weight mask that minimizes
  network error.
- **Pruning**: Select minimal subset of connections preserving
  accuracy.
- **Topology search**: Binary selection of connections from a
  candidate set.

Parameters
----------
penalty : float
    Constraint violation penalty (default 5.0).

- **__init__**(penalty)
- **weight_optimization**(target_output, candidate_weights, n_bits)
  - Formulate weight optimization as QUBO.
- **pruning**(adjacency, importance_scores, max_connections)
  - Formulate network pruning as QUBO.

### Class `SampleAggregator`
Post-process and aggregate quantum annealing samples.

Provides filtering, deduplication, energy histogram, and
Boltzmann-weighted statistics.

- **aggregate**(samples, energies, temperature)
  - Aggregate and analyze sample set.

### Class `SCPrecisionEncoder`
Encode SC probability values as qubit configurations.

SC values are continuous probabilities in &#91;0, 1&#93;. Quantum
annealers operate on binary variables. This encoder provides
three strategies for mapping SC precision to qubits:

- **binary**: k qubits encode 2^k levels (compact but coupled)
- **unary**: k qubits encode k+1 levels (robust but expensive)
- **one_hot**: k qubits encode k levels (good for categorical)

Parameters
----------
encoding : str
    One of ``binary``, ``unary``, ``one_hot``.
n_bits : int
    Number of qubits per SC value (default 8).

- **__init__**(encoding, n_bits)
- **n_levels**()
  - Number of representable precision levels.
- **encode**(sc_value)
  - Encode an SC probability as qubit configuration.
- **decode**(qubits)
  - Decode qubit configuration back to SC probability.
- **qubits_needed**(n_sc_values)
  - Total qubits needed to encode n SC values.
- **encode_array**(values)
  - Encode array of SC values into a single qubit dict.

### Class `ProblemDecomposer`
Decompose large QUBO/Ising into sub-problems for QPU.

When a model exceeds QPU capacity, this class partitions it
into smaller sub-problems that fit on hardware, solves each,
then merges the results.

Parameters
----------
max_subproblem_size : int
    Maximum qubits per sub-problem (default 64 for Chimera unit cell).
overlap : int
    Number of shared qubits between partitions (default 4).
n_iterations : int
    Number of decomposition-merge iterations (default 10).

- **__init__**(max_subproblem_size, overlap, n_iterations)
- **decompose**(model)
  - Partition Ising model into sub-problems.
- **solve_decomposed**(model, solver)
  - Decompose, solve sub-problems, and merge.

### Class `TTSAnalyzer`
Time-to-solution quality metric for quantum annealing.

TTS measures the total time required to find the ground state
with probability p_target, given:
- p_success: probability of finding ground state in a single run
- t_anneal: time per annealing run

TTS = t_anneal × (log(1 - p_target) / log(1 - p_success))

This is the standard benchmark metric used in D-Wave literature.

- **compute**(p_success, t_anneal_us, p_target)
  - Compute TTS metric.
- **from_samples**(energies, ground_state_energy, t_anneal_us, tolerance, p_target)
  - Compute TTS from a set of sample energies.
- **compare_solvers**(results, ground_state_energy, tolerance)
  - Compare TTS across multiple solvers.

### Function `export_ising_json(model, path)`
Export Ising model to JSON format.

Parameters
----------
model : IsingModel
    The model to export.
path : str
    Output file path.

### Function `export_qubo_json(model, path)`
Export QUBO model to JSON format.

### Function `export_bqm(model)`
Export Ising model as a dimod BinaryQuadraticModel.

Returns
-------
dimod.BinaryQuadraticModel or None
    BQM object, or None if dimod is not installed.

### Function `visualize_ising(model)`
Generate ASCII visualization of an Ising model.

Returns
-------
str
    Multi-line ASCII representation.

---

## Module `chaos.rng`

### Class `ChaoticRNG`
Logistic-map chaotic RNG for SC bitstream generation.

x_{n+1} = r * x_n * (1 - x_n)

At r=4.0 the logistic map is fully chaotic with Lyapunov exponent
ln(2) ~ 0.693 and an invariant density Beta(0.5, 0.5) on (0, 1).
The 100-step burn-in discards transients from the initial condition.

Parameters
----------
r : float
    Bifurcation parameter. Must be in (3.57, 4.0&#93; for chaos.
    Default 4.0 gives maximal chaos.
x : float
    Initial condition in (0, 1). Avoid 0.0, 0.5, 1.0 exactly
    (these are fixed/periodic points at r=4).
burn_in : int
    Steps to discard before first output.

Example
-------
>>> rng = ChaoticRNG(r=4.0, x=0.37)
>>> bits = rng.generate_bitstream(p=0.5, length=1000)
>>> 0.4 < bits.mean() < 0.6
True

- **__post_init__**()
- **random**(size)
  - Generate *size* chaotic floats in (0, 1).
- **random_vectorized**(size, n_maps)
  - Generate samples from *n_maps* independent logistic maps in parallel.
- **generate_bitstream**(p, length)
  - Generate SC bitstream where P(bit=1) ~ *p*.
- **lyapunov_exponent**(n_steps)
  - Estimate the maximal Lyapunov exponent via derivative averaging.
- **shannon_entropy**(n_samples, n_bins)
  - Estimate Shannon entropy of the chaotic sequence in bits.
- **autocorrelation**(n_samples, max_lag)
  - Compute autocorrelation of the chaotic sequence up to *max_lag*.
- **reset**(x)
  - Reset to initial condition (with fresh burn-in).
- **state**()
  - Current internal state.

### Class `TentMapRNG`
Tent map chaotic RNG — piecewise linear alternative to logistic map.

x_{n+1} = mu * min(x_n, 1 - x_n)

At mu=2.0 the tent map is topologically conjugate to the logistic map
at r=4.0 but has uniform invariant density on (0, 1) — better for
SC bitstream generation where uniform marginals are desired.

Parameters
----------
mu : float
    Slope parameter. Must be in (1, 2&#93; for chaos. Default 2.0.
x : float
    Initial condition in (0, 1).

- **__post_init__**()
- **_step**(s)
- **random**(size)
  - Generate *size* chaotic floats in (0, 1).
- **generate_bitstream**(p, length)
  - Generate SC bitstream where P(bit=1) ~ *p*.
- **reset**(x)
  - Reset to initial condition (with fresh burn-in).
- **state**()

---

## Module `chip_compiler.chip_spec`

### Class `CoreSpec`
Specification for one neuromorphic core.


### Class `ChipSpec`
Full neuromorphic chip specification.

Parameters
----------
name : str
    Chip identifier (e.g., 'loihi2', 'xylo', 'akida').
vendor : str
total_cores : int
core : CoreSpec
    Per-core specification (assumes homogeneous cores).
clock_mhz : float
power_mw_per_core : float
    Estimated dynamic power per active core.
routing_topology : str
    'mesh', 'crossbar', 'tree', 'ring'
max_fan_out : int
    Maximum outgoing connections per neuron.
analog_noise_cv : float
    Coefficient of variation for analog process variation.
    0.0 for fully digital chips.

- **total_neurons**()
- **total_power_mw**()
- **fits**(n_neurons, max_fan_out)
  - Check if a network fits on this chip.
- **cores_needed**(n_neurons)
  - Minimum cores needed for N neurons.

### Function `load_chip_spec(path)`
Load and validate a chip spec from a JSON file.

### Function `_validate_chip_payload(payload)`
### Function `_validate_core_payload(value)`
### Function `_validate_key_set(payload)`
### Function `_required_str(payload, key, source)`
### Function `_required_bool(payload, key, source)`
### Function `_required_str_list(payload, key, source)`
### Function `_required_positive_int(payload, key, source)`
### Function `_required_non_negative_int(payload, key, source)`
### Function `_required_int(payload, key, source)`
### Function `_required_positive_float(payload, key, source)`
### Function `_required_non_negative_float(payload, key, source)`
### Function `_required_float(payload, key, source)`
---

## Module `chip_compiler.compiler`

### Class `CoreMapping`
Mapping of neurons to one chip core.


### Class `CompilationResult`
Result of compiling an SNN to a chip target.

- **summary**()

### Function `compile_for_chip(layer_sizes, weights, neuron_types, target)`
Compile an SNN to a target neuromorphic chip.

Parameters
----------
layer_sizes : list of (n_inputs, n_neurons)
weights : list of ndarray, optional
    Weight matrices per layer. If provided, will be quantized.
neuron_types : list of str, optional
    Neuron type per layer (e.g., 'LIF', 'Izhikevich').
target : str or ChipSpec
    Target chip name or spec.

Returns
-------
CompilationResult

---

## Module `chiplet.chiplet_gen`

### Class `InterposerTech`

### Class `InterposerLink`
Timing model for a die-to-die link.

- **__post_init__**()
- **from_tech**(cls, src, dst, tech)
  - Create link with technology-specific defaults.
- **latency_cycles**()
  - Latency in clock cycles at 200 MHz (5 ns period).
- **fifo_depth_log2**()
  - Minimum async FIFO depth (log2) to absorb jitter.

### Class `ChipletDie`
One die in a chiplet package.

- **clock_period_ns**()

### Class `ChipletTopology`
Directed graph of dies + interposer links.

- **add_die**(die)
- **add_link**(link)
- **mesh_2d**(cls, rows, cols, tech)
  - Create a 2D mesh topology.
- **ring**(cls, n_dies, tech)
  - Create a ring topology.
- **star**(cls, n_dies, tech)
  - Create a star topology with die 0 as the hub.
- **get_links_from**(die_id)
- **get_links_to**(die_id)
- **get_die**(die_id)
- **num_dies**()

### Class `RoutingEntry`
One routing table entry: source neuron → target die + neuron.


### Class `RoutingTable`
Per-die AER routing table for inter-die communication.

- **add_route**(src, dst_die, dst_neuron, weight)
- **routes_to_die**(target_die)
- **num_entries**()
- **target_dies**()

### Class `PackageEnergyReport`
Energy report for the full chiplet package.

- **total_nj**()

### Class `CongestionReport`
Link utilisation analysis.


### Class `ChipletOutput`
Generated output files for one chiplet topology.

- **to_dict**()

### Class `ChipletGenerator`
Generates multi-die routing RTL from a chiplet topology.

- **generate**(topology, routing)
- **_emit_die_wrapper**(die, topo)
- **_emit_bridge**(link, decor_seed)
- **_emit_routing_table**(table, die)
- **_emit_top**(topo)
- **_emit_constraints**(topo)

### Class `TimingSimResult`
Result of interposer timing simulation.


### Class `CDCConfig`
Per-link CDC configuration for heterogeneous clock domains.

- **ratio**()
- **is_mesochronous**()

### Class `DieThermal`
Per-die thermal properties: power source, capacity, ambient path.

`power_mw` and `temperature_c` are runtime state; the rest are
geometry / packaging properties. The dataclass holds the data
consumed by the package-level conductance-matrix solver below.

- **is_throttled**()

### Class `PackageThermalReport`
Steady-state + transient thermal report for the full package.


### Class `LinkProtection`
ECC/CRC configuration for a die-to-die link.

- **__post_init__**()
- **effective_bandwidth_ratio**()

### Class `CreditConfig`
Per-link credit-based flow control parameters.

- **__post_init__**()
- **buffer_flits**()
- **credit_width**()

### Class `StackingType`

### Class `TSVLink`
Through-silicon via (TSV) link for 3D stacking.

- **latency_ns**()
- **bandwidth_gbps**()

### Class `PowerDomain`
Voltage island / power domain for a subset of dies.

- **__post_init__**()
- **is_gated**()
- **die_mask**()

### Class `PowerDomainMap`
Maps dies to power domains for isolation/gating.

- **add_domain**(domain)
- **domain_for_die**(die_id)
- **active_dies**()
- **gated_dies**()

### Class `PartitionAssignment`
Maps neurons/layers to dies from the hierarchical partitioner.

- **assign**(neuron_id, die_id)
- **neurons_on_die**(die_id)
- **to_routing_tables**(connectivity)
  - Convert partition + connectivity to per-die routing tables.

### Function `compute_decorrelation_seeds(topology)`
Assign independent LFSR seeds per link for bitstream decorrelation.

Uses golden-ratio hashing to ensure maximal separation between
LFSR sequences across dies (avoids correlated bitstreams).

### Function `_require_sv_identifier(value, field_name)`
Validate a generated SystemVerilog identifier fragment.

### Function `link_energy_pj(link, bits)`
Estimate total energy (pJ) for transmitting 'bits' over a link.

### Function `estimate_package_energy(topology, bits_per_link)`
Estimate total communication energy for one inference cycle.

### Function `estimate_congestion(topology, routing_tables, events_per_cycle)`
Estimate per-link utilisation given routing tables and traffic.

Utilisation = (events × data_width) / (bandwidth × 1e9 × period_ns × 1e-9).
A link with utilisation > 1.0 is over-saturated.

### Function `find_disjoint_paths(topology, src_die, dst_die, max_paths)`
Find up to max_paths link-disjoint paths between two dies.

Uses iterative BFS with link exclusion to find alternative routes
for fault-tolerant routing.

### Function `_bfs_path(topology, src, dst, excluded)`
BFS shortest path avoiding excluded links.

### Function `simulate_timing(topology, src_die, dst_die)`
BFS shortest-path timing simulation between two dies.

### Function `make_torus(rows, cols, tech)`
Create a 2D torus topology (mesh with wrap-around edges).

### Function `compute_cdc_configs(topology)`
Auto-compute per-link CDC configs from die clock frequencies.

### Function `_build_conductance_matrix(topology, die_state)`
Build (G, g_amb, die_id_order) for the thermal network.

G is the off-diagonal conductance matrix (W/K) for inter-die
couplings. g_amb is the per-die conductance to ambient.
die_id_order is the row → die_id mapping used by the solver.

For each link (i, j) in `topology.links` with technology
`tech`, the bond resistance is
    R_bond(i,j) = R_THERMAL&#91;tech&#93; + R_spread(i) + R_spread(j)
so the off-diagonal conductance is
    G&#91;i,j&#93; = G&#91;j,i&#93; = 1 / R_bond(i,j)

Per-die ambient conductance:
    g_amb&#91;i&#93; = 1 / R_to_ambient(i)

### Function `_solve_steady_state(G_off, g_amb, p_w, t_amb_c)`
Solve `(D - G_off) · T = P + g_amb · T_amb` for T.

where D is the diagonal of row-sums (Kirchhoff's current law
for the thermal network).

Returns the per-die steady-state temperature in °C.

### Function `_solve_transient(G_off, g_amb, capacities, p_w, t_amb_c, initial_t_c, dt_s, n_steps)`
Implicit-Euler integration of `C · dT/dt = -A · T + b`.

A and b are the same matrices as in `_solve_steady_state`.
Implicit Euler is unconditionally stable for the stiff
thermal system and converges to the steady-state solution
as t → ∞.

Per step: `(C/dt + A) · T_new = C/dt · T_old + b`.

Returns array of shape (n_steps, n_dies) with the temperature
trajectory.

### Function `simulate_thermal(topology, power_per_die_mw, ambient_c)`
Solve the full chiplet thermal network.

Builds a per-die conductance matrix from `topology.links` (using
per-technology thermal resistance) plus a per-die ambient path.
The steady-state temperature is the solution to a linear system;
a transient response can also be requested with implicit-Euler
time-stepping.

Parameters
----------
topology : ChipletTopology
    Dies + interposer links.
power_per_die_mw : dict, optional
    Per-die power dissipation in mW. Defaults to 100 mW per
    die when missing.
ambient_c : float
    Ambient air temperature in °C (default 25).
die_state : dict, optional
    Per-die thermal property overrides. When omitted, defaults
    from `DieThermal` are used (10×10 mm² silicon die, 1.5 K/W
    junction-to-ambient).
transient_steps : int
    If > 0, also compute a transient response of this many
    steps. The trajectory and time array are placed in the
    returned report.
transient_dt_s : float
    Time step (s) for transient integration. Default 1 ms is
    appropriate for ~0.08 J/K dies under ~0.1 W loading
    (thermal time constant ~0.1 s).

### Function `adaptive_route(topology, src_die, dst_die, congestion, congestion_threshold)`
Congestion-reactive routing: avoid saturated links.

Uses BFS but excludes links with utilisation above the threshold.
Falls back to shortest path if no uncongested route exists.

### Function `emit_crc32_sv(data_width)`
Emit IEEE 802.3 CRC-32 feedback logic for link error detection.

### Function `bandwidth_aware_route(topology, src_die, dst_die, required_gbps)`
Find a path where all links have bandwidth >= required_gbps.

### Function `emit_credit_controller_sv(config, link_name)`
Emit saturating credit-based flow control for a die-to-die link.

### Function `add_3d_stack(topology, bottom_die, top_die, stacking)`
Add a vertical (3D) link between stacked dies.

### Function `emit_power_gating_sv(domain)`
Emit sequenced isolation and switch control for a voltage island.

---

## Module `chiplet.hierarchical_partitioner`

### Class `HierarchyLevel`

### Class `CSRGraph`
Compressed Sparse Row graph for billion-neuron scale.

O(1) adjacency access per vertex, O(E) total memory.

- **from_edge_list**(cls, num_vertices, edges, vertex_weights)
  - Build CSR from edge list (symmetric: adds both directions).
- **neighbors**(v)
- **degree**(v)
- **edge_conn**(v)
- **edge_scc**(v)
- **num_edges**()

### Class `CorrelationEdge`
An edge with both connection weight and SC correlation weight.


### Class `CorrelationAwareGraph`
Adjacency representation with per-edge SCC weights.

Edge lookups (`edge_weight`, `edge_scc`) are O(1) via a cached
`(min_uv, max_uv) → CorrelationEdge` dict, lazily built on first
access. Was O(E) per call (linear scan), which made the
partitioner O(V²·E) on V vertices — see commit notes for #65.

- **_ensure_edge_cache**()
- **adjacency**()
- **edge_weight**(u, v)
- **edge_scc**(u, v)
- **num_edges**()
- **to_csr**()
  - Convert to CSR representation.

### Class `LFSRSeedAllocator`
Assigns independent LFSR seeds per partition.

Uses co-prime spacing to ensure maximal-length LFSR sequences
do not overlap between partitions.

- **__init__**(base_seed)
- **allocate**(num_partitions)
  - Return a list of unique, well-separated LFSR seeds.
- **verify_uniqueness**(seeds)

### Class `HierarchicalPartitioner`
Multi-level graph partitioner with correlation awareness.

- **__init__**(num_partitions, coarsen_threshold, kl_iterations, correlation_penalty, seed, refine_backend)
- **partition**(graph)
  - Partition the graph. Returns (partitions, seeds).
- **_dispatch_refine**(partitions, adj, graph)
  - Backend dispatch for the KL refinement step.
- **_recursive_bisect**(vertices, adj, graph, k)
  - Recursively bisect until we have k partitions.
- **_coarsen**(vertices, adj, graph)
  - Heavy-edge matching coarsening (merge low-SCC edges first).
- **_uncoarsen**(partition, mapping)
  - Expand coarsened partition back to original vertices.
- **_spectral_bisect**(vertices, adj, graph)
  - Spectral-heuristic bisection with correlation penalty.
- **_encode_csr**(partitions, adj, graph)
  - Pack the per-partition state into the flat CSR-style buffers
- **_decode_part_map**(part_map, n_parts)
  - Decode flat part_map&#91;V&#93; back into List&#91;List&#91;int&#93;&#93;.
- **_refine_rust**(partitions, adj, graph)
  - Rust dispatch for `_refine` — bit-exact parity with Python.
- **_refine_julia**(partitions, adj, graph)
  - Julia dispatch — bit-exact parity with Python + Rust.
- **_refine_go**(partitions, adj, graph)
  - Go dispatch via cgo + ctypes — bit-exact parity.
- **_refine_mojo**(partitions, adj, graph)
  - Mojo dispatch via raw-Int-addr ctypes — bit-exact parity.
- **_refine**(partitions, adj, graph)
  - Kernighan-Lin inspired local refinement.
- **_per_partition_cost**(v, n_parts, part_map, adj, graph)
  - Length-`n_parts` cost vector: `costs&#91;p&#93;` = cost of placing
- **_boundary_cost**(v, partition_id, part_map, adj, graph)
  - Cost of placing vertex v in partition_id (legacy single-target API).
- **repartition_incremental**(graph, partitions, max_moves)
  - Incremental repartitioning: migrate high-cost boundary vertices.

### Class `GhostCellManager`
Computes halo/ghost regions for boundary communication.

Ghost cells are copies of neurons on neighboring partitions that
a partition needs to read but not write.

- **compute_halos**(graph, partitions)
  - Return {partition_id: set of ghost vertex IDs needed}.
- **halo_sizes**(graph, partitions)
  - Return {partition_id: number_of_ghost_cells}.

### Class `BoundarySyncConfig`
Configuration for boundary synchronization.


### Class `BoundarySyncProtocol`
Manages decorrelation at partition boundaries.

Each boundary edge gets a decorrelation buffer (XOR with independent
LFSR seed) to prevent correlation blow-up at partition interfaces.

- **__init__**(config)
- **init_buffers**(graph, partitions, seeds)
  - Initialise decorrelation buffers at boundary edges.
- **check_scc_budget**(graph, partitions)
  - Check which boundary edges exceed SCC budget.
- **num_buffers**()

### Class `LoadMetrics`
Per-partition load metrics.


### Class `MigrationRecommendation`
Recommendation to migrate a vertex between partitions.


### Class `CorrelationLoadBalancer`
Runtime load balancer with SCC awareness.

Monitors per-partition load imbalance and boundary correlation,
and generates migration recommendations.

- **__init__**(imbalance_threshold, scc_weight)
- **compute_load_metrics**(graph, partitions)
  - Compute load metrics for each partition.
- **recommend_migrations**(graph, partitions, max_recommendations)
  - Generate migration recommendations.

### Class `RankMapper`
Maps partitions to MPI ranks with topology awareness.

- **__init__**(num_ranks, hierarchy)
- **assign**(partitions, graph)
  - Assign partition_id → rank.
- **cross_rank_edges**(graph, partitions)
  - Count edges that cross MPI rank boundaries.

### Class `PartitionReport`
Report from a partitioning run.

- **summary**()

### Function `_ensure_julia_kl_refine_loaded()`
Lazy-load the Julia KL refine module on first use.

### Function `_ensure_go_kl_refine_loaded()`
Lazy-load the Go KL refine shared library on first use.

### Function `_ensure_mojo_kl_refine_loaded()`
Lazy-load the Mojo KL refine shared library on first use.

### Function `_build_part_map(partitions)`
### Function `calculate_edge_cut(graph, partitions)`
Count cross-partition edges.

### Function `calculate_boundary_scc(graph, partitions)`
Maximum SCC on boundary edges.

### Function `calculate_mean_boundary_scc(graph, partitions)`
Mean SCC on boundary edges.

### Function `calculate_total_boundary_scc(graph, partitions)`
Total SCC on boundary edges.

### Function `calculate_imbalance_ratio(partitions)`
Imbalance ratio: max_size / ideal_size - 1.

0.0 = perfect balance, >0.0 = imbalanced.

### Function `calculate_comm_volume(graph, partitions, bytes_per_spike, bitstream_length)`
Estimate MPI communication volume.

### Function `build_partition_report(graph, partitions, seeds, scc_budget)`
Build a complete partition report.

---

## Module `cli`

### Class `_OutputAction`
Track whether ``--output`` was supplied explicitly.

- **__call__**(parser, namespace, values, option_string)

### Function `main()`
### Function `_cmd_compile_nir(args)`
Compile NIR/ONNX model to Verilog RTL artefacts.

### Function `_scnir_signal_kind_counts(document)`
### Function `_scnir_signal_routes(document)`
### Function `_scnir_hierarchy_port_count(document)`
### Function `_cmd_compile(args)`
Compile ODE equation string to Verilog RTL + optional synthesis.

Supports three compilation modes via CLI flags:

1. **Standard** (default): combinational datapath at the configured
   precision (``--data-width`` / ``--fraction``).
2. **Pipelined** (``--pipeline auto|N``): insert register stages at
   multiply outputs for high-frequency targets.  ``auto`` uses
   ``critical_path_depth()`` + ``pipeline_stages_needed()`` from
   ``static_analysis.py``.  ``--pipeline-points`` selects individual
   signals to register.
3. **Adaptive precision** (``--adaptive-precision``): generate a
   dual-datapath module with LP and HP sub-modules, hysteresis-based
   precision switching, and clock gating.  Configure LP/HP widths via
   ``--lp-width``, ``--lp-frac``, ``--hp-width``, ``--hp-frac``.

### Function `_cmd_serve(model_path, port, dt)`
Start streaming inference server.

### Function `_cmd_info()`
### Function `_cmd_collect_synthesis(args)`
Collect synthesis reports into optimiser evidence JSON.

### Function `_cmd_scnir(args)`
Validate or export SC-aware NIR metadata documents.

### Function `_cmd_formal(args)`
Compile and replay network-level formal verification artefacts.

### Function `_parse_antagonistic_pair(value)`
### Function `_parse_temporal_separation(value)`
### Function `_parse_population_silence(value)`
### Function `_print_optional_dependency_version(module_name, label)`
### Function `_format_engine_status(expected_version)`
### Function `_safe_simd_tier(engine)`
### Function `_cmd_benchmark()`
### Function `_cmd_deploy(model_path, target, output_dir, dt, bitstream_length)`
Deploy a model to FPGA or browser artefacts.

### Function `_cmd_map_nir(model_path, output_dir, hardware_targets, dt, bitstream_length)`
Generate deterministic silicon-mapping reports for a NIR graph.

### Function `_cmd_hub_init(output_dir, port, bind_host, image, offline)`
Generate a local self-hosted hub Compose bundle.

### Function `_auto_synthesize(output_dir, target, top_module, cfg)`
Run Yosys synthesis automatically if yosys is installed. Returns True on success.

### Function `_generate_project(output_dir, target, top_module)`
### Function `_cmd_studio(port)`
Launch the Visual SNN Design Studio (Equation Playground).

### Function `_cmd_preflight()`
---

## Module `comm.aer_udp`

### Class `AEREvent`
Single AER spike event.


### Class `AERSender`
Send AER spike events over UDP.

Parameters
----------
host : str
    Destination IP address.
port : int
    Destination UDP port.

- **__init__**(host, port)
- **send**(events)
  - Send a batch of AER events. Returns number of packets sent.
- **send_spikes**(spike_vector, timestamp)
  - Convert a binary spike vector to AER events and send.
- **close**()

### Class `AERReceiver`
Receive AER spike events over UDP.

Parameters
----------
host : str
    Bind address.
port : int
    Listen port.
timeout : float
    Socket timeout in seconds.

- **__init__**(host, port, timeout)
- **receive**()
  - Receive one packet of AER events. Returns empty list on timeout.
- **receive_as_vector**(n_neurons)
  - Receive and convert to binary spike vector.
- **close**()

---

## Module `compiler.adaptive_precision`

### Class `LayerPrecision`
Bitstream length assignment for one layer.


### Class `SynapsePrecision`
Precision assignment and conservative error bound for one synapse.

- **to_dict**()
  - Return a JSON-serialisable precision-plan row.

### Function `analyze_sensitivity(layer_weights, lengths, n_trials, seed)`
Measure per-layer sensitivity to bitstream length reduction.

For each layer, compute mean output error across trial inputs
when reducing bitstream length from max to min. Layers with high
sensitivity need longer bitstreams.

Parameters
----------
layer_weights : list of ndarray
    Weight matrices for each layer.
lengths : list of int
    Bitstream lengths to sweep (default: &#91;32, 64, 128, 256, 512, 1024&#93;).
n_trials : int
    Number of random input trials.
seed : int
    Random seed.

Returns
-------
list of float
    Per-layer sensitivity scores (higher = needs longer bitstream).

### Function `assign_synapse_precisions(layer_weights, layer_names, sensitivity_maps, target_error, min_bits, max_bits, min_length, max_length, confidence)`
Assign per-synapse bit widths and SC lengths with error bounds.

The bound is intentionally conservative: each synapse gets a local target
derived from the global target, quantisation error is bounded by half an
integer-quantisation step scaled by sensitivity, and stochastic sampling
error uses the existing Hoeffding bitstream-length helper.

### Function `precision_plan_manifest(assignments)`
Build a deterministic manifest for a per-synapse precision plan.

### Function `_select_bit_width(sensitivity, target, min_bits, max_bits)`
### Function `_quantization_error_bound(sensitivity, bits)`
### Function `_hoeffding_radius(length, confidence)`
### Function `assign_lengths(layer_weights, layer_names, total_budget, min_length, max_length, target_error, method)`
Assign per-layer bitstream lengths under a total budget.

Parameters
----------
layer_weights : list of ndarray
    Weight matrices for each layer.
layer_names : list of str, optional
    Human-readable layer names.
total_budget : int, optional
    Total bitstream cycles budget. If None, each layer gets its own
    minimum length for target_error.
min_length, max_length : int
    Bounds on per-layer bitstream length.
target_error : float
    Target per-layer accuracy (probability tolerance).
method : str
    'hoeffding' uses Hoeffding bound, 'sensitivity' uses empirical sweep.

Returns
-------
list of LayerPrecision
    Per-layer bitstream length assignments.

---

## Module `compiler.adaptive_runtime_precision`

### Function `_validate_lp_hp(lp_width, lp_frac, hp_width, hp_frac)`
Validate that the LP/HP pair is sensible.

### Function `compile_adaptive_precision(neuron, module_name, lp_width, lp_frac, hp_width, hp_frac)`
Compile an EquationNeuron to dual-datapath adaptive-precision Verilog.

Generates two complete neuron datapaths (LP and HP).  HP is emitted as
the authoritative output path, while LP drives only precision telemetry.

Parameters
----------
neuron : EquationNeuron
    The neuron defined by arbitrary ODE strings.
module_name : str
    Name of the generated Verilog module.
lp_width : int
    Low-precision total bit width (default 16).
lp_frac : int
    Low-precision fractional bits (default 8).
hp_width : int
    High-precision total bit width (default 32).
hp_frac : int
    High-precision fractional bits (default 16).
threshold_up_pct : float
    Fraction of LP range at which to switch to HP (default 0.8). Must satisfy
    0 < threshold_down_pct < threshold_up_pct < 1, and both thresholds must
    quantise to non-zero LP code points.
threshold_down_pct : float
    Fraction of LP range at which to switch back to LP (default 0.5). Must be
    strictly lower than threshold_up_pct and quantise on LP codes.
signed : bool
    True for signed two's complement.
overflow : str
    Overflow mode for both datapaths.
rounding : str
    Rounding mode for both datapaths.

Returns
-------
str
    Synthesisable Verilog source with HP-authoritative dual datapaths.

---

## Module `compiler.deployment`

### Class `ResourceEstimate`
Estimated FPGA resource usage.

Attributes
----------
luts : int
    Estimated look-up tables.
ffs : int
    Estimated flip-flops (registers).
dsps : int
    Estimated DSP blocks.
brams : int
    Estimated block RAMs.
mul_count : int
    Number of multiplications in the design.
add_count : int
    Number of additions/subtractions.
reg_bits : int
    Total register bits.


### Class `SLRPlacement`
SLR (Super Logic Region) placement for multi-die FPGAs.

Attributes
----------
module_name : str
    Module or instance name.
slr : int
    Target SLR index (0-based).
pblock_name : str
    Vivado PBLOCK name (auto-generated if empty).

- **__post_init__**()
  - Auto-generate pblock name if not set.

### Class `CertificationItem`
A single certification evidence item.

Attributes
----------
req_id : str
    Requirement identifier (e.g. ``"REQ-001"``).
description : str
    Requirement description.
design_ref : str
    Design artifact (e.g. Verilog module name).
verification_ref : str
    Verification artifact (e.g. SVA property, Cocotb test).
status : str
    ``"PASS"``, ``"FAIL"``, or ``"UNTESTED"``.


### Class `CompilationResult`
Per-target compilation result for comparison.

Attributes
----------
target : str
    Profile name.
verilog_lines : int
    Lines of generated Verilog.
data_width : int
    Total bit width.
fraction : int
    Fractional bits.
overflow : str
    Overflow mode.
rounding : str
    Rounding mode.
estimated_luts : int
    LUT estimate.
estimated_dsps : int
    DSP block estimate.
estimated_ffs : int
    Flip-flop estimate.
guard_bits : int
    Required guard bits.
max_freq_mhz : int | None
    Target max frequency, or None when unknown.


### Function `estimate_resources(verilog)`
Estimate FPGA resources from generated Verilog without synthesis.

Uses pattern matching on the Verilog source to count multipliers, adders,
registers, and LUTs. This is a heuristic — actual usage depends on the
synthesis tool, but estimates are within ~20% for typical designs.

Parameters
----------
verilog : str
    Generated Verilog source code.
data_width : int
    Neuron data width (for LUT estimation).
has_dsp : bool
    True if target has DSP blocks (multiplies go to DSP, not LUTs).

Returns
-------
ResourceEstimate
    Estimated resource usage.

### Function `generate_constraints(module_name)`
Generate timing constraint file for FPGA synthesis.

Parameters
----------
module_name : str
    Top-level module name.
target_freq_mhz : float
    Target clock frequency in MHz.
format : str
    ``"xdc"`` for Xilinx Vivado, ``"sdc"`` for Intel Quartus / generic.
clock_port : str
    Name of the clock input port.
reset_port : str
    Name of the reset input port.
data_width : int
    Data width for I/O delay estimation.

Returns
-------
str
    Complete constraint file content.

### Function `generate_host_driver(module_name, params)`
Generate host-side driver code for a bus-wrapped neuron.

Parameters
----------
module_name : str
    Neuron module name.
params : dict&#91;str, int&#93;
    Parameter names and bit widths.
language : str
    ``"python"`` or ``"c"``.
bus : str
    Bus protocol.
base_address : int
    Memory-mapped base address.
data_width : int
    Fixed-point data width.
fraction : int
    Fractional bits.

Returns
-------
str
    Complete driver source code.

### Function `_gen_python_driver(module_name, params, base_address, data_width, fraction)`
Generate Python MMIO driver.

### Function `_gen_c_driver(module_name, params, base_address, data_width, fraction)`
Generate C MMIO driver header.

### Function `generate_cocotb_testbench(module_name)`
Generate a Cocotb (Python) testbench for a compiled neuron.

Parameters
----------
module_name : str
    Verilog module name.
data_width : int
    Fixed-point data width.
fraction : int
    Fractional bits.
n_steps : int
    Number of simulation clock cycles.
input_current : float
    Input current value.

Returns
-------
str
    Complete Cocotb Python testbench.

### Function `generate_sby_script(module_name)`
Generate a SymbiYosys ``.sby`` formal verification script.

Enables one-command bounded model checking of compiled neurons
using open-source formal tools (SymbiYosys + Yosys + solver).

Parameters
----------
module_name : str
    Top-level Verilog module name.
sva_file : str, optional
    SystemVerilog assertions file. Defaults to ``{module}_sva.sv``.
depth : int
    BMC / induction depth in clock cycles.
mode : str
    ``"bmc"`` (bounded), ``"prove"`` (induction), ``"cover"``.
solver : str
    Solver backend (``"smtbmc"``, ``"aiger"``).
engine : str
    SMT engine (``"boolector"``, ``"z3"``, ``"yices"``).

Returns
-------
str
    Complete ``.sby`` configuration file.

### Function `generate_riscv_driver(module_name, params)`
Generate a RISC-V C driver for neuron control via MMIO.

Supports bare-metal, FreeRTOS, and Zephyr templates with timer-driven
neuron tick tasks for real-time operating system integration.

Parameters
----------
module_name : str
    Neuron module name.
params : dict&#91;str, int&#93;
    Parameter names and bit widths.
base_address : int
    MMIO base address.
data_width : int
    Fixed-point data width.
fraction : int
    Fractional bits.
rtos : str
    ``"baremetal"``, ``"freertos"``, or ``"zephyr"``.

Returns
-------
str
    Complete RISC-V C driver with optional RTOS integration.

### Function `generate_slr_constraints(placements)`
Generate Vivado XDC for multi-die SLR placement.

Emits PBLOCK constraints that pin modules to specific SLRs and
optionally adds inter-SLR pipeline register directives.

Parameters
----------
placements : list&#91;SLRPlacement&#93;
    Module-to-SLR assignments.
insert_pipeline_regs : bool
    Add register duplication directives for SLR crossings.
target_freq_mhz : float
    Target frequency for SLR crossing timing.

Returns
-------
str
    Complete XDC constraint block.

### Function `generate_certification_evidence(module_name, items)`
Generate XML traceability matrix for safety certification.

Produces a certification evidence document linking requirements to
design and verification artifacts in the format required by DO-254
(avionics), IEC 61508 (industrial), or ISO 26262 (automotive).

Parameters
----------
module_name : str
    Design module under certification.
items : list&#91;CertificationItem&#93;
    Requirement-to-evidence mapping.
standard : str
    ``"do254"``, ``"iec61508"``, or ``"iso26262"``.
dal_level : str
    Design Assurance Level or SIL/ASIL level.

Returns
-------
str
    XML certification evidence document.

### Function `compile_multi_target(equations, targets, module_name)`
Compile a neuron to multiple targets and collect metrics.

Parameters
----------
equations : dict
    Variable name → ODE RHS expression.
targets : list&#91;str&#93;
    Profile names to compile against.
module_name : str
    Base module name.

Returns
-------
list&#91;CompilationResult&#93;
    Per-target compilation results.

### Function `format_comparison_table(results)`
Format multi-target results as a markdown comparison table.

Parameters
----------
results : list&#91;CompilationResult&#93;
    Per-target compilation results.

Returns
-------
str
    Markdown table string.

---

## Module `compiler.equation_compiler`

### Class `Q88`
Fixed-point format configuration for Verilog code generation.

Supports arbitrary precision modes via ``data_width`` / ``fraction``,
with configurable signedness, overflow handling, and rounding.

============  ==========  ===============  =================  ===============
Mode          data_width  fraction         Integer range      Resolution
============  ==========  ===============  =================  ===============
**Q8.8**      16          8                &#91;-128, +127.996&#93;   1/256 ≈ 0.004
**Q4.12**     16          12               &#91;-8, +7.9998&#93;      1/4096 ≈ 0.0002
**Q16.16**    32          16               &#91;-32768, +32767&#93;   1/65536 ≈ 1.5e-5
**UQ8.8**     16          8  (unsigned)    &#91;0, +255.996&#93;      1/256 ≈ 0.004
============  ==========  ===============  =================  ===============

Overflow Modes
~~~~~~~~~~~~~~
- ``"saturate"`` — clamp to &#91;min, max&#93; (default, safest)
- ``"wrap"``     — two's complement wrap-around (Loihi 2 hardware behaviour)
- ``"trap"``     — emit ``$fatal`` assertion (DO-254 / IEC 61508 safety)

Rounding Modes
~~~~~~~~~~~~~~
- ``"truncate"``   — floor towards -∞ (default, fastest)
- ``"nearest"``    — round to nearest, ties away from zero
- ``"bankers"``    — round to nearest, ties to even (IEEE 754 default)
- ``"stochastic"`` — probabilistic rounding via LFSR (reduces long-run bias)

- **integer_bits**()
  - Number of integer bits (excluding sign bit if signed).
- **max_value**()
  - Maximum representable positive value.
- **min_value**()
  - Minimum (most negative) representable value.
- **resolution**()
  - Smallest representable non-zero step.
- **encode**(value)
  - Encode a float to unsigned Q-format integer representation.
- **encode_signed_literal**(value)
  - Encode a float as a Verilog signed decimal literal.
- **check_range**(value, label)
  - Check if a value fits in the integer range. Returns warnings.
- **precision_report**(dt, params)
  - Generate a human-readable precision diagnostics report.

### Class `_VerilogExprEmitter`
Walk a Python AST and emit equivalent Verilog fixed-point expressions.

Handles: +, -, *, /, **, unary minus, comparisons, names, constants.
Multiplications emit wide product with arithmetic right shift.

- **__init__**(state_vars, param_map, q)
  - Initialise the Verilog expression emitter.
- **_trunc**(wide_name)
  - Emit an intermediate wire for fixed-point truncation with rounding.
- **visit_BinOp**(node)
  - Emit Verilog for a binary operation (add, sub, mul, div).
- **visit_UnaryOp**(node)
  - Emit Verilog for a unary operation (negate, positive).
- **visit_Name**(node)
  - Resolve a Python name to its Verilog equivalent (register, parameter, or input).
- **visit_Constant**(node)
  - Encode a numeric constant as a Verilog signed literal in Q-format.
- **visit_Compare**(node)
  - Emit Verilog for comparison operators (>, >=, <, <=).
- **visit_Call**(node)
  - Emit Verilog for function calls (exp, log, sqrt, tanh, sigmoid, sin, cos, abs, clip, max, min).
- **_emit_lut_call**(lut_name, arg, entries)
  - Emit a 16-entry LUT indexed by top 4 bits of the input.
- **_exp_lut_entries**()
  - exp(x) for x in &#91;-8, +8) sampled at 16 points, Q8.8.
- **_log_lut_entries**()
  - log(x) for x in &#91;0.06, 8) sampled at 16 points, Q8.8.
- **_sqrt_lut_entries**()
  - sqrt(x) for x in &#91;0, 8) sampled at 16 points, Q8.8.
- **_tanh_lut_entries**()
  - tanh(x) for x in &#91;-8, +8) sampled at 16 points, Q8.8.
- **_sigmoid_lut_entries**()
  - sigmoid(x) = 1/(1+exp(-x)) for x in &#91;-8, +8), Q8.8.
- **_sin_lut_entries**()
  - sin(x) for x in &#91;-8, +8) sampled at 16 points, Q8.8.
- **_cos_lut_entries**()
  - cos(x) for x in &#91;-8, +8) sampled at 16 points, Q8.8.
- **generic_visit**(node)
  - Raise an error for any unsupported AST node type.

### Function `_emit_expr(expr_str, state_vars, param_map, q)`
Parse a Python expression string and return Verilog.

Returns (verilog_expr, intermediate_wires, mul_end, trunc_end, pipeline_regs).

### Function `compile_to_verilog(neuron, module_name, data_width, fraction)`
Compile an EquationNeuron to synthesizable Verilog RTL.

Parameters
----------
neuron : EquationNeuron
    The neuron defined by arbitrary ODE strings.
module_name : str
    Name of the generated Verilog module.
data_width : int
    Bit width for fixed-point arithmetic (default 16 = Q8.8).
fraction : int
    Number of fractional bits (default 8).
signed : bool
    True for signed two's complement (default), False for unsigned.
overflow : str
    Overflow mode: ``"saturate"`` (default), ``"wrap"``, or ``"trap"``.
rounding : str
    Rounding mode: ``"truncate"`` (default), ``"nearest"``,
    ``"bankers"``, or ``"stochastic"``.
pipeline_stages : int
    Number of pipeline register stages to insert at multiply outputs.
    0 = combinational (default). >0 = register every multiply output,
    enabling higher clock frequencies at the cost of latency.
pipeline_points : list&#91;str&#93;, optional
    Explicit list of intermediate signal names (e.g. ``&#91;"_mul0", "_mul2"&#93;``)
    where pipeline registers should be inserted.  When specified,
    only the named multiplies are registered instead of all.
    Ignored if ``pipeline_stages > 0`` (which registers all multiplies).

Returns
-------
str
    Synthesizable Verilog source code.

### Function `equation_to_fpga()`
One-liner: ODE string → (Python neuron, Verilog RTL).

>>> neuron, verilog = equation_to_fpga(
...     "dv/dt = -(v - E_L)/tau_m + I/C",
...     threshold="v > -50", reset="v = -65",
...     params=dict(E_L=-65, tau_m=10, C=1),
...     init=dict(v=-65),
... )

### Function `generate_testbench(neuron, module_name, n_steps, input_current, data_width, fraction)`
Generate a Verilog testbench for a compiled equation neuron.

Drives the module with constant current for n_steps clock cycles,
monitors spike_out and state outputs, and produces a VCD waveform.

Parameters
----------
neuron : EquationNeuron
    The neuron (same one passed to compile_to_verilog).
module_name : str
    Must match the module name used in compile_to_verilog.
n_steps : int
    Number of simulation clock cycles.
input_current : float
    Constant input current (Q-encoded internally).
data_width : int
    Bit width matching the compiled module.
fraction : int
    Fractional bits matching the compiled module.

Returns
-------
str
    Verilog testbench source code.

---

## Module `compiler.intelligence.core`

### Class `PositConfig`
Posit number format configuration.

Attributes
----------
nbits : int
    Total bit width (8 or 16).
es : int
    Exponent field size (0, 1, or 2).

- **useed**()
  - The useed value: 2^(2^es).
- **max_value**()
  - Maximum finite value.
- **min_positive**()
  - Smallest positive value.

### Class `MXFPConfig`
Microsoft Microscaling (MX) floating-point format.

Based on OCP Microscaling Formats Specification v1.0 (2024).

Attributes
----------
element_bits : int
    Bits per element (4, 6, or 8).
exp_bits : int
    Exponent bits per element.
mantissa_bits : int
    Mantissa bits per element (including implicit 1).
block_size : int
    Elements per shared-exponent block.
shared_exp_bits : int
    Shared exponent width (typically 8).

- **label**()
  - Human-readable format label.
- **bits_per_block**()
  - Total bits for one block including shared exponent.

### Class `QuantSweepResult`
Result of a quantisation sweep for one (width, fraction) pair.

Attributes
----------
data_width : int
    Total bit width tested.
fraction : int
    Fractional bits tested.
guard_bits : int
    Guard bits required.
estimated_luts : int
    Estimated LUT usage.
estimated_dsps : int
    Estimated DSP usage.
estimated_ffs : int
    Estimated flip-flop usage.
max_representable : float
    Maximum representable value.
min_step : float
    Minimum step size (LSB resolution).


### Class `OnChipLearningParams`
Parameters for on-chip STDP / reward-modulated plasticity.

Attributes
----------
learning_rule : str
    ``"stdp"``, ``"rstdp"`` (reward-modulated), or ``"triplet"``.
tau_plus_ms : float
    Pre→post time constant (ms).
tau_minus_ms : float
    Post→pre time constant (ms).
a_plus : float
    Potentiation amplitude.
a_minus : float
    Depression amplitude.
w_max : float
    Maximum synaptic weight.
w_min : float
    Minimum synaptic weight.
reward_tau_ms : float
    Reward signal time constant (ms), for RSTDP.
target_platform : str
    Target neuromorphic platform.


### Class `WeightNoiseProfile`
Device-variation noise model for analog/memristive targets.

Attributes
----------
noise_model : str
    ``"gaussian"``, ``"uniform"``, or ``"lognormal"``.
sigma : float
    Standard deviation of noise (fraction of weight range).
cycle_drift : float
    Weight drift per program/erase cycle (fraction).
retention_loss_per_day : float
    Daily retention loss (fraction).
target_platform : str
    Target platform.


### Class `TimescalePartition`
Partitioned ODE system by timescale.

Attributes
----------
fast_equations : dict&#91;str, str&#93;
    Fast dynamics (membrane, spikes).
slow_equations : dict&#91;str, str&#93;
    Slow dynamics (adaptation, homeostasis).
fast_clock_div : int
    Clock divider for fast domain (1 = full speed).
slow_clock_div : int
    Clock divider for slow domain.
cdc_signals : list&#91;str&#93;
    Signals requiring clock domain crossing.


### Class `DispatchPlan`
Multi-backend SNN dispatch plan.

Attributes
----------
backends : dict&#91;str, list&#91;str&#93;&#93;
    Backend name → list of assigned state variables.
sync_barriers : list&#91;str&#93;
    Synchronisation point descriptions.
total_neurons_per_backend : dict&#91;str, int&#93;
    Neuron count per backend.
estimated_speedup : float
    Estimated speedup vs single-backend.


### Class `TargetRecommendation`
Ranked hardware target recommendation.

Attributes
----------
profile_name : str
    Recommended profile.
score : float
    Fitness score (0-100).
rationale : str
    Why this target is recommended.


### Class `ReconfigPartition`
Partial reconfiguration partition plan.

Attributes
----------
partitions : list&#91;dict&#91;str, list&#91;str&#93;&#93;&#93;
    Each partition maps region name → assigned variables.
schedule : list&#91;str&#93;
    Time-ordered bitstream swap schedule.
total_regions : int
    Number of reconfigurable regions.
bitstream_count : int
    Total partial bitstreams needed.


### Class `CompilationCache`
Memoized compilation result cache.

Keyed by ``(equations_hash, target, data_width, fraction)``.
Avoids redundant recompilation when re-targeting.

- **__init__**()
- **_key**(equations, target, data_width, fraction)
- **get**(equations, target, data_width, fraction)
  - Look up a cached compilation result.
- **put**(equations, target, data_width, fraction, result)
  - Store a compilation result in cache.
- **size**()
  - Number of cached entries.

### Class `TopologyPlan`
Multi-chip network topology optimisation result.

Attributes
----------
chip_assignment : dict&#91;int, int&#93;
    Neuron index → chip index.
inter_chip_spikes : int
    Estimated inter-chip spikes per timestep.
intra_chip_spikes : int
    Estimated intra-chip spikes per timestep.
bandwidth_reduction : float
    Reduction vs naive assignment.
num_chips : int
    Total chips used.


### Class `NIRGraph`
Imported NIR/ONNX-SNN graph representation.

Attributes
----------
nodes : dict&#91;str, dict&#93;
    Node name → parameters.
edges : list&#91;tuple&#91;str, str&#93;&#93;
    Directed edges (source, target).
equations : dict&#91;str, str&#93;
    Extracted ODE equations per node.
framework : str
    Source framework.


### Class `DebugProbeSpec`
Auto-generated debug probe specification.

Attributes
----------
probe_type : str
    ``"ila"`` (Xilinx) or ``"signaltap"`` (Intel).
signals : list&#91;str&#93;
    Probed signal names.
depth : int
    Capture depth.
tcl_commands : str
    Vendor-specific TCL to insert probes.


### Function `verilog_to_vhdl_wrapper(module_name)`
Generate a VHDL-2008 entity/architecture wrapper for a Verilog module.

This produces a VHDL entity that matches the Verilog module's port list,
enabling mixed-language simulation and synthesis (Vivado, Questa, GHDL).
The VHDL wrapper instantiates the Verilog module as a component.

Parameters
----------
module_name : str
    Verilog module name.
data_width : int
    Fixed-point data width.
signed : bool
    Whether ports use signed types.

Returns
-------
str
    VHDL-2008 source code.

### Function `posit_encode(value, config)`
Encode a float to posit integer representation.

Parameters
----------
value : float
    Value to encode.
config : PositConfig
    Posit format.

Returns
-------
int
    Posit-encoded integer (nbits wide).

### Function `posit_decode(bits, config)`
Decode a posit integer to float.

Parameters
----------
bits : int
    Posit-encoded integer.
config : PositConfig
    Posit format.

Returns
-------
float
    Decoded value.

### Function `generate_tcl_project(module_name)`
Generate FPGA project TCL script.

Parameters
----------
module_name : str
    Top-level module name.
tool : str
    ``"vivado"`` or ``"quartus"``.
part : str
    Target FPGA part number.
verilog_files : list, optional
    Verilog source files.
constraint_file : str, optional
    Constraint file (XDC/SDC).

Returns
-------
str
    Complete TCL script.

### Function `_gen_vivado_tcl(module_name, part, verilog_files, constraint_file)`
Generate Xilinx Vivado project TCL.

### Function `_gen_quartus_tcl(module_name, part, verilog_files, constraint_file)`
Generate Intel Quartus project TCL.

### Function `generate_oss_makefile(module_name)`
Generate a Makefile for open-source FPGA synthesis (Yosys + nextpnr).

Parameters
----------
module_name : str
    Top-level module name.
target : str
    ``"ice40"`` or ``"ecp5"``.
device : str
    Device string (e.g. ``"hx8k"``, ``"um5g-85k"``).
package : str
    Package (e.g. ``"ct256"``, ``"CABGA381"``).
freq_mhz : float
    Target frequency.
verilog_files : list, optional
    Verilog source files.
pcf_file : str, optional
    Pin constraint file.

Returns
-------
str
    Complete Makefile content.

### Function `generate_dvs_aer_bridge(module_name)`
Generate a DVS (Dynamic Vision Sensor) to AER bridge in Verilog.

Converts Prophesee / Metavision / Sony IMX636 event packets into
the SC-NeuroCore AER address-event protocol for zero-copy sensor-
to-spike-network interfacing on FPGA.

Parameters
----------
module_name : str
    Output module name.
addr_width : int
    Pixel address width (covers X*Y event space).
polarity_bit : bool
    Include ON/OFF polarity in the event word.
timestamp_width : int
    Timestamp field width in bits.
fifo_depth : int
    Input event FIFO depth (power of 2).

Returns
-------
str
    Synthesisable Verilog module.

### Function `mxfp_encode_block(values, config)`
Encode a block of floats to MXFP format.

Parameters
----------
values : list&#91;float&#93;
    Block of float values (len must equal config.block_size).
config : MXFPConfig
    MXFP format configuration.

Returns
-------
tuple&#91;int, list&#91;int&#93;&#93;
    (shared_exponent, list_of_encoded_elements).

### Function `mxfp_decode_block(shared_exp, elements, config)`
Decode a block of MXFP elements to floats.

Parameters
----------
shared_exp : int
    Shared exponent.
elements : list&#91;int&#93;
    Encoded element integers.
config : MXFPConfig
    MXFP format configuration.

Returns
-------
list&#91;float&#93;
    Decoded float values.

### Function `generate_weight_rom(weights, module_name)`
Generate a weight ROM for synaptic connections.

Produces either a Verilog ROM module or a Xilinx ``.coe`` / Intel
``.mif`` memory initialisation file for BRAM-based weight storage.

Parameters
----------
weights : list&#91;list&#91;int&#93;&#93;
    2D weight matrix &#91;src_neuron&#93;&#91;dst_neuron&#93; in Q-format integers.
module_name : str
    ROM module name.
data_width : int
    Weight bit width.
output_format : str
    ``"verilog"`` (synthesisable ROM), ``"coe"`` (Xilinx), ``"mif"`` (Intel).

Returns
-------
str
    Weight ROM in the specified format.

### Function `auto_quantisation_sweep(equations, target)`
Sweep data widths to find accuracy-vs-resource trade-offs.

Compiles the same ODE equations at multiple quantisation levels
(Q4.2 through Q32.16) and reports the resource cost and numerical
precision for each. Enables rapid design-space exploration.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations mapping state variable names to expressions.
target : str
    Target platform name for resource estimation.
widths : list&#91;int&#93;, optional
    Data widths to sweep. Defaults to ``&#91;4, 8, 12, 16, 20, 24, 32&#93;``.
fraction_ratio : float
    Fraction of data_width used for fractional bits (default 0.5).

Returns
-------
list&#91;QuantSweepResult&#93;
    Sweep results sorted by data_width (ascending).

### Function `format_quantisation_report(results)`
Format a quantisation sweep into a readable markdown table.

Parameters
----------
results : list&#91;QuantSweepResult&#93;
    Results from ``auto_quantisation_sweep()``.

Returns
-------
str
    Markdown table comparing all quantisation levels.

### Function `generate_hls_cpp(module_name, equations)`
Translate compiled neuron equations to Vitis/Catapult HLS C++.

Generates a synthesisable C++ function with ``#pragma HLS`` directives
for Xilinx Vitis HLS or Siemens Catapult. Enables HW/SW co-design
workflows where the neuron runs as an HLS IP block alongside a
MicroBlaze or RISC-V soft processor.

Parameters
----------
module_name : str
    Function/module name.
equations : dict&#91;str, str&#93;
    ODE equations (state_var → C-style expression).
data_width : int
    Fixed-point total width.
fraction : int
    Fractional bits.
hls_tool : str
    ``"vitis"`` or ``"catapult"``.

Returns
-------
str
    Complete HLS C++ source file.

### Function `generate_learning_params()`
Generate on-chip learning parameters for neuromorphic targets.

Creates calibration parameters for platforms with in-situ
plasticity (BrainChip Akida 2, BrainScaleS-2, SpiNNaker 2).

Parameters
----------
learning_rule : str
    ``"stdp"`` (spike-timing), ``"rstdp"`` (reward-modulated),
    or ``"triplet"`` (triplet-based STDP).
tau_plus_ms : float
    LTP time constant.
tau_minus_ms : float
    LTD time constant.
a_plus : float
    Potentiation amplitude.
a_minus : float
    Depression amplitude.
w_max : float
    Weight ceiling.
w_min : float
    Weight floor.
reward_tau_ms : float
    Reward eligibility trace time constant.
target : str
    Target platform name.

Returns
-------
OnChipLearningParams
    Complete learning parameter set.

### Function `export_learning_config(params)`
Export on-chip learning parameters as a configuration file.

Parameters
----------
params : OnChipLearningParams
    Learning parameters from ``generate_learning_params()``.
output_format : str
    ``"json"`` or ``"yaml"``.

Returns
-------
str
    Configuration file content.

### Function `inject_weight_noise(weights)`
Inject device-variation noise into a weight matrix.

Simulates manufacturing variations and read noise in analog
compute-in-memory (Mythic, IBM PCM) and memristive crossbar
(Rain AI) targets. Enables robustness validation before tapeout.

Parameters
----------
weights : list&#91;list&#91;float | int&#93;&#93;
    Original weight matrix.
noise_model : str
    ``"gaussian"``, ``"uniform"``, or ``"lognormal"``.
sigma : float
    Noise magnitude (fraction of weight range).
seed : int, optional
    Random seed for reproducibility.

Returns
-------
list&#91;list&#91;float&#93;&#93;
    Weight matrix with injected noise.

### Function `create_noise_profile()`
Create a device-variation noise profile for analog targets.

Parameters
----------
noise_model : str
    Noise distribution type.
sigma : float
    Read noise standard deviation.
cycle_drift : float
    Weight drift per program/erase cycle.
retention_loss_per_day : float
    Daily state retention loss.
target : str
    Target platform.

Returns
-------
WeightNoiseProfile
    Complete noise characterisation.

### Function `partition_timescales(equations, time_constants)`
Partition ODE equations by timescale for multi-clock execution.

Identifies fast vs slow dynamics and assigns them to different
clock domains, inserting CDC synchronisers at domain boundaries.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.
time_constants : dict&#91;str, float&#93;, optional
    Known time constants per variable (ms). If None, estimated
    from equation structure.
threshold_ratio : float
    Ratio above which a variable is considered "slow".

Returns
-------
TimescalePartition
    Partitioned system with clock assignments.

### Function `plan_heterogeneous_dispatch(equations, backends)`
Plan multi-backend dispatch for an SNN model.

Splits ODE variables across heterogeneous backends based on
compute characteristics (fast dynamics → FPGA, slow → MCU,
learning → GPU).

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.
backends : list&#91;str&#93;
    Available backend targets.
neuron_count : int
    Total neurons.
time_constants : dict&#91;str, float&#93;, optional
    Time constants per variable.

Returns
-------
DispatchPlan
    Multi-backend assignment.

### Function `recommend_target(equations)`
Recommend optimal hardware targets for a neuron model.

Given ODE equations and constraints, ranks all registered
profiles and returns the top N recommendations.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.
max_power_mw : float, optional
    Maximum power budget.
min_freq_mhz : float, optional
    Minimum clock frequency.
max_data_width : int, optional
    Maximum data width.
require_class : str, optional
    Required platform class.
top_n : int
    Number of recommendations.

Returns
-------
list&#91;TargetRecommendation&#93;
    Ranked recommendations.

### Function `plan_partial_reconfiguration(equations)`
Plan FPGA partial reconfiguration for SNN time-multiplexing.

Splits neuron equations across reconfigurable regions and
generates a swap schedule.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.
max_regions : int
    Maximum reconfigurable regions.
time_slots : int
    Number of time-multiplexed slots.

Returns
-------
ReconfigPartition
    Partition plan with schedule.

### Function `optimize_network_topology(adjacency)`
Optimize SNN partitioning across multiple chips.

Minimises inter-chip spike communication by grouping
heavily-connected neurons onto the same chip.

Parameters
----------
adjacency : dict&#91;int, list&#91;int&#93;&#93;
    Neuron connectivity: source → list of targets.
num_chips : int
    Number of available chips.
neurons_per_chip : int, optional
    Max neurons per chip. Default: ceil(N / num_chips).

Returns
-------
TopologyPlan
    Optimised chip assignment.

### Function `import_nir_graph(nir_data)`
Import a Neuromorphic Intermediate Representation graph.

Converts NIR node definitions into ODE equations suitable
for the SC-NeuroCore compilation pipeline.

Parameters
----------
nir_data : dict
    NIR graph as dictionary with 'nodes' and 'edges'.
framework : str
    Source framework name.

Returns
-------
NIRGraph
    Imported graph with extracted equations.

### Function `insert_debug_probes(module_name, equations)`
Auto-insert ILA/SignalTap debug probes.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    ODE equations (state variables become probed signals).
vendor : str
    ``"xilinx"`` or ``"intel"``.
depth : int
    Capture depth in samples.

Returns
-------
DebugProbeSpec
    Probe specification with TCL commands.

---

## Module `compiler.intelligence.digital_twin`

### Class `DriftCompensator`
Analog drift compensation parameters.

Attributes
----------
refresh_interval_ms : float
    How often to re-calibrate (ms).
drift_rate_per_day : float
    Expected weight drift per day.
compensation_method : str
    ``"periodic_refresh"``, ``"adaptive"``, or ``"ecc"``.
verilog_controller : str
    Generated Verilog refresh controller.


### Class `HILCalibration`
Hardware-in-the-loop calibration protocol.

Attributes
----------
protocol_steps : list&#91;str&#93;
num_parameters : int
sweep_ranges : dict&#91;str, tuple&#91;float, float&#93;&#93;


### Class `ScrubSchedule`
Configuration memory scrubbing schedule.

Attributes
----------
interval_ms : float
strategy : str
frames_per_cycle : int
expected_seu_rate : float


### Class `TelemetryResult`
Hardware telemetry comparison result.

Attributes
----------
samples : int
max_drift : float
mean_drift : float
alerts : list&#91;str&#93;
healthy : bool


### Function `generate_drift_compensator(module_name)`
Generate analog drift compensation controller.

For analog/memristive targets, creates on-chip calibration
circuits that periodically refresh weights to compensate
for device aging and retention loss.

Parameters
----------
module_name : str
    Module name.
drift_rate_per_day : float
    Weight drift per day (fraction).
max_drift_tolerance : float
    Maximum acceptable drift before refresh.
clock_freq_mhz : int
    Clock frequency.
compensation_method : str
    Compensation strategy.

Returns
-------
DriftCompensator
    Controller with Verilog.

### Function `_validate_hil_contract(module_name, equations, parameters, sample_points, repetitions, settle_cycles, acceptance_tolerance)`
### Function `_coprime_stride(sample_points, start)`
### Function `_latin_hypercube_design(parameters, sample_points)`
### Function `generate_hil_calibration(module_name, equations)`
Generate hardware-in-the-loop calibration protocol.

Produces a step-by-step calibration procedure for compensating
analog drift, mismatch, and process variation on real hardware.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    State variable equations.
parameters : dict&#91;str, tuple&#91;float, float&#93;&#93;, optional
    Parameter sweep ranges {name: (min, max)}.
sample_points : int
    Deterministic Latin-hypercube design points across the parameter space.
repetitions : int
    Repeated measurements per design point for variance estimation.
settle_cycles : int
    Hardware cycles to wait before sampling after each parameter update.
acceptance_tolerance : float
    Maximum absolute drift allowed for each observable.
correction_model : str
    Calibration model fitted to measured residuals.
observables : tuple&#91;str, ...&#93;, optional
    State variables to compare against the software golden model.

Returns
-------
HILCalibration

### Function `generate_digital_twin(module_name, equations, profile_name)`
Generate a Python digital twin that mirrors deployed hardware.

The twin tracks identical state transitions in software, enabling
runtime comparison, anomaly detection, and predictive maintenance.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    State variable equations.
profile_name : str
    Target hardware profile.

Returns
-------
str
    Python source code for the digital twin class.

### Function `schedule_seu_scrubbing(config_bits)`
Generate scrubbing schedule for space-grade configuration memory.

Uses orbital altitude and shielding to estimate SEU rate, then
calculates optimal scrub interval for target reliability.

Parameters
----------
config_bits : int
    Total configuration memory bits.
orbit_altitude_km : float
    Orbital altitude (affects particle flux).
shielding_mm_al : float
    Aluminium shielding thickness.
strategy : str
    "blind" (full-chip) or "hybrid" (targeted + periodic full).

Returns
-------
ScrubSchedule

### Function `ingest_telemetry(telemetry_data, twin_states)`
Ingest hardware telemetry and compare against digital twin.

Parameters
----------
telemetry_data : list&#91;dict&#91;str, float&#93;&#93;
    List of hardware state snapshots {var: value}.
twin_states : list&#91;dict&#91;str, float&#93;&#93;
    Corresponding digital twin states.
drift_threshold : float
    Alert threshold for absolute drift.

Returns
-------
TelemetryResult

---

## Module `compiler.intelligence.exotic_physics`

### Class `MZIWeightEncoding`
Encoded weights for a Mach-Zehnder interferometer photonic array.

Attributes
----------
phases_theta : list&#91;list&#91;float&#93;&#93;
    Phase-shift θ values (radians) for each MZI in the mesh.
phases_phi : list&#91;list&#91;float&#93;&#93;
    Phase-shift φ values (radians) for external phase shifters.
transmission : list&#91;list&#91;float&#93;&#93;
    Effective transmission coefficients.
mesh_size : int
    Number of MZI columns in the Clements mesh.


### Class `OmniDispatchMap`

### Class `ReversibleNetlist`

### Class `MEAMapping`

### Class `Morphology`

### Class `CognitiveBounds`

### Class `AdiabaticPhase`

### Class `HolographicRouter`

### Function `encode_mzi_weights(weights)`
Encode a weight matrix as MZI phase-shift parameters.

Converts a real-valued weight matrix into the (θ, φ) phase-shift
representation used by photonic Mach-Zehnder interferometer meshes
(Lightmatter, iPronics, Xanadu). Uses the Clements decomposition
to map an arbitrary unitary matrix to a cascade of 2×2 beam splitters.

Parameters
----------
weights : list&#91;list&#91;float | int&#93;&#93;
    Weight matrix (N×M). Values are normalised to &#91;-1, 1&#93;.
mesh_type : str
    ``"clements"`` (triangular) or ``"reck"`` (rectangular).
loss_db_per_mzi : float
    Insertion loss per MZI in dB (for transmission estimation).

Returns
-------
MZIWeightEncoding
    Phase-shift parameters and transmission coefficients.

### Function `generate_mzi_config(encoding)`
Generate a photonic chip configuration file from MZI weights.

Parameters
----------
encoding : MZIWeightEncoding
    Phase-shift encoding from ``encode_mzi_weights()``.
output_format : str
    ``"json"`` or ``"csv"``.

Returns
-------
str
    Configuration file content.

### Function `dispatch_omni_paradigm(equations)`
### Function `synthesize_reversible_logic(equations, bits)`
### Function `map_wetware_mea(populations, connectivity)`
### Function `synthesize_morphology(equations, max_generations)`
### Function `enforce_cognitive_bounds(equations, state_bounds)`
### Function `generate_adiabatic_clocks(phases, freq_mhz)`
### Function `route_holographic_interconnects(num_neurons, connections)`
---

## Module `compiler.intelligence.power_and_thermal`

### Class `ThermalEstimate`
Thermal analysis result for a compiled neuron.

Attributes
----------
power_mw : float
    Estimated total power in milliwatts.
delta_t_c : float
    Estimated temperature rise in °C.
junction_temp_c : float
    Estimated junction temperature.
hotspot_delta_t_c : float
    Local temperature rise from concentrated DSP power.
derated_freq_mhz : float
    Frequency after thermal derating.
thermal_safe : bool
    True if junction temp is within limits.
hotspot_risk : str
    ``"none"``, ``"low"``, ``"medium"``, ``"high"``.


### Class `EnergySchedule`
Energy-aware neuron update schedule.

Attributes
----------
total_neurons : int
    Total neurons.
energy_budget_uj : float
    Energy budget per epoch (µJ).
neurons_per_epoch : int
    Neurons updatable within budget.
update_order : list&#91;int&#93;
    Priority-ordered neuron indices.
epoch_duration_ms : float
    Epoch duration.
duty_cycle : float
    Fraction of neurons updated per epoch.


### Class `ThermalEnvelopeEstimate`
Junction temperature estimate.

Attributes
----------
power_mw : float
    Estimated power dissipation (mW).
theta_ja : float
    Junction-to-ambient thermal resistance (°C/W).
t_ambient : float
    Ambient temperature (°C).
t_junction : float
    Estimated junction temperature (°C).
thermal_margin : float
    Margin to max T_j (°C).
pass_fail : str
    ``"PASS"`` or ``"FAIL"``.


### Class `ApproximationConfig`
Approximate computing configuration.

Attributes
----------
populations : dict&#91;str, dict&#93;
total_energy_savings_pct : float
max_output_error_pct : float


### Class `EnergyHarvestBudget`
Energy harvesting feasibility analysis.

Attributes
----------
harvester_power_uw : float
design_power_uw : float
energy_positive : bool
recommended_duty_cycle : float
margin_pct : float


### Function `thermal_analysis(estimated_power_mw, target_freq_mhz)`
Estimate thermal impact and frequency derating.

Estimates lumped junction rise from total power and applies timing
derating from both die temperature and DSP-column concentration.

Parameters
----------
estimated_power_mw : float
    Estimated power from ``estimate_power()`` or synthesis.
target_freq_mhz : float
    Nominal target frequency.
theta_ja : float
    Junction-to-ambient thermal resistance (°C/W).
    Typical: ~11.5 for Artix-7 BGA, ~3.5 for Versal with heatsink.
t_ambient_c : float
    Ambient temperature.
t_junction_max_c : float
    Maximum junction temperature.
process_nm : int
    Process node (affects derating sensitivity).
mul_count : int
    Number of DSP multipliers (affects hotspot risk).
dsp_columns : int
    Number of DSP columns to spread across.
dsp_power_mw : float, optional
    DSP-attributed dynamic power for local hotspot analysis. When omitted,
    local spreading rise is not added.
theta_spreading : float
    Local spreading resistance from a DSP column hotspot to the bulk
    junction node (°C/W).

Returns
-------
ThermalEstimate
    Thermal analysis with derating and hotspot risk.

### Function `_require_finite(value, name)`
### Function `_require_finite_positive(value, name)`
### Function `_require_finite_non_negative(value, name)`
### Function `generate_thermal_constraints(module_name, analysis)`
Generate XDC constraints for thermal-aware DSP placement.

Spreads DSP blocks across multiple columns to reduce thermal hotspots
and adds temperature-derated timing constraints.

Parameters
----------
module_name : str
    Module name.
analysis : ThermalEstimate
    Thermal analysis result.
dsp_columns : int
    Number of DSP columns to distribute across.

Returns
-------
str
    XDC constraint snippet for thermal-aware placement.

### Function `generate_energy_schedule(neuron_count)`
Generate energy-budget-aware neuron update schedule.

For energy-harvesting edge devices (solar, vibration, RF),
schedules neuron updates to fit within the available energy.

Parameters
----------
neuron_count : int
    Total neurons.
energy_budget_uj : float
    Available energy per epoch (µJ).
energy_per_neuron_nj : float
    Energy per neuron update (nJ).
epoch_duration_ms : float
    Epoch duration (ms).
priority_neurons : list&#91;int&#93;, optional
    High-priority neuron indices (updated first).

Returns
-------
EnergySchedule
    Update schedule.

### Function `estimate_thermal_envelope()`
Predict junction temperature from power dissipation.

Uses simple thermal resistance model: T_j = T_a + P × θ_ja.

Parameters
----------
power_mw : float
    Power dissipation (mW).
theta_ja : float
    Junction-to-ambient thermal resistance (°C/W).
t_ambient : float
    Ambient temperature (°C).
t_junction_max : float
    Maximum allowed junction temperature (°C).

Returns
-------
ThermalEnvelopeEstimate
    Temperature estimate with pass/fail.

### Function `generate_power_intent(module_name)`
Generate IEEE 1801 UPF power intent for neuron arrays.

Creates power domain definitions, isolation rules, and
retention strategies for multi-voltage SNN designs.

Parameters
----------
module_name : str
    Top module name.
num_domains : int
    Number of power domains.
always_on : bool
    Whether to include always-on domain.

Returns
-------
str
    UPF source text.

### Function `generate_power_state_machine(module_name)`
Generate sleep/wake/hibernate FSM for ultra-low-power.

Parameters
----------
module_name : str
    Module name.
states : list&#91;str&#93;, optional
    FSM states. Default: ACTIVE, IDLE, SLEEP, HIBERNATE.

Returns
-------
str
    Verilog FSM source.

### Function `configure_approximation(equations)`
Configure precision-energy tradeoff knobs per state variable.

Analyses each variable's dynamic range and recommends bit-width
reduction and stochastic rounding to achieve target energy savings
while bounding output error.

Parameters
----------
equations : dict&#91;str, str&#93;
    State variable equations.
target_savings_pct : float
    Target energy savings percentage.
max_error_pct : float
    Maximum acceptable output error percentage.

Returns
-------
ApproximationConfig

### Function `model_energy_harvest(design_power_uw)`
Model whether an energy harvester can sustain the neural workload.

Parameters
----------
design_power_uw : float
    Design power consumption in microwatts.
harvester_type : str
    "solar", "piezo", "thermal", or "rf".
harvester_area_cm2 : float
    Harvester active area.
environment : str
    "indoor", "outdoor", or "industrial".

Returns
-------
EnergyHarvestBudget

### Function `generate_dvfs_controller(module_name)`
Generate a Verilog DVFS controller FSM.

Parameters
----------
module_name : str
    Module name.
operating_points : list&#91;dict&#93;, optional
    &#91;{voltage_mv, freq_mhz}&#93;. Default: 3 points.
spike_rate_thresholds : list&#91;float&#93;, optional
    Spike rate thresholds for state transitions.

Returns
-------
str
    Synthesisable Verilog source code.

---

## Module `compiler.intelligence.reporting`

### Class `TargetComparison`
Compilation comparison for one target.

Attributes
----------
target : str
    Platform name.
data_width : int
    Selected data width.
fraction : int
    Fractional bits.
overflow : str
    Overflow mode.
dsp_block : str
    DSP block type.
max_freq_mhz : int | None
    Maximum frequency.
estimated_luts : int
    Estimated LUT usage.
estimated_dsps : int
    Estimated DSP usage.
pipeline_stages : int
    Required pipeline stages.
critical_path_depth : int
    DSP chain depth.


### Class `ProvenanceRecord`
Cryptographic audit trail entry.

Attributes
----------
stage : str
    Pipeline stage name.
input_hash : str
    SHA-256 of input artefact.
output_hash : str
    SHA-256 of output artefact.
timestamp : str
    ISO 8601 timestamp.
parameters : dict
    Compilation parameters used.


### Class `ModelComplexity`
Model compute-profile classification.

Attributes
----------
classification : str
    ``"compute_bound"``, ``"memory_bound"``, or ``"comm_bound"``.
compute_ops : int
    Total arithmetic operations.
memory_vars : int
    State variables (memory footprint proxy).
comm_ratio : float
    Inter-variable coupling ratio.
recommended_paradigm : str
    Best platform class.


### Class `PortabilityScore`
Cross-platform portability assessment.

Attributes
----------
score : float
    Portability score 0-100.
compatible_profiles : int
    Number of compatible profiles.
total_profiles : int
    Total profiles checked.
blockers : list&#91;str&#93;
    Portability blockers.


### Class `ParetoPoint`
A single Pareto-optimal design point.

Attributes
----------
config : dict
power_mw : float
area_luts : int
latency_ns : float


### Function `compare_targets(equations, targets)`
Compare compilation results across multiple hardware targets.

Compiles the same ODE equations for each target and reports
resource usage, precision, and pipeline requirements side-by-side.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.
targets : list&#91;str&#93;
    List of target platform names.

Returns
-------
list&#91;TargetComparison&#93;
    Comparison results for each target.

### Function `format_comparison_report(results)`
Format a multi-target comparison as a markdown table.

Parameters
----------
results : list&#91;TargetComparison&#93;
    Results from ``compare_targets()``.

Returns
-------
str
    Markdown comparison table.

### Function `generate_compilation_summary(module_name, equations, target)`
Generate a comprehensive human-readable compilation summary.

Produces a markdown document summarising all aspects of a
compilation: equations, target, precision, resources, pipeline,
guard bits, and applicable strategic features.

Parameters
----------
module_name : str
    Compiled module name.
equations : dict&#91;str, str&#93;
    ODE equations compiled.
target : str
    Target platform.
data_width : int
    Total bit width.
fraction : int
    Fractional bits.
verilog_lines : int
    Lines of generated Verilog (0 if not counted).

Returns
-------
str
    Markdown compilation summary.

### Function `generate_provenance_chain(module_name, equations, verilog_source)`
Generate a cryptographic provenance chain for compilation.

Creates a full audit trail from source equations through
compiled RTL, with SHA-256 hashes at every stage.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    Source ODE equations.
verilog_source : str
    Generated Verilog (if available).
target : str
    Target platform.
data_width : int
    Fixed-point width.
fraction : int
    Fractional bits.

Returns
-------
list&#91;ProvenanceRecord&#93;
    Ordered provenance records.

### Function `format_provenance_json(chain)`
Format provenance chain as JSON manifest.

Parameters
----------
chain : list&#91;ProvenanceRecord&#93;
    From ``generate_provenance_chain()``.

Returns
-------
str
    JSON manifest.

### Function `classify_model_complexity(equations)`
Classify a model's compute profile.

Determines whether the model is compute-bound, memory-bound,
or communication-bound and recommends the best platform paradigm.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.

Returns
-------
ModelComplexity
    Classification with recommended paradigm.

### Function `score_portability(equations)`
Score how portable a model is across all profiles.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.
min_data_width : int
    Minimum acceptable data width.

Returns
-------
PortabilityScore
    Portability assessment.

### Function `generate_compilation_report(module_name, equations, profile_name)`
Generate comprehensive compilation report.

Consolidates Verilog, timing, power, carbon, risk,
and reliability into a single markdown document.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    ODE equations.
profile_name : str
    Target profile.
include_carbon : bool
    Include carbon footprint section.
include_reliability : bool
    Include reliability prediction.

Returns
-------
str
    Markdown report.

### Function `explore_pareto(equations)`
Explore power/area/latency Pareto frontier.

Parameters
----------
equations : dict&#91;str, str&#93;
    State variable equations.
widths : list&#91;int&#93;, optional
    Bit widths to sweep. Default: &#91;8, 16, 24, 32&#93;.
pipeline_depths : list&#91;int&#93;, optional
    Pipeline stages to sweep. Default: &#91;1, 2, 4&#93;.

Returns
-------
list&#91;ParetoPoint&#93;
    Non-dominated design points.

---

## Module `compiler.intelligence.security_and_compliance`

### Class `SideChannelFinding`
Side-channel leakage finding.

Attributes
----------
signal : str
    Signal name.
risk_level : str
    ``"high"``, ``"medium"``, or ``"low"``.
category : str
    ``"timing"`` or ``"power"``.
description : str
    Explanation.
recommendation : str
    Mitigation suggestion.


### Class `SupplyChainRisk`
Supply chain risk assessment for a hardware profile.

Attributes
----------
profile_name : str
    Assessed profile.
risk_score : float
    Risk score 0-100 (higher = riskier).
risk_factors : list&#91;str&#93;
    Individual risk factor descriptions.
alternatives : list&#91;str&#93;
    Suggested alternative profiles.
export_control : str
    Export control classification.


### Class `CarbonEstimate`
Carbon footprint estimate per compilation target.

Attributes
----------
profile_name : str
    Target profile.
manufacturing_kg_co2 : float
    Estimated manufacturing CO₂ (kg).
operation_kg_co2_per_year : float
    Estimated annual operation CO₂ (kg).
total_5yr_kg_co2 : float
    Total 5-year lifecycle CO₂ (kg).
energy_mix : str
    Assumed energy source.


### Class `LicenseCheck`
IP core license compatibility result.

Attributes
----------
compatible : bool
conflicts : list&#91;str&#93;
licenses_found : list&#91;str&#93;


### Class `TrojanLintResult`
Hardware trojan lint analysis result.

Attributes
----------
suspicious_paths : list&#91;str&#93;
risk_level : str
total_checks : int


### Class `SBOM`
Software/Hardware Bill of Materials.

Attributes
----------
format : str
components : list&#91;dict&#93;
total_components : int


### Class `ObfuscationResult`
IP obfuscation report.

Attributes
----------
techniques_applied : list&#91;str&#93;
key_bits : int
original_signals : int
obfuscated_signals : int


### Class `WatermarkResult`
Netlist watermark embedding result.

Attributes
----------
watermark_hash : str
embedding_method : str
overhead_percent : float
verifiable : bool


### Class `PQCProtection`
Post-quantum cryptographic IP protection result.

Attributes
----------
algorithm : str
signature_hex : str
key_size_bits : int
quantum_safe : bool


### Function `embed_model_checksum(verilog)`
Embed a SHA-256 checksum of the compiled model in the Verilog source.

Enables bit-exact reproducibility verification — the hash of the
source equations and parameters is embedded as a Verilog comment and
a localparam, allowing downstream tools to verify that the RTL
matches the expected model.

Parameters
----------
verilog : str
    Generated Verilog source code.
equations : dict&#91;str, str&#93;, optional
    Original ODE equations (state_var → expression).
params : dict&#91;str, int | float&#93;, optional
    Compilation parameters (data_width, fraction, etc.).

Returns
-------
str
    Verilog with embedded checksum comment and localparam.

### Function `generate_bitstream_encryption(module_name)`
Generate bitstream encryption TCL/constraints for secure boot.

Produces the vendor-specific TCL commands and XDC constraints to
enable AES-256 bitstream encryption, protecting the compiled neuron
IP from reverse-engineering and tampering.

Parameters
----------
module_name : str
    Design module name.
vendor : str
    ``"xilinx"`` or ``"intel"``.
key_length : int
    AES key length: 128 or 256.
key_source : str
    Key storage: ``"efuse"`` (one-time programmable),
    ``"bbram"`` (battery-backed RAM), or ``"external"``.

Returns
-------
str
    TCL/Quartus script for bitstream encryption.

### Function `lint_side_channels(equations)`
Analyse equations for power/timing side-channel vulnerabilities.

Flags data-dependent timing paths and variable-activity patterns
in the generated RTL.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.
module_name : str
    Module name.
data_width : int
    Data width.

Returns
-------
list&#91;SideChannelFinding&#93;
    List of findings.

### Function `score_supply_chain_risk(profile_name)`
Assess supply chain risk for a hardware profile.

Scores based on vendor geography, sole-source status,
and export control classification.

Parameters
----------
profile_name : str
    Profile to assess.

Returns
-------
SupplyChainRisk
    Risk assessment.

### Function `estimate_carbon_footprint(profile_name)`
Estimate carbon footprint for a compilation target.

Parameters
----------
profile_name : str
    Target profile name.
power_mw : float
    Operating power (mW).
hours_per_day : float
    Operating hours per day.
grid_carbon_g_per_kwh : float
    Grid carbon intensity (g CO₂/kWh).

Returns
-------
CarbonEstimate
    Lifecycle carbon estimate.

### Function `check_license_compliance(project_license, dependencies)`
Verify IP core licensing compatibility.

Parameters
----------
project_license : str
    SPDX identifier of the project license.
dependencies : dict&#91;str, str&#93;
    Dependency name → SPDX license identifier.

Returns
-------
LicenseCheck

### Function `lint_hardware_trojans(equations)`
Detect suspicious combinational paths that could hide trojans.

Analyses the ODE dependency graph for dormant trigger conditions
and rarely-activated payload paths that are classic trojan signatures.

Parameters
----------
equations : dict&#91;str, str&#93;
    State variable equations.
check_dormant : bool
    Check for rarely-activated trigger paths.
check_payload : bool
    Check for suspicious payload injection points.

Returns
-------
TrojanLintResult

### Function `generate_sbom(module_name, profile_name)`
Generate SBOM/HBOM for IP core compliance.

Required by EU Cyber Resilience Act (2026). Generates a machine-
readable component inventory in CycloneDX or SPDX format.

Parameters
----------
module_name : str
    Module name.
profile_name : str
    Target hardware profile.
dependencies : dict&#91;str, str&#93;, optional
    External dependencies {name: version}.
sbom_format : str
    Output format: "CycloneDX" or "SPDX".

Returns
-------
SBOM

### Function `obfuscate_ip(module_name, equations)`
Apply logic locking and structural obfuscation for IP protection.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    State variable equations.
key_length : int
    Obfuscation key length in bits.
methods : list&#91;str&#93;, optional
    Techniques to apply. Default: logic_locking, constant_propagation_block,
    structural_transform.

Returns
-------
ObfuscationResult

### Function `embed_watermark(module_name, equations)`
Embed a verifiable watermark into the compiled netlist.

The watermark survives synthesis optimisation and can be verified
without access to the original design.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    State variable equations.
owner_id : str
    Owner identifier to embed.
method : str
    "constraint_based" or "don't_care_based".

Returns
-------
WatermarkResult

### Function `protect_ip_pqc(module_name, equations)`
Apply post-quantum cryptographic protection to IP core.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    State variable equations.
algorithm : str
    PQC algorithm. Default: CRYSTALS-Dilithium.
security_level : int
    NIST security level (2, 3, or 5).

Returns
-------
PQCProtection

---

## Module `compiler.intelligence.soc_and_chiplet`

### Class `StorageRecommendation`
Storage recommendation for neuron array state.

Attributes
----------
strategy : str
    ``"registers"``, ``"bram"``, or ``"uram"``.
neuron_count : int
    Number of neurons in the array.
total_bits : int
    Total state bits.
bram_18k_used : int
    Estimated 18Kb BRAM tiles consumed.
bram_36k_used : int
    Estimated 36Kb BRAM tiles consumed.
uram_used : int
    Estimated URAM tiles consumed (UltraScale+ only).
reason : str
    Human-readable explanation.


### Class `PIMLayout`
Memory layout plan for Processing-in-Memory targets.

Attributes
----------
bank_count : int
    Number of memory banks used.
neurons_per_bank : int
    Neurons assigned per bank.
weights_per_bank : int
    Weight entries per bank.
bank_utilisation : float
    Fraction of bank capacity used (0.0–1.0).
parallel_factor : int
    Number of banks that can compute in parallel.
layout_map : dict&#91;str, list&#91;int&#93;&#93;
    Mapping of data regions to bank IDs.


### Class `UCIePartition`
Partitioning plan for a neuron array across chiplet tiles.

Attributes
----------
tile_count : int
    Number of chiplet tiles used.
neurons_per_tile : int
    Neurons assigned per tile.
inter_tile_spikes : int
    Estimated spikes crossing tile boundaries per timestep.
die_to_die_bandwidth_gbps : float
    Required UCIe bandwidth (Gbps).
latency_penalty_ns : float
    Additional latency from die-to-die communication.
partition_map : dict&#91;int, list&#91;int&#93;&#93;
    Tile ID → list of neuron indices.


### Class `CXLMapping`
CXL.mem Type-3 mapping for neuron state.

Attributes
----------
device_count : int
    Number of CXL memory devices.
state_device_ids : list&#91;int&#93;
    Devices hosting neuron state.
weight_device_ids : list&#91;int&#93;
    Devices hosting synaptic weights.
total_capacity_gb : float
    Total CXL memory capacity.
host_bandwidth_gbps : float
    Required host→CXL bandwidth.
coherence_protocol : str
    CXL protocol used (``"CXL.mem"`` or ``"CXL.cache"``).


### Class `MemoryMap`
Address decoder specification for neuron arrays.

Attributes
----------
base_address : int
    Base address of neuron array.
entries : list&#91;dict&#91;str, int | str&#93;&#93;
    Address map entries.
total_bytes : int
    Total address space consumed.
decoder_verilog : str
    Generated address decoder Verilog.


### Class `FloorplanResult`
Multi-die/chiplet floorplan assignment.

Attributes
----------
die_assignment : dict&#91;str, int&#93;
    Block name → die index.
die_utilization : dict&#91;int, float&#93;
    Die index → utilization (0-1).
total_dies : int


### Class `UCIeMapping`
UCIe die-to-die protocol mapping result.

Attributes
----------
lanes : dict&#91;str, int&#93;
protocol_version : str
total_bandwidth_gbps : float


### Function `generate_cdc_synchroniser(signal_name)`
Generate a CDC (Clock Domain Crossing) synchroniser in Verilog.

Uses a multi-stage register chain to safely transfer signals between
clock domains. For multi-bit buses, use a gray-code converter or
handshake protocol instead.

Parameters
----------
signal_name : str
    Name of the signal being synchronised.
width : int
    Bit width (1 for single-bit CDC).
stages : int
    Number of synchroniser stages (2 minimum, 3 for MTBF).
src_clock : str
    Source clock name.
dst_clock : str
    Destination clock name.

Returns
-------
str
    Verilog CDC synchroniser module.

### Function `storage_recommendation(neuron_count, state_bits_per_neuron)`
Determine optimal storage strategy for a neuron array.

Decides between registers (small), BRAM (medium), and URAM (large)
based on total state bits and target capabilities.

Parameters
----------
neuron_count : int
    Number of neurons in the array.
state_bits_per_neuron : int
    State bits per neuron (e.g. 16 for Q8.8, 32 for Q16.16).
has_uram : bool
    True if the target has UltraRAM (UltraScale+ / Versal only).
register_threshold : int
    Max neurons for register-based storage.
uram_threshold : int
    Min neurons for URAM migration.

Returns
-------
StorageRecommendation
    Optimal storage strategy with resource estimates.

### Function `generate_bram_array(module_name)`
Generate a time-multiplexed BRAM-backed neuron array.

A single compute pipeline is shared across N neurons with BRAM-backed
state. The array processes one neuron per clock cycle.

Parameters
----------
module_name : str
    Module name.
neuron_count : int
    Number of neurons.
data_width : int
    Fixed-point data width.
state_vars : int
    State variables per neuron (e.g. 1 for LIF, 2 for Izhikevich).

Returns
-------
str
    Verilog module with BRAM-backed time-multiplexed neuron array.

### Function `generate_tmr_wrapper(module_name)`
Generate a Triple Modular Redundancy wrapper for any neuron module.

Instantiates three copies of the target module and a majority voter
to mask Single Event Upsets (SEUs) in aerospace/safety-critical
deployments. Compliant with DO-254 DAL-A and IEC 61508 SIL-4.

Parameters
----------
module_name : str
    Name of the inner neuron module to wrap.
data_width : int
    Data width of the inner module.
state_vars : list&#91;str&#93;, optional
    State variable names for output voting. Defaults to ``&#91;"v"&#93;``.
voter : str
    Voter type: ``"majority"`` (2-of-3) or ``"median"`` (middle value).

Returns
-------
str
    Synthesisable Verilog TMR wrapper module.

### Function `plan_pim_layout(neuron_count, synapse_count)`
Plan data placement across PIM memory banks.

Distributes neuron state and synaptic weights across memory banks
to maximise bank-level parallelism on PIM (UPMEM, Samsung HBM-PIM)
and CXL memory expander targets.

Parameters
----------
neuron_count : int
    Total neurons in the network.
synapse_count : int
    Total synaptic connections.
data_width : int
    Bits per value.
bank_size_kb : int
    Capacity of each memory bank in KB.
num_banks : int
    Number of available memory banks.
target : str
    Target platform name.

Returns
-------
PIMLayout
    Optimised memory layout plan.

### Function `generate_power_domain_wrapper(module_name)`
Generate a clock/power gating wrapper for always-on edge deployment.

Creates a wrapper module with:
- ICG (Integrated Clock Gating) cell for dynamic power reduction
- Power-down state retention latches
- Configurable wakeup latency
- Always-on domain for event detection (spike_detect)

Targets ultra-low-power edge platforms (Syntiant NDP120, Innatera
Pulsar, BrainChip Akida) where µW-level idle power is critical.

Parameters
----------
module_name : str
    Inner neuron module name.
data_width : int
    Data width.
state_vars : list&#91;str&#93;, optional
    State variables to retain. Defaults to ``&#91;"v"&#93;``.
always_on_signals : list&#91;str&#93;, optional
    Signals kept in the always-on domain. Defaults to ``&#91;"spike_out"&#93;``.
wakeup_cycles : int
    Clock cycles required to exit power-down.

Returns
-------
str
    Synthesisable Verilog power domain wrapper.

### Function `advise_ucie_partition(neuron_count, connectivity)`
Advise on neuron array partitioning across chiplet tiles.

Analyses a neuron array's connectivity to estimate inter-tile
spike traffic and UCIe bandwidth requirements when distributing
a network across multi-die chiplet systems (AMD MI300X,
Tenstorrent Galaxy, Intel Ponte Vecchio).

Parameters
----------
neuron_count : int
    Total neurons in the network.
connectivity : float
    Connection probability between any two neurons (0.0–1.0).
tile_count : int
    Number of chiplet tiles.
spike_rate_hz : float
    Average firing rate per neuron (Hz).
timestep_us : float
    Simulation timestep (µs).
ucie_lane_gbps : float
    UCIe lane bandwidth (Gbps per lane).
ucie_latency_ns : float
    UCIe die-to-die latency (ns).

Returns
-------
UCIePartition
    Partitioning plan with bandwidth and latency estimates.

### Function `advise_cxl_mapping(neuron_count, synapse_count)`
Advise on CXL.mem Type-3 device mapping for neuron state.

Plans the distribution of neuron state and synaptic weights
across CXL 3.0 Type-3 memory expander devices for large-scale
SNN simulations that exceed local DRAM capacity.

Parameters
----------
neuron_count : int
    Total neurons.
synapse_count : int
    Total synaptic connections.
data_width : int
    Bits per value.
device_capacity_gb : float
    Capacity per CXL device (GB).
max_devices : int
    Maximum CXL devices available.
access_pattern : str
    ``"streaming"`` (sequential) or ``"random"`` (scattered).

Returns
-------
CXLMapping
    Device mapping plan.

### Function `generate_pipeline_wrapper(module_name, equations)`
Generate a pipelined wrapper that inserts register stages.

Auto-computes the critical path depth and required pipeline stages
based on the target frequency, then wraps the neuron module with
input/output pipeline registers and a valid-pipeline shift register.

Parameters
----------
module_name : str
    Inner neuron module name.
equations : dict&#91;str, str&#93;
    ODE equations (for depth analysis).
data_width : int
    Data width.
target : str
    Target platform name (used for frequency lookup).
stages : int, optional
    Override pipeline stages. If None, auto-computed.

Returns
-------
str
    Synthesisable Verilog pipeline wrapper.

### Function `generate_memory_map(module_name, equations)`
Generate address decoder for multi-neuron SoC arrays.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    ODE equations (state variables define register set).
num_neurons : int
    Number of neuron instances.
data_width : int
    Register width in bits.
base_address : int
    Base address.

Returns
-------
MemoryMap
    Address map with decoder Verilog.

### Function `plan_multi_die_floorplan(blocks)`
Assign neuron blocks to chiplet/die positions.

Uses first-fit-decreasing bin packing.

Parameters
----------
blocks : dict&#91;str, int&#93;
    Block name → neuron count.
die_capacity : int
    Max neurons per die.
num_dies : int
    Available dies.

Returns
-------
FloorplanResult

### Function `map_ucie_protocol(blocks)`
Map neuron array blocks to UCIe die-to-die protocol lanes.

Parameters
----------
blocks : dict&#91;str, int&#93;
    Block name → data width in bits per cycle.
lane_bandwidth_gbps : float
    Bandwidth per UCIe lane.
protocol_version : str
    UCIe protocol version.

Returns
-------
UCIeMapping

---

## Module `compiler.intelligence.verification_and_safety`

### Class `EquivalenceSketch`
Formal equivalence proof skeleton between ODE and RTL.

Attributes
----------
module_name : str
    Module under verification.
equations : dict&#91;str, str&#93;
    Source ODE equations.
assertions : list&#91;str&#93;
    SVA assertion strings for equivalence checking.
proof_steps : list&#91;str&#93;
    Human-readable proof argument steps.
quantisation_bound : float
    Maximum quantisation error bound.


### Class `ComplianceEntry`
Single compliance requirement mapping.

Attributes
----------
req_id : str
    Requirement identifier.
standard : str
    Safety standard name.
description : str
    Requirement description.
verification : str
    How it is verified.
status : str
    ``"covered"``, ``"partial"``, or ``"gap"``.
artefact : str
    File or test that provides evidence.


### Class `StabilityResult`
ODE discretization stability analysis.

Attributes
----------
stable : bool
    True if discretization is stable.
max_eigenvalue : float
    Largest eigenvalue magnitude.
critical_dt : float
    Maximum stable timestep.
method : str
    Analysis method used.


### Class `ReliabilityEstimate`
Mean time to failure estimate.

Attributes
----------
mttf_hours : float
    Estimated MTTF in hours.
mttf_years : float
    Estimated MTTF in years.
failure_mode : str
    Dominant failure mechanism.
voltage_stress : float
    Normalised voltage stress factor.
temp_accel : float
    Arrhenius temperature acceleration factor.
mechanism_mttf_hours : dict&#91;str, float&#93;
    Per-mechanism MTTF estimates for NBTI, HCI, and TDDB.


### Class `FaultTree`
Fault Tree Analysis for safety certification.

Attributes
----------
top_event : str
    Top-level failure event.
gates : list&#91;dict&#93;
    Logic gates (AND/OR).
basic_events : list&#91;dict&#93;
    Leaf failure events with rates.
mcs : list&#91;list&#91;str&#93;&#93;
    Minimal cut sets.


### Class `CDCReport`
Clock domain crossing analysis result.

Attributes
----------
crossings : list&#91;dict&#93;
    Each crossing: signal, src_domain, dst_domain, sync_type.
violations : list&#91;str&#93;
    Unsynchronized crossings.
total_crossings : int
safe : bool


### Class `RegressionCheck`
Compilation regression check result.

Attributes
----------
metric : str
baseline : float
current : float
delta_pct : float
regression : bool


### Class `AgingPrediction`
Transistor aging prediction.

Attributes
----------
initial_fmax_mhz : float
degraded_fmax_mhz : float
degradation_pct : float
recommended_derating : float
dominant_mechanism : str
nbti_degradation_pct : float
hci_degradation_pct : float


### Class `FaultCampaignResult`
Fault injection campaign result.

Attributes
----------
total_injections : int
sdc_count : int
sdc_rate : float
critical_bits : list&#91;int&#93;
recommended_tmr_bits : list&#91;int&#93;


### Class `TimingReport`
Static timing analysis report.

Attributes
----------
critical_path : list&#91;str&#93;
critical_delay_ns : float
target_period_ns : float
slack_ns : float
timing_met : bool
recommendations : list&#91;str&#93;


### Function `generate_equivalence_sketch(module_name, equations)`
Generate a formal equivalence proof sketch for ODE→RTL translation.

Produces a structured argument that the compiled Verilog computes
the same function as the source ODE within quantisation error.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    ODE equations.
data_width : int
    Fixed-point total width.
fraction : int
    Fractional bits.

Returns
-------
EquivalenceSketch
    Proof skeleton with SVA assertions.

### Function `generate_compliance_matrix(module_name)`
Generate safety compliance matrix for certification.

Maps DO-254 / IEC 61508 / ISO 26262 requirements to SC-NeuroCore
verification artefacts.

Parameters
----------
module_name : str
    Module under certification.
standards : list&#91;str&#93;, optional
    Standards to cover. Default: all three.
has_tmr : bool
    TMR wrapper is present.
has_checksum : bool
    Model checksum is embedded.
has_sva : bool
    SVA assertions are generated.
has_provenance : bool
    Provenance chain exists.

Returns
-------
list&#91;ComplianceEntry&#93;
    Compliance matrix entries.

### Function `format_compliance_report(entries)`
Format compliance matrix as markdown.

Parameters
----------
entries : list&#91;ComplianceEntry&#93;
    From ``generate_compliance_matrix()``.

Returns
-------
str
    Markdown compliance table.

### Function `generate_bittrue_kernel(module_name, equations)`
Generate a bit-true simulation kernel matching RTL arithmetic.

Produces C (or Rust) code that computes exactly the same
fixed-point results as the generated Verilog — same truncation,
overflow, and pipeline latency.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    ODE equations.
data_width : int
    Fixed-point total width.
fraction : int
    Fractional bits.
language : str
    ``"c"`` or ``"rust"``.

Returns
-------
str
    Bit-true source code.

### Function `verify_ode_stability(equations)`
Verify numerical stability of discretized ODE system.

Uses eigenvalue analysis of the linearized system to determine
if the forward-Euler discretization is stable.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.
dt : float
    Timestep.
time_constants : dict&#91;str, float&#93;, optional
    Time constants per variable.

Returns
-------
StabilityResult
    Stability analysis result.

### Function `predict_reliability()`
Predict MTTF from voltage, temperature, and technology node.

Uses Arrhenius temperature acceleration and voltage stress factors.

Parameters
----------
voltage_v : float
    Operating voltage.
temperature_c : float
    Junction temperature (°C).
node_nm : int
    Technology node (nm).
base_mttf_hours : float
    Baseline MTTF at nominal conditions.

Returns
-------
ReliabilityEstimate
    MTTF prediction.

### Function `generate_fault_tree(module_name, equations)`
Generate FTA/FMEA for DO-254 Level A certification.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    ODE state variables (each becomes a failure point).

Returns
-------
FaultTree
    Fault tree with minimal cut sets.

### Function `generate_testbench(module_name, equations)`
Generate verification testbench for compiled neuron.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    ODE equations.
framework : str
    ``"cocotb"`` or ``"uvm"``.
num_cycles : int
    Simulation cycles.

Returns
-------
str
    Testbench source code.

### Function `analyze_cdc(equations)`
Analyze clock domain crossings in a neuron array.

Parameters
----------
equations : dict&#91;str, str&#93;
    ODE equations.
clock_domains : dict&#91;str, str&#93;, optional
    Variable → clock domain mapping. Default: all in ``clk_main``.

Returns
-------
CDCReport

### Function `check_regression(baseline, current)`
Detect performance regressions between compilations.

Parameters
----------
baseline : dict&#91;str, float&#93;
    Baseline metrics.
current : dict&#91;str, float&#93;
    Current metrics.
threshold_pct : float
    Regression threshold (%).

Returns
-------
list&#91;RegressionCheck&#93;

### Function `predict_aging(initial_fmax_mhz)`
Predict end-of-life Fmax after transistor aging.

Models NBTI and HCI degradation from temperature, voltage, and lifetime stress.

Parameters
----------
initial_fmax_mhz : float
    Initial maximum frequency.
voltage_v : float
    Operating voltage.
temperature_c : float
    Junction temperature in Celsius.
years : float
    Target lifetime in years.

Returns
-------
AgingPrediction

### Function `_require_finite_positive(value, name)`
### Function `_require_finite_non_negative(value, name)`
### Function `_require_celsius_above_absolute_zero(value, name)`
### Function `run_fault_campaign(equations, data_width)`
Run a fault injection campaign on the state register.

Parameters
----------
equations : dict&#91;str, str&#93;
    State variable equations.
data_width : int
    Total data width of state register.
num_injections : int
    Number of random bit-flip injections.
seed : int
    Random seed for reproducibility.

Returns
-------
FaultCampaignResult

### Function `verify_timing_closure(equations)`
Perform static timing analysis on the dataflow graph.

Parameters
----------
equations : dict&#91;str, str&#93;
    State variable equations.
target_freq_mhz : float
    Target clock frequency.
data_width : int
    Data width in bits.

Returns
-------
TimingReport

---

## Module `compiler.ir_type_checker`

### Class `SignalType`

### Class `IRNode`
A typed node in the Stochastic IR graph.


### Class `IREdge`

### Class `IRTypeError`
A type mismatch found during checking.


### Function `types_compatible(src, dst)`
Check if src can connect to dst without explicit conversion.

### Function `check_ir_types(nodes, edges)`
Type-check an IR graph and return all type errors.

Parameters
----------
nodes : dict mapping node name → IRNode
edges : list of IREdge connections

Returns
-------
list of IRTypeError (empty if all types check out)

---

## Module `compiler.mixed_precision`

### Class `PrecisionConfig`
Fixed-point configuration for a single variable.

Attributes
----------
data_width : int
    Total bit width.
fraction : int
    Number of fractional bits.
signed : bool
    True for signed two's complement.

- **int_bits**()
  - Number of integer bits (excluding sign).
- **max_value**()
  - Maximum representable value.
- **min_value**()
  - Minimum representable value.
- **resolution**()
  - Smallest representable step.
- **q_label**()
  - Human-readable Q-format label.
- **can_represent**(value)
  - Check if a value fits in this format without overflow.
- **encode**(value)
  - Encode a float to Q-format integer.

### Class `MixedPrecisionSpec`
Specification for mixed-precision compilation.

Maps each state variable to its own PrecisionConfig, enabling
heterogeneous datapaths in a single Verilog module.

Parameters
----------
var_configs : dict&#91;str, PrecisionConfig&#93;
    Per-variable precision configuration.

- **total_bits**()
  - Total bit count across all variables.
- **variables**()
  - List of variable names.
- **get**(var)
  - Get the precision config for a variable.
- **summary**()
  - Return a human-readable summary of the precision allocation.

### Function `_min_bits_for_range(lo, hi, signed)`
Compute minimum integer bits to cover a value range.

Parameters
----------
lo, hi : float
    The value range.
signed : bool
    Whether the format is signed.

Returns
-------
int
    Minimum integer bits needed.

### Function `_min_frac_for_resolution(resolution)`
Compute minimum fractional bits for a target resolution.

Parameters
----------
resolution : float
    Desired minimum step size.

Returns
-------
int
    Minimum fractional bits needed.

### Function `solve_precision(bounds)`
Automatically solve for optimal per-variable precision.

Uses a constraint-based approach:
1. For each variable, compute minimum integer bits from bounds
2. Compute minimum fractional bits from resolution requirements
3. If max_total_bits is set, iteratively reduce least-critical fractions

Parameters
----------
bounds : dict
    Mapping from variable name to (min, max) value bounds.
min_resolution : dict, optional
    Mapping from variable name to minimum required resolution.
    Defaults to 0.01 for all variables.
max_total_bits : int, optional
    If set, the solver will reduce fractional bits to fit.
signed : bool
    Whether to use signed formats.
align_to : int
    Align each variable's data_width to this multiple (1=no alignment,
    8=byte-align, 16=halfword-align).

Returns
-------
MixedPrecisionSpec
    Optimal per-variable precision configuration.

### Function `from_preset(var_presets)`
Create a MixedPrecisionSpec from named presets.

Parameters
----------
var_presets : dict&#91;str, str&#93;
    Mapping from variable name to preset name (e.g. ``{"v": "q88", "u": "q44"}``).

Returns
-------
MixedPrecisionSpec
    The resulting spec.

Raises
------
KeyError
    If a preset name is not found.

---

## Module `compiler.mlir_emitter`

### Class `MLIRNode`

### Class `MLIRBundle`
Generated MLIR file and evidence manifest.

- **to_dict**()

### Class `MLIREmitter`
Translates sc-neurocore objects into MLIR text formatted for CIRCT.

- **__init__**(module_name)
- **get_wire**()
- **_sanitize_ssa_name**(name, context)
- **emit_and**(lhs, rhs)
  - Emits a comb.and operation for stochastic multiplication.
- **emit_lfsr**(width, seed)
  - Emits an LFSR instantiation.
- **emit_xor**(lhs, rhs)
  - Emits a comb.xor operation.
- **emit_mux**(cond, true_val, false_val)
  - Emits a comb.mux operation (used for SC scaled addition).
- **generate**()
  - Generates the final MLIR string for the module.
- **write_bundle**(output_dir)
  - Write MLIR plus a manifest describing CIRCT lowering readiness.

### Function `generate_mlir_bundle(emitter, output_dir)`
Write a CIRCT-ready MLIR file and reproducibility manifest.

---

## Module `compiler.pipeline`

### Class `CompilerPipeline`
Automated hardware synthesis pipeline.

- **__init__**(work_dir)
- **_sanitize_name**(name)
  - Restrict output_name to alphanumeric + underscore.
- **compile_mlir_to_verilog**(mlir_content, output_name)
  - Invokes 'firtool' to lower MLIR to Verilog.
- **_validate_path**(path)
  - Ensure path resolves inside work_dir.
- **run_synthesis**(v_path, target_fpga)
  - Invokes 'yosys' for synthesis.
- **run_pnr**(json_path, target_device)
  - Invokes 'nextpnr' for place and route.

---

## Module `compiler.platforms.registry`

### Class `HardwareProfile`
Complete hardware configuration for a target platform.

Attributes
----------
name : str
    Short machine-readable identifier (e.g. ``"loihi2"``).
vendor : str
    Chip vendor (e.g. ``"Intel"``, ``"Xilinx"``).
family : str
    Product family (e.g. ``"Arria 10"``, ``"ECP5"``).
platform_class : str
    One of ``"fpga"``, ``"neuromorphic"``, ``"asic"``, ``"simulation"``.
data_width : int
    Total bit width for fixed-point arithmetic.
fraction : int
    Number of fractional bits.
signed : bool
    True for signed (two's complement), False for unsigned Q-format.
overflow : OverflowMode
    How to handle arithmetic overflow in next-state logic.
rounding : RoundingMode
    How to round after fixed-point multiplication truncation.
dsp_block : str
    Name of the DSP hard macro (e.g. ``"DSP48E2"``).
dsp_mult_a : int
    Width of the DSP A-port (multiplier input A).
dsp_mult_b : int
    Width of the DSP B-port (multiplier input B).
max_freq_mhz : int | None
    Typical maximum clock frequency (0 = unknown).
notes : str
    Human-readable rationale for the configuration.

- **int_bits**()
  - Number of integer bits (excluding sign bit if signed).
- **q_format_label**()
  - Human-readable Q-format string (e.g. ``'Q9.9'`` or ``'UQ8.8'``).
- **max_value**()
  - Maximum representable positive value.
- **min_value**()
  - Minimum representable value (most negative or zero).
- **resolution**()
  - Smallest representable step.
- **from_constraints**(cls, name)
  - Auto-construct an optimal profile from spec-sheet constraints.

### Function `_reg(p)`
Register a profile in the global registry.

### Function `get_profile(name)`
Look up a hardware profile by name.

Parameters
----------
name : str
    Case-insensitive profile name (e.g. ``"loihi2"``, ``"artix7"``).

Returns
-------
HardwareProfile
    The matching profile.

Raises
------
KeyError
    If no profile matches.

### Function `list_profiles()`
List all registered hardware profiles, optionally filtered.

Parameters
----------
platform_class : str, optional
    Filter by class: ``"fpga"``, ``"neuromorphic"``, ``"asic"``, ``"simulation"``
    ``"accelerator"``, ``"dsp"``, ``"photonic"``, ``"in_memory"``, ``"emerging"``.
vendor : str, optional
    Filter by vendor name (case-insensitive substring match).

Returns
-------
list&#91;HardwareProfile&#93;
    Matching profiles, sorted by (platform_class, vendor, name).

### Function `list_profile_names()`
Return all registered profile names, sorted.

### Function `load_toml_profile(path)`
Load a user-defined hardware profile from a TOML file.

Enables users to register custom hardware targets without modifying
the SC-NeuroCore source. TOML format::

    &#91;profile&#93;
    name = "my_chip"
    vendor = "My Corp"
    family = "ChipNet-1"
    platform_class = "accelerator"
    data_width = 16
    fraction = 8
    overflow = "saturate"
    rounding = "nearest"
    max_freq_mhz = 500
    dsp_block = "MAC"
    dsp_mult_a = 16
    dsp_mult_b = 16
    notes = "Custom chip description."

Parameters
----------
path : str
    Path to the TOML file.

Returns
-------
HardwareProfile
    The loaded and registered profile.

Raises
------
FileNotFoundError
    If the TOML file does not exist.
ValueError
    If required fields are missing.

### Function `load_toml_profiles_dir(directory)`
Load all TOML profiles from a directory.

Scans the directory for ``*.toml`` files and loads each as a hardware
profile. Useful for bulk-registering custom targets.

Parameters
----------
directory : str
    Path to the directory containing TOML profile files.

Returns
-------
list&#91;HardwareProfile&#93;
    All loaded profiles.

### Function `load_profiles_from_toml(path)`
Load custom hardware profiles from a TOML file.

Allows users and vendors to define profiles without modifying
SC-NeuroCore source code. This is the profile-extension path.

TOML format::

    &#91;&#91;profile&#93;&#93;
    name = "my_custom_chip"
    vendor = "MyVendor"
    family = "CustomFamily"
    platform_class = "custom"
    data_width = 16
    fraction = 8
    overflow = "saturate"
    rounding = "nearest"

Parameters
----------
path : str
    Path to TOML file.

Returns
-------
list&#91;str&#93;
    Names of loaded profiles.

### Function `register_platform_hook(hook_fn)`
Register a third-party platform discovery function.

The hook function should return a list of HardwareProfile instances
when called with no arguments. Profiles are registered at runtime.

Parameters
----------
hook_fn : callable
    Function returning list&#91;HardwareProfile&#93;.

### Function `discover_platforms()`
Execute all registered discovery hooks.

Returns
-------
list&#91;str&#93;
    Names of newly discovered profiles.

---

## Module `compiler.quantizer`

### Class `QFormat`
Fixed-point Q-format specification.

- **total_bits**()
- **scale**()
- **min_val**()
- **max_val**()
- **from_string**(cls, fmt)
  - Parse 'Q8.8', 'Q4.12', etc.

### Function `quantize_weights(weights, fmt, rounding, clip)`
Quantize float weights to fixed-point integers.

Parameters
----------
weights : np.ndarray
    Float weight matrix (any shape).
fmt : str
    Q-format string, e.g. "Q8.8" (8 integer + 8 fractional = 16-bit signed).
rounding : str
    "nearest" (round half to even), "stochastic" (probabilistic rounding),
    or "floor" (truncate toward negative infinity).
clip : bool
    If True, clip values to the representable range before quantization.

Returns
-------
np.ndarray
    Integer array (same shape) in the Q-format representation.
    To recover the float: result / (2^fraction_bits).

### Function `dequantize_weights(quantized, fmt)`
Convert quantized integer weights back to float.

### Function `q_weights_to_sc_probabilities(quantized, fmt)`
Convert quantized weights to SC probabilities in &#91;0, 1&#93;.

Maps the Q-format range &#91;min, max&#93; linearly to &#91;0, 1&#93; for
unipolar SC encoding.

### Function `quantization_error(weights, fmt, rounding)`
Compute quantization error statistics.

Returns
-------
dict with keys: max_abs_error, mean_abs_error, rmse, snr_db

---

## Module `compiler.static_analysis`

### Class `Interval`
A closed interval &#91;lo, hi&#93; for interval arithmetic.

- **__add__**(other)
  - Add two intervals: &#91;a,b&#93; + &#91;c,d&#93; = &#91;a+c, b+d&#93;.
- **__sub__**(other)
  - Subtract intervals: &#91;a,b&#93; - &#91;c,d&#93; = &#91;a-d, b-c&#93;.
- **__mul__**(other)
  - Multiply intervals: all four products, take min/max.
- **__truediv__**(other)
  - Divide intervals. Raises if divisor contains zero.
- **__neg__**()
  - Negate: -&#91;a,b&#93; = &#91;-b, -a&#93;.
- **contains**(lo, hi)
  - Check if this interval is contained within &#91;lo, hi&#93;.

### Class `_IntervalEvaluator`
Evaluate an AST expression using interval arithmetic.

- **__init__**(bounds)
  - Initialise with variable bounds.
- **visit_BinOp**(node)
  - Evaluate a binary operation on intervals.
- **visit_UnaryOp**(node)
  - Evaluate a unary operation on an interval.
- **visit_Name**(node)
  - Look up the interval for a variable name.
- **visit_Constant**(node)
  - A constant is a point interval &#91;c, c&#93;.
- **generic_visit**(node)
  - Raise for unsupported nodes.

### Class `OverflowProofResult`
Result of a formal overflow proof.

Attributes
----------
proven_safe : bool
    True if the expression provably cannot overflow at the given precision.
expr_interval : Interval
    The computed output interval of the expression.
q_min : float
    Minimum representable value in the Q-format.
q_max : float
    Maximum representable value in the Q-format.
margin_lo : float
    How far the expression minimum is from the Q-format minimum (positive = safe).
margin_hi : float
    How far the expression maximum is from the Q-format maximum (positive = safe).


### Class `PowerEstimate`
Estimated power consumption for a compiled neuron.

Attributes
----------
dynamic_mw : float
    Dynamic (switching) power in milliwatts.
static_mw : float
    Leakage power in milliwatts.
total_mw : float
    Total power (dynamic + static).
energy_per_spike_nj : float
    Energy per spike event in nanojoules.
toggle_rate : float
    Average toggle rate (transitions per clock per bit).


### Function `_count_additions(expr_str)`
Count the number of addition/subtraction nodes in an expression AST.

Parameters
----------
expr_str : str
    A Python-syntax arithmetic expression.

Returns
-------
int
    Number of Add/Sub operations in the AST.

### Function `compute_guard_bits(expr_str)`
Compute the number of guard bits needed for safe accumulation.

When summing N values, the accumulator needs ``ceil(log2(N+1))``
extra MSBs to guarantee no intermediate overflow. For a single
addition (a + b), 1 guard bit suffices. For ``a + b + c + d``,
2 guard bits are needed.

Parameters
----------
expr_str : str
    A Python-syntax ODE right-hand-side expression.

Returns
-------
int
    Minimum number of guard bits needed (0 if no additions).

Examples
--------
>>> compute_guard_bits("a + b")
1
>>> compute_guard_bits("a + b + c + d")
2
>>> compute_guard_bits("a * b")
0

### Function `compute_guard_bits_multi(equations)`
Compute guard bits for every state variable in a multi-ODE system.

Parameters
----------
equations : dict
    Mapping from variable name to RHS expression string.

Returns
-------
dict
    Mapping from variable name to required guard bits.

### Function `prove_no_overflow(expr_str, bounds, data_width, fraction, signed)`
Statically prove that an expression cannot overflow at a given precision.

Uses interval arithmetic to compute the range of possible output values,
then checks whether the output interval fits within the Q-format range.

Parameters
----------
expr_str : str
    Python-syntax arithmetic expression.
bounds : dict
    Mapping from variable name to (min, max) bounds.
data_width : int
    Total bit width.
fraction : int
    Fractional bits.
signed : bool
    Whether the format is signed.

Returns
-------
OverflowProofResult
    Contains ``proven_safe=True`` if the proof succeeds.

### Function `generate_sva(state_vars)`
Generate SystemVerilog Assertions for a compiled neuron module.

Produces three categories of formal properties:

1. **Overflow assertions** — check that no state variable exceeds
   the representable range after the next-state update.
2. **Reachability covers** — prove that spike output is reachable.
3. **Input assumptions** — constrain external inputs to valid bounds.

Parameters
----------
state_vars : list&#91;str&#93;
    Names of state variables (e.g. ``&#91;"v"&#93;``).
data_width : int
    Bit width of the fixed-point format.
fraction : int
    Fractional bits.
signed : bool
    True for signed format.
input_bounds : dict, optional
    Mapping from input names to (min_q, max_q) bounds in Q-format integers.
module_name : str
    Name of the target module.

Returns
-------
str
    SystemVerilog bind module with assertions.

### Function `_mul_div_depth(node)`
Return the longest chain of Mult/Div operations from root to leaf.

Parameters
----------
node : ast.AST
    Root AST node.

Returns
-------
int
    Maximum multiplicative depth.

### Function `critical_path_depth(expr_str)`
Count the longest chain of Mult/Div nodes in an expression.

This determines how many DSP blocks are chained in series in the
resulting Verilog datapath. At high frequencies, each DSP in series
adds ~2.5 ns of combinational delay.

Parameters
----------
expr_str : str
    Python-syntax arithmetic expression.

Returns
-------
int
    Maximum multiply/divide chain length (0 = no multiplies).

Examples
--------
>>> critical_path_depth("a * b + c")
1
>>> critical_path_depth("a * b * c * d")
3
>>> critical_path_depth("a + b")
0

### Function `pipeline_stages_needed(depth, target_freq_mhz, dsp_delay_ns, routing_overhead_ns)`
Compute how many pipeline registers are needed between DSP blocks.

Parameters
----------
depth : int
    Critical path depth (from ``critical_path_depth()``).
target_freq_mhz : int
    Target clock frequency in MHz.
dsp_delay_ns : float
    Propagation delay per DSP multiply (default 2.5 ns for Artix-7).
routing_overhead_ns : float
    Routing overhead per stage.

Returns
-------
int
    Number of pipeline registers to insert (0 = no pipelining needed).

### Function `pipeline_analysis(equations, target_freq_mhz, dsp_delay_ns)`
Analyse pipeline requirements for a multi-ODE system.

Parameters
----------
equations : dict
    Variable name → RHS expression.
target_freq_mhz : int
    Target clock frequency.
dsp_delay_ns : float
    DSP propagation delay.

Returns
-------
dict
    Per-variable analysis: ``{var: {"depth": int, "stages": int,
    "achievable_mhz": int}}``.

### Function `_load_vcd_text(activity_vcd)`
### Function `_bit_toggle_count(previous, current, width)`
### Function `_parse_vcd_activity(activity_vcd, time_units_per_cycle)`
Return measured toggles per cycle and average bit toggle rate from VCD.

### Function `estimate_power(verilog)`
Estimate power consumption from generated Verilog.

When a VCD trace is provided, dynamic power uses measured bit-level
switching activity. Otherwise the function falls back to structural
activity factors derived from registers, adders, multipliers, and
technology parameters.

Parameters
----------
verilog : str
    Generated Verilog source.
data_width : int
    Fixed-point data width.
freq_mhz : float
    Clock frequency in MHz.
vdd : float
    Supply voltage (V).
process_nm : int
    Process node in nm (7, 16, 28, 45, ...).
spike_rate_hz : float
    Expected average spike rate.
activity_vcd : str or Path, optional
    VCD trace text or a path to a VCD file. When provided, bit-level
    transitions in the trace drive dynamic power.
vcd_time_units_per_cycle : float
    Number of VCD timestamp units per target clock cycle.

Returns
-------
PowerEstimate
    Estimated power breakdown.

---

## Module `compiler_service`

### Class `LiveUpdateKind`
Kinds of update package a compiler service may emit.


### Class `DigitalTwinSyncContract`
Digital-twin sync requirements for a compiler-service request.

- **__post_init__**()
- **to_dict**()
  - Return a JSON-compatible sync contract.

### Class `LiveUpdatePolicy`
Policy deciding whether a compiler update needs re-synthesis.

- **classify**(changed_fields)
  - Classify changed fields into the minimum required update kind.
- **to_dict**()
  - Return a JSON-compatible policy manifest.

### Class `CompilerServiceRequest`
One request accepted by the compiler-service contract boundary.

- **__post_init__**()
- **to_dict**()
  - Return a deterministic request manifest.

### Class `LiveUpdatePackage`
Update package planned for a live FPGA/digital-twin loop.

- **to_dict**()
  - Return a JSON-compatible package summary.

### Class `CompilerServiceResponse`
Deterministic response from the contract boundary.

- **to_dict**()
  - Return a JSON-compatible compiler-service response.

### Function `build_compiler_service_contract()`
Build a deterministic compiler-service boundary manifest.

### Function `plan_live_update(request, policy)`
Classify a request into a live-update package.

### Function `build_compiler_service_response(request)`
Build a response without invoking a network service or toolchain.

### Function `_response_diagnostics(update_package)`
### Function `_target_key(target)`
### Function `_target_to_dict(target)`
### Function `_budget_to_dict(budget)`
### Function `_layer_to_dict(layer)`
### Function `_report_to_dict(report)`
### Function `_surrogate_layer_config_to_dict(config)`
---

## Module `compression.pruning`

### Class `PruningReport`
Results of a pruning operation.


### Function `prune_weights(weights, threshold, method)`
Prune small weights from layer weight matrices.

Parameters
----------
weights : list of ndarray
    Weight matrices for each layer.
threshold : float
    Pruning threshold. Weights with |w| <= threshold are zeroed.
method : str
    'magnitude' (default): prune by absolute value.
    'percentile': treat threshold as percentile (0-100) of weight
    magnitudes to prune.

Returns
-------
(pruned_weights, PruningReport)

### Function `prune_neurons(weights, firing_rates, activity_threshold)`
Structural pruning: remove neurons with low firing rates.

Removes entire rows from weight matrices (output neurons) and
corresponding columns from the next layer's weight matrix (input
connections). Reduces layer width, not just sparsity.

Parameters
----------
weights : list of ndarray
    Weight matrices &#91;W1, W2, ...&#93; where W_i has shape (n_out, n_in).
firing_rates : list of ndarray, optional
    Per-neuron firing rates for each layer. If None, uses output
    weight magnitude as a proxy for importance.
activity_threshold : float
    Neurons with firing rate (or weight norm) below this are pruned.

Returns
-------
(pruned_weights, PruningReport)

### Function `prune_stochastic(weights, bitstream_length, min_popcount_bits)`
Stochastic-aware pruning: score weights by bitstream contribution.

In SC networks, weight w encodes probability p = clip(|w|, 0, 1).
The expected popcount contribution per inference is:
    contribution = min(p, 1-p) * bitstream_length

Weights that produce nearly-deterministic bitstreams (p near 0 or 1)
contribute almost nothing to computation — they can be replaced with
constant 0/1 gates, saving AND+popcount hardware.

Parameters
----------
weights : list of ndarray
    Weight matrices (values in &#91;0, 1&#93; for unipolar SC).
bitstream_length : int
    Bitstream length (L). Longer streams = more bits per weight.
min_popcount_bits : float
    Minimum expected popcount contribution to keep a weight.
    Weights contributing fewer bits than this are zeroed.

Returns
-------
(pruned_weights, PruningReport)

---

## Module `compression.quantization`

### Function `quantize_weights(weights, bits, symmetric)`
Quantize weight matrices to fixed-point with given bit width.

Parameters
----------
weights : list of ndarray
    Float weight matrices.
bits : int
    Target bit width (default 8). Range: &#91;2, 16&#93;.
symmetric : bool
    Symmetric quantization around zero (default True).

Returns
-------
list of ndarray
    Quantized weights (still float dtype but with discrete values).

### Function `quantize_delays(delays, resolution, max_delay)`
Quantize continuous delays to integer grid.

Parameters
----------
delays : ndarray
    Continuous delay values.
resolution : int
    Delay step size (default 1). Resolution=2 means delays are
    rounded to {0, 2, 4, 6, ...}, halving the buffer depth.
max_delay : int, optional
    Clamp delays to this maximum.

Returns
-------
ndarray of int
    Quantized integer delays.

---

## Module `continual.engine`

### Class `PlasticityConfig`
Per-layer on-chip plasticity configuration.

Extracted from training for hardware deployment.

Parameters
----------
layer_name : str
rule : str
    Plasticity rule: 'stdp', 'r_stdp', 'homeostatic', 'none'.
tau_pre : float
    Pre-synaptic trace time constant (ms).
tau_post : float
    Post-synaptic trace time constant (ms).
lr_potentiation : float
    Potentiation learning rate (A+).
lr_depression : float
    Depression learning rate (A-).
w_min : float
    Minimum weight.
w_max : float
    Maximum weight.
homeostatic_target : float
    Target firing rate for homeostatic regulation.


### Class `ContinualReport`
Report from a continual learning session.

- **summary**()

### Class `ContinualLearner`
Continual learning engine with EWC and on-chip plasticity extraction.

Parameters
----------
weights : list of ndarray
    Initial trained weight matrices per layer.
layer_names : list of str
    Names for each layer.
ewc_lambda : float
    Regularization strength for EWC (0 = no protection).
plasticity_rule : str
    Default on-chip plasticity rule for all layers.

- **__init__**(weights, layer_names, ewc_lambda, plasticity_rule)
- **compute_fisher**(gradients_per_sample)
  - Compute Fisher Information diagonal from per-sample gradients.
- **ewc_penalty**()
  - Compute EWC regularization penalty.
- **register_task**(accuracy)
  - Register completion of a task.
- **update_weights**(new_weights)
  - Update weights (e.g., after training on a new task).
- **extract_plasticity_configs**()
  - Extract per-layer plasticity parameters for on-chip deployment.
- **report**()
  - Generate a continual learning report.

---

## Module `contrastive.ssl`

### Class `SpikeContrastiveLoss`
InfoNCE contrastive loss adapted for spike representations.

Computes similarity between spike-rate vectors from two augmented
views of the same input. Positive pairs = same input, different
augmentation. Negative pairs = different inputs.

Parameters
----------
temperature : float
    Contrastive temperature scaling.

- **__init__**(temperature)
- **compute**(view_a, view_b)
  - Compute contrastive loss for a batch of spike-rate pairs.

### Class `CSDPRule`
Contrastive Signal-Dependent Plasticity.

Local learning rule: weight update depends on (pre, post, contrastive_signal).
Positive phase: present real data → Hebbian update.
Negative phase: present corrupted data → anti-Hebbian update.

Generalizes Forward-Forward to spiking circuits.

Reference: Ororbia 2024, Science Advances

Parameters
----------
lr : float
    Learning rate.
decay : float
    Weight decay for regularization.

- **positive_update**(weights, pre_spikes, post_spikes)
  - Hebbian update from positive (real) data.
- **negative_update**(weights, pre_spikes, post_spikes)
  - Anti-Hebbian update from negative (corrupted) data.
- **contrastive_step**(weights, pos_pre, pos_post, neg_pre, neg_post)
  - Full contrastive update: positive + negative phase.
- **goodness**(activations)
  - Compute 'goodness' score (sum of squared activations).

---

## Module `control.adaptive_loop`

### Class `AdaptationEvent`
Record of a single adaptation cycle.


### Class `AdaptiveLoopConfig`
Configuration for the adaptive controller.


### Class `AdaptiveController`
Closed-loop: Runtime drift → Optimizer SA → New config.

Usage::

    ctrl = AdaptiveController(budget, layers)
    for bitstream_pair in stream:
        event = ctrl.step(bitstream_pair)
        if event and event.config_changed:
            apply_new_config(ctrl.current_config)

- **__init__**(budget, layers, config)
- **step**(bitstream_a, bitstream_b)
  - Feed a bitstream pair; returns AdaptationEvent if re-optimisation triggered.
- **adaptation_rate**()
  - Fraction of steps that triggered re-optimisation.
- **summary**()

---

## Module `control.controllers`

### Class `SpikingPID`
Population-coded PID controller.

Error → rate-coded spike populations → P/I/D populations →
output current. Gains are synaptic weights.

Parameters
----------
Kp, Ki, Kd : float
    PID gains (encoded as synaptic weights).
n_neurons : int
    Population size per channel.
dt : float
    Timestep.

- **__init__**(Kp, Ki, Kd, n_neurons, dt)
- **step**(error)
  - Compute PID output for one timestep.
- **step_spike**(error, rng)
  - Compute PID output as spike population.
- **reset**()

### Class `SpikingKalmanFilter`
Spike-domain Kalman filter for state estimation.

State prediction and correction using LIF-based integration.
Kalman gain encoded as synaptic weight matrix.

Parameters
----------
n_states : int
    State dimension.
n_measurements : int
    Measurement dimension.
A : ndarray
    State transition matrix.
H : ndarray
    Observation matrix.
Q : ndarray
    Process noise covariance.
R : ndarray
    Measurement noise covariance.

- **__init__**(n_states, n_measurements, A, H, Q, R)
- **predict**()
  - Predict step: x = A @ x, P = A @ P @ A^T + Q.
- **update**(z)
  - Update step with measurement z.
- **step**(z)
  - Predict + update in one call.
- **reset**()

### Class `SpikingLQR`
Spike-domain Linear Quadratic Regulator.

Computes optimal gain K from system matrices (A, B, Q, R).
Control law: u = -K @ x. Weights derived analytically.

Parameters
----------
A : ndarray (n, n) — state transition
B : ndarray (n, m) — control input
Q : ndarray (n, n) — state cost
R : ndarray (m, m) — control cost

- **__init__**(A, B, Q, R)
- **_solve_dare**(max_iter)
  - Solve discrete algebraic Riccati equation iteratively.
- **control**(x)
  - Compute optimal control: u = -K @ x.
- **gain_matrix**()

---

## Module `control.sc_runtime`

### Class `DecorrelatorType`

### Class `ECCMode`

### Class `ActivityZone`

### Class `RuntimeConfig`
Current runtime configuration for a bitstream engine.

- **effective_length**()
- **copy**()

### Class `AdaptationEvent`
Log entry for a runtime adaptation.


### Class `ActivityMonitor`
Rolling-window bitstream activity tracker.

Maintains running statistics of popcount density and SCC for
drift detection.

- **__init__**(window_size, drift_threshold)
- **observe**(bitstream, reference)
  - Record one observation.
- **_compute_scc**(a, b)
- **mean_density**()
- **mean_scc**()
- **drift_active**()
- **current_zone**()

### Class `HammingECC`
Hamming(7,4) encoder/decoder (Python mirror of Rust ScDoctor ECC).

- **encode**(data_4bit)
  - Encode 4-bit data to 7-bit Hamming codeword.
- **decode**(encoded_7bit)
  - Decode 7-bit Hamming codeword to 4-bit data, correcting 1-bit errors.
- **encode_bitstream**(bitstream)
  - Apply Hamming(7,4) ECC to an entire bitstream (groups of 4 bits).
- **decode_bitstream**(encoded)
  - Decode Hamming(7,4) encoded bitstream.

### Class `SECDEC_ECC`
SECDED (Single Error Correct, Double Error Detect) Hamming(8,4).

Extends Hamming(7,4) with an overall parity bit for 2-bit error
detection.  Corrects all 1-bit errors, detects all 2-bit errors.

- **__init__**()
- **encode**(data_4bit)
  - Encode 4-bit data to 8-bit SECDED codeword.
- **decode**(encoded_8bit)
  - Decode 8-bit SECDED codeword.
- **encode_bitstream**(bitstream)
  - Apply SECDED(8,4) to an entire bitstream.
- **decode_bitstream**(encoded)
  - Decode SECDED(8,4) encoded bitstream.

### Class `AdaptationPolicy`
Rules for runtime bitstream adaptation.

- **__init__**(scc_high, scc_low, min_length, max_length, ecc_trigger_length, enable_decorrelator_cascade)
- **decide**(config, metrics)
  - Evaluate metrics and return (new_config, trigger_reason_or_None).
- **_next_decorrelator**(current)
  - Get next decorrelator in escalation cascade.

### Class `RuntimeReport`
Summary report from a runtime session.

- **num_adaptations**()
- **adaptation_rate**(last_n)
  - Adaptations per observation (optionally over last_n events).
- **summary**()

### Class `SCRuntimeEngine`
Main runtime adapter: monitors + adapts + applies ECC.

- **__init__**(initial_config, policy, monitor_window)
- **observe**(bitstream, reference)
  - Feed one bitstream observation through the runtime.
- **protect**(bitstream)
  - Apply ECC protection if enabled in current config.
- **recover**(encoded)
  - Decode ECC-protected bitstream if enabled.
- **protect_batch**(bitstreams)
  - Apply ECC to a batch of bitstreams.
- **recover_batch**(encoded_list)
  - Decode a batch of ECC-protected bitstreams.

### Function `classify_activity(density)`
Map popcount density to activity zone.

---

## Module `conversion.ann_to_snn`

### Class `ConvertedSNN`
Rate-coded SNN converted from an ANN.

Attributes
----------
weights : list of ndarray
    Per-layer weight matrices.
biases : list of ndarray or None
    Per-layer biases (None if absent).
thresholds : list of float
    Per-layer firing thresholds after normalization.
T : int
    Number of simulation timesteps.
n_layers : int
    Number of layers.

- **__post_init__**()
- **run**(x)
  - Run the converted SNN for T timesteps on input x.
- **classify**(x)
  - Run SNN and return predicted class indices.

### Function `_extract_layers(model)`
Extract (weight, bias) pairs from a PyTorch Sequential model.

### Function `_compute_max_activations(model, calibration_data, percentile)`
Run calibration data through model, record per-layer max activation.

### Function `convert(model, calibration_data, T, percentile)`
Convert a trained PyTorch ANN to a rate-coded SNN.

Parameters
----------
model : nn.Module
    Trained PyTorch model (Sequential with Linear + ReLU).
calibration_data : Tensor, optional
    Sample input batch for threshold calibration. If None, uses
    default threshold of 1.0 per layer.
T : int
    Number of simulation timesteps (higher = more accurate, slower).
percentile : float
    Activation percentile for threshold normalization.

Returns
-------
ConvertedSNN
    Converted spiking network ready to run.

---

## Module `conversion.qcfs`

### Class `QCFSActivation`
QCFS activation: quantized clip-floor-shift ReLU replacement.

For T timesteps and threshold theta:
    QCFS(x) = clip(floor(x * T / theta + 0.5), 0, T) * theta / T

This quantizes activations to T+1 levels in &#91;0, theta&#93;, matching
the achievable spike rates of an IF neuron over T timesteps.

Parameters
----------
T : int
    Number of simulation timesteps.
theta : float
    Firing threshold (default 1.0).
learn_theta : bool
    Make threshold trainable (default False).

- **__init__**(T, theta, learn_theta)
- **forward**(x)
- **extra_repr**()

---

## Module `core.bipolar`

### Function `_validate_bitstream_length(length)`
### Function `_as_float_array(values, name)`
### Function `_as_bit_array(bits, name)`
### Function `bipolar_encode(value, L, rng)`
Encode a bipolar value in &#91;-1, 1&#93; as a Bernoulli bitstream.

p = (value + 1) / 2.  Bitstream has P(bit=1) = p.

### Function `bipolar_decode(bits)`
Decode a bitstream back to bipolar value in &#91;-1, 1&#93;.

value = 2 * mean(bits) - 1.

### Function `bipolar_multiply(a, b)`
XNOR gate: bipolar multiplication.

XNOR(a, b) = NOT(XOR(a, b)) = 1 when a == b, 0 when a != b.
E&#91;XNOR&#93; decodes to v_a * v_b in bipolar domain.

### Function `bipolar_mac(inputs, weights, L, seed)`
Bipolar multiply-accumulate: weighted sum via XNOR + popcount.

Parameters
----------
inputs : (N,) float array, values in &#91;-1, 1&#93;
weights : (M, N) float array, values in &#91;-1, 1&#93;
L : int, bitstream length
seed : int

Returns
-------
(M,) float array, dot product results (sum of N bipolar products)

### Function `bipolar_sc_layer(inputs, weights, bias, L, seed, activation)`
Single SC layer: bipolar MAC + optional bias + activation.

Parameters
----------
inputs : (N,) float, normalised to &#91;-1, 1&#93;
weights : (M, N) float, normalised to &#91;-1, 1&#93;
bias : (M,) float or None
L : bitstream length
activation : "relu", "none", or "tanh"

Returns
-------
(M,) float, layer output in &#91;-1, 1&#93;

### Function `float_to_bipolar_weights(weight_tensor)`
Normalise float weights to &#91;-1, 1&#93; for bipolar SC.

Preserves sign information (unlike unipolar to_sc_weights).

---

## Module `core.mdl_parser`

### Class `MDLSpecification`
Serializable MDL payload containing architecture and state sections.


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
  - Register a named module object for later pipeline execution.
- **set_attention**(module_name)
  - Focuses resources on a specific module.
- **execute_pipeline**(pipeline, initial_input)
  - Executes a sequence of modules.

---

## Module `core.tensor_stream`

### Class `TensorStream`
Unified Data Structure for sc-neurocore.
Handles automatic conversion between domains.

- **from_prob**(cls, probs)
  - Create a tensor stream whose data is already in probability form.
- **to_bitstream**(length)
  - Convert probability-domain data into Bernoulli bitstreams.
- **to_prob**()
  - Convert supported domains into probability-domain tensors.
- **to_quantum**()
  - Convert probability-domain data into two-amplitude quantum encoding.

---

## Module `core.types`

### Class `DecorrelationStrategy`
Decorrelation strategies for stochastic bitstreams.


### Class `ComputeMode`
Compute mode for a layer.


### Class `NeuronType`
Supported neuron models.


### Class `HardwareBudget`
Unified FPGA resource budget (shared by Optimizer, NAS, Runtime).

- **utilisation**(luts, ffs, bram, dsp)
  - Return resource utilisation fractions against this budget.

### Class `ResourceReport`
Unified resource consumption report.

- **meets_budget**(budget)
  - Return True when all tracked resources fit within a budget.
- **summary**()
  - Render the resource report as a compact human-readable string.

### Class `LayerSpec`
Layer specification usable by Optimizer, NAS, and Runtime.

This is the shared representation — each subsystem can extend it
but all core resource estimation uses this base.

- **estimate_luts**()
  - Unified LUT estimation shared across all subsystems.
- **estimate_power_mw**()
  - Estimate layer power in milliwatts from mode and workload size.
- **estimate_accuracy**()
  - Estimate stochastic-computing accuracy from mode and bitstream length.

### Function `estimate_network(layers)`
Estimate total resources for a network of layers.

---

## Module `dashboard.text_dashboard`

### Class `SCDashboard`
Simple CLI Dashboard for monitoring SC simulation.

- **__init__**(n_neurons)
- **update**(firing_rates, step)
- **_render**(step)

---

## Module `datasets.encoding`

### Function `poisson_encode(rates, T, dt_ms, seed)`
Convert firing-rate array to Poisson spike trains.

Parameters
----------
rates : array_like, shape (N,)
    Firing probabilities per timestep, clipped to &#91;0, 1&#93;.
T : int
    Number of timesteps.
dt_ms : float
    Timestep duration in ms (scales rates linearly).
seed : int or None
    RNG seed for reproducibility.

Returns
-------
spikes : ndarray, shape (T, N), dtype bool

### Function `latency_encode(values, T, tau, strict)`
Convert normalised values in &#91;0, 1&#93; to first-spike-time trains.

Higher values spike earlier. Each neuron fires exactly once.

Parameters
----------
values : array_like, shape (N,)
    Input values, expected in ``&#91;0, 1&#93;``.
T : int
    Number of timesteps.
tau : float
    Time constant controlling the spike-time spread.
strict : bool
    If True (default), raise ``ValueError`` when any value lies
    outside ``&#91;0, 1&#93;``. If False, silently clip the resulting
    spike times to ``&#91;0, T-1&#93;`` (the legacy behaviour). The
    clip happens regardless of ``strict``; this flag controls
    only whether the function raises before clipping.

Returns
-------
spikes : ndarray, shape (T, N), dtype bool

Raises
------
ValueError
    If ``strict=True`` (default) and any element of ``values``
    is outside ``&#91;0, 1&#93;``.

---

## Module `datasets.loaders`

### Function `_synthetic_event_dataset(n_samples, spatial_size, n_classes, T, dt_ms, seed)`
Generate synthetic Poisson-encoded event samples.

Each class gets a distinct random rate template. Events are returned
as (N_events, 4) arrays with columns &#91;x, y, polarity, timestamp_ms&#93;.

### Function `_check_root(root, dataset_name, url)`
Raise FileNotFoundError if *root* does not exist.

### Function `load_nmnist(root, train, dt_ms, T, synthetic, n_samples, seed)`
Load N-MNIST spiking vision dataset.

Neuromorphic-MNIST: 34x34 DVS recordings of MNIST digits moved on
an ATIS sensor via saccadic eye movements. 10 classes.

Orchard et al., "Converting Static Image Datasets to Spiking
Neuromorphic Datasets Using Saccades", Front. Neurosci. 2015.

Parameters
----------
root : path
    Directory containing the extracted dataset.
train : bool
    Load training split if True, test split otherwise.
dt_ms : float
    Temporal resolution for synthetic fallback.
T : int
    Number of timesteps for synthetic fallback.
synthetic : bool
    Force synthetic data generation.
n_samples : int
    Number of synthetic samples to generate.
seed : int
    RNG seed for reproducible synthetic data.

Returns
-------
samples : list of ndarray, each shape (N_events, 4)
    Columns: &#91;x, y, polarity, timestamp_ms&#93;.
labels : ndarray of int

### Function `_parse_nmnist_bin(path, dt_ms)`
Parse a single N-MNIST .bin file into (N, 4) event array.

### Function `load_shd(root, train, dt_ms, T, synthetic, n_samples, seed)`
Load Spiking Heidelberg Digits (SHD) dataset.

Audio digits 0-9 in English and German, spike-encoded through an
artificial cochlea model. 700 input channels, 20 classes.

Cramer et al., "The Heidelberg Spiking Data Sets for the Systematic
Evaluation of Spiking Neural Networks", IEEE TNNLS 2022.

Parameters
----------
root : path
    Directory containing shd_train.h5 / shd_test.h5.
train : bool
    Load training split if True, test split otherwise.
dt_ms : float
    Temporal resolution for binning spikes.
T : int
    Number of timesteps for synthetic fallback.
synthetic : bool
    Force synthetic data generation.
n_samples : int
    Number of synthetic samples to generate.
seed : int
    RNG seed for reproducible synthetic data.

Returns
-------
samples : list of ndarray, each shape (T, 700) dtype bool
    Binned spike trains.
labels : ndarray of int

### Function `_synthetic_shd(n_samples, T, dt_ms, seed)`
### Function `load_dvs_cifar10(root, train, dt_ms, T, synthetic, n_samples, seed)`
Load DVS-CIFAR10 event-camera dataset.

CIFAR-10 images displayed on a monitor and recorded by a DVS camera
at 128x128 resolution. 10 classes.

Li et al., "CIFAR10-DVS: An Event-Stream Dataset for Object
Classification", Front. Neurosci. 2017.

Parameters
----------
root : path
    Directory containing the extracted dataset.
train : bool
    Load training split if True, test split otherwise.
dt_ms : float
    Temporal resolution for synthetic fallback.
T : int
    Number of timesteps for synthetic fallback.
synthetic : bool
    Force synthetic data generation.
n_samples : int
    Number of synthetic samples to generate.
seed : int
    RNG seed for reproducible synthetic data.

Returns
-------
samples : list of ndarray, each shape (N_events, 4)
    Columns: &#91;x, y, polarity, timestamp_ms&#93;.
labels : ndarray of int

---

## Module `debug.analyzer`

### Class `DivergencePoint`
First point where two traces diverge.


### Class `CausalEvent`
One event in a causal spike chain.


### Function `find_divergence(trace_a, trace_b)`
Find the first timestep where two traces produce different spikes.

Useful for comparing ANN-converted SNN vs directly-trained SNN,
or Python simulation vs hardware output.

Returns None if traces are identical.

### Function `spike_diff(trace_a, trace_b)`
Summary of spike differences between two traces.

Returns
-------
dict with keys:
    total_mismatches: int
    mismatch_rate: float (fraction of timestep*neuron pairs)
    first_divergence: DivergencePoint or None
    per_neuron_mismatches: ndarray

### Function `causal_chain(trace, neuron_id, timestep, max_depth)`
Trace backward from a spike to find causal input events.

Starting from neuron_id at timestep, finds the chain of spikes
that contributed current to this neuron in preceding timesteps.

Parameters
----------
trace : ExecutionTrace
neuron_id : int
    Target neuron.
timestep : int
    Timestep of the spike to explain.
max_depth : int
    Maximum backward steps to trace.

Returns
-------
list of CausalEvent
    Causal chain from target backward to inputs.

---

## Module `debug.hil_client`

### Class `SpikeEvent`
Single telemetry sample from an FPGA or simulator.


### Class `SpikeRingBuffer`
Fixed-capacity circular buffer for SpikeEvent telemetry.

Lock-protected overwrite-on-full. Mirrors Go RingBuffer.

- **__init__**(capacity)
- **push**(evt)
- **snapshot**(n)
- **head**()
- **capacity**()

### Class `LayerAggregator`
Per-layer running statistics collector.

- **__init__**()
- **record**(evt)
- **get**(layer_id)
- **all**()
- **mean_correlation**(ls)
- **mean_precision**(ls)

### Class `ErrorBudget`
Threshold-based alerting for precision/correlation bounds.

- **check**(evt)

### Class `CorrelationWindow`
Sliding window for correlation values.

- **__init__**(size)
- **add**(v)
- **count**()
- **mean**()
- **max**()

### Class `PrecisionTracker`
Exponential moving average of precision.

- **update**(precision)

### Class `EventFilter`
Selects events matching criteria.

- **match**(evt)

### Class `TriggerCondition`
Conditional breakpoint for debugger.

- **evaluate**(evt)

### Class `TriggerLog`
Records fired trigger events for post-mortem analysis.

- **__init__**()
- **fire**(evt)
- **count**()

### Class `RateLimiter`
Token-bucket rate limiter for high-speed streams.

- **__init__**(capacity)
- **allow**()
- **refill**(n)
- **available**()

### Class `HealthStatus`
Debugger health snapshot.


### Function `filter_events(events, f)`
Apply a filter to a list of events.

### Function `check_health(events_received, uptime_seconds, buffer_head, buffer_capacity, clients_active)`
Compute health status from telemetry metrics.

### Function `export_csv(events)`
Export events to CSV string.

### Function `export_json(events)`
Export events to JSON array string.

---

## Module `debug.hil_debugger`

### Class `HILDebugger`
High-level wrapper for the HIL telemetry server.

- **__init__**(port)
- **start**()
  - Starts the HIL debugger server.
- **stop**()
  - Stops the HIL debugger server.
- **is_running**()
  - Returns True if the server is active.
- **url**()
  - Returns the base URL for the active telemetry server.

---

## Module `debug.hil_server`

### Class `HILServerDaemon`
Manages the background execution of the Go HIL Debugger service.

- **__post_init__**()
- **start**(build)
  - Compile and start the standalone HIL Debugger service.
- **_wait_for_ready**(timeout_sec)
- **stop**()
  - Gracefully terminate the background HIL debugger process.
- **is_running**()
  - Returns True if the daemon process is running.

---

## Module `debug.sc_doctor`

### Class `ScDoctor`
Adaptive bitstream length controller with optional ECC.

Correlation-driven feedback loop:
- High correlation (>0.15): double bitstream length
- Low correlation (<0.05): halve bitstream length (floor 256)
- ECC auto-enabled when length exceeds 2048

- **__init__**(initial_length, target_precision)
- **adapt**(current_correlation, popcount)
  - Analyze correlation and adjust bitstream length.
- **encode_ecc**(data)
  - Hamming(7,4) encode a 4-bit chunk → 7-bit codeword.
- **decode_ecc**(encoded)
  - Hamming(7,4) decode with single-bit error correction.

---

## Module `debug.sc_scope`

### Class `TransportType`

### Class `TransportConfig`
Configuration for a transport backend.


### Class `TransportBackend`
Pluggable transport adapter for bitstream acquisition.

Production backends (JTAG, UART, PYNQ DMA) require hardware;
the ``SIMULATED`` backend generates synthetic data for testing
and development.

- **connect**()
  - Establish connection to the target.
- **disconnect**()
- **read_bitstream**(num_words, layer_id)
  - Read packed bitstream words from the target.
- **_sim_read**(num_words, layer_id)
  - Generate simulated bitstream data.

### Class `BitstreamSample`
One timestamped bitstream capture.

- **bit_length**()
- **popcount**()
- **density**()
- **effective_bits**()
  - Shannon entropy-based effective precision.

### Class `AnalysisWindow`
Windowed statistics from recent samples.

- **__post_init__**()
- **push**(sample)
- **count**()
- **mean_density**()
- **std_density**()
- **mean_effective_bits**()
- **total_popcount**()
- **sample_rate_hz**()
  - Estimated sample rate from timestamps.

### Class `LiveAnalyzer`
Real-time SC bitstream analyzer with per-layer windows.

- **__init__**(num_layers, window_size)
- **ingest**(sample)
  - Process one incoming sample.
- **layer_stats**(layer_id)
  - Get summary stats for one layer.
- **all_stats**()

### Class `LayerErrorBudget`
Per-layer precision tracking against golden model expectations.

- **check**(measured_density)
  - Check if measured density is within tolerance.
- **current_error**()
- **mean_error**()
- **max_error**()
- **violations**()
- **pass_rate**()

### Class `TriggerType`

### Class `TriggerCondition`
Conditional capture trigger.


### Class `TriggerEvent`
A triggered capture event.


### Class `TriggerEngine`
Evaluates capture triggers against incoming samples.

- **__init__**()
- **add_trigger**(condition)
- **evaluate**(sample)
  - Check all triggers against a sample. Returns fired events.
- **event_count**()
- **clear**()

### Class `ScopeSession`
Manages a live debugging session.

- **start**()
  - Start the scope session.
- **stop**()
- **add_error_budget**(layer_id, expected_density, tol)
- **capture_one**(layer_id, neuron_id, num_words)
  - Capture one bitstream sample from the target.
- **capture_sweep**(num_layers, num_words)
  - Capture one sample from each layer.
- **status**()

### Class `ScopeRenderer`
Text-mode rendering of live scope data for CLI output.

- **render_density_bar**(cls, density, width)
  - Render a density as a text bar.
- **render_layer_summary**(cls, layer_id, stats)
- **render_session**(cls, session)
  - Render full session status as text.

### Function `_compute_scc_python(a, b)`
Pure-Python Alaghi-Hayes SCC (reference implementation + fallback).

### Function `compute_scc(a, b)`
Stochastic Computing Correlation between two u32-packed bitstreams.

Dispatches to the Rust ``stochastic_doctor_core.py_scc_packed`` when the
compiled extension is importable (the default when the repo is built with
``maturin develop --release``). Falls back to :func:`_compute_scc_python`
when the extension is missing — the fallback is numerically identical
(both implement the case-split Alaghi & Hayes 2013 form).

---

## Module `debug.tracer`

### Class `ExecutionTrace`
Complete execution trace of an SNN run.

Attributes
----------
n_neurons : int
    Total neurons across all populations.
n_steps : int
    Number of simulation timesteps.
spikes : ndarray of shape (n_steps, n_neurons)
    Binary spike matrix.
voltages : ndarray of shape (n_steps, n_neurons)
    Membrane voltages.
currents : ndarray of shape (n_steps, n_neurons)
    Input currents.
population_labels : list of str
    Population names.
population_ranges : list of (start, end)
    Neuron index ranges per population.

- **spike_count**()
  - Total spikes in the trace.
- **firing_rates**()
  - Per-neuron firing rate (spikes per step).
- **neuron_trace**(neuron_id)
  - Extract full trace for one neuron.
- **spike_times**(neuron_id)
  - Timesteps when a neuron spiked.
- **population_spikes**(pop_label)
  - Spike matrix for one population.

### Class `SpikeTracer`
Records execution trace during SNN simulation.

Wraps a Network and intercepts step_all to record spikes,
voltages, and currents at every timestep.

Usage
-----
>>> tracer = SpikeTracer(network)
>>> trace = tracer.run(duration=0.1, dt=0.001)
>>> divergence = find_divergence(trace, expected_spikes)

- **__init__**(network)
- **run**(duration, dt, seed)
  - Run the network and record full execution trace.

---

## Module `digital_twin.mismatch`

### Class `FPGAMismatchModel`
Wraps weight matrices and neuron parameters with FPGA imperfections.

Parameters
----------
quantization_bits : int
    Fixed-point bit width (default 16 for Q8.8).
weight_cv : float
    Coefficient of variation for weight perturbation (default 0.02 = 2%).
threshold_cv : float
    Per-neuron threshold variation (default 0.05 = 5%).
clock_jitter_pct : float
    Clock period variation (default 0.01 = 1%).
seed : int
    Random seed for reproducibility.

- **__post_init__**()
- **quantize**(values)
  - Apply Q-format quantization noise.
- **perturb_weights**(weights)
  - Add process variation noise to weights.
- **perturb_thresholds**(thresholds)
  - Add per-neuron threshold mismatch.
- **jitter_timing**(n_steps)
  - Generate clock jitter: per-step timing variation.
- **apply_to_network_weights**(weights)
  - Apply all hardware imperfections to a list of weight matrices.
- **mismatch_report**(weights)
  - Report expected mismatch statistics for given weights.

---

## Module `digital_twin.twinsync`

### Class `LamportClock`
Lamport logical clock for causal ordering.

- **__init__**()
- **tick**()
  - Local event: increment.
- **send**()
  - Prepare timestamp for sending.
- **receive**(remote_time)
  - Update on message receipt.

### Class `VectorClock`
Vector clock for full causal dependency tracking.

- **__init__**(node_id, num_nodes)
- **tick**()
- **send**()
- **receive**(remote_clock)
- **happened_before**(other)
  - Check if self happened-before other (self < other).
- **concurrent_with**(other)
  - Check if self is concurrent with other.

### Class `EventType`

### Class `TwinEvent`
One event in the time-warp simulation.


### Class `Checkpoint`
Deterministic state snapshot for rollback.

- **compute_checksum**()

### Class `CheckpointManager`
Manages state snapshots for time-warp rollback.

Preserves the identity substrate (ArcaneNeuron.v_deep) across
rollback: deep compartment is NEVER rolled back.

- **__init__**(max_checkpoints)
- **save**(node_id, virtual_time_ns, neuron_state, synapse_state, lfsr_state, identity_deep, lamport_time, vector_clock)
- **find_rollback_target**(node_id, target_time_ns)
  - Find the latest checkpoint at or before target_time.
- **discard_after**(node_id, time_ns)
  - Discard checkpoints after a given time (post-rollback cleanup).
- **total_checkpoints**()

### Class `NodeState`
Per-node state in the time-warp simulation.


### Class `TimeWarpEngine`
Optimistic parallel simulation with anti-message rollback.

Implements the Jefferson Time Warp protocol adapted for
SC neuromorphic simulation:

1. Each node advances optimistically at its own rate
2. Straggler events trigger rollback + anti-messages
3. Global Virtual Time (GVT) advances monotonically
4. Fossil collection prunes checkpoints below GVT
5. Identity (v_deep) is NEVER rolled back — it is the self

- **__init__**(num_nodes, checkpoint_interval_ns)
- **inject_event**(event)
  - Inject an event into the simulation.
- **process_next**()
  - Process the next event from the queue.
- **_rollback**(node, target_time_ns)
  - Roll back a node to a checkpoint at or before target_time.
- **compute_gvt**()
  - Compute Global Virtual Time (minimum of all LVTs + in-transit).
- **fossil_collect**()
  - Remove checkpoints below GVT.
- **status**()
- **inject_sync_barrier**(virtual_time_ns)
  - Inject a sync barrier event to all nodes at given time.
- **verify_causal_order**()
  - Verify causal ordering of processed events.
- **detect_starvation**(threshold_ns)
  - Detect nodes lagging behind GVT by more than threshold.
- **node_throughput**()
  - Events processed per node.

### Class `SyncMode`

### Class `DivergenceMetric`
Measures divergence between physical and digital twin.

- **total_divergence**()
- **within_tolerance**()

### Class `TwinSession`
Orchestrates physical ↔ digital twin synchronization.

Manages bidirectional data flow:
- Physical → Digital: sensor events (MEA spikes, EEG)
- Digital → Physical: stimulation commands (opto, TMS)

- **__init__**(num_nodes, mode, max_divergence)
- **start**()
- **stop**()
- **inject_physical_event**(spike_time_ns, neuron_id, target_node)
  - Inject a physical sensor event into the digital twin.
- **advance**(steps)
  - Advance the simulation by N steps.
- **update_divergence**(physical_rate_hz, digital_rate_hz, physical_identity)
  - Update divergence metrics.
- **status**()

### Class `LookaheadConfig`
Null-message lookahead for conservative synchronization.

Each node declares a minimum time advance (lookahead) it guarantees
before generating output events. Peers can safely advance by at
least this amount without rollback risk.

- **can_advance_to**(target_ns)
- **send_null_message**(current_ns)

### Class `NullMessageOptimizer`
Reduces rollbacks in mixed conservative/optimistic mode.

- **__init__**(num_nodes, default_lookahead_ns)
- **safe_advance_time**(node_id)
  - Maximum time this node can safely advance to.
- **broadcast_null**(node_id, current_ns)

### Class `DeltaCheckpoint`
Stores only the diff from a base checkpoint.

- **compute_delta**(base_state, new_state, base_id, new_id, virtual_time_ns, node_id)
- **compression_ratio**()
- **num_changes**()

### Class `ReplayVerifier`
Verifies bitstream-exact replay across runs.

Compares checkpoint hashes from two runs to prove determinism.

- **__init__**()
- **record_run_a**(checkpoint)
- **record_run_b**(checkpoint)
- **is_deterministic**()
- **first_divergence_index**()
- **compared_count**()

### Class `DriftCorrection`
One drift correction action.


### Class `DriftAutoCorrector`
Closed-loop drift correction between physical and digital twin.

- **__init__**(max_drift_ns, correction_gain)
- **check_and_correct**(physical_time_ns, digital_time_ns, node_id)
- **total_corrections**()

### Class `MPIRankMapping`
Maps MPI ranks to physical node topology.

- **neuron_count**()

### Class `MPITopology`
Physical→logical node layout for distributed twin.

- **__init__**()
- **add_rank**(mapping)
- **total_neurons**()
- **num_ranks**()
- **rank_for_neuron**(neuron_id)
- **co_located_ranks**(rank)
  - Ranks on the same host (cheap communication).

### Class `BackpressureController`
Prevents event overload by throttling injection rate.

- **__init__**(max_queue_depth, cooldown_ns)
- **should_accept**(current_queue_depth)
- **rejection_rate**()
- **is_backpressured**()

### Class `CheckpointAuditChain`
Tamper-evident chain of checkpoint hashes.

- **__init__**()
- **append**(checkpoint)
- **verify**()
- **length**()

### Class `SessionSnapshot`
Serializable session state for persistence.

- **from_session**(session)
- **to_dict**()

### Class `TwinEndpoint`
One twin in a federation.


### Class `TwinFederation`
Federates multiple digital twins for multi-subject studies.

- **__init__**()
- **register**(twin_id, session, priority)
- **twin_count**()
- **global_gvt**()
- **advance_all**(steps)
- **total_divergence**()

### Class `AdaptiveCheckpointInterval`
Dynamically adjusts checkpoint frequency based on rollback rate.

- **__init__**(base_interval, min_interval, max_interval)
- **update**(total_rollbacks, total_events)
  - Adjust interval: more rollbacks → more frequent checkpoints.
- **is_aggressive**()

---

## Module `distillation.distill`

### Class `TemporalDistillationLoss`
Temporal-aware distillation loss for SNN→SNN or ANN→SNN transfer.

Matches per-timestep output distributions from teacher to student,
with entropy regularization to prevent learning erroneous knowledge.

Parameters
----------
temperature : float
    Softmax temperature for logit matching.
alpha : float
    Weight of distillation loss vs task loss (0=task only, 1=distill only).
entropy_weight : float
    Entropy regularization strength.

- **__init__**(temperature, alpha, entropy_weight)
- **compute**(student_logits, teacher_logits, targets)
  - Compute distillation loss.
- **_softmax**(x)

### Class `SelfDistiller`
Self-distillation: use extended-T model as implicit teacher.

Run the same model at T_teacher timesteps (more accurate) to
generate soft targets for training at T_student timesteps (faster).

Parameters
----------
T_teacher : int
    Timesteps for teacher forward pass.
T_student : int
    Timesteps for student forward pass.
temperature : float

- **generate_targets**(run_fn, inputs)
  - Run model at T_teacher steps to generate soft targets.
- **_softmax**(x)

---

## Module `doctor.diagnose`

### Class `Severity`

### Class `Diagnosis`
Single diagnostic finding.


### Class `DiagnosticReport`
Full diagnostic report for an SNN architecture.

- **summary**()
- **has_critical**()
- **score**()
  - Health score 0-100. 100 = no issues.

### Function `diagnose(layer_sizes, weights, spike_rates, target, bitstream_length)`
Run architecture diagnostics.

Parameters
----------
layer_sizes : list of (n_inputs, n_neurons)
weights : list of ndarray, optional
    Weight matrices per layer.
spike_rates : list of ndarray, optional
    Per-neuron firing rates per layer (from profiling).
target : str
    FPGA target for hardware checks.
bitstream_length : int
    SC bitstream length.

Returns
-------
DiagnosticReport

### Function `_check_hardware(report, layer_sizes, target, bitstream_length)`
### Function `_check_weights(report, weights)`
### Function `_check_spike_rates(report, spike_rates)`
### Function `_check_architecture(report, layer_sizes)`
### Function `_check_coding_efficiency(report, layer_sizes, bitstream_length)`
---

## Module `drivers.physical_twin`

### Class `PhysicalTwinBridge`
Synchronise software neuron state with an explicit twin backend.

``mode="EMULATION"`` is a deterministic local noise model for development
and CI. ``mode="TCP"`` opens a JSON-line request/response connection for a
real hardware-twin service. The class never marks itself connected to
physical hardware unless a TCP exchange actually succeeds.

- **__init__**(ip, port)
- **sync_step**(sw_v_mem, sw_spike)
  - Send software state and return the twin membrane voltage.
- **_sync_step_tcp**(sw_v_mem, sw_spike)
- **_parse_reply**(response)
- **_log_divergence**(sw_v_mem, hw_v_mem)

---

## Module `drivers.sc_neurocore_driver`

### Class `RealityHardwareError`
Raised when physical hardware is required but missing.


### Class `SC_NeuroCore_Driver`
Primary driver for the sc-neurocore FPGA overlay on PYNQ-Z2.

This driver enforces 'Reality Checks'. It will NOT run on standard x86 CPUs
unless explicitly in 'EMULATION' mode.

- **__init__**(bitstream_path, mode, seed)
  - Construct a driver in HARDWARE or EMULATION mode.
- **_connect_to_fpga**()
  - Attempts to load the PYNQ libraries and flash the bitstream.
- **write_layer_params**(layer_id, params)
  - Writes parameters to a specific layer's AXI-Lite registers.
- **run_step**(input_vector)
  - Executes one integration step on the FPGA.

---

## Module `drivers.verify_hardware_link`

### Function `verify_link(extras)`
Run the hardware-link diagnostic CLI.

Parameters
----------
extras : bool
    Run the optional Evo 2 + Opentrons probes when True
    (default). Set to False to only check the FPGA subsystem;
    skips the imports of sibling-repo modules.

---

## Module `edge.aer_router`

### Class `AERRoutingDaemon`
Orchestrates the Go-based AER UDP mesh multi-FPGA router pipeline dynamically.

- **__init__**(port)
- **start**(build)
- **stop**()
  - Tears down the active background UDP topology safely.

---

## Module `edge.bitstream`

### Function `popcount32(word)`
Count set bits in a u32 word (Wilkes-Wheeler-Gill).

### Function `popcount_slice(words)`
Popcount over a packed u32 word slice.

### Function `sc_and(a, b)`
SC multiply (bitwise AND).

### Function `sc_or(a, b)`
SC saturating addition (bitwise OR).

### Function `sc_xor(a, b)`
SC absolute difference / HDC bind (bitwise XOR).

### Function `sc_sub(a, b)`
SC saturating subtraction: a AND NOT b.

### Function `sc_mux(a, b, sel)`
SC scaled addition (2:1 MUX): (a AND sel) OR (b AND NOT sel).

### Function `and_packed(a, b)`
SC AND over two packed word slices.

### Function `mux_packed(a, b, sel)`
SC MUX over two packed word slices with a select bitstream.

### Function `probability(words, bit_length)`
Estimated probability from a packed bitstream.

### Function `scc(a, b, bit_length)`
SCC between two packed u32 bitstreams (Alaghi & Hayes, 2013).

Returns a correlation coefficient in &#91;-1, 1&#93;.

---

## Module `edge.deploy`

### Function `generate_cargo_config(board)`
Generate .cargo/config.toml content for a target board.

### Function `generate_memory_x(board)`
Generate memory.x linker script content for a target board.

---

## Module `edge.lfsr`

### Class `Lfsr16`
16-bit Galois LFSR bitstream encoder.

Bit-compatible with the Rust core_engine::bitstream::Lfsr16.
Uses u32-packed output for MCU word alignment.

- **__init__**(seed)
- **step**()
  - Advance LFSR by one clock, return new state.
- **encode**(threshold, bit_length)
  - Encode probability (threshold/65535) into packed u32 words.
- **encode_float**(p, bit_length)
  - Encode a probability &#91;0.0, 1.0&#93; into a packed bitstream.

---

## Module `edge.neuron`

### Class `LifNeuron`
Leaky Integrate-and-Fire neuron (SC domain, integer arithmetic).

Membrane potential = running popcount of input bitstream.
Leak = right-shift per tick (exponential decay).
Fires when potential exceeds threshold.

- **tick**(input_words)
  - Process one timestep, return True if spike fired.
- **reset**()

### Class `IzhikevichNeuron`
Izhikevich neuron with integer SC-domain dynamics.

Uses fixed-point arithmetic (Q16.16) to avoid floating-point.
Supports regular spiking, fast spiking, chattering, and intrinsic burst.

- **tick**(input_current_q16)
  - Process one timestep. Returns True on spike.
- **reset**()
- **regular_spiking**(cls)
- **fast_spiking**(cls)
- **chattering**(cls)
- **intrinsic_burst**(cls)

---

## Module `edge.power_estimator`

### Class `Board`
Supported RISC-V MCU targets.

- **__init__**(label, ram_kb, flash_kb, active_uw, sleep_uw)

### Class `PowerProfile`
Estimated power profile for a target board at a given clock.

- **for_board**(cls, board, clock_mhz)
- **duty_cycled_uw**(duty)
  - Estimate µW for a given duty cycle (0.0=sleep, 1.0=active).

### Class `MemoryFootprint`
Memory footprint estimate for a tinySC network.

- **estimate**(cls, num_layers, neurons_per_layer, bs_words, board)
  - Estimate memory for a network configuration.
- **max_neurons**(board)
  - Maximum neurons that fit in a board's RAM (single layer).

---

## Module `edge.power_thermal`

### Class `PowerThermalConfig`
Inputs for an FPGA power and thermal deployment estimate.

- **__post_init__**()

### Class `VivadoPowerReport`
Power and thermal values parsed from a routed Vivado power report.


### Class `VivadoUtilizationReport`
Resource counts parsed from a Vivado utilisation report.


### Function `build_power_thermal_model(config)`
Build a deterministic JSON-compatible FPGA power/thermal model.

### Function `build_power_thermal_model_from_vivado_reports(report_dir, config)`
Build a power/thermal model seeded from Vivado implementation reports.

The returned model keeps the estimator resource breakdown for workload
context, but replaces headline power and thermal values with the routed
report values from Vivado.

### Function `parse_vivado_power_report(path)`
Parse headline power and thermal fields from a Vivado report_power file.

### Function `parse_vivado_utilization_report(path)`
Parse FPGA resource counts from a Vivado report_utilization file.

### Function `write_power_thermal_model(output_dir, config)`
Write `power_thermal_model.json` beside generated FPGA artefacts.

### Function `write_power_thermal_model_from_vivado_reports(report_dir, output_dir, config)`
Write report-derived FPGA power/thermal JSON beside deployable artefacts.

### Function `_target_payload(target)`
### Function `_resource_payload(report)`
### Function `_extract_header(text, label)`
### Function `_extract_summary_float(text, label)`
### Function `_extract_summary_text(text, label)`
### Function `_extract_utilisation_row(text, label)`
---

## Module `edge.sc_network`

### Class `SCLayer`
Single dense SC layer: weights × inputs via AND + popcount threshold.

- **__post_init__**()
- **_validate_configuration**()
- **_validate_weights**()
- **words_per_input**()
- **forward**(input_words, bit_length)
  - Run SC inference: AND each weight row with input, threshold popcount.

### Class `SCNetwork`
Multi-layer feed-forward SC network runner.

Usage::

    net = SCNetwork(bit_length=1024)
    net.add_layer(SCLayer(n_inputs=32, n_outputs=16))
    net.add_layer(SCLayer(n_inputs=16, n_outputs=8))
    output = net.run(&#91;0.5&#93; * 32)

- **__post_init__**()
- **add_layer**(layer)
- **encode_inputs**(probabilities)
  - Encode float probabilities into per-input packed bitstreams.
- **_spikes_to_bitstreams**(spikes, lfsr)
  - Re-encode spike booleans as bitstreams for the next layer.
- **_flatten_bitstreams**(streams)
  - Interleave per-input bitstreams into a flat word array.
- **run**(input_probabilities)
  - Full inference: encode → cascaded layer inference → spike output.
- **export_weights**()
  - Export all layer weights in serialization-ready format.
- **from_weights**(cls, layers_data, bit_length, lfsr_seed)
  - Construct network from deserialized weight data.
- **layer_count**()
- **total_neurons**()

---

## Module `edge.sobol`

### Class `SobolGenerator`
1D Sobol sequence generator with 16-bit resolution.

Uses Joe-Kuo direction numbers (dimension 1) and Gray-code indexing
so only one XOR per step.

- **__init__**(seed)
- **step**()
  - Advance by one step, return the next Sobol value in &#91;0, 65535&#93;.
- **encode**(threshold, length)
  - Encode a probability into packed u64 words using Sobol sequence.
- **reset**(seed)
  - Reset to initial state.

---

## Module `edge.telemetry`

### Class `TelemetryRing`
Fixed-size ring buffer for telemetry samples (u32 values).

- **__init__**(capacity)
- **push**(value)
- **mean**()
- **last**()
- **count**()
- **capacity**()

### Class `LayerTelemetry`
Telemetry counters for a single SC layer.

- **record_tick**(n_spikes, n_neurons)
  - Record one tick's worth of activity.
- **mean_spike_rate**()
- **mean_utilization**()
- **lifetime_spike_rate**()

### Class `DeviceTelemetry`
Aggregate telemetry for the full device/network.

- **get_layer**(layer_id)
- **record**(layer_id, n_spikes, n_neurons)
- **summary**()

---

## Module `edge.web_deploy`

### Class `WebDeploymentConfig`
Configuration for generating browser deployment artefacts.

- **__post_init__**()

### Class `WebDeploymentManifest`
Manifest consumed by the generated browser runtime.

- **to_dict**()
  - Return a JSON-serialisable manifest dictionary.

### Function `build_web_deployment(model_path, output_dir, config)`
Generate a static browser deployment scaffold for a model artefact.

### Function `_render_index_html()`
### Function `_render_runtime_js()`
### Function `_render_webgpu_shader()`
---

## Module `edge.weights`

### Class `WeightHeader`
Weight blob header (16 bytes).

- **to_bytes**()
- **from_bytes**(cls, data)
- **validate**()

### Class `LayerHeader`
Per-layer header (16 bytes).

- **to_bytes**()
- **from_bytes**(cls, data)
- **words_per_row**()

### Function `serialize_weights(layers)`
Serialize network weights to binary blob.

Parameters
----------
layers : list
    Each entry is (n_inputs, n_outputs, threshold, weight_rows).
    weight_rows is list&#91;list&#91;int&#93;&#93; (n_outputs × words_per_row u32 values).

Returns
-------
bytes
    Complete weight blob with headers.

### Function `deserialize_weights(data)`
Deserialize a weight blob into layer headers + weight matrices.

Returns
-------
list&#91;tuple&#91;LayerHeader, list&#91;list&#91;int&#93;&#93;&#93;&#93;
    Each entry is (header, weight_rows).

---

## Module `encoding.encoders`

### Function `rate_encode(values, T, seed)`
Rate coding: spike probability proportional to value.

Parameters
----------
values : ndarray of shape (N,)
    Input values in &#91;0, 1&#93;.
T : int
    Number of timesteps.
seed : int

Returns
-------
ndarray of shape (T, N), binary

### Function `latency_encode(values, T)`
Latency (Time-to-First-Spike) coding: higher value = earlier spike.

Parameters
----------
values : ndarray of shape (N,)
    Input values in &#91;0, 1&#93;.
T : int

Returns
-------
ndarray of shape (T, N), binary

### Function `delta_encode(values, threshold)`
Delta coding: spike when change exceeds threshold.

Parameters
----------
values : ndarray of shape (T,) or (T, N)
    Time-varying input signal.
threshold : float

Returns
-------
ndarray of same shape, binary

### Function `phase_encode(values, T, n_phases)`
Phase coding: value encoded as spike phase within oscillation cycle.

Parameters
----------
values : ndarray of shape (N,)
    Input values in &#91;0, 1&#93;.
T : int
n_phases : int
    Number of phase bins per cycle.

Returns
-------
ndarray of shape (T, N), binary

### Function `burst_encode(values, T, max_burst)`
Burst coding: value encoded as burst length (consecutive spikes).

Parameters
----------
values : ndarray of shape (N,)
    Input values in &#91;0, 1&#93;.
T : int
max_burst : int
    Maximum burst length for value=1.

Returns
-------
ndarray of shape (T, N), binary

### Function `rank_order_encode(values, T)`
Rank-order coding: neurons fire in order of decreasing value.

Parameters
----------
values : ndarray of shape (N,)
T : int

Returns
-------
ndarray of shape (T, N), binary

### Function `sigma_delta_encode(values, threshold)`
Sigma-delta coding: integrate error, spike when threshold exceeded.

Parameters
----------
values : ndarray of shape (T,) or (T, N)
    Time-varying signal.
threshold : float

Returns
-------
ndarray of same shape, binary

---

## Module `encoding.optimizer`

### Class `EncodingRecommendation`
Recommendation for one encoding scheme.


### Class `EncodingOptimizer`
Profile data and recommend optimal spike encoding.

Parameters
----------
T : int
    Number of simulation timesteps.

- **__init__**(T)
- **profile**(data)
  - Compute data statistics relevant to encoding selection.
- **recommend**(data)
  - Recommend encoding schemes ranked by suitability.
- **_info_score**(original, encoded)
  - Estimate how well encoding preserves input information.
- **_encodings**()
- **_reason**(name, stats)

---

## Module `energy.estimator`

### Class `LayerEstimate`
Resource estimate for one layer.


### Class `EnergyReport`
Complete pre-silicon energy estimate for an SNN on an FPGA target.

- **__post_init__**()
- **summary**()
  - Human-readable summary.

### Function `estimate(layer_sizes, target, bitstream_length, neuron_type, event_driven, clock_mhz, include_infra)`
Estimate FPGA resources and power for an SNN.

Parameters
----------
layer_sizes : list of (n_inputs, n_neurons)
    Architecture as list of layer dimensions.
target : str
    FPGA target: 'ice40', 'ecp5', 'artix7', 'zynq'.
bitstream_length : int
    SC bitstream length L (affects latency and precision).
neuron_type : str
    'lif' (clock-driven) or 'event' (event-driven).
event_driven : bool
    Use event-driven architecture (AER).
clock_mhz : float
    Target clock frequency.
include_infra : bool
    Include AXI/DMA infrastructure cost.

Returns
-------
EnergyReport
    Complete resource and power estimate.

---

## Module `energy.fpga_models`

### Class `FPGATarget`
FPGA target specification.


### Class `ModuleCost`
Resource cost for one SC module instance.


---

## Module `energy_accounting.accountant`

### Class `HardwareCostModel`
Energy costs for a specific hardware target.

All values in picojoules (pJ).


### Class `LayerEnergy`
Energy breakdown for one layer.


### Class `EnergyReport`
Complete energy accounting report.

- **summary**()
- **dominant_layer**()
- **energy_per_spike_pj**()

### Class `EnergyAccountant`
Per-spike energy accounting system.

Parameters
----------
hardware : str or HardwareCostModel
    Target hardware for cost mapping.

- **__init__**(hardware)
- **account**(layer_names, layer_sizes, spike_counts, n_timesteps)
  - Compute energy breakdown from spike activity.

---

## Module `energy_accounting.sustainability_profiler`

### Class `FPGAResourceReport`
Vivado-style utilisation report.

- **dynamic_power_mw**()
  - Estimate dynamic power: P = C_eff * V² * f * activity.
- **total_power_mw**()
- **power_breakdown**()
  - Per-component dynamic power breakdown in mW.
- **scale_dvfs**(clock_mhz, voltage_v)
  - Return a new report with scaled clock and voltage (DVFS).
- **from_vivado_dict**(cls, d)
  - Parse from a Vivado utilisation report dictionary.

### Class `GridRegion`

### Class `CarbonModel`
CO₂ emissions model for a given power profile.

- **co2_g_per_kwh**()
- **compute**(power_mw, duration_hours)
  - Return grams of CO₂ emitted.
- **annual_footprint_kg**(power_mw)
  - Annual CO₂ in kilograms (24/7 operation).

### Class `EmbodiedCarbon`
Manufacturing + end-of-life carbon footprint.

- **total_embodied_kg**()
- **amortised_annual_kg**()
  - Annual embodied carbon amortised over device lifetime.

### Class `EnergyHarvester`

### Class `HarvestProfile`
Energy-harvesting power curve.

- **__post_init__**()
- **average_power_mw**()
- **energy_over**(hours)
  - Total energy harvested in mWh.
- **power_at**(hour_of_day)
  - Instantaneous power at a given hour (0–24).

### Class `MultiHarvestStack`
Combines multiple energy-harvesting sources.

- **__init__**(profiles)
- **add**(profile)
- **average_power_mw**()
- **power_at**(hour_of_day)
- **energy_over**(hours)
- **num_sources**()

### Class `EnergyStorageSim`
Battery/supercap state-of-charge simulation.

- **__post_init__**()
- **step**(net_power_mw, dt_hours)
  - Advance one time step. Returns clamped SoC.
- **energy_stored_mwh**()
- **is_depleted**()

### Class `ThermalModel`
Simple junction-temperature model.

T_j = T_ambient + P_total * R_theta_ja

- **junction_temp**(power_mw)
- **is_safe**(power_mw)
- **max_power_mw**()
  - Maximum power before thermal limit.

### Class `DutyCycleConfig`
Optimised duty-cycle configuration for a deployment.


### Class `NetZeroReport`
Results of a net-zero feasibility analysis.


### Class `SustainabilityOptimizer`
Optimises SC deployment for net-zero energy operation.

- **__init__**(fpga, carbon, embodied, thermal)
- **analyze**(harvest, target_hours)
  - Run full sustainability analysis.
- **_optimize_duty_cycle**(total_power, harvest_power)
  - Compute optimal duty-cycle to match harvest.
- **_generate_suggestions**(total, harvest, deficit)
- **hourly_profile**(harvest, hours)
  - Generate an hourly power-balance profile.
- **simulate_storage**(harvest, storage, hours)
  - Simulate battery SoC over time with harvest and load.
- **energy_efficiency**(ops_per_second)
  - Compute energy efficiency metrics.
- **deployment_lifetime**(harvest, battery_mwh)
  - Estimate deployment lifetime and maintenance intervals.
- **adaptive_duty_cycle_sim**(harvest, hours, min_active)
  - Simulate time-varying adaptive duty cycling.

### Function `analyze_multi_harvest(fpga, stack, carbon)`
Run sustainability analysis with a stacked harvest profile.

---

## Module `energy_accounting.unified_reporter`

### Class `UnifiedEnergyReport`
Combined report from runtime profiling + sustainability analysis.

- **summary**()

### Class `UnifiedEnergyReporter`
Orchestrates power estimation → carbon accounting → thermal analysis.

Usage::

    reporter = UnifiedEnergyReporter(region=GridRegion.EU)
    report = reporter.analyze(total_power_mw=150.0, duration_h=1.0)

- **__init__**(region, ambient_temp_c, asic_power_mw)
- **analyze**(layer_configs, total_power_mw, inference_time_s)
  - Full analysis: power → carbon → thermal.

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

## Module `event_driven.simulator`

### Class `SpikeEvent`
One spike event in the priority queue.


### Class `EventStats`
Statistics from an event-driven simulation run.

- **summary**()

### Class `EventDrivenSimulator`
Event-driven asynchronous SNN simulator.

Parameters
----------
n_neurons : int
    Total neurons.
connectivity : list of (source, target, weight, delay)
    Synaptic connections.
threshold : float
    Spike threshold for LIF neurons.
tau_mem : float
    Membrane time constant (ms).
v_rest : float
    Resting potential.
v_reset : float
    Reset potential after spike.
refractory : float
    Refractory period (ms).

- **__init__**(n_neurons, connectivity, threshold, tau_mem, v_rest, v_reset, refractory)
- **inject_spikes**(events)
  - Inject external spike events.
- **inject_current**(events)
  - Inject current pulses.
- **run**(duration)
  - Run event-driven simulation.
- **reset**()
  - Reset all state.

---

## Module `evo_substrate.evo_substrate`

### Class `TopologyGene`
Encodes the network topology.

- **to_vector**()
- **from_vector**(cls, v)

### Class `NeuronGene`
Encodes neuron-level parameters (ArcaneNeuron-compatible).

- **to_vector**()
- **from_vector**(cls, v)

### Class `PlasticityGene`
Encodes plasticity rule parameters.

- **to_vector**()
- **from_vector**(cls, v)

### Class `Genome`
Complete genome for an evolving SC organism.

- **to_vector**()
- **from_vector**(cls, v, gen)
- **vector_dim**()
- **compute_id**()

### Class `MutationType`

### Class `MutationConfig`
Controls mutation rates and magnitudes.


### Class `MutationEngine`
Applies mutations to genomes.

- **__init__**(config, rng_seed)
- **mutate**(genome)
  - Apply a random mutation and return the mutated child.
- **_point_mutation**(genome)
- **_structural_mutation**(genome)
- **_duplication_mutation**(genome)
- **_swap_mutation**(genome)

### Class `CrossoverEngine`
Uniform crossover between two parent genomes.

- **__init__**(rng_seed)
- **crossover**(parent_a, parent_b)
  - Uniform crossover: each gene drawn from either parent.

### Class `LineageRecord`
One entry in the ancestry log.


### Class `LineageTracker`
Tracks ancestry graph for all organisms.

- **__init__**()
- **record**(organism, mutation_type)
- **get_ancestors**(genome_id)
  - Walk the ancestry chain to the root.
- **num_records**()

### Class `FitnessType`

### Class `FitnessResult`
Fitness evaluation result.

- **compute_composite**(w_acc, w_energy, w_latency)

### Class `FitnessEvaluator`
Evaluates organism fitness from simulation metrics.

- **__init__**(fitness_type)
- **evaluate**(genome, metrics)

### Class `Organism`
One evolving SC organism.


### Class `RuntimeFaultConfig`
Runtime fault-check settings for evolved SC organisms.


### Class `RuntimeFaultCheck`
Recorded runtime fault/degradation decision for one organism.

- **from_plan**(cls, organism, plan)
- **to_dict**()
  - Return a JSON-ready fault-check summary.

### Class `ReplicationEngine`
Manages organism reproduction, mutation, and deployment.

Selection → Replication → Mutation → Safety Check → Deploy

- **__init__**(mutation_engine, crossover_engine, fitness_evaluator, max_population, elitism, industrial_mode, runtime_fault_config, degradation_policy)
- **seed**(genome)
  - Seed the population with an initial organism.
- **replicate**(parent)
  - Create a mutated child from a parent.
- **replicate_crossover**(parent_a, parent_b)
  - Create a child via crossover of two parents.
- **evaluate_all**(metrics_fn)
  - Evaluate fitness for all living organisms.
- **verify_runtime_faults**(organism, config)
  - Run seeded runtime fault diagnosis and apply bounded degradation.
- **_runtime_bitstreams**(genome, config)
- **_apply_runtime_fault_plan**(organism, plan, config)
- **select_and_cull**(survival_fraction)
  - Select fittest organisms, cull the rest. Elitism preserved.
- **evolve_generation**(metrics_fn)
  - Run one full evolutionary generation.
- **best_organism**()
- **best_fitness**()
- **mean_fitness**()

### Class `OrganismEmitter`
Emits evolved organisms as NIR graph or Verilog.

- **to_nir**(genome)
  - Emit a simplified NIR-compatible graph dict.
- **to_verilog**(genome, module_name)
  - Emit Verilog wrapper for the organism.
- **to_photonic_netlist**(genome, pml_layers)
  - Emit a photonic netlist compatible with the optics PhotonicCompiler.

### Class `SafetyBounds`
Constrains the mutation space to prevent runaway replication.

Enforces hard limits on genome parameters that could cause
resource exhaustion or unsafe behaviour on FPGA tiles.

- **clamp**(genome)
- **is_within_bounds**(genome)

### Class `TileAllocation`
Maps an organism to a physical FPGA tile.


### Class `TileDeploymentTracker`
Tracks which organisms are deployed on which FPGA tiles.

- **__init__**(num_tiles)
- **deploy**(organism, tile_id)
- **evict**(tile_id)
- **free_tiles**()
- **utilisation**()

### Class `HallOfFame`
Maintains the top-N organisms across all generations.

- **__init__**(max_size)
- **update**(organism)
- **best_fitness**()
- **size**()

### Class `Island`
One sub-population (deme) in an island model.


### Class `IslandModel`
Multi-deme evolution with periodic migration.

- **__init__**(num_islands, migration_rate)
- **add_organism**(island_id, organism)
- **migrate**(rng)
  - Migrate best organisms between random island pairs.
- **total_population**()

### Class `GenomeSerializer`
Serializes/deserializes genomes for persistence.

- **to_dict**(genome)
- **from_dict**(d)

### Class `NoveltyArchive`
Behavioural novelty archive for novelty search.

- **__init__**(k_nearest, threshold)
- **novelty_score**(behaviour)
- **maybe_add**(behaviour)
- **size**()

### Class `ResourceBudget`
Per-organism resource constraints.

- **check**(genome)

### Class `ExtinctionDetector`
Detects population stagnation and triggers extinction events.

- **__init__**(stagnation_gens, kill_fraction)
- **check**(best_fitness)
- **apply**(population, rng)
  - Kill kill_fraction of population randomly.

### Class `CoevoRole`

### Class `CoevoOrganism`
Organism with a co-evolutionary role.


### Class `CoevolutionArena`
Runs predator-prey or symbiotic co-evolution.

- **__init__**()
- **add_predator**(organism)
- **add_prey**(organism)
- **evaluate_interactions**()
  - Evaluate predator-prey fitness from pairwise interactions.
- **total_organisms**()

### Class `SafetyCheckResult`
Result of a formal safety check on emitted Verilog/NIR.


### Class `FormalSafetyGuard`
Validates emitted organisms against safety constraints before deployment.

Links to the safety_cert module for IEC 61508 compliance.

- **__init__**(bounds)
- **check**(genome)
- **rejection_rate**()

### Class `TournamentSelector`
Tournament selection with configurable pressure.

- **__init__**(tournament_size)
- **select**(population, rng)
- **select_n**(population, n, rng)

### Class `ParetoFront`
Maintains a non-dominated Pareto front.

- **__init__**()
- **update**(organism)
- **size**()

### Class `AgeRegulator`
Culls organisms that exceed a maximum lifespan.

- **__init__**(max_age)
- **apply**(population, current_generation)

### Class `BloatMetrics`
Measures genome complexity for bloat detection.

- **is_bloated**()

### Class `BloatPenalizer`
Penalizes fitness for bloated genomes.

- **__init__**(penalty_weight, threshold)
- **penalize**(fitness, genome)

### Class `ActivationFunc`

### Class `CPPNNode`
One node in a CPPN network.


### Class `CPPNEdge`
One edge in a CPPN network.


### Class `CPPNGenome`
Compositional Pattern Producing Network for developmental encoding.

- **__init__**()
- **query**(x, y)
  - Query the CPPN at coordinates (x, y).
- **_activate**(x, func)
- **generate_weight_matrix**(rows, cols)
  - Generate a weight matrix by querying CPPN at grid positions.
- **num_nodes**()
- **num_edges**()

### Class `HWFitnessReport`
Fitness feedback from actual FPGA execution.

- **hw_composite**()

### Class `HWFitnessCollector`
Collects HW fitness from deployed organisms.

- **__init__**()
- **submit**(report)
- **get**(genome_id)
- **total_reports**()

### Class `GenerationStats`
Statistics for one generation.


### Class `EvoStatisticsTracker`
Records per-generation statistics for analytics.

- **__init__**()
- **record**(stats)
- **generations_tracked**()
- **fitness_trajectory**()
- **diversity_trajectory**()
- **improvement_rate**()

### Class `GenomeDiff`
Structural diff between two genomes.

- **is_identical**()

### Class `ComplexityTracker`
Tracks population complexity over generations.

- **__init__**()
- **record**(generation, population)
- **mean_trajectory**()
- **is_complexifying**()

### Function `genomic_distance(a, b)`
Normalised L1 distance between genome vectors.

Dispatches to the Rust ``evo_substrate_core.py_genomic_distance`` when
the compiled extension is importable. The NumPy fallback is kept as
the reference implementation and produces bit-exact identical values.

### Function `assign_species(population, threshold)`
Assign organisms to species by genomic distance.

First organism of each species is the representative.

### Function `population_diversity(population)`
Mean pairwise genomic distance (0 = clones, 1 = max diversity).

### Function `dominates(a, b)`
True if a Pareto-dominates b.

### Function `compute_bloat(genome, baseline_neurons)`
Compute bloat relative to a baseline.

### Function `shared_fitness(organism, population, sigma)`
Shared fitness: divide by niche count to prevent species domination.

### Function `genome_diff(a, b)`
Compute structural diff between two genomes.

### Function `genome_complexity(genome)`
Measure evolved complexity (information-theoretic).

---

## Module `exceptions`

### Class `SCNeuroError`
Base exception for all SC-NeuroCore errors.


### Class `SCEncodingError`
Probability or bitstream value outside valid range.


### Class `SCConfigError`
Invalid configuration parameter (layer size, threshold, etc.).

Reserved for future use — v3.14.0 callers raise plain
``ValueError`` for configuration errors. Migration tracked
under task #36.


### Class `SCWeightError`
Weight value or shape mismatch.

Reserved for future use — v3.14.0 weight loaders raise plain
``ValueError`` or ``KeyError`` for shape/value errors.
Migration tracked under task #36.


### Class `SCCompilerError`
Compiler pipeline configuration or target error.

Raised by ``sc_neurocore.compiler.pipeline`` for unknown FPGA
target, invalid output names, and path-escape detection.


### Class `SCDependencyError`
Optional dependency (JAX, Torch, PennyLane, Qiskit) not installed.

Raised by JAX backend, quantum hardware bridge, learning
callbacks (W&B, TensorBoard) and similar feature-gated paths.


### Class `SCHardwareError`
FPGA/hardware driver or bitstream error.

Raised by the PYNQ driver (missing IP block) and by Verilog
export when the model is not in the LIF whitelist.


### Class `BitstreamOverflowError`
Bitstream length exceeds the maximum supported width.

Reserved for future use — v3.14.0 bitstream encoders saturate
silently or raise plain ``ValueError`` for related issues
(e.g. the recent Q8.8 dt-underflow guard). Migration tracked
under task #36.


### Class `SeedCollisionError`
Two encoders received the same LFSR seed, breaking decorrelation.

Reserved for future use — v3.14.0 the encoder API does not
detect seed collisions. Implementing this would require a
global seed registry across all encoders. Tracked under
task #36.


### Class `BitwidthMismatchError`
Operands have incompatible fixed-point widths.

Reserved for future use — v3.14.0 the compiler hard-codes
Q8.8 throughout, so multi-width layouts are not supported.
This exception will fire once the compiler accepts mixed
widths. Tracked under task #36.


### Class `CoverageGateError`
Test coverage fell below the required threshold.

Reserved for future use — v3.14.0 coverage gating lives in
CI workflow YAML, not in Python code. The exception is here
so coverage-aware tooling has a stable raise target.
Tracked under task #36.


### Class `HardwareSimMismatchError`
Python golden model and Verilog RTL produced different results.

Reserved for future use — v3.14.0 the cosim suite raises
plain ``AssertionError`` from pytest assertions instead of
this typed exception. Migration tracked under task #36.


### Class `IRCompilationError`
IR graph failed verification or code generation.

Reserved for future use — v3.14.0 the compiler raises the
parent ``SCCompilerError`` (line 90 of pipeline.py)
rather than this leaf type. Migration tracked under task #36.


---

## Module `experimental.alternative_path`

### Class `AlternativePathMode`
Execution mode for an alternative path.


### Class `AlternativePathConfig`
Explicit policy for alternative-path execution.

The default is intentionally conservative: baseline only.


### Class `ComparisonStats`
Comparison summary between baseline and candidate outputs.


### Class `AlternativePathResult`
Structured result of a safe alternative-path execution.

- **to_report**()
  - Return a JSON-serialisable summary for logging or benchmarking.

### Class `AlternativePathCase`
Named input case for repeated comparison and benchmarking.


### Class `AlternativePathBatchSummary`
Aggregate report for a route evaluated over multiple named cases.

- **to_report**()
  - Return a JSON-serialisable summary of the batch run.

### Class `AlternativePathRoute`
Named pair of stable baseline and experimental candidate implementations.

- **describe**()
  - Return route metadata for discovery and documentation.
- **run**(config)
  - Execute the route according to the provided policy.
- **evaluate_cases**(cases, config)
  - Run the route across named cases and aggregate compare/benchmark results.

### Class `AlternativePathRegistry`
Registry for named alternative execution paths.

- **__init__**()
- **register**(route)
- **get**(name)
- **names**()
- **describe**()
- **run**(name, config)
- **evaluate**(name, cases, config)

### Function `_safe_stringify_error(exc)`
### Function `_numeric_comparison(baseline, candidate, config, context)`
### Function `_combine_stats(stats, context)`
### Function `compare_outputs(baseline, candidate, config, context)`
Compare outputs from baseline and candidate implementations.

### Function `_call_timed(fn)`
### Function `_median_runtime_ns(values)`
---

## Module `experimental.builtins`

### Function `build_builtin_registry()`
Register all built-in experimental routes.

### Function `builtin_cases_for_route(route_name)`
Return the default case set for a built-in route.

---

## Module `experimental.examples`

### Function `_baseline_affine_sigmoid(x, bias)`
### Function `_candidate_affine_sigmoid(x, bias)`
### Function `make_demo_sigmoid_route()`
Create a self-contained demo route for benchmarking and comparison.

### Function `build_demo_registry()`
Create a registry with the built-in demo route.

---

## Module `experimental.memory_routes`

### Function `_make_memory_neurons(n_neurons, seed)`
### Function `_run_delayed_recall_trial(cue)`
### Function `_run_delayed_recall_suite(delay_steps)`
### Function `_delayed_recall_local_baseline(delay_steps)`
### Function `_delayed_recall_shared_state_candidate(delay_steps)`
### Function `_compare_delayed_recall_gain(baseline, candidate, config)`
### Function `make_delayed_recall_shared_state_route()`
Route a bounded delayed-recall task against a shared-memory candidate.

The candidate is *quantum-inspired* only in the broad architectural sense:
it adds a non-local shared latent state to a real spiking baseline. It does
not claim to model ATP, Posner molecules, or validated in-vivo quantum
memory.

---

## Module `experimental.physics_routes`

### Function `_heat_cosine_mode_baseline(x_0, horizon, mode_index)`
### Function `_heat_cosine_mode_candidate(x_0, horizon, mode_index)`
### Function `make_heat_cosine_mode_route()`
Route Monte Carlo heat evolution against an exact Neumann cosine mode.

### Function `_harmonic_oscillator_rhs(_t, y)`
### Function `_harmonic_energy(y)`
### Function `_integrate_harmonic(solver, q0, p0, horizon)`
### Function `_harmonic_rk4_baseline(q0, p0, horizon)`
### Function `_harmonic_stormer_verlet_candidate(q0, p0, horizon)`
### Function `make_harmonic_symplectic_route()`
Route harmonic-oscillator integration against the symplectic solver.

### Function `_kuramoto_phase_velocity(phases, omegas, coupling)`
### Function `_kuramoto_order_parameter(phases)`
### Function `_kuramoto_interaction_energy(phases, coupling)`
### Function `_validate_kuramoto_route_inputs(initial_phases, horizon, omegas, coupling, dt)`
### Function `_wrap_phases(phases)`
### Function `_kuramoto_euler_baseline(initial_phases, horizon)`
### Function `_kuramoto_xy_lift_candidate(initial_phases, horizon)`
### Function `make_kuramoto_noiseless_symplectic_lift_route()`
Route a bounded noiseless Kuramoto regime against a symplectic XY lift.

---

## Module `experimental.reporting`

### Function `default_report_path(route_name)`
Return the default JSON report path for a route.

### Function `write_batch_report(summary, path)`
Write a batch summary to a JSON report file.

---

## Module `experimental.solver_routes`

### Function `_lif_rhs(_t, y)`
### Function `_lif_rk4_baseline(v0, current, horizon)`
### Function `_lif_exact_candidate(v0, current, horizon)`
### Function `make_lif_subthreshold_exact_route()`
Route subthreshold LIF integration against the analytical solution.

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
1. Encode scalar input current x_input in &#91;x_min, x_max&#93; as a unipolar
   bitstream of length `length`.
2. At each time step t, set:
   I_t = I_high if bitstream&#91;t&#93; == 1 else I_low
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

## Module `experiments.deep_research_demo`

### Function `run_deep_research_demo()`
---

## Module `experiments.demo_adaptive_audio`

### Function `_generate_eeg_chunk(rng, n_samples, target_hz, sample_rate, entrained_strength, noise_level)`
Generate simulated EEG with a target-frequency component + noise.

Parameters
----------
rng : RandomState
    Random number generator.
n_samples : int
    Number of samples to produce.
target_hz : float
    Entrainment target frequency.
sample_rate : int
    Sampling rate in Hz.
entrained_strength : float
    Amplitude of the entrained sinusoidal component (0-1).
noise_level : float
    Amplitude of background noise.

Returns
-------
np.ndarray of shape (n_samples,)

### Function `run_demo()`
---

## Module `experiments.demo_param_sweep`

### Function `run_pattern_trials(label, x_inputs, weight_values, n_neurons, T, noise_std, n_trials, base_seed)`
Run multiple trials of SCDenseLayer for a given pattern (x_inputs).
Return matrix of shape (n_trials, n_neurons) with firing rates.

### Function `nearest_centroid_multi(sample, centroids)`
Nearest-centroid classifier over K classes.
centroids&#91;k&#93;: firing-rate centroid for class k.

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
centroids&#91;k&#93;: firing-rate centroid for class k.

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

## Module `experiments.demo_sleep_optimization`

### Function `generate_eeg_epoch(stage, n_samples, sample_rate, rng)`
Return *n_samples* of simulated EEG voltage for *stage*.

### Function `_night_schedule(n_epochs)`
Return a realistic stage sequence for *n_epochs* epochs.

Follows a simplified 90-minute cycle pattern:
WAKE -> N1 -> N2 -> N3 -> N2 -> REM -> repeat

### Function `run_demo()`
---

## Module `experiments.demo_swarm_control`

### Function `_bar(value, width)`
Tiny ASCII bar chart helper.

### Function `run_demo()`
---

## Module `experiments.demo_tcbo_consciousness`

### Function `run_demo()`
---

## Module `experiments.demonstration_convergence`

### Function `_emit_hardware_evidence(mlir_str)`
### Function `run_demonstration()`
---

## Module `experiments.exascale_demo`

### Function `run_exascale_demo()`
---

## Module `experiments.experimental_horizons_demo`

### Function `run_horizons_demo()`
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

## Module `experiments.quantum_neuromorphic_demo`

### Function `run_demo()`
---

## Module `experiments.spatial_generative_demo`

### Function `run_spatial_gen_demo()`
---

## Module `experiments.system_level_demo`

### Function `run_system_demo()`
---

## Module `experiments.tcbo_demo_engine`

### Class `ScenarioName`

### Class `ScenarioConfig`

### Class `SyntheticEEGGenerator`
Configurable Kuramoto oscillator network for synthetic EEG.

- **__init__**(N, dt, seed)
- **set_coupling_scale**(scale)
- **apply_anesthesia**(strength)
- **apply_alpha_boost**(factor)
- **apply_coupling_decay**(rate)
- **step**(perturbation)
  - One Kuramoto timestep. Returns phases in &#91;0, 2pi).
- **run**(n_steps)
  - Run n_steps, return (n_steps, N) history.
- **get_order_parameter**()
- **reset**(seed)

### Class `TCBOController`
PI controller for gap-junction coupling kappa.

- **__init__**(tau_h1, Kp, Ki, kappa_min, kappa_max)
- **step**(p_h1, kappa, dt)
  - Compute new kappa from consciousness deficit.
- **reset**()

### Class `TCBODemoSnapshot`
Per-step snapshot of the TCBO demo state.

- **to_dict**()

### Class `TCBODemoEngine`
Orchestrates TCBO consciousness detection scenarios.

- **__init__**(N, dt, seed)
- **get_scenarios**()
- **start_scenario**(name)
  - Initialize and start a named scenario.
- **step**()
  - Advance one timestep.
- **run_scenario**(name, duration_s, subsample)
  - Run a full scenario, returning subsampled snapshots.
- **get_state**()
- **get_history**(last_n)
- **reset**()

### Function `_compute_order_parameter(theta)`
Kuramoto order parameter R = |<e^(i*theta)>|.

### Function `_compute_p_h1_lightweight(phase_history, tau_h1, beta)`
Lightweight p_h1 proxy using phase-locking statistics.

Computes average PLV across pairs from recent history,
then applies logistic squash.

### Function `get_tcbo_demo_engine()`
### Function `reset_tcbo_demo_engine()`
---

## Module `experiments.whitepaper_benchmark`

### Function `run_whitepaper_benchmark()`
---

## Module `explain.spike_explain`

### Class `ExplanationResult`
Result of an explanation method.

- **top_k**(k)
  - Return top-k most important (timestep, neuron_id, score) tuples.
- **summary**()

### Class `SpikeAttributor`
Backward spike attribution via eligibility-trace chain.

Traces the contribution of each input spike to the output
through intermediate layers using eligibility trace products.
Approximation of temporal backpropagation attribution.

Parameters
----------
decay : float
    Temporal decay factor for backward attribution (0-1).

- **__init__**(decay)
- **attribute**(spikes, weights, output_neuron)
  - Compute per-input-spike attribution scores.

### Class `TemporalSaliency`
Perturbation-based temporal saliency.

For each input spike, measure the change in output when that spike
is removed. Spikes whose removal causes large output change are
salient (important).

Parameters
----------
run_fn : callable
    Function that takes input spikes (T, N) and returns output
    spike counts or rates (N_output,).

- **__init__**(run_fn)
- **explain**(spikes, output_neuron)
  - Compute perturbation-based saliency for each input spike.

### Class `CausalImportance`
Causal importance via forward intervention.

Silence each neuron (clamp to zero) across all timesteps and
measure the impact on classification output. Builds a per-neuron
causal importance score.

Parameters
----------
run_fn : callable
    Function that takes input spikes (T, N) and returns output (N_output,).

- **__init__**(run_fn)
- **explain**(spikes, output_neuron)
  - Compute causal importance by silencing each neuron.

---

## Module `explainability.explainability`

### Class `LFSRReplay`
Deterministic LFSR-16 replay engine.

Mirrors ``core_engine::Lfsr16`` polynomial: x^16 + x^14 + x^13 + x^11 + 1.
Given the same seed, reproduces the exact same bitstream for formal
verification and replay-based auditing.

- **__init__**(seed)
- **step**()
  - Advance one step, return new register value.
- **encode**(threshold, length)
  - Generate a bitstream by comparing LFSR output against threshold.
- **reset**()
  - Reset to initial seed for replay.

### Class `SpikeDecision`

### Class `DecisionMargin`
How close a decision was to flipping.


### Class `DecisionNode`
One node in a spike decision tree.

- **is_leaf**()
- **margin**()

### Class `SpikeDecisionTree`
Captures "why this spike fired" as a verifiable decision tree.

- **__init__**()
- **add_decision**(neuron_id, bitstream, threshold, scc, parent, timestep, layer_id, contributing_neurons, threshold_q16)
  - Record a spike decision from bitstream observation.
- **depth**()
- **_compute_depth**(node)
- **num_spikes**()
- **num_nodes**()
- **nodes_at_layer**(layer_id)
  - Return all nodes at a given layer.
- **nodes_at_timestep**(timestep)
  - Return all nodes at a given timestep.
- **get_node**(neuron_id)
  - Look up a node by neuron ID.
- **spike_path**()
  - Return the chain of spiking nodes from root down.
- **_collect_spike_path**(node, path)
- **to_dict**()
- **_node_to_dict**(node)

### Class `ProvenanceStep`
One step in a full provenance chain.


### Class `ProvenanceTrace`
Full chain from input → encoding → computation → spike decision.

- **__init__**()
- **add_step**(stage, description, data, metadata)
  - Record one provenance step.
- **finalize**()
- **is_complete**()
- **num_steps**()
- **chain_hash**()
  - Hash of the entire provenance chain for tamper detection.
- **to_list**()

### Class `RegulatoryMetadata`
IEC 62304 / FDA SaMD traceability fields.


### Class `FormalPropertyLink`
Cross-reference to SymbiYosys formal verification.


### Class `VerifiabilityReport`
Formal audit report with hash verification.


### Class `SensitivityResult`
Result of a 'what-if' threshold perturbation.


### Class `SensitivityAnalyzer`
Counterfactual analysis: 'would the decision flip if threshold ±N?'

- **analyze**(node, perturbations)
- **critical_delta**(node)
  - Smallest threshold change that flips the decision.

### Class `CausalAttribution`
Attribution of a spike to upstream neurons.

- **top_contributors**()
  - Sorted (descending) list of contributing neurons.

### Class `CausalAttributor`
Computes causal attribution from input neurons to output spike.

- **attribute**(target, input_bitstreams, weights)
  - Compute per-input-neuron contribution to the target popcount.

### Class `DiffEntry`
One field that differs between two explanations.


### Class `ExplanationDiff`
Compares two decision nodes to find divergence points.

- **diff**(a, b)

### Class `TemporalWindow`
Records decisions across timesteps for temporal attribution.

- **__init__**()
- **add**(node)
- **spike_rate_at**(timestep)
- **active_timesteps**()
- **peak_timestep**()
  - Timestep with the highest spike rate.
- **num_timesteps**()

### Class `NaturalLanguageExplainer`
Generates human-readable explanation strings.

- **explain_node**(node)
- **explain_attribution**(attr)
- **explain_sensitivity**(results)
- **enhance_with_local_llm**(text)
  - Enhance a deterministic explanation through the local LLM bridge.
- **explain_node_with_local_llm**(node)
  - Generate a local-LLM-enhanced explanation for one decision node.

### Class `MultiLayerTrace`
Traces decisions across network layers.

- **__init__**()
- **add**(node)
- **layer_ids**()
- **spikes_at_layer**(layer_id)
- **spike_rate_at_layer**(layer_id)
- **propagation_path**()
  - Per-layer spike rate for visualising propagation.

### Class `SymbolicPathStep`
One step in a symbolic decision path.


### Class `SymbolicPath`
Human-readable symbolic path: input → encoding → decision.

- **__init__**()
- **add**(neuron_id, decision, reason)
- **length**()
- **to_list**()

### Class `ExplainabilityEngine`
End-to-end explainability: replay + decision tree + provenance.

- **__init__**(seed)
- **explain_spike**(neuron_id, threshold_q16, bitstream_length, spike_threshold_count, scc, timestep, layer_id, contributing_neurons)
  - Explain one spike decision via deterministic replay.
- **verify**(regulatory, formal_properties)
  - Generate a verifiability report with full replay check.
- **replay_bitstream**(threshold_q16, length)
  - Replay a bitstream from the engine's seed (for external comparison).
- **sensitivity**(node, perturbations)
  - Run sensitivity analysis on a decision node.
- **attribute**(target, input_bitstreams, weights)
  - Compute causal attribution for a decision.
- **explain_spike_with_local_llm**(neuron_id, threshold_q16, bitstream_length, spike_threshold_count)
  - Run the deterministic explainability path, then enhance it locally.

---

## Module `export.compiler_export`

### Class `SSAEnvironment`
Manages Static Single Assignment (SSA) registers for MLIR/Relay.

- **__init__**()
- **allocate**(edge_name)
- **get**(edge_name)

### Class `ShapeInference`
Infers tensor shapes dynamically across the SNN graph.

- **__init__**(input_shapes)
- **infer**(node)

### Class `CompilerExporter`
- **__init__**(target)
- **_topological_sort**(nodes)
  - Kahn's algorithm for topological sorting of the DAG.
- **_format_mlir_type**(shape, dtype)
- **export_to_mlir**(ir_graph, input_shapes)
  - Emits strict SSA MLIR text via topological traversal.

---

## Module `export.onnx_export`

### Class `ONNXTensorType`
- **to_dict**()

### Class `ONNXNode`
- **to_dict**()

### Class `ONNXGraph`
- **to_dict**()
- **to_json**(indent)

### Class `ONNXExporter`
Exports SC-NeuroCore IR graphs to ONNX-compatible representation.

- **__init__**(graph_name)
- **_infer_type**(node_type, shape)
- **export**(ir_graph, input_shapes, metadata)
  - Convert SC-IR graph to ONNX graph representation.

---

## Module `export.onnx_exporter`

### Class `SCOnnxExporter`
Export SC networks to ONNX protobuf or legacy JSON.

- **export**(layers, filename)
  - Export *layers* to *filename*.
- **_export_protobuf**(layers, filename)
- **_export_json**(layers, filename)

### Function `_classify_op(layer)`
---

## Module `export.pipeline`

### Class `PipelineStageResult`
Result from a single pipeline stage.

`output` accepts strings (verilog / relay / mlir text) and the
richer ONNXGraph object so each stage can store its native
representation without stringifying.


### Class `PipelineResult`
Full pipeline result.

- **success**()
- **summary**()

### Class `ExportPipeline`
End-to-end: NeuronPlugin → ONNX → TVM → MLIR → Verilog.

Usage::

    pipeline = ExportPipeline()
    result = pipeline.run("LIF", n_neurons=64, bitstream_length=256)
    print(result.verilog)

- **__init__**(registry, target)
- **run**(neuron_name, n_neurons, bitstream_length, module_name)
  - Execute the full export pipeline for a neuron model.
- **_stage_model_zoo**(neuron_name, n_neurons, bitstream_length)
  - Stage 1: Look up neuron plugin and validate.
- **_stage_verilog**(neuron_name, n_neurons, bitstream_length, module_name)
  - Stage 2: Generate SystemVerilog.
- **_stage_onnx**(ir_graph, n_neurons, bitstream_length)
  - Stage 3: Export to ONNX graph.
- **_stage_tvm**(ir_graph, n_neurons, bitstream_length)
  - Stage 4: Lower to TVM Relay.
- **_stage_mlir**(ir_graph, n_neurons, bitstream_length)
  - Stage 5: Export to MLIR/SSA.

### Function `_build_ir_graph(neuron_name, n_neurons, bitstream_length, meta)`
Build a lightweight IR graph from neuron plugin metadata.

---

## Module `export.tvm_lowering`

### Class `TargetDevice`

### Class `TargetSchedule`
- **for_fpga**(cls, vendor)
- **for_gpu**(cls)
- **for_cpu**(cls)

### Class `RelayFunction`
- **to_relay_text**()

### Class `TVMLowering`
Lowers SC-NeuroCore IR to TVM Relay IR text representation.

- **__init__**(schedule)
- **_shape_str**(shape, dtype)
- **_lower_node**(node, shapes)
  - Returns (relay_line, output_type_str).
- **lower**(ir_graph, input_shapes, func_name)
  - Lower SC-IR graph to Relay IR text.
- **emit_build_script**(relay_text)
  - Generate a self-contained TVM build script for the lowered IR.

---

## Module `fault_injection.fault_injection`

### Class `FaultModel`

### Class `RadiationProfile`
Preset radiation environments with typical BER (bit error rate).

- **leo**(cls)
- **geo**(cls)
- **deep_space**(cls)
- **terrestrial**(cls)

### Class `FaultInjectionResult`
- **probability_original**()
- **probability_corrupted**()
- **absolute_error**()

### Class `ResilienceReport`
- **summary**()

### Class `FaultInjector`
Applies configurable faults to SC bitstreams.

- **__init__**(seed)
- **inject**(bitstream, model, ber)
  - Inject faults into a boolean bitstream.
- **inject_at_positions**(bitstream, positions)
  - Flip specific bit positions (deterministic injection).

### Class `ResilienceBenchmark`
Monte-Carlo resilience benchmarking harness.

- **__init__**(seed)
- **_generate_bitstream**(length, probability)
  - Generate a random SC bitstream encoding a given probability.
- **run**()
  - Run Monte-Carlo fault injection trials.
- **sweep_ber**()
  - Sweep across multiple BER values to produce a degradation curve.

---

## Module `fault_injection.resilience_mode`

### Class `ResilienceModeConfig`
Configuration for a resilience-mode run over one bitstream layer.

- **__post_init__**()

### Class `ResilienceModeTrialReport`
Aggregate measurements for one fault model.

- **to_dict**()
  - Return a JSON-ready report.

### Class `ResilienceModeReport`
Full resilience-mode output for one layer and radiation profile.

- **requires_replay**()
  - Whether any fault model requires deterministic replay.
- **to_dict**()
  - Return a JSON-ready report.

### Class `FaultInjectionResilienceMode`
Run seeded fault-injection trials and degradation policy together.

- **__init__**(config)
- **run**(bitstreams)
  - Evaluate resilience for a binary ``(neurons, bits)`` layer.
- **_run_fault_model**(streams, flat, fault_model, model_index)
- **_validate_bitstreams**(bitstreams)

---

## Module `fault_injection.resilience_policy`

### Class `DegradationAction`
Runtime action recommended after seeded fault diagnosis.


### Class `SeededFaultObservation`
Fault-injection observation linked to deterministic replay seed.


### Class `DegradationPlan`
Graceful-degradation decision derived from seeded diagnostics.

- **to_dict**()
  - Return a JSON-ready summary without expanding full bitstreams.

### Class `GracefulDegradationPolicy`
Combine seeded fault injection with stochastic-doctor diagnosis.

- **evaluate**(bitstreams)
  - Inject seeded faults, audit the layer, and recommend degradation.
- **_inject_layer**(bitstreams, fault_model, ber, seed)
- **_plan**(observation)
- **_make_plan**(action, observation, multiplier, reason)
- **_validate_bitstreams**(bitstreams)

---

## Module `federated.federated_sc`

### Class `DPMechanism`
Bitstream-level differential privacy via calibrated bit-flipping.

Instead of adding Gaussian/Laplace noise to real-valued gradients,
we flip bits in the SC bitstream with probability p. This achieves
(ε,0)-differential privacy where ε = ln((1-p)/p) per bit.

For a bitstream of length L, the total privacy cost is ε_total via
Rényi-DP composition (tighter than naive composition).

- **flip_probability**()
  - Calibrated bit-flip probability for the target ε.
- **privatise**(bitstream, rng)
  - Apply DP noise by flipping bits with calibrated probability.
- **per_bit_epsilon**()
  - Privacy cost per individual bit.
- **total_epsilon**(bitstream_length)
  - Total ε under advanced composition (Rényi-based bound).

### Class `PrivacyAccountant`
Rényi Differential Privacy (RDP) composition tracker.

Tracks cumulative privacy budget across federated rounds.
Uses the moments accountant for tight composition.

- **consume_round**(mechanism, bitstream_length)
  - Account for one round. Returns True if budget remains.
- **current_epsilon**()
  - Convert accumulated RDP to (ε,δ)-DP.
- **remaining_epsilon**()
  - How much budget is left.
- **is_exhausted**()

### Class `SecretShare`
Additive secret sharing over GF(2) for bitstream aggregation.

Splits a bitstream into N shares such that XOR of all shares
recovers the original. Individual shares reveal nothing.

- **split**(bitstream, rng)
  - Split bitstream into additive GF(2) shares.
- **reconstruct**(shares)
  - Reconstruct bitstream from all shares (XOR).
- **verify_reconstruction**(original, shares)
  - Verify that shares reconstruct correctly.

### Class `CommitmentScheme`
SHA-256 commitment for verifiable aggregation.

Compatible with the ZKPVerifier in security/zkp.py.

- **commit**(data, nonce)
  - Create a binding commitment to data.
- **verify**(data, commitment, nonce)
  - Verify a commitment.
- **generate_nonce**(rng)
  - Generate a random 32-byte nonce.

### Class `SCGradientEncoder`
Encodes real-valued gradients as SC bitstreams with DP noise.

- **encode**(gradients, seeds, rng)
  - Encode gradient vector as privatised SC bitstreams.
- **decode**(bitstreams, g_min, g_max)
  - Decode SC bitstreams back to gradient values.

### Class `FederatedClient`
One participant in federated SC learning.

- **__post_init__**()
- **local_train**(data, labels, lr)
  - Simulate one local training step (gradient computation).
- **encode_gradients**(gradients)
  - Encode gradients as privatised SC bitstreams + commitment.

### Class `FederatedAggregator`
Secure aggregation server for federated SC bitstreams.

- **aggregate_bitstreams**(client_bitstreams, weights)
  - Weighted majority-vote aggregation of SC bitstreams.
- **detect_outliers**(client_bitstreams, threshold)
  - Detect malicious/outlier clients via cosine similarity.
- **verify_commitments**(client_bitstreams, commitments, nonces)
  - Verify that submitted bitstreams match commitments.

### Class `ConvergenceTracker`
Tracks loss/gradient-norm across federated rounds.

- **record**(aggregated_gradient)
  - Record one round's gradient norm.
- **record_loss**(loss)
- **converged**()
  - Simple heuristic: converged if last 5 grad norms < 0.01.
- **trend**()
  - Return trend direction.

### Class `FederatedRound`
Orchestrates one complete round of federated SC learning.

- **run**(data_per_client, labels_per_client, client_weights)
  - Execute one federated round.
- **status**()
  - Return current federated learning status.

### Class `DPCertificate`
Exportable privacy proof document for regulatory compliance.

- **from_accountant**(cls, accountant, mechanism, bitstream_length)
- **to_dict**()
- **is_compliant**()

### Class `AdaptiveEpsilonScheduler`
Dynamically adjust ε per round based on convergence state.

- **__post_init__**()
- **step**(converging)
  - Compute ε for the next round.

### Class `ErrorFeedback`
Error feedback for gradient sparsification.

Accumulates the residual from sparsification to avoid information
loss over rounds.

- **accumulate**(gradients)
  - Add residual from previous round.
- **update**(original, sparse)
  - Store the residual (what was lost to sparsification).

### Class `AuditEntry`
Single audit log entry for one federated round.

- **__post_init__**()

### Class `AuditLog`
Per-round provenance record for regulatory compliance.

- **log_round**(round_number, num_active, epsilon_consumed, grad_norm)
- **to_list**()
- **total_rounds**()
- **max_epsilon**()

### Function `_lfsr16_step(reg)`
Single step of LFSR-16 (polynomial x^16+x^14+x^13+x^11+1).

### Function `lfsr_encode(value, seed, length)`
Encode a probability &#91;0,1&#93; into a packed bitstream using LFSR-16.

### Function `bitstream_probability(bits)`
Estimate probability from a bitstream.

### Function `clip_gradients(gradients, max_norm)`
L2 gradient clipping for formal DP sensitivity bounds.

Clips the gradient vector so its L2 norm ≤ max_norm.
This is required for provable (ε,δ)-DP guarantees.

### Function `sparsify_topk(gradients, k)`
Top-k gradient sparsification.

Returns (sparse_gradients, mask) where only k largest-magnitude
entries are non-zero.  Reduces communication cost proportionally.

### Function `poisson_subsample(clients, sampling_rate, rng)`
Poisson subsampling of clients for privacy amplification.

Each client is independently included with probability ``sampling_rate``.
This provides privacy amplification by a factor of O(sampling_rate).

### Function `stochastic_quantize(gradients, levels, rng)`
Stochastic quantization to reduce communication bits.

Quantizes each gradient to one of ``levels`` levels with
unbiased randomized rounding. E&#91;Q(g)&#93; = g.

### Function `krum_select(client_vectors, num_byzantine)`
Multi-Krum selection: find the vector closest to most others.

Returns the index of the most "central" client update.
Tolerates up to ``num_byzantine`` malicious clients.

### Function `trimmed_mean(client_vectors, trim_fraction)`
Coordinate-wise trimmed mean aggregation.

Removes the top and bottom ``trim_fraction`` of values per
coordinate before averaging. Robust to Byzantine clients.

### Function `fedprox_gradient(gradients, local_weights, global_weights, mu)`
Add FedProx proximal term for non-IID robustness.

grad_proximal = grad + μ * (w_local - w_global)

### Function `amplified_epsilon(base_epsilon, sampling_rate)`
Compute privacy amplification from Poisson subsampling.

Uses the Balle et al. (2018) bound: ε' ≈ log(1 + q(e^ε - 1))
where q is the sampling rate.

---

## Module `few_shot.haam`

### Class `HebbianFewShot`
Hebbian few-shot learner using associative memory.

Support examples stored via one-shot Hebbian weight update.
Query classified by comparing spike-rate representation to
stored prototypes.

Parameters
----------
n_features : int
    Input feature dimension.
n_classes : int
    Number of classes to support.
lr_hebbian : float
    Hebbian learning rate for support storage.

- **__init__**(n_features, n_classes, lr_hebbian)
- **store**(spike_pattern, label)
  - Store one support example via Hebbian update.
- **query**(spike_pattern)
  - Classify a query pattern by cosine similarity to stored memories.
- **few_shot_episode**(support_x, support_y, query_x)
  - Run a complete few-shot episode.
- **reset**()

### Class `SpikePrototypeNet`
Prototypical network in spike domain.

Compute class prototypes as mean spike-rate vectors from support set.
Classify queries by nearest prototype (Euclidean or cosine).

Parameters
----------
n_features : int
metric : str
    'cosine' or 'euclidean'

- **classify**(support_x, support_y, query_x)
  - Classify query set using support set prototypes.

---

## Module `formal.counterexample_replay`

### Class `RateBoundReplayResult`
Replay result for an aligned-window network rate-bound property.


### Class `RefractoryReplayResult`
Replay result for a monitored-output refractory invariant.


### Class `AntagonisticReplayResult`
Replay result for a mutually-exclusive output-pair invariant.


### Class `TemporalSeparationReplayResult`
Replay result for a bidirectional temporal-separation invariant.


### Class `PopulationCoactivationReplayResult`
Replay result for a population-level output coactivation cap.


### Class `PopulationSilenceReplayResult`
Replay result for post-coactivation global output silence.


### Class `PopulationInactivityReplayResult`
Replay result for bounded consecutive population inactivity.


### Function `replay_rate_bound_counterexample(spike_trace, rate_bound)`
Replay a spike trace against the same aligned-window rate bound used by SVA.

### Function `replay_refractory_counterexample(spike_trace, refractory)`
Replay a spike trace against a monitored-output refractory invariant.

### Function `replay_antagonistic_counterexample(spike_trace, exclusion)`
Replay a spike trace against a mutually-exclusive output-pair invariant.

### Function `replay_temporal_separation_counterexample(spike_trace, separation)`
Replay a spike trace against a bidirectional output temporal separation.

### Function `replay_population_coactivation_counterexample(spike_trace, population)`
Replay a spike trace against a population coactivation cap.

### Function `replay_population_silence_counterexample(spike_trace, silence)`
Replay a spike trace against a post-coactivation global silence contract.

### Function `replay_population_inactivity_counterexample(spike_trace, inactivity)`
Replay a spike trace against a bounded consecutive-inactivity contract.

### Function `_select_binary_spike(sample, output_index)`
### Function `_count_binary_spikes(sample)`
### Function `_as_binary_bool(value)`
---

## Module `formal.lean_bridge`

### Class `FormalProofEngine`
Invokes Lean 4 `safety_bounds.lean` to formally verify mathematical parameters dynamically.

- **__init__**()
- **is_available**()
- **check_proofs**()
  - Invokes the native `lean --check` verification boundary safely testing axioms.

---

## Module `formal.network_properties`

### Class `DenseLIFNetworkSpec`
Formal boundary contract for a dense LIF network HDL module.

- **__post_init__**()

### Class `NetworkRateBound`
Bound a selected output neuron's spike count inside a fixed time window.

- **__post_init__**()

### Class `NetworkRefractoryInvariant`
Forbid a selected output from spiking during its refractory window.

- **__post_init__**()

### Class `NetworkAntagonisticOutputExclusion`
Forbid two antagonistic network outputs from spiking in the same cycle.

- **__post_init__**()

### Class `NetworkOutputTemporalSeparation`
Forbid two outputs from spiking within a bounded cycle window.

- **__post_init__**()

### Class `NetworkPopulationCoactivationCap`
Bound the number of simultaneously active outputs in a sample cycle.

- **__post_init__**()

### Class `NetworkPopulationSilenceAfterCoactivation`
Require global output silence after a population coactivation event.

- **__post_init__**()

### Class `NetworkPopulationInactivityBound`
Bound consecutive valid cycles with no active network outputs.

- **__post_init__**()

### Function `validate_systemverilog_identifier(value)`
Return ``value`` after checking it is a plain SystemVerilog identifier.

### Function `_validate_positive_int(value)`
### Function `_validate_non_negative_int(value)`
---

## Module `formal.property_compiler`

### Function `compile_dense_lif_fixture_rtl(network)`
Compile a deterministic dense LIF fixture RTL module for formal runs.

### Function `compile_network_rate_bound_sva(network, rate_bound)`
Compile a network output spike-rate contract into deterministic SVA.

### Function `compile_network_refractory_sva(network, refractory)`
Compile a network output refractory contract into deterministic SVA.

### Function `compile_network_antagonistic_exclusion_sva(network, exclusion)`
Compile a mutually-exclusive output-pair contract into deterministic SVA.

### Function `compile_network_temporal_separation_sva(network, separation)`
Compile a bidirectional output temporal-separation contract into SVA.

### Function `compile_network_population_coactivation_sva(network, population)`
Compile a population-level output coactivation cap into deterministic SVA.

### Function `compile_network_population_silence_sva(network, silence)`
Compile a post-coactivation global silence contract into deterministic SVA.

### Function `compile_network_population_inactivity_sva(network, inactivity)`
Compile a bounded global-output inactivity contract into deterministic SVA.

---

## Module `formal.report_schema`

### Class `FormalReportValidationError`
Raised when a formal network verification report violates its schema.


### Function `validate_formal_network_report(payload)`
Validate a formal network verification report without external dependencies.

### Function `_validate_artifacts(artifacts)`
### Function `_validate_rate_replay(value, field)`
### Function `_validate_refractory_replay(value, field)`
### Function `_validate_antagonistic_replay(value, field)`
### Function `_validate_temporal_replay(value, field)`
### Function `_validate_population_replay(value, field)`
### Function `_validate_population_silence_replay(value, field)`
### Function `_validate_population_inactivity_replay(value, field)`
### Function `_expect_artifact_path(artifacts, key)`
### Function `_expect_mapping(value, field)`
### Function `_expect_equal(value, expected, field)`
### Function `_expect_str(value, field)`
### Function `_expect_string(value, field)`
### Function `_expect_bool(value, field)`
### Function `_expect_non_negative_int(value, field)`
### Function `_expect_optional_non_negative_int(value, field)`
---

## Module `fusion.multimodal`

### Class `ModalityConfig`
Configuration for one sensor modality.


### Class `MultiModalFusion`
Fuse spike trains from multiple sensor modalities.

Parameters
----------
modalities : list of ModalityConfig
    Sensor modality definitions.
output_dt_us : float
    Output time bin width in microseconds (common timebase).
mode : str
    Fusion mode: 'concatenate', 'sum', or 'attention'.

- **__init__**(modalities, output_dt_us, mode)
- **fuse**(spike_trains, duration_us)
  - Fuse spike trains from all modalities into a unified output.

---

## Module `fusion.sensor_fusion`

### Class `SensorModality`

### Class `EventStream`
Timestamped event stream from a single sensor modality.

- **num_events**()
- **duration_us**()
- **event_rate**()
- **to_bitstream**(length, num_channels)
  - Convert event stream to SC bitstream matrix (channels × length).

### Class `BitstreamDecorrelator`
On-the-fly decorrelation for heterogeneous bitstreams.

Uses per-stream LFSR-based scrambling to break inter-stream
correlations introduced by shared clock domains or spatial
proximity.

- **__init__**(seed)
- **decorrelate**(streams, method)
  - Decorrelate a list of bitstream matrices.
- **_generate_mask**(shape, seed, method)
- **_lfsr_mask**(shape, seed)
- **_sobol_mask**(shape, seed)
- **measure_scc**(a, b)
  - Compute stochastic cross-correlation between two bitstreams.

### Class `CrossModalAttention`
SC-domain cross-modal attention kernel.

Implements query-key-value attention using stochastic arithmetic:
  - Q·K similarity: SC-AND (bitwise AND = multiplication)
  - Weighted V: SC-MUX (bitwise multiplexer = scaled addition)

- **__init__**(num_channels, seed)
- **_sc_and**(a, b)
- **_sc_mux**(a, b, sel)
- **attend**(query_stream, key_stream, value_stream)
  - Compute cross-modal attention in SC domain.
- **_project**(stream, weights)

### Class `FusionMetrics`
Metrics from a sensor fusion pass.


### Class `SensorFusionLayer`
Multi-stream sensor fusion with per-modality weighting.

- **__init__**(num_channels, bitstream_length, seed)
- **set_weight**(modality, weight)
- **fuse**(streams, use_attention)
  - Fuse multiple event streams into a single SC bitstream.

### Class `HDCBinding`
Hyperdimensional computing for cross-modal representation binding.

Uses binary hypervectors for modality-independent representation.
Binding: XOR (permutation-invariant association).
Bundling: majority vote (superposition).

- **__init__**(dim, seed)
- **get_hypervector**(key)
  - Get or create a random hypervector for a key.
- **bind**(a, b)
  - XOR binding: associate two representations.
- **bundle**(vectors)
  - Majority-vote bundling: superpose multiple vectors.
- **similarity**(a, b)
  - Cosine-like similarity via Hamming distance.
- **encode_stream**(stream, num_channels)
  - Encode an event stream as a single hypervector.

### Class `DVSAdapter`
Adapter for Dynamic Vision Sensor (event camera).

- **encode_events**(timestamps, x, y, polarities, resolution)

### Class `CochleaAdapter`
Adapter for silicon cochlea (frequency-to-channel mapping).

- **__init__**(num_channels, freq_min_hz, freq_max_hz)
- **freq_to_channel**(freq_hz)
  - Log-scale frequency to channel mapping (tonotopic).
- **encode_spikes**(timestamps, frequencies)

### Class `TactileAdapter`
Adapter for e-skin / tactile sensor arrays.

- **encode_pressure**(timestamps, taxel_ids, pressures, threshold)
  - Convert pressure readings to ON/OFF events.

### Class `IMUAdapter`
Adapter for IMU / proprioceptive streams.

- **encode_angular_rate**(timestamps, axis_id, rates_dps, deadzone_dps)
  - Convert angular rate to events (above deadzone).

### Class `TemporalAligner`
Aligns heterogeneous event streams to a common time window.

- **__init__**(window_us)
- **align**(streams)
  - Slice all streams to their overlapping time window.
- **slice_windows**(stream)
  - Slice a stream into fixed-width time windows.

### Class `FusionVerilogEmitter`
Generates SystemVerilog for configurable multi-modal fusion.

- **emit**(module_name, num_streams, bitstream_width, use_attention)

### Class `FusionEnergyEstimate`
Sub-mW energy estimate for fusion pipeline.

- **total_mw**()

### Class `FusionEnergyEstimator`
Estimates per-inference energy for SC fusion on FPGA.

- **__init__**(tech_node_nm, vdd_v)
- **estimate**(num_streams, num_channels, bitstream_length, use_attention, clock_mhz)

---

## Module `generative.audio_synthesis`

### Class `SCAudioSynthesizer`
SC Audio Synthesis engine.
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
Generator for 3D mesh and point cloud outputs from stochastic voxel data.

Implements Marching Cubes algorithm for isosurface extraction.

- **export_point_cloud_json**(points, intensities, filename)
  - Export a point cloud to JSON format.
- **generate_surface_mesh**(voxel_grid, iso_level)
  - Generate a surface mesh from a voxel grid using Marching Cubes.
- **_get_edge_vertices**(i, j, k, cube_vals, iso_level)
  - Compute vertex positions along cube edges.
- **_compute_normals**(vertices, faces)
  - Compute vertex normals from face normals.
- **export_mesh_obj**(mesh, filename)
  - Export mesh to OBJ format.
- **export_mesh_json**(mesh, filename)
  - Export mesh to JSON format.
- **bitstream_to_voxels**(bitstreams, grid_size)
  - Convert bitstream outputs to a voxel grid.
- **generate_from_scpn**(scpn_outputs, grid_size)
  - Generate 3D mesh directly from SCPN layer outputs.

---

## Module `graphs.gnn`

### Class `StochasticGraphLayer`
Event-Based Graph Convolution Layer.
Message Passing happens via Bitstreams.

- **__init__**(adj_matrix, n_features)
- **forward**(node_features)
  - node_features: (N, Features)

---

## Module `hardware.constraints`

### Class `Violation`
A single hardware constraint violation.

Attributes:
    neuron_id: Index of the offending neuron.
    constraint: Name of the violated constraint.
    value: Actual value that violates the constraint.
    limit: Maximum allowed value.
    message: Human-readable description.


### Class `HardwareConstraints`
Constraint set for a target device.

Derived from a ``DeviceSpec``, or specified manually.

- **from_device**(cls, device)
  - Derive constraints from a device specification.

### Class `ConstraintChecker`
Check and optionally fix hardware constraint violations.

- **check**(adjacency, constraints, weights, delays)
  - Check all constraints. Returns list of violations (empty if clean).
- **auto_fix**(adjacency, constraints)
  - Attempt automatic fixes: prune weakest connections to satisfy fan-in/out.

---

## Module `hardware.deployment`

### Class `DeploymentPackage`
Self-contained deployment artifact for neuromorphic hardware.

Attributes:
    device: Target device specification.
    placements: Neuron-to-core mapping.
    config_blob: Binary configuration data for the target.
    metadata: Additional deployment metadata.


### Class `Deployer`
Create and validate deployment packages.

- **package**(adjacency, device, placements, weights)
  - Create a deployment package.
- **validate**(package)
  - Validate a deployment package for consistency.
- **summary**(package)
  - Human-readable deployment summary.
- **_build_config**(weights, placements, device)
  - Build binary configuration blob.

---

## Module `hardware.device`

### Class `DeviceFamily`
Supported neuromorphic hardware families.


### Class `DeviceSpec`
Physical specification of a neuromorphic device.

Attributes:
    family: Hardware family identifier.
    cores: Number of neuro-cores on the chip.
    neurons_per_core: Maximum neurons per core.
    synapses_per_core: Maximum synaptic connections per core.
    axons_per_core: Maximum input axons per core.
    tick_ns: Duration of one simulation tick in nanoseconds.
    precision_bits: Weight precision in bits.
    supports_learning: Whether on-chip learning is supported.
    power_per_core_mw: Estimated power per active core (mW).
    max_fan_in: Maximum fan-in per neuron.
    max_fan_out: Maximum fan-out per neuron.
    weight_bits: Synaptic weight bit-width.
    delay_bits: Synaptic delay bit-width.
    max_delay_ticks: Maximum synaptic delay in ticks.


### Function `get_device(family)`
Look up a device specification by family name or enum.

---

## Module `hardware.mapping`

### Class `NeuronPlacement`
Placement of a single neuron on hardware.


### Class `Mapper`
Map neurons to cores using different strategies.

- **map_greedy**(adjacency, device)
  - Greedy sequential mapping: fill cores one by one.
- **map_balanced**(adjacency, device)
  - Balanced mapping: distribute neurons evenly across cores.
- **map_locality**(adjacency, device)
  - Locality-aware mapping: cluster connected neurons on same core.

---

## Module `hardware.resource_estimator`

### Class `ResourceEstimate`
Hardware resource estimation result.

Attributes:
    cores_needed: Minimum cores to host the network.
    neurons_mapped: Total neurons to place.
    synapses_mapped: Total synapses to route.
    utilization_pct: Average core utilization (%).
    power_mw: Estimated total power (mW).
    latency_us: Estimated single-tick latency (µs).
    fits: Whether the network fits on the target device.


### Class `ResourceEstimator`
Estimate hardware cost for deploying an SC-NeuroCore network.

- **estimate**(adjacency, device)
  - Estimate resources from an adjacency matrix.
- **fits**(adjacency, device)
  - Quick check: does the network fit on the device?
- **compare**(adjacency, devices)
  - Compare resource requirements across multiple devices.

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

## Module `hdl.resources`

### Function `list_baseline_primitive_rtl()`
Return the static baseline Verilog primitives bundled with the wheel.

### Function `baseline_primitive_path(name)`
Return an importlib resource handle for a known baseline primitive.

### Function `baseline_primitive_text(name)`
Read a known baseline primitive as UTF-8 text.

---

## Module `hdl_gen._ident`

### Function `sanitize_ident(name, context)`
Validate an HDL-facing identifier before interpolating it into source.

---

## Module `hdl_gen.aer_emitter`

### Class `AEREmitter`
Emit a research-stage AER wrapper around the existing sync HDL path.

This is intentionally conservative: the compute pipeline remains clocked
and the output is wrapped in a 4-phase AER-style request/acknowledge
interface. It is not a QDI async network replacement.

- **__init__**(module_name, bus_width)
- **add_layer**(layer_type, name, params)
- **generate**()
- **_require_positive_int**(value, name)
- **_validate_layers**()
- **_dense_input_width**(params, previous_width)
- **_dense_output_width**(params)
- **_dense_layer_widths**()

---

## Module `hdl_gen.bus_interface`

### Function `generate_bus_wrapper(inner_module, params)`
Generate a bus-attached wrapper around a compiled neuron module.

Parameters
----------
inner_module : str
    Name of the inner Verilog neuron module (e.g. ``"sc_lif"``).
params : dict&#91;str, int&#93;
    Mapping from Verilog parameter name to bit width.
bus : str
    Bus protocol: ``"axi_lite"`` or ``"wishbone"``.
data_width : int
    Neuron fixed-point data width (e.g. 16 for Q8.8).
addr_width : int
    Address bus width (default 8 → 256 bytes → 64 registers).
bus_data_width : int
    Bus data width (default 32 for standard SoC buses).
base_address : int
    Base address for documentation (not used in RTL).

Returns
-------
str
    Complete SystemVerilog source for the bus wrapper module.

### Function `generate_register_map(params)`
Return the register map for a neuron's parameters.

Parameters
----------
params : dict&#91;str, int&#93;
    Parameter names and their bit widths.
base_address : int
    Starting address.

Returns
-------
dict&#91;str, int&#93;
    Mapping from register name to byte address.

### Function `_generate_axi_lite(inner_module, params, data_width, addr_width, bus_data_width)`
Generate an AXI4-Lite slave wrapper.

### Function `_generate_wishbone(inner_module, params, data_width, addr_width, bus_data_width)`
Generate a Wishbone B4 pipelined slave wrapper.

---

## Module `hdl_gen.ip_xact`

### Function `generate_ip_xact(module_name)`
Generate IP-XACT component XML.

Parameters
----------
module_name : str
    Top-level Verilog module name.
vendor : str
    IP vendor identifier.
library : str
    IP library name.
version : str
    IP version string.
data_width : int
    Neuron data width.
params : dict, optional
    Verilog parameters.
bus : str
    Bus interface type.

Returns
-------
str
    IP-XACT XML string.

### Function `_add_port(parent, name, direction, width)`
Add a port element to the ports section.

Parameters
----------
parent : Element
    Parent ports element.
name : str
    Port name.
direction : str
    ``"in"`` or ``"out"``.
width : int
    Bit width.

---

## Module `hdl_gen.kuramoto_emitter`

### Class `KuramotoEmitter`
Emit a bounded fixed-point Kuramoto phase core for HDL experiments.

The generated RTL is intentionally narrow in scope:

- noiseless only
- all-to-all scalar coupling only
- fixed-point phase state
- LUT-based sine approximation

This is a synthesis exploration scaffold, not a drop-in replacement for
the production Kuramoto solvers.

- **__init__**(module_name)
- **_fixed_int**(value)
- **_signed_literal**(value)
- **_lut_lines**()
- **generate**()

---

## Module `hdl_gen.lfsr16_emitter`

### Class `Lfsr16Emitter`
Emit a synthesisable standalone LFSR-16 Verilog module.

- **__init__**(module_name, seed)
- **generate**()
  - Return the standalone LFSR-16 Verilog module.
- **_advance**(state)

---

## Module `hdl_gen.online_learning_emitter`

### Class `OnlineO1ResourceEstimate`
Deterministic pre-synthesis resource estimate for one online-learning block.

- **as_dict**()
  - Return a deterministic JSON-ready estimate payload.

### Class `OnlineO1LearningEmitter`
Emit a synthesisable reward-modulated STDP state machine.

- **__init__**()
- **generate**()
  - Return Verilog for one bounded online-learning synapse lane.
- **estimate_resources**()
  - Return a conservative pre-synthesis resource estimate.
- **manifest**()
  - Return deterministic metadata for the generated learning lane.

---

## Module `hdl_gen.quasirandom_emitter`

### Class `Halton16Emitter`
Emit a synthesisable standalone Halton-16 (Van der Corput base-2) module.

Architecture: pure counter + bit-reversal wiring.
Zero multipliers, zero LUTs for core logic.

- **__init__**(module_name)
- **generate**()
  - Return the standalone Halton-16 Verilog module.

### Class `QuasiRandomEmitter`
Unified factory for quasi-random SNG emitters.

Parameters
----------
method : str
    ``"sobol"`` or ``"halton"``.
module_name : str, optional
    Override the default module name.
seed : int, optional
    Seed for Sobol (ignored for Halton).

- **__init__**(method, module_name, seed)
- **generate**()
  - Generate the Verilog source for the selected method.
- **module_name**()
  - Return the sanitised module name.

---

## Module `hdl_gen.side_channel_encoding_emitter`

### Class `SideChannelEncodingEmitter`
Emit a synthesisable ROM-style wrapper for one protected encoding record.

- **__init__**()
- **generate**()
  - Return a Verilog module exposing payload and dummy stream bits.
- **manifest**()
  - Return transport metadata linking the HDL hook to analytic evidence.

### Function `_bits_literal(bits)`
---

## Module `hdl_gen.sobol16_emitter`

### Class `Sobol16Emitter`
Emit a synthesisable standalone Sobol-16 Verilog module.

- **__init__**(module_name, seed)
- **generate**()
  - Return the standalone Sobol-16 Verilog module.

---

## Module `hdl_gen.spice_generator`

### Class `SpiceGenerator`
Generates SPICE netlists for Memristive Crossbars.

- **generate_crossbar**(weights, filename)
  - weights: (Rows, Cols) - Conductance values &#91;0, 1&#93; mapped to &#91;G_off, G_on&#93;.

---

## Module `hdl_gen.tmr_wrapper`

### Function `generate_tmr_wrapper(module_name, inputs, outputs)`
Generate a TMR wrapper for a given Verilog module.

Parameters
----------
module_name : str
    Name of the target module to triplicate.
inputs : list of (name, width) tuples
    Input ports of the target module.
outputs : list of (name, width) tuples
    Output ports to protect with majority voting.
voter_module : str
    Name of the voter module (default: ``sc_tmr_voter``).

Returns
-------
str
    Generated Verilog source for the TMR wrapper.

---

## Module `hdl_gen.verilog_generator`

### Class `VerilogGenerator`
Generates Top-Level Verilog for a defined SC Network.

- **__init__**(module_name, bus_width)
  - Initialise with a top-level module name.
- **add_layer**(layer_type, name, params)
  - Add a layer definition to the network.
- **generate**(mode)
  - Emits Verilog code.
- **_validate_sync_layers**()
  - Reject sync RTL configurations that cannot be emitted faithfully.
- **_require_positive_int**(value, name)
  - Return value as int after rejecting booleans and non-positive values.
- **_dense_input_width**(params, previous_width)
- **_dense_output_width**(params)
- **_sync_layer_widths**()
  - Return per-layer ``(input_width, output_width)`` and reject mismatches.
- **emit_lfsr16_source**(module_name, seed)
  - Emit a standalone LFSR-16 stochastic source module.
- **emit_sobol16_source**(module_name, seed)
  - Emit a standalone Sobol-16 stochastic source module.
- **emit_sources_from_ir**(ir)
  - Emit standalone stochastic source modules declared in an IR payload.
- **emit_async_aer**(module_name)
  - Emit the research-stage async AER wrapper.
- **emit_kuramoto_phase**(module_name)
  - Emit the bounded research Kuramoto phase core.
- **emit_halton16_source**(module_name)
  - Emit a standalone Halton-16 stochastic source module.
- **emit_quasirandom_source**(method, module_name, seed)
  - Emit a quasi-random source via the unified factory.
- **emit_decorrelator**()
  - Return the path to the sc_decorrelator HDL module.
- **emit_edt_controller**()
  - Return an instantiation template for the EDT controller.
- **emit_tmr_wrapper**(module_name, inputs, outputs)
  - Generate a TMR wrapper for the given module.
- **save_to_file**(path)
  - Write generated Verilog to a file.

### Function `emit_sources_from_ir(ir)`
Emit LFSR-16 and Sobol-16 source modules from a lightweight IR payload.

The helper accepts the mapping shapes already used by documentation,
tests, and compiler-service payloads: ``{"nodes": &#91;...&#93;}``,
``{"nodes": {"node_id": {...}}}``, or a direct iterable of node mappings.
Non-source nodes are ignored. Source nodes must identify their generator
through ``source_type``, ``decorrelator``, ``generator``, ``strategy``, or
the node ``type``/``node_type`` itself.

### Function `_iter_ir_nodes(ir)`
Iterate IR nodes, yielding (node_id, node) pairs.

### Function `_source_kind(node)`
Determine the stochastic source kind from an IR node.

### Function `_source_module_name(node)`
Extract or generate a unique module name for a stochastic source.

### Function `_source_seed(node)`
Extract the PRNG seed from an IR node.

### Function `_node_params(node)`
Extract the params/attributes sub-mapping from an IR node.

### Function `_node_value(node)`
Look up the first matching key in a node mapping or object.

### Function `_normalise(value)`
Normalise a node type string to lowercase with underscores.

---

## Module `homeostasis.regulator`

### Class `StabilityMetrics`
Network stability measurements.

- **summary**()

### Class `NetworkRegulator`
Network-wide homeostatic regulator.

Monitors population firing rates and adjusts thresholds, learning rates,
and weights to maintain target activity levels.

Parameters
----------
target_rate : float
    Target mean firing rate (spikes per step).
rate_tolerance : float
    Acceptable deviation from target (fraction).
threshold_step : float
    Per-step threshold adjustment magnitude.
lr_scale_factor : float
    Multiplicative LR adjustment factor.

- **__init__**(target_rate, rate_tolerance, threshold_step, lr_scale_factor)
- **regulate**(firing_rates, thresholds, learning_rate, weights)
  - Apply homeostatic regulation.

### Class `SleepConsolidation`
Sleep-phase synaptic renormalization for memory consolidation.

During sleep: suppress external input, apply power-law weight decay,
allow spontaneous replay through recurrent dynamics.

Reference: Sleep-Based Homeostatic Regularization (arXiv Jan 2026)

Parameters
----------
decay_exponent : float
    Power-law exponent for weight decay (higher = more aggressive).
noise_amplitude : float
    Spontaneous activity noise during sleep.
duration_fraction : float
    Sleep duration as fraction of epoch (0.1 = 10% of time sleeping).

- **__init__**(decay_exponent, noise_amplitude, duration_fraction)
- **apply**(weights, seed)
  - Apply sleep consolidation to weights.
- **should_sleep**(epoch, total_epochs)
  - Determine if this epoch should include a sleep phase.

---

## Module `hub.bundle`

### Class `HubBundleConfig`
Configuration for a local self-hosted hub bundle.

- **__post_init__**()

### Function `build_model_zoo_index()`
Build a deterministic model-zoo index for hub manifests.

### Function `build_benchmark_plan(config)`
Build the benchmark-runner plan included in the hub bundle.

### Function `build_hub_manifest(config)`
Build a deterministic manifest for a self-hosted hub bundle.

### Function `write_hub_bundle(output_dir, config)`
Write a local Docker Compose hub bundle and return generated paths.

### Function `_compose_yaml(config, repo_context)`
### Function `_env_example(config)`
### Function `_readme(config)`
### Function `_offline_environment(config)`
### Function `_studio_container_command(config)`
### Function `_ingress_scope(bind_host)`
### Function `_json(payload)`
### Function `_relative_repo_context(output_dir)`
### Function `_repo_root()`
### Function `_validate_relative_path(label, value)`
### Function `_validate_bind_host(value)`
---

## Module `hypervisor.hypervisor`

### Class `RegionState`

### Class `HWRegion`
One isolated hardware region on the fabric.

- **axi_end_addr**()
- **is_free**()
- **contains_addr**(addr)

### Class `TenantPriority`

### Class `QoSPolicy`
Quality-of-Service policy for a tenant.


### Class `TenantState`
Checkpointable state for live migration.

- **compute_checksum**()

### Class `Tenant`
One SC network tenant on the hypervisor.


### Class `FirewallRule`
One address-range access rule.

- **end_addr**()

### Class `BitstreamFirewall`
AXI address-range isolation preventing cross-tenant access.

Each tenant can only access its own region's AXI address space.
Any cross-region access is blocked and logged as a violation.

- **__init__**()
- **add_rule**(rule)
- **remove_tenant_rules**(tenant_id)
- **check_access**(tenant_id, addr, is_write)
  - Check if a tenant can access an address.
- **_log_violation**(tenant_id, addr, reason)
- **violation_count**()
- **clear_violations**()

### Class `SchedulingPolicy`

### Class `ScheduleSlot`
One time slot in the schedule.

- **end_cycle**()

### Class `Scheduler`
Multi-tenant temporal scheduler with preemption.

Supports priority-based, round-robin, fair-share, and EDF scheduling.

- **__init__**(policy)
- **generate_schedule**(tenants, num_cycles)
  - Generate a schedule for the given tenants.
- **_round_robin**(tenants, total)
- **_priority**(tenants, total)
- **_fair_share**(tenants, total)
- **_edf**(tenants, total)

### Class `MigrationRequest`
Request to migrate a tenant between regions.


### Class `MigrationResult`
Result of a migration attempt.


### Class `MigrationEngine`
Live migration of tenants between hardware regions.

Migration steps:
1. Pause tenant on source region
2. Checkpoint state (voltages, weights, LFSR, spike queues)
3. Verify checkpoint integrity (SHA-256)
4. Restore state on target region
5. Update firewall rules
6. Resume tenant

- **__init__**()
- **checkpoint**(tenant)
  - Checkpoint tenant state for migration.
- **restore**(tenant, state)
  - Restore checkpointed state to a tenant.
- **migrate**(tenant, source, target, firewall)
  - Execute live migration.

### Class `HypervisorConfig`
Hypervisor configuration.


### Class `Hypervisor`
Multi-tenant neuromorphic hypervisor.

Manages tenant lifecycle, hardware allocation, scheduling,
firewall enforcement, and live migration.

- **__init__**(config)
- **add_region**(region)
  - Register a hardware region.
- **register_tenant**(tenant)
  - Register a new tenant.
- **allocate**(tenant_id)
  - Allocate a free region to a tenant.
- **deallocate**(tenant_id)
  - Release a tenant's hardware region.
- **remove_tenant**(tenant_id)
  - Remove a tenant entirely.
- **schedule**(num_cycles)
  - Generate a schedule for active tenants.
- **migrate**(tenant_id, target_region_id)
  - Migrate a tenant to a different region.
- **check_access**(tenant_id, addr, is_write)
  - Check if a tenant can access an address (firewall).
- **status**()
  - Get hypervisor status.
- **tenant_report**(tenant_id)
  - Get a report for one tenant.
- **compute_utilisation**()
  - Compute utilisation fraction per region.
- **check_overcommit**()
  - Check if total tenant QoS exceeds fabric capacity.
- **get_faulted_regions**()
  - List regions in FAULTED state.
- **mark_region_faulted**(region_id)
  - Mark a region as faulted and evict its tenant.

### Class `BandwidthMeter`
Per-tenant throughput metering (spikes/cycles per window).

- **record**(tenant_id, spike_count, cycle)
- **throughput**(tenant_id)
  - Spikes per cycle (averaged over window).
- **exceeds_quota**(tenant_id, max_mbps)

### Class `PreemptionEvent`
Record of a preemption event.


### Class `PreemptionManager`
Handles preemption with state checkpoint/restore.

- **__init__**()
- **preempt**(victim, preemptor, region, cycle)
  - Preempt victim and give region to preemptor.
- **restore_preempted**(tenant)
  - Restore a previously preempted tenant's state.

### Class `SLAViolation`
One SLA violation.


### Class `SLAMonitor`
Monitors per-tenant QoS compliance and detects violations.

- **__init__**()
- **check_latency**(tenant, measured_us, cycle)
- **check_bandwidth**(tenant, measured_mbps, cycle)
- **total_violations**()
- **violations_for**(tenant_id)

### Class `UsageRecord`
One billing record.


### Class `ResourceAccounting`
Tracks per-tenant resource usage for metered billing.

- **__init__**()
- **record**(tenant_id, cycles, spikes)
- **total_cycles**(tenant_id)
- **total_spikes**(tenant_id)
- **invoice**(tenant_id, cost_per_cycle)
  - Compute billing amount.

### Class `RegionHealth`
Health score with degradation model.

- **health_score**()
  - 0.0 = dead, 1.0 = perfect.
- **is_degraded**()
- **record_error**()

### Class `AuditEventType`

### Class `AuditEntry`
- **__post_init__**()

### Class `SecurityAuditLog`
Structured, append-only audit trail for compliance.

- **__init__**(max_entries)
- **log**(event)
- **query**(event_type, tenant_id)
- **count**()
- **checksum**()

### Class `MigrationThrottle`
Rate-limits migration requests to prevent storms.

- **allow**()
  - Check if a migration is allowed under the rate limit.
- **record**()
- **recent_count**()

### Function `select_region_multi_die(regions, min_neurons, preferred_die)`
Select best free region, preferring a specific die.

### Function `admission_check(tenant, regions, existing_tenants)`
Check if a new tenant can be admitted without overcommitting.

### Function `verify_isolation(firewall, regions)`
Verify that no two tenants share address ranges.

Returns list of violation descriptions (empty = sound).

---

## Module `identity.checkpoint`

### Class `Checkpoint`
Save and restore complete IdentitySubstrate state.

- **save**(substrate, path)
  - Save complete state to .npz file.
- **load**(path)
  - Restore substrate from checkpoint.
- **merge**(paths)
  - Merge multiple checkpoints by averaging weights and concatenating history.

### Function `_restore_voltages(population, voltages)`
Write voltage array back into individual neuron objects.

---

## Module `identity.decoder`

### Class `StateDecoder`
Extract cognitive state from spiking network for session priming.

- **__init__**(substrate)
- **_recent_trains**(n_neurons, window)
  - Get recent binary spike trains for the first n_neurons.
- **extract_dominant_patterns**(n_components)
  - PCA on recent spike trains -> dominant activity patterns.
- **extract_attractor_states**(threshold)
  - Find stable attractor states via correlation clustering.
- **extract_connectivity_signature**()
  - Functional connectivity matrix summarizing learned structure.
- **generate_priming_context**()
  - Generate a text summary of current network state.

---

## Module `identity.director`

### Class `DirectorController`
L16 self-monitoring and self-regulation for the identity substrate.

- **__init__**(substrate)
- **monitor**()
  - Measure current dynamics from recent spike history.
- **diagnose**()
  - Identify problems in network dynamics.
- **correct**()
  - Apply corrective actions based on diagnosis.
- **report**()
  - Generate human-readable health report.

### Function `_add_weight_noise(data, scale)`
Add Gaussian noise to nonzero weights, clip to non-negative.

### Function `_homeostatic_scale(data, factor)`
Scale all weights toward the mean. Turrigiano 2008.

### Function `_prune_weak(data, threshold)`
Zero out weights below threshold.

### Function `_grow_synapses(data, fraction, seed)`
Reinitialize a fraction of zero weights to small positive values.

---

## Module `identity.encoder`

### Class `TraceEncoder`
Encode reasoning traces as temporal spike patterns.

LSH maps text chunks to neuron groups. Poisson spike trains
weighted by chunk salience deliver the encoded signal.

- **__init__**(n_neurons, hash_dims, seed)
- **_hash_to_neurons**(text)
  - Map text to an activation vector over neurons via LSH.
- **encode**(text, duration_ms, dt)
  - Convert text to spike pattern array (n_neurons, n_steps).
- **encode_key_value**(key, value)
  - Encode a key-value pair as a combined spike pattern.

### Function `_tokenize(text)`
Split text into sentence-like chunks on punctuation boundaries.

### Function `_word_set(chunk)`
Extract lowercased alphanumeric tokens.

### Function `_salience(chunk, position, total)`
Salience score based on position and lexical density.

First and last sentences score higher (inverted-U weighting).
Longer chunks with more unique words score higher.

---

## Module `identity.substrate`

### Class `IdentitySubstrate`
Persistent spiking neural network for identity continuity.

Three populations:
- cortical: HodgkinHuxley excitatory (main processing)
- inhibitory: WangBuzsaki fast-spiking (balance/stability)
- memory: HindmarshRose bursting (pattern storage via attractors)

Connectivity: small-world E->E with STDP, random E->I, I->E, E->M, M->E.

- **__init__**(n_cortical, n_inhibitory, n_memory, seed)
- **_build_projections**(seed)
- **_build_monitors**()
- **_build_network**()
- **step**(stimuli, dt)
  - Advance one timestep. Inject external current into cortical neurons.
- **run**(duration, dt, stimuli_sequence)
  - Run for *duration* seconds. Optional time-varying stimuli array.
- **inject_experience**(reasoning_trace)
  - Encode a reasoning trace as spike patterns and inject via run().
- **extract_state**()
  - Extract current network state for session priming.
- **health_check**()
  - L16 Director: check network dynamics are healthy.
- **spike_history**()
- **ee_weights**()

---

## Module `industrial_applications`

### Class `IndustrialDomain`
Supported industrial application domains.


### Class `EvidenceCategory`
Evidence categories expected in an industrial readiness pack.


### Class `EvidenceRequirement`
One evidence requirement for an application profile.

- **to_dict**()
  - Return a JSON-ready requirement.

### Class `IndustrialApplicationProfile`
Readiness profile for one SC-NeuroCore industrial use case.

- **to_dict**()
  - Return a JSON-ready profile.

### Class `IndustrialReadinessAssessment`
Evidence coverage assessment for one industrial application profile.

- **ready**()
  - Whether all mandatory evidence categories are present.
- **mandatory_coverage**()
  - Mandatory evidence coverage ratio.
- **to_dict**()
  - Return a JSON-ready assessment.

### Class `IndustrialApplicationRegistry`
Registry of application profiles and evidence-readiness checks.

- **__init__**(profiles)
- **get**(domain)
  - Return the profile for a domain.
- **list_profiles**()
  - Return all registered profiles in deterministic order.
- **assess**(domain, evidence_bag)
  - Assess whether the evidence bag covers a domain profile.

### Function `default_industrial_profiles()`
Return conservative built-in industrial application profiles.

### Function `assess_industrial_readiness(domain, evidence_bag)`
Assess readiness for a built-in industrial application domain.

### Function `_requirements()`
### Function `_normalise_evidence_categories(categories)`
---

## Module `integrations.lava_bridge`

### Class `LoihiNetworkConfig`
Configuration for a deployed Loihi network.


### Class `SCtoLavaConverter`
Convert SC-NeuroCore layer stack to Lava Process network.

- **__init__**(weight_bits)
- **convert_dense_layer**(sc_layer)
  - Convert an SCDenseLayer or VectorizedSCLayer to Loihi config.
- **convert_training_model**(spiking_net)
  - Convert a trained SpikingNet to a list of LoihiNetworkConfigs.

### Function `export_weights_loihi(weights, weight_bits, weight_exp)`
Convert SC probability weights &#91;0,1&#93; to Loihi fixed-point format.

Loihi uses signed integer weights with configurable precision.
Maps &#91;0,1&#93; → &#91;-128, 127&#93; for 8-bit weights.

### Function `loihi_threshold_from_sc(sc_threshold, weight_bits)`
Convert SC normalised threshold to Loihi integer threshold.

---

## Module `interfaces.bci`

### Class `BCIEncoder`
Encode continuous neural signals into spike trains.

Replaces the old BCIDecoder (misleading name — it encodes, not decodes).
Uses seeded RNG for deterministic, reproducible encoding.

Parameters
----------
n_channels : int
    Number of recording channels.
sampling_rate : int
    Input signal sampling rate (Hz).
window_ms : float
    Encoding window duration in milliseconds.
seed : int
    RNG seed for reproducibility.

- **encode**(signal, T)
  - Encode a signal block into spike trains via rate coding.
- **encode_stream**(signal)
  - Encode a multi-window signal stream.
- **_normalize**(values)
  - Normalize to &#91;0, 1&#93; for probability encoding.
- **normalize_signal**(signal)
  - Normalize signal to &#91;0, 1&#93;. Legacy API — use _normalize().
- **encode_to_bitstream**(signal, length)
  - Legacy API. Encodes (channels, time) → (channels, length).

### Class `BCIDecoder`
Legacy alias. Use BCIEncoder instead.

- **__init__**(channels, sampling_rate)

---

## Module `interfaces.bci_closed_loop`

### Class `ClosedLoopBCIConfig`
Configuration for one raw-waveform to feedback HIL loop.


### Class `FeedbackFrame`
Feedback vector emitted to an implant emulator or hardware adapter.


### Class `ClosedLoopBCIResult`
One processed BCI/HIL loop window.


### Class `SpikeDecoder`
Decoder interface for closed-loop spike windows.

- **decode**(spike_raster)
  - Decode a binary spike raster into feedback control values.

### Class `FeedbackSink`
Feedback interface for an implant emulator or hardware adapter.

- **apply_feedback**(values, timestamp_us)
  - Apply decoded feedback values and return the emitted frame.

### Class `RateSpikeDecoder`
Decode spike rasters as per-channel firing rates.

- **decode**(spike_raster)

### Class `ImplantEmulator`
Deterministic feedback sink used by the closed-loop template.

- **apply_feedback**(values, timestamp_us)

### Class `ClosedLoopBCITemplate`
WaveformCodec + AER + telemetry closed-loop BCI scaffold.

- **__post_init__**()
- **process_window**(waveform)
  - Process one raw electrode window through the closed-loop template.
- **_validate_waveform**(waveform)
- **_detect_spike_raster**(waveform)

---

## Module `interfaces.bci_hil_manifest`

### Class `BCIHILBoardProfile`
Board/input profile for a closed-loop BCI reference pipeline.

- **to_dict**()
  - Return a deterministic manifest dictionary.

### Function `available_bci_hil_profiles()`
Return all reference profiles in deterministic order.

### Function `get_bci_hil_profile(profile_id)`
Return one reference profile by identifier.

### Function `build_bci_hil_reference_manifest(profile_id)`
Build a deterministic closed-loop BCI/HIL reference manifest.

### Function `create_bci_hil_template(profile_id)`
Create a `ClosedLoopBCITemplate` from a reference profile.

### Function `_profile_to_config(profile)`
---

## Module `interfaces.ccw_bridge`

### Class `CCWMode`
CCW modulation modes aligned with VIBRANA.


### Class `CCWParameters`
Parameters for CCW audio generation.


### Class `VIBRANAState`
State for VIBRANA visualization sync.


### Class `CCWBridge`
Bridge between SC-NeuroCore and CCW/VIBRANA systems.

Converts bitstream outputs from SCPN layers into audio parameters
and visualization states for the CCW application.

- **__init__**(params)
- **bitstream_to_frequency**(bitstream, freq_min, freq_max)
  - Convert a bitstream to a frequency value.
- **scpn_metrics_to_ccw**(metrics)
  - Convert SCPN global metrics to CCW audio parameters.
- **glyph_vector_to_vibrana**(glyph_vector)
  - Convert L7 glyph vector to VIBRANA visualization parameters.
- **generate_binaural_sample**(ccw_params, duration_samples)
  - Generate binaural audio samples from CCW parameters.
- **generate_ccw_metadata**(scpn_outputs, glyph_vector)
  - Generate complete CCW metadata package for audio/visual sync.
- **export_glyph_stream**(glyph_vector, cosmic_vector, filepath)
  - Export glyph stream data for VIBRANA/CCW hardware playback.
- **create_session_config**(mode, duration_minutes)
  - Create a complete CCW session configuration.

### Function `create_bridge(ccw_params)`
Factory function to create a CCW bridge instance.

---

## Module `interfaces.dvs_input`

### Class `DVSInputLayer`
Interface for Dynamic Vision Sensors (Event Cameras).
Converts AER events (x, y, t, p) into SC Bitstreams.

- **__post_init__**()
- **process_events**(events)
  - Integrate a batch of events.
- **_validate_events**(events)
- **generate_bitstream_frame**(length)
  - Generate a HxWxLength bitstream cube from current surface state.

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

## Module `ir.scnir_compatibility`

### Class `SCNIRCompatibilityRow`
One compatibility row for a NIR primitive.

- **as_dict**()
  - Return a deterministic JSON-ready row.

### Function `scnir_compatibility_matrix()`
Return the deterministic SC-NIR compatibility matrix.

### Function `scnir_compatibility_matrix_dicts()`
Return the matrix as deterministic JSON-ready dictionaries.

### Function `build_scnir_compatibility_audit(evidence_root)`
Build a versioned closure-audit report for the SC-NIR compatibility matrix.

The report is intentionally derived from the executable matrix after
validation, so release automation consumes the same data that enforces
parser coverage and evidence-path freshness.

### Function `validate_scnir_compatibility_matrix(evidence_root)`
Fail if the matrix drifts from parser-declared support or stale evidence paths.

Parameters
----------
evidence_root:
    Optional repository root used to verify that every ``audit_evidence``
    path in the matrix resolves to an existing file.

---

## Module `ir.scnir_convert`

### Class `SCNIRConversionConfig`
Configuration for deterministic SC-NIR metadata export.

- **__post_init__**()
- **resolved_accumulator_bits**()
  - Accumulator width used by exported precision metadata.

### Function `build_scnir_from_neuron_graph(neuron_graph)`
Build an SC-NIR document from an existing NIR-derived NeuronGraph.

### Function `_hierarchy_from_graph(neuron_graph, streams)`
Materialise SC-NIR hierarchy metadata from inlined NeuronGraph provenance.

### Function `_hierarchy_stream_ids(neuron_graph, instance)`
Return deterministic stream ids generated by one inlined hierarchy instance.

### Function `_hierarchy_port_prefix(signal_kind)`
### Function `_connection_transforms(conn)`
Return explicit SC-NIR transform metadata for a weighted connection.

### Function `_online_learning_annotation(config, stream_id)`
### Function `_connection_delay_steps(conn)`
### Function `export_scnir_from_nir(model_path)`
Read a NIR model, export SC-NIR metadata, and write it to JSON.

### Function `_precision(config)`
### Function `_source(config, stream_index)`
### Function `_population_encoding(neuron_type)`
### Function `_population_signal_kind(neuron_type)`
### Function `_population_stream_id(name)`
### Function `_connection_stream_id(src, dst)`
### Function `_stream_fragment(value)`
### Function `_nth_prime(index)`
---

## Module `ir.scnir_handoff_audit`

### Class `SCNIRHDLHandoffAuditError`
Raised when a compile-nir HDL handoff directory is incomplete or inconsistent.


### Class `SCNIRHDLHandoffAuditReport`
Deterministic summary of a validated SC-NIR HDL handoff directory.

- **as_dict**()
  - Return a stable JSON-ready report.

### Function `audit_scnir_hdl_handoff(directory)`
Validate a ``compile-nir`` HDL output directory and return an audit report.

The audit is intentionally structural and fail-closed: every SC-NIR stream
must have exactly one matching source-manifest row and emitted source
module, aggregate counts must match the typed document, and top-level
SC-NIR localparams must agree with the JSON handoff metadata.

### Function `write_scnir_hdl_handoff_audit(directory, output_path)`
Validate a handoff directory and write the JSON audit report.

### Function `_load_manifest(path)`
### Function `_verify_manifest_header(manifest)`
### Function `_external_inputs(manifest)`
### Function `_verify_source_rows(root, document, sources)`
### Function `_verify_hierarchy_modules(root, document)`
### Function `_verify_hierarchy_top_instances(verilog, document)`
### Function `_verify_source_row_keys(row, index)`
### Function `_verify_source_row_matches_stream(row, stream, index)`
### Function `_stream_transform_rows(stream)`
### Function `_delay_steps_for_row(delay_steps)`
### Function `_signal_kind_counts(document)`
### Function `_signal_routes(document)`
### Function `_hierarchy_instances(document)`
### Function `_expect_top_localparam(verilog, name, value)`
### Function `_expect_equal(actual, expected, field)`
### Function `_require_file(path, label)`
### Function `_expect_mapping_sequence(manifest, key)`
### Function `_expect_non_empty_string(mapping, key)`
### Function `_expect_int(mapping, key)`
### Function `_expect_positive_int(mapping, key)`
### Function `_expect_non_negative_int(mapping, key)`
---

## Module `ir.scnir_hdl`

### Class `SCNIRHDLSourceManifestEntry`
Serialisable manifest row for one emitted stochastic source module.

- **as_dict**()
  - Return a deterministic JSON-ready representation.

### Class `SCNIRHDLSourceBundle`
Concrete HDL source modules plus the manifest that explains them.

- **manifest_dicts**()
  - Return deterministic JSON-ready manifest rows.

### Function `build_scnir_source_bundle(document)`
Emit deterministic HDL source modules for every SC-NIR stream.

Only source kinds with the standard threshold-bit output contract are
materialised here. Unsupported SC-NIR source kinds fail closed instead of
being lowered to semantically incompatible RTL.

### Function `_emit_stream_source(stream)`
### Function `_manifest_entry(stream)`
### Function `_require_seed(stream)`
### Function `_module_name_for_stream(stream, index)`
---

## Module `ir.scnir_schema`

### Class `SCNIRValidationError`
Raised when an SC-NIR payload violates the fail-closed contract.


### Class `SCNIRPrecision`
Fixed-point interpretation attached to one stochastic stream.


### Class `SCNIRSource`
Random or deterministic source metadata for a stochastic stream.


### Class `SCNIRCorrelationConstraint`
Correlation rule between two stochastic streams.


### Class `SCNIRStreamTransform`
Deterministic transform applied before a logical stochastic stream.


### Class `SCNIRStream`
SC metadata for one logical stochastic bitstream.


### Class `SCNIRHierarchyPort`
One typed port on a hierarchical SC-NIR hardware instance.


### Class `SCNIRHierarchyInstance`
One hierarchy instance boundary for future nested hardware handoff.


### Class `SCNIRDocument`
Top-level SC-NIR metadata document.


### Function `validate_scnir_dict(payload)`
Validate a decoded SC-NIR payload or raise ``SCNIRValidationError``.

### Function `scnir_from_dict(payload)`
Build a typed SC-NIR document from a decoded mapping.

### Function `scnir_to_dict(document)`
Convert a typed SC-NIR document to deterministic JSON-ready data.

### Function `load_scnir(path)`
Load and validate an SC-NIR JSON document.

### Function `write_scnir(path, document)`
Write an SC-NIR JSON document after validating it.

### Function `upgrade_scnir_dict(payload)`
Upgrade supported SC-NIR payloads to the current canonical schema.

Version ``v0.1`` did not encode recurrent connection delay, and versions
before ``v0.3`` did not distinguish spiking, analogue-state, and weight
streams.  Version ``v0.4`` added explicit stream transform metadata for
threshold comparators.  Version ``v0.5`` permits ``delay_steps`` to be
either a scalar integer or a per-source-column integer vector.  Version
``v0.6`` adds top-level hierarchy instance and port metadata.  Version
``v0.7`` adds optional validated per-weight-stream online-learning
annotations.  Legacy upgrades insert the missing fields before validating
through the typed schema.  Current documents are canonicalised through the
same deterministic writer.

### Function `_validate_stream(stream, path)`
### Function `_validate_online_learning_annotation(annotation, path)`
### Function `_validate_transform(transform, parent_path)`
### Function `_validate_precision(precision, parent_path)`
### Function `_validate_source(source, parent_path)`
### Function `_validate_correlation(constraint, path)`
### Function `_validate_hierarchy(hierarchy, stream_signal_kinds)`
### Function `_stream_from_dict(stream)`
### Function `_transform_from_dict(transform)`
### Function `_correlation_from_dict(constraint)`
### Function `_hierarchy_instance_from_dict(instance)`
### Function `_hierarchy_port_from_dict(port)`
### Function `_stream_to_dict(stream)`
### Function `_hierarchy_instance_to_dict(instance)`
### Function `_hierarchy_port_to_dict(port)`
### Function `_infer_legacy_signal_kind(stream_id)`
### Function `_expect_keys(payload, allowed, path)`
### Function `_expect_mapping(value, path)`
### Function `_expect_sequence(value, path)`
### Function `_expect_non_empty_string(value, path)`
### Function `_expect_stream_id(value, path)`
### Function `_expect_hdl_identifier(value, path)`
### Function `_expect_positive_int(value, path)`
### Function `_expect_non_negative_int(value, path)`
### Function `_expect_delay_steps(value, path)`
Validate scalar or per-source-column stream delay metadata.

### Function `_delay_steps_from_value(value, path)`
### Function `_delay_steps_to_json(value)`
### Function `_expect_enum(value, allowed, path)`
### Function `_is_prime(value)`
---

## Module `layers.attention`

### Class `StochasticAttention`
Stochastic Computing Attention Block.

Two modes:

- ``forward()`` — row-sum normalised (SC-native, no exp). Matches Rust engine ``forward()``.
- ``forward_softmax()`` — proper softmax with temperature scaling.

Example
-------
>>> Q = np.random.default_rng(0).uniform(0, 1, (4, 8))
>>> K = np.random.default_rng(1).uniform(0, 1, (6, 8))
>>> V = np.random.default_rng(2).uniform(0, 1, (6, 5))
>>> attn = StochasticAttention(dim_k=8)
>>> attn.forward(Q, K, V).shape
(4, 5)
>>> attn.forward_softmax(Q, K, V).shape
(4, 5)

- **__post_init__**()
- **_ensure_2d**(Q, K, V)
- **_validate_unipolar_bitstream_inputs**(Q, K, V, length)
- **forward**(Q, K, V)
  - Row-sum normalised attention (SC-native, no exp).
- **forward_softmax**(Q, K, V)
  - Proper softmax attention with temperature scaling.
- **forward_bitstream**(Q, K, V, length, use_sobol)
  - SC-native attention via bitstream AND gates.

---

## Module `layers.circuit_primitives`

### Class `LateralInhibition`
Lateral inhibition: each neuron inhibits its neighbors.

Models the surround suppression found in retinal ganglion cells,
cortical simple cells, and throughout sensory processing.

The inhibition kernel is a Gaussian centered on each neuron with
width `radius`, producing a Mexican-hat (center-surround) response
when combined with the neuron's own excitation.

- **__post_init__**()
- **apply**(rates)
  - Apply lateral inhibition to firing rates.

### Class `WinnerTakeAll`
k-Winner-Take-All circuit.

Only the top-k neurons remain active; all others are suppressed to zero.
Models competitive dynamics in cortical columns and basal ganglia
action selection.

With k=1, this is a hard argmax over the population.

- **apply**(rates)
  - Apply k-WTA to firing rates.
- **winners**(rates)
  - Return indices of the k winning neurons.

---

## Module `layers.fusion`

### Class `SCFusionLayer`
Fuses multiple data modalities using stochastic multiplexing (MUX).

Example
-------
>>> import numpy as np
>>> layer = SCFusionLayer(
...     input_dims={"audio": 4, "visual": 4},
...     fusion_weights={"audio": 0.7, "visual": 0.3},
... )
>>> out = layer.forward({"audio": np.ones(4), "visual": np.zeros(4)})
>>> out.shape
(4,)

- **__post_init__**()
- **forward**(inputs)
  - inputs: {'modality': np.array(&#91;values&#93;)}

---

## Module `layers.hardware_aware`

### Class `HardwareAwareSCLayer`
SC layer with memristive hardware defect injection.

Parameters
----------
n_inputs : int
    Number of input channels.
n_neurons : int
    Number of output neurons.
length : int
    Bitstream length.
stuck_rate : float
    Fraction of synapses with stuck-at faults (0 or 1). Default 0.05.
variability : float
    Additive weight noise std. Default 0.02.
seed : int
    Random seed for defect generation.

- **__post_init__**()
- **_apply_defects**()
- **forward**(input_values)
- **update_weights**(gradient, lr)
  - Update weights with gradient, respecting stuck-at mask.
- **weights**()
- **n_stuck**()
- **stuck_fraction**()

---

## Module `layers.jax_dense_layer`

### Class `JaxSCDenseLayer`
JAX-accelerated stochastic dense layer of LIF neurons.

Example
-------
>>> layer = JaxSCDenseLayer(n_neurons=10, n_inputs=5, seed=0)  # doctest: +SKIP
>>> import jax.numpy as jnp  # doctest: +SKIP
>>> spikes = layer.step(jnp.ones(10) * 0.5)  # doctest: +SKIP
>>> spikes.shape  # doctest: +SKIP
(10,)

- **__post_init__**()
- **_positive_int**(name, value)
- **_finite_param**(name, value)
- **_positive_param**(cls, name, value)
- **_nonnegative_param**(cls, name, value)
- **_validate_config**()
- **_initialise_weights**()
- **_validate_current_vector**(currents)
- **_current_from_step_input**(values)
- **_validate_current_sequence**(currents)
- **step**(I_t)
  - Advance the entire layer by one time step.
- **run**(currents)
  - Run for multiple steps.
- **reset**()

---

## Module `layers.memristive`

### Class `MemristiveDenseLayer`
Dense layer mapped to a memristor crossbar with hardware non-idealities.

Defect parameters from Prezioso et al., Nature 521:61-64, 2015.

- **__post_init__**()
- **apply_hardware_defects**()
  - Corrupt weights based on physical properties.

---

## Module `layers.predictive_coding`

### Class `PredictiveCodingSCLayer`
Zero-multiplication predictive coding in SC.

Parameters
----------
n_inputs : int
    Number of input channels.
n_neurons : int
    Number of predictive neurons.
length : int
    Bitstream length.
lr : float
    STDP-like learning rate for prediction weights.
seed : int or None
    Random seed.

- **__post_init__**()
- **forward**(inputs)
  - Process one timestep.
- **reset**()

---

## Module `layers.rall_dendrite`

### Class `RallDendrite`
Dendritic tree with Rall branching and compartmental dynamics.

Parameters
----------
n_branches : int
    Number of dendritic branches.
branch_length : int
    Number of compartments per branch.
tau : float
    Membrane time constant (ms).
coupling : float
    Inter-compartment coupling strength (0 to 1).
dt : float
    Timestep (ms).

- **__post_init__**()
- **step**(branch_inputs)
  - Advance one timestep.
- **branch_voltages**()
  - Current compartment voltages, shape (n_branches, branch_length).
- **reset**()

---

## Module `layers.recurrent`

### Class `SCRecurrentLayer`
SC recurrent / reservoir layer (echo state network).

Spectral radius bound follows Jaeger, GMD Report 148, 2001.

Example
-------
>>> import numpy as np
>>> res = SCRecurrentLayer(n_inputs=3, n_neurons=10, seed=0)
>>> state = res.step(np.array(&#91;0.5, 0.3, 0.8&#93;))
>>> state.shape
(10,)

- **__post_init__**()
- **step**(input_vector)
  - Process one time step (e.g., one frame of audio).
- **reset**()

---

## Module `layers.sc_conv_layer`

### Class `SCConv2DLayer`
SC 2D convolutional layer using stochastic probability encodings.

``sc_mode="unipolar"`` accepts probabilities in ``&#91;0, 1&#93;`` and uses
probability-product accumulation. ``sc_mode="bipolar"`` accepts signed
values in ``&#91;-1, 1&#93;`` and uses signed XNOR-equivalent products.

Example
-------
>>> import numpy as np
>>> conv = SCConv2DLayer(in_channels=1, out_channels=2, kernel_size=3, padding=1)
>>> img = np.random.rand(1, 8, 8)
>>> out = conv.forward(img)
>>> out.shape
(2, 8, 8)

- **__post_init__**()
- **_validate_configuration**()
- **_validate_input**(input_image)
- **_output_shape**(height, width)
- **forward**(input_image)
  - input_image: (in_channels, H, W)

---

## Module `layers.sc_dense_layer`

### Class `SCDenseLayer`
Stochastic-computing dense layer of LIF neurons.

Each neuron receives SC dot-product input current and produces independent
spike trains. ``weight_values`` may be either a single shared vector of
length ``n_inputs`` or a dense matrix shaped ``(n_neurons, n_inputs)``.
Software-only but fully SC-driven at the input/synapse level.

Example
-------
>>> layer = SCDenseLayer(
...     n_neurons=4, x_inputs=&#91;0.5, 0.3&#93;, weight_values=&#91;0.8, 0.6&#93;,
...     x_min=0.0, x_max=1.0, w_min=0.0, w_max=1.0, length=256,
... )
>>> layer.run(T=100)
>>> trains = layer.get_spike_trains()
>>> trains.shape
(4, 100)

- **__post_init__**()
- **_normalise_weight_values**()
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
SC dense layer with integrated STDP learning.

Each neuron has per-input STDP synapses. Plasticity follows
Bi & Poo 1998 asymmetry convention.

- **__post_init__**()
- **run_epoch**(input_values)
  - Run one bitstream epoch (length 'length').
- **get_weights**()

---

## Module `layers.vectorized_layer`

### Class `VectorizedSCLayer`
High-performance SC layer using packed bitwise operations.

Uses GPU (CuPy) when available, otherwise pure NumPy.
Optional sparse connectivity via ``scipy.sparse``.

Example
-------
>>> import numpy as np
>>> layer = VectorizedSCLayer(n_inputs=8, n_neurons=4, length=512)
>>> out = layer.forward(np.random.rand(8))
>>> out.shape
(4,)
>>> (out >= 0).all() and (out <= 1).all()
True

- **__post_init__**()
- **from_exported_weights**(cls, exported_layer)
  - Build a packed SC inference layer from ``to_sc_weights()`` output.
- **_random**(size)
- **_uniform**(low, high, size)
- **_choice**(n_items, size)
- **_apply_bias**(outputs)
- **_refresh_packed_weights**()
- **_init_sparse**()
- **_pack_sparse_weights**()
  - Pack bitstreams only for non-zero synapses, stored in a flat array.
- **forward**(input_values)
  - Compute output firing rates for the layer.
- **_forward_dense**(in_probs)
- **_forward_sparse**(in_probs)
- **_forward_sparse_gpu**(packed_inputs)
  - CuPy CSR matmul path for sparse connectivity on GPU.

### Function `_get_scipy_sparse()`
### Function `_has_scipy_sparse()`
### Function `_popcount_rows(packed)`
Vectorized Hamming-weight popcount across rows of a uint64 array.

### Function `_bipolar_prob(values)`
### Function `_mask_unused_tail_bits(packed, length)`
### Function `_as_float_array(value, name)`
---

## Module `learning.advanced`

### Class `BPTTLearner`
Backpropagation Through Time for spiking networks.

Uses fast-sigmoid surrogate gradient (Neftci et al. 2019) to handle
the spike non-differentiability.

- **__init__**(network, loss_fn, lr)
- **train_step**(inputs, targets)
  - One BPTT step: forward pass, loss, backward with surrogate gradients.

### Class `TBPTTLearner`
Truncated Backpropagation Through Time for long sequences.

Splits input into chunks of ``k`` timesteps, backpropagating gradients
only within each chunk while carrying forward state (membrane voltage)
across boundaries. Reduces memory from O(T) to O(k).

Williams & Peng 1990.

- **__init__**(network, loss_fn, lr, k)
- **train_step**(inputs, targets)
  - One TBPTT step over the full sequence, chunked into windows of k.

### Class `EligibilityTrace`
E-prop eligibility trace: three-factor learning (pre x post x error).

Bellec et al. 2020.

- **__init__**(tau_e, dt)
- **update**(pre_spike, post_spike, error_signal)
  - Compute weight delta from three-factor rule.

### Class `RewardModulatedLearner`
Reward-modulated STDP (R-STDP).

Maintains per-synapse eligibility traces and applies weight updates
scaled by a global reward signal.

- **__init__**(network, tau_reward)
- **_init_traces**()
- **step**(reward)
  - Apply reward-modulated weight update.

### Class `MetaLearner`
MAML-style meta-learning for spiking networks.

Finn et al. 2017. Inner loop: fast adaptation on a task.
Outer loop: meta-gradient across tasks.

- **__init__**(network, inner_lr, outer_lr)
- **_snapshot_weights**()
- **_restore_weights**(snapshot)
- **inner_loop**(task_data, n_steps)
  - Fast adaptation: n_steps of gradient descent on task_data.
- **outer_step**(tasks)
  - Meta-gradient update across multiple tasks.

### Class `HomeostaticPlasticity`
Homeostatic synaptic scaling to maintain target firing rate.

Turrigiano 2008. Multiplicatively scales all incoming weights to keep
the population mean rate near target_rate.

- **__init__**(target_rate, tau)
- **update**(population)
  - Scale weights of all incoming projections to *population*.

### Class `ShortTermPlasticity`
Tsodyks-Markram short-term plasticity (STP).

Tsodyks & Markram 1997. Models depression (tau_d) and facilitation (tau_f)
with use parameter u_se.

- **__init__**(tau_d, tau_f, u_se)
- **update**(pre_spikes)
  - Compute effective weight scaling given pre-synaptic spikes.

### Class `StructuralPlasticity`
Activity-dependent synapse creation and elimination.

Grows new synapses between correlated neurons and prunes weak ones.

- **__init__**(growth_rate, prune_threshold)
- **update**(projection)
  - Grow or prune synapses in a Projection based on activity.

### Function `_fast_sigmoid_surrogate(v, threshold)`
Surrogate gradient: d/dv of fast-sigmoid spike function.

Neftci et al. 2019, Eq. 5.

---

## Module `learning.callbacks`

### Class `TrainingCallback`
Base class for training callbacks.

- **log**(metrics, step)
- **close**()

### Class `TensorBoardCallback`
Log scalars to TensorBoard via ``torch.utils.tensorboard``.

- **__init__**(log_dir)
- **log**(metrics, step)
- **close**()

### Class `WandBCallback`
Log metrics to Weights & Biases.

- **__init__**(project)
- **log**(metrics, step)
- **close**()

### Class `CSVCallback`
Log metrics to a CSV file (no dependencies).

- **__init__**(path)
- **log**(metrics, step)
- **close**()

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
- **apply_ewc_penalty**(step_size)
  - Push weights back toward consolidated values, weighted by Fisher info.

---

## Module `learning.neuroevolution`

### Class `SNNGeneticEvolver`
Genetic Algorithm for evolving SNN weights/parameters.

- **__init__**(layer_factory, fitness_func)
- **evolve**(generations)
- **_crossover**(p1, p2)
- **_mutate**(ind)

---

## Module `learning.online_o1`

### Class `OnlineO1Config`
Hardware-bounded configuration for local reward-modulated STDP.

- **__post_init__**()
- **max_weight**()
  - Maximum unsigned fixed-point weight.
- **max_trace**()
  - Maximum unsigned trace value.
- **min_eligibility**()
  - Minimum signed eligibility value.
- **max_eligibility**()
  - Maximum signed eligibility value.
- **min_reward**()
  - Minimum signed reward input.
- **max_reward**()
  - Maximum signed reward input.
- **per_synapse_state_bits**()
  - Stored bits per synapse: weight plus three bounded traces.
- **to_scnir_annotation**()
  - Return deterministic SC-NIR metadata for online-learning synapses.

### Class `OnlineO1Snapshot`
Immutable synapse state snapshot after one online update.


### Class `OnlineO1Synapse`
One fixed-point reward-modulated STDP synapse with O(1) state.

- **__post_init__**()
- **state_fields**()
  - Names of state fields retained between timesteps.
- **state_bit_count**()
  - Stored state bits for one synapse.
- **snapshot**()
  - Return the current bounded state.
- **step**()
  - Advance one streamed timestep and return the bounded state.

### Function `build_online_o1_memory_proof()`
Return a sequence-length independent memory proof for the rule.

### Function `_decay_unsigned(value, shift, max_value)`
### Function `_decay_signed(value, shift)`
### Function `_saturate(value, lower, upper)`
---

## Module `learning.schedulers`

### Class `StepScheduler`
Drop learning rate by *gamma* every *step_size* steps.

- **__init__**(lr_init, step_size, gamma)
- **step**()
- **reset**()

### Class `ExponentialScheduler`
Multiply learning rate by *gamma* each step.

- **__init__**(lr_init, gamma)
- **step**()
- **reset**()

### Class `CosineScheduler`
Cosine annealing from *lr_init* to *lr_min* over *total_steps*.

- **__init__**(lr_init, lr_min, total_steps)
- **step**()
- **reset**()

### Class `WarmupCosineScheduler`
Linear warmup followed by cosine decay.

- **__init__**(lr_init, lr_min, warmup_steps, total_steps)
- **step**()
- **reset**()

---

## Module `license`

### Class `CommercialLicenseStatus`
Current AGPL/commercial licence state.

The raw licence key is intentionally never stored in this object.


### Function `get_license_status()`
Return the current local licence state without contacting a network.

### Function `reset_license_status()`
Reset process-local licence state to the default AGPL mode.

### Function `set_license_key(key)`
Validate and install an explicit commercial licence key for this process.

### Function `load_license_from_env()`
Validate ``SC_NEUROCORE_LICENSE_KEY`` when it is explicitly configured.

### Function `validate_license_key(key)`
Validate a Polar customer-portal licence key.

Validation is opt-in. AGPL users who never call this function, never call
``set_license_key()``, and do not set ``SC_NEUROCORE_LICENSE_KEY`` are not
blocked and do not need the HTTP dependency.

### Function `_status_from_polar_response(response)`
### Function `_post_json_with_httpx(endpoint, payload)`
### Function `_string_value(value)`
### Function `_optional_string(value)`
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
  - Map quantum probability |beta|^2 to concentration &#91;0, 10&#93; uM.
- **bio_to_stochastic**(concentration, length)
  - Map concentration to bitstream.
- **get_functor**(source, target)

---

## Module `math.topology`

### Function `winding_number(phases)`
Compute the winding number of a phase trajectory around S^1.

The winding number counts how many times the phase wraps around
the circle &#91;0, 2*pi). It is a topological invariant — continuous
deformations of the trajectory cannot change it.

Parameters
----------
phases : np.ndarray, shape (T,)
    Time series of phase values (radians).

Returns
-------
int
    Number of complete windings (positive = counterclockwise).

### Function `ollivier_ricci_curvature(knm, i, j)`
Compute Ollivier-Ricci curvature between nodes i and j on the coupling graph.

Ollivier (2009), "Ricci curvature of Markov chains on metric spaces."
The curvature kappa(i,j) measures how much the neighborhoods of i and j
overlap. Positive curvature = neighborhoods converge (community structure).
Negative curvature = neighborhoods diverge (bottleneck).

Approximation: kappa(i,j) = 1 - W1(mu_i, mu_j) / d(i,j)
where mu_i is the lazy random walk distribution from node i,
and W1 is the Wasserstein-1 distance on the graph.

Simplified version: uses the coupling strength ratio as a proxy.

Parameters
----------
knm : np.ndarray, shape (N, N)
    Coupling matrix (non-negative, not necessarily symmetric).
i, j : int
    Node indices.

Returns
-------
float
    Estimated Ollivier-Ricci curvature in &#91;-1, 1&#93;.

### Function `sheaf_consistency_defect(phases, knm)`
Compute the sheaf consistency defect for the SCPN phase state.

In sheaf theory, a global section exists iff the gluing conditions
are satisfied on all overlaps. For the SCPN, the coupling matrix
defines the overlaps, and the phase differences weighted by coupling
measure the failure to glue.

defect = (1/N^2) * sum_{i,j} |K_ij| * |1 - cos(theta_i - theta_j)|

When phases are synchronized (all equal), defect = 0.
When phases are maximally incoherent, defect approaches max(|K|).

This is equivalent to (1 - Kuramoto_R) weighted by coupling.

Parameters
----------
phases : np.ndarray, shape (N,)
    Phase values (radians) for each layer/oscillator.
knm : np.ndarray, shape (N, N)
    Coupling matrix.

Returns
-------
float
    Sheaf consistency defect >= 0. Zero means globally coherent.

### Function `connection_curvature(phases, knm)`
Compute the connection curvature from PGBO phase dynamics.

The PGBO covariant derivative u_mu = dphi_mu - alpha * A_mu
defines a U(1) connection. The curvature F_{ij} = K_{ij} * cos(theta_i - theta_j)
measures the obstruction to parallel transport between layers i and j.

Parameters
----------
phases : np.ndarray, shape (N,)
    Phase values.
knm : np.ndarray, shape (N, N)
    Coupling matrix.

Returns
-------
np.ndarray, shape (N, N)
    Connection curvature matrix. Diagonal is zero.

---

## Module `memristor.memristor_mapper`

### Class `MemristorTechnology`
Supported memristor fabrication technologies.


### Class `ConductanceModel`
Per-device conductance variability model.

Models both device-to-device (D2D) fabrication variability and
cycle-to-cycle (C2C) read/write noise as Gaussian distributions.

- **__post_init__**()
- **dynamic_range**()
  - ON/OFF conductance ratio.
- **level_step**()
  - Conductance step between adjacent levels.
- **target_conductance**(level)
  - Nominal conductance for a given quantisation level.
- **sample_d2d**(level, rng)
  - Sample actual conductance with device-to-device variability.
- **sample_rw**(conductance, rng)
  - Apply read/write noise to a conductance value.
- **drift**(conductance, elapsed_s, alpha)
  - Model conductance drift over time.
- **thermal_shift**(conductance, temp_c, ref_c)
  - Temperature-dependent conductance shift.

### Class `SneakPathModel`
Estimates sneak-path leakage in passive crossbar arrays.

In an M×N passive crossbar, selecting cell (r,c) exposes parallel
leakage paths through unselected devices. Worst-case sneak current
is proportional to (M+N-2) × G_off.

- **worst_case_sneak**(rows, cols, g_off, v_read)
  - Worst-case sneak current (A) through unselected paths.
- **signal_to_sneak_ratio**(g_on, g_off, rows, cols)
  - Ratio of desired signal current to sneak current.

### Class `IRDropModel`
Models interconnect wire resistance and voltage drop.

- **voltage_drop**(row, col)
  - Accumulated IR drop at cell (row, col) from corner.
- **effective_conductance**(g_nominal, row, col, v_read)
  - Conductance seen at read amplifier after IR drop.

### Class `StuckFaultMap`
Map of stuck-at ON/OFF devices in a crossbar.

- **generate**(cls, rows, cols, fault_rate, seed)
  - Generate random stuck faults at given rate.
- **is_stuck**(row, col)
  - Return 'on', 'off', or None.
- **num_faults**()
- **fault_rate**()

### Class `AgingReport`
Results of aging simulation.


### Class `AgingSimulator`
Simulates conductance drift over device lifetime.

- **__init__**(model, alpha)
- **simulate**(conductances, elapsed_s)
  - Apply drift to all conductances, return (drifted, report).

### Class `SCAbsorbEncoder`
Adjusts SC encoding thresholds to absorb known device variability.

Instead of compensating post-silicon, pre-distorts the bitstream
encoding thresholds so that the effective computation matches ideal.

- **compute_adjusted_thresholds**(ideal_weights, actual_conductances, model, q_bits)
  - Return Q8.8 adjusted thresholds that absorb device error.

### Class `WriteVerifyResult`
Outcome of iterative write-verify programming.


### Class `WriteVerifyProtocol`
Iterative program-verify loop for memristor cells.

- **__init__**(model, max_iterations, tolerance, seed)
- **program_cell**(target_level)
  - Program a single cell to target level with verify.

### Class `CrossbarPowerEstimate`
Power and performance estimate for a crossbar tile.


### Class `CrossbarEstimator`
Estimates power, latency, and area for crossbar arrays.

- **estimate**(cls, crossbar)

### Class `CrossbarTopology`

### Class `CrossbarArray`
Physical crossbar array specification.

- **num_devices**()
- **conductance_model**()

### Class `VariabilityInjector`
Injects fab-realistic conductance variability into weight matrices.

- **__init__**(model, seed)
- **quantize_weights**(weights)
  - Map floating-point weights &#91;0, 1&#93; to conductance levels.
- **inject_d2d**(levels)
  - Apply device-to-device variability to quantised levels.
- **inject_rw**(conductances)
  - Apply read/write noise to conductance values.
- **inject_full**(weights)
  - Full pipeline: quantise → D2D → R/W noise. Returns (levels, conductances).
- **compute_error**(weights, conductances)
  - Compute variability-induced error statistics.

### Class `CompensationStrategy`

### Class `CompensationLUT`
Per-device compensation lookup table.

Maps nominal level → compensated threshold to absorb known D2D
variability at design time (program-verify or digital pre-distortion).

- **build**(cls, device_id, model, measured_g)
  - Build compensation LUT from measured or modelled conductances.
- **max_compensation**()
  - Maximum compensation ratio (deviation from 1.0).

### Class `CrossbarMapping`
Mapping of a weight matrix to a crossbar array.


### Class `MappingResult`
Full result of a memristor mapping pass.


### Class `MemristorMapper`
Maps SC network weight matrices to physical crossbar arrays.

- **__init__**(technology, topology, max_crossbar_size, compensation, seed)
- **map_weights**(weights)
  - Map a weight matrix (or list of matrices) to crossbar arrays.

### Class `MonteCarloReport`
Results from Monte Carlo variability co-simulation.


### Class `MonteCarloSimulator`
Monte Carlo co-simulation with variability injection.

Runs N trials of a crossbar multiply-accumulate operation with
independently sampled D2D + R/W variability to estimate output
error distributions and yield.

- **__init__**(model, num_trials, tolerance, seed)
- **simulate_mac**(weights, inputs)
  - Simulate multiply-accumulate with variability.

### Class `VerilogEmitter`
Generates crossbar-aware SystemVerilog with compensation LUTs.

- **__init__**(bit_width, frac_bits)
- **emit_crossbar**(mapping, module_name)
  - Generate SystemVerilog for a single crossbar tile.
- **emit_top**(result, module_name)
  - Generate top-level module instantiating all crossbar tiles.

---

## Module `meta_plasticity.meta_plasticity`

### Class `STDPParams`
Mutable STDP parameters.

- **to_vector**()
- **from_vector**(cls, v)

### Class `STPParams`
Mutable short-term plasticity parameters.

- **to_vector**()
- **from_vector**(cls, v)

### Class `HomeostaticParams`
Homeostatic plasticity: target rate + gain modulation.

- **adapt**(measured_rate_hz)
  - Adjust gain to push firing rate toward target.

### Class `BitstreamParams`
Mutable SC bitstream parameters.


### Class `PlasticityRuleSet`
Complete set of mutable plasticity rules.

This is the "genome" that the meta-controller evolves.

- **to_vector**()
  - Serialise all mutable params to a flat vector.
- **from_vector**(cls, v, gen)
- **vector_dim**()
- **copy**()

### Class `MetaSignalType`
Types of meta-control signals from L16.


### Class `MetaControlSignal`
One meta-control directive.


### Class `MetaController`
SC-domain meta-controller that rewrites plasticity rules.

Observes network performance metrics (surprise, novelty, GCI,
firing rates) and emits rule modifications. The controller itself
uses SC bitstream logic for decision-making.

- **__init__**(sensitivity, rng_seed)
- **observe**(metrics)
  - Record one observation of network state.
- **decide**()
  - Decide what meta-control signals to emit.
- **apply_signals**(rules, signals)
  - Apply meta-control signals to a rule set (in-place).

### Class `RuleEvolver`
Evolutionary engine for plasticity rule sets.

Maintains a population of candidate rules, evaluates fitness,
and evolves via selection + crossover + mutation.

- **__post_init__**()
- **evaluate_fitness**(rules, metrics)
  - Compute fitness from network performance metrics.
- **select_parents**()
  - Tournament selection of two parents.
- **crossover**(p1, p2)
  - Uniform crossover of two rule sets.
- **mutate**(rules)
  - Gaussian mutation of rule parameters.
- **evolve**()
  - Run one generation of evolution.
- **best**()
- **mean_fitness**()

### Class `NeuromodulatorType`

### Class `NeuromodulatorState`
Simulated neuromodulatory tone that modulates meta-plasticity.

- **update**(novelty, surprise, gci)
  - Update neuromodulator levels from network state.
- **modulation_factor**(param)
  - Get a combined modulation factor for a specific parameter type.

### Class `EngineConfig`
Configuration for the meta-plasticity engine.


### Class `MetaPlasticityEngine`
Top-level orchestrator for self-evolving meta-plasticity.

Connects ArcaneNeuron populations, the meta-controller, and the
rule evolver into a single lifelong learning system.

- **step**(metrics)
  - Process one timestep of meta-plasticity.
- **status**()

### Class `RuleCheckpoint`
Serialisable snapshot of a PlasticityRuleSet at a point in time.

- **restore**()

### Class `CheckpointStore`
Persistent store for rule checkpoints.

- **save**(rules, step, tag)
- **restore_best**()
- **restore_by_tag**(tag)
- **count**()

### Class `EWCProtection`
Elastic Weight Consolidation for plasticity parameters.

Penalises large deviations from previously learned rule vectors
to protect against catastrophic forgetting.

- **consolidate**(rules)
  - Set the current rules as the anchor point.
- **penalty**(rules)
  - Compute EWC penalty for deviating from anchor.
- **regularise**(rules, max_penalty)
  - Pull rules back toward anchor if penalty exceeds threshold.

### Class `CuriositySignal`
Intrinsic curiosity based on prediction error of network state.

Tracks expected next-state and computes curiosity as the prediction
error magnitude. High curiosity → explore (increase meta-plasticity).

- **update**(state_vector)
  - Update prediction model and return curiosity score.

### Class `MetaLearningRate`
Learning rate of the learning rate.

Adapts the meta-controller sensitivity based on whether recent
rule changes improved fitness.

- **update**(fitness_delta)
  - Adjust meta_lr from fitness delta. Positive = good → speed up.

### Class `SleepPhase`
Offline memory consolidation via experience replay.

Stores recent metric snapshots and replays them during sleep to
stabilise rule sets without new input.

- **record**(metrics)
- **sleep**(engine_step_fn)
  - Run consolidation by replaying buffered experiences.
- **buffer_size**()

### Class `SynapticTag`
Synaptic tag for early→late LTP conversion.

- **is_expired**()

### Class `TaggingModel`
Synaptic tagging and capture model.

Early-phase LTP creates a tag. If a consolidation signal arrives
before the tag decays, it is "captured" into late-phase LTP.

- **create_tag**(synapse_id, strength, time_ms)
- **decay_tags**(dt_ms)
- **consolidate**(consolidation_strength)
  - Attempt capture on all active tags. Returns count of captured.
- **prune_expired**()
- **active_tags**()

### Class `ContextRuleBank`
Maintains separate rule sets per context/task.

Allows rapid switching between learned plasticity configurations
without catastrophic interference.

- **store**(context, rules)
- **switch**(context)
- **contexts**()
- **num_contexts**()

### Class `FitnessTrajectory`
Tracks fitness over time and detects trends.

- **record**(fitness)
- **trend**()
  - Return slope of fitness over recent window. >0 = improving.
- **is_improving**()
- **is_stagnant**()
- **best_ever**()

### Class `RuleConstraints`
Hard constraints on plasticity parameters to prevent pathological values.

- **enforce**(rules)
  - Clamp all parameters to valid ranges.
- **is_valid**(rules)
  - Check if all parameters are within constraints.

### Function `population_diversity(evolver)`
Compute diversity as mean pairwise L2 distance of rule vectors.

### Function `inject_diversity(evolver, n_random)`
Replace worst individuals with fresh random rule sets.

---

## Module `model_zoo.configs`

### Function `mnist_classifier(n_hidden)`
784-128-10 feedforward SNN for MNIST-like digit classification.

Architecture follows Zenke & Ganguli 2018 (SuperSpike), Table 1:
input Poisson layer -> hidden LIF -> output LIF.  Weights are
Xavier-uniform initialised (not trained).

Reference: Zenke & Ganguli, Neural Computation 30(6), 2018.

### Function `dvs_gesture_classifier(n_classes)`
Event-camera gesture recognition SNN (Amir et al. 2017 / IBM DVS128).

2-layer feedforward: 256 input -> 256 hidden -> n_classes output.
Poisson input simulates DVS event stream at 500 Hz.

Reference: Amir et al., CVPR 2017.

### Function `shd_speech_classifier()`
Spiking Heidelberg Digits (SHD) recurrent architecture.

Recurrent hidden layer with sparse recurrent connectivity,
projecting to 20-class readout.  Topology matches
Cramer et al. 2020, Table 2: 700 input -> 256 recurrent -> 20 out.

Reference: Cramer et al., IEEE TNNLS 33(7), 2022.

### Function `brunel_balanced_network(n_exc, n_inh, g, eta)`
Brunel 2000 sparse balanced E/I network.

Parameters default to the synchronous irregular (SI) regime:
g=5.0 (inhibitory strength ratio), eta=2.0 (external rate / threshold rate).
Connectivity probability = 0.1 (epsilon in the paper).

Reference: Brunel, J. Comput. Neurosci. 8(3), 2000, Sec. 2.

### Function `cortical_column(n_layers)`
Potjans-Diesmann 2014 cortical microcircuit (scaled down).

4-layer column (L2/3, L4, L5, L6) with E and I populations per
layer, using Pospischil RS neurons (excitatory) and Golomb FS
neurons (inhibitory).  Sizes scaled to ~5% of the original model.

Reference: Potjans & Diesmann, Cerebral Cortex 24(3), 2014.

### Function `central_pattern_generator(n_oscillators)`
Half-centre CPG for quadruped locomotion.

Pairs of mutually inhibiting HindmarshRose oscillators produce
alternating burst patterns.  Adjacent pairs are coupled with
phase lag ~pi/2 (walk gait).

Reference: Ijspeert, Neural Networks 21(4), 2008, Sec. 3.

### Function `decision_making_circuit(n_per_pool)`
Wang 2002 / Wong & Wang 2006 spiking attractor decision circuit.

Two selective excitatory pools compete via a shared inhibitory
population.  Uses HH neurons for excitatory pools and WangBuzsaki
for inhibitory interneurons.

Reference: Wang, Neuron 36(5), 2002, Fig. 1.

### Function `working_memory_circuit(n_neurons)`
Compte et al. 2000 bump attractor for spatial working memory.

Ring of NMDA-based excitatory neurons with distance-dependent
connectivity and uniform inhibition.  Transient cue creates a
persistent activity bump encoding a remembered location.

Reference: Compte et al., J. Neurophysiol. 84(3), 2000.

### Function `auditory_processing(n_channels)`
Cochlear filterbank -> SNN spectro-temporal processing.

Tonotopic input layer (one population per frequency channel) ->
lateral inhibition (onset detection) -> integration layer.
HodgkinHuxley neurons model auditory nerve fibre dynamics.

Reference: Goodman & Brette, Front. Neurosci. 4, 2010.

### Function `visual_cortex_v1(n_orientation, n_per_orientation)`
Simple/complex cell model of primary visual cortex.

Orientation-tuned simple cells (HodgkinHuxley) feed into complex
cells (WangBuzsaki) that pool over phase.  Cross-orientation
inhibition sharpens selectivity.

Reference: Hubel & Wiesel, J. Physiol. 160, 1962;
           Carandini & Heeger, Nat. Rev. Neurosci. 13, 2012.

---

## Module `model_zoo.model_zoo`

### Class `NeuronState`
Generic state container for neuron models.

- **__getitem__**(key)
- **__setitem__**(key, value)
- **copy**()
- **as_dict**()

### Class `PluginMeta`
Metadata carried by every neuron plugin.


### Class `NeuronPlugin`
Abstract base class for pluggable neuron models.

- **meta**()
- **default_state**()
- **default_params**()
- **ode_dynamics**(state, current, params, dt)
  - Advance the neuron state by one timestep *dt*.
- **threshold_check**(state, params)
  - Return True if the neuron has fired.
- **reset**(state, params)
  - Reset state after a spike.
- **simulate**(current_trace, dt, params)
  - Convenience: simulate a full current trace.

### Class `LIFPlugin`
Leaky Integrate-and-Fire neuron model.

- **meta**()
- **default_state**()
- **default_params**()
- **ode_dynamics**(state, current, params, dt)
- **threshold_check**(state, params)
- **reset**(state, params)

### Class `IzhikevichPlugin`
Izhikevich (2003) simple model of spiking neurons.

- **meta**()
- **default_state**()
- **default_params**()
- **ode_dynamics**(state, current, params, dt)
- **threshold_check**(state, params)
- **reset**(state, params)

### Class `AdExPlugin`
Adaptive Exponential Integrate-and-Fire (Brette & Gerstner 2005).

- **meta**()
- **default_state**()
- **default_params**()
- **ode_dynamics**(state, current, params, dt)
- **threshold_check**(state, params)
- **reset**(state, params)

### Class `HodgkinHuxleyPlugin`
Hodgkin–Huxley conductance-based model (1952).

- **meta**()
- **default_state**()
- **default_params**()
- **ode_dynamics**(state, current, params, dt)
- **threshold_check**(state, params)
- **reset**(state, params)

### Class `PluginRegistry`
Discovers and manages neuron model plugins.

- **__init__**()
- **register**(plugin)
- **get**(name)
- **list_plugins**()
- **__len__**()
- **__contains__**(name)
- **with_builtins**(cls)
  - Create a registry pre-loaded with all built-in neuron models.

### Class `VerilogGenerator`
Generates synthesisable SystemVerilog from a NeuronPlugin.

- **__init__**(bit_width, frac_bits)
- **generate**(plugin)
  - Produce a complete SystemVerilog module for the given plugin.
- **_to_fixed**(value)

### Class `DocGenerator`
Generates markdown documentation from plugin metadata.

- **generate**(plugin)
- **generate_index**(registry)
  - Generate a summary index for all registered plugins.

---

## Module `model_zoo.pretrained`

### Function `_dense_to_csr(dense)`
Convert dense weight matrix (fan_in, fan_out) to CSR arrays.

### Function `_apply_weights(proj, dense)`
Overwrite a Projection's CSR data with a dense weight matrix.

### Function `_validate_archive_members(name, archive)`
### Function `_load_weight_matrix(name, archive, key, expected_shape)`
### Function `load_pretrained(name)`
Load a network with pre-initialised weights.

Supported names: ``'mnist'``, ``'shd'``, ``'dvs_gesture'``.

Raises ``ValueError`` for unknown names.

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

## Module `nas.darts_sc_nas`

### Class `BitstreamCandidate`
Represents an operation simulated with a specific SC bitstream length.
In a forward pass, this acts as an identity/quantizer (simulating SC variance).

- **__init__**(length, lut_cost, power_cost)
- **forward**(x)

### Class `SCMixedOp`
Continuous relaxation over discrete SC bitstream configurations.

- **__init__**(c_in, c_out, kernel_size, stride, padding)
- **forward**(x)
- **expected_resource_cost**()
- **extract_optimal_config**()

### Class `SCNASNetwork`
A small hardware-aware search network representing a deep SNN.

- **__init__**()
- **forward**(x)
- **hardware_penalty**()

---

## Module `nas.equiv`

### Class `EquivResult`
Result of a formal equivalence check.

- **summary**()

### Function `generate_miter(dut_module, ref_module, top_name, data_width, fraction)`
Generate a Verilog miter circuit for two modules.

Both modules must have identical port signatures:
clk, rst_n, leak_k, gain_k, I_t, noise_in -> spike_out, v_out

### Function `generate_sby(top_name, verilog_files, depth, engine)`
Generate a SymbiYosys .sby proof script.

### Function `check_equivalence(dut_verilog, ref_verilog, depth, run)`
Check formal equivalence between DUT and reference.

Parameters
----------
dut_verilog : str
    DUT module name (must exist in hdl/).
ref_verilog : str
    Reference module name (must exist in hdl/equiv/).
depth : int
    BMC depth (number of clock cycles to check).
run : bool
    If True, actually run SymbiYosys. Requires sby + z3 installed.
    If False, generate proof files and return without running.

Returns
-------
EquivResult

---

## Module `nas.sc_nas_engine`

### Class `DecorrelationStrategy`

### Class `NeuronType`

### Class `FPGAResourceBudget`
Hardware resource constraints for the target FPGA.

- **utilisation**(luts, ffs, bram, dsp)

### Class `NASObjective`
Search objectives and constraints.


### Class `LayerConfig`
Configuration for a single network layer.

- **lut_cost**()
- **ff_cost**()
- **dsp_cost**()
- **bram_cost_kb**()
- **power_cost**()

### Class `SCCandidate`
A candidate SC network architecture.

- **evaluate_resources**()
- **meets_budget**(budget)
- **fingerprint**()

### Class `SCFitnessEvaluator`
Pure-Python SC simulation fitness evaluator.

Uses the SC variance model: for a bitstream of length N encoding
probability p, the variance is p*(1-p)/N.  Accuracy is estimated
as 1 − mean_variance across all layers.

- **__init__**(seed)
- **evaluate**(candidate, target_p)
  - Evaluate candidate accuracy via SC variance model.

### Class `EvolutionaryNAS`
µ+λ evolutionary search with tournament selection.

- **__init__**(objective, budget, population_size, num_generations, mutation_rate, seed, convergence_patience, surrogate_optimizer)
- **_random_layer**()
- **_random_candidate**(gen)
- **_mutate**(candidate, gen)
- **_crossover**(a, b, gen)
- **_tournament_select**(population, k)
- **_evaluate_candidate**(candidate)
  - Evaluate a candidate through the configured NAS scoring path.
- **search**()
  - Run the evolutionary search. Returns the final Pareto front.

### Class `NASReport`
Summary report from an SC-NAS search.

- **best_accuracy**()
- **most_efficient**()
- **summary**()

### Class `NASVerilogEmitter`
Emits SystemVerilog for Pareto-optimal SC-NAS candidates.

- **emit**(candidate, module_name)
  - Generate SystemVerilog for a searched architecture.
- **emit_pareto**(front)
  - Emit Verilog for all Pareto-optimal candidates.

### Function `pareto_front(candidates, objectives)`
Extract the Pareto-optimal front (NSGA-II non-dominated sorting).

Maximises accuracy, minimises resource usage.

### Function `_assign_crowding_distance(front)`
Assign NSGA-II crowding distance to Pareto front members.

### Function `run_nas(objective, budget, population_size, num_generations, seed, convergence_patience, surrogate_optimizer)`
Convenience entry point for SC-NAS search.

---

## Module `nas.search`

### Class `NASResult`
Result of a NAS run.

- **best_accuracy**()
  - Architecture with highest accuracy on the Pareto front.
- **best_efficiency**()
  - Architecture with lowest energy on the Pareto front.
- **summary**()

### Function `_evaluate(arch, target, accuracy_fn)`
Evaluate one architecture: hardware cost + optional accuracy.

### Function `_dominates(a, b)`
True if a Pareto-dominates b (higher accuracy AND lower energy).

### Function `_non_dominated_sort(population)`
NSGA-II non-dominated sorting. Returns list of fronts.

### Function `_crowding_distance(front)`
Compute NSGA-II crowding distance for a front.

### Function `_tournament_select(population, fronts, rng)`
Binary tournament selection using front rank + crowding distance.

### Function `nas(space, target, population_size, generations, max_luts, accuracy_fn, seed)`
Run hardware-aware NAS using NSGA-II.

Parameters
----------
space : SearchSpace
    Architecture search space definition.
target : str
    FPGA target for hardware cost evaluation.
population_size : int
    Number of architectures per generation.
generations : int
    Number of evolutionary generations.
max_luts : int, optional
    Hard LUT budget. Architectures exceeding this are penalized.
    If None, uses the target's total LUT count.
accuracy_fn : callable, optional
    Function(Architecture) -> float accuracy in &#91;0, 1&#93;.
    If None, uses a proxy based on network capacity.
seed : int
    Random seed.

Returns
-------
NASResult
    Pareto front + all evaluated architectures.

---

## Module `nas.search_space`

### Class `Architecture`
One point in the NAS search space.

- **n_layers**()
- **layer_sizes**()
- **total_params**()

### Class `SearchSpace`
Configurable NAS search space.

Parameters
----------
n_inputs : int
    Input dimension.
n_outputs : int
    Output dimension (width of final layer).
min_layers, max_layers : int
    Range of hidden layer count.
width_choices : list of int
    Candidate widths per layer.
neuron_choices : list of str
    Candidate neuron models.
L_choices : list of int
    Candidate bitstream lengths.
delay_choices : list of int
    Candidate max-delay values.

- **random_architecture**(rng)
  - Sample a random architecture from the space.
- **mutate**(arch, rng)
  - Mutate one random gene in the architecture.
- **crossover**(a, b, rng)
  - Uniform crossover between two architectures of equal layer count.
- **space_size**()
  - Approximate total architectures in the search space.

---

## Module `nas.surrogate_bridge`

### Class `SupportsSurrogateOptimise`
Minimal optimiser interface required by SC-NAS integration.

- **optimise**(network)
  - Return a surrogate report for NAS layer profiles.

### Class `NASPolicyLayer`
Per-layer compiler policy attached to an SC-NAS candidate.


### Class `NASPolicyPlan`
Surrogate optimiser policy projected onto an SC-NAS candidate.


### Class `NASPolicyEvaluation`
Result of calling the surrogate optimiser from inside NAS evaluation.


### Function `candidate_layer_profiles(candidate)`
Convert an SC-NAS candidate into optimiser layer profiles.

### Function `optimise_candidate_policy(candidate, optimiser)`
Run the surrogate optimiser for a candidate and optionally apply its policy.

### Function `evaluate_candidate_with_surrogate(candidate, optimiser)`
Score a candidate through the surrogate optimiser for NAS search loops.

### Function `build_nas_policy_plan(candidate, report)`
Project a surrogate optimiser report onto a NAS candidate.

### Function `apply_surrogate_policy(candidate, report)`
Return a candidate copy with compatible bitstream/decorrelator settings.

### Function `_layer_ids(candidate, layer_ids)`
### Function `_required_config(report, layer_id)`
### Function `_to_nas_decorrelation(name)`
---

## Module `network._torch_bridge`

### Class `_PopulationSpec`

### Class `NetworkTorchBridge`
Differentiable bridge for a bounded subset of declarative ``Network`` graphs.

- **__init__**(populations, projections, surrogate_fn)
- **_find_input_populations**()
- **_find_output_populations**()
- **_projection_weight**(projection)
- **_initial_state**(batch, device, dtype)
- **forward**(inputs)
  - Run the differentiable bridge on ``(T, batch, input_dim)`` current input.
- **sync_to_network**()
  - Copy learned bridge weights back into the underlying CSR projections.

### Function `_csr_to_dense(projection)`
### Function `_csr_mask(projection)`
### Function `_validate_projection_csr(projection)`
### Function `_build_population_spec(population, surrogate_fn)`
---

## Module `network.cortical_column`

### Class `CorticalColumn`
Potjans & Diesmann 2014 8-population cortical microcircuit.

Parameters
----------
scale : float, optional
    Population-size multiplier in (0, 1&#93;. Default 0.1
    (≈ 7700 neurons), which is the smallest size where the
    published Table 4 firing rates are reproduced within
    tolerance. `scale=1.0` yields the full ~77 000-neuron model.
bg_rate : float, optional
    Background Poisson rate per channel (Hz). Defaults to the
    published 8.0; setting to 0.0 disengages the drive (useful
    for fidelity tests that confirm cells go silent).
g_inh : float, optional
    Relative inhibitory weight. Defaults to the published 4.0.
scale_correction : bool, optional
    When True (default), scales per-synapse weights by 1/scale
    so that mean drive per cell is preserved at sub-full scale
    (van Albada et al. 2015). Disable to study finite-size
    effects directly.
seed : int or None, optional
    Per-instance RNG seed for connectivity, voltages and
    background Poisson. None → fresh entropy.

- **__init__**(scale, bg_rate, g_inh, scale_correction, delay_distribution, n_delay_bins, use_block_csr, seed, backend)
- **_stack_block**(triples, n_rows, n_cols)
  - Concatenate per-pair (rows, cols, data) into a CSR.
- **_init_buffers**(dt)
- **step**(dt)
  - Advance the network by one timestep `dt` (ms).
- **_inject_block**(dt)
  - Block-CSR injection. With the batched Rust kernel
- **_spmv_into**(block, x, y, arrays)
  - `y += block @ x`. Uses the Rust rayon-parallel kernel
- **_inject_per_pair**(dt)
  - Legacy per-pair injection: one mat-vec per (pair, bin).
- **_integrate_and_detect**(dt)
  - Per-population LIF Euler step + spike detect + buffer push.
- **simulate**(duration_ms, dt)
  - Run the network for `duration_ms` ms.
- **population_rates**(rasters, dt, burn_in_ms)
  - Return mean firing rate (Hz) per population.
- **total_indegree**(target)
  - Return mean total synaptic in-degree of a target cell.
- **reset_state**()
  - Re-randomise voltages, currents and refractory state.
- **__repr__**()
- **population_names**()

### Function `_is_inhibitory(pop_name)`
---

## Module `network.export`

### Function `_check_exportable(network)`
Validate all populations use LIF-compatible models.

### Function `_emit_lif_module(pop, idx)`
Generate Verilog for one LIF population.

### Function `_emit_top(network, target)`
Emit top-level Verilog wrapper.

### Function `export_verilog(network, output_dir, target)`
Export a LIF-based network to Verilog files.

---

## Module `network.gamma_oscillation`

### Class `PINGCircuit`
Conductance-based PING circuit (Börgers-Kopell 2003).

Parameters mirror the publication; defaults reproduce the 40 Hz
weak-PING regime from Fig 2A. Override `i_drive_e_mean` to scan
drive-frequency curves; override `w_EI` / `w_IE` to scan the
gain-loop strength.

- **__post_init__**()
- **step**(dt)
  - Advance one timestep (dt in ms); return (spikes_e, spikes_i).
- **_step_rust**(dt)
  - Rust-backed per-step kernel — matches the Python path
- **_step_julia**(dt)
- **_step_go**(dt)
- **_step_mojo**(dt)
- **_step_python**(dt)
  - Reference Python implementation. Bit-identical to
- **reset_state**()
  - Re-initialise voltages, conductances and refractory state.
- **population_rate**(spike_log, dt, bin_ms)
  - Bin per-step spike booleans into a population rate (Hz).
- **dominant_frequency**(spike_log, dt, bin_ms, f_min, f_max)
  - Return the dominant frequency (Hz) in the population rate.

---

## Module `network.monitor`

### Class `SpikeMonitor`
Records (neuron_idx, timestep) pairs from a population.

- **__init__**(population, label)
- **record**(spikes, t_step)
  - Store spike events for this timestep (from binary spike vector).
- **record_event**(neuron_id, t_step)
  - Store a single spike event directly (from Rust backend).
- **spike_times**()
  - All spike timesteps as 1-D array.
- **spike_trains**()
  - Per-neuron spike timestep arrays.
- **count**()
  - Total number of spikes recorded.
- **raster_data**()
  - Return (timesteps, neuron_ids) arrays for raster plots.
- **firing_rates**(n_steps, dt)
  - Mean firing rate (Hz) per neuron over the simulation.
- **isi**(neuron)
  - Inter-spike intervals (timestep units) for a single neuron.
- **cross_correlation**(i, j, max_lag)
  - Cross-correlogram between neurons i and j.

### Class `StateMonitor`
Records state variable traces from a population.

- **__init__**(population, variables, record)
- **snapshot**(t_step)
  - Capture current state variables.
- **traces**()
  - Variable traces as {name: (n_steps, n_neurons)} arrays.
- **t**()
  - Timestep array.

### Class `RateMonitor`
Population firing rate in time bins.

- **__init__**(population, bin_ms)
- **record**(spikes, t_step, dt)
  - Accumulate spikes; flush when a bin completes.
- **rate**()
  - Firing rate (Hz) per bin.
- **t**()
  - Bin edge timestep array.

---

## Module `network.mpi_runner`

### Class `MPIRunner`
MPI-distributed network simulation.

Partitions populations across MPI ranks via round-robin assignment.
Each rank steps only its local populations; spikes propagate via
``MPI_Allgatherv`` every timestep.

Each rank steps supported local populations through the Rust engine's
``step_population`` API when the extension is importable and every
local model on the rank is supported. Otherwise the runner falls back
to ``Population.step_all`` for CPU-only environments. ``spike_gating``
and ``fim_lambda`` are unsupported by this runner — the
``Network._run_mpi`` dispatcher raises ``NotImplementedError`` when
either is requested with ``backend='mpi'``.

- **__init__**(network)
- **_initialize_rust_dispatch**()
  - Prepare a rank-local Rust runner when the installed engine supports it.
- **_partition_populations**()
  - Round-robin assignment of populations to ranks.
- **_identify_cross_rank_projections**()
  - Separate projections into local and cross-rank.
- **_exchange_spikes**(local_spikes)
  - Allgatherv spike vectors so every rank knows who spiked.
- **_step_local**(pop_to_currents, last_spikes)
  - Step only this rank's populations, return local spike dict.
- **run**(n_steps, dt)
  - Run the distributed simulation for *n_steps* timesteps.

### Function `_require_mpi()`
### Function `_get_rust_engine()`
### Function `_rust_supports_model(model_name)`
---

## Module `network.network`

### Class `Network`
Declarative network: collects objects, runs the simulation loop.

- **__init__**()
- **add**(obj)
  - Register a simulation object by type.
- **_can_use_rust**()
- **run**(duration, dt, progress, backend, spike_gating)
  - Run the simulation for *duration* seconds at timestep *dt*.
- **_run_mpi**(duration, dt)
- **_run_rust**(duration, dt)
- **_run_python**(duration, dt, progress)
- **_apply_stimuli**(pop_to_currents, t, dt)
  - Inject stimulus currents into target populations.
- **_apply_projections**(pop_to_currents, last_spikes)
  - Propagate spikes through all projections.
- **_record**(pop, spikes, t, dt)
  - Feed spikes/states to all monitors attached to this population.
- **_update_plasticity**(last_spikes)
  - Apply plasticity rules to projections that have them.
- **_apply_fim**(last_spikes)
  - Fisher Information Metric self-observation feedback.
- **to_torch**(surrogate_fn)
  - Build an explicit differentiable bridge without altering NumPy/Rust execution.

### Function `_load_network_runner_class()`
### Function `_get_rust_engine()`
### Function `_rust_supports_model(model_name)`
---

## Module `network.population`

### Class `Population`
A group of N identical neurons with vectorized state access.

- **__init__**(model, n, params, label)
  - Create *n* neurons of *model* (class or string name).
- **_sync_voltages**()
  - Pull membrane voltage from each neuron into the flat array.
- **step_all**(currents, spike_gating)
  - Advance all neurons one timestep; return binary spike vector.
- **reset_all**()
  - Reset every neuron to its initial state.
- **get_states**()
  - Collect all neuron states into arrays keyed by variable name.
- **set_voltages**(voltages)
  - Sync voltages from an external source (e.g. Rust backend) into neurons.
- **voltages**()
  - Current membrane voltages (read-only view).

### Function `_resolve_model(model)`
Return a model class from a string name or pass through a class.

---

## Module `network.projection`

### Class `Projection`
Synaptic projection from source to target population.

Parameters
----------
delay : float, array-like, or 0
    - 0: no delay (default)
    - scalar > 0: uniform axonal delay (all synapses share one delay)
    - 1-D array of length n_synapses: per-synapse delay in timesteps.
      Enables heterogeneous axonal/synaptic delays.

- **__init__**(source, target, weight, probability, delay, topology, plasticity, seed, weight_threshold)
  - Create projection with CSR connectivity and optional delay/plasticity.
- **_init_delays**(delay)
  - Set up delay buffers based on delay specification.
- **n_synapses**()
  - Number of synaptic connections.
- **delay_mode**()
  - Delay mode: 'none', 'uniform', or 'per_synapse'.
- **max_delay**()
  - Maximum delay in timesteps across all synapses.
- **_build_connectivity**(topology, probability, seed)
  - Build CSR arrays from topology name or pre-built tuple.
- **propagate**(source_spikes)
  - Compute target currents from source spikes through CSR connectivity.
- **update_plasticity**(src_spikes, tgt_spikes, a_plus, a_minus, tau, directional_bias)
  - Trace-based STDP weight update.
- **_enforce_symmetry**()
  - Symmetrise CSR weight data: W_ij = W_ji = (W_ij + W_ji) / 2.

### Function `_csr_matvec(indptr, indices, data, x, n_out, weight_threshold)`
CSR matrix-vector product: result&#91;j&#93; += data&#91;k&#93; * x&#91;i&#93; for each (i,j).

Skips source neurons with x&#91;i&#93;==0 (spike-driven) and optionally
skips synapses with |data&#91;k&#93;| <= weight_threshold (sparse weights).

### Function `_csr_delayed_matvec(indptr, indices, data, delay_steps, spike_history, hist_idx, n_out)`
CSR matrix-vector product with per-synapse delays.

For each synapse k connecting source i to target j with delay d_k:
    current&#91;j&#93; += data&#91;k&#93; * spike_history&#91;(hist_idx - d_k) % max_delay, i&#93;

### Function `validate_csr_topology(indptr, indices, data, n_source, n_target)`
Validate and normalize CSR connectivity arrays.

---

## Module `network.stimulus`

### Class `TimedArray`
Time-varying current from a pre-computed array.

- **__init__**(values, dt)
- **get_current**(t_step)
  - Return the value at timestep t_step (clamps to last value).

### Class `PoissonInput`
Random Poisson spike input producing weighted current.

- **__init__**(n, rate_hz, weight, dt, seed)
- **get_current**(t_step, dt)
  - Generate Poisson spikes and return weighted current vector.

### Class `StepCurrent`
Rectangular step current between onset and offset timesteps.

- **__init__**(onset, offset, amplitude)
- **get_current**(t_step, dt)
  - Return amplitude if within &#91;onset, offset), else 0.

---

## Module `network.topology`

### Function `_to_csr(n_rows, n_cols, rows, cols, weights)`
Convert COO arrays to CSR (indptr, indices, data).

### Function `random_connectivity(n_src, n_tgt, p, weight, seed)`
Erdos-Renyi random connectivity.

### Function `small_world(n, k, p_rewire, weight, seed)`
Watts-Strogatz small-world graph (n-by-n adjacency).

### Function `scale_free(n, m, weight, seed)`
Barabasi-Albert preferential attachment (n-by-n adjacency).

### Function `ring_topology(n, k, weight)`
Ring topology with k nearest neighbours in each direction.

### Function `grid_topology(rows_count, cols_count, radius, weight)`
2D lattice connectivity within Manhattan radius.

### Function `all_to_all(n_src, n_tgt, weight)`
Full connectivity (every source to every target).

---

## Module `neuro_symbolic.agent`

### Class `PredictiveAgentConfig`
Configuration for a hybrid symbolic-spiking predictive agent.

- **__post_init__**()

### Class `SCErrorSignature`
SC-domain prediction-error signature.

`xor_bits` is the stochastic-computing error carrier. `popcount`
is the integer error magnitude used by hardware-friendly decision
logic.

- **to_dict**()
  - Return a JSON-compatible representation.

### Class `HybridInferenceResult`
Result of one high-level neuro-symbolic predictive pass.

- **to_dict**()
  - Return a compact JSON-compatible summary.

### Class `NeuroSymbolicPredictiveAgent`
Hybrid predictive-coding agent for symbolic-spiking workflows.

- **__init__**(config)
- **num_symbols**()
  - Number of registered symbolic labels.
- **register_symbols**(symbols)
  - Register additional symbolic labels.
- **observe**(observation)
  - Run one predictive-symbolic observation pass.
- **_validate_observation**(observation)

### Function `build_sc_error_signature(observation, prediction)`
Build an XOR/popcount error signature from observation and prediction.

---

## Module `neuro_symbolic.predictive_coding`

### Class `BindOp`
Supported HDC binding operations.


### Class `ReasoningStep`
Single step in a symbolic reasoning trace.


### Class `ReasoningTrace`
Captures a symbolic reasoning chain for audit and formal verification.

Each step records the symbol query, the operation applied, the
similarity score to the best match, and a confidence metric derived
from the Hamming margin between the best and second-best candidates.

- **add**(symbol, operation, similarity, confidence)
- **length**()
- **mean_confidence**()
- **is_complete**()
- **finalize**()
- **to_dict**()

### Class `Hypervector`
Packed binary hypervector (pure-Python mirror of the Rust Hypervector).

Uses ``np.uint64`` packed bitstream layout compatible with the
``neuro_symbolic`` crate's ``Vec<u64>`` representation.

- **__init__**(data, length)
- **zeros**(cls, dim)
- **random**(cls, seed, dim)
- **bind**(other)
  - XOR binding (self-inverse, dimension-preserving).
- **permute**(shift)
  - Cyclic right rotation by *shift* bits.
- **hamming_distance**(other)
  - Normalised Hamming distance (0.0 = identical, 1.0 = opposite).
- **similarity**(other)
  - Cosine-like similarity: 1 − 2·hamming.
- **popcount**()
- **density**()
- **threshold_bundle**(vectors)
  - Majority-vote bundle across N vectors.

### Class `SymbolEncoder`
Deterministic symbol → hypervector mapping (mirrors Rust SymbolEncoder).

- **__init__**(base_seed)
- **encode**(symbol)
- **encode_sequence**(symbols)
- **vocabulary_size**()
- **_symbol_seed**(symbol)

### Class `PredictiveCodingLayer`
Single layer in a hierarchical predictive coding network.

Maintains a generative model: top-down predictions are compared
against bottom-up observations to produce prediction errors that
drive weight updates and symbolic trace emission.

- **__init__**(input_dim, hidden_dim, lr, precision, seed)
- **predict**(hidden)
  - Generate a top-down prediction from the hidden state.
- **compute_error**(observation, hidden)
  - Bottom-up prediction error: weighted residual.
- **update**(observation, hidden)
  - One-step gradient update on both weights and hidden state.
- **mean_recent_error**()
- **converged**()

### Class `VerifiableInference`
Wraps prediction + HDC symbol matching with an auditable trace.

- **__init__**(encoder, layer, symbol_library)
- **register_symbol**(name)
  - Register a symbol into the lookup library.
- **register_symbols**(names)
- **num_symbols**()
- **infer**(observation, top_k)
  - Run inference: prediction → error → HDC symbol match.

### Function `_unpack(hv)`
### Function `_pack(bits, length)`
---

## Module `neuro_symbolic.self_verification`

### Class `VerificationStatus`
Status of one self-verification obligation.


### Class `VerificationObligation`
One checked condition in a self-verification trace.

- **to_dict**()
  - Return a JSON-ready obligation.

### Class `NeuroSymbolicSelfVerificationTrace`
Machine-checkable summary of a neuro-symbolic inference result.

- **passed**()
  - Whether every obligation passed.
- **failed_obligations**()
  - Names of failed obligations.
- **to_dict**()
  - Return a JSON-ready trace.

### Class `NeuroSymbolicSelfVerifier`
Build checked self-verification traces for inference outputs.

- **verify_result**(result)
  - Verify a high-level hybrid inference result against its observation.
- **verify_trace_only**(trace)
  - Verify a trace when only symbolic evidence is available.
- **_validate_vector**(values, name)
- **_check_shape**(name, lhs, rhs)
- **_check_prediction_error**(observation, prediction, error)
- **_check_signature**(observation, prediction, signature)
- **_check_reasoning_trace**(trace)
- **_check_symbol_scores**(symbol_scores)
- **_build_trace**(result, obligations)

### Function `build_self_verification_trace(result)`
Convenience wrapper for high-level neuro-symbolic inference results.

### Function `_stable_digest(payload)`
### Function `_json_default(value)`
---

## Module `neurons._units`

### Function `require_pint()`
### Function `is_quantity(value)`
### Function `require_quantity(value, label)`
### Function `quantity_to_base(value)`
### Function `_dimensionless_magnitude(value, fn_name)`
### Function `_exp(value)`
### Function `_log(value)`
### Function `_sin(value)`
### Function `_cos(value)`
### Function `_tanh(value)`
### Function `_sinh(value)`
### Function `_cosh(value)`
### Function `_sigmoid(value)`
### Function `_sqrt(value)`
### Function `_clip(value, low, high)`
### Function `build_quantity_namespace()`
### Function `validate_quantity_expression(expr, env)`
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
XOR-nonlinearity neuron with shunting inhibition.

Implements ``d1 + d2 - 2*d1*d2`` (XOR truth table for binary inputs).
Based on Koch, *Biophysics of Computation*, 1999, Ch. 12.

- **step**(input_a, input_b)
- **reset_state**()
  - Reset internal state to defaults.
- **get_state**()
  - Return dict with internal state.

---

## Module `neurons.dsl_cli`

### Function `cmd_list(args)`
List all bundled schemas.

### Function `cmd_validate(args)`
Validate schemas.

### Function `cmd_info(args)`
Show model information.

### Function `cmd_compile(args)`
Compile model to Verilog with optional hardware target profile.

### Function `cmd_simulate(args)`
Simulate model for N steps.

### Function `cmd_precision(args)`
Show precision diagnostics for a model at every supported format.

Analyses all 9 precision modes, checking parameter range,
dt encoding, and providing a recommended mode.

### Function `cmd_platforms(args)`
List all available hardware target profiles.

### Function `main()`
Parse CLI arguments and dispatch to the requested subcommand.

---

## Module `neurons.equation_builder`

### Class `EquationNeuron`
Neuron defined by arbitrary ODE equations as strings.

Each equation is a right-hand-side expression for dX/dt.
Variables can reference other state variables, parameters,
and the special variable `I` (input current).

``units="strict"`` enables opt-in pint-based dimensional
validation before the expressions are compiled for runtime.

- **__init__**(equations, parameters, state, threshold, reset, constants, dt, method, units, input_unit)
  - Initialise an equation-defined neuron from ODE strings.
- **_validate_expr**(expr)
  - Validate an expression against the AST whitelist.
- **_ast_depth**(node)
  - Return the maximum nesting depth of an AST.
- **_prepare_strict_runtime**()
  - Convert pint quantities to base-unit floats for runtime.
- **_convert_runtime_value**(name, value)
  - Convert a runtime value from pint quantity to float.
- **_build_env**()
  - Build the eval environment with parameters, state, and noise.
- **step**(I)
  - Advance the neuron by one timestep; return 1 if it spikes.
- **get_state**()
  - Return current state, with units if in strict mode.
- **reset**()
  - Reset state to initial values.
- **__repr__**()
  - Human-readable representation of the neuron equations.

### Function `from_equations()`
Factory: build EquationNeuron from Brian2-style equation strings.

Example:
    lif = from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
    )

Use ``units="strict"`` with pint quantities to validate the
equation dimensions before runtime compilation.

---

## Module `neurons.fixed_point_lif`

### Class `FixedPointLIFNeuron`
Bit-true fixed-point model of the Verilog ``sc_lif_neuron``.

All arithmetic is performed in signed Q(FRACTION) fixed-point with
explicit bit-width masking so that overflow/wrap behaviour matches
the hardware exactly.

Parameters
----------
data_width : int
    Total bit width of all fixed-point values (default 16).
fraction : int
    Number of fractional bits (default 8, giving Q8.8).
v_rest, v_reset, v_threshold : int
    Membrane parameters in Q(FRACTION) fixed-point.
refractory_period : int
    Number of clock cycles to hold after a spike.

Example
-------
>>> neuron = FixedPointLIFNeuron()
>>> spike, v = neuron.step(leak_k=240, gain_k=16, I_t=100)
>>> spike in (0, 1)
True
>>> neuron.reset()

- **__post_init__**()
- **step**(leak_k, gain_k, I_t, noise_in)
  - Execute one clock cycle — bit-true match to Verilog RTL.
- **reset**()
  - Reset neuron state to power-on defaults.
- **reset_state**()
  - Reset internal state (alias for :meth:`reset`).
- **get_state**()
  - Return dict with internal state.

### Class `FixedPointLFSR`
Bit-true model of the 16-bit LFSR in ``sc_bitstream_encoder.v``.

Polynomial: x^16 + x^14 + x^13 + x^11 + 1
Taps (0-indexed): 15, 13, 12, 10

Example
-------
>>> lfsr = FixedPointLFSR(seed=0xACE1)
>>> vals = &#91;lfsr.step() for _ in range(10)&#93;
>>> len(set(vals)) > 1  # produces varying pseudo-random values
True

- **__post_init__**()
- **step**()
  - Advance one clock cycle; return new register state.
- **reset**(seed)

### Class `FixedPointBitstreamEncoder`
Bit-true model of ``sc_bitstream_encoder.v``.

Combines LFSR + comparator to produce a stochastic bitstream
where P(bit=1) ~ x_value / (2^DATA_WIDTH - 1).

Example
-------
>>> enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
>>> bits = &#91;enc.step(x_value=128) for _ in range(100)&#93;
>>> all(b in (0, 1) for b in bits)
True

- **__post_init__**()
- **step**(x_value)
  - Return 1 if LFSR < x_value, else 0 (one clock cycle).
- **reset**()

### Function `_mask(value, width)`
Mask to DATA_WIDTH bits and interpret as signed two's complement.

---

## Module `neurons.homeostatic_lif`

### Class `HomeostaticLIFNeuron`
LIF neuron with homeostatic threshold adaptation.

Self-regulates firing rate toward a target setpoint via exponential
moving average of spike rate. Based on Turrigiano (2012).

Example
-------
>>> neuron = HomeostaticLIFNeuron(target_rate=0.1, noise_std=0.0)
>>> for _ in range(200):
...     neuron.step(1.5)
>>> neuron.v_threshold != 1.0  # threshold adapted
True

- **__post_init__**()
- **step**(input_current)
- **get_state**()

---

## Module `neurons.models.adaptive_threshold_if`

### Class `AdaptiveThresholdIFNeuron`
Integrate-and-fire with dynamic threshold. Platkiewicz & Bhatt 2010.

C dV/dt = -g_L(V - V_rest) + I
dtheta/dt = -(theta - theta_rest) / tau_theta
On spike: V -> V_reset, theta += delta_theta

Reference: Platkiewicz, J. & Brette, R. (2010). J. Neurosci. 30:6891–6902.

- **step**(current)
- **reset**()

---

## Module `neurons.models.adaptive_threshold_moe`

### Class `AdaptiveThresholdMoENeuron`
Adaptive threshold MoE spiking neuron (SpikingBrain).

Parameters
----------
k : float
    Firing rate control (higher k -> lower threshold -> more spikes).
    Default: 4.0 (SpikingBrain recommended).
ema_alpha : float
    EMA decay for running mean of |input|. Default: 0.1.

- **step**(current)
  - Advance one timestep. Returns integer spike count (>= 0).
- **step_collapsed**(activation)
  - Time-collapsed single-step: s_INT = round(x / V_th).
- **sparsity**()
  - Current activation sparsity (1.0 if below threshold, 0.0 if firing).
- **reset**()
  - Reset state to initial conditions.

---

## Module `neurons.models.adex`

### Class `AdExNeuron`
Adaptive Exponential Integrate-and-Fire. Brette & Gerstner 2005.

dv/dt = -(v - v_rest)/tau + delta_T * exp((v - v_rh)/delta_T) / tau - w/C + I/C
dw/dt = (a * (v - v_rest) - w) / tau_w
if v >= v_threshold: v = v_reset, w += b

Reference: Brette, R. & Gerstner, W. (2005). J. Neurophysiol. 94:3637–3642.

Integrator options:
- ``baseline_euler`` preserves the historical explicit-Euler path
- ``rk4`` is an explicit higher-order alternative path
- ``rosenbrock`` is a linearly implicit stiff-system path over the same
  AdEx ODEs

- **__post_init__**()
- **step**(current)
- **_rhs**(_t, state, current)
- **_step_baseline_euler**(current)
- **_step_rk4**(current)
- **_step_rosenbrock**(current)
- **reset**()

---

## Module `neurons.models.ai_optimized`

### Class `MultiTimescaleNeuron`
Three-compartment memory neuron with fast/medium/slow timescales.

dv_fast/dt   = (-v_fast + I) / tau_fast
dv_medium/dt = (-v_medium + alpha * spike_fast) / tau_medium
dv_slow/dt   = (-v_slow + beta * v_medium) / tau_slow
theta_eff    = theta_base - gamma * v_slow

Spike when v_fast >= theta_eff. The slow compartment accumulates
context over seconds, modulating excitability.

Reference: Custom SC-NeuroCore model — no external publication.

- **step**(current)
- **reset**()

### Class `AttentionGatedNeuron`
Spiking neuron with learned sigmoid attention gate.

gate = sigmoid(w_key * I + w_query * v)
dv/dt = (-v + gate * I) / tau

Each neuron learns which input magnitudes to attend to
and which to suppress, via key/query weights.

- **step**(current)
- **reset**()

### Class `PredictiveCodingNeuron`
Fires only on prediction errors.

dpred/dt = (I - pred) / tau_pred
surprise = |I - pred|
dv/dt    = (-v + surprise) / tau

Silent when input matches prediction. Fires on novel stimuli.

- **step**(current)
- **reset**()

### Class `SelfReferentialNeuron`
Introspects on its own spike history to modulate dynamics.

self_rate = count(recent_spikes) / window
effective_tau = tau * (1 + self_rate / target_rate)
theta_eff = theta * (1 + regularity)

Regular firing lowers threshold (maintain pattern).
Chaotic firing raises threshold (stabilize).

- **step**(current)
- **reset**()

### Class `CompositionalBindingNeuron`
Phase-coding neuron for variable binding.

dphi/dt = omega + coupling * sin(phi_input - phi)
dA/dt   = (-A + I) / tau
Spike when A * cos(phi) > theta.

Two neurons in-phase encode bound concepts.
Phase offset encodes relational structure.

- **step**(current)
- **reset**()

### Class `DifferentiableSurrogateNeuron`
Spiking neuron with learnable surrogate gradient parameters.

Forward: spike = int(v >= theta)
Backward: surrogate = 1 / (1 + beta * |v - theta|)^2   &#91;conceptual&#93;
v = alpha * v * (1 - spike) + I

alpha (decay), beta (steepness), theta (threshold) all trainable.

- **step**(current)
- **reset**()
- **surrogate_grad**()
  - Smooth surrogate gradient for backprop.

### Class `ContinuousAttractorNeuron`
Ring attractor for continuous working memory.

u_i += (-u_i + f(sum_j w_ij u_j) + I_i) / tau * dt
f(x) = max(0,x)^2 / (1 + max(0,x)^2)
w_ij = A * exp(-d_ij^2 / (2*sigma_e^2)) - B   (Mexican hat)

Holds a continuous value (angle/position) in persistent activity.
Output: position of the activity bump.

- **__post_init__**()
- **_build_weights**()
- **_activation**(x)
- **step**(current)
- **bump_position**()
- **reset**()

### Class `MetaPlasticNeuron`
Neuron with self-regulating meta-learning rate.

dv/dt = (-v + I) / tau
error_trace += (-error_trace + |reward - expected|) / tau_meta * dt
meta_lr = lr0 * sigmoid(kappa * (error_trace - target_error))

High error: faster learning. Low error: stabilize.

- **step**(current)
- **update_meta**(reward)
- **meta_lr**()
- **reset**()

---

## Module `neurons.models.aihara_map_neuron`

### Class `AiharaMapNeuron`
Aihara 1990 chaotic map neuron.

2D discrete map exhibiting chaotic spiking dynamics. The fast variable
x is driven by a sigmoidal self-feedback modulated by k_f, minus the
slow recovery variable y.

x(n+1) = k_f · x(n) · σ(x(n) + α) - y(n) + I
y(n+1) = k_s · y(n) + δ · x(n)

where σ(z) = 1 / (1 + exp(-z)).

Reference: Aihara et al. (1990) Phys Lett A 144:333–340.

- **step**(current)
- **reset**()

---

## Module `neurons.models.akida_neuron`

### Class `AkidaNeuron`
BrainChip Akida 2021 — event-domain rank-order IF neuron.

Membrane integrates weighted spikes with rank-order decay:
V += weight * modulation^rank
Spike when V >= threshold. No leak between events.

Reference: BrainChip Inc. (2021). Akida Neuromorphic Processor Reference Manual.

- **step**(weight)
  - Process one input spike event with given synaptic weight.
- **reset**()

---

## Module `neurons.models.alpha`

### Class `AlphaNeuron`
Alpha-synapse neuron. Rall 1967.

Dual excitatory/inhibitory synaptic currents with alpha-function kinetics.

Reference: Gerstner, W. & Kistler, W.M. (2002). Spiking Neuron Models. Cambridge Univ. Press, §4.1.

- **step**(exc_current, inh_current)
- **reset**()

---

## Module `neurons.models.alpha_motor_neuron`

### Class `AlphaMotorNeuron`
Alpha motor neuron — spinal cord, innervates extrafusal muscle fibres.

WB Na⁺/K⁺ core + persistent inward current (PIC, L-type Ca²⁺ for
plateau potentials) + Ca²⁺-dependent AHP (SK channels for rate
limiting). Larger soma (C_m=1.5).

Reference: Powers & Binder (2001) J Neurophysiol 86;
Heckman & Enoka (2012) Compr Physiol 2(4).

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.amari_field`

### Class `AmariNeuralField`
Amari 1977 — continuous neural field, discretized on N nodes.

tau du_i/dt = -u_i + sum_j w(|i-j|) f(u_j) * dx + I_i
w(x) = A * exp(-a*|x|) - B * exp(-b*|x|)    (Mexican hat)
f(u) = max(0, u)                              (Heaviside-linear)

Reference: Amari, S. (1977). Biol. Cybern. 27:77–87.

- **__post_init__**()
- **_build_kernel**()
- **step**(current)
  - Advance one timestep. Returns mean activation.
- **reset**()

---

## Module `neurons.models.arcane_neuron`

### Class `ArcaneNeuron`
Unified self-referential cognition neuron for persistent identity.

- **step**(current)
- **reset**()
- **identity_state**()
  - The deep compartment value — the accumulated identity.
- **confidence**()
- **novelty**()
- **identity_drift**()
  - Cumulative absolute magnitude of identity mutation.
- **meta_learning_rate**()
- **get_recent_pre_activity**()
  - Get proxy for pre-synaptic activation (recent spike behavior).
- **get_state**()

---

## Module `neurons.models.astrocyte`

### Class `AstrocyteModel`
Li & Bhatt 1994 — astrocyte Ca2+ signaling via IP3 receptor.

3 ODEs: Ca (cytosolic), h (IP3R de-inactivation), IP3.
Ca release from ER through IP3 receptor + SERCA pump + leak.

Reference: Postnov, D.E. et al. (2009). Neural Comput. 21:2746–2782.

- **step**(current)
  - Return cytosolic Ca concentration (uM). current = glutamate-driven IP3 production.
- **reset**()

---

## Module `neurons.models.astrocyte_adapter`

### Class `AstrocyteNeuron`
Population-compatible wrapper for AstrocyteModel.

Parameters
----------
ca_threshold : float
    Ca²⁺ concentration (µM) above which the astrocyte "fires"
    (releases gliotransmitter). Default 0.3 µM.
dt : float
    Timestep in seconds.

- **__post_init__**()
- **step**(current)
  - Advance one timestep. Returns 1 if Ca > threshold, else 0.
- **ca**()
- **ip3**()
- **reset**()

---

## Module `neurons.models.astrocyte_lif`

### Class `AstrocyteLIFNeuron`
Astrocyte-LIF hybrid with tripartite synapse (Perea et al. 2009).

Parameters
----------
tau_m : float
    Membrane time constant (ms). Default: 20.0.
tau_ca : float
    Calcium decay time constant (ms). Default: 500.0.
e_l : float
    Leak reversal potential (mV). Default: -65.0.
theta : float
    Spike threshold (mV). Default: -50.0.
v_reset : float
    Post-spike reset potential (mV). Default: -65.0.
ca_delta : float
    Calcium increment per presynaptic spike. Default: 0.1.
ca_thresh : float
    Calcium threshold for gliotransmitter release. Default: 0.5.
g_glio : float
    Gliotransmitter current amplitude. Default: 2.0.
dt : float
    Integration timestep (ms). Default: 0.1.

- **step_with_pre**(i_ext, pre_spike)
  - Step with external current and presynaptic spike indicator.
- **step**(current)
  - Step without presynaptic spike info (no glial feedback).
- **reset**()
  - Reset state to initial conditions.

---

## Module `neurons.models.atype_k_neuron`

### Class `ATypeKNeuron`
A-type K⁺ (IA) neuron — WB base + transient outward K⁺ current.

IA activates rapidly at subthreshold voltages and inactivates over
tens of milliseconds. Controls first-spike latency and interspike
intervals.

a∞ = 1 / (1 + exp(-(v + 50) / 20))   τ_a = 2 ms
b∞ = 1 / (1 + exp((v + 70) / 6))     τ_b = 50 ms

Reference: Connor & Stevens (1971) J Physiol 213:31;
Hoffman et al. (1997) Nature 387:869.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.av_ron_cardiac`

### Class `AvRonCardiacNeuron`
Av-Ron, Parnas & Segel 1993 — cardiac ganglion Type III bursting.

Reduced HH-type with slow inactivation producing plateau bursts.

Reference: Av-Ron, E. et al. (1991). Biol. Cybern. 65:487–500.

- **step**(current)
- **reset**()

---

## Module `neurons.models.balanced_resonate_and_fire`

### Class `BalancedResonateAndFireNeuron`
Balanced RF neuron with refractory threshold and smooth reset.

State variables follow the paper notation ``u = x + i y`` and refractory
state ``q``. One scalar update computes:

``b_t = p(omega) - b_offset - q_{t-1}``

``u_t = u_{t-1} + dt * ((b_t + i * omega) * u_{t-1} + current)``

``theta_t = theta_c + q_{t-1}``

``z_t = Heaviside(Re(u_t) - theta_t)``

``q_t = gamma * q_{t-1} + z_t``

Reference: Higuchi, Kairat, Bohte, and Otte (2024), "Balanced
Resonate-and-Fire Neurons", Proceedings of ICML 2024, Algorithm 1.

- **__post_init__**()
- **p_omega**()
  - Current divergence boundary for ``omega`` and ``dt``.
- **damping**()
  - Current smooth-reset damping ``b_t`` before the next step.
- **dynamic_threshold**()
  - Current refractory threshold ``theta_c + q``.
- **step**(current)
  - Advance one BRF timestep and return the binary spike ``z_t``.
- **reset**()
  - Reset membrane and refractory state to rest.
- **state**()
  - Return a compact state snapshot for reproducibility tests.
- **_validate_parameters**()

### Function `sustain_oscillation_boundary(omega, dt)`
Return the BRF divergence boundary ``p(omega)``.

``p(omega) = (-1 + sqrt(1 - (dt * omega)^2)) / dt``.
The value is real only when ``0 < dt * omega <= 1``.

---

## Module `neurons.models.benda_herz`

### Class `BendaHerzNeuron`
Benda & Herz 2003 — phenomenological spike-frequency adaptation.

f = f_onset(I - A)          instantaneous f-I curve
dA/dt = -A/tau_a + delta_a * f
f_onset(x) = f_max / (1 + exp(-beta*(x - I_half)))

Reference: Benda, J. & Herz, A.V.M. (2003). Neural Comput. 15:2523–2564.

- **__post_init__**()
- **_f_onset**(x)
- **step**(current)
- **reset**()

---

## Module `neurons.models.bertram_phantom`

### Class `BertramPhantomBurster`
Bertram et al. 2008 — phantom burster with dual slow variables.

C dV/dt  = -(I_Ca + I_K + I_s1 + I_s2 + I_L) + I_ext
ds1/dt   = (s1_inf(V) - s1) / tau_s1
ds2/dt   = (s2_inf(V) - s2) / tau_s2

Two slow variables (s1, s2) with different timescales produce
bursting via a phantom slow manifold.

Reference: Bertram, R. et al. (1995). Biophys. J. 68:2323–2332.

- **_boltz**(v, vh, k)
- **step**(current)
- **reset**()

---

## Module `neurons.models.bk_neuron`

### Class `BKNeuron`
BK (Big Conductance Ca²⁺-Activated K⁺) channel neuron.

Wang-Buzsáki base (Na⁺ + Kdr + leak) extended with a BK (KCa1.1)
current that depends jointly on membrane voltage and intracellular
Ca²⁺ concentration.

BK channels are the primary mediator of the fast
afterhyperpolarisation (fAHP) following action potentials. Their
joint voltage–Ca²⁺ dependence means they open during the spike peak
when Ca²⁺ influx is maximal, producing rapid repolarisation.

C_m dv/dt = -g_Na m³∞ h (v - E_Na) - g_K n⁴ (v - E_K)
            - g_BK BK∞(v, &#91;Ca²⁺&#93;) (v - E_K) - g_L (v - E_L) + I

BK∞ = 1 / (1 + exp(-(v - V_half_BK) / 15))
V_half_BK = 10 - 30 · &#91;Ca²⁺&#93; / (&#91;Ca²⁺&#93; + 0.5)

d&#91;Ca²⁺&#93;/dt = -&#91;Ca²⁺&#93; / τ_Ca  (+ 0.3 on spike)

Reference: Contreras et al. (2021) J Comput Neurosci;
Wang & Bhatt (1996) BK biophysics; Wang & Buzsáki (1996) base model.

- **step**(current)
  - Advance one dt. Returns 1 if spike, 0 otherwise.
- **reset**()
  - Reset to default initial conditions.

### Function `_safe_rate(a, vhalf, v, k, fallback)`
Rate function with safe handling of (v + vhalf) ≈ 0.

---

## Module `neurons.models.booth_rinzel`

### Class `BoothRinzelNeuron`
Booth & Rinzel 1995 — bistable motoneuron, 2-compartment.

C dVs/dt = -I_Na(Vs) - I_K(Vs) - I_L(Vs) - gc*(Vs - Vd)/p + I/p
C dVd/dt = -I_Ca(Vd) - I_KCa(Vd) - I_L(Vd) - gc*(Vd - Vs)/(1-p)
dq/dt   = (q_inf(Vd) - q) / tau_q
dCa/dt  = -f * (alpha_Ca * I_Ca + k_Ca * Ca)

Reference: Booth, V. & Rinzel, J. (1995). J. Neurophysiol. 73:1934–1945.

- **_safe_exp**(x)
- **step**(current)
- **reset**()

---

## Module `neurons.models.brainscales_adex`

### Class `BrainScaleSAdExNeuron`
BrainScaleS-2 — analog AdEx (1000x real-time). Schemmel 2010.

Reference: Schemmel, J. et al. (2010). Proc. ISCAS 2010: 1947–1950.

- **step**(current)
- **reset**()

---

## Module `neurons.models.brunel_wang`

### Class `BrunelWangNeuron`
LIF neuron with NMDA, AMPA, and GABA synaptic currents.

Reference: Brunel, N. & Wang, X.J. (2001). Effects of neuromodulation in a
cortical network model of object working memory dominated by
recurrent inhibition. J Comput Neurosci 11:63-85.

Used in decision-making and working memory models. The key feature
is the voltage-dependent NMDA conductance with Mg2+ block.

- **__post_init__**()
- **_nmda_voltage_dep**(v)
  - Mg2+ block factor: 1 / (1 + &#91;Mg2+&#93;/3.57 * exp(-0.062 * V)).
- **step**(i_ampa_ext, s_ampa_rec, s_nmda_rec, s_gaba)
  - Advance one timestep.
- **reset**()
- **get_state**()

---

## Module `neurons.models.butera_respiratory`

### Class `ButeraRespiratoryNeuron`
Butera, Rinzel & Smith 1999 — pre-Botzinger respiratory neuron.

Reference: Butera, R.J. et al. (1999). J. Neurophysiol. 82:382–397.

- **_sexp**(x)
- **_scosh**(x)
- **step**(current)
- **reset**()

---

## Module `neurons.models.cazelles_map`

### Class `CazellesMapNeuron`
Cazelles et al. 2001 — simple 2D bursting map neuron.

x(n+1) = f(x(n)) - y(n) + I
y(n+1) = y(n) + epsilon * (x(n) - sigma)

f(x) = a*x*(1 - x)    (logistic-like fast dynamics)

Bursting arises from slow y modulation of fast x.

Reference: Cazelles, B., Courbage, M. & Rabinovich, M. (2001). Europhys. Lett. 56(4):504–509.

- **step**(current)
- **reset**()

---

## Module `neurons.models.cerebellar_basket_neuron`

### Class `CerebellarBasketNeuron`
Cerebellar basket cell — perisomatic-targeting interneuron.

WB core + A-type K⁺ (transient outward) + Ca²⁺-dependent K⁺ (AHP).
Distinct from cortical PV+ by A-current and pronounced AHP.

Reference: Midtgaard (1992); Häusser & Clark (1997); WB (1996).

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.cfc`

### Class `ClosedFormContinuousNeuron`
Hasani et al. 2022 — closed-form continuous-depth neuron (CfC).

Analytical solution of the LTC ODE between timesteps:
x(t+dt) = x(t)*exp(-dt/tau_eff) + f_target*(1 - exp(-dt/tau_eff))
where tau_eff and f_target depend on input.

Reference: Canavier, C.C. et al. (1993). Biophys. J. 65:2373–2382.

- **step**(current)
- **reset**()

---

## Module `neurons.models.chandelier_neuron`

### Class `ChandelierNeuron`
Chandelier cell — axo-axonic fast-spiking interneuron.

WB core + Kv1 (D-type delay current, slow activation → first-spike
delay) + Kv3.1 (fast activation for AP sharpening). Targets AIS.

Reference: Woodruff et al. (2011); Wang & Buzsáki (1996).

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.chay`

### Class `ChayNeuron`
Chay 1985 — pancreatic beta cell burster.

Reference: Chay, T.R. (1985). Physica D 16:233–242.

- **step**(current)
- **reset**()

---

## Module `neurons.models.chay_keizer`

### Class `ChayKeizerNeuron`
Chay & Keizer 1983 — pancreatic beta cell with Ca-dependent K.

3 ODEs: V, n (delayed rectifier K), Ca (intracellular).
I_Ca fast, I_K delayed rectifier, I_K(Ca) slow.

Reference: Chay, T.R. & Keizer, J. (1983). Biophys. J. 42:181–190.

- **step**(current)
- **reset**()

---

## Module `neurons.models.chialvo_map`

### Class `ChialvoMapNeuron`
Chialvo 1995 — 2D discrete map neuron.

x&#91;n+1&#93; = x²·exp(y-x) + k + I
y&#91;n+1&#93; = a·y - b·x + c

Reference: Chialvo, D.R. (1995). Chaos, Solitons & Fractals 5:461–479.

- **step**(current)
- **reset**()

---

## Module `neurons.models.clif`

### Class `ComplementaryLIFNeuron`
Complementary LIF — ICML 2024, dual positive/negative spike paths.

Maintains separate excitatory/inhibitory membrane potentials; spike
emitted when their difference exceeds threshold.

Reference: Hunsberger, E. & Eliasmith, C. (2015). Neural Comput. 27(12):2548–2586.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.coba_lif`

### Class `COBALIFNeuron`
Conductance-based LIF — Destexhe et al. 2001.

C dV/dt = -g_L(V - E_L) - g_e(V - E_e) - g_i(V - E_i) + I
dg_e/dt = -g_e / tau_e
dg_i/dt = -g_i / tau_i

Reference: Brunel, N. (2000). J. Comput. Neurosci. 8:183–208.

- **step**(current, delta_ge, delta_gi)
- **reset**()

---

## Module `neurons.models.cochlear_hair_cell`

### Class `CochlearHairCell`
Cochlear inner hair cell with Boltzmann MET channel activation.

Parameters
----------
g_max : float
    Maximum MET channel conductance. Default: 10.0.
e_met : float
    MET channel reversal potential (mV). Default: 0.0.
g_l : float
    Leak conductance. Default: 1.0.
e_l : float
    Leak reversal / resting potential (mV). Default: -60.0.
cap : float
    Membrane capacitance (pF). Default: 10.0.
x0 : float
    Boltzmann half-activation displacement (nm). Default: 0.0.
delta : float
    Boltzmann slope factor (nm). Default: 0.1.
dt : float
    Integration timestep (ms). Default: 0.01.

- **p_open**(displacement)
  - Boltzmann activation of MET channels.
- **step**(displacement)
  - Step with basilar membrane displacement.
- **reset**()
  - Reset state to resting potential.

---

## Module `neurons.models.compte_wm`

### Class `CompteWMNeuron`
Compte et al. 2000 — NMDA-based working memory neuron.

C dV/dt = -I_L - I_AMPA - I_NMDA - I_GABA + I_ext
NMDA includes voltage-dependent Mg2+ block:
  B(V) = 1 / (1 + &#91;Mg&#93;/3.57 * exp(-0.062*V))
ds_NMDA/dt = -s_NMDA/tau_NMDA + alpha*x*(1-s_NMDA)
dx/dt      = -x/tau_x

Reference: Compte, A. et al. (2000). J. Neurophysiol. 84:853–868.

- **_mg_block**(v)
- **step**(current, spike_in)
- **reset**()

---

## Module `neurons.models.connor_stevens`

### Class `ConnorStevensNeuron`
Connor-Stevens 1977 — A-type potassium current, Type-I excitability.

4 ODEs: v, m (Na activation), h (Na inactivation), n (K), a (A-type), b (A-type inactivation).

Reference: Connor, J.A. & Stevens, C.F. (1971). J. Physiol. 213:31–53.

- **step**(current)
- **reset**()

---

## Module `neurons.models.courage_nekorkin_map`

### Class `CourageNekorkinMapNeuron`
Courbage, Nekorkin & Vdovin 2007 — piecewise-linear Lorenz-type map.

Reference: Courbage, M., Nekorkin, V.I. & Vdovin, L.V. (2007). Chaos 17:043109.

- **_f**(x)
- **step**(current)
- **reset**()

---

## Module `neurons.models.dcn_neuron`

### Class `DCNNeuron`
Deep cerebellar nuclei neuron — main output of the cerebellum.

WB Na⁺/K⁺ core + T-type Ca²⁺ (rebound bursting), Ih (pacemaker),
persistent Na⁺ (subthreshold), Ca²⁺-dependent AHP. 7 currents total.

Reference: Llinás & Mühlethaler (1988) J Physiol 404:241;
Jahnsen (1986) J Physiol 372:129.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.de_schutter_purkinje`

### Class `DeSchutterPurkinjeNeuron`
Single-compartment Purkinje-cell conductance model after De Schutter & Bower.

The maintained Python model exposes five active gating variables:
h_Na, n_K, m_CaP, h_CaP, and q_KCa. Use the audit index before treating this
compact implementation as a full multi-compartment reconstruction.

Reference: De Schutter, E. & Bower, J.M. (1994). J. Neurophysiol. 71:375–400.

- **step**(current)
  - Advance the compact conductance model and return a spike indicator.
- **reset**()
  - Restore voltage, gates, and calcium state to their defaults.

---

## Module `neurons.models.dendrify`

### Class `DendrifyNeuron`
Beniaguev et al. 2022 — active dendrite 2-compartment reduced model.

Soma: standard LIF with reset.
Dendrite: has a dendritic spike mechanism (NMDA-like) that produces
a supralinear calcium plateau when input exceeds d_threshold.

Reference: Pagkalos, M. et al. (2023). Nat. Commun. 14:3234.

- **step**(current)
- **reset**()

---

## Module `neurons.models.dendritic_nmda`

### Class `DendriticNMDANeuron`
Two-compartment neuron with NMDA Mg2+ block (Jahr & Stevens 1990).

Parameters
----------
g_nmda : float
    NMDA conductance. Default: 1.5.
e_nmda : float
    NMDA reversal potential (mV). Default: 0.0.
mg_conc : float
    Extracellular Mg2+ concentration (mM). Default: 1.0.
g_coupling : float
    Soma-dendrite coupling conductance. Default: 0.5.
tau_soma : float
    Soma time constant (ms). Default: 20.0.
tau_dend : float
    Dendrite time constant (ms). Default: 50.0.
theta : float
    Spike threshold (mV). Default: -50.0.
dt : float
    Integration timestep (ms). Default: 0.1.

- **mg_block**(v)
  - Mg2+ block factor: B(V) = 1/(1 + &#91;Mg&#93;/3.57 * exp(-0.062*V)).
- **step**(i_soma, glutamate)
  - Step with somatic input current and dendritic glutamate.
- **reset**()
  - Reset state to resting potential.

---

## Module `neurons.models.destexhe_thalamic`

### Class `DestexheThalamicNeuron`
Destexhe 1993 — thalamocortical relay with T-current and I_h.

6 ODEs: V, m_Na, h_Na, n_K, m_T, h_T (+ optional h-current).

Reference: Destexhe, A. et al. (1996). J. Comput. Neurosci. 3:19–46.

- **step**(current)
- **reset**()

---

## Module `neurons.models.direction_selective_rgc`

### Class `DirectionSelectiveRGC`
Direction-selective retinal ganglion cell.

Parameters
----------
tau : float
    Membrane time constant (ms). Default: 10.0.
theta : float
    Spike threshold. Default: 0.5.
is_on_centre : bool
    True for On-centre, False for Off-centre. Default: True.
w_centre : float
    Centre weight. Default: 1.0.
w_surround : float
    Surround inhibition weight. Default: 0.3.
direction_pref : float
    Preferred direction angle (radians). Default: 0.0.
dt : float
    Integration timestep. Default: 1.0.

- **new_on**(cls)
  - Create an On-centre cell.
- **new_off**(cls)
  - Create an Off-centre cell.
- **step_rf**(intensity, surround_mean)
  - Step with local intensity and surround mean intensity.
- **step**(current)
  - Simple step (no surround).
- **reset**()
  - Reset state to initial conditions.

---

## Module `neurons.models.dpi_neuron`

### Class `DPINeuron`
Indiveri et al. 2011 — DYNAP-SE differential-pair integrator.

Subthreshold log-domain dynamics modelling analog VLSI circuits.
tau dI_mem/dt = -I_mem + I_syn + I_leak
Spike when I_mem >= I_threshold, reset to I_reset.
All variables in current domain (nA), mirroring transistor currents.

Reference: Chicca, E. et al. (2014). Proc. IEEE 102:1367–1388.

- **step**(i_syn)
- **reset**()

---

## Module `neurons.models.durstewitz_dopamine`

### Class `DurstewitzDopamineNeuron`
Durstewitz, Seamans & Sejnowski 2000 — PFC neuron with D1 modulation.

D1 agonism enhances NMDA (g_nmda_scale), reduces Na window current
(v_shift_na), and enhances K (g_k_scale), stabilising up-states.

Reference: Durstewitz, D. et al. (2000). J. Neurophysiol. 83:1733–1750.

- **step**(current)
- **reset**()

---

## Module `neurons.models.e_prop_alif`

### Class `EPropALIFNeuron`
Bellec et al. 2020 — ALIF with eligibility traces for e-prop.

Adaptive LIF: threshold increases after each spike and decays.
Eligibility trace e_t tracks how synaptic weight changes affect
future spiking, enabling three-factor learning.

Reference: Bellec, G. et al. (2020). Nat. Commun. 11:3625.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.energy_lif`

### Class `EnergyLIFNeuron`
Fardet & Levina 2020 — LIF with metabolic energy constraint.

Reference: Blouw, P. et al. (2019). Proc. NeurIPS Workshop on Energy Efficient ML.

- **step**(current)
- **reset**()

---

## Module `neurons.models.ermentrout_kopell_map_neuron`

### Class `ErmentroutKopellMapNeuron`
Ermentrout-Kopell 1986 canonical Type I (theta neuron) map.

The canonical model for Type I (saddle-node) excitability. Phase
variable θ advances on a circle; spike occurs when θ crosses π.

θ(n+1) = θ(n) + dt · &#91;(1 - cos θ) + (1 + cos θ) · I&#93;

Reference: Ermentrout & Kopell (1986) SIAM J Appl Math 46:233–253.

- **step**(current)
- **reset**()

---

## Module `neurons.models.ermentrout_kopell_pop`

### Class `ErmentroutKopellPopulation`
Montbrio, Pazo & Roxin 2015 — exact mean-field of QIF/theta network.

Reference: Ermentrout, G.B. & Kopell, N. (1986). SIAM J. Appl. Math. 46:233–253.

- **step**(ext_input)
- **reset**()

---

## Module `neurons.models.escape_rate`

### Class `EscapeRateNeuron`
Gerstner 2000 — stochastic threshold (escape noise model).

Reference: Gerstner, W. (2000). Neural Comput. 12:43–89.

- **step**(current)
- **reset**()

---

## Module `neurons.models.expif`

### Class `ExpIFNeuron`
Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003.

Reference: Fourcaud-Trocmé, N. et al. (2003). J. Neurosci. 23:11628–11640.

- **step**(current)
- **reset**()

---

## Module `neurons.models.fitzhugh_nagumo`

### Class `FitzHughNagumoNeuron`
FitzHugh-Nagumo 1961 — 2D qualitative spike model.

dv/dt = v - v³/3 - w + I
dw/dt = ε(v + a - bw)

Reference: FitzHugh, R. (1961). Biophys. J. 1:445–466.

Integrator options:
- ``baseline_euler`` preserves the historical explicit-Euler path
- ``rk4`` is an explicit fourth-order path over the same two-state ODE
- ``rosenbrock`` is a linearly implicit path for stiff slow-fast regimes

- **__post_init__**()
- **step**(current)
- **_rhs**(_t, state, current)
- **_step_baseline_euler**(current)
- **_step_rk4**(current)
- **_step_rosenbrock**(current)
- **reset**()

---

## Module `neurons.models.fitzhugh_rinzel`

### Class `FitzHughRinzelNeuron`
FitzHugh 1976 / Rinzel 1987 — FHN + slow variable for bursting.

Reference: Rinzel, J. (1987). In: Mathematical Topics in Population Biology. Springer, pp. 267–281.

- **step**(current)
- **reset**()

---

## Module `neurons.models.fractional_lif`

### Class `FractionalLIFNeuron`
Fractional-order LIF — memory-dependent dynamics.

Uses Grünwald-Letnikov fractional derivative approximation.
D^α v(t) = -(v - v_rest) + R·I, where 0 < α ≤ 1.
α < 1 introduces memory (power-law decay instead of exponential).
Lundstrom et al. 2008.

Reference: Teka, W. et al. (2014). PLoS Comput. Biol. 10:e1003526.

- **__post_init__**()
- **_compute_gl_coefficients**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.galves_locherbach`

### Class `GalvesLocherbachNeuron`
Galves-Löcherbach 2013 — stochastic point process neuron.

P(spike at t | history) = φ(V(t))
V(t) = Σ w_j · spike_j(past) · decay + leak
Purely probabilistic, no ODE.

Reference: Galves, A. & Löcherbach, E. (2013). J. Stat. Phys. 151:896–921.

- **_firing_prob**()
- **step**(weighted_input)
- **reset**()

---

## Module `neurons.models.gamma_motor_neuron`

### Class `GammaMotorNeuron`
Gamma motor neuron — innervates intrafusal fibres of muscle spindles.

Simple LIF with spike-frequency adaptation. Two subtypes: dynamic
(bag1, velocity-sensitive) and static (bag2/chain, length-sensitive).

Reference: Prochazka & Hulliger (1989) Prog Brain Res 80;
Taylor et al. (1999) J Physiol 519(3).

- **static_type**(cls)
  - Static gamma — bag2/chain intrafusal fibres (length-sensitive).
- **step**(drive)
- **reset**()

---

## Module `neurons.models.gamma_renewal`

### Class `GammaRenewalNeuron`
Gamma renewal process neuron. Keat et al. 2001.

ISI ~ Gamma(k, k/rate). Hazard h(t) evaluated at elapsed time
since last spike. P(spike in dt) = h(t)*dt.

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §7.4.

- **__post_init__**()
- **step**(rate_override)
- **reset**()

### Function `_log_gamma_int(k)`
ln(Gamma(k)) for positive integer k = ln((k-1)!).

### Function `_gamma_survival(k, x)`
P(X > x) for Gamma(k, 1) via series for upper incomplete gamma.

---

## Module `neurons.models.gated_lif`

### Class `GatedLIFNeuron`
Yao et al. 2022 NeurIPS — LIF with learnable gates.

Reference: Yao, M. et al. (2022). Proc. NeurIPS 35:19606–19618.

- **step**(current)
- **reset**()

---

## Module `neurons.models.gif_population`

### Class `GIFPopulationNeuron`
Mensi et al. 2012 — Generalized IF with escape-rate spiking.

Stochastic threshold: P(spike|V) = lambda_0 * exp((V - theta) / delta_v).
Adaptation currents eta decay after each spike.

Reference: Mensi, S. et al. (2012). J. Neurophysiol. 107:1756–1775.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.glif`

### Class `GLIFNeuron`
Allen Institute GLIF5 — Generalized LIF, 5-level hierarchy.

Teeter et al. 2018, Nat Comm. Level 5: LIF + reset rules +
instantaneous threshold + threshold adaptation + after-spike currents.

Reference: Teeter, C. et al. (2018). Nat. Commun. 9:709.

- **step**(current)
- **reset**()

---

## Module `neurons.models.glm_neuron`

### Class `GLMNeuron`
Pillow et al. 2008 — generalized linear model (point-process GLM).

lambda(t) = exp(k . stim(t) + h . spike_history(t) + mu)
P(spike in dt) = lambda(t) * dt

k: stimulus filter (length n_k)
h: post-spike filter (length n_h), typically negative (refractoriness)

Reference: Pillow, J.W. et al. (2008). Nature 454:1058–1062.

- **__post_init__**()
- **step**(stimulus)
- **reset**()

---

## Module `neurons.models.golgi_cell`

### Class `GolgiCell`
Cerebellar Golgi cell — Solinas et al. 2007 full model.

11 ionic currents: INa_t (m³h), INa_p, IKdr (n⁴), IKA (a³b),
IKM (w), ICaT (mT²s), ICaN (c²), IBK (V+Ca²⁺), ISK (Ca²⁺),
Ih (r), IL. Spontaneously active 3–10 Hz.

Reference: Solinas et al. (2007) Front Cell Neurosci 1:2.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
### Function `_boltz(v, vh, k)`
---

## Module `neurons.models.golomb_fs`

### Class `GolombFSNeuron`
Golomb et al. 2007 — fast-spiking interneuron with Kv3.

C dV/dt = -I_Na - I_Kd - I_Kv3 - I_L + I_ext
Kv3 channel enables narrow spikes and high sustained firing.

Reference: Golomb, D. et al. (2007). J. Neurophysiol. 97:3831–3843.

- **step**(current)
- **reset**()

---

## Module `neurons.models.granule_cell`

### Class `GranuleCell`
Cerebellar granule cell — D'Angelo et al. 2001 model.

Smallest and most numerous neuron in the brain. 7 ionic currents:
INa (m³h), IKdr (n⁴), IKA (a³b), ICaT (mT²s), IKCa (Hill), Ih (r),
plus tonic GABA. Ca²⁺ dynamics with KCa half-saturation.

Reference: D'Angelo et al. (2001) J Neurosci 21:759–770.

- **_boltz**(v, vh, k)
- **step**(current)
- **reset**()

---

## Module `neurons.models.gutkin_ermentrout`

### Class `GutkinErmentroutNeuron`
Gutkin & Ermentrout 1998 — persistent Na + K minimal conductance.

Reference: Gutkin, B.S. & Ermentrout, G.B. (1998). Neural Comput. 10:1047–1065.

- **step**(current)
- **reset**()

---

## Module `neurons.models.hay_l5`

### Class `HayL5PyramidalNeuron`
Hay et al. 2011 — Layer 5 thick-tufted pyramidal cell (reduced).

Simplified 3-compartment reduction of the full morphological model.
Compartments: soma (Na/K fast), apical trunk (Ca/Ih), apical tuft (Ca/NMDA).
Reproduces BAC firing (backpropagation-activated calcium spike).

Original: Hay, Hill, Schurmann, Markram & Segev, PLoS Comput Biol 7(7), 2011.

Reference: Hay, E. et al. (2011). PLoS Comput. Biol. 7:e1002107.

- **step**(current_soma, current_tuft)
- **reset**()

---

## Module `neurons.models.hill_tononi`

### Class `HillTononiNeuron`
Hill & Tononi 2005 — thalamocortical sleep/wake model.

I_Na + I_K + I_h + I_T + I_KNa + I_L with Na-dependent K current.

Reference: Hill, S. & Tononi, G. (2005). J. Neurophysiol. 93:1671–1698.

- **step**(current)
- **reset**()

---

## Module `neurons.models.hindmarsh_rose`

### Class `HindmarshRoseNeuron`
Hindmarsh-Rose 1984 — 3D chaotic bursting model.

dx/dt = y - x³ + bx² - z + I
dy/dt = 1 - 5x² - y
dz/dt = r(s(x - x_rest) - z)

Reference: Hindmarsh, J.L. & Rose, R.M. (1984). Proc. R. Soc. Lond. B 221:87–102.

- **step**(current)
- **reset**()

---

## Module `neurons.models.hodgkin_huxley`

### Class `HodgkinHuxleyNeuron`
Hodgkin-Huxley 1952 — 4-ODE ion channel model.

C_m dv/dt = -g_Na m³h(v-E_Na) - g_K n⁴(v-E_K) - g_L(v-E_L) + I
dm/dt = α_m(1-m) - β_m·m
dh/dt = α_h(1-h) - β_h·h
dn/dt = α_n(1-n) - β_n·n

Reference: Hodgkin, A.L. & Huxley, A.F. (1952). J. Physiol. 117:500–544.

Integrator options:
- ``baseline_euler`` preserves the historical explicit-Euler sub-step path
- ``rk4`` is an explicit higher-order alternative path over the same
  sub-step schedule
- ``rosenbrock`` is a linearly implicit stiff-system path over the same
  Hodgkin-Huxley ODEs

- **__post_init__**()
- **_alpha_m**(v)
- **_beta_m**(v)
- **_alpha_h**(v)
- **_beta_h**(v)
- **_alpha_n**(v)
- **_beta_n**(v)
- **step**(current)
- **_rhs**(_t, state, current)
- **_step_baseline_euler**(current)
- **_step_rk4**(current)
- **_step_rosenbrock**(current)
- **reset**()

---

## Module `neurons.models.huber_braun`

### Class `HuberBraunNeuron`
Braun, Huber et al. 1998 — cold receptor, temperature-dependent.

4 ODEs: V, a_sd (slow depolarizing), a_sr (slow repolarizing), a_r.

Reference: Braun, H.A. et al. (1998). Int. J. Bifurcation Chaos 8:881–889.

- **step**(current)
- **reset**()

---

## Module `neurons.models.hybrid_linear_attention`

### Class `HybridLinearAttentionNeuron`
Hybrid linear attention spiking neuron.

Parameters
----------
dim : int
    Dimension of recurrent KV state. Default: 16.
lambda_decay : float
    Exponential decay for recurrent state. Default: 0.95.
window_size : int
    Sliding window size for local attention. Default: 16.

- **__post_init__**()
- **_phi**(x)
  - Feature map: elu(x) + 1.
- **step_qkv**(query, key, value)
  - Step with explicit query, key, value (scalar projections).
- **step**(current)
  - Simple step (input treated as combined qkv). Returns spike (0 or 1).
- **reset**()
  - Reset state to initial conditions.

---

## Module `neurons.models.ibarz_tanaka_map`

### Class `IbarzTanakaMapNeuron`
Ibarz et al. 2007 / Tanaka — piecewise-linear bursting map.

x(n+1) = f(x(n)) + y(n) + I
y(n+1) = y(n) - mu*(x(n) + 1) + mu*sigma

f(x) = alpha/(1-x)       if x <= 0
     = alpha + beta*x     if 0 < x < alpha+beta (spiking)
Reset x -> x_reset when x >= x_threshold.

Reference: Ibarz, B. et al. (2011). Phys. Rep. 501:1–74.

- **_f**(x)
- **step**(current)
- **reset**()

---

## Module `neurons.models.ih_neuron`

### Class `IhNeuron`
Ih (HCN) neuron — WB base + hyperpolarisation-activated cation current.

Ih activates upon hyperpolarisation and conducts a mixed Na⁺/K⁺
current with reversal ≈ -40 mV. Produces voltage sag, rebound
excitation, and pacemaker oscillations.

r∞ = 1 / (1 + exp((v + 80) / 10))
τ_r = 100 + 200 / (1 + exp((v + 70) / 10))

Reference: Robinson & Siegelbaum (2003) Annu Rev Physiol 65:453;
Pape (1996) Annu Rev Physiol 58:299.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.ilif`

### Class `InhibitoryLIFNeuron`
Inhibitory LIF — 2025, temporal inhibitory mechanism.

After spiking, a decaying inhibitory trace suppresses the membrane
for a learned duration, shaping temporal coding.

Reference: Fourcaud-Trocmé, N. et al. (2003). J. Neurosci. 23:11628–11640.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.inhomogeneous_poisson`

### Class `InhomogeneousPoissonNeuron`
Cox 1955 — doubly stochastic Poisson (time-varying rate).

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §7.3.

- **step**(rate_hz)
- **reset**()

---

## Module `neurons.models.iqif`

### Class `IntegerQIFNeuron`
Lo et al. 2021 — fixed-point quadratic integrate-and-fire.

V&#91;t+1&#93; = V&#91;t&#93; + (V&#91;t&#93;^2 >> k) + I, all integer arithmetic.

Reference: Izhikevich, E.M. (2007). Dynamical Systems in Neuroscience. MIT Press, §4.1.

- **step**(current)
- **reset**()

---

## Module `neurons.models.izhikevich2007`

### Class `Izhikevich2007Neuron`
Izhikevich 2007 biophysical quadratic integrate-and-fire neuron.

Equations from Izhikevich, E. M. (2007), *Dynamical Systems in
Neuroscience*, using the NeuroML 2 ``izhikevich2007Cell`` parameterisation:

``C dv/dt = k (v - vr) (v - vt) - u + I``
``du/dt = a (b (v - vr) - u)``

If ``v >= vpeak`` after integration, the neuron emits one spike and applies
``v <- c`` and ``u <- u + d``. Units are the NeuroML base units used by the
importer: pF, nS, mV, ms, and pA.

- **__post_init__**()
- **_require_finite**(name, value)
- **_require_positive**(cls, name, value)
- **_rhs**(v, u, input_current)
- **step**(input_current)
- **_step_euler**(input_current)
- **_step_rk4**(input_current)
- **_apply_threshold_reset**()
- **reset_state**()
- **get_state**()

---

## Module `neurons.models.jansen_rit`

### Class `JansenRitUnit`
Jansen & Rit 1995 — neural mass model for EEG generation.

6 ODEs: 3 populations (pyramidal, excitatory, inhibitory) x 2 states.

Reference: Jansen, B.H. & Rit, V.G. (1995). Biol. Cybern. 73:357–366.

- **_sigmoid**(x)
- **step**(p_ext)
- **reset**()

---

## Module `neurons.models.kilinc_bhatt_map_neuron`

### Class `KilincBhattMapNeuron`
Kilinc-Bhatt 2023 sigmoid map with adaptive threshold.

Minimal 2D map with built-in spike frequency adaptation via a slow
threshold variable. Designed for efficient hardware implementation
while retaining biologically relevant dynamics.

x(n+1) = -x(n) + k · σ(4·(x(n) - θ(n))) + I
θ(n+1) = β · θ(n) + γ · H(x(n) - θ_spike)

where σ(z) = 1 / (1 + exp(-z)), H() is Heaviside.

Reference: Kilinc & Bhatt (2023).

- **step**(current)
- **reset**()

---

## Module `neurons.models.klif`

### Class `KLIFNeuron`
KLIF — LIF with learnable scaling factor k.

V&#91;t+1&#93; = alpha * V&#91;t&#93; + k * I; spike when V >= threshold.
The scaling factor k is a trainable parameter for SNN backprop.

Reference: Eshraghian, J.K. et al. (2021). Proc. IEEE 109:935–950.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.lapicque`

### Class `LapicqueNeuron`
Lapicque 1907 — classical RC integrate-and-fire.

tau * dv/dt = -(v - v_rest) + R * I

Reference: Lapicque, L. (1907). J. Physiol. Pathol. Gén. 9:620–635.

- **step**(current)
- **reset**()

---

## Module `neurons.models.larter_breakspear`

### Class `LarterBreakspearNeuron`
Breakspear, Terry & Friston 2003 — neural mass with ion channels.

3 ODEs per node. Combines Wilson-Cowan population dynamics with
conductance-based ion channel kinetics for whole-brain modelling.
Used in The Virtual Brain (TVB) simulator.

Reference: Larter, R. et al. (1999). Chaos 9:795–804.; Breakspear, M. et al. (2003). Cereb. Cortex 13:189–202.

- **_m_ca**(v)
- **_m_na**(v)
- **_m_k**(v)
- **step**(coupling)
- **reset**()

---

## Module `neurons.models.leaky_compete_fire`

### Class `LeakyCompeteFireNeuron`
Oster, Douglas & Liu 2009 — winner-take-all with lateral inhibition.

Reference: Oster, M. et al. (2009). Neural Comput. 21(9):2437–2465.

- **__post_init__**()
- **step**(currents)
- **reset**()

---

## Module `neurons.models.lnm`

### Class `LearnableNeuronModel`
Jahns et al. 2025 — fully parameterized learnable neuron.

V&#91;t+1&#93; = alpha * V&#91;t&#93; + beta * I&#91;t&#93; + gamma * f(V&#91;t&#93;)
where alpha, beta, gamma are trainable scalars and f is a
learnable activation (here sigmoid).

Reference: Jolivet, R. et al. (2006). J. Comput. Neurosci. 21:35–45.

- **step**(current)
- **reset**()

---

## Module `neurons.models.loihi2`

### Class `Loihi2Neuron`
Intel Loihi 2, 2021 — programmable 3-state-variable neuron.

State variables (s1, s2, s3) with configurable decay, threshold,
and cross-coupling. Generalises CUBA, COBA, and Izhikevich on-chip.
All integer arithmetic with configurable bit-shift decays.

Reference: Intel Corp. (2021). Loihi 2 Neuromorphic Processor Technical Brief.

- **step**(weighted_input)
- **reset**()

---

## Module `neurons.models.loihi_cuba`

### Class `LoihiCUBANeuron`
Loihi CUBA LIF — Intel Loihi fixed-point neuron. Davies 2018.

Reference: Davies, M. et al. (2018). IEEE Micro 38:82–99.

- **step**(weighted_input)
- **reset**()

---

## Module `neurons.models.ltc`

### Class `LiquidTimeConstantNeuron`
Hasani et al. 2021 — liquid time-constant neuron.

dx/dt = -(1/tau(x,I)) * x + (1/tau(x,I)) * f(x,I)
where tau depends on input, making the neuron's time constant
adaptive and input-driven.

Reference: Hasani, R. et al. (2021). Proc. AAAI Conf. Artif. Intell. 35(9):7657–7666.

- **step**(current)
- **reset**()

---

## Module `neurons.models.lugaro_cell`

### Class `LugaroCell`
Cerebellar Lugaro cell — rare fusiform granular layer interneuron.

LIF with adaptation, serotonin (5-HT) modulation, depolarised leak
for spontaneous firing. Inhibits Golgi cells and molecular layer INs.

Reference: Dieudonné & Bhatt (2003) J Physiol 548:97;
Lainé & Bhatt (2007) Front Syst Neurosci 1:4.

- **with_serotonin**(cls, level)
- **step**(current)
- **reset**()

---

## Module `neurons.models.mainen_sejnowski`

### Class `MainenSejnowskiNeuron`
Mainen & Sejnowski 1996 — axonal Na spike initiation model.

2-compartment: soma (passive) + axon (active Na + K).
Axon initiates spike via fast Na kinetics; soma follows passively.
C_s dV_s/dt = -g_L(V_s - E_L) + gc(V_a - V_s) + I
C_a dV_a/dt = -I_Na - I_K + gc(V_s - V_a)

Reference: Mainen, Z.F. & Sejnowski, T.J. (1996). Nature 382:363–366.

- **step**(current)
- **reset**()

### Function `_safe_exp(x)`
---

## Module `neurons.models.marder_stg`

### Class `MarderSTGNeuron`
Marder & Selverston 1992 — stomatogastric ganglion neuron.

7 currents: I_Na, I_CaT, I_CaS, I_A, I_KCa, I_Kd, I_H, I_L.
LP-like model from the pyloric CPG.

Reference: Marder, E. & Calabrese, R.L. (1996). Physiol. Rev. 76:687–717.

- **_boltz**(v, v_half, k)
- **step**(current)
- **reset**()

---

## Module `neurons.models.martinotti_neuron`

### Class `MartinottiNeuron`
Martinotti cell — adapting interneuron targeting layer 1 apical dendrites.

Pospischil gating with Na⁺, K⁺, strong M-current (Kv7 for pronounced
adaptation), T-type Ca²⁺ (rebound), leak. Overlaps SST+ phenotype
but with stronger adaptation (higher g_m) and lower rheobase.

Reference: Silberberg & Markram (2007); Toledo-Rodriguez et al. (2005).

- **step**(current)
- **reset**()

---

## Module `neurons.models.mat`

### Class `MATNeuron`
Kobayashi 2009 — Multi-timescale Adaptive Threshold.

Reference: Kobayashi, R. et al. (2009). Front. Comput. Neurosci. 3:9.

- **step**(current)
- **reset**()

---

## Module `neurons.models.mcculloch_pitts`

### Class `McCullochPittsNeuron`
McCulloch & Pitts 1943 — binary threshold neuron.

y = 1 if sum(w_i * x_i) >= theta, else 0.

Reference: McCulloch, W.S. & Pitts, W. (1943). Bull. Math. Biophys. 5:115–133.

- **step**(weighted_input)
- **reset**()

---

## Module `neurons.models.mckean`

### Class `McKeanNeuron`
McKean 1970 — piecewise-linear FitzHugh-Nagumo caricature.

dv/dt = f(v) - w + I
dw/dt = epsilon * (v - gamma*w)

f(v) = -v          if v < a/2
     = v - a       if a/2 <= v < (1+a)/2
     = 1 - v       if v >= (1+a)/2

Reference: McKean, H.P. (1970). Adv. Math. 4:209–223.

- **_f**(v)
- **step**(current)
- **reset**()

---

## Module `neurons.models.medvedev_map`

### Class `MedvedevMapNeuron`
Medvedev 2005 — 1D piecewise-monotone spiking map.

Reference: Medvedev, G.S. (2005). Physica D 202:37–59.

- **step**(current)
- **reset**()

---

## Module `neurons.models.mihalas_niebur`

### Class `MihalasNieburNeuron`
Mihalas-Niebur Generalized IF — captures 20 spike patterns.

Mihalas & Niebur 2009. Multiple internal thresholds and
adaptation currents enable tonic/phasic/burst/accommodation patterns.

Reference: Mihalas, S. & Niebur, E. (2009). Neural Comput. 21:704–718.

- **step**(current)
- **reset**()

---

## Module `neurons.models.morris_lecar`

### Class `MorrisLecarNeuron`
Morris-Lecar 1981 — calcium-potassium oscillator.

C dv/dt = -g_Ca m_∞(v)(v-E_Ca) - g_K w(v-E_K) - g_L(v-E_L) + I
dw/dt = λ(v)(w_∞(v) - w)

Reference: Morris, C. & Lecar, H. (1981). Biophys. J. 35:193–213.

Integrator options:
- ``baseline_euler`` preserves the historical explicit-Euler path
- ``rk4`` is an explicit fourth-order path over the same Morris-Lecar ODEs
- ``rosenbrock`` is a linearly implicit stiff-system path over the same
  conductance equations

- **__post_init__**()
- **_m_inf**(v)
- **_w_inf**(v)
- **_lam**(v)
- **step**(current)
- **_rhs**(_t, state, current)
- **_step_baseline_euler**(current)
- **_step_rk4**(current)
- **_step_rosenbrock**(current)
- **reset**()

---

## Module `neurons.models.motor_unit`

### Class `MotorUnit`
Motor unit — alpha motor neuron + muscle fibre.

Each spike triggers a muscle twitch. Force output is summation of
overlapping twitches (rate coding). Twitch modelled as critically-
damped second-order: f(t) = A · (t/τ) · exp(1 - t/τ).

Reference: Fuglevand et al. (1993) J Neurophysiol 70(6);
Heckman & Enoka (2012) Compr Physiol 2(4).

- **slow**(cls)
  - Slow motor unit (type S): small, fatigue-resistant, low force.
- **fast**(cls)
  - Fast motor unit (type FF): large, fatigable, high force.
- **step**(drive)
- **reset**()

---

## Module `neurons.models.multicompartment_mcn`

### Class `MulticompartmentMCNNeuron`
Multi-compartment neuron (Spiking-WM, PNAS 2025).

Parameters
----------
tau : float
    Soma time constant. Default: 2.0.
tau_b : float
    Basal dendrite time constant. Default: 2.0.
tau_a : float
    Apical dendrite time constant. Default: 2.0.
g_ratio : float
    Basal-to-soma conductance ratio (g_B/g_L). Default: 1.0.
beta : float
    Sigmoid steepness for apical gating. Default: 1.0.
v_th : float
    Spike threshold. Default: 1.0.
dt : float
    Integration timestep. Default: 1.0.

- **_sigma**(x)
  - Sigmoid gating: sigma(x) = 1/(1 + exp(-beta*x)).
- **step_compartments**(x_basal, x_apical, i_soma)
  - Step with basal input, apical input, and direct somatic input.
- **step**(current)
  - Simple step: input goes to basal dendrite only.
- **reset**()
  - Reset state to initial conditions.

---

## Module `neurons.models.neurogrid`

### Class `NeuroGridNeuron`
Boahen 2014 — Neurogrid subthreshold analog 2-compartment.

Soma (v_s) and dendrite (v_d) coupled by conductance g_c.
Soma is LIF with exponential spike initiation.
Dendrite integrates synaptic input.

Reference: Benjamin, B.V. et al. (2014). Proc. IEEE 102:699–716.

- **step**(current)
- **reset**()

---

## Module `neurons.models.nlif`

### Class `NonlinearLIFNeuron`
Nonlinear LIF with cubic term. Touboul & Brette 2008.

C dV/dt = a*(V - V_rest)*(V - V_crit) - w + I
dw/dt = (b*(V - V_rest) - w) / tau_w

Reference: Brette, R. (2004). Neural Comput. 16:2263–2278.

- **step**(current)
- **reset**()

---

## Module `neurons.models.nmda_neuron`

### Class `NMDANeuron`
NMDA receptor neuron — WB base + NMDA-type glutamate receptor current.

NMDA receptors require both glutamate binding (modelled as input
current) AND membrane depolarisation (Mg²⁺ block removal). The
voltage-dependent Mg²⁺ block follows Jahr & Stevens (1990):

B(V) = 1 / (1 + &#91;Mg²⁺&#93;/3.57 · exp(-0.062 · V))

I_NMDA = g_NMDA · s_NMDA · B(V) · (V - E_NMDA)

Reference: Jahr & Stevens (1990) J Neurosci 10:1830;
Wang (1999) Neuron 22:409.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.non_resetting_lif`

### Class `NonResettingLIFNeuron`
Adaptive multi-timescale threshold (aMAT) variant — non-resetting LIF.

tau_m dV/dt = -(V - V_rest) + R*I
On spike: threshold rises by delta_theta, V does NOT reset.
dtheta/dt  = -(theta - theta_rest) / tau_theta

Kobayashi et al. 2009, Jolivet et al. 2004.

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §1.3.

- **step**(current)
- **reset**()

---

## Module `neurons.models.perfect_integrator`

### Class `PerfectIntegratorNeuron`
Non-leaky integrate-and-fire. Lapicque 1907 (no leak).

dV/dt = I / C

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §1.3.

- **step**(current)
- **reset**()

---

## Module `neurons.models.pernarowski`

### Class `PernarowskiNeuron`
Pernarowski 1994 — simplified pancreatic beta cell burster.

3 ODEs (V, w, z) with two slow variables. Captures square-wave
and parabolic bursting depending on parameters.

Reference: Pernarowski, M. (1994). SIAM J. Appl. Math. 54:814–832.

- **step**(current)
- **reset**()

---

## Module `neurons.models.persistent_na_neuron`

### Class `PersistentNaNeuron`
Persistent Na⁺ (INaP) neuron — WB base + non-inactivating Na⁺ current.

INaP activates at subthreshold voltages (-60 to -40 mV) and does not
inactivate, providing a sustained depolarising drive for subthreshold
oscillations, plateau potentials, and burst generation.

p∞ = 1 / (1 + exp(-(v + 48) / 5))
τ_p = 10 + 40 / (1 + ((v + 48) / 10)²)

Reference: Crill (1996) Annu Rev Physiol 58:349;
French et al. (1990) Neuroscience 42:363.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.pinsky_rinzel`

### Class `PinskyRinzelNeuron`
Pinsky-Rinzel 1994 — 2-compartment pyramidal cell.

Soma (fast Na/K) coupled to dendrite (Ca/KAHP) via gc.
Minimal model for burst generation in cortical pyramidal cells.

Reference: Pinsky, P.F. & Rinzel, J. (1994). J. Comput. Neurosci. 1:39–60.

- **step**(current_soma, current_dend)
- **reset**()

---

## Module `neurons.models.plant_r15`

### Class `PlantR15Neuron`
Plant 1981 — Aplysia R15 parabolic burster, 5 ODEs.

dV/dt  = (-I_Na - I_K - I_Ca - I_L - I_KCa + I_ext) / C
dm/dt  = (m_inf(V) - m) / tau_m(V)
dh/dt  = (h_inf(V) - h) / tau_h(V)
dn/dt  = (n_inf(V) - n) / tau_n(V)
dCa/dt = -k_Ca * I_Ca - Ca / tau_Ca

Reference: Plant, R.E. & Kim, M. (1976). Biophys. J. 16:227–244.

- **step**(current)
- **reset**()

---

## Module `neurons.models.plif`

### Class `ParametricLIFNeuron`
Fang et al. 2021 — Parametric LIF (PLIF) with learnable decay.

V(t+1) = alpha * V(t) * (1 - spike(t)) + I(t)
alpha  = sigmoid(a)    (learnable parameter)
spike  = Theta(V - threshold)

Reference: Fang, W. et al. (2021). Proc. AAAI Conf. Artif. Intell. 35(3):2661–2669.

- **alpha**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.poisson`

### Class `PoissonNeuron`
Poisson spike generator — stochastic firing at rate λ.

P(spike in dt) = λ · dt. Essential for input layer generation.

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §7.2.

- **__post_init__**()
- **step**(rate_override)
- **reset**()

---

## Module `neurons.models.pospischil`

### Class `PospischilNeuron`
Pospischil et al. 2008 — minimal HH for 5 cortical cell types.

C dV/dt = -I_Na - I_Kd - I_M - I_L + I_ext
I_M (slow K+) provides adaptation; its conductance distinguishes:
  RS (g_m=0.07), FS (g_m=0), IB (g_m=0.03), LTS (g_m=0.03+I_T).
Default parameters: RS cortical pyramidal.

Reference: Pospischil, M. et al. (2008). Biol. Cybern. 99:427–441.

- **step**(current)
- **reset**()

---

## Module `neurons.models.prescott`

### Class `PrescottNeuron`
Prescott 2008 — Type I/II/III excitability via M-current tuning.

Reference: Prescott, S.A. et al. (2008). PLoS Comput. Biol. 4:e1000198.

- **step**(current)
- **reset**()

---

## Module `neurons.models.psn`

### Class `ParallelSpikingNeuron`
Parallel Spiking Neuron — 2024, linear filter over all timesteps.

Applies a learned 1D convolution kernel over an internal buffer,
enabling non-causal temporal aggregation during training.
At each step: score = sum(kernel * buffer); spike if score >= threshold.

Reference: Comsa, I.-M. et al. (2020). Proc. ICLR 2020.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.pv_fast_spiking_neuron`

### Class `PVFastSpikingNeuron`
PV+ (parvalbumin) fast-spiking interneuron.

Wang-Buzsáki 1996 core extended with Kv3.1 for narrow APs and
high-frequency sustained firing (>200 Hz). No spike frequency
adaptation.

Reference: Wang & Buzsáki (1996) J Neurosci 16:6402–6413.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.quadratic_if`

### Class `QuadraticIFNeuron`
Quadratic Integrate-and-Fire — canonical Type-I excitability.

dv/dt = v² + I
Reset when v >= v_peak.

Reference: Latham, P.E. et al. (2000). J. Neurophysiol. 83:808–827.

- **step**(current)
- **reset**()

---

## Module `neurons.models.quantum_inspired_lif`

### Class `QuantumInspiredLIFNeuron`
Quantum-inspired LIF with complex amplitude and stochastic firing.

Parameters
----------
tau : float
    Membrane time constant (ms). Default: 20.0.
theta : float
    Firing threshold for |z|. Default: 1.0.
dt : float
    Integration timestep (ms). Default: 0.1.
v_reset : float
    Reset value for z_re and z_im after spike. Default: 0.0.
seed : int
    Initial RNG state for xorshift64. Default: 12345.

- **__post_init__**()
- **_xorshift64**()
  - Xorshift64 PRNG returning uniform in &#91;0, 1).
- **step_complex**(i_re, i_im)
  - Step with real and imaginary current components.
- **step**(current)
  - Step with real-only current (imaginary = 0).
- **reset**()
  - Reset state to initial conditions.

---

## Module `neurons.models.rall_cable`

### Class `RallCableNeuron`
Rall 1962 — N-compartment passive cable discretization.

Each compartment: C dV_i/dt = -g_L(V_i - E_L) + g_a(V_{i-1} - 2V_i + V_{i+1})
Soma is compartment 0; input injected at distal end (N-1).
Spike detection at soma.

Reference: Rall, W. (1959). Exp. Neurol. 1:491–527.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.renshaw_cell`

### Class `RenshawCell`
Renshaw cell — spinal inhibitory interneuron for recurrent inhibition.

WB gating core with strong adaptation to produce burst-then-decay
response to motor axon collateral input.

Reference: Renshaw (1941); Windhorst (1996) Prog Neurobiol 46(5).

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.resonate_and_fire`

### Class `ResonateAndFireNeuron`
Resonate-and-Fire — subthreshold oscillation + threshold.

Izhikevich 2001. Complex dynamics: z = x + i*y,
dz/dt = (b + iω)z + I, fire when |z| > threshold.
Implemented as 2 real ODEs.

Reference: Izhikevich, E.M. (2001). Neural Networks 14:883–894.

- **step**(current)
- **reset**()

---

## Module `neurons.models.rulkov_map`

### Class `RulkovMapNeuron`
Rulkov 2001 — discrete map-based neuron (no ODE, O(1) per step).

x&#91;n+1&#93; = f(x&#91;n&#93;, y&#91;n&#93;) + I
y&#91;n+1&#93; = y&#91;n&#93; - μ(x&#91;n&#93; + 1) + μσ
Fast iteration, exhibits spiking and bursting.

Reference: Rulkov, N.F. (2002). Phys. Rev. E 65:041922.

- **step**(current)
- **reset**()

---

## Module `neurons.models.sfa`

### Class `SFANeuron`
Benda & Herz 2003 — Spike Frequency Adaptation IF.

Reference: Benda, J. & Herz, A.V.M. (2003). Neural Comput. 15:2523–2564.

- **step**(current)
- **reset**()

---

## Module `neurons.models.sherman_rinzel_keizer`

### Class `ShermanRinzelKeizerNeuron`
Sherman, Rinzel & Keizer 1988 — pancreatic beta cell (reduced).

Reference: Sherman, A. et al. (1988). Biophys. J. 54:411–425.

- **step**(current)
- **reset**()

---

## Module `neurons.models.siegert`

### Class `SiegertTransferFunction`
Siegert 1951 — mean-field LIF firing rate.

Analytical stationary firing rate of a LIF neuron driven by
Gaussian white noise: r = &#91;tau_rp + tau_m * sqrt(pi) *
integral(exp(u^2)*(1+erf(u)), u_reset..u_thresh)&#93;^{-1}
Uses Gauss-Hermite quadrature approximation.

Reference: Siegert, A.J.F. (1951). Phys. Rev. 81:617–623.

- **step**(current)
  - Return instantaneous firing rate (Hz) for given mean input current.
- **reset**()

### Function `_erf_approx(x)`
Abramowitz & Stegun 7.1.26 rational approximation.

---

## Module `neurons.models.sigma_delta`

### Class `SigmaDeltaNeuron`
Yoon 2017 — event-driven sigma-delta encoding.

Reference: Yoon, Y.J. (2016). LIF and simplified SRM as APSDM. arXiv:1605.02226.

- **step**(current)
- **reset**()

---

## Module `neurons.models.sigmoid_rate`

### Class `SigmoidRateNeuron`
Continuous rate model with sigmoidal transfer. Wilson & Cowan 1972 style.

tau dr/dt = -r + sigma(beta * (input - theta))

Reference: Wilson, H.R. & Cowan, J.D. (1972). Biophys. J. 12:1–24.

- **step**(current)
- **reset**()

---

## Module `neurons.models.sk_neuron`

### Class `SKNeuron`
SK (Small Conductance Ca²⁺-Activated K⁺) channel neuron.

Wang-Buzsáki base extended with an SK (KCa2.x) current that depends
solely on intracellular Ca²⁺ (no voltage dependence). SK channels
have slower kinetics than BK and produce the medium
afterhyperpolarisation (mAHP) lasting 50–200 ms.

SK∞ = &#91;Ca²⁺&#93;² / (&#91;Ca²⁺&#93;² + 0.25)   (Hill function, n=2)
τ_Ca = 150 ms (slower than BK's 50 ms)

Reference: Stocker (2004) Nat Rev Neurosci 5:758–770;
Wang & Buzsáki (1996) base model.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.spike_response`

### Class `SpikeResponseNeuron`
Spike Response Model (SRM0) — kernel-based, no ODEs.

v(t) = η(t - t_last) + Σ κ(t - t_in) · w
Spike when v(t) ≥ threshold.
Gerstner 1995.

Reference: Gerstner, W. (1995). Neural Comput. 7:1049–1071.

- **step**(weighted_input)
- **reset**()

---

## Module `neurons.models.spinnaker2`

### Class `SpiNNaker2Neuron`
TU Dresden / SpiNNaker2 2024 — ARM Cortex-M4F software LIF.

Fixed-point LIF on M4F with exponential decay via integer multiply-shift.
Includes refractory counter and configurable precision.

Reference: Mayr, C. et al. (2019). Proc. DATE 2019: 1547–1552.

- **step**(current)
- **reset**()

---

## Module `neurons.models.spinnaker_lif`

### Class `SpiNNakerLIFNeuron`
SpiNNaker LIF — ARM Cortex-M4 digital. Furber 2014.

Reference: Furber, S.B. et al. (2014). Proc. IEEE 102:652–665.

- **step**(current)
- **reset**()

---

## Module `neurons.models.srm0`

### Class `SRM0Neuron`
Spike Response Model, zeroth order.

Reference: Gerstner, W. & Kistler, W.M. (2002). Spiking Neuron Models. Cambridge Univ. Press.

v(t) = eta(t - t_hat) + integral(kappa(t - s) * I(s) ds)

Simplified discrete version:
  eta decays after spike: eta(s) = -eta_reset * exp(-s / tau_eta)
  kappa integrates input: v += (I * R - v) * dt / tau_m + eta
  Spike when v > threshold.

Gerstner, W. & Kistler, W.M. (2002). Spiking Neuron Models.
Cambridge University Press. Ch. 4.

- **__post_init__**()
- **step**(current)
- **reset**()
- **get_state**()

---

## Module `neurons.models.sst_neuron`

### Class `SSTNeuron`
SST+ (somatostatin) low-threshold spiking interneuron.

Pospischil et al. 2008 LTS parameterisation with Na⁺, K⁺, M-current
(Kv7 for adaptation), T-type Ca²⁺ (rebound bursting), Ih (sag), leak.

Reference: Pospischil et al. (2008) Biol Cybern 99:427–441.

- **step**(current)
- **reset**()

---

## Module `neurons.models.stellate_cell`

### Class `StellateCell`
Cerebellar stellate cell — fast-spiking molecular layer interneuron.

WB Na⁺/K⁺ core + Kv3.1 for narrow APs. Feedforward inhibition onto
Purkinje cell dendrites. Smaller than basket cells.

Reference: Sultan & Bower (1999) J Comp Neurol 409:63;
Häusser & Clark (1997) Neuron 19:665.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.stochastic_if`

### Class `StochasticIFNeuron`
Brunel & Hakim 1999 — Ornstein-Uhlenbeck driven IF.

Reference: Tuckwell, H.C. (1988). Introduction to Theoretical Neurobiology, Vol. 2. Cambridge Univ. Press.

- **step**(current)
- **reset**()

---

## Module `neurons.models.superspike_neuron`

### Class `SuperSpikeNeuron`
Zenke & Ganguli 2018 — LIF with SuperSpike surrogate gradient.

Uses Van Rossum filtered eligibility traces and a smooth surrogate
gradient sigma'(V) = 1/(beta * |V - V_th| + 1)^2.

Reference: Zenke, F. & Ganguli, S. (2018). Neural Comput. 30:1514–1541.

- **__post_init__**()
- **surrogate_grad**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.tc_lif`

### Class `TwoCompartmentLIFNeuron`
Two-compartment LIF — Yang et al. AAAI 2024.

Soma:     tau_s dV_s/dt = -(V_s - V_rest) + kappa*(V_d - V_s) + I_ext
Dendrite: tau_d dV_d/dt = -(V_d - V_rest) + I_d
Spike when V_s >= theta; V_s -> V_reset, V_d unchanged.
Dendrite provides history-dependent input for sequential tasks.

Reference: Destexhe, A. et al. (1996). J. Comput. Neurosci. 3:19–46.

- **step**(i_soma, i_dend)
- **reset**()

---

## Module `neurons.models.terman_wang`

### Class `TermanWangOscillator`
Terman & Wang 1995 — relaxation oscillator for LEGION networks.

dv/dt = f(v) - w + I + rho
dw/dt = epsilon * (g(v) - w)

f(v) = 3*v - v^3 + 2                (cubic nullcline)
g(v) = alpha * (1 + tanh(v/beta))    (sigmoid recovery)

Reference: Terman, D. & Wang, D.L. (1995). Neural Comput. 7:507–517.

- **step**(current)
- **reset**()

---

## Module `neurons.models.theta`

### Class `ThetaNeuron`
Theta neuron — canonical Type-I on the unit circle.

dθ/dt = (1 - cos θ) + (1 + cos θ) · I
Spike when θ crosses π.
Ermentrout & Kopell 1986.

Reference: Ermentrout, G.B. & Kopell, N. (1986). SIAM J. Appl. Math. 46:233–253.

- **step**(current)
- **reset**()

---

## Module `neurons.models.threshold_linear_rate`

### Class `ThresholdLinearRateNeuron`
Threshold-linear (ReLU) rate neuron. Dayan & Abbott 2001.

r = gain * max(0, input - theta)

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §15.2.

- **step**(current)
- **reset**()

---

## Module `neurons.models.traub_miles`

### Class `TraubMilesNeuron`
Traub & Miles 1991 — reduced hippocampal CA3 pyramidal.

Reference: Traub, R.D. & Miles, R. (1991). Neuronal Networks of the Hippocampus. Cambridge Univ. Press.

- **step**(current)
- **reset**()

---

## Module `neurons.models.truenorth`

### Class `TrueNorthNeuron`
Merolla 2014 — IBM TrueNorth digital neuron.

Reference: Merolla, P.A. et al. (2014). Science 345:668–673.

- **step**(weighted_input)
- **reset**()

---

## Module `neurons.models.tsodyks_markram`

### Class `TsodyksMarkramNeuron`
Tsodyks & Markram 1997 — LIF with short-term synaptic plasticity.

LIF membrane: tau_m dV/dt = -(V - V_rest) + R*I_syn + R*I_ext
dx/dt = (1 - x)/tau_d - u*x*delta(spike_in)
du/dt = (U - u)/tau_f + U*(1-u)*delta(spike_in)
I_syn = A * u * x on presynaptic spike

Reference: Tsodyks, M. et al. (1998). Neural Comput. 10:821–835.

- **step**(current, presynaptic_spike)
- **reset**()

---

## Module `neurons.models.ttype_ca_neuron`

### Class `TTypeCaNeuron`
T-type Ca²⁺ (IT) neuron — WB base + low-voltage-activated Ca²⁺ current.

IT activates at subthreshold voltages (-65 to -50 mV) and inactivates
slowly. When de-inactivated by hyperpolarisation, IT produces a
low-threshold spike (LTS) that can trigger a burst of Na⁺ spikes.

m_T∞ = 1 / (1 + exp(-(v + 52) / 5))
s∞ = 1 / (1 + exp((v + 81) / 4))
τ_s = 30 + 100 / (1 + exp((v + 75) / 10))

Reference: Huguenard (1996) Annu Rev Physiol 58:329;
Destexhe et al. (1996) J Neurophysiol 76:2049.

- **step**(current)
- **reset**()

### Function `_safe_rate(a, vhalf, v, k, fallback)`
---

## Module `neurons.models.unipolar_brush_cell`

### Class `UnipolarBrushCell`
Unipolar brush cell (UBC) — excitatory vestibular cerebellum interneuron.

LIF with slow NMDA-like persistent current that prolongs mossy fibre
bursts into sustained granule cell activation. Giant 1:1 synapse.

Reference: Bhatt et al. (1994) J Comp Neurol 349:560;
Diana et al. (2007) J Neurosci 27:4374.

- **step**(current)
- **reset**()

---

## Module `neurons.models.upper_motor_neuron`

### Class `UpperMotorNeuron`
Upper motor neuron — layer 5 pyramidal cell, corticospinal projection.

Pospischil 2008 RS parameterisation with Na⁺, K⁺, M-current (Kv7
for adaptation) + high-threshold Ca²⁺ for dendritic Ca²⁺ spikes.

Reference: Pospischil et al. (2008) Biol Cybern 99:427–441;
Larkum (2013) Trends Neurosci 36(3).

- **step**(current)
- **reset**()

---

## Module `neurons.models.vip_neuron`

### Class `VIPNeuron`
VIP (vasoactive intestinal peptide) irregular-spiking interneuron.

Na⁺, K⁺, A-type K⁺ (Kv4 for accommodation), leak. High input
resistance, small soma. Disinhibitory role (inhibits SST+ and PV+).

Reference: Porter et al. (1998); Bhatt et al. (2019).

- **step**(current)
- **reset**()

---

## Module `neurons.models.wang_buzsaki`

### Class `WangBuzsakiNeuron`
Wang-Buzsáki 1996 — fast-spiking GABAergic interneuron.

3 ODEs. Simplified HH with only Na + K delayed rectifier.
Designed for gamma (30-80 Hz) oscillation modelling.

Reference: Wang, X.-J. & Buzsáki, G. (1996). J. Neurosci. 16:6402–6413.

- **step**(current)
- **reset**()

---

## Module `neurons.models.wendling`

### Class `WendlingNeuron`
Wendling et al. 2002 — extended Jansen-Rit with slow GABA_B inhibition.

10 ODEs: 4 populations (pyramidal, excitatory, fast inhibitory, slow
inhibitory) x 2 states each + 2 for slow inhibitory PSP.
Reproduces epileptiform EEG patterns.

Reference: Wendling, F. et al. (2002). Biol. Cybern. 86:97–108.

- **_sigmoid**(x)
- **step**(p_ext)
- **reset**()

---

## Module `neurons.models.wilson_cowan`

### Class `WilsonCowanUnit`
Wilson-Cowan 1972 — excitatory/inhibitory population rate model.

τ_e dE/dt = -E + S(w_ee·E - w_ei·I + I_ext)
τ_i dI/dt = -I + S(w_ie·E - w_ii·I)
S(x) = 1/(1 + exp(-a(x-θ))) - 1/(1 + exp(aθ))

Reference: Wilson, H.R. & Cowan, J.D. (1972). Biophys. J. 12:1–24.

- **_sigmoid**(x)
- **step**(ext_input)
- **reset**()

---

## Module `neurons.models.wilson_hr`

### Class `WilsonHRNeuron`
Wilson 1999 — polynomial cortical model.

dV/dt = -(17.81 + 47.71*V + 32.63*V^2)*(V - 0.55) - 26*R*(V + 0.92) + I
dR/dt = (-R + 1.35*V + 1.03) / tau_R

V in dimensionless units, spike at V > V_peak.

Reference: Wilson, H.R. (1999). Spikes, Decisions, and Actions. Oxford Univ. Press.

- **step**(current)
- **reset**()

---

## Module `neurons.models.wong_wang`

### Class `WongWangUnit`
Wong & Wang 2006 — reduced decision-making attractor model.

Reference: Wong, K.-F. & Wang, X.-J. (2006). J. Neurosci. 26:1314–1328.

- **_phi**(i_syn)
- **step**(stim1, stim2)
- **reset**()

---

## Module `neurons.models.yamada`

### Class `YamadaNeuron`
Yamada, Kashimori & Kambara 1989 — subcritical Hopf burster.

3 ODEs: V, n (fast K recovery), q (slow variable for bursting).
Exhibits square-wave bursting via slow modulation of a Hopf bifurcation.

Reference: Yamada, W.M. et al. (1989). In: Methods in Neuronal Modeling. MIT Press, pp. 97–133.

- **step**(current)
- **reset**()

---

## Module `neurons.sc_izhikevich`

### Class `SCIzhikevichNeuron`
Stochastic Izhikevich neuron (software-only).

Standard Izhikevich model (IEEE TNN 14(6), 2003):
v' = 0.04*v^2 + 5*v + 140 - u + I + noise
u' = a*(b*v - u)

When v >= 30 mV: spike, then v <- c, u <- u + d.

Example
-------
>>> neuron = SCIzhikevichNeuron(noise_std=0.0)
>>> spikes = &#91;neuron.step(10.0) for _ in range(100)&#93;
>>> sum(spikes) > 0  # regular spiking with I=10
True

Integrator options:
- ``baseline_half_euler`` preserves the historical two-half-step path
- ``rk4`` is an explicit higher-order alternative path

- **__post_init__**()
- **_require_finite**(name, value)
- **_require_positive**(cls, name, value)
- **_require_nonnegative**(cls, name, value)
- **step**(input_current)
- **_rhs**(v, u, input_current)
- **_apply_noise_and_threshold**()
- **_step_baseline_half_euler**(input_current)
- **_step_rk4**(input_current)
- **reset_state**()
- **get_state**()

---

## Module `neurons.schema_validator`

### Class `SchemaError`
A single validation error.

- **__init__**(level, message, section)
- **__repr__**()

### Function `validate_schema_dict(data, name)`
Validate a parsed schema dictionary.

Returns a list of SchemaError objects. Empty list = valid.

### Function `validate_schema(name)`
Validate a bundled schema by name.

Checks both TOML and JSON versions if they exist, and verifies parity.

### Function `validate_all_bundled()`
Validate all bundled schemas.

Returns a dict mapping schema name to its list of errors/warnings.

---

## Module `neurons.stochastic_lif`

### Class `StochasticLIFNeuron`
Discrete-time noisy leaky integrate-and-fire neuron.

dv/dt = -(v - v_rest) / tau_mem + R * I + noise

Parameters use normalised units (voltage &#91;0,1&#93;, time in ms).
Defaults from Gerstner & Kistler, *Spiking Neuron Models*, 2002.

Example
-------
>>> neuron = StochasticLIFNeuron(v_threshold=1.0, tau_mem=20.0, noise_std=0.0)
>>> spikes = &#91;neuron.step(1.5) for _ in range(50)&#93;
>>> sum(spikes) > 0
True
>>> neuron.get_state()  # membrane voltage + refractory counter
{'v': ..., 'refractory': 0}

Process a bitstream as input current:

>>> import numpy as np
>>> bits = np.array(&#91;1, 0, 1, 1, 0, 1, 0, 0&#93;, dtype=np.uint8)
>>> neuron.reset_state()
>>> out = neuron.process_bitstream(bits, input_scale=2.0)
>>> out.shape
(8,)

- **__post_init__**()
- **step**(input_current)
- **reset_state**()
- **get_state**()
- **process_bitstream**(input_bits, input_scale)
  - Process a bitstream (array of 0s and 1s) as input current.

---

## Module `neurons.universal_dsl`

### Class `UniversalNeuron`
Schema-driven neuron model — parallel meta-layer over existing models.

Does NOT replace hand-crafted model files.  Delegates simulation to
:class:`~sc_neurocore.neurons.equation_builder.EquationNeuron`, ensuring
all existing AST safety, integration methods, and compilation paths
remain available.

Parameters
----------
schema : dict
    Parsed schema dictionary (from :func:`load_schema`).
parameter_overrides : dict, optional
    Override default parameter values (e.g. for parameter sweeps).
dt_override : float, optional
    Override the schema's default timestep.
method_override : str, optional
    Override the integration method.

- **__init__**(schema)
  - Initialise from a validated schema dictionary.
- **from_schema**(cls, source)
  - Create a UniversalNeuron from a schema file or bundled name.
- **from_dict**(cls, schema)
  - Create from an in-memory schema dictionary.
- **step**()
  - Advance one timestep.  Keyword args are external inputs (e.g. I=10.0).
- **reset**()
  - Reset state to initial conditions.
- **state**()
  - Current state variable values.
- **name**()
  - Model name from metadata.
- **doi**()
  - DOI of the source publication, if available.
- **schema**()
  - Return a deep copy of the underlying schema.
- **extensions**()
  - Forward-compatible extension fields.
- **to_json**()
  - Export the model schema as JSON (Studio interchange format).
- **to_toml**()
  - Export the model schema as TOML (version control format).
- **to_equation_neuron**()
  - Return the underlying EquationNeuron instance.
- **to_verilog**(module_name)
  - Compile the model to Verilog via the equation compiler.
- **list_state_variables**()
  - Return the names of all state variables.
- **list_parameters**()
  - Return the names of all parameters.
- **list_equations**()
  - Return the ODE right-hand sides.
- **__repr__**()
  - Return a human-readable summary string.

### Function `_load_toml(path)`
Load a TOML file, using tomllib (3.11+) or tomli fallback.

### Function `_load_json(path)`
Load a JSON schema file.

### Function `load_schema(source)`
Load a neuron model schema from a TOML or JSON file.

Parameters
----------
source : str or Path
    Path to a ``.toml`` or ``.json`` schema file, or a bare name
    (e.g. ``"lif"``) which is resolved against the bundled schemas.

Returns
-------
dict
    Parsed schema dictionary.

Raises
------
FileNotFoundError
    If the schema file cannot be found.
ValueError
    If the schema version is unsupported.

### Function `schema_to_json(schema)`
Export a schema dictionary as a JSON string (for Studio interchange).

Parameters
----------
schema : dict
    A schema dictionary (as returned by :func:`load_schema`).

Returns
-------
str
    JSON string with 2-space indentation.

### Function `schema_to_toml(schema)`
Export a schema dictionary as a TOML string (for version control).

Parameters
----------
schema : dict
    A schema dictionary.

Returns
-------
str
    TOML-formatted string.

### Function `list_bundled_schemas()`
Return names of all bundled model schemas (without extensions).

Returns
-------
list of str
    Sorted list of available schema names.

---

## Module `nir_bridge.export`

### Function `_node_to_nir(name, node)`
Convert a single SC-NeuroCore node to its NIR equivalent.

### Function `to_nir(network, path)`
Export an SC-NeuroCore SCNetwork to NIR format.

Parameters
----------
network : SCNetwork
    The network to export.
path : str or Path, optional
    If provided, write the NIR graph to this file.

Returns
-------
nir.NIRGraph

---

## Module `nir_bridge.fpga_compiler`

### Class `SCNIRExternalInputManifestEntry`
Stable flattened input-bus layout entry for one external source.

- **as_dict**()
  - Return deterministic JSON-ready external input metadata.

### Class `NetworkCompilationResult`
All artefacts from a network-level FPGA compilation.

Attributes
----------
neuron_modules : dict&#91;str, str&#93;
    Mapping from neuron type to Verilog source.
weight_rom : str
    Weight ROM Verilog source.
top_module : str
    Top-level interconnect Verilog source.
module_name : str
    Top-level module name.
total_neurons : int
    Total neuron count.
total_synapses : int
    Total synapse count.
q_format : str
    Q-format label (e.g. ``"Q8.8"``).
interconnect : str
    ``"direct"`` or ``"aer"``.
warnings : list&#91;str&#93;
    Quantisation and compilation warnings.
scnir_document : SCNIRDocument
    SC-aware metadata document consumed by the compilation artefacts.
scnir_source_modules : dict&#91;str, str&#93;
    Concrete stochastic source HDL modules keyed by Verilog module name.
scnir_source_manifest : tuple&#91;SCNIRHDLSourceManifestEntry, ...&#93;
    Deterministic manifest mapping SC-NIR streams to source modules.
scnir_external_inputs : tuple&#91;SCNIRExternalInputManifestEntry, ...&#93;
    Deterministic flattened input-bus layout for external source names.
scnir_hierarchy_modules : dict&#91;str, str&#93;
    Standalone SC-NIR hierarchy boundary modules keyed by module name.


### Function `_require_homogeneous_param(values, label)`
Return the scalar value of a per-neuron parameter or fail closed.

### Function `_resolved_population_params(neuron_type, pop)`
Resolve population parameters without averaging per-neuron values.

### Function `_population_module_signature(pop)`
Build the exact parameter signature for shared module reuse.

### Function `_signed_hex(value, width)`
Emit a width-limited signed Verilog literal.

### Function `_ceil_log2_at_least_one(value)`
Return ceil(log2(value)) with a lower bound of 1.

### Function `_connection_sources_are_analogue(pop)`
Whether a population output should be routed as an analogue state.

### Function `_external_input_layout(conns, pop_by_name, pops)`
Assign stable flattened input-bus lanes to each external source name.

### Function `_external_input_manifest(graph)`
Return the flattened input-bus layout used by generated top-level RTL.

### Function `_scnir_stream_fragment(value)`
### Function `_scnir_connection_stream_id(src, dst)`
### Function `_hierarchy_weight_literals(document, qgraph)`
Return flattened quantised weight literals owned by hierarchy output ports.

### Function `_hierarchy_output_wires_by_stream(hierarchy)`
Return top-level hierarchy output wire names keyed by SC-NIR stream id.

### Function `_connection_has_thresholds(conn)`
Whether a connection carries explicit NIR Threshold metadata.

### Function `_normalise_connection_delay_steps(delay_steps, source_width, label)`
Return one validated delay value per source column.

### Function `_build_scnir_hierarchy_modules(document)`
Emit standalone boundary modules for preserved SC-NIR hierarchy instances.

### Function `_build_scnir_hierarchy_module(instance)`
Emit one synthesisable hierarchy boundary module from typed SC-NIR ports.

### Function `_hierarchy_port_declaration(port)`
### Function `_hierarchy_zero_literal(port)`
### Function `_build_scnir_hierarchy_instance_block(hierarchy)`
Emit top-level hierarchy contract instances for preserved SC-NIR boundaries.

### Function `_hierarchy_top_wire_declaration(wire_name, port)`
### Function `_hierarchy_weight_expr(hierarchy_output_wires, stream_id)`
### Function `_build_neuron_module(neuron_type, pop)`
Build a Verilog module for one canonical neuron type.

Uses the existing ``equation_compiler.compile_to_verilog()`` with
canonical ODE templates.

Parameters
----------
neuron_type : str
    Canonical neuron type (``"lif"``, ``"if"``, etc.).
pop : NeuronSpec
    Representative population (for parameter defaults).
data_width : int
    Fixed-point data width.
fraction : int
    Fractional bits.

Returns
-------
str
    Synthesisable Verilog module source.

### Function `_build_weight_rom(qgraph)`
Generate a combined weight ROM for all connections.

All connection weight matrices are flattened into a single ROM
addressed by a global index.  Each connection gets a base address
offset.

Parameters
----------
qgraph : QuantisedGraph
    Quantised graph with integer weight matrices.
data_width : int
    Weight data width.

Returns
-------
str
    Verilog weight ROM module source.

### Function `_build_top_direct(module_name, qgraph)`
Generate direct-wired per-neuron top-level interconnect.

Every neuron gets its own instance of the type-specific module.  NIR
affine weights are emitted explicitly in fixed-point arithmetic:

* external analogue inputs use ``(input * weight) >>> fraction``;
* analogue source populations use ``(v_out * weight) >>> fraction``;
* spiking source populations contribute their fixed-point weight on
  spikes and zero otherwise;
* all fan-in terms and biases accumulate in a widened signed accumulator
  before saturation back to the neuron input Q-format.

Parameters
----------
module_name : str
    Top-level module name.
qgraph : QuantisedGraph
    Quantised graph.
data_width : int
    Fixed-point data width.

Returns
-------
str
    Verilog top-level module source.

### Function `_build_top_aer(module_name, qgraph)`
Generate weighted event-bus top-level interconnect.

The emitted datapath keeps the same population instances as the direct
interconnect but routes spike-producing source populations through an
address-event fan-out block.  All active source spikes in a cycle contribute
their signed fixed-point weights to every destination accumulator, so
simultaneous events preserve the dense affine semantics of the NIR graph.
External analogue inputs and analogue source populations remain direct
fixed-point multiply-accumulate terms because they are not sparse events.

Parameters
----------
module_name : str
    Top-level module name.
qgraph : QuantisedGraph
    Quantised graph.
data_width : int
    Fixed-point data width.
fraction : int
    Fractional bits for analogue multiply downshift.

Returns
-------
str
    Verilog top-level module source.

### Function `compile_network_to_fpga(graph)`
Compile a NeuronGraph to synthesisable Verilog RTL.

End-to-end pipeline:

1. Quantise all parameters to the target Q-format.
2. Generate one Verilog module per unique neuron type.
3. Generate a combined weight ROM.
4. Generate a top-level interconnect module (direct or AER).

Parameters
----------
graph : NeuronGraph
    Network description (from ``from_scnetwork()``).
module_name : str
    Top-level Verilog module name.
data_width : int
    Fixed-point total width (16 for Q8.8, 32 for Q16.16).
fraction : int
    Fractional bits.
bitstream_length : int
    SC-NIR bitstream length metadata propagated into compilation artefacts.
source_kind : {"lfsr", "sobol"}
    Hardware stochastic source family materialised from SC-NIR metadata.
base_seed : int
    First deterministic source seed; stream index increments from this base.
target : str
    FPGA target for resource estimation hints.
online_learning : Mapping&#91;str, Mapping&#91;str, Any&#93;&#93; | None
    Optional validated per-weight-stream SC-NIR online-learning annotations,
    keyed by deterministic stream id such as ``"conn.src_to_dst.weight"``.

Returns
-------
NetworkCompilationResult
    All generated Verilog sources and compilation metadata.

Raises
------
ValueError
    If the graph is empty or contains unsupported neuron types.

---

## Module `nir_bridge.hardware_targets`

### Class `SCMappingConstraints`
SC-specific constraints used before lowering NIR graphs to a target.

- **to_dict**()
  - Return a JSON-serialisable representation.

### Class `NeuromorphicHardwareProfile`
NIR extension profile for a named neuromorphic target.

- **to_manifest**()
  - Return the profile in deterministic manifest form.

### Class `HardwareNoiseAnnotation`
Measured target noise that can be replayed in simulation.

- **to_dict**()
  - Return a JSON-serialisable noise annotation.

### Function `available_hardware_profiles()`
Return all known hardware profiles in deterministic order.

### Function `get_hardware_profile(target_id)`
Return one hardware profile by identifier.

### Function `build_nir_hardware_manifest(targets)`
Build a deterministic manifest for NIR hardware-extension planning.

### Function `build_noise_annotation(target_id, observations)`
Validate measured hardware noise and prepare it for simulation replay.

---

## Module `nir_bridge.neuromorphic_adapters`

### Class `NeuromorphicAdapterPackage`
Deterministic handoff package for one neuromorphic hardware target.

- **manifest**()
  - Return a JSON-serialisable adapter manifest.
- **files**()
  - Return deterministic package files keyed by relative path.

### Function `build_neuromorphic_adapter_package(source, target_id, config)`
Build one Loihi 2 or SpiNNaker2 adapter handoff package.

### Function `build_neuromorphic_adapter_bundle(source, targets, config)`
Build deterministic adapter packages for multiple targets.

### Function `write_neuromorphic_adapter_bundle(output_dir, source, targets, config)`
Write Loihi 2/SpiNNaker2 adapter manifests and reports to disk.

### Function `_normalise_adapter_target(target_id)`
### Function `_target_config(target, config)`
---

## Module `nir_bridge.neuron_graph`

### Class `NeuronSpec`
One neuron population (layer) in the compiled graph.

Attributes
----------
name : str
    Unique population identifier (matches the NIR node name).
neuron_type : str
    Canonical neuron type: ``"lif"``, ``"if"``, ``"li"``,
    ``"cuba_lif"``, ``"cuba_li"``.
n_neurons : int
    Number of neurons in this population.
params : dict&#91;str, np.ndarray&#93;
    Neuron parameters keyed by canonical names:
    ``tau``, ``r``, ``v_leak``, ``v_threshold``, ``v_reset``,
    ``tau_syn``, ``tau_mem``, ``w_in`` (type-dependent).
dt : float
    Simulation timestep used during NIR import.


### Class `ConnectionSpec`
Weighted edge between two neuron populations.

Attributes
----------
src : str
    Source population name.
dst : str
    Destination population name.
weights : np.ndarray
    Weight matrix of shape ``(n_dst, n_src)`` in float32.
    Row *i* contains the weights from all source neurons to
    destination neuron *i*.
bias : np.ndarray | None
    Optional bias vector of shape ``(n_dst,)``.
delay_steps : int | tuple&#91;int, ...&#93;
    Number of explicit unit-delay timesteps on this connection.  A scalar
    applies to all source columns; a tuple carries one delay per source
    column for heterogeneous NIR ``Delay`` vectors.
source_threshold : np.ndarray | None
    Optional threshold vector applied to source signals before the weight
    matrix.  Represents NIR ``Threshold`` on the source side.
destination_threshold : np.ndarray | None
    Optional threshold vector applied after this connection's affine
    accumulation and before the destination population input.


### Class `HierarchyInstanceSpec`
Flattened nested graph provenance preserved for SC-NIR hierarchy export.


### Class `NeuronGraph`
Complete network description ready for FPGA compilation.

Attributes
----------
populations : list&#91;NeuronSpec&#93;
    Ordered list of neuron populations (topological order).
connections : list&#91;ConnectionSpec&#93;
    Weighted connections between populations.
input_pop : str
    Name of the input population.
output_pop : str
    Name of the output population.
dt : float
    Global simulation timestep.
hierarchy : tuple&#91;HierarchyInstanceSpec, ...&#93;
    Nested NIR graph instances that were inlined for flat hardware lowering
    but must remain visible in SC-NIR hierarchy metadata.

- **total_neurons**()
  - Total neuron count across all populations.
- **total_synapses**()
  - Total synapse count across all connections.
- **neuron_types**()
  - Set of unique neuron types in the graph.
- **summary**()
  - Human-readable summary of the network graph.

### Function `_extract_neuron_params(node, neuron_type)`
Extract canonical parameters from an SC neuron node.

Parameters
----------
node : Any
    SC node instance (e.g. ``SCLIFNode``, ``SCCubaLIFNode``).
neuron_type : str
    Canonical neuron type string.

Returns
-------
dict&#91;str, np.ndarray&#93;
    Parameter dictionary with type-appropriate keys.

### Function `_topological_order(nodes, edges)`
Return a deterministic topological order for an already cycle-broken graph.

### Function `_hdl_identifier_fragment(value)`
Return a conservative Verilog identifier fragment for generated metadata.

### Function `_inline_single_port_subgraphs(network)`
Inline parser-executable single-port subgraphs for SC-NIR/FPGA lowering.

The runtime parser keeps nested NIR graphs as executable wrapper nodes.  HDL
lowering needs a single explicit dataflow graph, so this helper namespaces
each nested subgraph and rewires the outer edges through the nested boundary
nodes.  Multi-port subgraphs are accepted only when the parent graph supplies
an exact ordered one-edge-per-input and one-edge-per-output boundary mapping.

### Function `_delay_steps(node, node_name)`
Return scalar or per-source delay metadata for an explicit NIR Delay node.

### Function `_delay_steps_array(delay_steps)`
Return delay metadata as a one-dimensional integer array.

### Function `_compose_delay_steps(left, right)`
Compose adjacent delay nodes with scalar/vector broadcasting.

### Function `_fit_delay_steps_to_width(delay_steps, width, label)`
Validate scalar/vector delay metadata against a source width.

### Function `_scale_vector(node, node_name)`
Return a finite one-dimensional scale vector from an SCScaleNode.

### Function `_threshold_vector(node, node_name)`
Return a finite one-dimensional threshold vector from an SCThresholdNode.

### Function `_compose_scale(left, right)`
Compose adjacent scale vectors under NumPy broadcasting rules.

### Function `_broadcast_scale(scale, size, label)`
Broadcast a scalar/vector scale to ``size`` or fail closed.

### Function `_broadcast_threshold(threshold, size, label)`
Broadcast a scalar/vector threshold to ``size`` or fail closed.

### Function `_shape_width(shape)`
Return the element count for a finite NIR shape or fail closed.

### Function `_flatten_widths(node, node_name)`
Return input/output element counts for a shape-typed SCFlattenNode.

### Function `_conv1d_to_dense_matrix(node, node_name)`
Lower a shape-known NIR Conv1d node to an exact dense matrix.

### Function `_conv2d_to_dense_matrix(node, node_name)`
Lower a shape-known NIR Conv2d node to an exact dense matrix.

### Function `_pool2d_to_dense_matrix(node, node_name)`
Lower a shape-known NIR Pool2d node to an exact dense matrix.

### Function `_weight_matrix_and_bias(node, node_name)`
Return dense weight and bias arrays for a weight-carrying NIR node.

### Function `_node_logical_width(node)`
Return the flattened channel width for a source/destination node.

### Function `_resolve_weight_source(node_name)`
Resolve the population/input source feeding a weight node.

Traverses pass-through nodes immediately upstream of ``Affine``/``Linear``
nodes and accumulates explicit NIR Delay metadata.  Ambiguous fan-in fails
closed so the compiler does not invent a source for hardware handoff.

### Function `_resolve_weight_destination(node_name)`
Resolve the neuron destination fed by a weight node.

Traverses pass-through nodes immediately downstream of ``Affine``/``Linear``
and accumulates post-weight Scale metadata.  The scale is later folded into
connection rows and bias terms.

### Function `_fold_connection_scales(weights, bias)`
Fold adjacent Scale nodes into a connection's weights and bias.

### Function `from_scnetwork(network, dt)`
Convert a parsed SCNetwork to a NeuronGraph for FPGA compilation.

Walks the topologically-sorted node list and partitions nodes into
neuron populations and weighted connections.  Pass-through nodes
(Input, Output, Scale, Flatten, Threshold) are folded into the
adjacent edges.

Parameters
----------
network : SCNetwork
    A parsed SC-NeuroCore network (from ``from_nir()``).
dt : float, optional
    Override the simulation timestep.  If ``None``, uses the
    timestep stored in the network's neuron nodes.

Returns
-------
NeuronGraph
    Network description ready for FPGA compilation.

Raises
------
ValueError
    If the network contains no neuron populations or no connections.

---

## Module `nir_bridge.node_map`

### Class `SCInputNode`
Graph entry point — passes input through unchanged.

- **forward**(x)

### Class `SCOutputNode`
Graph exit point — collects output.

- **forward**(x)

### Class `SCLIFNode`
LIF neuron mapped from NIR LIF primitive.

NIR LIF: tau*dv/dt = (v_leak - v) + R*I, spike when v > v_threshold
Euler: v += ((v_leak - v) + R*I) * dt/tau

- **from_nir**(cls, name, node, dt, reset_mode)
- **__post_init__**()
- **_broadcast_to**(size)
- **forward**(x)
- **reset**()

### Class `SCIFNode`
IF neuron — integrator with threshold, no leak.

NIR IF: dv/dt = R*I, spike when v > v_threshold
Euler: v += R*I*dt

- **from_nir**(cls, name, node, dt, reset_mode)
- **__post_init__**()
- **_broadcast_to**(size)
- **forward**(x)
- **reset**()

### Class `SCLINode`
Leaky integrator — LIF without threshold.

NIR LI: tau*dv/dt = (v_leak - v) + R*I

- **from_nir**(cls, name, node, dt)
- **__post_init__**()
- **_broadcast_to**(size)
- **forward**(x)
- **reset**()

### Class `SCAffineNode`
Dense linear transform with bias: y = Wx + b

- **from_nir**(cls, name, node)
- **forward**(x)

### Class `SCLinearNode`
Matrix multiply without bias: y = Wx

- **from_nir**(cls, name, node)
- **forward**(x)

### Class `SCScaleNode`
Element-wise scaling: y = s * x

- **from_nir**(cls, name, node)
- **forward**(x)

### Class `SCThresholdNode`
Spike threshold: y = 1 if x > threshold else 0

- **from_nir**(cls, name, node)
- **forward**(x)

### Class `SCFlattenNode`
Reshape tensor — flatten dimensions.

- **from_nir**(cls, name, node)
- **forward**(x)

### Class `SCIntegratorNode`
Pure integrator: dv/dt = R*I (no leak, no threshold). Euler: v += R*I*dt

- **from_nir**(cls, name, node, dt)
- **n_neurons**()
  - Number of integrator state channels.
- **__post_init__**()
- **forward**(x)
- **reset**()

### Class `SCDelayNode`
Temporal delay: output = input(t - delay).

NIR Delay: I(t - tau). Implemented as a circular buffer per element.
Delay values are rounded to integer timesteps.

- **from_nir**(cls, name, node, dt)
- **__post_init__**()
- **forward**(x)
- **reset**()

### Class `SCCubaLIFNode`
Current-based LIF with synaptic filter.

NIR CubaLIF: tau_syn * dI_syn/dt = -I_syn + w_in * I
             tau_mem * dv/dt = (v_leak - v) + R * I_syn
             spike when v > v_threshold, reset to v_reset

- **from_nir**(cls, name, node, dt, reset_mode)
- **__post_init__**()
- **_broadcast_to**(size)
  - Broadcast scalar params to match actual input size.
- **forward**(x)
- **reset**()

### Class `SCCubaLINode`
Current-based leaky integrator (CubaLIF without threshold).

NIR CubaLI: tau_syn * dI_syn/dt = -I_syn + w_in * I
            tau_mem * dv/dt = (v_leak - v) + R * I_syn

- **from_nir**(cls, name, node, dt)
- **__post_init__**()
- **_broadcast_to**(size)
- **forward**(x)
- **reset**()

### Class `SCSumPool2dNode`
2D sum pooling: sum over spatial kernel windows.

- **from_nir**(cls, name, node)
- **forward**(x)

### Class `SCAvgPool2dNode`
2D average pooling: SumPool / kernel_area.

- **from_nir**(cls, name, node)
- **forward**(x)

### Class `SCConv1dNode`
1D convolution: y = conv1d(x, weight) + bias.

- **from_nir**(cls, name, node)
- **forward**(x)

### Class `SCConv2dNode`
2D convolution: y = conv2d(x, weight) + bias.

- **from_nir**(cls, name, node)
- **forward**(x)

### Function `_shape_tuple_from_type(type_map, key)`
Return a positive integer shape tuple from a NIR type map.

### Function `_shape3_tuple_from_type(type_map, key)`
Return a rank-3 positive integer shape tuple from a NIR type map.

### Function `map_node(name, node)`
Convert a single NIR node to its SC-NeuroCore equivalent.

---

## Module `nir_bridge.parser`

### Class `_UnitDelayNode`
Implicit unit-delay inserted on recurrent (back) edges.

Acts as a DAG source: outputs the previous timestep's buffered value.
Buffer is updated externally by SCNetwork.step() after execution.

- **forward**(x)
- **update_buffer**(value)
- **reset**()

### Class `SCSubgraphNode`
Executable wrapper for a nested NIR subgraph (single I/O port).

- **__post_init__**()
- **forward**(x)
- **reset**()

### Class `SCMultiPortSubgraphNode`
Executable wrapper for a nested NIR subgraph with multiple I/O ports.

Supports modular architectures where subgraphs expose multiple named
inputs and outputs (e.g., encoder-decoder, skip connections).

- **__post_init__**()
- **input_ports**()
- **output_ports**()
- **forward**(x)
  - Single-input convenience: feeds x to first input, returns first output.
- **forward_multi**(inputs)
  - Multi-port forward: provide named inputs, get named outputs.
- **reset**()

### Class `SCNetwork`
Executable network parsed from a NIR graph.

Nodes are stored by name. Edges define the forward pass order.
Calling ``run()`` feeds input through the graph for the given
number of timesteps and returns the output node's accumulated result.

Recurrent edges (cycles) are automatically handled by inserting
unit-delay nodes that feed from the previous timestep.

- **from_nir**(cls, source, dt, reset_mode)
  - Build an ``SCNetwork`` directly from a NIR graph or file path.
- **to_hardware**()
  - Compile this parsed network to the existing FPGA artefact bundle.
- **_find_back_edges**()
  - DFS-based back-edge detection.
- **_break_cycles**()
  - Replace back edges with unit-delay source nodes.
- **_topological_sort**()
  - Kahn's algorithm with automatic cycle breaking via delay nodes.
- **topo_order**()
- **step**(inputs)
  - Execute one timestep through the graph.
- **run**(inputs, steps)
  - Run the network for multiple timesteps.
- **reset**()
  - Reset all stateful nodes.
- **summary**()
  - Human-readable network summary.

### Function `from_nir(source, dt, reset_mode)`
Convert a NIR graph to an executable SC-NeuroCore network.

Parameters
----------
source : nir.NIRGraph or str or Path
    NIR graph object, or path to a .nir file.
dt : float
    Timestep for leaky integrator dynamics.
reset_mode : str
    Spike reset mechanism: "reset" (v = v_reset, NIR spec default)
    or "subtract" (v = v - v_threshold, used by snnTorch).

Returns
-------
SCNetwork
    Executable network with topologically sorted forward pass.

### Function `_validate_import_options(dt, reset_mode)`
### Function `_read_nir_file(source)`
### Function `_validate_nir_graph_boundary(graph, context)`
### Function `_parse_graph(graph, dt, reset_mode)`
Recursively parse a NIR graph into an SCNetwork.

---

## Module `nir_bridge.quantise_params`

### Class `QuantisedGraph`
NeuronGraph with all parameters converted to Q-format integers.

Attributes
----------
populations : list&#91;NeuronSpec&#93;
    Populations with integer-valued parameters (Q-encoded).
connections : list&#91;ConnectionSpec&#93;
    Connections with integer-valued weight matrices (Q-encoded).
q : Q88
    The fixed-point format configuration used.
input_pop : str
    Input population name.
output_pop : str
    Output population name.
dt : float
    Global timestep.
warnings : list&#91;str&#93;
    Overflow/underflow warnings generated during quantisation.
total_neurons : int
    Total neuron count.
total_synapses : int
    Total synapse count.


### Function `_quantise_array(arr, q, label, warnings)`
Quantise a float array to Q-format integers with clamping.

Parameters
----------
arr : np.ndarray
    Float values to quantise.
q : Q88
    Fixed-point format.
label : str
    Label for warning messages.
warnings : list&#91;str&#93;
    Accumulator for overflow/underflow warnings.

Returns
-------
np.ndarray
    Integer array of Q-encoded values (dtype int64).

### Function `_check_dt_quantisation(dt, q, warnings)`
Verify that the timestep survives Q-format quantisation.

Parameters
----------
dt : float
    Simulation timestep.
q : Q88
    Fixed-point format.
warnings : list&#91;str&#93;
    Accumulator for warnings.

### Function `quantise_graph(graph, q)`
Convert all floating-point parameters to Q-format integers.

Parameters
----------
graph : NeuronGraph
    Network with float32 parameters.
q : Q88
    Target fixed-point format.

Returns
-------
QuantisedGraph
    Network with integer-valued parameters and quantisation warnings.

---

## Module `nir_bridge.silicon_mapping`

### Class `SiliconMappingConfig`
Configuration for NIR silicon mapping report generation.

- **__post_init__**()

### Class `_GraphView`

### Function `build_silicon_mapping_report(source, config)`
Build a deterministic target-mapping report for a parsed NIR network.

### Function `write_silicon_mapping_report(output_dir, source, config)`
Write `nir_silicon_mapping_report.json` in deterministic form.

### Function `_coerce_graph(source)`
### Function `_node_payload(name, node)`
### Function `_node_type(node)`
### Function `_resource_estimate(node, node_type)`
### Function `_target_report(target_id, node_payloads, edges, config)`
### Function `_lowering_result(node_type, supported, unsupported, native_bitstream)`
### Function `_summary(mapped_nodes, edges, native_bitstream)`
### Function `_status(summary)`
### Function `_fallback_requirements(mapped_nodes, native_bitstream)`
### Function `_noise_hooks(channels)`
### Function `_mapping_or_attr(node, key)`
### Function `_int_attr(node, key, default)`
### Function `_shape(value)`
### Function `_shape_size(shape)`
### Function `_shape_product(value)`
---

## Module `online_learning.eprop`

### Class `EpropTrainer`
E-prop online trainer for a single-layer recurrent SNN.

Parameters
----------
n_inputs : int
    Input dimension.
n_neurons : int
    Number of LIF neurons.
n_outputs : int
    Output dimension.
tau_mem : float
    Membrane time constant (ms).
tau_trace : float
    Eligibility trace decay time constant (ms).
threshold : float
    Spike threshold.
lr : float
    Learning rate.
dt : float
    Timestep (ms).

- **__post_init__**()
- **reset**()
  - Reset all internal state and eligibility traces.
- **step**(x, target)
  - Process one timestep with optional learning.
- **train_sequence**(inputs, targets)
  - Train on one sequence, return mean loss.
- **predict_sequence**(inputs)
  - Run inference on a sequence without learning.
- **memory_per_step**()
  - Memory usage per timestep in parameters (O(1) in T).

---

## Module `online_learning.online_trainer`

### Class `OnlineLIFLayer`
Single LIF layer with online (eligibility-based) learning.

Parameters
----------
n_inputs : int
n_neurons : int
tau_mem : float
    Membrane time constant.
threshold : float
lr : float
    Learning rate for local weight updates.

- **__post_init__**()
- **reset**()
- **step**(x)
  - Forward one timestep. Returns spike vector.
- **apply_learning_signal**(signal)
  - Apply a top-down learning signal to update weights.

### Class `OnlineTrainer`
Feedforward online trainer: stacks OnlineLIFLayers with eligibility learning.

Parameters
----------
layer_sizes : list of int
    &#91;n_input, n_hidden1, ..., n_output&#93;
tau_mem : float
threshold : float
lr : float

- **__post_init__**()
- **reset**()
- **step**(x, target)
  - Forward one timestep through all layers with optional learning.
- **train_sequence**(inputs, targets)
  - Train on one sequence, return mean loss.
- **n_layers**()
- **memory_per_step**()
  - Total parameters stored per timestep (O(1) in T).

---

## Module `optics.photonic_emitter`

### Class `PhotonicEmitter`
Industrial-grade Photonic Emitter.
Uses topological sorting to ensure optical ports are defined before being coupled.

- **__init__**(target_pdk)
- **_topological_sort**(nodes)
- **emit_lumerical_netlist**(ir_graph)

### Class `OpticalModulation`
Optical modulation scheme.


### Class `PhotonicTarget`
Hardware target specification for a photonic backend.

- **lightmatter**(cls)
- **silicon_photonics**(cls)
- **two_d_waveguide**(cls)

### Class `OpticalPulse`
Single optical pulse with phase and amplitude.


### Class `BitstreamToOptical`
Converts SC bitstreams into optical pulse trains.

- **__init__**(target)
- **convert**(bitstream, pulse_duration_ps)
  - Map a boolean SC bitstream to an optical pulse train.
- **to_phase_array**(bitstream)
  - Vectorised phase extraction.
- **to_amplitude_array**(bitstream)
  - Vectorised amplitude extraction.
- **optical_power_profile**(bitstream, input_power_mw)
  - Compute output power profile accounting for insertion loss.

### Class `FDTDSolver`
1D finite-difference time-domain solver for waveguide co-simulation.

Solves Maxwell's equations on a 1D grid with Yee discretisation.
Suitable for verifying pulse propagation, dispersion, and loss in
simple waveguide geometries.

Boundary condition: a quadratic-ramp multiplicative **absorbing
boundary** at each end (not a Berenger split-field PML; 1D does not
require the σ-matched split formulation because there is no transverse
dimension into which energy could scatter). Field amplitude is tapered
by ``s_i = 1 − 0.8·((N−i)/N)²`` for the outermost ``N`` cells on each
side. Reflection is < −30 dB for wavelengths much smaller than the
boundary depth; for higher-fidelity absorption use :class:`FDTD2DSolver`
which implements full split-field Berenger PML.

- **__init__**(grid_size, dx_um, dt_factor, refractive_index, boundary_cells)
- **set_loss**(loss_db_per_cm)
  - Set propagation loss.
- **inject_pulse**(position, wavelength_nm, amplitude, phase)
  - Inject a Gaussian-envelope optical pulse at a grid position.
- **step**(n_steps)
  - Advance the simulation by *n_steps* timesteps.
- **field_energy**()
  - Total electromagnetic energy in the grid.
- **snapshot**()
  - Return a copy of the current E and H fields.

### Class `CompilationResult`
Result of a photonic compilation pass.

- **to_gdsii**(filename, mzi_length_um, pitch_um)
  - Export the compiled MZI cascade to a GDSII file via gdsfactory.

### Class `PhotonicCompiler`
End-to-end compiler: SC IR → optical mapping → netlist → co-sim.

- **__init__**(target)
- **compile_bitstream**(bitstream, run_fdtd, fdtd_steps)
  - Compile a single SC bitstream to a photonic deployment.
- **generate_mzi_verilog**(bit_width)
  - Generate SystemVerilog for an MZI modulator.
- **generate_microring_verilog**(bit_width)
  - Generate SystemVerilog for a microring resonator.

### Class `FDTD2DSolver`
2D finite-difference time-domain solver (TE mode).

Yee grid with optional PML absorbing boundaries.
Solves for Ez, Hx, Hy on a 2D cross-section of the photonic chip.

- **__init__**(nx, ny, dx_um, dy_um, dt_factor, pml_layers)
- **_build_pml**()
  - Construct Berenger PML conductivity profiles.
- **set_waveguide**(y_center, width_cells, refractive_index, x_start, x_end)
  - Define a horizontal waveguide stripe.
- **inject_source**(x, y, wavelength_nm, amplitude, sigma_cells)
  - Inject a 2D Gaussian source.
- **step**(n_steps)
  - Advance TE simulation by n_steps.
- **field_energy**()
  - Total EM energy in the grid.
- **field_at_point**(x, y)
  - Read Ez at a grid point.
- **cross_section**(x)
  - Return Ez cross-section at a given x position.
- **snapshot**()
  - Return copies of all field components.

### Class `MeepAdapter`
Optional adapter for the Meep FDTD solver.

Provides a thin wrapper that translates SC-NeuroCore photonic
configurations into Meep simulation objects. Meep is an optional
dependency; all methods gracefully degrade if not installed.

- **is_available**()
  - Check if Meep is installed.
- **build_waveguide_geometry**(target, waveguide_width_um, length_um, substrate_index)
  - Build a Meep geometry dict (usable even without Meep).
- **run_simulation**(geometry, run_time)
  - Run a Meep simulation or return a mock result if unavailable.

### Class `WaveguidePair`
A pair of adjacent waveguides for crosstalk analysis.

- **effective_index_diff**()
  - Approximate effective index difference between even/odd modes.
- **coupling_coefficient**()
  - Coupling coefficient κ (per micrometre).
- **coupling_ratio**()
  - Power coupling ratio at the end of the coupling length.
- **isolation_db**()
  - Isolation in dB (lower coupling = better isolation).

### Class `CrosstalkModel`
Models evanescent crosstalk between adjacent waveguides.

Uses coupled-mode theory to compute power transfer between
parallel waveguide runs on a photonic chip.

- **__init__**()
- **add_pair**(pair)
- **transfer_matrix**(pair)
  - 2×2 transfer matrix for a directional coupler.
- **compute_crosstalk**(pair, input_power)
  - Compute output power on both waveguides.
- **worst_case_isolation**()
  - Worst-case isolation across all pairs (minimum dB).
- **analyze_bank**(waveguides, gap_nm, coupling_length_um, wavelength_nm, core_index, cladding_index)
  - Geometric crosstalk for a uniform bank of parallel waveguides.
- **analyze_pairs**(pair_indices, gaps_nm, coupling_lengths_um, wavelength_nm, core_index, cladding_index)
  - Per-pair crosstalk for arbitrary waveguide geometry — the O(N²) path.

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

## Module `optimizer.feedback_loop`

### Class `SynthesisFeedbackResult`
Result of one measured-evidence optimiser feedback pass.


### Function `optimise_from_synthesis_reports()`
Parse synthesis reports and immediately rerun the SC optimiser.

This helper is the local closed loop for the first production path:
report files are parsed into strict evidence, evidence becomes measured
observations, and those observations bias the surrogate optimiser for the
supplied layer network.  It never invokes vendor tools or fabricates
missing metrics; callers must provide reports and measured accuracy.

### Function `optimise_from_evidence_payload()`
Rerun the SC optimiser from an in-memory evidence payload.

---

## Module `optimizer.observation_loader`

### Class `ObservationLoadError`
Raised when a benchmark/synthesis observation cannot be trusted.


### Function `load_observations(path)`
Load benchmark observations from a JSON evidence file.

### Function `load_synthesis_observation(report_paths)`
Load one observation from Vivado/Quartus report files plus design metadata.

Raw vendor reports do not describe the compiler decision that produced the
hardware, and many do not carry model accuracy.  The caller must therefore
provide the design fields and measured accuracy explicitly.

### Function `observation_from_synthesis_reports(reports)`
Build one observation from raw Vivado/Quartus text reports.

### Function `observations_from_payload(payload)`
Convert an in-memory benchmark/synthesis payload into observations.

### Function `_metrics_from_synthesis_reports(reports)`
### Function `_first_numeric_match(text, patterns)`
### Function `_first_power_mw(text)`
### Function `_extract_records(payload)`
### Function `_observation_from_record(record)`
### Function `_merged_views(record)`
### Function `_mapping(value)`
### Function `_required_int(view, key, source, index)`
### Function `_required_int_any(view, keys, source, index)`
### Function `_required_float_any(view, keys, source, index)`
### Function `_required_str(view, key, source, index)`
### Function `_required(view, key, source, index)`
### Function `_required_any(view, keys, source, index)`
### Function `_to_int(value, key, source, index)`
### Function `_to_float(value, key, source, index)`
---

## Module `optimizer.resource_optimizer`

### Class `OptimizationStep`
One step in the optimization process.


### Class `OptimizationResult`
Result of the resource optimization process.

- **summary**()

### Function `fit_to_target(layer_sizes, weights, target, max_iterations, min_bitstream_length, initial_bitstream_length)`
Automatically compress an SNN to fit a target FPGA.

Iteratively applies:
1. Bitstream length reduction (halving L)
2. Weight pruning (increasing threshold)
3. Weight quantization (reducing bit width)

Stops when the energy estimator says the network fits on the target.

Parameters
----------
layer_sizes : list of (n_inputs, n_neurons)
weights : list of ndarray
target : str
    FPGA target ('ice40', 'ecp5', 'artix7', 'zynq').
max_iterations : int
    Maximum optimization steps.
min_bitstream_length : int
    Minimum allowed L.
initial_bitstream_length : int
    Starting bitstream length.

Returns
-------
OptimizationResult

---

## Module `optimizer.sc_optimizer`

### Class `DecorrelationStrategy`

### Class `ComputeMode`

### Class `HardwareBudget`

### Class `LayerProfile`

### Class `LayerConfig`

### Class `OptimizerReport`
- **summary**()

### Class `SCOptimizer`
- **__init__**(budget)
- **_estimate_resources**(mac_count, length, decorr, mode)
  - Returns (LUTs, Power_mW, Accuracy_score, Latency_cycles).
- **_generate_candidates**(layer)
- **_is_feasible**(config)
- **_score**(config, network)
- **_build_report**(config, network, pareto)
- **optimize**(network)
  - Greedy knapsack optimization maximizing weighted accuracy.
- **optimize_annealing**(network)
  - Simulated annealing for larger design spaces.
- **_optimize_annealing_rust**(network)
  - Rust-accelerated SA path.
- **_optimize_annealing_python**(network)
  - Pure-Python SA fallback.
- **_extract_pareto**(points)
  - Extract non-dominated Pareto frontier from (luts, power, accuracy) tuples.

---

## Module `optimizer.surrogate_sc_optimizer`

### Class `TargetHardwareProfile`
Target device budget and compiler preference weights.


### Class `BenchmarkObservation`
Measured or externally supplied design-point observation.

The optimiser treats these as higher-priority training points than its
analytical generated points.  Callers should only pass observations that
come from real benchmark or synthesis outputs.


### Class `SurrogateLayerConfig`
Selected SC compiler settings for one layer.


### Class `SurrogateOptimizerReport`
Budgeted per-layer compiler configuration.

- **feasible**()
  - Whether every layer received a configuration.

### Class `_Candidate`

### Class `_Label`

### Class `_RidgeSurrogate`
Small multi-output ridge regressor backed by NumPy.

- **__init__**(alpha)
- **fit**(features, labels)
- **predict**(features)

### Class `SurrogateSCOptimizer`
Compiler optimiser using a learned surrogate over SC design points.

- **__init__**(target)
- **optimise**(network)
  - Select budgeted layer settings for ``network``.
- **_fit_surrogate**(network)
- **_rank_layer_candidates**(layer, remaining_luts, remaining_power)
- **_candidate_grid**(layer)
- **_analytical_label**(cand)
- **_observation_label**(cand)
- **_to_config**(cand, pred)
- **_rebalance**(selected, network)
- **_features**(cand)
- **_normalise_label**(label)
- **_denormalise_prediction**(values)
- **_utility**(label)
- **_polynomial_quality**(polynomial)
- **_weighted_accuracy**(selected, network)

---

## Module `optimizer.synthesis_evidence`

### Function `load_design(path)`
Load explicit compiler-design metadata for one synthesis observation.

### Function `observation_to_record(observation)`
Convert an optimiser observation into the stable evidence JSON shape.

### Function `build_payload_from_reports()`
Build an evidence payload from report files and explicit metadata.

### Function `build_payload(args)`
Build an evidence payload from parsed command-line arguments.

### Function `energy_payload(observation)`
Compute report-derived energy only when workload metadata is explicit.

### Function `write_payload(payload, output)`
Write evidence JSON to a file or stdout.

### Function `build_parser()`
Build the command-line parser.

### Function `main(argv)`
Run the evidence collector.

---

## Module `physics.heat`

### Class `FeynmanKacHeatSolver`
Solve the 1D heat equation via Feynman-Kac path-integral expectation.

PDE: ``∂u/∂t = α · ∂²u/∂x²`` on ``x ∈ &#91;0, L&#93;`` with
     reflective BC ``u'(0,t) = u'(L,t) = 0``
     and initial condition ``u(x, 0) = f(x)``.

Solution (Feynman-Kac):
    ``u(x_eval, T) = E&#91; f(X_T) | X_0 = x_eval &#93;``
where ``X_t`` is reflected Brownian motion on ``&#91;0, L&#93;`` with
variance ``2α t``.

Parameters
----------
length : float
    Domain extent L (same units as the diffusion coefficient
    squared per unit time). Defaults to a 1.0-unit interval.
diffusivity : float
    α — the heat-equation diffusion coefficient (m²/s in SI;
    units are caller's responsibility).
num_walkers : int
    Number of Monte Carlo walkers; estimator variance scales
    as 1/sqrt(N).
dt : float
    Time step for the Euler-Maruyama integration of the
    Brownian motion. Must be small enough that
    ``sqrt(2 α dt) << L`` so reflection is well-resolved.
seed : int
    RNG seed for reproducibility.

- **__post_init__**()
- **set_initial_distribution**(f, n_grid)
  - Sample walker positions from the initial distribution f(x).
- **set_initial_delta**(x_0)
  - All walkers start at position x_0 (delta-function initial condition).
- **step**(n_substeps)
  - Advance walkers by `n_substeps` Brownian steps.
- **evolve_to**(T)
  - Step walkers from current time to ``T`` (T > current t).
- **get_density**(n_bins)
  - Return the Monte Carlo histogram density over &#91;0, L&#93;.
- **expectation**(observable)
  - Compute the Feynman-Kac expectation `E&#91;observable(X_T)&#93;`.
- **time**()
  - Current simulation time.

---

## Module `physics.wolfram_hypergraph`

### Class `WolframHypergraph`
Simulates the Wolfram Physics Project Hypergraph.
Universe is a set of relations (Hyperedges).

- **evolve**(steps)
  - Applies a rewrite rule.
- **dimension_estimate**()
  - Estimate effective dimension via BFS neighborhood growth.

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
  - Train weights in a multimodal fusion layer via per-sample updates.

---

## Module `privacy.dp_snn`

### Class `PrivacyAccountant`
Track cumulative privacy budget across training steps.

Uses simple composition theorem: total epsilon = sum of per-step epsilons.
For tighter bounds, use Renyi DP (future extension).

Parameters
----------
target_epsilon : float
    Privacy budget limit.
target_delta : float
    Failure probability.

- **record_step**(step_epsilon)
  - Record privacy cost of one training step.
- **spent_epsilon**()
- **remaining_epsilon**()
- **budget_exhausted**()
- **summary**()

### Class `SpikeLevelDP`
Spike-level differential privacy mechanism.

Adds stochastic spike noise to provide (epsilon, delta)-DP.
Two mechanisms:
- Spike randomized response: each spike independently flipped with probability p
- Spike subsampling: randomly drop spikes with probability 1-q

Parameters
----------
epsilon : float
    Per-step privacy budget.
mechanism : str
    'randomized_response' or 'subsampling'.
seed : int

- **__init__**(epsilon, mechanism, seed)
- **privatize**(spikes)
  - Apply DP mechanism to a spike tensor.
- **per_step_epsilon**()

### Class `MembershipAudit`
Audit SNN for membership inference vulnerability.

Given a trained model (as a callable), test whether it leaks
information about training data membership. Uses shadow model
methodology: compare model confidence on training vs non-training
samples.

Parameters
----------
run_fn : callable
    Model function: takes spikes (T, N) → output (N_out,).

- **__init__**(run_fn)
- **audit**(member_samples, non_member_samples)
  - Run membership inference audit.

---

## Module `privacy.governance`

### Class `ConsentBoundary`
Participant-level legal basis and telemetry permissions.

- **__post_init__**()
- **to_dict**()
- **from_dict**(cls, data)

### Class `RetentionPolicy`
Retention windows for neural telemetry and artefacts.

- **__post_init__**()
- **to_dict**()
- **from_dict**(cls, data)

### Class `RedactionPolicy`
Field-level redaction policy for protected telemetry and logs.

- **__post_init__**()
- **to_dict**()
- **from_dict**(cls, data)

### Class `TelemetryPolicy`
Telemetry sink and sampling policy.

- **__post_init__**()
- **to_dict**()
- **from_dict**(cls, data)

### Class `ProvenanceRecord`
Cryptographic provenance record for model and dataset artefacts.

- **__post_init__**()
- **to_dict**()
- **from_dict**(cls, data)

### Class `IntegratorResponsibility`
Operational responsibilities for an integrator in a deployment pipeline.

- **__post_init__**()
- **to_dict**()
- **from_dict**(cls, data)

### Class `PrivacyFeatureFlags`
Feature activation and audit flags for a governed workflow.

- **__post_init__**()
- **to_dict**()
- **from_dict**(cls, data)

### Class `GovernanceContract`
Full privacy governance contract for BCI/neural workflows.

- **__post_init__**()
- **audit_required_features**()
  - Return sorted feature keys that require audit flags.
- **active_features**()
  - Return names of enabled privacy features in deterministic order.
- **to_dict**()
  - Return a deterministic JSON-serialisable representation.
- **from_dict**(cls, data)

### Function `_expect_mapping(value)`
Return ``value`` as dict or raise a typed ValueError.

### Function `_require_fields(payload, section, required)`
Raise when required fields are missing from a manifest section.

### Function `_ensure_positive_days(value, name)`
Validate a duration field.

### Function `_ensure_non_empty_str(value, name)`
### Function `_ensure_bool(value, name)`
### Function `_to_str_tuple(values, name)`
---

## Module `profiler.platform_profiler`

### Class `PlatformResult`
Performance result for one platform.


### Function `compare(layer_sizes, duration, dt, bitstream_length, platforms)`
Compare SNN performance across platforms.

Parameters
----------
layer_sizes : list of (n_inputs, n_neurons)
    Network architecture.
duration : float
    Simulation duration in seconds.
dt : float
    Timestep.
bitstream_length : int
    SC bitstream length for FPGA estimates.
platforms : list of str, optional
    Platforms to compare. Default: all available.
    Options: 'python', 'rust', 'fpga_ice40', 'fpga_artix7'

Returns
-------
list of PlatformResult
    One result per platform, sorted by energy efficiency.

### Function `format_table(results)`
Format comparison results as a readable table.

### Function `_profile_python(layer_sizes, duration, dt)`
Measure Python/NumPy backend performance.

### Function `_profile_rust(layer_sizes, duration, dt)`
Estimate Rust NetworkRunner performance.

### Function `_profile_fpga(layer_sizes, target, bitstream_length)`
Estimate FPGA performance using energy estimator.

---

## Module `profiling.energy`

### Class `EnergyMetrics`
- **reset**()
- **estimate_energy**()
- **co2_emission_g**(carbon_intensity_g_per_kwh)

### Function `track_energy(func)`
Decorator to track energy of a layer call (simulated).

---

## Module `profiling.spike_profiler`

### Class `Severity`

### Class `Pathology`
One detected training pathology.


### Class `LayerStats`
Accumulated statistics for one layer across recorded steps.


### Class `ProfileReport`
Complete profiling report with per-layer stats and detected pathologies.

- **summary**()
- **has_critical**()

### Class `SpikeProfiler`
Instruments SNN training to detect pathologies and compute diagnostics.

Record spike tensors, voltage tensors, and optionally gradient tensors
per layer per training step. Call report() to get a ProfileReport with
detected pathologies and fix suggestions.

Parameters
----------
dead_threshold : float
    Firing rate below which a neuron is considered dead (default 0.01).
saturated_threshold : float
    Firing rate above which a neuron is considered saturated (default 0.95).
gradient_explosion_ratio : float
    Max/mean gradient norm ratio above which gradient explosion is flagged.

- **__init__**(dead_threshold, saturated_threshold, gradient_explosion_ratio)
- **record_step**(layer, spikes, voltages, gradients)
  - Record one timestep of data for a layer.
- **reset**()
  - Clear all accumulated data.
- **report**()
  - Analyze accumulated data and return a ProfileReport.
- **_detect_pathologies**(layer_stats)

### Class `_LayerAccumulator`
Internal: accumulates per-step data for one layer.

- **__init__**(name)
- **add**(spikes, voltages, gradients)
- **compute_stats**()

---

## Module `qat.quantize`

### Class `TernaryWeights`
Ternary weight quantization: {-1, 0, +1}.

94% memory reduction. Each weight is one of three values.
Threshold-based: weights with |w| < threshold become 0.

Parameters
----------
threshold_ratio : float
    Fraction of max(|w|) below which weights are zeroed.

- **__init__**(threshold_ratio)
- **quantize**(weights)
- **sparsity**(weights)

### Class `QuantizedSNNLayer`
SNN layer with quantization-aware forward pass.

During training: weights quantized in forward, full-precision in backward (STE).
At export: weights are already at target precision.

Parameters
----------
n_inputs : int
n_neurons : int
weight_bits : int
    Target weight precision (2, 4, 8, 16).
threshold : float
tau_mem : float

- **__post_init__**()
- **forward**(x, dt)
  - Quantization-aware forward pass.
- **export_weights**()
  - Export quantized weights for hardware deployment.
- **reset**()

### Function `_ste_quantize(x, bits, symmetric)`
Quantize with straight-through estimator (forward quantized, backward identity).

### Function `quantize_aware_train_step(layer, x, target, lr)`
One QAT training step with STE.

Parameters
----------
layer : QuantizedSNNLayer
x : ndarray of shape (n_inputs,)
target : ndarray of shape (n_neurons,)
lr : float

Returns
-------
dict with 'output', 'loss'

---

## Module `qat.torch_qat`

### Class `_STEQuantize`
Straight-through estimator for uniform quantization.

- **forward**(ctx, x, n_bits, symmetric)
- **backward**(ctx, grad_output)

### Class `QuantizedLinear`
Linear layer with STE weight quantization.

- **__init__**(in_features, out_features, n_bits, bias)
- **forward**(x)
- **export_quantized**()
  - Export integer weights at target precision.

### Class `QuantizedLIFNet`
Feedforward SNN with quantized weights for QAT.

Drop-in replacement for SpikingNet with configurable bit precision.

Example
-------
>>> net = QuantizedLIFNet(784, 128, 10, n_bits=4)
>>> x = torch.randn(25, 32, 784)  # (T, batch, features)
>>> spikes, mem = net(x)
>>> spikes.shape
torch.Size(&#91;32, 10&#93;)

- **__init__**(n_input, n_hidden, n_output, n_layers, n_bits, beta, surrogate_fn)
- **forward**(x)
  - x: (T, batch, n_input). Returns (spike_counts, membrane_acc).
- **export_quantized**()
  - Export all layers as quantized integer weights.
- **effective_bits**()
  - Average effective bits across all weights (for reporting).

### Class `SCAwareLinear`
Linear layer with SC noise injection during training.

During training: injects Gaussian noise with std = sqrt(p*(1-p)/L)
to simulate bitstream variance. Weights clamped to &#91;-1, 1&#93;.

During eval: no noise, standard linear.

- **__init__**(in_features, out_features, bitstream_length, bias)
- **forward**(x)

### Class `SCAwareLIFNet`
SNN with SC-aware training: noise injection + weight clamping.

Trains the model to be robust to stochastic computing bitstream
variance. Weights are constrained to &#91;-1, 1&#93; (bipolar SC range).

Example
-------
>>> net = SCAwareLIFNet(784, 128, 10, bitstream_length=256)
>>> x = torch.randn(25, 32, 784)
>>> spikes, mem = net(x)

- **__init__**(n_input, n_hidden, n_output, n_layers, bitstream_length, beta, surrogate_fn)
- **forward**(x)
  - x: (T, batch, n_input). Returns (spike_counts, membrane_acc).
- **export_bipolar_weights**()
  - Export weights clamped to &#91;-1, 1&#93; for bipolar SC deployment.

### Function `ste_quantize(x, n_bits, symmetric)`
Quantize tensor with straight-through estimator.

---

## Module `quantum.hardware_bridge`

### Class `QuantumHardwareLayer`
Executes a Quantum-Classical Hybrid Layer on Qiskit/PennyLane.
Maps bitstream probability -> Qubit Rotation -> True Measurement.

- **__post_init__**()
- **forward**(input_bitstreams)
  - input_bitstreams: (n_qubits, length)
- **_run_qiskit**(theta)
  - Runs the circuit on Qiskit AerSimulator for `length` shots.
- **_run_pennylane**(theta)
  - Runs the circuit on PennyLane for `length` shots.

---

## Module `quantum.hybrid`

### Class `QuantumStochasticLayer`
Simulates a Quantum-Classical Hybrid Layer.
Input bitstream probability -> Qubit Rotation -> Measurement Probability.

Mapping: p_in -> theta = p_in * pi
P_out = |<0|Ry(theta)|0>|^2 = cos^2(theta/2)
This non-linearity is useful for classification.

- **__post_init__**()
- **forward**(input_bitstreams)
  - input_bitstreams: (n_qubits, length)

---

## Module `quantum.hybrid_pipeline`

### Class `HybridQuantumClassicalPipeline`
- **__init__**(n_qubits, n_layers, noise_model)
- **circuit**(params)
  - Parameterized Ry-CNOT circuit → ⟨Z⊗Z⟩ expectation.
- **train**(n_steps, lr)
  - VQE-style optimization: minimize ⟨Z⊗Z⟩.
- **evaluate**(params)

### Function `_ry(theta)`
Ry rotation gate.

### Function `_cnot()`
### Function `_kron_gate(gate, qubit, n_qubits)`
Embed single-qubit gate into n-qubit space.

---

## Module `quantum.noise_models`

### Class `HeronR2NoiseParams`
IBM Heron r2 calibration parameters (2024).


### Class `HeronR2NoiseModel`
- **__init__**(params)
- **depolarizing_channel**(p)
  - Kraus operators for single-qubit depolarizing channel.
- **amplitude_damping**(gamma)
  - Kraus operators for amplitude damping (T1 decay).
- **phase_damping**(gamma)
  - Kraus operators for phase damping (T2 decay).
- **apply_single_qubit_noise**(rho)
  - Apply single-qubit noise channel to density matrix.
- **apply_readout_noise**(measurement)
  - Apply asymmetric readout error.
- **gate_fidelity_1q**()
- **gate_fidelity_2q**()

---

## Module `quantum.param_shift`

### Class `ParameterShiftOptimizer`
- **__init__**(circuit_fn, n_params, lr)
- **compute_gradient**(params)
- **step**(params)

### Function `parameter_shift_gradient(circuit_fn, params, shift)`
Gradient via parameter-shift rule.

f'(θ_i) = &#91;f(θ_i + s) - f(θ_i - s)&#93; / (2 sin(s))

---

## Module `quantum.qec`

### Class `QecShield`
Repetition code QEC shield for stochastic-quantum bitstreams.

- **__init__**(code_type, distance)
- **encode**(bitstream)
  - repetition code (d=3): 0 -> 000, 1 -> 111
- **extract_syndromes**(physical_bits)
- **decode**(physical_bits)
- **get_error_rate**(syndromes)

### Class `SurfaceCodeShield`
Distance-d rotated surface code for stochastic-quantum bitstreams.

Encodes 1 logical qubit into d² physical data qubits. X and Z stabilizers
detect bit-flip and phase-flip errors. Decoding uses a lookup table for d=3.

Ref: Fowler et al., "Surface codes: Towards practical large-scale quantum
computation", Phys. Rev. A 86, 032324 (2012).

- **__init__**(distance)
- **_build_stabilizers**(d)
  - Build X and Z stabilizer generators for rotated surface code.
- **_build_d3_lut**(stabilizers)
  - Build syndrome → correction lookup for d=3 single-qubit errors.
- **encode**(bitstream)
  - Encode logical bitstream into surface code physical qubits.
- **measure_syndrome**(physical_bits)
  - Measure X and Z stabilizer syndromes.
- **decode**(physical_bits)
  - Decode surface code: measure syndromes, correct single-qubit errors, majority vote.
- **_apply_lut_correction**(physical, syndromes, lut)
  - Apply lookup-table correction for each bitstream position.
- **get_error_rate**(x_syn, z_syn)
  - Estimated error rate from syndrome density.

---

## Module `quantum.sc_quantum_compiler`

### Class `QuantumGate`
A quantum gate applied to specific qubits.


### Class `SCQuantumCircuit`
Quantum circuit compiled from SC operations.

- **simulate**()
  - Simulate the circuit and return the full statevector.
- **output_probability**()
  - Simulate and return P(output_qubit = |1⟩).
- **simulate_noisy**(noise_model)
  - Simulate with noise: evolve density matrix through Kraus channels.
- **output_probability_noisy**(noise_model, n_shots)
  - Simulate with noise and return P(output=1) via measurement sampling.
- **summary**()

### Function `sc_prob_to_statevector(p)`
Encode SC probability as a single-qubit state vector.

|ψ⟩ = √(1-p)|0⟩ + √p|1⟩  →  P(measure |1⟩) = p exactly.

### Function `statevector_to_prob(sv)`
Extract SC probability from a single-qubit state vector via Born rule.

### Function `ry_gate(theta)`
Ry rotation gate: encodes probability via rotation angle.

### Function `prob_to_ry_angle(p)`
Compute Ry angle that encodes probability p: sin²(θ/2) = p.

### Function `_apply_gate(state, gate, qubits, n_qubits)`
Apply a gate to specific qubits in a full statevector.

### Function `_apply_single_qubit_gate(state, gate, qubit, n_qubits)`
Apply a 2x2 gate to one qubit in a multi-qubit state.

### Function `_apply_two_qubit_gate(state, gate, q0, q1, n_qubits)`
Apply a 4x4 gate to two qubits.

### Function `_apply_single_qubit_channel(rho, noise_model, qubit, n_qubits)`
Apply single-qubit noise channel to one qubit of a multi-qubit density matrix.

### Function `compile_sc_multiply(p_a, p_b)`
Compile SC AND gate (multiplication) to a quantum circuit.

SC: P(a AND b) = P(a) * P(b) for independent streams.
Quantum: encode probabilities as Ry rotations, use CNOT for correlation.

### Function `compile_sc_layer(weights, input_probs)`
Compile an SC dense layer to quantum gate descriptions.

Parameters
----------
weights : np.ndarray
    Shape (n_neurons, n_inputs), values in &#91;0, 1&#93;.
input_probs : np.ndarray
    Shape (n_inputs,), SC input probabilities.

Returns
-------
list of dicts, one per neuron, each containing:
    'neuron_idx': int
    'ry_angles': list of (input_angle, weight_angle) pairs
    'expected_output': float — SC computation result
    'quantum_output': float — quantum simulation result

---

## Module `quantum_cognition.__main__`

### Function `_emit_snn_stimulus(snn_dir, chunk_summary, directive, step_index)`
Write an SNN stimulus JSON file for downstream orchestration.

### Function `_make_llm_endpoint(model)`
Create an agentic-shared Endpoint if model override requested.

### Function `cmd_learn(args)`
One-shot learning from a repository.

### Function `cmd_daemon(args)`
Continuous learning daemon — shuffles GOTM, learns in cycles.

### Function `cmd_status(args)`
Print saved learning state.

### Function `main(argv)`
CLI entry point.

---

## Module `quantum_cognition.bridge_adapter`

### Class `FisherPosnerQuantumBridge`
Bridge between SpinPoolMPS and quantum hardware / orchestrator.

Supports three operational modes:

1. **Emulated** (default): Pure NumPy phase optimisation via the
   MPS emulator.  No external dependencies required.
2. **PennyLane**: Gradient-based phase optimisation using PennyLane
   autograd on simulated qubits.
3. **Orchestrator**: Accepts global phase vectors from the
   scpn-phase-orchestrator and combines them with local
   optimisation.

Parameters
----------
n_qubits : int
    Number of qubits / spin sites.
backend : str
    Backend selection: ``"auto"``, ``"pennylane"``, ``"ibm_qiskit"``,
    or ``"emulated"``.

- **__init__**(n_qubits, backend)
- **_init_ibm_backend**()
  - Initialise IBM Quantum backend via qiskit-ibm-runtime.
- **backend**()
  - Active backend name.
- **execute_non_local_sync**(entangle_pairs)
  - Execute non-local synchronisation via entanglement.
- **_sync_ibm_qiskit**(entangle_pairs, shots)
  - Build and execute Bell pair circuit on IBM backend or AerSimulator.
- **execute_posner_circuit**(shots)
  - Dispatch an actual 8q Posner Hamiltonian circuit to IBM QPU.
- **_extract_qiskit_counts**(pub_result)
  - Extract counts from a SamplerV2 result regardless of register name.
- **_dispatch_qiskit_circuit**(qc, shots)
  - Dispatch a circuit and return ⟨Z⟩ expectation values.
- **_dispatch_qiskit_circuit_raw**(qc, shots)
  - Dispatch a circuit and return raw bitstring counts.
- **_sync_ibm_aer**(qc, shots)
  - Execute circuit on explicit local AerSimulator backend.
- **_counts_to_expvals**(counts, shots)
  - Convert bitstring counts to ⟨Z⟩ expectation values.
- **_sync_pennylane**(entangle_pairs)
  - PennyLane Bell pair circuit → PauliZ expectations.
- **_sync_emulated**(entangle_pairs)
  - Pure-numpy emulation of Bell pair correlations.
- **optimize_phases**(target_coherence, learning_rate, n_steps)
  - Optimise qubit phases towards target coherence.
- **apply_orchestrator_bias**(global_phases, target_coherence, learning_rate)
  - Combine global orchestrator phases with local optimisation.
- **to_qpu_artifact_metadata**()
  - Produce metadata for QPUBridgeArtifact integration.
- **__repr__**()

### Function `compute_max_qubits(safety_factor)`
Compute maximum PennyLane qubits that fit in available RAM.

PennyLane ``default.qubit`` uses a dense state vector of shape
``(2**n_qubits,)`` with ``complex128`` (16 bytes per amplitude).
This function reads available RAM and computes the largest qubit
count that stays within the safety budget.

Falls back to ``/proc/meminfo`` if ``psutil`` is unavailable
(common on minimal containers).

Parameters
----------
safety_factor : float
    Fraction of available RAM to allow (0, 1&#93;.  Default 0.5.

Returns
-------
int
    Maximum qubit count, clamped to &#91;``_QUBIT_FLOOR``, ``_QUBIT_CEILING``&#93;.

### Function `_get_available_ram()`
Return available RAM in bytes.  psutil → /proc/meminfo fallback.

---

## Module `quantum_cognition.content_indexer`

### Class `ContentChunk`
A single indexed content chunk from a GOTM repository.

Attributes
----------
repo_name : str
    Repository name (e.g. ``"SC-NEUROCORE"``).
file_path : str
    Relative path within the repository.
chunk_index : int
    Sequential index within the file.
text : str
    Raw text content of the chunk.
content_type : str
    One of ``"docstring"``, ``"comment"``, ``"markdown"``, ``"code"``,
    ``"metadata"``.
weight : float
    Priority weight based on file type.
sha256 : str
    SHA-256 hash of the chunk text (provenance).

- **__post_init__**()
- **summary**()
  - First 200 characters of the chunk text.
- **to_dict**()
  - Serialise to JSON-compatible dict.

### Function `_should_skip_dir(name)`
Check if a directory should be skipped during indexing.

### Function `_extract_python_docstrings(text)`
Extract docstrings and significant comments from Python source.

### Function `_extract_rust_doc_comments(text)`
Extract /// and //! doc comments from Rust source.

### Function `_chunk_text(text, target_size)`
Split text into chunks of approximately target_size characters.

### Function `index_file(file_path, repo_name, repo_root)`
Index a single file into content chunks.

Parameters
----------
file_path : Path
    Absolute path to the file.
repo_name : str
    Name of the repository.
repo_root : Path
    Root directory of the repository.

Returns
-------
list&#91;ContentChunk&#93;
    Extracted content chunks with provenance metadata.

### Function `index_gotm_repo(repo_path, repo_name)`
Index an entire GOTM repository into content chunks.

Parameters
----------
repo_path : str or Path
    Path to the repository root.
repo_name : str, optional
    Override repository name (default: directory name).

Returns
-------
list&#91;ContentChunk&#93;
    All indexed chunks sorted by weight (descending).

### Function `embed_chunks(chunks, n_dims, seed)`
Convert content chunks to numerical vectors for neural input.

Uses a lightweight deterministic hashing approach (not a neural
embedding model) to produce fixed-size feature vectors.  Each
dimension captures a different statistical property of the text:

- Character frequency distribution (dims 0–25)
- Text length features (dim 26–27)
- Weight and content type (dim 28–29)
- Hash-derived features (dim 30–31)

Parameters
----------
chunks : list&#91;ContentChunk&#93;
    Content chunks to embed.
n_dims : int
    Output vector dimensionality (default 32).
seed : int
    Random seed for hash-derived features.

Returns
-------
np.ndarray
    Shape ``(len(chunks), n_dims)``, values normalised to &#91;0, 1&#93;.

### Function `embed_tfidf(chunks, n_dims, min_df, max_df_ratio)`
Compute proper TF-IDF vectors from a corpus of chunks.

Unlike ``embed_chunks()`` which uses character statistics, this
computes true TF-IDF with corpus-wide Inverse Document Frequency:

    TF(t,d) = log(1 + count(t,d))
    IDF(t) = log(N / df(t))
    TF-IDF(t,d) = TF(t,d) × IDF(t)

Parameters
----------
chunks : list&#91;ContentChunk&#93;
    Corpus of content chunks.
n_dims : int
    Number of top-IDF terms to use as feature dimensions.
min_df : int
    Minimum document frequency for a term to be included.
max_df_ratio : float
    Maximum document frequency ratio (terms in >85% of docs removed).

Returns
-------
tuple&#91;np.ndarray, dict&#91;str, int&#93;&#93;
    - TF-IDF matrix of shape ``(len(chunks), n_dims)``, L2-normalised.
    - Vocabulary mapping {term: dimension_index}.

---

## Module `quantum_cognition.dashboard`

### Class `TerminalDashboard`
ANSI terminal dashboard for quantum cognition monitoring.

Parameters
----------
max_raster_steps : int
    Number of recent steps to show in the spike raster.
clear_screen : bool
    Whether to clear terminal before drawing.

- **__init__**(max_raster_steps, clear_screen)
- **draw**(brain)
  - Render a single dashboard frame.
- **__repr__**()

### Function `_heat_char(value, max_val)`
Map a value to a coloured block character.

### Function `_bar(value, max_val, width)`
Render a simple bar with fill fraction.

### Function `_directive_colour(directive)`
Return ANSI colour for a directive.

---

## Module `quantum_cognition.fisher_posner`

### Class `HybridFisherPosnerLIF`
LIF neuron with quantum-metabolic coupling via spin pool.

Parameters
----------
neuron_id : int
    Site index in the spin pool (determines entanglement location).
spin_pool : SpinPoolMPS
    Shared spin pool providing non-local ATP modulation.
dt : float
    Integration timestep in ms.
v_rest : float
    Resting membrane potential in mV.
v_threshold : float
    Spike threshold in mV.
v_reset : float
    Post-spike reset potential in mV.
tau_m : float
    Membrane time constant in ms.
atp_initial : float
    Initial ATP level (normalised, 0–1).
atp_consumption : float
    ATP consumed per spike.
atp_basal_regeneration : float
    First-order ATP recovery rate independent of Posner singlet yield.
    Posner efficiency modulates this baseline rather than replacing
    cellular metabolism.

- **__init__**(neuron_id, spin_pool, dt, v_rest, v_threshold, v_reset, tau_m, atp_initial, atp_consumption, atp_basal_regeneration)
- **step**(I_in)
  - Advance the neuron by one timestep.
- **get_state**()
  - Return full neuron state for checkpointing.
- **reset_state**()
  - Reset to resting potential and full ATP.
- **reset**()
  - Alias for reset_state (NeuronProtocol compatibility).
- **v**()
  - Membrane voltage alias for Population compatibility.
- **v**(value)
- **__repr__**()

### Class `HybridFisherPosnerLIFNeuron`
Population-compatible wrapper for HybridFisherPosnerLIF.

This class creates its own SpinPoolMPS and wraps ``step()`` to return
``0`` or ``1`` (integer spike flag) instead of ``(Vm, bool)``, making it
compatible with ``Population(model='HybridFisherPosnerLIFNeuron', n=N)``.

The underlying SpinPoolMPS is shared among all neurons in the same
Population via class-level pool management.

- **__init__**(dt, v_rest, v_threshold, v_reset, tau_m, atp_initial, atp_consumption, atp_basal_regeneration, n_sites)
- **step**(I_in)
  - Advance one timestep, return 1 if spiked else 0.
- **v**()
- **v**(value)
- **v_threshold**()
- **v_rest**()
- **get_state**()
- **reset**()
- **reset_state**()
- **__repr__**()
- **_reset_pools**(cls)
  - Reset shared pool registry (for testing).

---

## Module `quantum_cognition.fs_watcher`

### Class `GOTMWatcher`
Watch the GOTM collection for new/modified files.

Parameters
----------
watch_path : str or Path
    Root directory to monitor.
repo_name : str
    Repository name for content chunks.
debounce_s : float
    Minimum seconds between re-processing the same file.
poll_interval_s : float
    Polling interval when using the fallback backend.
use_polling : bool or None
    Force polling mode.  ``None`` = auto-detect (use watchdog if
    available, poll otherwise).

- **__init__**(watch_path, repo_name, debounce_s, poll_interval_s, use_polling)
- **start**()
  - Start the watcher in a background daemon thread.
- **stop**()
  - Stop the watcher gracefully.
- **get_chunks**(max_items)
  - Drain the chunk queue (non-blocking).
- **is_running**()
  - Whether the watcher thread is active.
- **_should_process**(file_path)
  - Check debounce timer for a file.
- **_process_file**(file_path)
  - Index a single file and enqueue its chunks.
- **_run**()
  - Main watcher loop (runs in background thread).
- **_run_polling**()
  - Polling-based file watcher for NTFS compatibility.
- **_run_watchdog**()
  - Watchdog (inotify) based file watcher.
- **__repr__**()

---

## Module `quantum_cognition.gotm_brain`

### Class `LearningStep`
Record of a single learning step for telemetry.

- **to_dict**()
  - Serialise to JSON-compatible dict.

### Class `GOTMBrain`
Self-learning brain for the God of the Math collection.

Composes quantum cognition classes with a local LLM to create a
neural system that learns from GOTM content.

Parameters
----------
n_neurons : int
    Number of neurons (also determines spin pool sites and qubit count).
bridge_backend : str
    Quantum bridge backend (``"emulated"``, ``"pennylane"``,
    ``"ibm_aer"``, ``"ibm_qiskit"``, or ``"auto"``). The default is
    explicit emulation so repository learning never silently switches to
    an expensive simulator because an optional package is installed.
seed : int or None
    Random seed for reproducibility.

- **__init__**(n_neurons, bridge_backend, seed, llm_endpoint)
- **get_llm_guidance**(context_summary)
  - Query the configured local LLM for a learning directive.
- **process_content**(input_vector, directive)
  - Process a content vector through the neural network.
- **learn_step**(chunk, vector)
  - Execute a single learning step on one content chunk.
- **learn_from_repo**(repo_path, repo_name, max_chunks)
  - Index and learn from an entire GOTM repository.
- **get_learning_state**()
  - Return full learning state for inspection and persistence.
- **get_history**()
  - Return learning history as a list of dicts.
- **save_state**(path)
  - Persist full brain state (v_deep) to a JSON file.
- **load_state**(path)
  - Restore brain state (v_deep) from a previously saved JSON file.
- **reset**()
  - Reset all neurons, spin pool, and history.
- **__repr__**()

---

## Module `quantum_cognition.kane_mapper`

### Class `KaneRegisterLayout`
Physical layout of a Si:P qubit register.

Attributes
----------
n_qubits : int
    Number of ³¹P donor qubits.
qubit_positions : np.ndarray
    Shape ``(n_qubits, 2)`` — (x, y) coordinates in nanometres.
coupling_matrix : np.ndarray
    Shape ``(n_qubits, n_qubits)`` — exchange coupling J(d) in meV.
    Symmetric, diagonal is zero.
depth_nm : float
    Implantation depth below Si surface.
t2_budget_ms : float
    T₂ decoherence budget for the register in milliseconds.
max_gate_depth : int
    Maximum circuit depth achievable within the T₂ budget.
gate_schedule : list&#91;dict&#93;
    Ordered list of gate operations with timing.

- **to_dict**()
  - Serialise to JSON-compatible dict.

### Class `KaneSiliconMapper`
Map SpinPoolMPS sites to a Kane-architecture Si:P register.

Parameters
----------
spacing_nm : float
    Target inter-donor spacing in nanometres (default 20 nm).
depth_nm : float
    Implantation depth below silicon surface (default 20 nm).
topology : str
    Layout topology: ``"linear"``, ``"grid"``, ``"triangular"``, or
    ``"hexagonal"``.

- **__init__**(spacing_nm, depth_nm, topology)
- **map_pool_to_register**(n_sites)
  - Compute physical qubit placement and coupling matrix.
- **_compute_positions**(n)
  - Compute qubit positions based on topology.
- **_compute_coupling_matrix**(positions)
  - Compute exchange coupling J(d) between all donor pairs.
- **_exchange_coupling**(distance_nm)
  - Compute exchange coupling J(d) in meV.
- **_build_gate_schedule**(n, coupling)
  - Build a gate schedule with DAG-based parallel scheduling.
- **get_constraints**(n_sites)
  - Return design constraints for a register of given size.
- **__repr__**()

---

## Module `quantum_cognition.radical_pair`

### Class `RadicalPairParams`
Parameters for a radical pair system.

Attributes
----------
hyperfine_a : float
    Isotropic hyperfine coupling constant in MHz for the exact default
    one-nucleus model.  Posner-specific calculations must pass explicit
    tensors via ``hyperfine_tensors_1`` and/or ``hyperfine_tensors_2``.
exchange_j : float
    Exchange coupling in MHz.
    Typical: 0–10 MHz for separated radicals in Posner molecules.
recombination_rate : float
    Radical pair recombination rate in µs⁻¹.
    Typical: 0.01–1.0 µs⁻¹.
lifetime_us : float
    Radical pair lifetime in µs.
    Posner molecules: ~1–1000 µs (protected by Ca₉(PO₄)₆ cage).
hyperfine_tensors_1, hyperfine_tensors_2 : list&#91;np.ndarray&#93;
    3×3 hyperfine tensors in MHz for nuclei coupled to electron 1 and
    electron 2 respectively.
quadrature_order : int
    Gauss-Legendre points for finite-lifetime recombination integration.


### Class `RadicalPairModel`
Density-matrix radical pair mechanism for ATP hydrolysis gating.

Computes singlet yield (probability of productive ATP hydrolysis)
as a function of local magnetic field and radical pair parameters.

Parameters
----------
params : RadicalPairParams, optional
    RPM parameters.  Defaults are an exact one-nucleus isotropic RPM,
    not a Posner molecule parameterization.

- **__init__**(params)
- **_isotropic_tensor**(a_mhz)
- **from_hyperfine_tensors**(cls)
  - Construct an exact RPM model from explicit 3×3 hyperfine tensors.
- **_validated_tensors**()
- **_spin_operator**(n_spins, target, component)
- **_singlet_density_with_nuclear_bath**(n_nuclei)
- **_hamiltonian**(b_local)
- **singlet_yield**(b_local)
  - Compute singlet yield Φ_S for the current parameters.
- **singlet_yield_field_sweep**(b_range)
  - Compute singlet yield over a range of magnetic fields.
- **atp_efficiency**(b_local, entanglement_boost)
  - Compute ATP hydrolysis efficiency from singlet yield.
- **get_state**()
  - Return parameters as a dict for serialisation.
- **__repr__**()

---

## Module `quantum_cognition.spin_pool`

### Class `SpinCouplingTensor`
Two-spin coupling tensor in MHz.

``tensor_mhz&#91;a, b&#93;`` multiplies ``S_i^a S_j^b`` for
``a,b ∈ {x,y,z}``.  This supports isotropic exchange, anisotropic
dipolar coupling, and off-diagonal tensor terms without adding hidden
model constants.


### Class `SpinPoolMPS`
Non-local spin storage using true Matrix Product States.

Simulates entangled :sup:`31`\ P nuclear spins in Posner molecules.
Each site corresponds to a phosphorus nuclear spin represented as a
rank-3 tensor A&#91;α, σ, β&#93; where:
- α, β are bond indices (dimension ``bond_dim``)
- σ is the physical index (0 = spin-up, 1 = spin-down)

The full state |Ψ⟩ = Σ Tr(A¹&#91;σ₁&#93;·A²&#91;σ₂&#93;·…·Aⁿ&#91;σₙ&#93;) |σ₁…σₙ⟩

Parameters
----------
n_sites : int
    Number of nuclear spin sites.
bond_dim : int
    Maximum bond dimension (controls entanglement capacity).
correlation_length : float
    Initialisation parameter for inter-site correlation decay.
update_rate : float
    Mixing rate α for entanglement map updates on spike events.

- **__init__**(n_sites, bond_dim, correlation_length, update_rate, seed)
- **_init_product_state**()
  - Initialise MPS as the pure product state |00…0⟩.
- **to_statevector**()
  - Return the exact statevector represented by this MPS.
- **set_statevector**(statevector)
  - Load a statevector into MPS form without silent truncation.
- **_full_spin_operator**(n_sites, site, component)
- **evolve_exact**(couplings, time_us)
  - Evolve under explicit two-spin coupling tensors.
- **_compute_rdm_single**(site)
  - Compute single-site reduced density matrix by contracting MPS.
- **_compute_rdm_two_site**(site)
  - Compute two-site reduced density matrix for sites (site, site+1).
- **_compute_entanglement_entropy**(site)
  - Compute von Neumann entropy of bipartition at site.
- **_update_entanglement_map**()
  - Recompute entanglement map from MPS bond entropies.
- **apply_measurement**(site_idx, intensity)
  - Apply a Born-rule projective measurement at one spin site.
- **_apply_adjacent_unitary**(i, unitary)
  - Apply a two-site unitary to adjacent sites ``i`` and ``i + 1``.
- **_swap_adjacent**(i)
  - Apply an exact adjacent SWAP gate inside the MPS.
- **_apply_heisenberg_between**(i, j, coupling)
  - Apply the Heisenberg gate to arbitrary sites via an exact SWAP network.
- **_apply_tebd_gate**(i, j, coupling)
  - Apply a two-site TEBD gate between adjacent sites i and j.
- **get_local_atp_efficiency**(site_idx)
  - Return ATP hydrolysis probability at a given site.
- **rho**()
  - Return site-0 reduced density matrix (backward compat).
- **get_status**()
  - Return summary status for telemetry and visualisation.
- **get_state**()
  - Return full internal state for checkpointing.
- **set_state**(state)
  - Restore internal state from a checkpoint dictionary.
- **reset**()
  - Reset to product state.
- **to_scpn_payload**()
  - Produce metadata compatible with SCPNDatastream format.
- **__repr__**()

---

## Module `quantum_cognition.studio_hook`

### Class `QuantumCognitionLayerMetadata`
Structured metadata for the quantum cognition visualisation layer.

- **to_dict**()
  - Serialise to a JSON-compatible dict.

### Class `QuantumStudioHook`
Telemetry endpoint for quantum cognition layer visualisation.

Provides structured metadata and streaming data for SNN Visual Studio
and SCPN Studio frontends.

Parameters
----------
spin_pool : SpinPoolMPS
    The spin pool to observe.
bridge : FisherPosnerQuantumBridge
    The quantum bridge to observe.

- **__init__**(spin_pool, bridge)
- **get_layer_metadata**()
  - Return layer metadata for the Studio layer panel.
- **get_layer_metadata_dict**()
  - Return layer metadata as a plain dict (JSON-serialisable).
- **get_realtime_data**()
  - Return streaming data for live entanglement graph.
- **get_entanglement_snapshot**()
  - Return a timestamped snapshot of entanglement and ATP state.
- **to_json_event**(event_type)
  - Produce a JSON event string for external frontend streaming.
- **__repr__**()

---

## Module `quantum_cognition.tests.test_fisher_posner`

### Class `TestHybridFisherPosnerLIF`
- **test_resting_state**(pool_and_neuron)
- **test_subthreshold_no_spike**(pool_and_neuron)
  - Small input should not trigger a spike.
- **test_suprathreshold_spike**(pool_and_neuron)
  - Large input should trigger a spike.
- **test_metabolic_failure**(pool_and_neuron)
  - Depleted ATP should prevent spiking (metabolic failure).
- **test_atp_regeneration**(pool_and_neuron)
  - ATP should regenerate over time via quantum efficiency.
- **test_spike_feeds_back_to_pool**(pool_and_neuron)
  - Spiking should call apply_measurement on the spin pool.
- **test_v_property**(pool_and_neuron)
  - v property should alias Vm.
- **test_reset**(pool_and_neuron)
- **test_get_state**(pool_and_neuron)
- **test_invalid_neuron_id**()
- **test_invalid_type**()

### Class `TestHybridFisherPosnerLIFNeuron`
- **test_wrapper_step**()
  - Wrapper step should return int (0 or 1).
- **test_wrapper_v_property**()

### Function `pool_and_neuron()`
---

## Module `quantum_cognition.tests.test_gotm_brain_inline`

### Class `TestGOTMBrainPersistence`
- **test_save_load_roundtrip**(tmp_path)
  - State should survive save → load cycle.
- **test_load_mismatched_neurons**(tmp_path)
  - Loading state with wrong neuron count should raise.

### Class `TestGOTMBrainLLM`
- **test_fallback_directive**()
  - Without LLM, should return STABILIZE.
- **test_process_content**()
  - process_content should return list of spike indices.

### Class `TestGOTMBrainMisc`
- **test_reset**()
- **test_get_learning_state**()
- **test_repr**()
- **test_learning_step_to_dict**()

---

## Module `quantum_cognition.tests.test_kane_mapper`

### Class `TestKaneSiliconMapper`
- **test_linear_positions**()
  - Linear topology should place qubits in a line.
- **test_grid_positions**()
  - Grid topology should place qubits in a 2D arrangement.
- **test_coupling_matrix_symmetry**()
  - Coupling matrix must be symmetric with zero diagonal.
- **test_coupling_decay**()
  - Coupling should decay with distance.
- **test_coupling_positive**()
  - All coupling values should be non-negative.
- **test_t2_budget**()
  - T₂ budget must be positive.
- **test_single_qubit**()
  - Single qubit register should work.
- **test_constraints**()
  - Constraints dict should contain all expected fields.
- **test_constraints_infeasible**()
  - Wide spacing should be infeasible (coupling too weak).
- **test_serialisation**()
  - to_dict should produce JSON-compatible output.
- **test_invalid_spacing**()
- **test_invalid_topology**()
- **test_repr**()

---

## Module `quantum_cognition.tests.test_radical_pair`

### Class `TestRadicalPairParams`
- **test_defaults**()
- **test_custom**()

### Class `TestRadicalPairModel`
- **test_singlet_yield_zero_field**()
  - At zero field, singlet yield depends only on exchange/hyperfine ratio.
- **test_singlet_yield_range**()
  - Singlet yield must be bounded &#91;0, 1&#93;.
- **test_strong_exchange_preserves_singlet**()
  - Strong exchange coupling should preserve singlet character.
- **test_weak_exchange_reduces_singlet**()
  - Weak exchange coupling leads to more mixing → lower singlet yield.
- **test_field_sweep**()
  - Field sweep should return array of correct length.
- **test_atp_efficiency_rejects_entanglement_boost**()
  - Classical boost is not a radical-pair Hamiltonian parameter.
- **test_atp_efficiency_range**()
  - ATP efficiency is singlet-yield-derived and bounded.
- **test_get_state**()
  - State dict should contain all params.
- **test_repr**()

---

## Module `quantum_cognition.tests.test_spin_pool`

### Class `TestSpinPoolMPS`
- **test_init_defaults**()
- **test_entanglement_map_normalised**()
  - Entanglement map should sum to 1 after init.
- **test_measurement_updates_map**()
  - Measurement at a site should shift entanglement towards that site.
- **test_normalisation_preserved**()
  - Entanglement map should remain normalised after measurements.
- **test_atp_efficiency_range**()
  - ATP efficiency is a singlet probability in &#91;0, 1&#93;.
- **test_invalid_site_index**()
- **test_negative_intensity**()
- **test_reset**()
- **test_state_roundtrip**()
  - get_state → set_state should preserve state.
- **test_scpn_payload**()
- **test_invalid_params**()
- **test_singlet_statevector_roundtrip_preserves_atp_observable**()
- **test_statevector_import_rejects_invalid_public_contracts**()
- **test_statevector_import_rejects_silent_bond_truncation**()
- **test_checkpoint_without_map_recomputes_quantum_diagnostics**()
- **test_corrupted_checkpoint_zero_norm_fails_on_statevector_export**()
- **test_single_site_pool_rejects_two_site_atp_observable**()
- **test_exact_evolution_rejects_invalid_coupling_contracts**()
- **test_internal_heisenberg_same_site_is_noop**()
- **test_internal_heisenberg_adjacent_gate_preserves_state_norm**()
- **test_internal_heisenberg_nonadjacent_swap_network_preserves_state_norm**()

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

## Module `reservoir.auto_reservoir`

### Class `ReservoirMetrics`
Reservoir quality metrics.

- **summary**()

### Class `AutoCriticalReservoir`
Spiking Liquid State Machine with automatic criticality tuning.

Parameters
----------
n_inputs : int
n_neurons : int
    Reservoir size.
n_outputs : int
    Readout dimension.
threshold : float
    LIF spike threshold.
leak : float
    Membrane leak factor (0-1). Higher = faster decay.
connectivity : float
    Fraction of possible synapses that exist (sparsity).
seed : int

- **__init__**(n_inputs, n_neurons, n_outputs, threshold, leak, connectivity, seed)
- **spectral_radius**()
- **reset**()
- **step**(x)
  - Process one timestep, return reservoir state (spikes).
- **run**(inputs)
  - Run input sequence through reservoir, return state matrix.
- **fit_readout**(states, targets, ridge)
  - Train readout via ridge regression.
- **predict**(states)
  - Predict from reservoir states.
- **train_and_predict**(train_inputs, train_targets, test_inputs)
  - Full pipeline: run train, fit readout, run test, predict.
- **metrics**(inputs)
  - Compute reservoir quality metrics.

---

## Module `residual.blocks`

### Class `MembraneShortcutBlock`
MS-ResNet residual block with membrane shortcut.

Skips the inter-block LIF neuron. Residual connection adds
directly to membrane potential, not to spikes.

Parameters
----------
n_features : int
threshold : float
tau_mem : float

- **__init__**(n_features, threshold, tau_mem, seed)
- **forward**(x)
  - Forward pass: x -> W1 -> LIF -> W2 -> add residual -> LIF -> spikes.
- **reset**()

### Class `SEWBlock`
SEW-ResNet block: activation-before-addition.

spike(W@x) + x instead of spike(W@x + x).
Prevents identity mapping issues in spiking residual networks.

Parameters
----------
n_features : int
threshold : float

- **__init__**(n_features, threshold, seed)
- **forward**(x)
  - Forward: spike(W@x) + x (element-wise, clamped to &#91;0,1&#93;).
- **reset**()

### Class `DeepSNNStack`
Stack of residual blocks for building deep SNNs.

Parameters
----------
n_features : int
n_blocks : int
block_type : str
    'ms' for MembraneShortcut, 'sew' for SEW.

- **__init__**(n_features, n_blocks, block_type)
- **forward**(x)
- **reset**()
- **n_blocks**()
- **depth**()

---

## Module `resilience.fault_suite`

### Class `FaultType`

### Class `FaultModel`
One fault injection configuration.


### Class `FaultResult`
Result of one fault injection run.


### Class `ResilienceReport`
Full fault resilience report.

- **degradation_curve**(fault_type)
  - Get (fault_rate, degradation) pairs for one fault type.
- **most_vulnerable_layer**()
  - Return the layer index with highest average degradation.
- **summary**()

### Class `FaultResilienceSuite`
Systematic fault injection and resilience analysis.

Parameters
----------
eval_fn : callable
    Function(weights) -> accuracy. Takes list of weight matrices,
    returns accuracy in &#91;0, 1&#93;.
weights : list of ndarray
    Baseline (unfaulted) weight matrices.

- **__init__**(eval_fn, weights)
- **baseline_accuracy**()
- **inject_fault**(fault)
  - Apply a fault model to weights, return faulted copies.
- **run_single**(fault)
  - Run one fault injection experiment.
- **sweep**(fault_type, rates, per_layer)
  - Sweep fault rate and optionally per-layer.
- **full_audit**()
  - Run all fault types at standard rates, per-layer.

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

## Module `safety_cert.safety_cert`

### Class `SafetyStandard`

### Class `SILLevel`

### Class `ASILLevel`

### Class `Requirement`
One safety requirement with traceability links.


### Class `TraceabilityMatrix`
Requirement → implementation → verification traceability.

Each requirement links to:
- Implementation artifacts (RTL files, Python modules)
- Verification artifacts (formal proofs, UVM results, tests)

- **__init__**()
- **add_requirement**(req)
- **link_implementation**(req_id, impl_ref)
- **link_verification**(req_id, verif_ref)
- **_update_status**(req)
- **coverage**()
- **open_count**()
- **generate_report**()
  - Generate text traceability report.

### Class `FailureCategory`

### Class `FailureMode`
One failure mode in the FMEDA.

- **safe_failure_fraction**()
  - Fraction of failures that are safe or detected dangerous.

### Class `FMEDA`
Failure Modes, Effects, and Diagnostic Analysis.

Aggregates failure modes for SC neuromorphic modules and computes
Safe Failure Fraction (SFF) and Diagnostic Coverage (DC).

- **__init__**()
- **add_failure_mode**(fm)
- **add_sc_standard_modes**(component)
  - Add standard SC-specific failure modes for a component.
- **total_failure_rate**()
- **safe_failure_fraction**()
  - SFF = (safe + no_effect + DC*dangerous_detected) / total.
- **diagnostic_coverage**()
- **residual_risk_fit**()
  - Dangerous-undetected failure rate (residual risk).
- **sff_by_component**()
  - Per-component safe failure fraction.
- **max_achievable_sil**()
  - Determine max achievable SIL from SFF and DC.
- **generate_report**()

### Class `FormalProperty`
One formally verified property.


### Class `FormalProofCertificate`
Certificate packaging formal verification evidence.

- **add_property**(prop)
- **proven_count**()
- **total_count**()
- **pass_rate**()
- **compute_hash**()
- **generate_report**()

### Class `WCETPath`
Worst-case execution time for one SC computation path.

- **total_cycles**()
- **wcet_ns**(clock_mhz)

### Class `WCETAnalyzer`
Worst-case execution time analysis for SC bitstream paths.

Uses static analysis of the SC pipeline stages:
- LFSR encoding: bitstream_length cycles
- Dot product: num_inputs cycles
- LIF evaluation: fixed (3 cycles)
- AER encoding: num_neurons worst-case
- STP update: 1 cycle

- **analyze**(cls, bitstream_length, num_inputs, num_neurons, has_stp)
- **analyze_multistage**(cls, layers)
  - Analyze a multi-layer SC network.

### Class `ChecklistItem`
One item in a compliance checklist.


### Class `ComplianceChecklist`
Standard-specific compliance checklist generator.

- **generate**(cls, standard)

### Class `CertificationPackage`
Complete certification package output.

- **checklist_coverage**()

### Class `CertificationGenerator`
Top-level generator for safety certification packages.

- **generate**(standard, target_sil, modules, formal_properties, network_config)
  - Generate a complete certification package.

### Class `CCFDefence`
One defence against common cause failures.


### Class `CCFAnalysis`
IEC 61508 Annex D — β-factor estimation for common cause failures.

β = fraction of dangerous failures that are common cause.
Defences reduce β. Lower β → higher achievable SIL.

- **__init__**()
- **mark_implemented**(defence_id)
- **beta_factor**()
  - Resulting β-factor (start at 0.10, reduce by implemented defences).
- **implemented_count**()
- **sil_compatible**(target_sil)
  - Check if β is low enough for the target SIL.

### Class `ProofTestCoverage`
Maps formal proof pass/fail to diagnostic coverage for SIL claims.

- **coverage_from_proofs**(properties)
  - Formal proof coverage = proven / total asserts.
- **dc_to_sil**(dc)
  - Map diagnostic coverage to max SIL per IEC 61508 Table 3.
- **uncovered_modules**(properties, all_modules)
  - Modules with no formal proofs.

### Class `HFTLevel`

### Class `HFTAssessment`
Hardware Fault Tolerance assessment per IEC 61508 Table 2.

- **required_hft**()
  - Determine required HFT from SFF and target SIL.
- **is_simplex_ok**()

### Class `ChangeRecord`
One tracked change affecting safety.


### Class `ChangeImpactTracker`
Tracks changes and their impact on certification artifacts.

- **__init__**()
- **add_change**(change)
- **affected_requirements**()
- **high_risk_count**()
- **needs_re_certification**()

### Class `SafetyManualGenerator`
Generates operator safety manual skeleton per IEC 61508-2 §7.4.15.

- **generate**(product_name, sil_level, modules, wcet_ns)

### Class `SWClass`

### Class `IEC62304Assessment`
IEC 62304 software safety classification for medical devices.

- **requires_unit_testing**()
- **requires_architectural_design**()
- **from_sil**(sil)

### Class `ReliabilityMetrics`
System-level reliability from FMEDA data.

- **mtbf_hours**()
  - Mean Time Between Failures (hours).
- **mtbf_years**()
- **pfh_d**()
  - Probability of dangerous failure per hour.
- **pfh_sil**()
  - Max SIL from PFHd per IEC 61508 Table 3.
- **from_fmeda**(fmeda)

### Class `EvidenceItem`
One item in the evidence bag.


### Class `EvidenceBag`
Manifest of all certification evidence artifacts.

- **__init__**()
- **add**(item)
- **add_from_package**(pkg)
- **file_count**()
- **manifest**()
- **compute_hashes**()

### Class `CrossStandardMapper`
Maps equivalent clauses across IEC 61508 / ISO 26262 / FDA / DO-254.

- **equivalent_clauses**(standard, clause)
- **coverage_overlap**(checklist_a, checklist_b)
  - Count shared compliance coverage between two checklists.

### Class `PropertyGap`
One module with insufficient formal coverage.

- **coverage**()

### Class `FormalPropertyGapDetector`
Detects modules with insufficient formal verification coverage.

- **detect**(cls, properties, required_modules)
- **is_fully_covered**(cls, properties, required_modules)

---

## Module `safety_cert.safety_monitor`

### Class `SafetyLimits`
Configurable safety thresholds matching SV parameters.


### Class `SafetyMonitor`
Software mirror of the hardware neuro_safe_monitor.

Enforces all 6 formally proven properties:
  &#91;P1&#93; monitor_soundness — halt when current/voltage/coherence out of bounds
  &#91;P2&#93; safe_transition — coherence must not decrease (monotone)
  &#91;P3&#93; sc_precision_bound — popcount must be in &#91;0, N&#93;
  &#91;P4&#93; sc_add_preserves_range — SC addition result ≤ denominator
  &#91;P5&#93; lif_membrane_bounded — membrane ≤ v_max
  &#91;P6&#93; correlation_range — |SCC numerator| ≤ denominator

- **reset**()
  - Reset monitor state (equivalent to rst_n pulse).
- **check**(current, voltage, coherence, popcount_k, sc_add_result, membrane, scc_numerator, scc_denominator)
  - Check all 6 safety properties. Returns True if any violation detected.
- **property_names**()
  - Return names of violated properties.

---

## Module `scpn.datastream`

### Class `SCPNDatastream`
In-memory representation of one deterministic SCPN stream.

- **n_steps**()
  - Number of timesteps in the stream.
- **n_layers**()
  - Number of SCPN layer channels in the stream.
- **firing_rates**()
  - Mean spike probability per layer over this stream window.
- **rotation_angles_rad**()
  - ``Ry`` angles for quantum-control bridges: firing rate times pi.
- **quantum_amplitudes**()
  - Real amplitude encoding of firing rates as ``&#91;alpha, beta&#93;`` pairs.
- **to_json_dict**()
  - Serialise the datastream to a stable JSON-compatible mapping.
- **from_json_dict**(cls, payload)
  - Load and validate a datastream from a JSON-compatible mapping.

### Function `generate_scpn_datastream()`
Generate a deterministic 16-layer stream for inter-repository tests.

The probability envelope is a bounded phase oscillator driven by the
canonical SCPN natural frequencies and a small normalised coupling
bias derived from ``K_nm``. The binary spike train is sampled from
that envelope with a local RNG seeded by ``seed``.

### Function `validate_scpn_datastream(stream)`
Validate shape, bounds, and canonical matrix invariants.

### Function `write_scpn_datastream(path, stream)`
Write a stream payload to JSON.

### Function `read_scpn_datastream(path)`
Read a stream payload from JSON.

### Function `_float_scalar_from_payload(value)`
### Function `_int_scalar_from_payload(value)`
### Function `_numeric_array_from_payload(value)`
### Function `_validate_numeric_json_tree(value)`
### Function `generate_scpn_datastream_payload()`
Generate a JSON-compatible stream payload in one call.

---

## Module `scpn.layers.l10_boundary`

### Class `L10_StochasticParameters`

### Class `L10_BoundaryLayer`
Topological firewall with dissonance rejection.

- **__init__**(params)
- **step**(dt, l9_input, external_noise)
- **_integrity**()
- **get_global_metric**()
- **_validate_params**(params)
- **_retrieval_quality**(value)
- **_qec_residual**(l9_input, n_boundary_nodes)
- **_bounded_l9_vector**(value, n_boundary_nodes, name)
- **_memory_complexity_flux**(l9_input)
- **_boundary_context**(l9_input)
- **_noise_vector**(external_noise, n_boundary_nodes)

---

## Module `scpn.layers.l11_morphic`

### Class `L11_StochasticParameters`

### Class `L11_MorphicLayer`
Noospheric spin-glass with memetic spreading dynamics.

- **__init__**(params)
- **step**(dt, l10_input)
- **get_global_metric**()
- **_validate_params**(params)
- **_integrity_signal**(value)
- **_l10_boundary_effect**(cls, l10_input, n_nodes)
- **_noospheric_context**(l10_input)
- **_project_nonnegative_vector**(value, n_nodes, name)
- **_project_rejection_mask**(value, n_nodes)
- **_nonnegative_scalar**(value, name)
- **_unit_scalar**(cls, value, name)
- **_update_info_density**(dt, transmission)

---

## Module `scpn.layers.l12_quantum_info`

### Class `L12_StochasticParameters`

### Class `L12_QuantumInfoLayer`
ENAQT-inspired ecological coherence transport.

- **__init__**(params)
- **step**(dt, l11_input)
- **_von_neumann_entropy**()
- **get_global_metric**()
- **_validate_params**(params)
- **_info_saturation**(value)
- **_l11_noospheric_effect**(l11_input)
- **_gaian_context**(l11_input)
- **_nonnegative_scalar**(value, name)
- **_unit_scalar**(cls, value, name)

---

## Module `scpn.layers.l13_temporal`

### Class `L13_StochasticParameters`

### Class `L13_TemporalLayer`
Temporal binding via cross-correlation within a sliding window.

- **__init__**(params)
- **step**(dt, l12_input)
- **get_global_metric**()
- **_validate_params**(params)
- **_coherence_signal**(coherence, n_channels)
- **_l12_source_sampling_effect**(l12_input)
- **_source_context**(l12_input)
- **_scalar**(value, name)
- **_pearson**(a, b)
- **_max_lag_binding_matrix**(history)

---

## Module `scpn.layers.l14_integration`

### Class `L14_StochasticParameters`
- **__post_init__**()

### Class `L14_IntegrationLayer`
Weighted integration across SCPN layer metrics.

- **__init__**(params)
- **step**(dt, layer_metrics, l13_input)
- **get_global_metric**()
- **_validate_params**(params)
- **_validate_step_inputs**(cls, dt, layer_metrics, l13_input)
- **_normalised_weights**(weights)
- **_metric_vector**(layer_metrics, limit)
- **_finite_mean**(values, name)
- **_l13_bridge_effect**(cls, l13_input)
- **_bridge_context**(l13_input)
- **_finite_scalar**(value, name)
- **_unit_scalar**(cls, value, name)

---

## Module `scpn.layers.l15_meta`

### Class `L15_StochasticParameters`

### Class `L15_MetaLayer`
Self-monitoring meta-cognitive layer with GCI computation.

- **__init__**(params)
- **step**(dt, l14_input)
- **get_global_metric**()
- **_validate_params**(params)
- **_validate_step_inputs**(dt, l14_input, params)
- **_consilium_context**(l14_input)
- **_nonnegative_scalar**(value, name)
- **_unit_scalar**(cls, value, name)

---

## Module `scpn.layers.l16_director`

### Class `L16_StochasticParameters`

### Class `_L16StepInputs`

### Class `L16_DirectorLayer`
Cybernetic closure with PI control and Lyapunov monitoring.

- **__init__**(params)
- **step**(dt, l15_input)
- **get_global_metric**()
- **_validate_params**(params)
- **_validate_step_inputs**(dt, l15_input, params)
- **_validate_boundary_context**(l15_input)
- **_nonnegative_scalar**(value, name)
- **_unit_scalar**(cls, value, name)

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
- **_validate_params**(params)
- **_validate_step_inputs**(dt, external_field, n_qubits)

---

## Module `scpn.layers.l2_neurochemical`

### Class `L2_StochasticParameters`
Parameters for the Stochastic L2 Neurochemical Layer.


### Class `L2_NeurochemicalLayer`
Stochastic implementation of the Neurochemical Signaling Layer.

Models receptor-ligand binding, neurotransmitter dynamics, and
second messenger cascades using bitstream representations.

- **__init__**(params)
- **step**(dt, nt_release, l1_input)
  - Advance the layer by one time step.
- **release_neurotransmitter**(nt_type, amount)
  - Trigger neurotransmitter release.
- **get_global_metric**()
  - Return the global neurochemical activity metric.
- **get_neuromodulation_state**()
  - Return named neurotransmitter levels for external use.
- **_validate_params**(params)
- **_validate_step_inputs**(cls, dt, nt_release, l1_input, n_neurotransmitter_types)
- **_finite_mean**(values, name)

---

## Module `scpn.layers.l3_genomic`

### Class `L3_StochasticParameters`
Parameters for the Stochastic L3 Genomic Layer.


### Class `L3_GenomicLayer`
Stochastic implementation of the Genomic-Epigenomic Layer.

Models gene expression, epigenetic modifications, and bioelectric
pattern formation using bitstream representations.

- **__init__**(params)
- **_init_regulatory_network**()
  - Initialize gene regulatory network with sparse connections.
- **step**(dt, l2_input, bioelectric_signal)
  - Advance the layer by one time step.
- **get_global_metric**()
  - Return the global genomic activity metric.
- **get_ciss_coherence**()
  - Return CISS spin coherence metric.
- **_validate_params**(params)
- **_validate_step_inputs**(cls, dt, l2_input, bioelectric_signal, n_genes)
- **_finite_mean**(values, name)
- **_bioelectric_signal**(values, n_genes)

---

## Module `scpn.layers.l4_cellular`

### Class `L4_StochasticParameters`
Parameters for the Stochastic L4 Cellular Layer.


### Class `L4_CellularLayer`
Stochastic implementation of the Cellular-Tissue Synchronization Layer.

Models collective cellular behavior, gap junction coupling, and
tissue-level pattern formation using bitstream representations.

- **__init__**(params)
- **_init_gap_junctions**()
  - Initialize gap junction connectivity.
- **_build_neighbor_matrix**()
  - Build 2D grid neighbor connectivity matrix.
- **step**(dt, l3_input, external_stimulus)
  - Advance the layer by one time step.
- **get_global_metric**()
  - Return the global synchronization metric (Kuramoto order parameter).
- **get_tissue_pattern**()
  - Return 2D tissue activity pattern.
- **_validate_params**(params)
- **_validate_step_inputs**(cls, dt, l3_input, external_stimulus, n_cells)
- **_finite_mean**(values, name)

---

## Module `scpn.layers.l5_organismal`

### Class `L5_StochasticParameters`
Parameters for the Stochastic L5 Organismal Layer.


### Class `L5_OrganismalLayer`
Stochastic implementation of the Organismal-Psychoemotional Layer.

Models whole-organism integration, autonomic regulation, and
emotional dynamics using bitstream representations.

- **__init__**(params)
- **_init_emotional_attractors**()
  - Initialize emotional attractor states.
- **step**(dt, l4_input, external_event)
  - Advance the layer by one time step.
- **_compute_rmssd**()
  - Compute RMSSD (root mean square of successive differences).
- **get_global_metric**()
  - Return the global organismal coherence metric.
- **get_emotional_valence**()
  - Return current emotional valence.
- **_validate_params**(params)
- **_validate_step_inputs**(cls, dt, l4_input, external_event)
- **_validate_external_event**(cls, external_event)
- **_apply_external_event**(external_event)
- **_dimension_names**(cls)

---

## Module `scpn.layers.l6_ecological`

### Class `L6_StochasticParameters`
Parameters for the Stochastic L6 Ecological Layer.


### Class `L6_EcologicalLayer`
Stochastic implementation of the Ecological-Planetary Layer.

Models planetary-scale electromagnetic fields, Schumann resonances,
and biospheric network dynamics using bitstream representations.

- **__init__**(params)
- **step**(dt, l5_input, solar_activity, lunar_phase)
  - Advance the layer by one time step.
- **get_global_metric**()
  - Return the global planetary coherence metric.
- **get_schumann_spectrum**()
  - Return current Schumann resonance spectrum.
- **get_circadian_time**()
  - Return current circadian time (0-24 hours).
- **_validate_params**(params)
- **_validate_step_inputs**(cls, dt, l5_input, solar_activity, lunar_phase)
- **_finite_mean**(values, name)
- **_l5_organismal_effect**(cls, l5_input)
- **_unit_mean**(cls, values, name)

---

## Module `scpn.layers.l7_symbolic`

### Class `L7_StochasticParameters`
Parameters for the Stochastic L7 Symbolic Layer.


### Class `L7_SymbolicLayer`
Stochastic implementation of the Geometric-Symbolic Layer.

Models sacred geometry patterns, symbolic resonances, and
acupuncture point dynamics using bitstream representations.

- **__init__**(params)
- **step**(dt, l6_input, symbol_input, acupoint_stimulus)
  - Advance the layer by one time step.
- **get_global_metric**()
  - Return the global symbolic coherence metric.
- **get_glyph_vector_normalized**()
  - Return normalized glyph vector for external use.
- **stimulate_meridian**(meridian_id, intensity)
  - Stimulate a specific meridian.
- **get_acupoint_map**()
  - Return clinically common named acupoint activations.
- **_e8_roots**()
- **_validate_params**(params)
- **_symbol_input**(symbol_input)
- **_finite_mean**(value, name)
- **_l6_symbolic_effect**(cls, l6_input)
- **_unit_mean**(value, name)
- **_acupoint_stimulus**(stimulus)

---

## Module `scpn.layers.l8_phase_field`

### Class `L8_StochasticParameters`
- **__post_init__**()

### Class `L8_PhaseFieldLayer`
Stochastic cosmic phase-locking via Kuramoto-coupled PTA oscillators.

- **__init__**(params)
- **step**(dt, l7_input)
- **_order_parameter**()
- **_memory_imprint_drive**()
- **get_global_metric**()
- **_validate_params**(params)
- **_glyph_drive**(glyph_vector)
- **_l7_phase_drive**(cls, l7_input)
- **_nonnegative_scalar**(value, name)

---

## Module `scpn.layers.l9_memory`

### Class `L9_StochasticParameters`

### Class `L9_MemoryLayer`
Hopfield associative memory with stochastic bitstream encoding.

- **__init__**(params)
- **store**(pattern)
  - Hebbian imprint: W += pattern ⊗ pattern.
- **step**(dt, l8_input, boundary_cue, ebs_context)
- **_retrieval_quality**()
- **get_global_metric**()
- **_validate_params**(params)
- **_pattern_vector**(pattern)
- **_cosmic_alignment**(value)
- **_l8_phase_reference_drive**(cls, l8_input)
- **_memory_imprint_drive**(payload)
- **_boundary_cue_vector**(boundary_cue)
- **_boundary_context**(ebs_context)
- **_holographic_entropy**(activation)

---

## Module `scpn.params`

### Function `build_knm_matrix(n_layers)`
Build the Knm inter-layer coupling matrix.

Construction: exponential decay baseline, calibration anchor overrides,
cross-hierarchy boosts, symmetrisation, zero diagonal.

---

## Module `security.checkpoint_loading`

### Class `CheckpointTrustError`
Raised when a checkpoint is not present in the trusted digest set.


### Function `_checkpoint_digest(path)`
### Function `safe_load_checkpoint(path)`
Load a tensor/state-dict checkpoint only after SHA-256 verification.

The trust map accepts either the file name or the resolved full path as key.
PyTorch is always invoked with ``weights_only=True`` after digest validation.

### Function `safe_load_legacy_checkpoint(path)`
Load a metadata checkpoint through pickle only after SHA-256 verification.

Use this only for legacy internal checkpoints whose dictionaries contain
non-state-dict metadata and cannot yet be represented by ``weights_only``.

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

## Module `security.side_channel_benchmark`

### Class `SideChannelBenchmarkError`
Raised when side-channel benchmark inputs or outputs are invalid.


### Class `SideChannelBenchmarkArm`
One benchmark arm with class-activity leakage proxy evidence.


### Class `SideChannelBenchmarkRecord`
Per-sample benchmark record with realised protected probability.


### Class `SideChannelDeployManifest`
Deploy/evidence manifest for an analytic side-channel benchmark.


### Class `SideChannelBenchmarkReport`
Analytic baseline-versus-protected side-channel benchmark report.


### Function `run_side_channel_leakage_benchmark()`
Compare correlated baseline streams against activity-balanced streams.

### Function `write_side_channel_benchmark_report(output_path)`
Run the analytic benchmark and write a canonical JSON artifact.

### Function `_normalise_probabilities(probabilities)`
### Function `_normalise_labels(labels)`
### Function `_correlated_activity_fixture_stream(probability, bitstream_length)`
### Function `_report_payload(report)`
### Function `_with_artifact_path(report, path)`
### Function `_deploy_manifest_payload(manifest)`
### Function `_arm_payload(arm)`
### Function `_class_proxy_payload(proxy)`
---

## Module `security.side_channel_metrics`

### Class `SideChannelMetricError`
Raised when side-channel metric inputs are malformed or unsupported.


### Class `SwitchingActivitySummary`
Transition-count summary for a rectangular binary bitstream matrix.


### Class `ClassActivityProxy`
Class-conditioned analytic proxy for activity-dependent leakage.

``label_activity_correlation`` is Pearson correlation between numeric labels
and per-sample mean switching rate. It is ``None`` when either side has zero
variance, avoiding fabricated correlation claims.


### Function `compute_switching_activity(bitstreams)`
Compute per-stream switching activity for rows of binary bitstreams.

### Function `compute_class_activity_proxy(bitstreams_by_sample, labels)`
Summarise class-conditioned switching activity for simulated samples.

### Function `_normalise_sample_collection(bitstreams_by_sample)`
### Function `_normalise_bitstream_matrix(bitstreams)`
### Function `_normalise_bit(value)`
### Function `_normalise_labels(labels)`
### Function `_pearson_correlation(labels, rates)`
---

## Module `security.thermal_sc_encoding`

### Class `ThermalSCEncodingError`
Raised when thermal side-channel encoder inputs violate the contract.


### Class `ThermalSCEncodingConfig`
Configuration for deterministic activity-balanced SC encoding.


### Class `ActivityBalancedEncoding`
A single activity-shaped stochastic-computing bitstream.


### Class `ActivityBalancedEncodingSummary`
Batch-level analytic evidence for activity-shaped encodings.


### Class `ActivityBalancedEncodingBatch`
Batch result for activity-shaped stochastic encodings.


### Function `encode_activity_balanced_probability(probability, config)`
Encode one probability with distributed ones and deterministic rotation.

### Function `encode_activity_balanced_probabilities(probabilities, config)`
Encode a probability batch and attach analytic class-activity evidence.

### Function `_validate_config(config)`
### Function `_validate_probability(probability)`
### Function `_distribute_ones(ones, bitstream_length)`
### Function `_activity_preserving_rotation_offset(bitstream, config, stream_index)`
### Function `_build_dummy_bitstreams(config, stream_index)`
### Function `_candidate_offsets(desired, distance, bitstream_length)`
### Function `_rotate(bitstream, offset)`
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

## Module `sensors.dvs`

### Class `DVSLoader`
Load and preprocess DVS event camera data.

Parameters
----------
width : int
    Sensor width in pixels.
height : int
    Sensor height in pixels.

- **n_pixels**()
- **from_numpy**(events)
  - Load events from structured numpy array.
- **from_tonic**(dataset_name, index)
  - Load events from a Tonic dataset (requires tonic package).

### Function `events_to_spike_trains(events, width, height, dt_us, duration_us)`
Convert DVS events to binary spike train matrix.

Parameters
----------
events : structured ndarray with x, y, t, p fields
width, height : int
    Sensor dimensions.
dt_us : float
    Time bin width in microseconds (default 1000 = 1ms).
duration_us : float, optional
    Total duration. If None, inferred from event timestamps.

Returns
-------
ndarray of shape (n_bins, width * height * 2)
    Binary spike trains. Channels: &#91;ON pixels, OFF pixels&#93;.

### Function `events_to_frames(events, width, height, dt_us, duration_us)`
Convert DVS events to event count frames.

Parameters
----------
events : structured ndarray
width, height : int
dt_us : float
    Frame duration in microseconds (default 10000 = 10ms).
duration_us : float, optional

Returns
-------
ndarray of shape (n_frames, 2, height, width)
    Event count frames with ON and OFF channels.

---

## Module `serve.server`

### Class `SpikeServer`
Streaming SNN inference server.

Parameters
----------
network : SCNetwork or Network
    The SNN to run.
host : str
    Bind address (default '0.0.0.0').
port : int
    Listen port (default 8001).

- **__init__**(network, host, port)
- **step**(inputs)
  - Run one network timestep and return output spikes.
- **start**(blocking)
  - Start the HTTP server.
- **stop**()
  - Shut down the server.

---

## Module `sleep.circadian_optimizer`

### Class `Chronotype`
Sleep chronotype labels (after Michael Breus' model).


### Class `CircadianProfile`
A chronotype's circadian parameters.

Attributes
----------
chronotype : Chronotype
    The chronotype label.
bedtime_hour : float
    Ideal bedtime in decimal hours (0-24, can exceed 24 for next-day).
wake_hour : float
    Ideal wake time in decimal hours.
default_protocol : str
    Name of the recommended sleep audio protocol.
melatonin_peak_hour : float
    Hour at which endogenous melatonin peaks (typically ~2 h after
    bedtime onset for most chronotypes).
core_body_temp_nadir_hour : float
    Hour of the core body temperature nadir (~2 h after melatonin peak).


### Class `CircadianOptimizer`
Chronotype-aware circadian rhythm optimizer.

Parameters
----------
chronotype : Chronotype
    The user's chronotype.

Example::

    opt = CircadianOptimizer(Chronotype.BEAR)
    profile = opt.get_profile()
    print(opt.melatonin_level(23.0))  # near-peak for a Bear

- **__init__**(chronotype)
- **get_profile**()
  - Return the full circadian profile for the configured chronotype.
- **get_sleep_window**()
  - Return ``(bedtime_hour, wake_hour)``.
- **get_recommended_protocol**()
  - Return the default protocol name for this chronotype.
- **is_in_sleep_window**(hour)
  - Check whether *hour* (0-24) falls inside the sleep window.
- **melatonin_level**(hour)
  - Estimate melatonin level at *hour* using a sinusoidal model.
- **to_dict**()
  - Serialise the optimizer state to a plain dict.

---

## Module `sleep.protocol_library`

### Class `StageAudioParams`
Audio-entrainment parameters for a single sleep stage.

Attributes
----------
binaural_hz : float
    Binaural beat frequency (Hz).
noise_color : str
    Background noise colour (``"pink"``, ``"brown"``, ``"white"``, etc.).
base_freq_hz : float
    Carrier / base tone frequency (Hz).
volume : float
    Relative volume in ``&#91;0, 1&#93;``.
isochronic_hz : float
    Isochronic pulse frequency (Hz); 0 disables.
spatial_rotation : float
    Spatial audio rotation speed in degrees per second.


### Class `SleepProtocol`
A named sleep-entrainment protocol.

Attributes
----------
name : str
    Human-readable identifier (must match the registry key).
description : str
    Short description of the protocol's therapeutic goal.
stage_audio : Dict&#91;SleepStage, StageAudioParams&#93;
    Audio parameters keyed by target stage.
stage_targets : Dict&#91;SleepStage, float&#93;
    Target fraction of total sleep time per stage (must sum to 1.0).
total_duration_min : float
    Recommended session length in minutes.

- **get_audio_for_stage**(stage)
  - Return audio parameters for *stage*, falling back to WAKE params.
- **get_target_stage**(progress)
  - Return the ideal stage for a given session *progress* in &#91;0, 1&#93;.
- **to_dict**()
  - Serialise the protocol to a plain dict.

### Function `_build_insomnia_relief()`
### Function `_build_jet_lag_reset()`
### Function `_build_deep_sleep_boost()`
### Function `_build_rem_enhancement()`
### Function `_build_shift_worker()`
### Function `_build_power_nap()`
### Function `get_protocol(name)`
Look up a protocol by name.  Raises ``KeyError`` if not found.

### Function `list_protocols()`
Return a sorted list of all available protocol names.

---

## Module `sleep.report_generator`

### Class `SleepReport`
Aggregate report for a completed sleep session.

Attributes
----------
total_duration_min : float
    Total session length in minutes.
sleep_onset_latency_min : float
    Minutes from session start until the first non-WAKE epoch.
sleep_efficiency_pct : float
    Percentage of total time spent asleep (non-WAKE).
quality_score : float
    Composite quality score in ``&#91;0, 100&#93;``.
stage_durations_min : Dict&#91;str, float&#93;
    Time (minutes) per stage.
stage_percentages : Dict&#91;str, float&#93;
    Percentage of total time per stage.
stage_targets : Dict&#91;str, float&#93;
    Protocol target percentages for comparison.
hypnogram : List&#91;int&#93;
    Stage codes per epoch.
wakeups : int
    Number of WAKE epochs that occurred after initial sleep onset.
reinductions : int
    Number of re-induction sequences triggered.
recommendations : List&#91;str&#93;
    Plain-language suggestions.
grade : str
    Letter grade (A-F).


### Class `SleepReportGenerator`
Generates a :class:`SleepReport` from a completed optimiser session.

- **generate**(optimizer)
  - Analyse *optimizer*'s tick history and return a report.

---

## Module `sleep.sleep_optimizer`

### Class `SleepOptimizerConfig`
Tuneable knobs for the optimiser loop.


### Class `SleepTick`
Snapshot produced every ``stage_check_interval`` samples.

Attributes
----------
tick : int
    Monotonic tick counter.
elapsed_min : float
    Wall-clock minutes since session start.
current_stage : SleepStage
    Detected stage at this tick.
target_stage : SleepStage
    Protocol's ideal stage at this progress point.
stage_match : bool
    Whether current == target.
audio_params : StageAudioParams
    Audio parameters being delivered.
band_powers : Dict&#91;str, float&#93;
    Most recent EEG band-power decomposition.
reinduction_active : bool
    Whether a re-induction sequence is currently running.


### Class `SleepOptimizer`
Closed-loop sleep optimiser.

Parameters
----------
protocol : SleepProtocol or str
    The protocol instance (or its registry name) to follow.
config : SleepOptimizerConfig, optional
    Operational parameters.

Example::

    opt = SleepOptimizer("insomnia_relief")
    opt.start_session()
    for sample in eeg_stream:
        opt.add_sample(sample)
        tick = opt.check_and_adapt()
        if tick is not None:
            apply_audio(tick.audio_params)
    report = opt.stop_session()

- **__init__**(protocol, config)
- **start_session**()
  - Begin a new optimisation session, resetting all state.
- **stop_session**()
  - End the current session and return the full tick history.
- **add_sample**(sample)
  - Feed a single EEG voltage sample.
- **add_samples**(samples)
  - Feed an array of EEG voltage samples.
- **check_and_adapt**()
  - Run stage detection and protocol adaptation.
- **get_history**()
  - Return a copy of all recorded ticks.
- **get_stage_durations**()
  - Compute time (minutes) spent in each detected stage.
- **get_hypnogram**()
  - Return the detected-stage sequence as a list of integer codes.
- **get_state**()
  - Return a summary dict of the optimiser's current state.

---

## Module `sleep.sleep_stage_detector`

### Class `SleepStage`
AASM sleep-stage labels.


### Class `DetectorConfig`
Parameters for the sleep-stage detector.


### Class `SleepStageDetector`
Real-time sleep-stage detector from single-channel EEG.

Usage::

    det = SleepStageDetector()
    for sample in eeg_stream:
        det.add_sample(sample)
        stage = det.detect()
        if stage is not None:
            print(stage.name)

- **__init__**(config)
- **add_sample**(sample)
  - Append a single EEG voltage sample to the internal buffer.
- **add_samples**(samples)
  - Append an array of EEG voltage samples.
- **detect**()
  - Return the smoothed sleep-stage classification, or ``None`` if
- **get_band_powers**()
  - Return the most recently computed band-power dict, or ``None``.
- **reset**()
  - Clear all internal state.
- **_compute_band_powers**()
  - Compute absolute band powers from the current buffer via FFT.
- **_classify**(power_vec)
  - Classify by cosine similarity to canonical signatures.
- **_smooth**()
  - Majority-vote smoothing over the recent stage history.

---

## Module `snn_optimizer.passes`

### Class `LayerNode`
One layer in the SNN computation graph.

- **n_params**()

### Class `SNNGraph`
SNN computation graph: sequence of layers.

- **total_params**()
- **total_neurons**()
- **copy**()

### Class `PassResult`
Result of one optimization pass.


### Class `OptimizationReport`
Report from running all optimization passes.

- **compression_ratio**()
- **summary**()

### Function `dead_neuron_elimination(graph, threshold)`
Remove neurons that never fire (firing rate below threshold).

Requires firing_rates to be set on each layer (from profiling).
Removes rows from weight matrices and corresponding columns from
the next layer's weight matrix.

### Function `layer_fusion(graph)`
Fuse adjacent linear layers with compatible dimensions.

If two consecutive layers have no nonlinearity between them
(both LIF with same type), fuse W2 @ W1 into a single layer.
Caveat: this is valid only when the intermediate layer has
effectively linear behavior (high threshold, no spikes).

### Function `redundancy_elimination(graph, correlation_threshold)`
Merge neurons with near-identical weight vectors.

If two neurons in the same layer have weight correlation > threshold,
merge them: keep one, remove the other, scale outgoing weights by 2.

### Function `optimize(graph, passes)`
Run optimization passes on an SNN graph.

Parameters
----------
graph : SNNGraph
passes : list of str, optional
    Pass names to run. Default: all passes.
    Options: 'dead_neuron_elimination', 'layer_fusion', 'redundancy_elimination'

Returns
-------
(optimized_graph, OptimizationReport)

---

## Module `solvers.exact_lif`

### Class `ExactLIFSolver`
Event-driven exact integration for LIF neurons.

Reference: Rotter, S. & Diesmann, M. (1999). Biol. Cybern. 81:381–402.

- **evolve_to_time**(v0, t, current)
  - Compute V(t) given initial voltage v0 and constant current.
- **next_spike_time**(v0, current)
  - Compute time until next spike under constant current.
- **isi**(current)
  - Compute the inter-spike interval for constant current.
- **firing_rate**(current)
  - Compute steady-state firing rate (Hz) for constant current.
- **simulate**(current, t_end, v0)
  - Simulate LIF with constant current from t=0 to t=t_end.

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

## Module `solvers.ode`

### Class `ODESolver`
Base class for ODE solvers: dy/dt = f(t, y).

- **step**(f, y, t, dt)
  - Advance one step. Returns (y_new, dt_used).

### Class `EulerSolver`
Forward Euler — O(h).

y_{n+1} = y_n + h * f(t_n, y_n)

- **step**(f, y, t, dt)

### Class `HeunSolver`
Heun's method (improved Euler / explicit trapezoidal) — O(h²).

k1 = f(t_n, y_n)
k2 = f(t_n + h, y_n + h*k1)
y_{n+1} = y_n + h/2 * (k1 + k2)

- **step**(f, y, t, dt)

### Class `RK4Solver`
Classical Runge-Kutta — O(h⁴).

k1 = f(t, y)
k2 = f(t + h/2, y + h/2 * k1)
k3 = f(t + h/2, y + h/2 * k2)
k4 = f(t + h, y + h * k3)
y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)

Reference: Kutta, W. (1901). Z. Math. Phys. 46:435–453.

- **step**(f, y, t, dt)

### Class `DormandPrinceSolver`
Dormand-Prince adaptive RK45 — embedded pair for step-size control.

Uses the 7-stage, 5th-order solution with a 4th-order embedded error
estimate. Step size is adapted to maintain local truncation error
below the specified tolerance.

Reference: Dormand, J.R. & Prince, P.J. (1980). J. Comput. Appl. Math. 6:19–26.

- **__init__**(atol, rtol, max_factor, min_factor, safety)
- **step**(f, y, t, dt)
- **integrate**(f, y0, t_span, dt0)
  - Integrate over &#91;t0, tf&#93;. Returns (t_array, y_array).

### Class `ExponentialEuler`
Exponential Euler for linear ODEs: dy/dt = A*y + b.

Exact solution: y(t+dt) = exp(A*dt) * y(t) + (exp(A*dt) - I) * A^{-1} * b

For scalar diagonal systems (like LIF):
y(t+dt) = y_rest + (y(t) - y_rest) * exp(-dt/tau) + R*I * (1 - exp(-dt/tau))

The callable f must return A*y + b; the solver extracts the decay constant.

- **__init__**(tau, y_rest, r_m)
- **step**(f, y, t, dt)

### Function `get_solver(name)`
Factory function: return an ODE solver by name.

Supported names: 'euler', 'heun', 'rk4', 'dp45', 'exponential_euler',
'rosenbrock', and 'rosenbrock_euler'.

---

## Module `solvers.stiff`

### Class `RosenbrockEuler`
Linearly implicit one-stage Rosenbrock solver.

The step solves ``(I - gamma*h*J) k = h*f(t, y)`` and returns
``y + k``. With ``gamma=1`` this is the Rosenbrock-Euler method:
first-order, L-stable for linear stiff decay, and suitable as a
stiffness-specific alternative path for small neuron state vectors.

Reference: Hairer, E. & Wanner, G. (1996). Solving ODEs II. Springer.

- **__init__**(gamma, jacobian_epsilon)
- **step**(f, y, t, dt)

### Class `ImplicitEuler`
Backward (implicit) Euler — L-stable, 1st order.

y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

Solved via fixed-point iteration: for stiff neuron ODEs,
3–5 iterations typically suffice.

Reference: Hairer, E. & Wanner, G. (1996). Solving ODEs II. Springer.

- **__init__**(max_iterations, tol)
- **step**(f, y, t, dt)

### Class `TrapezoidalRule`
Trapezoidal rule (Crank-Nicolson) — A-stable, 2nd order.

y_{n+1} = y_n + h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))

Higher-order than implicit Euler while retaining A-stability.
Solved via fixed-point iteration.

Reference: Crank, J. & Nicolson, P. (1947). Proc. Camb. Phil. Soc. 43:50–67.

- **__init__**(max_iterations, tol)
- **step**(f, y, t, dt)

### Function `_finite_difference_jacobian(f, t, y, f_y, epsilon)`
---

## Module `solvers.symplectic`

### Class `StormerVerlet`
Störmer-Verlet (velocity Verlet) — symplectic 2nd order.

For separable Hamiltonians H = T(p) + V(q):
    p_{1/2} = p_n - (h/2) * ∇V(q_n)
    q_{n+1} = q_n + h * ∇T(p_{1/2})
    p_{n+1} = p_{1/2} - (h/2) * ∇V(q_{n+1})

Here we split the state y = &#91;q, p&#93; where q are position-like (voltage)
and p are momentum-like (recovery/gating) variables.

Reference: Hairer, E. et al. (2006). Geometric Numerical Integration. Springer.

- **step**(f, y, t, dt)

### Class `LeapfrogSolver`
Leapfrog (kick-drift-kick) — symplectic 2nd order.

Equivalent to Störmer-Verlet but staggered:
    p_{n+1/2} = p_n + (h/2) * f_p(q_n)
    q_{n+1} = q_n + h * f_q(p_{n+1/2})
    p_{n+1} = p_{n+1/2} + (h/2) * f_p(q_{n+1})

State vector: y = &#91;q₀..q_{n-1}, p₀..p_{n-1}&#93;

Reference: Yoshida, H. (1990). Phys. Lett. A 150:262–268.

- **step**(f, y, t, dt)

---

## Module `sources.bitstream_current_source`

### Class `BitstreamCurrentSource`
Multi-channel bitstream current source.

- Takes scalar inputs x_i in &#91;x_min, x_max&#93;
- Encodes each into a bitstream via BitstreamEncoder
- Passes them through BitstreamSynapses
- Decodes the realised post-synaptic bitstreams into a per-cycle
  current trace for neuron simulation.

Static inputs and weights are encoded once at construction time.
The stochastic realisation is then fixed until a new source is
constructed with different parameters or seeds.

- **__post_init__**()
- **_apply_bipolar_xnor**()
- **_map_bipolar_to_current**(value)
- **_decode_current_trace**()
- **reset**()
  - Reset the realised current trace cursor to its first timestep.
- **current_trace**()
  - Return the realised per-cycle decoded current trace.
- **step**()
  - Return the current I_t at the current time index and advance.
- **full_current_estimate**()
  - Return the mean current over the realised bitstream duration.

---

## Module `sources.quantum_entropy`

### Class `QuantumEntropySource`
Generates entropy based on simulated Quantum Measurement Collapse.
Used to inject 'True' (Simulated) Quantum Indeterminacy into Neural Models.

Physics:
- Maintains a Qubit State |psi>
- Applies Hadamard (Superposition) and Phase Rotations
- Measures (Collapse) to generate noise

- **__post_init__**()
- **_hadamard**()
  - Apply Hadamard gate H = (1/√2)&#91;&#91;1,1&#93;,&#91;1,-1&#93;&#93; to each qubit.
- **_measure**()
  - Apply Hadamard, measure via Born rule, collapse state.
- **sample_normal**(mean, std)
  - Two independent measurements → Box-Muller → Gaussian sample.
- **sample**()
  - Return one default normal sample from the simulated measurement source.

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

## Module `spike_codec.aer_codec`

### Class `AERCompressionResult`
Compression result with AER codec metrics.


### Class `AERSpikeCodec`
AER spike codec: event-list encoding for sparse spike data.

Converts spike raster (T, N) to a compact stream of (timestamp,
neuron_id) events. Delta-encodes timestamps for further compression.

Parameters
----------
timestamp_bits : int
    Bits for delta-coded timestamps. 16 = max gap of 65535 samples.
    Larger windows between spikes use escape codes.
neuron_bits : int
    Bits for neuron ID. Auto-sized from N if 0.

- **__init__**(timestamp_bits, neuron_bits)
- **compress**(spikes)
  - Compress spike raster to AER event stream.
- **decompress**(data, T, N)
  - Decompress AER event stream to spike raster.

---

## Module `spike_codec.codec`

### Class `CompressionResult`
Result of spike train compression.

- **summary**()

### Class `SpikeCodec`
ISI spike train codec with configurable entropy backend.

Compression strategy:
1. Extract per-neuron spike times from binary raster
2. Compute inter-spike intervals (ISIs) per neuron
3. Encode ISIs with chosen backend:
   'varint': LEB128 variable-length integers (fast, simple)
   'huffman': Adaptive Huffman (30-60% smaller on dense data)

Each neuron is encoded independently. No inter-channel modeling.
For inter-channel compression, use DeltaSpikeCodec.

Parameters
----------
mode : str
    'lossless' (exact reconstruction) or 'lossy' (preserve rates only).
timing_precision : int
    For lossy mode: quantize spike times to this resolution.
entropy : str
    'varint' (default) or 'huffman'.

- **__init__**(mode, timing_precision, entropy)
- **compress**(spikes)
  - Compress a spike raster.
- **decompress**(data, T, N)
  - Decompress to spike raster.
- **_quantize_timing**(spikes)
- **_pick_entropy**(n_spikes, total_bins)
  - Auto-select entropy backend based on data density.
- **_encode_events**(events, T, N)
  - Encode spike events using ISI + auto-selected entropy backend.
- **_encode_events_huffman**(events, T, N)
  - Encode events using Huffman-coded ISIs.
- **_decode_events**(data, N)
  - Decode ISI-encoded spike events (auto-detects entropy backend).
- **_decode_events_huffman**(data, N)
  - Decode Huffman-coded ISI events.
- **_encode_varint**(value)
  - Encode integer using variable-length encoding (LEB128-style).
- **_decode_varint**(data, pos)
  - Decode variable-length integer, return (value, new_position).

---

## Module `spike_codec.delta_codec`

### Class `DeltaCompressionResult`
Compression result with delta coding metrics.


### Class `DeltaSpikeCodec`
Delta spike codec: compress inter-channel XOR residuals.

Channels are grouped spatially. Within each group, one reference
channel is transmitted raw; others are XOR'd against the reference
and ISI-compressed. When channels are correlated, the XOR residuals
are much sparser than the raw data.

Parameters
----------
group_size : int
    Channels per group. Larger groups = more sharing but weaker
    correlation with distant channels. 4-16 typical for probes.
mode : str
    'lossless' or 'lossy' for the underlying ISI codec.
timing_precision : int
    For lossy mode: quantize timing resolution.

- **__init__**(group_size, mode, timing_precision)
- **compress**(spikes)
  - Compress spike raster using inter-channel delta coding.
- **decompress**(data, T, N)
  - Decompress delta-coded spike raster.

---

## Module `spike_codec.entropy`

### Class `HuffmanEncoder`
Encode integer streams using adaptive Huffman coding.

Builds code table from the input data, stores table in header,
then encodes symbols as variable-length bit sequences.

- **encode**(values)
  - Encode integer list to compressed bytes.
- **decode**(data, n_symbols)
  - Decode compressed bytes to integer list.

### Function `_build_huffman_table(symbols)`
Build Huffman codes from a symbol stream.

Returns dict mapping symbol → (code_bits, code_length).
Uses package-merge algorithm for optimal length-limited codes.

### Function `_walk_tree(node, lengths, depth)`
### Function `_canonical_codes(lengths)`
Generate canonical Huffman codes from bit lengths.

Canonical codes: sorted by (length, symbol), sequential assignment.
Decoder only needs the length table to reconstruct codes.

---

## Module `spike_codec.predictive_codec`

### Class `PredictiveCompressionResult`
Compression result with predictive coding metrics.


### Class `_RatePredictor`
Per-channel EMA rate predictor (legacy, kept for reference).

- **__init__**(n_channels, alpha, threshold)
- **predict**()
- **update**(actual)
- **reset**()

### Class `PredictiveSpikeCodec`
Predictive spike codec: compress prediction errors, not raw spikes.

Four predictor modes:
    'ema' (default): float EMA rate tracking + threshold comparison.
    'lfsr': Q8.8 fixed-point rate + LFSR comparator. Bit-true with
            sc_bitstream_encoder.v — maps directly to Verilog RTL.
    'context': Markov context predictor. Hashes last K spike states
               per channel, predicts from accumulated statistics.
    'world_model': Learnable autoregressive predictor (LMS-trained).
               Predicts spike&#91;t&#93; from spike&#91;t-K:t&#93; via linear model
               with sigmoid activation. Learns cross-channel correlations.

Compression pipeline:
    1. For each timestep t:
       a. predicted&#91;t&#93; = predictor.predict()
       b. error&#91;t&#93; = actual&#91;t&#93; XOR predicted&#91;t&#93;
       c. predictor.update(actual&#91;t&#93;)
    2. ISI-compress the error matrix (sparser than raw spikes)
    3. Pack with header (predictor params for decoder sync)

Parameters
----------
alpha : float
    EMA smoothing factor (ema mode). Ignored in lfsr/context mode.
threshold : float
    Spike prediction threshold (ema mode). Ignored in lfsr/context mode.
predictor : str
    'ema', 'lfsr', or 'context'.
alpha_q8 : int
    Q8.8 smoothing factor for lfsr mode. 1 = 1/256 ≈ 0.004.
seed : int
    LFSR seed for lfsr mode (non-zero, 16-bit).
context_bits : int
    Context history length for context mode (default 8 = last 8 spikes).
base_mode : str
    'lossless' or 'lossy' for the underlying ISI codec.
timing_precision : int
    For lossy mode: quantize timing resolution.

- **__init__**(alpha, threshold, predictor, alpha_q8, seed, context_bits, base_mode, timing_precision)
- **compress**(spikes)
  - Compress spike raster using predictive error coding.
- **decompress**(data, T, N)
  - Decompress to spike raster.

### Function `_predict_and_xor_context(spikes, N, context_bits)`
Context-model predict-XOR loop. Returns (errors, correct_count).

Per-channel Markov predictor: hash last K spike states as context key,
predict based on accumulated statistics for that context.
Captures temporal patterns (bursts, oscillations, refractory periods)
that EMA misses.

### Function `_xor_and_recover_context(errors, N, context_bits)`
Context-model XOR-recover loop for decoder.

### Function `_predict_and_xor(spikes, N, alpha, threshold)`
EMA predict-XOR loop. Returns (errors, correct_count).

### Function `_xor_and_recover(errors, N, alpha, threshold)`
EMA XOR-recover loop for decoder.

### Function `_lfsr16_step(reg)`
One step of 16-bit Galois LFSR. Matches sc_bitstream_encoder.v.

### Function `_predict_and_xor_lfsr(spikes, N, alpha_q8, seed)`
LFSR-based predict-XOR loop. Bit-true with Verilog.

Uses Q8.8 fixed-point rate tracking + LFSR comparator for prediction.
Same polynomial and step semantics as sc_bitstream_encoder.v.

Parameters
----------
spikes : (T, N) int8
N : int
alpha_q8 : int
    Q8.8 smoothing factor. 1 = 1/256 ≈ 0.004.
seed : int
    LFSR seed (non-zero, 16-bit).

### Function `_xor_and_recover_lfsr(errors, N, alpha_q8, seed)`
LFSR-based XOR-recover loop for decoder.

---

## Module `spike_codec.registry`

### Function `get_codec(name)`
Get a codec by name.

Parameters
----------
name : str
    One of: 'isi', 'predictive', 'delta', 'streaming', 'aer'.
**kwargs
    Passed to the codec constructor.

Returns
-------
Codec instance with compress/decompress methods.

### Function `list_codecs()`
List available codec names.

### Function `recommend_codec(n_channels, firing_rate, latency_ms, correlated, neuromorphic)`
Recommend a codec based on data characteristics.

Parameters
----------
n_channels : int
    Number of recording channels.
firing_rate : float
    Mean firing rate in Hz (per neuron).
latency_ms : float
    Maximum acceptable latency in milliseconds.
correlated : bool
    True if nearby channels are spatially correlated.
neuromorphic : bool
    True if target is neuromorphic hardware (Loihi, SpiNNaker).

Returns
-------
str — codec name

---

## Module `spike_codec.streaming_codec`

### Class `StreamingCompressionResult`
Compression result with streaming codec metrics.


### Class `StreamingSpikeCodec`
Streaming spike codec: fixed-latency, independently decodable frames.

Each time window is compressed as a self-contained frame. No inter-frame
dependencies. Worst-case latency = window_size samples.

Parameters
----------
window_size : int
    Samples per frame. 20 = 1ms at 20kHz (typical BCI).
    Smaller = lower latency but less compression.

- **__init__**(window_size)
- **compress**(spikes)
  - Compress spike raster into independently decodable frames.
- **decompress**(data, T, N)
  - Decompress streaming frames to spike raster.
- **compress_frame**(window)
  - Compress a single time window (for real-time streaming).
- **decompress_frame**(frame)
  - Decompress a single frame.

### Function `_pack_window(window)`
Pack a (W, N) spike window into compact frame bytes.

Format per frame:
    n_channels: uint16
    window_size: uint16
    For each channel:
        If silent (no spikes): store nothing, mark in skip bitmap
        If active: store raw bitmask (ceil(W/8) bytes)

    skip_bitmap: ceil(N/8) bytes — 1 = silent, 0 = active
    active_bitmasks: concatenated, ceil(W/8) bytes each

### Function `_unpack_window(frame, offset)`
Unpack one frame. Returns (window, new_offset).

---

## Module `spike_codec.waveform_codec`

### Class `WaveformCompressionResult`
Result of waveform compression.


### Class `WaveformCodec`
End-to-end neural waveform codec.

Pipeline: detect → separate → compress each component optimally.

Parameters
----------
threshold_sigma : float
    Spike detection threshold in units of per-channel noise sigma.
    Typical: 4.0-5.0 (4 sigma catches ~99.99% of noise).
snippet_samples : int
    Waveform samples to extract around each spike (before + after peak).
max_templates : int
    Maximum number of spike waveform templates to maintain.
template_threshold : float
    Correlation threshold for template matching (0-1).
quantize_bits : int
    Background signal quantization (fewer bits = more compression).
mode : str
    Compression mode controlling what is preserved:
    - ``"full"``: spike timing + waveform templates + background LFP (~137x)
    - ``"waveform"``: spike timing + waveform templates, no background (~1700x)
    - ``"spike"``: spike timing only, Neuralink-equivalent (~4500x)

- **__init__**(threshold_sigma, snippet_samples, max_templates, template_threshold, quantize_bits, mode)
- **compress**(waveform)
  - Compress raw electrode waveform.
- **_detect_spikes**(waveform, thresholds)
  - Threshold-crossing spike detection with refractory period.
- **_extract_snippets**(waveform, times_per_ch, N)
  - Extract waveform clips around detected spikes.
- **_template_match**(snippets)
  - Build template library and match snippets.
- **_compress_snippets**(templates, template_ids, residuals)
  - Compress templates + IDs + quantised residuals.
- **_extract_background**(waveform, times_per_ch)
  - Extract low-frequency background (remove spikes).
- **_compress_background**(background)
  - Delta-encode + quantize background signal.

---

## Module `spike_dsp.filters`

### Class `SpikeFIR`
Finite Impulse Response filter in spike domain.

Computes weighted sum of delayed input spike trains.
Output at time t = sum(coefficients&#91;k&#93; * input&#91;t-k&#93;) > threshold.

Parameters
----------
coefficients : ndarray
    FIR tap weights.
threshold : float
    Output spike threshold (on weighted sum).

- **filter**(spikes)
  - Apply FIR filter to spike train.

### Class `SpikeIIR`
Infinite Impulse Response filter using leaky integration.

Membrane-like dynamics: state = decay * state + input_spike.
Fires when state exceeds threshold. Natural IIR in spike domain.

Parameters
----------
decay : float
    Decay factor per timestep (0-1).
threshold : float
gain : float
    Input spike gain.

- **filter**(spikes)
  - Apply IIR filter to spike train.

### Function `spike_convolve(spikes, kernel, threshold)`
Convolve a spike train with a kernel in spike domain.

Parameters
----------
spikes : ndarray of shape (T,)
kernel : ndarray of shape (K,)
threshold : float

Returns
-------
ndarray of shape (T,), binary

---

## Module `spike_dsp.spectral`

### Function `spike_fft(spikes, dt, window_size)`
Compute FFT of a spike train.

Parameters
----------
spikes : ndarray of shape (T,) or (T, N)
    Binary spike train(s).
dt : float
    Timestep in seconds.
window_size : int
    Sliding window for instantaneous rate estimation.

Returns
-------
(frequencies, magnitudes) tuple
    frequencies: ndarray of shape (F,)
    magnitudes: ndarray of shape (F,) or (F, N)

### Function `spike_power_spectrum(spikes, dt, window_size)`
Compute power spectral density of a spike train.

Returns
-------
(frequencies, psd) tuple

---

## Module `spike_dsp.wavelets`

### Function `spike_wavelet_decompose(spikes, n_scales, base_window)`
Decompose spike train into frequency bands via multi-scale filtering.

Uses cascaded moving-average filters at doubling window sizes:
scale 0: window=base_window, scale 1: window=2*base_window, etc.
Each scale captures a different frequency band.

Parameters
----------
spikes : ndarray of shape (T,) or (T, N)
n_scales : int
    Number of wavelet scales.
base_window : int
    Window size for finest scale.

Returns
-------
list of ndarray
    One array per scale, shape (T,) or (T, N). Binary spike representation
    of activity at each frequency band.

---

## Module `spike_gnn.spike_gnn`

### Class `SpikeGraphConv`
Spike-based graph convolution layer.

Message passing: each node aggregates spike trains from neighbors,
applies a learned weight transform via LIF integration.

Parameters
----------
in_features : int
    Input feature dimension per node.
out_features : int
    Output feature dimension per node.
threshold : float
tau_mem : float

- **__init__**(in_features, out_features, threshold, tau_mem, seed)
- **forward**(node_features, adjacency, T)
  - Spike-based graph convolution.

### Class `SpikeGNNLayer`
Multi-layer spike GNN for graph classification/regression.

Parameters
----------
layer_dims : list of int
    &#91;in_features, hidden1, ..., out_features&#93;
threshold : float
T : int
    Simulation timesteps per layer.

- **__post_init__**()
- **forward**(node_features, adjacency)
  - Forward pass through all layers.
- **graph_classify**(node_features, adjacency)
  - Classify a graph by global readout (sum pooling + argmax).
- **n_layers**()

---

## Module `spike_norm.normalizers`

### Class `ThresholdDependentBN`
tdBN: incorporates firing threshold into normalization.

BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
where mean/var are computed across batch, adjusted by V_threshold.

Parameters
----------
n_features : int
threshold : float
momentum : float

- **__post_init__**()
- **forward**(x, training)

### Class `PerTimestepBN`
BNTT: separate BN statistics per timestep.

Each timestep t has its own mean_t, var_t, gamma_t, beta_t.

Parameters
----------
n_features : int
T : int
    Number of timesteps.

- **__post_init__**()
- **forward**(x, t, training)

### Class `TemporalEffectiveBN`
TEBN: rescales presynaptic inputs per timestep.

Applies BN then per-timestep scaling factor lambda_t.

Parameters
----------
n_features : int
T : int

- **__post_init__**()
- **forward**(x, t, training)

### Class `MembranePotentialBN`
MPBN: BN on membrane potential before spike function.

At inference: fold BN into threshold (zero overhead).
new_threshold = (V_th - beta) * sqrt(var + eps) / gamma + mean

Parameters
----------
n_features : int
threshold : float

- **__post_init__**()
- **forward**(membrane, training)
- **fused_threshold**()
  - Compute per-neuron threshold that absorbs BN at inference.

### Class `TemporalAccumulatedBN`
TAB: normalizes accumulated membrane potential.

Tracks running accumulated potential across timesteps.
Addresses Temporal Covariate Shift directly.

Parameters
----------
n_features : int

- **__post_init__**()
- **forward**(x, training)
- **reset**()

---

## Module `spike_ode.ode_layer`

### Class `ODELIFDynamics`
LIF membrane ODE dynamics.

dv/dt = -(v - v_rest) / tau_mem + I(t) / C_mem

Parameters
----------
tau_mem : float
    Membrane time constant (ms).
v_rest : float
v_threshold : float
v_reset : float
C_mem : float
    Membrane capacitance (normalized).

- **dvdt**(v, I)
  - Compute membrane voltage derivative.

### Class `SpikingODELayer`
Spiking Neural ODE layer with event-driven integration.

Integrates the membrane ODE with adaptive Euler stepping.
Detects threshold crossings via bisection, emits spikes, resets.

Parameters
----------
n_inputs : int
n_neurons : int
dynamics : ODELIFDynamics
dt_init : float
    Initial integration step size.
dt_min : float
    Minimum step size.
max_steps_per_interval : int
    Max ODE steps per simulation interval.
seed : int

- **__init__**(n_inputs, n_neurons, dynamics, dt_init, dt_min, max_steps_per_interval, seed)
- **step**(x, interval)
  - Integrate ODE over one interval, return spike counts.
- **forward**(inputs, interval)
  - Process a sequence of inputs.
- **reset**()
  - Reset membrane voltages to the configured resting potential.
- **voltage**()
  - Return a defensive copy of the current membrane voltages.

---

## Module `spintronic.spintronic_mapper`

### Class `SpintronicTech`

### Class `MaterialParams`
Micromagnetic material parameters.

- **cofeb_mgo**(cls)
  - CoFeB/MgO — standard STT-MTJ / SOT-MRAM stack.
- **pt_co_multilayer**(cls)
  - Pt/Co multilayer — skyrmion host.
- **w_cofeb**(cls)
  - W/CoFeB — SOT switching with heavy-metal underlayer.

### Class `SpintronicDeviceConfig`
Configuration for a single spintronic device.

- **__post_init__**()
- **from_tech**(cls, tech)
- **area_nm2**()
- **switching_energy_fj**()
  - Write-path switching energy, E = I² × R_write × t.
- **thermal_stability**()
  - Thermal stability factor Δ = Ku × V / (kB × T).
- **read_disturb_probability**()
  - Probability of read disturb (accidental flip during read).
- **endurance_cycles**()
  - Estimated write endurance cycles.

### Class `VariabilityModel`
Process variability for spintronic devices.

Models dimension, anisotropy, and DMI variations from fab data.

- **apply**(device, rng)
  - Return a variability-injected copy of the device.

### Class `SpintronicCell`
One cell in a spintronic array.

- **resistance_ohm**()

### Class `SpintronicArray`
Crossbar array of spintronic devices for SC computation.

- **__init__**(rows, cols, tech, variability, rng_seed)
- **total_cells**()
- **total_area_um2**()
- **program_weights**(weights_q88)
  - Program Q8.8 weights into the array.
- **read_weights**()
  - Read weights back from the array.
- **power_breakdown**(bitstream_length)
  - Per-array power breakdown in femtojoules.

### Class `MappingResult`
Result of mapping SC bitstreams to a spintronic array.


### Class `SpintronicMapper`
Maps SC bitstreams + weights to a spintronic crossbar.

Performs:
1. Weight quantisation to device states (P/AP)
2. Array sizing from network dimensions
3. Energy/timing estimation
4. Monte-Carlo variability yield analysis

- **__init__**(tech, variability, rng_seed)
- **map_network**(weights_q88, bitstream_length)
  - Map a weight matrix to a spintronic array.
- **monte_carlo_yield**(weights_q88, n_trials, tolerance_q88)
  - Run Monte-Carlo yield analysis.

### Class `MuMax3ScriptGenerator`
Generates MuMax3 (.mx3) scripts for micromagnetic co-simulation.

- **generate_switching**(device, current_density_a_m2, duration_ns)
- **generate_skyrmion**(device)

### Class `SpintronicVerilogGenerator`
Generates Verilog wrappers for spintronic crossbar arrays.

- **generate**(array_name, rows, cols, tech)

### Class `RacetrackShiftRegister`
Domain-wall racetrack memory with current-driven bit shifting.

Models a nanotrack with N bit positions; SOT current pulses shift
all domain walls one position per clock edge.

- **__post_init__**()
- **load**(data)
- **shift_right**(n, rng)
- **shift_left**(n, rng)
- **shift_energy_fj**()

### Class `SkyrmionHallCorrector`
Corrects skyrmion trajectory for the skyrmion Hall effect.

Skyrmions drift at an angle θ_H to the applied current direction.
θ_H = arctan(4π Q / (α D)) where Q is topological charge.

- **hall_angle_deg**()
- **corrected_position**(x_drive, track_width_nm)
  - Return (x, y) position accounting for Hall drift.
- **needs_confinement**()

### Class `MLCConfig`
Multi-level cell configuration for spintronic devices.

- **__post_init__**()
- **resistance_margins**()
  - Resistance levels evenly spaced between R_P and R_AP.
- **quantize_weight**(weight_float)
  - Quantize a &#91;0, 1&#93; weight to an MLC level.
- **dequantize**(level)
- **density_improvement**()

### Class `WriteVerifyResult`
Result of a write-verify cycle.

- **error**()

### Class `AgingModel`
Endurance degradation model for spintronic devices.

TMR and thermal stability degrade with write cycles.

- **tmr_degradation**(initial_tmr, endurance_limit)
  - TMR degrades linearly as cycling approaches endurance limit.
- **stability_degradation**(initial_delta, endurance_limit)
  - Thermal stability degrades with oxide breakdown.
- **is_worn_out**()
- **write**(n)

### Class `RadiationModel`
SEU and TID models for spintronic devices in radiation environments.

- **seu_rate**(flux_particles_cm2_s, n_devices)
  - SEU events per second.
- **tid_degradation**(dose_krad)
  - Fraction of performance remaining after TID.
- **is_rad_hard**()
  - Spintronic devices are inherently radiation-hard (non-charge-based).

### Class `DefectEntry`
One defective cell in the array.


### Class `DefectMap`
Tracks and remaps defective cells in a spintronic array.

- **__init__**()
- **add_defect**(row, col, defect_type)
- **defect_count**()
- **defect_rate**(total_cells)
- **add_remap**(bad, spare)
- **is_defective**(row, col)
- **effective_address**(row, col)

### Class `MuMax3Result`
Parsed result from a MuMax3 simulation.

- **magnetisation_magnitude**()

### Class `MuMax3OutputParser`
Parses MuMax3 table output for co-simulation integration.

- **parse_table**(text)
  - Parse a MuMax3 .table output (TSV with header).
- **is_switching_successful**(result)

### Function `switching_current_vs_temperature(i_c0_ua, delta_0, temperature_k, temp_ref_k)`
Ic(T) = Ic0 × (1 - T/T_ref × kBT/(Δ0 × kBT_ref)).

Simplified Néel-Arrhenius model for thermally activated switching.

### Function `switching_time_vs_temperature(t_sw0_ns, temperature_k, temp_ref_k)`
Switching time increases with temperature due to thermal fluctuations.

### Function `retention_failure_probability(thermal_stability, time_seconds, attempt_freq_hz)`
P_fail = 1 - exp(-t × f0 × exp(-Δ)).

### Function `write_verify(cell, target_q88, max_attempts, rng)`
Program a cell with write-verify loop.

---

## Module `stochastic_doctor.diagnostics`

### Class `AuditSeverity`
Audit finding severity levels.


### Class `BitstreamAuditFinding`
Single finding from a bitstream audit.


### Class `BitstreamAuditReport`
Full bitstream-level audit report for a network layer.

JSON-serializable via ``to_json()`` for pipeline integration.

- **to_dict**()
  - Serialize to a plain dict (JSON-compatible).
- **to_json**(indent)
  - Serialize to JSON string.

### Class `DriftDetector`
Exponential moving average drift detector for SCC monitoring.

Tracks the running EMA of SCC values. Flags a drift event when
|EMA| exceeds the threshold.

Parameters
----------
alpha : float
    EMA smoothing factor (0.0–1.0; lower = smoother).
threshold : float
    Absolute SCC value above which to flag drift.

- **__init__**(alpha, threshold)
- **observe**(scc_value)
  - Feed a new SCC observation. Returns True if drift detected.
- **reset**()
  - Reset detector state.
- **history**()
  - EMA history for plotting/logging.

### Class `StochasticDoctor`
Bitstream-level stochastic diagnostics engine.

Parameters
----------
correlation_threshold : float
    |SCC| above this triggers a WARNING (default 0.3).
critical_threshold : float
    |SCC| above this triggers a CRITICAL (default 0.7).

- **__init__**(correlation_threshold, critical_threshold)
- **compute_correlation**(a, b)
  - Compute SCC between two bitstreams.
- **estimate_precision**(bitstream)
  - Estimate probability and variance bound for a bitstream.
- **compute_histogram**(bitstream, word_size)
  - Compute per-word popcount histogram.
- **audit_layer**(layer_id, bitstreams)
  - Audit a full layer of bitstreams.

### Function `_scc_python(a, b)`
Pure Python SCC (fallback when Rust core unavailable).

### Function `compute_scc(a, b)`
Compute SCC between two bitstreams.

Uses Rust PyO3 acceleration when available, falls back to pure Python.
Set ``SC_NEUROCORE_NO_RUST=1`` to force Python path.

---

## Module `studio.analysis`

### Function `bifurcation_sweep(simulate_fn, base_config, param_name, param_min, param_max, n_values)`
Sweep one parameter and extract voltage attractors at each value.

Returns {param_values, attractors} where attractors&#91;i&#93; is a list
of voltage extrema in the second half of the simulation (the attractor).

### Function `sensitivity_analysis(simulate_fn, base_config, param_names, perturbation)`
Compute firing rate sensitivity to each parameter (±perturbation fraction).

### Function `nullclines_2d(equations, params, var_names, ranges, grid_size)`
Compute nullclines for a 2-variable ODE system on a grid.

### Function `heatmap_2d(simulate_fn, base_config, param_x, x_min, x_max, x_steps, param_y, y_min, y_max, y_steps)`
Sweep two parameters and compute firing rate heatmap.

### Function `spike_triggered_average(time, voltage, spikes, dt, window_ms)`
Compute spike-triggered average of voltage around each spike.

### Function `frequency_response(simulate_fn, base_config, freq_min, freq_max, n_freqs, amplitude)`
Sweep sinusoidal current frequency and measure spike rate response.

### Function `precision_compare(equations, threshold, reset, params, init, dt, duration, current)`
Compare float64 vs Q8.8 fixed-point simulation of the same ODE.

---

## Module `studio.app`

### Class `SimulateRequest`

### Class `ModelSimulateRequest`

### Class `FICurveRequest`

### Class `CompileRequest`

### Class `BifurcationRequest`

### Class `SensitivityRequest`

### Class `NullclineRequest`

### Class `PrecisionRequest`

### Class `CompareRequest`

### Class `FreqResponseRequest`

### Class `HeatmapRequest`

### Class `NetworkRequest`

### Class `CodegenRequest`

### Class `_SimCache`
LRU cache for simulation results keyed by JSON hash.

- **__init__**(maxsize)
- **_key**(data)
- **get**(params)
- **put**(params, result)

### Function `_safe(fn)`
### Function `_make_simulate_fn(req_dict)`
Build a simulate callable from request params (ODE or model).

### Function `create_app()`
---

## Module `studio.characterize`

### Function `characterize_model(simulate_fn, base_config)`
Run a full characterisation suite on a neuron model.

Returns a dict with:
- trace: simulation result at default current
- pattern: firing pattern classification
- fi_curve: firing rate vs current (20 points)
- threshold_current: estimated rheobase (lowest current that produces spikes)
- max_rate: maximum firing rate in the f-I sweep
- isi_stats: ISI statistics at default current
- sensitivity: top 5 most sensitive parameters
- state_var_ranges: min/max of each state variable

---

## Module `studio.codegen`

### Function `generate_model_script(model_name, params, duration, current, dt)`
Generate a standalone Python script that reproduces the current simulation.

### Function `generate_ode_script(equations, threshold, reset, params, init, duration, current, dt)`
Generate a standalone Python script for a custom ODE simulation.

### Function `generate_oneliner(model_name, params, current)`
Generate a copy-paste Python one-liner for notebook use.

### Function `classify_firing_pattern(spikes, n_steps, dt)`
Classify the firing pattern from spike indices.

---

## Module `studio.compiler`

### Function `build_ir_from_equation(equations, params, threshold, reset, dt)`
Build an SC IR graph from ODE equations.

Uses the Rust ScGraphBuilder to construct a stochastic computing IR
that represents the neuron's ODE as a hardware pipeline:
input current → encode → multiply (leak, gain) → LIF step → output spike.

### Function `verify_ir(ir_text)`
Parse and verify an IR text representation.

### Function `emit_systemverilog(ir_text)`
Parse IR text and emit synthesisable SystemVerilog.

### Function `emit_sv_from_equation(equations, params, threshold, reset)`
Direct equation → SystemVerilog via the Python equation compiler.

### Function `cosim_traces(equations, threshold, reset, params, init, dt, duration, current)`
Run Python float and Q8.8 fixed-point simulations side by side.

---

## Module `studio.model_scan`

### Function `scan_all_models(current, duration)`
Simulate every model at a given current and classify its firing pattern.

Results are cached after first call.

---

## Module `studio.models`

### Class `RustStudioBackendUnavailable`
Raised when the Studio Rust batch-simulation path is unavailable.


### Class `RustStudioBackendError`
Raised when the Studio Rust batch-simulation path fails at runtime.


### Class `ModelMetadataError`
Raised when Studio model metadata loading fails for a known model.


### Function `_load_class(name)`
### Function `_classify_fields(cls)`
Split dataclass fields into state variables and parameters.

### Function `_categorize(name)`
### Function `list_models()`
Return metadata for all 118 neuron models with categories.

Results are cached after first call — subsequent calls return instantly.

### Function `get_model_detail(name)`
Return full metadata for a single model.

### Function `_load_rust_batch_simulate()`
Load the Rust batch-simulation bridge entrypoint.

Import failure means the backend is unavailable; it must not be conflated
with runtime failure inside an otherwise available backend.

### Function `_is_rust_unsupported_model_error(exc)`
True when the Rust backend rejected a model as unsupported.

### Function `_detect_step_kwarg(cls)`
Figure out what keyword the .step() method uses for current injection.

### Function `_try_rust_simulate(name, n_steps, current_trace, actual_dt)`
Attempt Rust batch simulation.

Returns ``None`` only when the backend is unavailable or the model is not
implemented in Rust. Runtime failures in an available backend are raised so
the caller does not silently degrade to Python.

### Function `simulate_model(name, param_overrides, dt, duration, current, protocol, frequency_hz)`
Simulate a named model. Uses Rust engine when model has default params.

---

## Module `studio.network`

### Function `simulate_ei_network(n_exc, n_inh, w_ee, w_ei, w_ie, w_ii, p_conn, ext_rate, duration, dt)`
Simulate a balanced E-I network. Uses Rust engine when available.

### Function `_simulate_rust(n_exc, n_inh, w_ee, w_ei, w_ie, w_ii, p_conn, ext_rate, duration, dt)`
Entire simulation in Rust — connectivity, Poisson input, stepping, recording.

### Function `_simulate_numpy(n_exc, n_inh, w_ee, w_ei, w_ie, w_ii, p_conn, ext_rate, duration, dt)`
Pure NumPy fallback (no Rust engine).

---

## Module `studio.network_graph`

### Class `ModelDiscoveryError`
Raised when Studio model discovery cannot produce a trustworthy list.


### Function `available_models()`
Return names of all neuron models available for populations.

### Function `create_population(label, model, count, neuron_type, x, y)`
Create a population node for the network canvas.

### Function `create_projection(source_id, target_id, weight, delay, probability)`
Create a projection edge between two populations.

### Function `validate_graph(graph)`
Validate a network graph. Returns list of error messages (empty = valid).

### Function `simulate_graph(graph)`
Simulate a network graph using the E-I network backend.

Maps populations and projections to the existing E-I simulation.
Only graphs with exactly 2 populations (1 exc + 1 inh) are
currently supported; other topologies fail closed instead of
being collapsed into an unfaithful surrogate.

### Function `graph_to_nir(graph)`
Export network graph to NIR-compatible format.

### Function `nir_to_graph(nir_data)`
Import NIR-compatible format to network graph.

---

## Module `studio.presets`

### Function `list_presets()`
### Function `get_preset(preset_id)`
---

## Module `studio.progress`

### Function `_characterize_with_progress(simulate_fn, base_config, q)`
Run characterisation with progress updates pushed to queue.

### Function `_heatmap_with_progress(simulate_fn, base_config, param_x, x_vals, param_y, y_vals, q)`
Run 2D heatmap sweep with progress updates.

### Function `_scan_with_progress(q)`
Scan all models with progress updates.

---

## Module `studio.project`

### Function `_validate_hdl_identifiers(payload)`
Reject workspace content that would later interpolate into HDL/MLIR source.

### Function `_ensure_dir()`
### Function `_projects_root()`
Return the resolved Studio project root.

### Function `_safe_name(name)`
Validate a project name that maps to one JSON file in the project root.

### Function `_safe_path(name)`
Build a resolved project file path confined to the Studio project root.

### Function `save_project(name, state)`
Save full studio state to a JSON file.

### Function `load_project(name)`
Load a saved project by name.

### Function `list_projects()`
List all saved projects.

### Function `delete_project(name)`
Delete a saved project.

### Function `run_pipeline(graph, target)`
Full pipeline: graph → compile equations → emit SV → synthesise.

If the graph has ODE equations in population params, compiles them
to SystemVerilog and runs synthesis. Otherwise uses a default LIF
compile path.

---

## Module `studio.simulation`

### Function `_spike_stats(spike_indices, dt, n_steps)`
Compute spike statistics from spike index list.

### Function `_make_current_trace(protocol, current, n_steps, dt, frequency_hz, step_onset, step_offset, ramp_start, ramp_end)`
Generate a current injection trace for the given protocol.

### Function `simulate(equations, threshold, reset, params, init, dt, duration, current, protocol, frequency_hz)`
Run an ODE neuron simulation and return time series data.

### Function `fi_curve(equations, threshold, reset, params, init, dt, duration, i_min, i_max, i_steps)`
Sweep current and compute firing rate at each level.

---

## Module `studio.svg_export`

### Function `traces_to_svg(time, states, spikes, model_name, dt, width, height)`
Generate publication-quality SVG from simulation traces.

Returns a complete SVG string with axes, grid, legend, spike markers,
and colour-coded polylines. Traces are downsampled to <=2000 points.

### Function `_empty_svg(width, height)`
---

## Module `studio.synthesis`

### Function `check_tools()`
Detect which EDA tools are installed.

### Function `run_synthesis(verilog_source, target)`
Run Yosys synthesis and return resource usage.

### Function `_parse_yosys_json(json_path)`
Extract resource counts from Yosys JSON output.

### Function `estimate_resources(ir_op_count, target)`
Quick resource estimate from IR operation count, no Yosys needed.

Heuristic: each IR op maps to ~2 LUTs + 1 FF on average.
LIF step op maps to ~12 LUTs + 8 FFs + 1 DSP (multiplier).

### Function `multi_target_synthesis(verilog_source)`
Run synthesis on all supported targets, return comparison.

### Function `run_pnr(json_path, target)`
Run nextpnr place-and-route and return timing report.

---

## Module `studio.templates`

### Function `list_templates()`
### Function `get_template(name)`
---

## Module `studio.training`

### Class `TrainingJob`
Manages a single training run in a background thread.

- **__init__**(config)
- **start**()
- **stop**()
- **_emit**(event_type, data)
- **_run**()
- **_train**()

### Function `list_surrogates()`
### Function `list_cell_types()`
### Function `_make_synthetic(batch_size)`
Generate synthetic classification data for quick demos.

### Function `_load_mnist(batch_size)`
Load MNIST via torchvision if available, else synthetic fallback.

### Function `start_training(config)`
### Function `stop_training(job_id)`
### Function `get_training_status(job_id)`
### Function `stream_metrics(job_id)`
Generator that yields SSE-formatted metric events.

### Function `list_jobs()`
---

## Module `swarm.agent`

### Class `AgentConfig`
Hyper-parameters for a single swarm agent.


### Class `SwarmAgent`
Spiking-neural-network agent with soft-LIF dynamics.

Parameters
----------
cfg : AgentConfig
    Neuron and network parameters.
agent_id : int
    Unique identifier within the swarm.

- **__init__**(cfg, agent_id)
- **n_weights**()
- **weights**()
  - Return all trainable weights as a flat 1-D vector.
- **weights**(flat)
- **think**(sensory)
  - Run one SNN tick and return ``(speed, turn_angle)``.
- **act**(speed, turn)
  - Update position and heading given motor commands.
- **reset**(rng, width, height)
  - Reset kinematic and neural state (weights untouched).

---

## Module `swarm.collective_fields`

### Class `FieldConfig`
Field layer hyper-parameters.


### Class `CollectiveFields`
Chemical, emotional, and symbolic field layers for swarm communication.

Parameters
----------
cfg : FieldConfig
    Field configuration.
env_width : float
    Physical width of the environment (for coordinate mapping).
env_height : float
    Physical height of the environment.
n_agents : int
    Number of agents (for emotional field sizing).

- **__init__**(cfg, env_width, env_height, n_agents)
- **_to_grid**(x, y)
- **diffuse**(dt)
  - Apply Laplacian diffusion + exponential decay to the chemical field.
- **deposit_chemical**(x, y, amount)
  - Add *amount* of chemical at world coordinate ``(x, y)``.
- **get_chemical_gradient**(x, y)
  - Return normalised (dx, dy) chemical gradient at ``(x, y)``.
- **synchronize_emotions**(coupling)
  - Pull each agent's emotional vector toward the swarm mean.
- **get_symbolic_at**(x, y)
  - Return the 2-channel symbolic vector at ``(x, y)``.
- **deposit_symbolic**(x, y, channel, amount)
  - Deposit into a symbolic channel at ``(x, y)``.
- **update**(agents, env, dt)
  - Run one collective-field tick.

### Function `_apply_laplacian(field)`
Apply 3x3 Laplacian via manual convolution (zero-padded edges).

Parameters
----------
field : ndarray, shape (H, W)

Returns
-------
lap : ndarray, shape (H, W)

---

## Module `swarm.fitness`

### Class `SwarmFitness`
Static fitness functions for swarm evaluation.

- **coverage_score**(positions, area)
  - Fraction of the arena covered by the swarm.
- **cohesion_score**(positions)
  - Reward moderate inter-agent distance (not too spread, not too clumped).
- **alignment_score**(headings)
  - Mean resultant length of heading angles (Rayleigh statistic).
- **target_score**(positions, targets)
  - Proximity reward: inverse mean distance to nearest target per agent.
- **obstacle_penalty**(positions, obstacles)
  - Fraction of agents inside any obstacle (surface penetration).
- **composite**(env)
  - Weighted sum of all objectives.

---

## Module `swarm.neuroevolution_swarm`

### Class `EvolverConfig`
Neuroevolution hyper-parameters.


### Class `SwarmEvolver`
Genetic algorithm that evolves SNN weights for swarm control.

Parameters
----------
cfg : EvolverConfig
    Evolution and evaluation parameters.

- **__init__**(cfg)
- **_make_env**()
  - Build a fresh environment with the correct agent config.
- **evaluate_individual**(weights)
  - Create environment, inject *weights* into every agent, run, score.
- **_select_elite**()
  - Return the top-N weight vectors by fitness.
- **_crossover**(parent_a, parent_b)
  - Uniform crossover: each gene randomly from either parent.
- **_mutate**(individual)
  - Gaussian mutation applied to a random subset of genes.
- **evolve_generation**()
  - Evaluate population, select, reproduce.  Return best fitness.
- **get_best_weights**()
  - Return the weight vector with the highest fitness.
- **run**(n_generations)
  - Run *n_generations* of evolution.  Return list of best fitnesses.

---

## Module `swarm.swarm_env`

### Class `EnvConfig`
Environment hyper-parameters.


### Class `SwarmEnvironment`
2-D continuous arena for swarm simulation.

Parameters
----------
cfg : EnvConfig
    Environment configuration.

- **__init__**(cfg)
- **_random_target_pos**()
- **_apply_boundary**(agent)
- **get_positions**()
  - Return (n_agents, 2) position array.
- **get_headings**()
  - Return (n_agents,) heading array.
- **get_pairwise_distances**()
  - Return (n_agents, n_agents) Euclidean distance matrix.
- **get_neighbor_distances**(agent_idx, k)
  - Return sorted distances to the *k* nearest neighbours.
- **get_obstacle_distances**(agent_idx, k)
  - Distances to the *k* nearest obstacle surfaces (negative = inside).
- **get_target_distances**(agent_idx, k)
  - Distances to the *k* nearest targets.
- **step**(dt, fields)
  - Advance the simulation by one tick.
- **get_state**()
  - Return a JSON-serialisable snapshot.

---

## Module `symbolic.spike_logic`

### Class `SpikeGate`
Spike-based logic gate.

Parameters
----------
gate_type : str
    'AND', 'OR', 'NOT', 'NAND', 'XOR'

- **__call__**()
- **lif_config**()
  - LIF neuron configuration for this gate.

### Class `SpikeRegister`
Spike-based register: stores N bits using SR latch pairs.

Each bit is held by two neurons in mutual inhibition (bistable).
Write: inject spike to set/reset neuron.
Read: check which neuron of each pair is active.

Parameters
----------
n_bits : int
    Register width.

- **__init__**(n_bits)
- **write**(value)
  - Write an integer value to the register.
- **read**()
  - Read the register as an integer.
- **write_bits**(bits)
  - Write raw bit array.
- **read_bits**()
  - Read raw bit array.
- **clear**()

### Class `SpikeALU`
Spike-based Arithmetic Logic Unit.

Operations: ADD, SUB, AND, OR, XOR, CMP, SHIFT_LEFT, SHIFT_RIGHT.
All implemented via spike-gate compositions.

Parameters
----------
n_bits : int
    Word width.

- **__init__**(n_bits)
- **add**(a, b)
  - Ripple-carry addition. Returns (result, carry_out).
- **sub**(a, b)
  - Subtraction via two's complement: a - b = a + (~b + 1).
- **bitwise_and**(a, b)
- **bitwise_or**(a, b)
- **bitwise_xor**(a, b)
- **compare**(a, b)
  - Compare: returns -1, 0, or 1.
- **shift_left**(a, n)
- **shift_right**(a, n)

### Function `spike_sort(values, n_bits)`
Sort integers using spike-based comparison network.

Uses a bubble-sort topology where each compare-and-swap
is implemented via SpikeALU.compare.

Parameters
----------
values : list of int
n_bits : int

Returns
-------
list of int, sorted ascending

---

## Module `synapses.bcm`

### Class `BCMSynapse`
BCM synapse with sliding modification threshold.

Parameters
----------
eta : float
    Learning rate.
tau_theta : float
    Time constant for sliding threshold (ms).
theta_init : float
    Initial threshold value.
w_min, w_max : float
    Weight bounds.

- **__post_init__**()
- **step**(pre_rate, post_rate, dt)
  - Advance one timestep.
- **reset**()

---

## Module `synapses.clopath_stdp`

### Class `ClopathSTDP`
Voltage-based STDP (Clopath et al. 2010).

Parameters
----------
a_ltd : float
    LTD amplitude. Default: 14e-5 (Clopath 2010, Table 1).
a_ltp : float
    LTP amplitude. Default: 8e-5.
tau_x : float
    Pre-synaptic trace decay (ms). Default: 15.
tau_minus : float
    Slow voltage trace decay (ms). Default: 10.
tau_plus : float
    Fast voltage trace decay (ms). Default: 7.
theta_minus : float
    LTD voltage threshold (mV). Default: -70.6 (rest).
theta_plus : float
    LTP voltage threshold (mV). Default: -45.3 (depolarization).
w_min, w_max : float
    Weight bounds.

- **__post_init__**()
- **step**(pre_spike, u_post, dt)
  - Advance one timestep.
- **reset**()

---

## Module `synapses.dopamine_stdp`

### Class `DopamineStdpSynapse`
Dopamine-gated STDP synapse (Izhikevich 2007).

Parameters
----------
weight : float
    Synaptic weight. Default: 0.5.
w_min : float
    Minimum weight. Default: 0.0.
w_max : float
    Maximum weight. Default: 1.0.
tau_e : float
    Eligibility trace time constant (ms). Default: 1000.0.
tau_da : float
    Dopamine decay time constant (ms). Default: 200.0.
tau_pre : float
    Pre-synaptic trace time constant (ms). Default: 20.0.
tau_post : float
    Post-synaptic trace time constant (ms). Default: 20.0.
a_plus : float
    LTP amplitude. Default: 1.0.
a_minus : float
    LTD amplitude (negative). Default: -1.0.
lr : float
    Learning rate. Default: 0.001.
dt : float
    Integration timestep (ms). Default: 1.0.

- **step**(pre_spike, post_spike, reward)
  - Advance one timestep with spike indicators and reward signal.
- **reset**()
  - Reset state to initial conditions.

---

## Module `synapses.dot_product`

### Class `BitstreamDotProduct`
Bitstream-level dot product via SC synapses.

For each input i, applies synapse_i (AND gate), then sums decoded
probabilities: y ~ sum_i w_i * x_i.

Example
-------
>>> import numpy as np
>>> from sc_neurocore import BitstreamSynapse
>>> syns = &#91;BitstreamSynapse(w_min=0.0, w_max=1.0, w=0.5, length=256)
...         for _ in range(3)&#93;
>>> dp = BitstreamDotProduct(synapses=syns)
>>> pre = np.ones((3, 256), dtype=np.uint8)
>>> post_matrix, y_scalar = dp.apply(pre)
>>> post_matrix.shape
(3, 256)

- **__post_init__**()
- **n_inputs**()
- **apply**(pre_matrix, y_min, y_max)
  - Apply all synapses to the pre-synaptic bitstreams and compute

---

## Module `synapses.gap_junction`

### Class `GapJunction`
Bidirectional electrical synapse.

Parameters
----------
conductance : float
    Gap junction conductance g_c (nS). Typical: 0.01-1.0 nS.
    Bennett & Zukin, Neuron 2004.
rectification : float
    Rectification factor in &#91;0, 1&#93;. 0 = fully bidirectional (ohmic),
    1 = fully rectifying (current flows in one direction only).
    Default 0 (standard gap junction).

- **current**(v_pre, v_post)
  - Compute gap junction current flowing INTO v_post.
- **current_matrix**(voltages, adjacency)
  - Compute gap junction currents for a population.

---

## Module `synapses.r_stdp`

### Class `RewardModulatedSTDPSynapse`
Reward-modulated STDP synapse (Izhikevich, Cerebral Cortex 17(10), 2007).

Eligibility trace accumulates Hebbian coincidences; weight update
fires only when a global reward signal arrives.

Example
-------
>>> syn = RewardModulatedSTDPSynapse(w_min=0.0, w_max=1.0, w=0.5, length=64)
>>> for _ in range(20):
...     syn.process_step(pre_bit=1, post_bit=1)
>>> syn.apply_reward(reward=1.0)  # positive reward → potentiate
>>> syn.w >= 0.5
True

- **process_step**(pre_bit, post_bit)
- **apply_reward**(reward)
  - Global reward signal triggers weight update.

---

## Module `synapses.sc_synapse`

### Class `BitstreamSynapse`
Stochastic-computing synapse using bitstreams.

Each synapse has a weight w in &#91;w_min, w_max&#93;.
SC multiplication via bitwise AND: P(out=1) ~ P(pre=1) * P(w=1).

Example
-------
>>> import numpy as np
>>> syn = BitstreamSynapse(w_min=0.0, w_max=1.0, w=0.5, length=1024, seed=42)
>>> pre = np.ones(1024, dtype=np.uint8)  # all-ones input
>>> post = syn.apply(pre)
>>> abs(post.mean() - 0.5) < 0.1  # output ~50% ones
True

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

## Module `synapses.short_term_plasticity`

### Class `ShortTermPlasticitySynapse`
Short-term plasticity synapse (Tsodyks-Markram 1997).

Parameters
----------
x : float
    Available resources (depression). Default: 1.0.
u : float
    Release probability (facilitation). Default: 0.5.
u_base : float
    Baseline release probability U. Default: 0.5.
tau_d : float
    Depression recovery time constant (ms). Default: 200.0.
tau_f : float
    Facilitation decay time constant (ms). Default: 20.0.
amplitude : float
    Maximum PSC amplitude. Default: 1.0.
dt : float
    Integration timestep (ms). Default: 1.0.

- **new_depressing**(cls)
  - Create a depressing synapse (cortical pyr-pyr).
- **new_facilitating**(cls)
  - Create a facilitating synapse (cortical pyr-interneuron).
- **step**(pre_spike)
  - Advance one timestep. Returns post-synaptic current.
- **reset**()
  - Reset state to initial conditions.

---

## Module `synapses.stochastic_stdp`

### Class `StochasticSTDPSynapse`
Stochastic synapse with spike-timing-dependent plasticity.

LTP on pre→post coincidence, LTD on pre-without-post.
Asymmetry ratio from Bi & Poo, J. Neurosci. 18(24), 1998.

Example
-------
>>> syn = StochasticSTDPSynapse(w_min=0.0, w_max=1.0, w=0.5, length=64)
>>> for _ in range(100):
...     syn.process_step(pre_bit=1, post_bit=1)  # correlated activity → LTP
>>> syn.w >= 0.5  # weight increased or stayed
True

- **__post_init__**()
- **process_step**(pre_bit, post_bit)
  - Process one timestep: compute output, update trace, apply STDP.
- **_potentiate**()
- **_depress**()

---

## Module `synapses.tripartite`

### Class `TripartiteSynapse`
Synapse with bidirectional astrocyte coupling.

Parameters
----------
base_weight : float
    Baseline synaptic weight.
glut_per_spike : float
    IP3 production rate per pre-synaptic spike (µM/s).
ca_threshold : float
    Astrocyte Ca²⁺ threshold for gliotransmitter release (µM).
facilitation : float
    Multiplicative gain when astrocyte is active (> 1 for facilitation).
depression_rate : float
    Weight depression rate when astrocyte Ca²⁺ is below threshold.
w_min, w_max : float
    Weight bounds.

- **__post_init__**()
- **step**(pre_spike, post_spike, dt)
  - Advance one timestep.
- **ca**()
  - Current astrocyte Ca²⁺ concentration (µM).
- **ip3**()
  - Current astrocyte IP3 concentration (µM).
- **effective_weight**()
  - Current effective synaptic weight.
- **reset**()

---

## Module `synapses.triplet_stdp`

### Class `TripletSTDP`
Triplet STDP synapse (Pfister-Gerstner 2006).

Parameters
----------
tau_plus : float
    Pre-synaptic trace decay (ms). Default: 16.8 (visual cortex fit).
tau_minus : float
    Post-synaptic trace decay (ms). Default: 33.7.
tau_x : float
    Slow pre-synaptic trace decay (ms). Default: 101.
tau_y : float
    Slow post-synaptic trace decay (ms). Default: 125.
a2_plus : float
    Pair LTP amplitude. Default: 7.5e-10.
a3_plus : float
    Triplet LTP amplitude. Default: 9.3e-3.
a2_minus : float
    Pair LTD amplitude. Default: 7.0e-3.
a3_minus : float
    Triplet LTD amplitude. Default: 2.3e-4.
w_min : float
    Minimum weight. Default: 0.0.
w_max : float
    Maximum weight. Default: 1.0.

- **__post_init__**()
- **step**(pre_spike, post_spike, dt)
  - Advance one timestep.
- **reset**()

---

## Module `temporal_hierarchy.multi_clock`

### Class `ClockDomain`
One clock domain in a multi-clock SNN.

Parameters
----------
name : str
tick_interval : int
    Steps between updates (1 = every step, 10 = every 10th step).
layers : list of str
    Layer names assigned to this clock domain.


### Class `HetSynLayer`
Layer with heterogeneous per-synapse time constants.

Each synapse has its own tau, initialized log-normally (mean=5ms, std=1ms).
The synaptic trace at each synapse decays at its own rate:
  trace&#91;i,j&#93; = exp(-dt/tau&#91;i,j&#93;) * trace&#91;i,j&#93; + input_spike&#91;j&#93;

Parameters
----------
n_inputs : int
n_neurons : int
tau_mean : float
    Mean synaptic time constant (ms).
tau_std : float
    Std of log(tau) for log-normal initialization.
threshold : float
seed : int

- **__init__**(n_inputs, n_neurons, tau_mean, tau_std, threshold, seed)
- **step**(x, dt)
  - Process one timestep.
- **reset**()
- **tau_stats**()

### Class `MultiClockSNN`
Multi-clock SNN with different temporal resolutions per layer.

Parameters
----------
layers : list of HetSynLayer
    Network layers.
clock_domains : list of ClockDomain
    Clock domain assignments.

- **__init__**(layers, layer_names, clock_intervals)
- **step**(x, dt)
  - Process one global timestep.
- **run**(inputs, dt)
  - Run full sequence.
- **reset**()

---

## Module `topology.analyzer`

### Class `TopologyReport`
Network topology analysis report.

- **summary**()

### Class `TopologyAnalyzer`
Analyse SNN connectivity structure.

Parameters
----------
adjacency : ndarray of shape (N, N)
    Binary adjacency matrix or weight matrix (nonzero = edge).
directed : bool
    If True, treat as directed graph.
n_path_samples : int
    Maximum number of source nodes used by ``_avg_path_length``.
    For ``N <= n_path_samples`` the result is the true all-pairs
    average; for larger graphs it is a sample mean over the first
    ``n_path_samples`` nodes (deterministic, not randomised).
    Default 100. Set to 0 or a very large value to force full
    all-pairs (expensive at N >> 100).

- **__init__**(adjacency, directed, n_path_samples)
- **analyze**()
  - Run full topology analysis.
- **_modularity**(communities)
  - Newman's modularity Q for the (undirected projection) graph.
- **_connected_components**(A)
  - Label each node with its connected-component index (BFS).
- **_clustering**()
  - Local clustering coefficient averaged over nodes.
- **_avg_path_length**()
  - Average shortest path length via BFS.
- **_bfs**(A, src)
- **_assortativity**(degrees)
  - Degree assortativity coefficient.

---

## Module `training.delay_linear`

### Class `DelayLinear`
Linear layer with trainable per-synapse delays.

Parameters
----------
in_features : int
    Number of input neurons.
out_features : int
    Number of output neurons.
max_delay : int
    Maximum delay in timesteps. Delay buffer stores this many steps.
bias : bool
    Include bias term (default False for SNN).
learn_delay : bool
    Make delays trainable (default True).
init_delay : float
    Initial delay value for all synapses (default 1.0).

Forward pass
-------------
Call ``step(input_spikes)`` at each timestep. The module maintains
an internal spike history buffer. For each synapse (i, j):

    delayed_input&#91;j&#93; = sum_i W&#91;j,i&#93; * interp(history, t - D&#91;j,i&#93;)

where interp linearly interpolates between integer delay bins,
making D differentiable.

Export
------
``delays_int`` returns quantized integer delays for hardware deployment.
``to_nir_delay_array()`` returns delays in the CSR format expected by
``Projection(delay=array)``.

- **__init__**(in_features, out_features, max_delay, bias, learn_delay, init_delay)
- **reset**()
  - Clear spike history. Call between sequences.
- **step**(x)
  - Process one timestep.
- **delays_int**()
  - Quantized integer delays for hardware export.
- **to_nir_delay_array**()
  - Export delays as flat array matching CSR data order.
- **extra_repr**()

---

## Module `training.encoding`

### Function `rate_encode(x, n_timesteps)`
Poisson rate coding. Higher values spike more often.

x: values in &#91;0, 1&#93;, shape (*batch). Returns (T, *batch).

### Function `latency_encode(x, n_timesteps, tau)`
Time-to-first-spike latency coding. Stronger input → earlier spike.

x: values in &#91;0, 1&#93;, shape (*batch). Returns (T, *batch).

### Function `delta_encode(x, threshold)`
Delta coding: spike on temporal change exceeding threshold.

x: shape (T, *batch). Returns same shape with spikes where |dx| > threshold.

---

## Module `training.equilibrium_propagation`

### Class `EPNetwork`
Multi-layer Equilibrium Propagation network.

Parameters
----------
layer_sizes : list of int
    Number of units per layer (including input and output).
rng_seed : int
    Random seed for reproducibility.

- **__init__**(layer_sizes, rng_seed)
- **_energy**(states)
  - Compute the Hopfield energy of the network.
- **_settle**(x)
  - Settle the network to (near) equilibrium.
- **train**(x_batch, y_batch)
  - Train on a mini-batch using the EP two-phase protocol.
- **predict**(x, n_settle)
  - Predict output by settling in free phase.
- **get_params**()
  - Return serialisable parameter dict.

### Function `_rho(x)`
Hard-sigmoid activation (hardware-friendly, no exp).

### Function `_rho_prime(x)`
Derivative of hard-sigmoid.

---

## Module `training.loops`

### Function `_device_usable(device)`
Return whether a Torch device can execute a minimal tensor operation.

### Function `auto_device()`
Select best available device: CUDA > MPS > CPU.

### Function `train_epoch(model, loader, optimizer, n_timesteps, loss_fn, device, max_grad_norm, flatten_input)`
One training epoch. Returns (avg_loss, accuracy).

Parameters
----------
flatten_input : bool
    If True (default), flatten data to (batch, features) for feedforward SNNs.
    Set to False for convolutional models that need spatial dimensions.

### Function `evaluate(model, loader, n_timesteps, loss_fn, device, flatten_input)`
Evaluate model. Returns (avg_loss, accuracy).

---

## Module `training.losses`

### Function `spike_count_loss(spike_counts, targets)`
Cross-entropy on spike counts. Bohte 2011.

### Function `membrane_loss(membrane_acc, targets)`
Cross-entropy on accumulated membrane potential.

### Function `spike_rate_loss(spike_counts, targets, n_timesteps, target_rate)`
MSE between output spike rates and one-hot target pattern.

### Function `spike_l1_loss(spike_counts, n_timesteps)`
L1 penalty on mean spike rate. Encourages sparse firing.

### Function `spike_l2_loss(spike_counts, n_timesteps)`
L2 penalty on mean spike rate. Penalizes high-firing neurons.

---

## Module `training.sc_correlation_regularizers`

### Function `_require_stream_bank(streams)`
### Function `correlation_matrix(streams)`
Return Pearson correlation matrix across bitstream rows.

### Function `pairwise_correlation_penalty(streams)`
Penalize off-diagonal stream correlations above ``threshold``.

### Function `correlation_penalty(observed)`
Differentiable mean-square penalty toward a target correlation.

---

## Module `training.sc_estimators`

### Class `DifferentiableSCConfig`
Validated contract for differentiable SC training operators.

- **__post_init__**()

### Class `RelaxedSCProduct`
Result bundle for a differentiable relaxed SC product.

- **length_cost**()

### Class `SCBitstreamSample`
Sampled SC bitstreams with decoded values.


### Class `SCBitstreamStatistics`
Rate, variance, and correlation evidence for sampled bitstreams.


### Class `SampledSCProduct`
Decoded sampled SC multiply result and stream statistics.


### Function `_require_domain(name, value, lower, upper)`
### Function `_bernoulli_product_expectation(p_input, p_weight, correlation)`
### Function `_role_seed(config, role)`
### Function `_probabilities_from_values(values, config)`
### Function `_decode_probabilities(probabilities, config)`
### Function `_sample_bernoulli_row(probability, length, seed)`
### Function `_sample_sobol_row(probability, length, seed)`
### Function `_radical_inverse_base2(index)`
### Function `_sample_halton_row(probability, length, seed)`
### Function `_lfsr16_step(state)`
### Function `_sample_lfsr_row(probability, length, seed)`
### Function `_sample_row(probability, length, seed, generator)`
### Function `_sample_paired_sobol_rows(input_probability, weight_probability, length, seed)`
### Function `sample_sc_bitstreams(values, config)`
Sample deterministic SC bitstreams for training-time statistics.

### Function `estimate_bitstream_statistics(streams)`
Return rate, variance, and Pearson correlation evidence for bitstreams.

### Function `relaxed_sc_multiply(input_value, weight_value, config)`
Return differentiable expected SC multiplication under the config contract.

### Function `sampled_sc_multiply(input_value, weight_value, config)`
Sample SC bitstreams, multiply them, and decode the empirical product.

### Function `finite_difference_gradients(input_value, weight_value, config)`
Central finite-difference gradients for deterministic relaxed SC operators.

---

## Module `training.snn_modules`

### Class `SCWeightNoiseModel`
Deterministic export-time noise model for SC weight probabilities.

- **__post_init__**()
- **metadata**()

### Class `LIFCell`
Single-step Leaky Integrate-and-Fire with surrogate backward.

v&#91;t&#93; = beta * v&#91;t-1&#93; + I&#91;t&#93;
spike&#91;t&#93; = H(v&#91;t&#93; - threshold)
v&#91;t&#93; -= spike&#91;t&#93; * threshold  (subtract reset)

- **__init__**(beta, threshold, surrogate_fn, learn_beta, learn_threshold)
- **beta**()
- **threshold**()
- **forward**(current, v)

### Class `IFCell`
Integrate-and-Fire (no leak, beta=1).

Simplest spiking model: v&#91;t&#93; = v&#91;t-1&#93; + I&#91;t&#93;, fire when v >= threshold.

- **__init__**(threshold, surrogate_fn, learn_threshold)
- **threshold**()
- **forward**(current, v)

### Class `SynapticCell`
Dual-exponential synaptic LIF. Two state variables: synapse current + membrane.

i_syn&#91;t&#93; = alpha * i_syn&#91;t-1&#93; + I&#91;t&#93;
v&#91;t&#93; = beta * v&#91;t-1&#93; + i_syn&#91;t&#93;

- **__init__**(alpha, beta, threshold, surrogate_fn, learn_beta, learn_threshold)
- **beta**()
- **threshold**()
- **forward**(current, i_syn, v)
  - Returns (spike, i_syn_next, v_next).

### Class `ALIFCell`
Adaptive LIF. Bellec et al. 2020.

Threshold adapts based on recent spiking: theta&#91;t&#93; = theta_0 + beta_adapt * a&#91;t&#93;
where a&#91;t&#93; = rho * a&#91;t-1&#93; + spike&#91;t-1&#93;.

- **__init__**(beta, threshold, rho, beta_adapt, surrogate_fn)
- **forward**(current, v, a)
  - Returns (spike, v_next, a_next).

### Class `ExpIFCell`
Exponential Integrate-and-Fire. Fourcaud-Trocmé et al. 2003.

v&#91;t&#93; = beta * v&#91;t-1&#93; + delta_T * exp((v&#91;t-1&#93; - v_rh) / delta_T) + I&#91;t&#93;
Exponential term creates sharp upstroke near threshold.

- **__init__**(beta, threshold, delta_t, v_rh, surrogate_fn, learn_beta, learn_threshold)
- **beta**()
- **threshold**()
- **forward**(current, v)

### Class `AdExCell`
Adaptive Exponential IF. Brette & Gerstner 2005.

v&#91;t&#93; = beta * v&#91;t-1&#93; + delta_T * exp((v - v_rh) / delta_T) - w&#91;t-1&#93; + I&#91;t&#93;
w&#91;t&#93; = rho * w&#91;t-1&#93; + a * (v&#91;t-1&#93; - v_rest) + b * spike&#91;t&#93;

- **__init__**(beta, threshold, delta_t, v_rh, a, b, rho, v_rest, surrogate_fn, learn_beta, learn_threshold)
- **beta**()
- **threshold**()
- **forward**(current, v, w)

### Class `LapicqueCell`
Lapicque IF with membrane resistance. Lapicque 1907.

tau * dv/dt = -(v - v_rest) + R * I
Discretised: v&#91;t&#93; = (1 - dt/tau) * v&#91;t-1&#93; + (R * dt / tau) * I&#91;t&#93;

- **__init__**(tau, r, dt, threshold, v_rest, surrogate_fn, learn_threshold)
- **threshold**()
- **forward**(current, v)

### Class `AlphaCell`
Alpha synapse neuron. Rall 1967.

Two-state alpha function: i_exc and i_inh with separate time constants.
v&#91;t&#93; = beta * v&#91;t-1&#93; + i_exc&#91;t&#93; - i_inh&#91;t&#93;

- **__init__**(alpha_exc, alpha_inh, beta, threshold, surrogate_fn, learn_beta, learn_threshold)
- **beta**()
- **threshold**()
- **forward**(exc_current, inh_current, i_exc, i_inh, v)

### Class `SecondOrderLIFCell`
Second-order LIF with inertial term. Dayan & Abbott 2001.

Adds a second state variable (acceleration) for smoother dynamics:
a&#91;t&#93; = alpha * a&#91;t-1&#93; + I&#91;t&#93;
v&#91;t&#93; = beta * v&#91;t-1&#93; + a&#91;t&#93;

- **__init__**(alpha, beta, threshold, surrogate_fn, learn_beta, learn_threshold)
- **beta**()
- **threshold**()
- **forward**(current, a, v)

### Class `RecurrentLIFCell`
LIF with trainable recurrent weights.

- **__init__**(n_neurons, beta, threshold, surrogate_fn, learn_beta, learn_threshold)
- **forward**(current, v, spike_prev)

### Class `SpikingNet`
Multi-layer feedforward SNN for classification.

Architecture: &#91;Linear -> LIF&#93; x (n_layers+1)
Readout: spike count and membrane accumulation over T timesteps.

- **__init__**(n_input, n_hidden, n_output, n_layers, beta, surrogate_fn, learn_beta, learn_threshold)
- **forward**(x)
  - x: (T, batch, n_input). Returns (spike_counts, membrane_acc).
- **to_sc_weights**(include_bias, noise_model, encoding)
  - Export weight matrices for SC bitstream deployment.

### Class `ConvSpikingNet`
Convolutional SNN for image classification.

Conv2d(1,32,5)→LIF→AvgPool→Conv2d(32,64,5)→LIF→AvgPool→Flatten→Linear→LIF→Linear→LIF

- **__init__**(n_output, beta, surrogate_fn, learn_beta, learn_threshold)
- **forward**(x)
  - x: (T, batch, 1, 28, 28). Returns (spike_counts, membrane_acc).
- **to_sc_weights**(include_bias, noise_model, encoding)
  - Export weight matrices for SC bitstream deployment.

### Function `_coerce_sc_weight_noise_model(noise_model)`
### Function `_normalise_sc_weight_tensor(weight, encoding)`
### Function `_sc_weight_scale(weight, encoding)`
### Function `_normalise_sc_bias_tensor(bias, scale, encoding)`
### Function `_apply_sc_weight_noise(weight, model, layer_index, encoding)`
### Function `_logit(p)`
Inverse sigmoid: logit(p) = log(p / (1 - p)).

---

## Module `training.stochastic_backprop`

### Class `SCTrainingObjectiveConfig`
Weights and targets for SC-aware training objective components.

- **__post_init__**()

### Class `SCResourceProxy`
Hardware proxy values used in SC-aware objective shaping.

- **__post_init__**()
- **normalized_cost**()
  - Return a dimensionless bounded-scale resource proxy.

### Class `SCObjectiveBreakdown`
Named scalar components of an SC-aware training objective.


### Class `SCBackpropDesignSpace`
Discrete SC architecture choices exposed to differentiable joint training.

- **__post_init__**()

### Class `SCBackpropJointReport`
Joint stochastic-backpropagation result with exportable SC design metadata.


### Function `_scalar_like(reference, value)`
### Function `_tensor_like(reference, values)`
### Function `_require_tensor_domain(name, value, lower, upper)`
### Function `relaxed_sc_linear(input_value, weight, bias, sc_config)`
Linear layer whose multiply-accumulate path uses relaxed SC products.

### Function `_bernoulli_product_expectation_with_correlation_tensor(p_input, p_weight, correlation)`
### Function `_relaxed_sc_linear_with_correlation_tensor(input_value, weight, bias, sc_config, correlation)`
### Function `stochastic_training_objective(task_loss)`
Compose task loss with SC length, correlation, variance, and resource costs.

### Function `stochastic_backprop_joint_objective(inputs, targets)`
Train a relaxed SC layer jointly over weights and SC architecture variables.

The selected config remains discrete and exportable, while length, encoding,
and correlation costs are computed from differentiable proxies so one
objective can update model parameters and stochastic-computing design
variables in the same backward pass.

---

## Module `training.stochastic_backprop_export`

### Function `build_stochastic_backprop_export_manifest(benchmark_report, sc_config)`
Build a deterministic SC-NIR handoff manifest for stochastic backpropagation.

### Function `_estimator_variance_export(benchmark_report)`
### Function `_joint_design_export(benchmark_report, sc_config)`
### Function `write_stochastic_backprop_export_manifest(path, benchmark_report, sc_config)`
Write a canonical stochastic backpropagation SC-NIR export manifest.

### Function `write_stochastic_backprop_handoff_bundle(export_manifest, output_dir)`
Materialise an auditable SC-NIR HDL handoff bundle from an export manifest.

### Function `_validate_benchmark_matches_config(benchmark_report, sc_config)`
### Function `_source_rows_from_scnir(scnir_document)`
### Function `_source_module_name(index, stream_id)`
### Function `_shared_bitstream_length(scnir_document)`
### Function `_signal_kind_counts(scnir_document)`
### Function `_write_top_module(path)`
### Function `_write_source_module(path)`
### Function `_write_trained_design_parity_bundle(export_manifest, root)`
### Function `_write_trained_design_module(path)`
### Function `_q16(value)`
### Function `_build_scnir_document(sc_config)`
### Function `_stream()`
### Function `_constraint()`
### Function `_empty_source()`
### Function `_source_kind(generator)`
### Function `_source_specific(source_kind, sc_config)`
### Function `_validate_manifest(manifest)`
### Function `_mapping(value, path)`
---

## Module `training.surrogate`

### Class `_FastSigmoidLegacy`
Zenke & Vogels 2021 (legacy autograd path).

- **forward**(ctx, x, slope)
- **backward**(ctx, grad_output)

### Class `_SuperSpikeLegacy`
Zenke & Ganguli 2018 (legacy autograd path).

- **forward**(ctx, x, beta)
- **backward**(ctx, grad_output)

### Class `_ATanLegacy`
Fang et al. 2021 (legacy autograd path).

- **forward**(ctx, x, alpha)
- **backward**(ctx, grad_output)

### Class `_SigmoidLegacy`
Standard sigmoid surrogate (legacy autograd path).

- **forward**(ctx, x, slope)
- **backward**(ctx, grad_output)

### Class `_StraightThroughLegacy`
Straight-through estimator (legacy autograd path).

- **forward**(ctx, x)
- **backward**(ctx, grad_output)

### Class `_TriangularLegacy`
Triangular window surrogate (legacy autograd path).

- **forward**(ctx, x, width)
- **backward**(ctx, grad_output)

### Function `_heaviside(x)`
### Function `_fast_sigmoid_grad(x, slope)`
### Function `_superspike_grad(x, beta)`
### Function `_atan_grad(x, alpha)`
### Function `_sigmoid_grad(x, slope)`
### Function `_triangular_grad(x, width)`
### Function `_fake_like_float(x)`
### Function `_fast_sigmoid_custom_op(x, slope)`
### Function `_fast_sigmoid_fake(x, slope)`
### Function `_fast_sigmoid_setup_context(ctx, inputs, output)`
### Function `_fast_sigmoid_backward(ctx, grad_output)`
### Function `_superspike_custom_op(x, beta)`
### Function `_superspike_fake(x, beta)`
### Function `_superspike_setup_context(ctx, inputs, output)`
### Function `_superspike_backward(ctx, grad_output)`
### Function `_atan_custom_op(x, alpha)`
### Function `_atan_fake(x, alpha)`
### Function `_atan_setup_context(ctx, inputs, output)`
### Function `_atan_backward(ctx, grad_output)`
### Function `_sigmoid_custom_op(x, slope)`
### Function `_sigmoid_fake(x, slope)`
### Function `_sigmoid_setup_context(ctx, inputs, output)`
### Function `_sigmoid_backward(ctx, grad_output)`
### Function `_straight_through_custom_op(x)`
### Function `_straight_through_fake(x)`
### Function `_straight_through_backward(ctx, grad_output)`
### Function `_triangular_custom_op(x, width)`
### Function `_triangular_fake(x, width)`
### Function `_triangular_setup_context(ctx, inputs, output)`
### Function `_triangular_backward(ctx, grad_output)`
### Function `_dispatch_path(path, custom_impl, legacy_impl)`
### Function `fast_sigmoid_custom_op(x, slope)`
Heaviside forward, fast-sigmoid backward via ``torch.library.custom_op``.

### Function `fast_sigmoid_legacy(x, slope)`
Heaviside forward, fast-sigmoid backward via legacy autograd.

### Function `fast_sigmoid(x, slope)`
Heaviside forward, fast-sigmoid backward.

### Function `superspike_custom_op(x, beta)`
Heaviside forward, SuperSpike backward via ``torch.library.custom_op``.

### Function `superspike_legacy(x, beta)`
Heaviside forward, SuperSpike backward via legacy autograd.

### Function `superspike(x, beta)`
Heaviside forward, SuperSpike backward.

### Function `atan_surrogate_custom_op(x, alpha)`
Heaviside forward, arctan backward via ``torch.library.custom_op``.

### Function `atan_surrogate_legacy(x, alpha)`
Heaviside forward, arctan backward via legacy autograd.

### Function `atan_surrogate(x, alpha)`
Heaviside forward, arctan backward.

### Function `sigmoid_surrogate_custom_op(x, slope)`
Heaviside forward, sigmoid backward via ``torch.library.custom_op``.

### Function `sigmoid_surrogate_legacy(x, slope)`
Heaviside forward, sigmoid backward via legacy autograd.

### Function `sigmoid_surrogate(x, slope)`
Heaviside forward, sigmoid backward.

### Function `straight_through_custom_op(x)`
Heaviside forward, identity backward via ``torch.library.custom_op``.

### Function `straight_through_legacy(x)`
Heaviside forward, identity backward via legacy autograd.

### Function `straight_through(x)`
Heaviside forward, identity backward (STE).

### Function `triangular_custom_op(x, width)`
Heaviside forward, triangular backward via ``torch.library.custom_op``.

### Function `triangular_legacy(x, width)`
Heaviside forward, triangular backward via legacy autograd.

### Function `triangular(x, width)`
Heaviside forward, triangular window backward.

---

## Module `training.utils`

### Class `SpikeMonitor`
Record spikes per layer during forward pass.

Attach to a SpikingNet or ConvSpikingNet to capture spike activity
at each layer per timestep. Useful for raster plots and diagnostics.

    monitor = SpikeMonitor(model)
    spk, mem = model(x)
    raster = monitor.get("lifs.0")  # (T, batch, n_neurons)
    monitor.reset()

- **__init__**(model)
- **_attach**()
- **_make_hook**(name)
- **get**(name)
  - Get recorded spikes for a named module. Returns (T, *shape) or None.
- **layer_names**()
- **reset**()
- **remove**()

### Function `reset_states(monitors)`
Clear SpikeMonitor recorded data.

Parameters
----------
monitors : list of SpikeMonitor, optional
    Monitors to reset. If None, does nothing.

Note: this does NOT reset membrane voltages or adaptation variables.
To reset neuron state, re-initialize the model or call forward()
with fresh zero-initialized hidden states. To reset a single
monitor, call ``monitor.reset()`` directly.

### Function `model_info(model)`
Quick architecture summary for SNN models.

### Function `population_decode(spike_counts, preferred_values)`
Population vector decoding.

Instead of argmax, computes a weighted average of preferred values
based on spike counts. More informative than winner-take-all.

Parameters
----------
spike_counts : torch.Tensor
    Shape (batch, n_neurons). Spike counts per neuron.
preferred_values : torch.Tensor or None
    Shape (n_neurons,) or (n_neurons, d). If None, uses neuron indices.

Returns
-------
torch.Tensor
    Decoded values, shape (batch,) or (batch, d).

---

## Module `transfer.checkpoint`

### Class `SNNCheckpoint`
Complete SNN model checkpoint.

- **n_layers**()
- **total_params**()

### Function `save_checkpoint(checkpoint, path)`
Save SNN checkpoint to .npz + .json.

Parameters
----------
checkpoint : SNNCheckpoint
path : str or Path
    Base path (without extension). Creates path.npz and path.json.

### Function `load_checkpoint(path)`
Load SNN checkpoint from .npz + .json.

Parameters
----------
path : str or Path
    Base path (without extension).

Returns
-------
SNNCheckpoint

### Function `_validate_metadata(meta)`
### Function `_require_string_list(meta, key)`
### Function `_optional_string_list(meta, key)`
### Function `_validate_layer_size(size)`
### Function `_validate_weight_array(array, key)`
---

## Module `transfer.fine_tune`

### Class `TransferConfig`
Configuration for transfer learning.

Parameters
----------
freeze_until : str or int
    Freeze all layers up to (and including) this layer name or index.
lr_backbone : float
    Learning rate for frozen layers (usually 0 or very small).
lr_head : float
    Learning rate for unfrozen layers.


### Function `freeze_layers(checkpoint, layer_names, until_index)`
Freeze layers (mark as non-trainable).

Parameters
----------
checkpoint : SNNCheckpoint
layer_names : list of str, optional
    Specific layers to freeze.
until_index : int, optional
    Freeze all layers with index <= until_index.

Returns
-------
SNNCheckpoint with frozen_layers updated

### Function `unfreeze_layers(checkpoint, layer_names, all_layers)`
Unfreeze layers (mark as trainable).

Parameters
----------
checkpoint : SNNCheckpoint
layer_names : list of str, optional
    Specific layers to unfreeze.
all_layers : bool
    If True, unfreeze everything.

### Function `apply_transfer_config(checkpoint, config)`
Apply transfer config: freeze layers, compute per-layer LR.

Returns
-------
(checkpoint, per_layer_lr)

---

## Module `transformers.block`

### Class `_AttentionHead`
- **forward**(Q, K, V)

### Class `StochasticTransformerBlock`
Spiking Transformer Block (S-Former).
Structure:
Input -> Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm -> Output

- **__post_init__**()
- **forward**(x)
  - x: (d_model,) or (Sequence_Length, d_model). Returns same shape.
- **_multi_head_attention**(x)

---

## Module `transformers.spikformer`

### Class `SpikeDrivenAttention`
Spike-Driven Self-Attention (SSA).

Replaces Q*K^T softmax with spike-based masking:
  Attention = SpikeFn(Q_linear(S)) * SpikeFn(K_linear(S))^T * V_linear(S)

All operations reduce to AND gates on binary spikes —
zero multiplications, pure SC-compatible logic.

Parameters
----------
embed_dim : int
    Embedding dimension.
num_heads : int
    Number of attention heads.
T : int
    Number of simulation timesteps.
threshold : float
    Spike threshold for Q/K projections.

- **__post_init__**()
- **_spike_fn**(membrane)
  - Integrate-and-fire: returns (spikes, new_membrane).
- **forward**(x)
  - Forward pass: spike-driven attention over T timesteps.
- **num_multiply_ops**()
  - Zero multiplications in the attention core (AND gates only).

### Class `SpikyStateSpace`
Spiking State-Space Model (S4-SNN hybrid).

Combines linear state-space dynamics with spiking nonlinearity:
  h_t = A * h_{t-1} + B * spike_input_t
  y_t = C * h_t
  spike_t = IF(y_t > threshold)

Runs in O(1) memory per timestep (no BPTT unrolling needed).
Reference: SpikySpace (2025).

Parameters
----------
d_model : int
    Input/output dimension.
d_state : int
    Hidden state dimension.
threshold : float
    Spiking threshold.
dt : float
    Discretization timestep.

- **__post_init__**()
- **reset**()
  - Reset hidden state and membrane potential.
- **step**(x)
  - Process one timestep.
- **forward**(x_seq)
  - Process a full sequence.

### Class `CPGPositionalEncoding`
Central Pattern Generator positional encoding.

Replaces sinusoidal positional encoding with biologically-inspired
CPG oscillators. Each dimension has a different frequency and phase,
generating spike-compatible temporal position signals.

Parameters
----------
d_model : int
    Encoding dimension.
max_len : int
    Maximum sequence length.

- **__post_init__**()
- **encode**(seq_len)
  - Generate positional encoding.
- **encode_spikes**(seq_len, rng)
  - Generate spike-encoded positional encoding.

---

## Module `utils.adapter_discovery`

### Function `discover_adapters()`
Auto-discover adapter plugins via importlib.metadata entry points.

Looks for entry points in group ``sc_neurocore.adapters``.
Each entry point should point to a class inheriting BaseStochasticAdapter.

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
Sliding-window probability estimator for bitstreams.

Example
-------
>>> avg = BitstreamAverager(window=100)
>>> for _ in range(100):
...     avg.push(1)
>>> avg.estimate()
1.0
>>> avg.push(0)
>>> avg.estimate() < 1.0
True

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

### Function `generate_bipolar_bitstream(x, length, rng)`
Generate a bipolar SC bitstream encoding a value in &#91;-1, +1&#93;.

Bipolar encoding: value x in &#91;-1, 1&#93; maps to probability p = (x + 1) / 2.
Bit=1 with probability p, bit=0 with probability 1-p.
Decoding: x = 2 * mean(bits) - 1.

Bipolar multiplication uses XNOR: P(A XNOR B) encodes A*B in bipolar.

### Function `bipolar_to_value(bitstream)`
Decode a bipolar bitstream to a value in &#91;-1, +1&#93;.

x = 2 * mean(bits) - 1

### Function `value_to_bipolar_prob(x)`
Map a value in &#91;-1, 1&#93; to the unipolar probability used in bipolar encoding.

p = (x + 1) / 2. This p is then used with standard Bernoulli generation.

### Function `value_to_unipolar_prob(x, x_min, x_max, clip)`
Map a scalar x from &#91;x_min, x_max&#93; into a unipolar probability &#91;0,1&#93;.
Linear mapping:
    p = (x - x_min) / (x_max - x_min)
If clip=True, x is clipped into &#91;x_min, x_max&#93;.

### Function `unipolar_prob_to_value(p, x_min, x_max)`
Map a unipolar probability p in &#91;0,1&#93; back to a scalar in &#91;x_min, x_max&#93;.
Inverse of value_to_unipolar_prob.

### Function `adaptive_length(p, epsilon, confidence, method, min_length, max_length)`
Compute minimum bitstream length for target precision.

Given probability p and error tolerance epsilon, returns the smallest L
such that |p_hat - p| < epsilon with the given confidence.

Parameters
----------
p : float
    Encoded probability in &#91;0, 1&#93;.
epsilon : float
    Maximum acceptable absolute error.
confidence : float
    Confidence level (e.g. 0.95 for 95%).
method : str
    Bound type: "hoeffding" (tighter), "chebyshev", or "variance" (no confidence).
min_length : int
    Minimum returned length.
max_length : int
    Maximum returned length (hardware cap).

Returns
-------
int
    Minimum bitstream length (rounded up to nearest power of 2 for Sobol compatibility).

### Function `sc_divide(numerator, denominator)`
Stochastic computing division via CORDIV circuit.

Li, Qian, Riedel & Bazargan, IEEE Trans. Signal Process. 62(9), 2014.

Sequential circuit: at each bit position t,
  - x&#91;t&#93;=1         → z&#91;t&#93; = 1
  - x&#91;t&#93;=0, y&#91;t&#93;=1 → z&#91;t&#93; = 0
  - x&#91;t&#93;=0, y&#91;t&#93;=0 → z&#91;t&#93; = z&#91;t-1&#93; (hold)

Converges to P(z=1) ≈ P(x=1) / P(y=1) when P(x) ≤ P(y).

Parameters
----------
numerator : np.ndarray
    Bitstream (uint8, {0,1}) of length L.
denominator : np.ndarray
    Bitstream (uint8, {0,1}) of length L. Must have higher or equal density.

Returns
-------
np.ndarray
    Quotient bitstream of length L.

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

## Module `utils.deprecation`

### Function `deprecated(since, removal, alternative)`
Mark a function or class as deprecated.

Example::

    @deprecated(since="3.11", removal="4.0", alternative="new_func")
    def old_func():
        ...

Parameters
----------
since : str
    Version where deprecation was introduced.
removal : str
    Version where the function will be removed.
alternative : str, optional
    Name of the replacement function/class.

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

## Module `utils.lds_decorrelation`

### Function `generate_decorrelated_bitstreams(probabilities, length, method, seed)`
Generate decorrelated bitstreams for a probability matrix.

Each element of the probability matrix gets its own LDS dimension,
ensuring zero correlation between any pair of bitstreams.

Parameters
----------
probabilities : np.ndarray
    Probability matrix, any shape. Values in &#91;0, 1&#93;.
length : int
    Bitstream length per element.
method : str
    "sobol" or "halton".
seed : int or None
    Random seed for scrambling.

Returns
-------
np.ndarray
    Shape (*probabilities.shape, length), dtype uint8.

### Function `star_discrepancy_estimate(samples, n_test)`
Estimate star discrepancy of a sample set (quality metric for LDS).

Lower discrepancy → more uniform coverage → better SC precision.

Parameters
----------
samples : np.ndarray
    Shape (n_samples, d), values in &#91;0, 1&#93;.
n_test : int
    Number of random test points.

Returns
-------
float
    Estimated star discrepancy.

---

## Module `utils.logging`

### Class `JSONFormatter`
Emit log records as single-line JSON objects.

- **format**(record)

### Function `configure_logging(level, json, stream)`
Configure the ``sc_neurocore`` logger hierarchy.

Parameters
----------
level : str or int
    Log level name (``"DEBUG"``, ``"INFO"``, etc.) or numeric level.
json : bool
    If *True*, use :class:`JSONFormatter` for machine-parseable output.
stream
    Output stream. Defaults to ``sys.stderr``.

---

## Module `utils.model_bridge`

### Class `SCBridge`
Bridge between standard DL frameworks (like PyTorch) and SC-NeuroCore.

- **load_from_state_dict**(state_dict, layer_mapping)
  - Load weights from a state_dict (numpy or torch tensors) into SC layers.
- **export_to_numpy**(layers)
  - Export SC weights back to numpy dictionary.

### Function `normalize_weights(weights)`
Normalizes weights to &#91;0, 1&#93; range for unipolar SC.

---

## Module `utils.numerics`

### Function `safe_exp(x)`
exp() with argument clipped to &#91;-500, 500&#93; to prevent overflow.

### Function `safe_cosh(x)`
cosh() with argument clipped to &#91;-500, 500&#93; to prevent overflow.

### Function `safe_tanh(x)`
tanh() with argument clipped to &#91;-500, 500&#93;.

### Function `boltzmann(v, v_half, k)`
Boltzmann sigmoid: 1 / (1 + exp((v_half - v) / k)). Overflow-safe.

### Function `boltzmann_inv(v, v_half, k)`
Inverse Boltzmann: 1 / (1 + exp((v - v_half) / k)). Overflow-safe.

### Function `clip_gating(x)`
Clip gating variable to physiological range &#91;0, 1&#93;.

### Function `clip_voltage(v, v_min, v_max)`
Clip membrane voltage to safe range (default &#91;-200, 100&#93; mV).

---

## Module `utils.profiling`

### Function `estimate_memory(layers, unit)`
Estimate memory usage of a list of SC layers.

Example::

    from sc_neurocore import VectorizedSCLayer
    from sc_neurocore.utils.profiling import estimate_memory

    layers = &#91;
        VectorizedSCLayer(n_inputs=50, n_neurons=128, length=256),
        VectorizedSCLayer(n_inputs=128, n_neurons=10, length=256),
    &#93;
    print(estimate_memory(layers))
    # {'weights_bytes': 56320, 'packed_bytes': 112640, ...}

Parameters
----------
layers : list
    SC layer objects with ``.weights`` and ``.length`` attributes.
unit : str
    "B", "KB", or "MB".

Returns
-------
dict
    Breakdown: weights_bytes, packed_bytes, neuron_state_bytes,
    total_bytes, total_human.

---

## Module `utils.registry`

### Class `ComponentRegistry`
Thread-safe component registry with namespace support.

- **__init__**()
- **_ensure_namespace_loaded**(namespace)
  - Load namespace-side registration modules on first access.
- **register**(namespace, name)
  - Decorator to register a class under *namespace*/*name*.
- **get**(namespace, name)
  - Retrieve a registered class. Raises ``KeyError`` if missing.
- **list**(namespace)
  - Return sorted names in *namespace*.
- **namespaces**()
  - Return all registered namespaces.
- **clear**(namespace)
  - Remove all entries (or just one namespace).

---

## Module `utils.rng`

### Class `RNG`
Thin wrapper around NumPy RNG for reproducible per-neuron streams.

Example
-------
>>> rng = RNG(seed=42)
>>> vals = rng.random(5)
>>> vals.shape
(5,)
>>> RNG(seed=42).random(5) == vals  # deterministic
array(&#91; True,  True,  True,  True,  True&#93;)

- **__init__**(seed)
- **normal**(mean, std, size)
- **uniform**(low, high, size)
- **bernoulli**(p, size)
- **random**(size)
- **shuffle**(x)

---

## Module `uvm_gen.uvm_gen`

### Class `PortDirection`

### Class `PortType`

### Class `ModulePort`
Parsed RTL port.

- **sv_decl**()
- **is_clock**()
- **is_reset**()

### Class `ModuleParam`
RTL module parameter.


### Class `RTLModule`
Parsed RTL module specification.

- **from_verilog_source**(cls, source)
  - Parse a Verilog/SystemVerilog module header.
- **input_ports**()
- **output_ports**()
- **clock_port**()
- **reset_port**()
- **total_input_bits**()
- **total_output_bits**()

### Class `StimulusConfig`
Configuration for randomised SC bitstream stimulus.


### Class `CoverageSpec`
Functional coverage specification.


### Class `ScoreboardConfig`
Scoreboard configuration for golden model comparison.


### Class `FormalLink`
Link between UVM assertions and SymbiYosys formal proofs.


### Class `SimTarget`
Simulation tool target for Makefile generation.


### Class `UVMBenchmark`
Complete generated UVM testbench.

- **to_dict**()

### Class `UVMGenerator`
Generates complete UVM verification IP from an RTL module spec.

- **__init__**(stimulus, coverage, scoreboard)
- **generate**(rtl)
  - Generate the complete UVM testbench for an RTL module.
- **generate_multi**(modules)
  - Generate UVM testbenches for multiple modules.
- **_emit_transaction**(rtl)
- **_emit_sequence**(rtl)
- **_emit_driver**(rtl)
- **_emit_monitor**(rtl)
- **_emit_scoreboard**(rtl)
- **_emit_coverage**(rtl)
- **_emit_agent**(rtl)
- **_emit_env**(rtl)
- **_emit_top**(rtl)
- **_emit_sby**(rtl)
- **_filelist**(rtl)
- **_emit_bind**(rtl)
- **_emit_makefile**(rtl, sim)
- **_emit_regression_list**(rtl)
- **generate_formal_links**(rtl)
  - Generate formal-to-dynamic links for existing SymbiYosys modules.

---

## Module `verification.formal_proofs`

### Class `Interval`
- **__add__**(other)
- **__mul__**(other)
- **__repr__**()

### Class `FormalVerifier`
Interval arithmetic checker for stochastic probability bounds and
energy safety constraints. Not an SMT solver.

- **verify_probability_bounds**(input_interval, weight_interval)
  - Prove that Output Probability is always in &#91;0, 1&#93;.
- **verify_energy_safety**(energy, cost)
  - Prove that operation will not consume more energy than available.

---

## Module `verification.safety`

### Class `CodeSafetyVerifier`
AST blocklist screen for auto-generated code.

Walks the AST and rejects code containing known-dangerous patterns:
filesystem mutation, process spawning, network access, code execution,
and unrestricted imports.

Limitations: this is a static blocklist, not a sandbox. It catches
common dangerous patterns but cannot prevent all malicious code.
Obfuscated calls (getattr chains, importlib indirection) may bypass it.
Do not use as a security boundary without additional sandboxing.

- **verify_code_safety**(source_code)
  - Static analysis of source code for dangerous patterns.
- **verify_logic_invariant**(func, input_sample, expected_condition)
  - Dynamic verification: run func and check output against condition.

---

## Module `verification.snn_standard`

### Class `VerificationLevel`
Evidence levels for SNN verification claims.


### Class `VerificationEvidenceKind`
Kinds of evidence accepted by the standard.


### Class `VerificationClaimStatus`
Status of one evidence item or standard requirement.


### Class `SNNVerificationEvidence`
One evidence item used in a formal SNN verification claim.

- **__post_init__**()
- **to_dict**()
  - Return a JSON-ready evidence record.

### Class `SNNVerificationRequirement`
One mandatory or optional requirement in a standard profile.

- **__post_init__**()
- **to_dict**()
  - Return a JSON-ready requirement.

### Class `SNNVerificationStandardProfile`
Named set of requirements for a formal SNN verification claim.

- **__post_init__**()
- **to_dict**()
  - Return a JSON-ready profile.

### Class `SNNVerificationRequirementResult`
Evaluation of one profile requirement.

- **to_dict**()
  - Return a JSON-ready requirement result.

### Class `SNNVerificationConformanceReport`
Conformance report for a profile and evidence set.

- **passed**()
  - Whether all mandatory requirements passed.
- **missing_mandatory**()
  - Mandatory requirement ids with missing evidence.
- **failed_mandatory**()
  - Mandatory requirement ids with failing evidence.
- **mandatory_coverage**()
  - Coverage ratio for mandatory requirements.
- **to_dict**()
  - Return a JSON-ready conformance report.

### Class `SNNVerificationStandard`
Evaluate SNN verification evidence against a standard profile.

- **__init__**(profile)
- **assess**(evidence)
  - Assess evidence against the configured profile.
- **_assess_requirement**(requirement, evidence_items)

### Function `publication_grade_snn_standard_profile()`
Return the default formal SNN verification standard profile.

### Function `assess_snn_verification_standard(evidence, profile)`
Assess evidence against the default or supplied SNN verification profile.

---

## Module `verification.temporal_properties`

### Class `PropertyResult`

### Class `Counterexample`
Input that violates a property.


### Class `VerificationResult`
Result of a temporal property check.

- **summary**()

### Function `fires_within(spikes, neuron_id, stimulus_times, max_latency)`
Verify that neuron fires within max_latency steps of each stimulus.

Parameters
----------
spikes : ndarray of shape (T, N)
neuron_id : int
stimulus_times : list of int
    Timesteps when stimulus was applied.
max_latency : int
    Maximum allowed response latency in timesteps.

### Function `mutual_exclusion(spikes, neuron_set)`
Verify that no two neurons in the set fire at the same timestep.

Parameters
----------
spikes : ndarray of shape (T, N)
neuron_set : list of int
    Neuron IDs that should never co-fire.

### Function `rate_bound(spikes, neuron_id, max_rate, window_size)`
Verify firing rate stays below max_rate in every sliding window.

Parameters
----------
spikes : ndarray of shape (T, N)
neuron_id : int
max_rate : float
    Maximum allowed firing rate (spikes per step).
window_size : int
    Sliding window size in timesteps.

### Function `refractory_guarantee(spikes, neuron_id, min_gap)`
Verify minimum inter-spike interval.

Parameters
----------
spikes : ndarray of shape (T, N)
neuron_id : int
min_gap : int
    Minimum required gap between consecutive spikes (timesteps).

### Function `causal_order(spikes, neuron_a, neuron_b, max_delay)`
Verify that neuron A fires before neuron B within max_delay.

For every spike of neuron B, there must be a spike of neuron A
within the preceding max_delay timesteps.

Parameters
----------
spikes : ndarray of shape (T, N)
neuron_a, neuron_b : int
max_delay : int

### Function `bounded_activity(spikes, neuron_set, window_size, max_total_spikes)`
Verify total spike count in neuron set stays bounded per window.

Parameters
----------
spikes : ndarray of shape (T, N)
neuron_set : list of int
window_size : int
max_total_spikes : int

---

## Module `viz.neuro_art`

### Class `NeuroArtGenerator`
Generates Art (Images) from Neural State.

- **generate_visual**(state_vector)
  - Maps a 1D state vector to a 2D RGB abstract image.

---

## Module `viz.plots`

### Function `_require_mpl()`
### Function `_get_ax(ax)`
Return existing axes or create a new figure+axes pair.

### Function `raster_plot(spike_monitor, ax, color, marker, s)`
Spike raster from a SpikeMonitor.

### Function `voltage_trace(state_monitor, neuron_ids, ax)`
Membrane voltage traces from a StateMonitor.

### Function `firing_rate_plot(spike_monitor, bin_ms, ax)`
Population firing rate histogram (spikes per bin).

### Function `isi_histogram(spike_monitor, neuron_id, bins, ax)`
Inter-spike interval distribution for a single neuron.

### Function `cross_correlogram(spike_monitor, i, j, max_lag_ms, ax)`
Spike cross-correlation between neurons *i* and *j*.

### Function `population_activity(spike_monitor, bin_ms, ax)`
Heatmap of binned spike counts per neuron.

### Function `phase_portrait(state_monitor, var_x, var_y, neuron_id, ax)`
2-D phase-plane trajectory for a single neuron.

### Function `weight_matrix(projection, ax, cmap)`
Connectivity weight heatmap from a Projection's CSR data.

### Function `network_graph(network, ax)`
Node/edge diagram of populations and projections.

### Function `psd_plot(spike_monitor, neuron_id, ax)`
Power spectral density of a single neuron's spike train.

### Function `instantaneous_rate_plot(spike_monitor, neuron_id, sigma_ms, ax)`
Gaussian-kernel-smoothed instantaneous firing rate.

### Function `spike_train_comparison(trains, labels, ax)`
Overlay multiple spike trains as event plots.

Parameters
----------
trains : list of np.ndarray
    Each element is a 1-D array of spike timesteps.
labels : list of str, optional
    Labels for each train.

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

### Class `LinearGaussianSSM`
Parameters of a discrete-time linear Gaussian state-space model.

All matrices are NumPy arrays of dtype float64. Shapes:

- ``A``: (d, d) state transition
- ``B``: (d, m) control input
- ``C``: (p, d) observation
- ``D``: (p, m) feed-through
- ``Q``: (d, d) process noise covariance (PSD)
- ``R``: (p, p) observation noise covariance (PSD)
- ``mu_0``: (d,) prior mean
- ``Sigma_0``: (d, d) prior covariance

- **state_dim**()
- **obs_dim**()
- **control_dim**()
- **__post_init__**()
- **random**(cls, state_dim, obs_dim, control_dim, seed)
  - Construct a stable random LGSSM for smoke tests / initialisation.

### Class `FilterResult`
Output of `KalmanFilter.filter()`.

Attributes
----------
means : np.ndarray, shape (T, d)
    Filtered means E&#91;x_t | y_{1:t}, u_{1:t}&#93;.
covariances : np.ndarray, shape (T, d, d)
    Filtered covariances Cov&#91;x_t | y_{1:t}, u_{1:t}&#93;.
pred_means : np.ndarray, shape (T, d)
    One-step-ahead predicted means E&#91;x_t | y_{1:t-1}, u_{1:t-1}&#93;.
pred_covariances : np.ndarray, shape (T, d, d)
    One-step-ahead predicted covariances.
log_likelihood : float
    Log p(y_{1:T} | u_{1:T}) under the model — used by EM.


### Class `KalmanFilter`
Forward Kalman filter for `LinearGaussianSSM`.

Computes the filtering distribution p(x_t | y_{1:t}, u_{1:t})
for a fully-observed sequence of observations + (optional)
controls. Algorithm: standard Kalman update equations
(Bishop 2006 §13.3.1).

- **__init__**(model)
- **filter**(observations, controls, backend)
  - Run forward filtering on a sequence.
- **_filter_rust**(observations, controls)
  - Forward-pass dispatch to the Rust LGSSM Kalman filter.
- **_filter_julia**(observations, controls)
  - Forward-pass dispatch to the Julia LGSSM Kalman filter.
- **_filter_go**(observations, controls)
  - Forward-pass dispatch to the Go LGSSM Kalman filter.
- **_filter_mojo**(observations, controls)
  - Forward-pass dispatch to the Mojo LGSSM Kalman filter.

### Class `SmoothResult`
Output of `RTSSmoother.smooth()`.

Attributes
----------
means : np.ndarray, shape (T, d)
    Smoothed means E&#91;x_t | y_{1:T}, u_{1:T}&#93;.
covariances : np.ndarray, shape (T, d, d)
    Smoothed covariances Cov&#91;x_t | y_{1:T}, u_{1:T}&#93;.
cross_covariances : np.ndarray, shape (T-1, d, d)
    Lag-1 smoothed cross-covariances Cov&#91;x_t, x_{t+1} | y_{1:T}&#93;.
    Required by the EM M-step.


### Class `RTSSmoother`
Rauch-Tung-Striebel backward smoother (1965).

Consumes a `FilterResult` (forward pass) and produces the
smoothing distribution p(x_t | y_{1:T}, u_{1:T}) for every t.

- **__init__**(model)
- **smooth**(filter_result)

### Class `EMLearner`
Expectation-Maximisation parameter estimator for LGSSM.

Reference: Shumway & Stoffer (1982), Bishop (2006) §13.3.2.

Per iteration:
  E-step: run Kalman filter + RTS smoother → posterior means,
          covariances, cross-covariances.
  M-step: closed-form update for {A, C, Q, R, mu_0, Sigma_0}
          from the smoothed posterior. (B and D are treated
          as known; estimating them simultaneously requires
          a more involved derivation.)

Convergence: log-likelihood is monotone non-decreasing
across iterations under exact arithmetic.

- **__init__**(max_iter, tol)
- **fit**(observations, initial_model, controls)
  - Estimate model parameters from a single observation sequence.

### Class `PredictiveWorldModel`
Probabilistic predictive world model based on a Linear Gaussian SSM.

The legacy 65-LOC `predict_next_state` / `forecast` API is
preserved as a thin wrapper on the proper `LinearGaussianSSM`
+ `KalmanFilter` infrastructure above. The previous
deterministic linear matmul + clip implementation was replaced
2026-04-17 per `feedback_sophisticated_from_start.md`.

- **__post_init__**()
- **reset**()
- **predict_next_state**(current_state, action)
  - Predict E&#91;x_{t+1} | x_t, u_t&#93; under the SSM dynamics.
- **predict_next_state_with_cov**(current_state, current_cov, action)
  - Predict mean + covariance of x_{t+1} given (x_t, Σ_t, u_t).
- **forecast**(initial_state, actions)
  - Multi-step deterministic forecast (mean trajectory).
- **forecast_with_cov**(initial_state, initial_cov, actions)
  - Multi-step probabilistic forecast (mean + cov trajectory).

### Function `_missing_rust_kalman_filter()`
### Function `_load_rust_kalman_filter()`
### Function `_ensure_mojo_loaded()`
Lazy-load the Mojo LGSSM shared library on first request.

Built once via:
  cd src/sc_neurocore/accel/mojo/world_model
  mojo build --emit shared-lib -o liblgssm.so lgssm.mojo

Per `feedback_mojo_026_ffi_pattern.md`, the @export signature
accepts only non-parametric types — we pass every matrix as a
raw `int64` address (numpy `arr.ctypes.data`) and the Mojo
side reconstructs `UnsafePointer&#91;Float64, MutAnyOrigin&#93;`
inside the function.

### Function `_ensure_go_loaded()`
Lazy-load the Go LGSSM shared library on first request.

The .so is built once via:
  cd src/sc_neurocore/accel/go/lgssm
  go build -buildmode=c-shared -o liblgssm.so lgssm.go

Returns True if the .so loads + has the expected symbol;
False otherwise (with reason logged into the caller's
`unavailable_reason` channel).

### Function `_ensure_julia_loaded()`
Lazy-load the Julia LGSSM module on first request.

Returns True if Julia + the .jl module are available; False otherwise.
Caches the loaded module in `_julia_module` for subsequent calls.
Julia startup latency is ~5 s — never paid unless `backend='julia'`
is explicitly requested.

---

## Module `world_model.spike_predictor`

### Class `SpikePredictor`
Online autoregressive spike pattern predictor.

Learns to predict spike&#91;t&#93; from spike&#91;t-K:t&#93; per channel.
Weight matrix W of shape (N, N*K) maps flattened history to
per-channel firing probabilities. Binary prediction via threshold.

Training: LMS update after each timestep.
    W += lr * outer(error, history)
where error = actual - predicted_prob.

Parameters
----------
n_channels : int
    Number of spike channels.
history_len : int
    Number of past timesteps to use as context (K).
lr : float
    LMS learning rate.
threshold : float
    Probability threshold for binary prediction.
seed : int
    RNG seed for weight initialization.

- **__post_init__**()
- **_features**()
  - Flatten history buffer into feature vector.
- **predict_probs**()
  - Predict per-channel firing probabilities from history.
- **predict**()
  - Predict binary spike pattern.
- **update**(actual)
  - Update weights with observed spike pattern (LMS rule).
- **reset**()
  - Reset to initial state (same seed → same weights).

### Function `predict_and_xor_world_model(spikes, n_channels, history_len, lr, threshold, seed)`
World-model predict-XOR loop for codec compression.

Returns (errors, correct_count).

### Function `xor_and_recover_world_model(errors, n_channels, history_len, lr, threshold, seed)`
World-model XOR-recover loop for codec decompression.

---
