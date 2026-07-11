# SC-NeuroCore API Reference

## Module `_native.array_guards`

### Function `require_c_contiguous(arr, name, dtype)`
Validate array layout before a zero-copy native call.

---

## Module `_native.core_engine_bridge`

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
  - Return the Rust-reported fixed-point state footprint in bits.
- **__del__**()

### Class `RustPlasticityRule`
RAII wrapper around a Rust plasticity rule handle.

rule_type: RULE_STDP(0), RULE_BCM(1), RULE_REWARD_STDP(2), RULE_ELIGENT(3)

- **__init__**(rule_type, weight, param_a, param_b)
- **step**(pre_spike, post_spike, dt, reward)
  - Advance one plasticity-rule timestep through the Rust FFI.
- **step_batched**(pre_spikes, post_spikes, rewards, dt)
  - Process Numpy arrays in a single FFI boundary crossing.
- **weight**()
  - Return the current Rust-managed rule weight.
- **reset**()
  - Reset rule traces while preserving the Rust rule handle.
- **__del__**()

### Class `RustEligentLearner`
RAII wrapper around a Rust ELIGENT learner handle.

- **__init__**(threshold, target_rate, weight)
- **step**(fired, pre_spike, global_reward, dt)
  - Advance one ELIGENT learner timestep through the Rust FFI.
- **step_batched**(fired_slice, pre_spikes, rewards, dt)
  - Process batched ELIGENT learner events in one FFI call.
- **__del__**()

### Class `RustRuleLayer`
RAII wrapper around a Rust RuleLayerHandle for parallelized spatial (layer) execution.

- **__init__**(count, rule_type, weight, param_a, param_b)
- **__getstate__**()
- **get_state_dict**()
  - Return a Python state dictionary with Rust serialization bytes.
- **__setstate__**(state)
- **load_state_dict**(state_dict)
  - Restore a Rust rule layer from a Python state dictionary.
- **step**(pre_spikes, post_spikes, rewards, dt)
  - Process spatial Numpy arrays natively across Rayon Rust threads.
- **step_analog**(pre_probs, post_probs, rewards, dt, seed)
  - Process MTJ analogue probability states directly into sampled stochastic spikes using Rayon RNG.
- **get_weights**()
  - Return layer weights copied from the Rust rule layer.
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
  - Advance one Rust-WGPU rule-layer step.
- **step_analog**(pre_probs, post_probs, rewards, dt, seed)
  - Advance analog probabilities through the Rust-WGPU step path.
- **get_weights**()
  - Return weights copied from the Rust-WGPU layer.
- **get_state_dict**()
  - Return the minimal serializable Rust-WGPU weight state.
- **load_state_dict**(state_dict)
  - Warn that Rust-WGPU state restore relies on reinitialization.
- **reset**()
  - Zero WGSL plasticity traces; weights preserved. See `WgpuRuleLayer::reset`.
- **__del__**()

### Function `set_deterministic_mode(seed)`
Force exact random generator seeds for bit-true SC formal verification.

### Function `is_available()`
Return True if the Rust learning engine is loaded.

---

## Module `accel.backend`

### Class `Backend`
Acceleration backend handle exposing the stable SC inference contract.

- **sc_forward**(weights_packed, input_probs)
  - Run the stochastic forward pass; see :func:`sc_neurocore.accel.sc_forward`.

### Class `NumpyBackend`
NumPy fallback backend (always available, the bit-true floor).

- **sc_forward**(weights_packed, input_probs)
  - Run the NumPy bit-true stochastic forward pass.

### Class `RustBackend`
Rust-accelerated backend over the compiled engine.

- **sc_forward**(weights_packed, input_probs)
  - Run the compiled Rust stochastic forward pass.

### Function `available_backends()`
Report which SC inference backends resolve, in fastest-first order.

Returns
-------
dict
    Mapping of backend name to availability; ``numpy`` is always ``True``.

### Function `get_backend(name)`
Return an SC inference backend handle.

Parameters
----------
name : str, optional
    ``"auto"`` (default) returns the fastest available backend in
    :data:`PRIORITY` order; a specific name (``"rust"``, ``"numpy"``) forces
    that backend.

Returns
-------
Backend
    A handle whose ``sc_forward`` matches the documented contract.

Raises
------
ValueError
    If ``name`` is not ``"auto"`` or a known backend name.
RuntimeError
    If an explicitly requested backend is unavailable.

---

## Module `accel.backend_order`

### Function `with_floor(floor)`
Return :data:`ACCELERATORS` with ``floor`` appended as the always-available tier.

---

## Module `accel.backend_selection`

### Function `current_cpu()`
Return the host CPU model string, matching the benchmark harness convention.

Mirrors ``benchmarks/bench_*.py::_cpu_model`` so a live host matches a stored
``meta.cpu`` exactly: the ``/proc/cpuinfo`` ``model name`` line, else
:func:`platform.processor`, else ``"unknown"``.

### Function `measured_orders()`
Build ``{cpu: {kernel: (backends fastest-measured-first)}}`` from the JSONs.

Cached: the on-disk benchmarks are immutable for the life of the process.
Malformed or non-comparison files (no ``backends`` / ``kernel`` / ``meta.cpu``)
are skipped silently — they are simply not a usable measurement.

### Function `select_backend_order(kernel)`
Return the dispatch order for ``kernel``, reordered from measured benchmarks.

Compiled backends measured on this host's CPU lead, fastest-first; any backend
in ``static`` without a measurement keeps its static position after them; the
floor (``static&#91;-1&#93;``) is always last. With no matching measurement the static
order is returned verbatim.

---

## Module `accel.gpu_backend`

### Function `to_device(arr)`
Move a NumPy array to the active backend (GPU copy or no-op).

### Function `to_host(arr)`
Bring an array back to host RAM as a NumPy array.

### Function `gpu_pack_bitstream(bits)`
Pack uint8 {0,1} array into uint64 words.

Works on both CuPy and NumPy arrays.

Args:
    bits: Shape ``(N,)`` or ``(B, N)`` of uint8.

Returns
-------
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

Returns
-------
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
Pack a uint8 bitstream into a uint64 word array.

Parameters
----------
bitstream : numpy.ndarray of shape (N,), uint8
    Input bits valued in ``{0, 1}``.
packed_arr : numpy.ndarray of shape (N // 64,), uint64
    Output array receiving the packed 64-bit words.

### Function `jit_vec_mac(packed_weights, packed_inputs, outputs)`
Accumulate a packed bitwise multiply-accumulate (MAC).

Computes ``outputs&#91;i&#93; = sum(popcount(packed_weights&#91;i&#93; AND packed_inputs))``.

Parameters
----------
packed_weights : numpy.ndarray of shape (n_neurons, n_inputs, n_words), uint64
    Packed synaptic weight bitstreams.
packed_inputs : numpy.ndarray of shape (n_inputs, n_words), uint64
    Packed input bitstreams.
outputs : numpy.ndarray of shape (n_neurons,)
    Output array receiving the accumulated MAC results.

---

## Module `accel.mojo.runner`

### Class `MojoKernelRunner`
Run the maintained monolithic Mojo kernel suite from Python.

The runner is a subprocess façade over ``kernels.mojo``. It discovers the
kernel bundle at construction time, invokes the pixi-managed Mojo toolchain
for builds and benchmark runs, and keeps Python fallbacks for scalar helper
methods. Those scalar helpers do not attempt hidden Mojo IPC; benchmark
execution remains the explicit Mojo subprocess surface.

Parameters
----------
_mojo_dir:
    Directory expected to contain ``kernels.mojo``. The default is the
    installed package directory.
_pixi_bin:
    Absolute pixi executable used to launch the Mojo toolchain.

- **__post_init__**()
  - Validate the configured Mojo kernel directory.
- **build**()
  - Build ``kernels.mojo`` through pixi.
- **run_benchmark**(timeout_sec)
  - Run the kernel benchmark suite and parse millisecond timings.
- **popcount**(data)
  - Return the Python Hamming weight of packed stochastic words.
- **lfsr_encode**(seed, threshold, bits)
  - Encode a threshold stream with the maintained LFSR-16 fallback.

---

## Module `accel.mpi_driver`

### Class `MPIDriver`
Distributed sc-neurocore driver built on MPI.

Handles partitioning and synchronisation of bitstreams across cluster
nodes.

- **__init__**()
- **scatter_workload**(global_inputs)
  - Distribute a large input array across nodes along axis 0.
- **gather_results**(local_results)
  - Collect per-node result arrays back to the root rank.
- **barrier**()
  - Synchronize all nodes.

---

## Module `accel.sc_inference`

### Function `sc_forward_numpy(weights_packed, input_probs, length, seed)`
NumPy reference for :func:`sc_forward` — the bit-true floor.

Parameters
----------
weights_packed : array_like
    Pre-packed unipolar weight bitstreams, shape ``(n_out, n_in, n_words)``
    ``uint64`` with ``n_words = ceil(length / 64)``.
input_probs : array_like
    Input probabilities, shape ``(n_in,)`` ``float64`` in ``&#91;0, 1&#93;``.
length : int
    Bitstream length.
seed : int, optional
    Base LFSR seed for the input encoder.

Returns
-------
numpy.ndarray
    ``(n_out,)`` ``float64`` AND-then-popcount estimate of
    ``weights @ input_probs`` divided by ``length``.

Raises
------
ValueError
    If shapes are inconsistent or probabilities lie outside ``&#91;0, 1&#93;``.

### Function `sc_forward(weights_packed, input_probs)`
Stochastic forward pass over caller-owned packed weight bitstreams.

Encodes ``input_probs`` into LFSR bitstreams, ANDs them against the pre-packed
weights and returns the popcount estimate of ``weights @ input_probs``. The
accelerated and NumPy backends are bit-identical for a fixed ``seed``.

Parameters
----------
weights_packed : array_like
    ``(n_out, n_in, n_words)`` ``uint64`` packed unipolar weight bitstreams.
input_probs : array_like
    ``(n_in,)`` ``float64`` input probabilities in ``&#91;0, 1&#93;``.
length : int
    Bitstream length (keyword-only).
backend : str or Backend, optional
    ``"auto"`` selects the fastest available backend; a name or a
    :class:`~sc_neurocore.accel.backend.Backend` forces one.
seed : int, optional
    Base LFSR seed for the input encoder.

Returns
-------
numpy.ndarray
    ``(n_out,)`` ``float64`` estimate of ``weights @ input_probs``.

---

## Module `accel.vector_ops`

### Function `pack_bitstream(bitstream)`
Pack a uint8 bitstream into uint64 words for 64-way parallel processing.

Parameters
----------
bitstream : numpy.ndarray of shape (N,) or (Batch, N), uint8, or list&#91;int&#93;
    Input bits valued in ``{0, 1}``. Python lists are accepted for the
    one-dimensional compatibility path and are converted to ``uint8``.

Returns
-------
numpy.ndarray of shape (ceil(N / 64),) or (Batch, ceil(N / 64)), uint64
    The packed 64-bit words.

### Function `unpack_bitstream(packed, original_length, original_shape)`
Unpacks uint64 array back to uint8 bitstream.

Args:
    packed: Packed uint64 array (1D or 2D)
    original_length: Total number of bits to extract
    original_shape: Optional tuple for reshaping output (batch, length)

Returns
-------
    Unpacked bitstream of shape (original_length,) or original_shape

### Function `vec_and(a_packed, b_packed)`
Bitwise-AND two packed arrays, realising stochastic multiplication.

### Function `vec_xnor(a_packed, b_packed)`
Bitwise XNOR on packed arrays. SC bipolar multiplication: P(A XNOR B) = P(A)*P(B) + (1-P(A))*(1-P(B)).

### Function `vec_not(packed)`
Bitwise NOT on packed arrays. SC complement: P(NOT A) = 1 - P(A).

### Function `vec_mux(select_packed, a_packed, b_packed)`
Bitwise MUX on packed arrays. SC scaled addition: P(out) = P(sel)*P(A) + (1-P(sel))*P(B).

When sel is a Bernoulli(0.5) stream, this computes the average (A+B)/2.

### Function `vec_popcount(packed)`
Count the total set bits in a packed array, for integration/accumulation.

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
- **step_jax**(dt, inputs)
  - Advances the L12 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Gaian Synchrony Index.
- **get_metrics**()
  - Returns L12-specific metrics like Coherence and Flow.

---

## Module `adapters.holonomic.l13_source`

### Class `L13_HolonomicParameters`
Configuration for the Layer 13 source-field adapter.

Parameters
----------
n_vacuum_nodes:
    Number of nodes in the one-dimensional vacuum lattice.
bitstream_length:
    Number of stochastic bits emitted per vacuum node.
j_primordial_coupling:
    Finite nearest-neighbour primordial lattice coupling.
h_potential_bias:
    Finite scalar source-field bias applied to each node.
lambda_scission:
    Finite non-negative symmetry-breaking drive.


### Class `L13_SourceAdapter`
JAX-traceable adapter for the SCPN source-field layer.

- **__init__**(params, seed)
  - Initialise the Layer 13 source-field adapter.
- **encode**(domain_state)
  - Map vacuum potential to stochastic source-field bitstreams.
- **step_jax**(dt, inputs)
  - Advance the L13 holonomic dynamics using JAX-compatible arrays.
- **decode**(bitstreams)
  - Map bitstreams back to primordial source coherence.
- **get_metrics**()
  - Return L13-specific vacuum and Fisher-metric telemetry.

---

## Module `adapters.holonomic.l14_trans`

### Class `L14_HolonomicParameters`
Parameters derived from Paper 14 and Keystone Tuning specs.


### Class `L14_TransdimensionalAdapter`
JAX-traceable adapter for the SCPN Transdimensional layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps resonance alignment to stochastic bitstreams.
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
- **encode**(domain_state)
  - Maps neurochemical concentrations to stochastic bitstreams.
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
- **step_jax**(dt, inputs)
  - Advances the L5 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to average valence and HRV coherence.
- **get_metrics**()
  - Returns L5-specific metrics.

---

## Module `adapters.holonomic.l6_plan`

### Class `L6_HolonomicParameters`
Configuration for the Layer 6 Gaia-field adapter.

Parameters
----------
n_regions:
    Number of regional planetary field nodes.
bitstream_length:
    Number of stochastic bits emitted per region.
f_schumann:
    Positive finite Schumann resonance frequency in hertz.
q_factor:
    Positive finite cavity quality factor.
alpha_gaia:
    Positive finite regional-to-planetary coupling strength.
p_percolation:
    Critical percolation threshold in the open interval ``(0, 1)``.


### Class `L6_PlanetaryAdapter`
JAX-traceable adapter for the SCPN planetary-biospheric layer.

- **__init__**(params, seed)
  - Initialise the Layer 6 planetary adapter.
- **encode**(domain_state)
  - Map planetary coherence to stochastic regional bitstreams.
- **step_jax**(dt, inputs)
  - Advance the L6 holonomic dynamics using JAX-compatible arrays.
- **decode**(bitstreams)
  - Map bitstreams back to the global coherence index.
- **get_metrics**()
  - Return L6-specific Gaia and Schumann telemetry.

---

## Module `adapters.holonomic.l7_sym`

### Class `L7_HolonomicParameters`
Parameters derived from Paper 7 and Metatron's Cube geometry.


### Class `L7_SymbolicAdapter`
JAX-traceable adapter for the SCPN geometrical-symbolic layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Map symbolic phases to stochastic bitstreams.
- **step_jax**(dt, inputs)
  - Advances the L7 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Map bitstreams back to symbolic coherence telemetry.
- **get_metrics**()
  - Return L7-specific routing and phase-stability metrics.

---

## Module `adapters.holonomic.l8_cosm`

### Class `L8_HolonomicParameters`
Parameters derived from Paper 8 and PTA specifications.


### Class `L8_CosmicAdapter`
JAX-traceable adapter for the SCPN Cosmic Phase-Locking layer.

- **__init__**(params, seed)
- **encode**(domain_state)
  - Maps cosmic phases to stochastic bitstreams.
- **step_jax**(dt, inputs)
  - Advances the L8 holonomic dynamics using JAX.
- **decode**(bitstreams)
  - Maps bitstreams back to Cosmic Alignment.
- **get_metrics**()
  - Returns L8-specific metrics.

---

## Module `adapters.holonomic.l9_mem`

### Class `L9_HolonomicParameters`
Configuration for the Layer 9 TSVF memory adapter.

Parameters
----------
n_memory_slots:
    Number of holographic memory rows retained by the adapter.
bitstream_length:
    Number of stochastic bits carried by each memory row.
retrieval_gain:
    Non-negative multiplier applied to the total TSVF overlap.
weak_measurement_strength:
    Bounded measurement coupling reserved for the TSVF update kernel.
temporal_window:
    Positive temporal-memory window used by downstream Layer 9 consumers.


### Class `L9_MemoryAdapter`
JAX-traceable adapter for the SCPN existential-memory layer.

- **__init__**(params, seed)
  - Initialise the Layer 9 memory adapter.
- **encode**(domain_state)
  - Map memory imprints to stochastic bitstreams via TSVF overlap.
- **step_jax**(dt, inputs)
  - Advance the L9 holonomic dynamics using JAX-compatible arrays.
- **decode**(bitstreams)
  - Map bitstreams back to memory-retrieval quality.
- **get_metrics**()
  - Return L9-specific overlap and imprint-density metrics.

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

## Module `adapters.importers`

### Class `NeuroMLImporter`
Class-oriented registry surface for NeuroML 2 cell imports.

- **import_cells**(path)
  - Parse a NeuroML 2 XML file into imported cell definitions.
- **create_neuron**(cell)
  - Instantiate a neuron from a parsed NeuroML cell definition.

### Class `SONATAImporter`
Class-oriented registry surface for SONATA network imports.

- **import_network**(nodes_path, edges_path)
  - Import a SONATA network from nodes and optional edges files.

### Class `SpikeInterfaceImporter`
Class-oriented registry surface for spike-train conversion imports.

- **to_bitstreams**(spike_times, duration_ms, dt)
  - Convert spike times to a binary bitstream matrix.
- **to_population_input**(spike_times, duration_ms, dt)
  - Convert spike times to ``Population.step_all`` input.
- **to_probabilities**(spike_times, duration_ms, max_rate_hz)
  - Convert spike trains into bounded stochastic-computing probabilities.

---

## Module `adapters.neuroml`

### Class `ImportedCell`
Result of importing a NeuroML cell definition.


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
  - Return the bundled BrainScaleS-3 substrate profile.
- **dynapse2**(cls)
  - Return the bundled DynapSE-2 substrate profile.

### Class `AEREvent`
Address-Event Representation spike event.


### Class `AnalogBridge`
Quantize stochastic weights and thresholds for analog substrates.

- **__init__**(g_range, v_range, dac_res, profile)
- **emit_analog_config**(nodes)
  - Emit DAC configuration dictionaries for SC weights and LIF nodes.

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
Map spike-vector activity to human-readable semantic concepts.

- **__init__**(concept_map)
- **explain**(spikes)
  - Describe the concepts implied by an active spike vector.

---

## Module `analysis.phi_estimation`

### Function `phi_star(data, tau, backend)`
Geometric integrated information Phi* (Barrett & Seth 2011).

``Phi* = MI(past; future) - min_partition Σ_k MI(past_k; future_k)`` under the
Gaussian assumption, with each mutual information a difference of covariance
log-determinants (Cholesky form).

Parameters
----------
data : numpy.ndarray
    Shape ``(n_channels, n_timesteps)`` — spike counts or continuous signals.
tau : int, optional
    Time lag for the past→future mapping.
backend : str, optional
    ``"auto"`` selects the fastest available backend (Rust engine when present,
    otherwise the NumPy reference); ``"python"``, ``"rust"``, ``"julia"``,
    ``"go"`` and ``"mojo"`` force a specific path.

Returns
-------
float
    Phi* estimate in nats. Non-negative; ``0`` means fully reducible.

### Function `phi_from_spike_trains(spikes, bin_size, tau, backend)`
Compute Phi* from binary spike trains by binning into spike counts.

Parameters
----------
spikes : numpy.ndarray
    Shape ``(n_neurons, n_timesteps)``, binary ``{0, 1}``.
bin_size : int, optional
    Number of timesteps per bin.
tau : int, optional
    Time lag in bins.
backend : str, optional
    Forwarded to :func:`phi_star`.

Returns
-------
float
    Phi* in nats.

---

## Module `analysis.spike_stats.basic`

### Function `spike_times(binary_train, dt)`
Extract spike times (seconds) from a binary 0/1 array.

### Function `isi(binary_train, dt)`
Inter-spike intervals (seconds) from a binary train.

### Function `firing_rate(binary_train, dt)`
Mean firing rate (Hz).

### Function `spike_count(binary_train)`
Return the total number of spikes in a binary spike train.

### Function `bin_spike_train(binary_train, bin_size)`
Bin a binary spike train into spike counts per bin.

---

## Module `analysis.spike_stats.causality`

### Function `pairwise_granger_causality(source, target, bin_size, order)`
Return pairwise Granger causality from ``source`` to ``target``.

Parameters
----------
source:
    One-dimensional binary or count-valued source spike train.
target:
    One-dimensional binary or count-valued target spike train.
bin_size:
    Positive number of samples per spike-count bin.
order:
    Positive autoregressive model order.

Returns
-------
float
    Log-likelihood ratio. Positive values indicate that past source counts
    reduce target prediction error under the regularised Granger model.

Raises
------
ValueError
    If ``bin_size`` or ``order`` is not positive, or if either train is not
    one-dimensional and finite.

### Function `conditional_granger_causality(source, target, condition, bin_size, order)`
Return conditional Granger causality from ``source`` to ``target``.

The reduced model predicts ``target`` from its own history and the
``condition`` history. The full model adds ``source`` history, following the
Geweke conditional Granger construction.

Parameters
----------
source:
    One-dimensional binary or count-valued source spike train.
target:
    One-dimensional binary or count-valued target spike train.
condition:
    One-dimensional binary or count-valued conditioning spike train.
bin_size:
    Positive number of samples per spike-count bin.
order:
    Positive autoregressive model order.

Returns
-------
float
    Log-likelihood ratio after controlling for ``condition``. Positive
    values indicate source-specific predictive information.

Raises
------
ValueError
    If ``bin_size`` or ``order`` is not positive, or if any train is not
    one-dimensional and finite.

### Function `spectral_granger_causality(trains, bin_size, order, n_freqs)`
Return frequency-domain Granger causality for a spike-train population.

Parameters
----------
trains:
    Non-empty list of one-dimensional spike trains. All trains must produce
    the same number of bins.
bin_size:
    Positive number of samples per spike-count bin.
order:
    Positive autoregressive model order.
n_freqs:
    Positive number of frequencies in the closed interval ``&#91;0, 0.5&#93;``.

Returns
-------
np.ndarray
    Array with shape ``(n_neurons, n_neurons, n_freqs)``. Singular transfer
    matrices are skipped and leave the corresponding frequency slice at
    zero rather than raising during the inverse.

Raises
------
ValueError
    If any domain parameter is not positive, if the population is empty, if
    any train is not one-dimensional and finite, or if binned lengths differ.

### Function `partial_directed_coherence(trains, bin_size, order, n_freqs)`
Return partial directed coherence for a spike-train population.

Parameters
----------
trains:
    Non-empty list of one-dimensional spike trains. All trains must produce
    the same number of bins.
bin_size:
    Positive number of samples per spike-count bin.
order:
    Positive autoregressive model order.
n_freqs:
    Positive number of frequencies in the closed interval ``&#91;0, 0.5&#93;``.

Returns
-------
np.ndarray
    Normalised PDC tensor with shape ``(n_neurons, n_neurons, n_freqs)``.

Raises
------
ValueError
    If any domain parameter is not positive, if the population is empty, if
    any train is not one-dimensional and finite, or if binned lengths differ.

### Function `directed_transfer_function(trains, bin_size, order, n_freqs)`
Return the directed transfer function for a spike-train population.

Parameters
----------
trains:
    Non-empty list of one-dimensional spike trains. All trains must produce
    the same number of bins.
bin_size:
    Positive number of samples per spike-count bin.
order:
    Positive autoregressive model order.
n_freqs:
    Positive number of frequencies in the closed interval ``&#91;0, 0.5&#93;``.

Returns
-------
np.ndarray
    Normalised DTF tensor with shape ``(n_neurons, n_neurons, n_freqs)``.
    Singular transfer matrices are skipped and leave the corresponding
    frequency slice at zero.

Raises
------
ValueError
    If any domain parameter is not positive, if the population is empty, if
    any train is not one-dimensional and finite, or if binned lengths differ.

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

### Function `spike_train_pca(trains, n_components, bin_size, backend)`
PCA on a binned spike-count matrix (neurons × time bins).

Parameters
----------
trains : list of numpy.ndarray
    Binary spike trains, one per neuron.
n_components : int, optional
    Number of principal components to keep.
bin_size : int, optional
    Number of timesteps per bin.
backend : str, optional
    ``"auto"`` selects the fastest measured path — the NumPy/LAPACK reference,
    which outperforms the compiled dense-eigendecomposition backends;
    ``"python"``, ``"rust"``, ``"julia"``, ``"go"`` and ``"mojo"`` force a
    specific path (the compiled backends are kept for cross-language parity
    and portability).

Returns
-------
tuple of numpy.ndarray
    ``(projected, explained_variance_ratio)``; ``projected`` is
    ``(n_components, n_bins)``. Empty arrays when no trains are supplied.

### Function `demixed_pca(trains_by_condition, n_components, bin_size, backend)`
Demixed PCA (Kobak et al. 2016) on condition-mean activity.

Separates condition-dependent variance by projecting the grand-mean-centred
condition means onto the leading eigenvectors of their covariance.

Parameters
----------
trains_by_condition : dict
    ``{condition_id: &#91;binary trains per neuron&#93;}``.
n_components : int, optional
    Number of components to keep.
bin_size : int, optional
    Number of timesteps per bin.
backend : str, optional
    See :func:`spike_train_pca`.

Returns
-------
tuple of numpy.ndarray
    ``(projected, explained_variance_ratio)``; empty arrays when fewer than
    two conditions carry data.

### Function `factor_analysis(trains, n_factors, bin_size, n_iter, backend)`
Factor analysis via EM (Rubin & Thayer 1982) on binned activity.

The loadings start from a deterministic PCA initialisation (so the result is
reproducible and seed-independent) and each EM step solves its symmetric
positive-definite systems by Cholesky factorisation.

Parameters
----------
trains : list of numpy.ndarray
    Binary spike trains, one per neuron.
n_factors : int, optional
    Number of latent factors.
bin_size : int, optional
    Number of timesteps per bin.
n_iter : int, optional
    Number of EM iterations.
backend : str, optional
    See :func:`spike_train_pca`.

Returns
-------
tuple of numpy.ndarray
    ``(loadings, uniquenesses)`` of shapes ``(n_neurons, n_factors)`` and
    ``(n_neurons,)``.

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

### Function `gpfa_pca_init(Y, n_latents, bin_ms)`
Deterministic PCA initialisation of the GPFA parameters.

The loading matrix ``C`` is the top ``n_latents`` left singular vectors of the
centred data, scaled by their singular values; a fixed sign convention (each
column's largest-magnitude entry made positive) makes the result reproducible
across runs and BLAS/LAPACK implementations. This replaces the former random
initialisation, so every backend can start the EM from an identical ``C``.

Parameters
----------
Y : numpy.ndarray
    Binned spike counts, shape ``(n_neurons, n_bins)``.
n_latents : int
    Number of latent dimensions.
bin_ms : float
    Bin width in milliseconds, used to set the GP timescales ``tau``.

Returns
-------
tuple
    ``(C, d, R, tau)`` — loading matrix ``(n_neurons, n_latents)``, offset
    ``(n_neurons,)``, observation-noise covariance ``(n_neurons, n_neurons)``
    and GP timescales ``(n_latents,)``.

### Function `gpfa_em(Y, C0, d0, R0, tau, max_iter, tol)`
Run the GPFA EM loop from a fixed initialisation (NumPy reference floor).

The GP timescales ``tau`` are held fixed, so the kernel matrices are constant
across iterations. Returns the filtered trajectories together with the final
parameters and the per-iteration marginal log-likelihoods.

Parameters
----------
Y : numpy.ndarray
    Binned spike counts, shape ``(n_neurons, n_bins)``.
C0, d0, R0 : numpy.ndarray
    Initial loading matrix, offset and observation-noise covariance.
tau : numpy.ndarray
    GP timescales, shape ``(n_latents,)``.
max_iter : int
    Maximum EM iterations.
tol : float
    Convergence tolerance on the log-likelihood increment.

Returns
-------
tuple
    ``(trajectories, C, d, R, log_likelihoods)``.

### Function `gpfa(trains, n_latents, bin_ms, dt, max_iter, tol, seed, backend)`
Extract smooth latent trajectories from parallel spike trains via EM.

The initialisation is deterministic (PCA, see :func:`gpfa_pca_init`), so the
result is reproducible and identical across acceleration backends up to
floating-point round-off. ``seed`` is retained for API compatibility but no
longer affects the result.

Parameters
----------
trains : list of numpy.ndarray
    Parallel binary/integer spike trains.
n_latents : int, optional
    Number of latent dimensions (clamped to ``min(n_neurons, n_bins)``).
bin_ms, dt : float, optional
    Bin width (ms) and simulation timestep (s).
max_iter, tol : int, float, optional
    EM iteration cap and log-likelihood convergence tolerance.
seed : int, optional
    Retained for API compatibility; initialisation is deterministic.
backend : str, optional
    ``"auto"`` selects the fastest measured backend — the ``nalgebra``-backed
    Rust path when the engine is present, otherwise the NumPy reference
    (``"python"``); ``"rust"``, ``"julia"``, ``"go"`` and ``"mojo"`` run the
    parity-verified compiled paths on request.

Returns
-------
dict
    Keys ``trajectories``, ``C``, ``d``, ``R``, ``log_likelihoods``, ``tau``.

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
  - Initialise the POYO decoder weights from the seed.
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
  - Initialise the POSSM decoder state-space parameters from the seed.
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
  - Initialise the NDT-3 decoder weights from the seed.
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
  - Initialise the CEBRA encoder weights from the seed.
- **encode**(x)
  - Encode neural data through 2-layer MLP.
- **cosine_similarity**(a, b)
  - Pairwise cosine similarity matrix.
- **infonce_loss**(anchors, positives)
  - InfoNCE contrastive loss. van den Oord et al. (2018).
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

### Function `isolation_distance(cluster, noise, backend)`
Isolation distance (Harris et al. 2001).

The Mahalanobis distance at which the number of noise points reaching the
cluster equals the cluster size — the squared Mahalanobis radius of the
``n_cluster``-th nearest noise point.

Parameters
----------
cluster : numpy.ndarray
    Shape ``(n_cluster, n_features)`` — the cluster's feature vectors.
noise : numpy.ndarray
    Shape ``(n_noise, n_features)`` — competing (noise) feature vectors.
backend : str, optional
    ``"auto"`` selects the fastest available backend (Rust engine when
    present, otherwise the NumPy reference); ``"python"``, ``"rust"``,
    ``"julia"``, ``"go"`` and ``"mojo"`` force a specific path.

Returns
-------
float
    Isolation distance; ``nan`` when ``n_cluster < 2`` or fewer noise points
    than cluster points are supplied.

### Function `l_ratio(cluster, noise, backend)`
L-ratio cluster quality (Schmitzer-Torbert et al. 2005).

The mean over noise points of the chi-squared survival weight
``exp(-½ (d²_Mahalanobis - n_features))`` (clamped to ``&#91;0, 1&#93;``), normalised
by the cluster size — small for well-isolated clusters.

Parameters
----------
cluster : numpy.ndarray
    Shape ``(n_cluster, n_features)`` — the cluster's feature vectors.
noise : numpy.ndarray
    Shape ``(n_noise, n_features)`` — competing (noise) feature vectors.
backend : str, optional
    Forwarded to the polyglot dispatch (see :func:`isolation_distance`).

Returns
-------
float
    L-ratio; ``nan`` when ``n_cluster < 2`` or no noise points are supplied.

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
- **step**(current)
  - Step the unified physical simulation one tick forward.
- **step_from_bio_rates**(rates)
  - Modulate phenomenological bounds leveraging a multi-channel biological firing rate map.
- **evaluate_bio_pathway_resilience**(rates)
  - Run deterministic fault-injection resilience over biological pathways.
- **run_meta_learning_episode**(currents)
  - Run a full outer-loop adaptation episode over a current sequence.
- **export_reasoning_trace**()
  - Export a compact symbolic trace for outer-loop introspection.
- **export_symbolic_reasoning_log**()
  - Export a short symbolic self-verification log for downstream audit.
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

### Function `validate_pdk(pdk)`
Check PDK configuration for obvious errors.

### Function `validate_pdk_installation(pdk, pdk_root, require_signoff)`
Check whether the resolved open-source PDK files are present locally.

---

## Module `audio.adaptive_engine`

### Class `SessionPhase`
Adaptive audio control phase for a closed-loop session.


### Class `AdaptiveSessionReport`
Summary of a completed adaptive audio session.

- **to_dict**()
  - Return a JSON-compatible summary of the adaptive session.

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
- **on_evs_update**(snapshot)
  - Process one EVS update and return adapted audio parameters.
- **get_session_report**()
  - Generate summary report of the current session.
- **current_phase**()
  - Return the active adaptive-control phase.
- **tick**()
  - Return the number of processed EVS updates.
- **reset**()
  - Reset session state (does not reset SSGF or EVS).

---

## Module `audio.evs_engine`

### Class `EVSConfig`
Configuration for FFT-based entrainment scoring.

Attributes
----------
sample_rate:
    EEG sample rate in hertz.
fft_window:
    Number of samples retained in the ring buffer for FFT scoring.
baseline_duration_s:
    Baseline collection duration in seconds.
update_interval_samples:
    Nominal sample interval between external EVS updates.


### Class `EVSSnapshot`
Single-tick entrainment verification observation.

Attributes
----------
evs_score:
    Composite entrainment score in the inclusive range 0 to 100.
relative_increase:
    Target-band power increase relative to baseline.
peak_alignment:
    Alignment between the spectral peak and target frequency.
band_dominance:
    Fraction of total spectral power in the target band.
temporal_consistency:
    Stability score computed from recent EVS values.
is_verified:
    Whether score and confidence clear the verification threshold.
confidence:
    Confidence score derived from the number of scoring updates.
target_hz:
    Target entrainment frequency in hertz.
peak_hz:
    Dominant measured frequency in hertz.
band_powers:
    Per-band FFT power estimates.
timestamp:
    Snapshot creation time from ``time.time()``.

- **to_dict**()
  - Serialise the snapshot into JSON-compatible telemetry.

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
  - Initialise the EVS ring buffer and scoring state.
- **start_baseline**()
  - Begin baseline EEG collection.
- **add_sample**(voltage)
  - Feed one raw EEG voltage sample.
- **set_target**(hz)
  - Set the entrainment target frequency.
- **compute**()
  - Compute current EVS snapshot.
- **baseline_done**()
  - Whether baseline EEG collection has been finalised.
- **score_history**()
  - Return a copy of accumulated EVS scores.
- **reset**()
  - Clear buffers, baseline state, and score history.

---

## Module `audio.ssgf_engine`

### Class `SSGFConfig`
Configuration for the SSGF geometry-coupled oscillator engine.

Attributes
----------
N:
    Number of oscillators in the Kuramoto field.
z_dim:
    Length of the latent geometry vector decoded into the symmetric
    coupling matrix.
lr_z:
    Gradient-descent step size for the latent geometry vector.
sigma_g:
    Scale applied to geometry-derived phase coupling.
micro_steps:
    Number of Kuramoto integration steps per outer geometry update.
dt:
    Integration timestep in seconds.
noise:
    Standard deviation of phase noise injected during each micro-step.
K_base:
    Baseline Kuramoto coupling retained for compatibility with profile
    tuning surfaces.
K_alpha:
    Adaptive coupling multiplier retained for compatibility with profile
    tuning surfaces.
field_pressure:
    Cosine field pressure applied as a global steering term.
seed:
    Deterministic NumPy random seed for reproducible initial conditions.


### Class `SSGFEngine`
Lightweight SSGF geometry-coupled Kuramoto solver.

Maintains a latent vector *z* whose decoded geometry matrix W(t)
feeds back into the micro-cycle, steering oscillators toward
higher global coherence R.  Audio-mapping observables are derived
from the resulting phase dynamics and spectral properties of W.

- **__init__**(cfg)
  - Initialise the SSGF state from a deterministic configuration.
- **outer_step**()
  - Advance one SSGF outer cycle.
- **get_audio_mapping**()
  - Derive CCW audio parameters from current SSGF state.
- **get_state**()
  - Return a JSON-compatible snapshot of the current SSGF state.

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
  - Populate chronotype-derived defaults for omitted profile maps.
- **get_best_target_hz**()
  - Return the best entrainment target for this user.
- **update_from_session**(avg_evs, peak_evs, best_target_hz, band_powers)
  - Update profile after a completed session.
- **to_dict**()
  - Serialise the profile into JSON-compatible primitive values.
- **from_dict**(cls, data)
  - Build a profile from a dictionary produced by :meth:`to_dict`.

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

- **timesteps**(epoch)
  - Sequence length for this epoch.
- **rate_scale**(epoch)
  - Return the firing-rate multiplier for the given epoch.
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

---

## Module `benchmarks.online_o1_adaptation`

### Function `build_online_o1_adaptation_benchmark()`
Return a deterministic Python/Rust adaptation benchmark report.

### Function `write_online_o1_adaptation_benchmark(path)`
Write a canonical benchmark report and return the output path.

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
  - Initialise the scKG-BERT interface weights from the seed.
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
  - Initialise the Geneformer interface weights from the seed.
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
  - Create a configuration preset for a standard MEA layout.

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

### Class `AERToSCConverter`
Converts AER event streams to SC bitstreams.

Uses a time-windowed rate code: count events per neuron per window,
then LFSR-encode the resulting firing probabilities.

- **convert**(events)
  - Convert AER events to per-neuron SC bitstreams.

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
  - Apply the BCM update to a saturated Q8.8 synaptic weight.

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
  - Return a dataclass field through the legacy mapping interface.
- **__contains__**(key)
  - Return whether ``key`` names a public result field.
- **keys**()
  - Return the mapping-view field names in dataclass declaration order.

### Class `BioHybridSession`
Manages a complete bio-hybrid experiment session.

Orchestrates: MEA recording → spike detection → AER transcoding →
SC processing → optogenetic feedback → plasticity update.

- **process_frame**(voltage_data, t_start_s, stim_times_s)
  - Process one MEA data frame through the full pipeline.

### Class `SpikeSorter`
Research spike sorter using PCA feature extraction and K-Means clustering.

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
  - Return the arithmetic mean of recorded loop latencies.
- **p99_latency_us**()
  - Return the 99th percentile closed-loop latency.
- **compliance_ratio**()
  - Return the fraction of samples inside the latency budget.

### Class `PharmModel`
Simulates effect of pharmacological agents on spike rate.

Models excitatory (e.g., bicuculline) or inhibitory (e.g., TTX) agents
as gain factors on firing rate.

- **apply**(t_current_s)
  - Mark the pharmacological agent as applied at the current time.
- **effective_gain**(t_current_s)
  - Return the active firing-rate gain at an experiment timestamp.
- **modulate_spikes**(spike_counts, t_current_s)
  - Modulate spike counts by pharmacological gain.
- **modulate_spike_events**(spikes, t_current_s)
  - Apply pharmacological rate gain to spike events.

### Class `WellConfig`
One well in a multi-well MEA plate.

- **label**()
  - Return the stable plate label for this well.

### Class `MultiWellPlate`
Multi-well plate (e.g., 6/24/48/96-well MEA plate).

- **add_well**(well)
  - Append a well configuration to the plate.
- **standard_6_well**(cls, layout)
  - Construct a six-well plate with uniform MEA layout presets.
- **num_wells**()
  - Return the number of configured wells.
- **get_well**(well_id)
  - Return a well by identifier.

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
  - Append one audit entry to the session log.
- **total_rounds**()
  - Return the number of recorded audit entries.
- **to_list**()
  - Serialise audit entries to deterministic dictionaries.
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
  - Return the nucleotide length of this strand.
- **gc_content**()
  - Return the fraction of nucleotides that are G or C bases.
- **complement**()
  - Return the reverse-complement strand sequence.
- **max_homopolymer_run**()
  - Return the longest repeated-base run in the strand.
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
  - Return the number of strands implementing this gate.

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
  - Return the total strand count across circuit inputs, outputs, fuel, and gates.
- **total_gates**()
  - Return the number of DNA logic gates in the circuit.
- **total_nucleotides**()
  - Return the total nucleotide count across all circuit strands.
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
  - Return whether optional NUPACK thermodynamic analysis is installed.
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

### Function `derive_probabilities_from_adjacency(adjacency, floor, ceiling)`
Derive per-node SC probabilities from inbound absolute weight mass.

### Function `encode_bitstream_bank(probabilities)`
Encode probabilities into deterministic LFSR-backed SC bitstreams.

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
  - Configure a standard linear anneal from s=0 to s=1.
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

### Function `simulate_timing(topology, src_die, dst_die)`
BFS shortest-path timing simulation between two dies.

### Function `make_torus(rows, cols, tech)`
Create a 2D torus topology (mesh with wrap-around edges).

### Function `compute_cdc_configs(topology)`
Auto-compute per-link CDC configs from die clock frequencies.

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

### Function `main()`
Run the command-line interface and return a process exit status.

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

## Module `compiler._sby_runner`

### Class `SbyRun`
The raw outcome of one :func:`run_sby_task` invocation.

Attributes
----------
verdict : str
    The parsed SymbiYosys verdict (``"PASS"``, ``"FAIL"``, ``"ERROR"``, ...).
rc : int
    The return code reported inside the ``DONE (..., rc=N)`` summary line.
returncode : int
    The ``sby`` process exit code (distinct from ``rc``: the process may exit
    non-zero on a ``FAIL`` while ``rc`` classifies the proof outcome).
counterexample : str or None
    The failing-assertion description when the output carries one, else
    ``None``.
trace_path : str or None
    Absolute path to the counterexample trace (resolved against the work
    directory) when the output names one, else ``None``.
summary : list&#91;str&#93;
    The ``sby`` ``summary:`` lines, retained for diagnostics.
stdout : str
    The full captured standard output, retained for error reporting.


### Function `formal_tools_available(engine)`
Return ``True`` when the full proof toolchain is on ``PATH``.

Checks ``sby`` and ``yosys`` plus the SMT solver binary for ``engine`` — for
the ``smtbmc`` backend the engine name is also the solver executable name
(``z3``, ``boolector``, ``yices`` …). A runner may have ``sby`` and ``yosys``
installed yet lack the solver (as on a CI image that ships only the HDL
toolchain), in which case a proof would error out at the engine stage; this
guard reports that case as unavailable so callers and tests can skip cleanly.

Parameters
----------
engine : str
    The ``smtbmc`` SMT engine whose solver binary must also be present.

Returns
-------
bool
    ``True`` only when ``sby``, ``yosys``, and the ``engine`` solver all
    resolve on ``PATH``.

### Function `parse_verdict(stdout)`
Extract the ``(verdict, rc)`` from the ``sby`` summary.

SymbiYosys prints one ``DONE (<verdict>, rc=<n>)`` line per finished task; the
last such line is the authoritative outcome (an earlier ``DONE (ERROR, ...)``
can precede a retried task). Returns ``("UNKNOWN", -1)`` when no ``DONE`` line
is present at all — a truncated or crashed run.

Parameters
----------
stdout : str
    The captured ``sby`` standard output.

Returns
-------
tuple&#91;str, int&#93;
    The verdict token and the ``sby`` return code it reported.

### Function `run_sby_task(workdir, sby_name)`
Run one already-written ``.sby`` task in ``workdir`` and read its verdict.

The ``.sby`` script and every source file it reads must already exist in
``workdir``; this call only invokes ``sby -f <sby_name>`` there, captures the
output, and parses it into an :class:`SbyRun`. Counterexample text and trace
path are extracted opportunistically — they are populated regardless of
verdict when present, and the caller decides which fields matter for its
result type.

Parameters
----------
workdir : Path
    Directory holding the ``.sby`` script and its sources; also the ``sby``
    run tree and the base for resolving a relative counterexample trace path.
sby_name : str
    File name of the ``.sby`` script within ``workdir``.
timeout_s : float
    Wall-clock limit for the ``sby`` process.

Returns
-------
SbyRun
    The parsed run outcome.

Raises
------
RuntimeError
    If the ``sby`` process exceeds ``timeout_s``.

### Function `is_inconclusive(run)`
Return ``True`` for an inconclusive result — proved nothing, disproved nothing.

A k-induction (``mode prove``) run whose base case holds but whose induction
step does not converge reports ``UNKNOWN`` with :data:`_INCONCLUSIVE_RC`: the
property may well be true, but was not proved and no counterexample was found
(the induction step reached the goal from an *unreachable* predecessor). This
is a real, honest outcome distinct from both a disproof and a tool failure.

Parameters
----------
run : SbyRun
    The raw run to inspect.

Returns
-------
bool
    ``True`` only for the ``UNKNOWN`` / inconclusive-return-code signature.

### Function `raise_for_incomplete(run)`
Raise when the run is a tool or setup failure, not a verdict about the design.

A ``PASS`` (proved) or ``FAIL`` (disproved with a counterexample) is a
conclusive outcome, and an inconclusive k-induction result
(:func:`is_inconclusive`) is a real — if unhelpful — outcome; none of these
raise. Any *other* verdict — ``ERROR``, a crash with no ``DONE`` line, a
timeout — is a tool or setup failure that must not be read as a claim about
the design, so callers invoke this before interpreting a result.

Parameters
----------
run : SbyRun
    The raw run to inspect.
what : str
    Short label for the check (``"equivalence proof"``, ``"property proof"``)
    used in the raised message.

Raises
------
RuntimeError
    When the run neither proved, disproved, nor came back inconclusive.

---

## Module `compiler.auto_tune`

### Function `precision_plan_manifest(assignments)`
Build a deterministic manifest for a per-synapse precision plan.

### Function `auto_tune_synapse_precisions(layer_weights)`
Auto-tune per-synapse precision for an explicit percent error target.

---

## Module `compiler.block_floating`

### Class `BlockFloatingMode`
Shared-exponent block floating-point specification.

- **label**()
  - Human-readable label.
- **exponent_bias**()
  - Bias applied to shared exponents.
- **min_exponent**()
  - Minimum unbiased exponent.
- **max_exponent**()
  - Maximum unbiased exponent.
- **mantissa_range**()
  - Largest signed mantissa magnitude.
- **emit_fraction**()
  - Conservative fixed-point fallback fraction for RTL emission.
- **metadata**()
  - Deterministic metadata payload for cross-target telemetry.
- **block_exponent_count**(parameter_count)
  - Return the exact number of shared exponents for a flat parameter payload.
- **block_exponent_layout**(parameter_count)
  - Return the explicit exponent-vector layout for downstream emitters.
- **validate_exponents**(exponents)
  - Validate exponent vector length and code range for a parameter payload.
- **from_string**(cls, fmt)
  - Parse strict canonical format like 'BFP16E3'.
- **from_aliases**(cls, fmt)
  - Parse tolerant aliases such as BFP16_E3, BFP16.3, BFP16-3, and BFP16E3X32.

### Class `BlockExponentLayout`
Concrete shared-exponent layout for flattened block-floating parameters.

- **__post_init__**()
- **last_block_size**()
  - Number of parameters carried by the final exponent block.
- **manifest**()
  - Deterministic block-exponent layout manifest.
- **validate_exponents**(exponents)
  - Validate exponent vector length and encoded range.

---

## Module `compiler.block_floating_quantization`

### Class `CompiledBlockFloatingDense`
Dense operator compiled with shared-exponent block-floating weights.

- **__post_init__**()
- **output_size**()
  - Number of dense output channels.
- **input_size**()
  - Number of dense input channels.
- **reconstructed_weights**()
  - Float reconstruction of the compiled block-floating weight matrix.
- **manifest**()
  - Deterministic deployment metadata for block-floating dense weights.
- **forward_float**(inputs)
  - Return dense outputs from BFP weights and quantised fixed-point inputs.
- **forward_with_overflow**(inputs)
  - Return saturated fixed-point output codes and per-output overflow flags.
- **forward_accumulator_codes**(inputs)
  - Return saturated output codes in the configured fixed-point input format.
- **precision_trap_report**(inputs)
  - Return saturation telemetry suitable for a hardware trap register.
- **precision_envelope_report**(inputs)
  - Return a conservative absolute-output envelope for this workload.

### Function `compile_dense_block_floating(weights, fmt)`
Compile a dense matrix into block-floating weights with Q-format inputs.

### Function `quantize_block_floating(weights, fmt)`
Quantize float weights into shared-exponent block-floating blocks.

### Function `dequantize_block_floating(quantized, exponents, fmt)`
Reconstruct floats from block-floating mantissas and exponents.

---

## Module `compiler.c_expr_emitter`

### Class `CExprEmitter`
Lower a Python expression AST to a C++ (``ap_fixed``) expression string.

Parameters
----------
state_vars : set of str
    ODE state-variable names; emitted verbatim (they are struct members /
    locals in the generated function).
param_map : dict, optional
    Mapping from parameter names to their C++ identifiers.
math_ns : str
    Namespace prefix for transcendental calls (``"hls"`` for Vitis HLS math,
    ``"std"`` for a portable ``<cmath>`` build).
fp_type : str
    Fixed-point type name used to cast numeric literals.

Attributes
----------
free_vars : list of str
    Identifiers referenced by the expression that are not state variables,
    parameters, or the input current — collected in first-seen order for the
    exporter to declare as inputs.

- **__init__**(state_vars, param_map)
- **visit_BinOp**(node)
  - Emit C++ for a binary operation (add, sub, mul, div, pow).
- **visit_UnaryOp**(node)
  - Emit C++ for a unary operation (negate, positive).
- **visit_Name**(node)
  - Resolve a Python name to its C++ identifier, recording free variables.
- **visit_Constant**(node)
  - Emit a numeric constant cast to the fixed-point type.
- **visit_Compare**(node)
  - Emit C++ for comparison operators (>, >=, <, <=).
- **visit_Call**(node)
  - Emit C++ for a supported function call.
- **generic_visit**(node)
  - Raise for any unsupported AST node type.

### Function `emit_c_expr(expr_str, state_vars, param_map)`
Parse an ODE expression and return its C++ form plus its free variables.

Parameters
----------
expr_str : str
    Python-syntax ODE expression.
state_vars : set of str
    ODE state-variable names.
param_map : dict, optional
    Parameter-name to C++-identifier mapping.
math_ns : str
    Namespace prefix for transcendental calls.
fp_type : str
    Fixed-point type used to cast numeric literals.

Returns
-------
tuple of (str, list of str)
    The C++ expression string and the free identifiers it references (in
    first-seen order).

---

## Module `compiler.c_fixed_emitter`

### Function `signed_q(q, value)`
Encode ``value`` as the signed integer its Verilog ``'sd`` literal denotes.

This is the two's-complement bit pattern of ``round(value * 2**fraction)``
truncated to ``data_width`` bits, reinterpreted as a signed integer — exactly
what :meth:`Q88.encode_signed_literal` writes into the RTL, but as a Python
``int`` suitable for a C/Rust literal.

### Function `emit_c_fixed_expr(expr_str, state_map, param_map, q)`
Lower an ODE expression to a bit-exact integer C/Rust expression.

Parameters
----------
expr_str : str
    Python-syntax ODE right-hand-side expression.
state_map : dict
    ODE variable name to its register-read source lvalue.
param_map : dict
    Parameter name to its signed Q-format integer value.
q : Q88
    Fixed-point configuration.
lang : str
    ``"c"`` or ``"rust"``.
input_ref : str
    Source expression reading the input current ``I``.
lut_start : int
    Starting index for LUT table naming, so several expressions of one kernel
    get unique table names.

Returns
-------
tuple
    ``(expr, statements, tables, free_vars, lut_count, input_used)`` — the
    64-bit-integer expression string, the helper statements it depends on, the
    LUT tables it references, the free identifiers it introduced (first-seen
    order), the next free LUT index, and whether the input current was read.

---

## Module `compiler.certification_gen`

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

---

## Module `compiler.cocotb_gen`

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

---

## Module `compiler.compiler_impl`

### Function `compile_adaptive_precision(neuron, module_name, lp_width, lp_frac, hp_width, hp_frac)`
Compile an EquationNeuron to dual-datapath adaptive-precision Verilog.

---

## Module `compiler.constraint_gen`

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

---

## Module `compiler.equivalence_check`

### Class `EquivalenceResult`
Outcome of a machine-checked equivalence proof.

Attributes
----------
proven : bool
    ``True`` when the checker proved output equivalence to ``depth`` cycles.
verdict : str
    The raw SymbiYosys verdict (``"PASS"``, ``"FAIL"``, ``"ERROR"``, ...).
mode : str
    ``"bmc"`` (bounded) or ``"prove"`` (k-induction).
depth : int
    Checked depth in clock cycles.
engine : str
    SMT engine used (e.g. ``"z3"``).
returncode : int
    ``sby`` process exit code.
counterexample : str or None
    Failing-assertion description on a ``FAIL`` verdict, else ``None``.
trace_path : str or None
    Path to the counterexample VCD trace on ``FAIL``, else ``None``.
summary : list&#91;str&#93;
    The ``sby`` summary lines, retained for diagnostics.


### Function `prove_equivalence(dut_verilog, ref_verilog, io_ports)`
Prove ``dut_verilog`` equivalent to ``ref_verilog`` via SymbiYosys.

Parameters
----------
dut_verilog, ref_verilog : str
    Verilog sources defining ``dut_top`` and ``ref_top`` respectively.
io_ports : list&#91;MiterPort&#93;
    Shared interface passed to :func:`build_equivalence_miter`.
dut_top, ref_top : str
    Module names under verification (must differ).
dut_params, ref_params : dict&#91;str, int&#93;, optional
    Per-instance parameter overrides.
depth : int
    BMC / induction depth in clock cycles.
engine : str
    SMT engine (``"z3"`` by default; the ``smtbmc`` backend).
mode : {"bmc", "prove"}
    Bounded model checking or k-induction.
reset_cycles : int
    Leading clocks to hold reset before comparing.
clock, reset_n : str
    Clock and active-low reset port names.
timeout_s : float
    Wall-clock limit for the ``sby`` process.
workdir : str or Path, optional
    Directory for the generated sources and ``sby`` run tree. A temporary
    directory is created and left in place if omitted (caller cleans up).

Returns
-------
EquivalenceResult
    The parsed verdict.

Raises
------
RuntimeError
    If the formal tools are absent, the ``sby`` run errors out (a tool or
    setup failure, distinct from a ``FAIL`` disproof), or times out.

---

## Module `compiler.equivalence_miter`

### Class `MiterPort`
One port of the module interface shared by the DUT and the reference.

Attributes
----------
name : str
    Port identifier.
width : int
    Bit width (``1`` for a scalar port).
signed : bool
    Whether the port is declared ``signed``.
direction : str
    ``"input"`` or ``"output"``.

- **declaration**(suffix)
  - Return a Verilog ``wire`` declaration for this port.

### Function `parse_module_interface(verilog, top)`
Parse the ANSI port interface of a Verilog module.

Parameters
----------
verilog : str
    Verilog source containing the module.
top : str
    Module name whose interface to parse.
params : dict&#91;str, int&#93;, optional
    Values for any parameters used in port width expressions (e.g.
    ``{"DATA_WIDTH": 16}``). Widths that reference an unlisted parameter
    raise :class:`ValueError`.

Returns
-------
list&#91;MiterPort&#93;
    The ports in declaration order.

### Function `build_equivalence_miter(dut_top, ref_top, io_ports)`
Build a sequential-equivalence miter for two interface-compatible modules.

Both ``dut_top`` and ``ref_top`` are instantiated with the same ``io_ports``;
the miter exposes the clock and every non-reset input as free top-level
inputs (the model checker explores all their values), derives an active-low
reset held for ``reset_cycles`` clocks from a counter, and asserts that every
output agrees between the two instances once reset is released.

Parameters
----------
dut_top, ref_top : str
    Module names of the device-under-test and the reference. Must differ so
    both sources can be read into one design.
io_ports : list&#91;MiterPort&#93;
    The shared interface. Must contain ``clock`` and ``reset_n`` inputs and
    at least one output.
miter_name : str
    Name of the generated miter module.
dut_params, ref_params : dict&#91;str, int&#93;, optional
    Parameter overrides applied to the respective instance (the two modules
    may take different parameter sets).
reset_cycles : int
    Number of leading clocks to hold reset asserted before comparing.
clock, reset_n : str
    Port names of the clock and active-low reset.

Returns
-------
str
    The complete miter Verilog module.

---

## Module `compiler.expr_lut_tables`

### Function `const_float(node)`
Constant-fold a literal or simple literal-arithmetic node to a float.

Recognises fractional exponents such as ``1.0 / 3.0`` in ``x ** p`` by
folding compile-time-constant expressions built from literals and ``+``,
``-``, ``*``, ``/`` (and unary minus).

Parameters
----------
node : ast.AST
    Expression node to fold.

Returns
-------
float or None
    The folded value, or ``None`` if the node is not a compile-time
    constant.

### Function `symmetric_sample_points()`
Return the 256 sample points over ``&#91;-16, 16)`` at 0.125 spacing.

The symmetric transcendental LUTs (exp, tanh, sigmoid, sin, cos, cosh,
exprel, cbrt) are tabulated on this grid; the value ``x == 0`` falls at
index 128. Must match the LUT-call defaults (``lut_min=-16``, ``step=0.125``)
in the emitting backends.

Returns
-------
list of float
    The 256 tabulation points.

### Function `exp_lut_entries(data_width, fraction)`
Quantised ``exp`` LUT over the symmetric grid, saturated to the word max.

Parameters
----------
data_width : int
    Fixed-point word width; sets the signed saturation cap.
fraction : int
    Number of fractional bits (the Q-format scale ``1 << fraction``).

Returns
-------
list of int
    256 integer Q-format entries.

### Function `log_lut_entries(fraction)`
Quantised ``log`` LUT (16 entries over ``&#91;0.06, 7.56)`` at 0.5 spacing).

Parameters
----------
fraction : int
    Number of fractional bits.

Returns
-------
list of int
    16 integer Q-format entries.

### Function `sqrt_lut_entries(fraction)`
Quantised ``sqrt`` LUT (16 entries over ``&#91;0, 7.5&#93;`` at 0.5 spacing).

Parameters
----------
fraction : int
    Number of fractional bits.

Returns
-------
list of int
    16 integer Q-format entries.

### Function `tanh_lut_entries(fraction)`
Quantised ``tanh`` LUT over the symmetric grid.

Parameters
----------
fraction : int
    Number of fractional bits.

Returns
-------
list of int
    256 integer Q-format entries.

### Function `cosh_lut_entries(data_width, fraction)`
Quantised ``cosh`` LUT over the symmetric grid, saturated to the word max.

Parameters
----------
data_width : int
    Fixed-point word width; sets the signed saturation cap (cosh grows fast).
fraction : int
    Number of fractional bits.

Returns
-------
list of int
    256 integer Q-format entries.

### Function `cbrt_lut_entries(fraction)`
Quantised cube-root LUT over the symmetric grid (odd, sign-preserving).

Parameters
----------
fraction : int
    Number of fractional bits.

Returns
-------
list of int
    256 integer Q-format entries.

### Function `exprel_lut_entries(data_width, fraction)`
Quantised ``exprel(z) = (exp(z)-1)/z`` LUT, with the removable limit 1 at 0.

Grows like ``exp(z)/z`` for large ``z``, so entries saturate to the word max.

Parameters
----------
data_width : int
    Fixed-point word width; sets the signed saturation cap.
fraction : int
    Number of fractional bits.

Returns
-------
list of int
    256 integer Q-format entries.

### Function `sigmoid_lut_entries(fraction)`
Quantised logistic-sigmoid LUT over the symmetric grid.

Parameters
----------
fraction : int
    Number of fractional bits.

Returns
-------
list of int
    256 integer Q-format entries.

### Function `sin_lut_entries(fraction)`
Quantised ``sin`` LUT over the symmetric grid.

Parameters
----------
fraction : int
    Number of fractional bits.

Returns
-------
list of int
    256 integer Q-format entries.

### Function `cos_lut_entries(fraction)`
Quantised ``cos`` LUT over the symmetric grid.

Parameters
----------
fraction : int
    Number of fractional bits.

Returns
-------
list of int
    256 integer Q-format entries.

---

## Module `compiler.fixed_point_quantization`

### Function `quantize_weights(weights, fmt, rounding, clip)`
Quantize float weights to fixed-point integers.

Parameters
----------
weights : np.ndarray
    Float weight matrix (any shape).
fmt : str | QFormat | QFormatMixed
    Q-format string/object, e.g. ``"Q8.8"`` or ``QFormatMixed()``.
rounding : str
    ``nearest`` (round half to even), ``stochastic``, or ``floor``.
clip : bool
    If True, clip values to the representable range before quantization.

### Function `dequantize_weights(quantized, fmt, scale)`
Convert quantized fixed-point weights back to float.

### Function `dequantize(quantized, fmt, scale)`
Alias matching the mixed-precision public API.

### Function `q_weights_to_sc_probabilities(quantized, fmt)`
Convert fixed-point quantized weights to SC probabilities in &#91;0, 1&#93;.

### Function `quantization_error(weights, fmt, rounding)`
Compute quantization error statistics.

---

## Module `compiler.formal_evidence`

### Function `write_precision_formal_evidence_bundle(output_dir, assignments)`
Write a SymbiYosys evidence bundle for adaptive-precision claims.

Renders the bounded-error monitor RTL, the bound assertion checker, a runnable
``.sby`` script, and a manifest describing the claim. The bundle is
deterministic: identical arguments produce byte-identical artefacts.

Parameters
----------
output_dir : str or Path
    Directory to write the bundle into (created if absent).
assignments : list&#91;SynapsePrecision&#93;
    The precision plan. Must be non-empty.
module_name : str
    Top-level module name for the generated RTL and SVA.
execute : bool
    When ``False`` (default) the bundle is only written and the manifest
    records ``symbiyosys_executed = False`` — no external tools are invoked,
    keeping the call deterministic and CI-safe. When ``True`` and the formal
    toolchain (``sby`` / ``yosys`` / ``z3``) is on ``PATH``, the proof is run
    and the real verdict recorded; when the toolchain is absent, a skip reason
    is recorded instead of a fabricated pass.
unbounded : bool
    Selects the proof method for both the emitted ``.sby`` and the executed
    proof. ``False`` (default) uses bounded model checking to ``length + 2``
    cycles — complete because the accumulator saturates. ``True`` uses
    k-induction (``mode prove``), an *unbounded* proof whose depth does not
    scale with the bitstream length (the monitor carries the 1-inductive
    strengthening lemma the induction needs). k-induction can come back
    inconclusive, which is recorded honestly rather than as a pass.

Returns
-------
dict&#91;str, Any&#93;
    The manifest that was written.

Raises
------
ValueError
    If ``assignments`` is empty.

---

## Module `compiler.formal_property_check`

### Class `PropertyProofResult`
Outcome of a machine-checked RTL property proof.

Attributes
----------
proven : bool
    ``True`` when the checker proved every bound assertion holds to ``depth``
    cycles under the SVA environment assumptions.
verdict : str
    The raw SymbiYosys verdict (``"PASS"``, ``"FAIL"``, ``"ERROR"``, ...).
mode : str
    ``"bmc"`` (bounded) or ``"prove"`` (k-induction).
depth : int
    Checked depth in clock cycles.
engine : str
    SMT engine used (e.g. ``"z3"``).
returncode : int
    ``sby`` process exit code.
counterexample : str or None
    Failing-assertion description on a ``FAIL`` verdict, else ``None``.
trace_path : str or None
    Path to the counterexample VCD trace on ``FAIL``, else ``None``.
summary : list&#91;str&#93;
    The ``sby`` summary lines, retained for diagnostics.


### Function `prove_property(rtl_verilog, sva_verilog)`
Prove ``rtl_verilog`` satisfies the assertions in ``sva_verilog`` via SymbiYosys.

Parameters
----------
rtl_verilog : str
    Synthesisable Verilog source defining ``top``.
sva_verilog : str
    SystemVerilog source carrying the environment ``assume`` constraints and
    the ``assert`` obligations, bound onto ``top`` with a ``bind`` statement.
top : str
    The RTL module name under verification.
mode : {"bmc", "prove"}
    Bounded model checking (default, complete for state-stationary designs) or
    k-induction.
depth : int
    BMC / induction depth in clock cycles.
engine : str
    SMT engine (``"z3"`` by default; the ``smtbmc`` backend).
timeout_s : float
    Wall-clock limit for the ``sby`` process.
workdir : str or Path, optional
    Directory for the generated sources and ``sby`` run tree. A temporary
    directory is created and left in place if omitted (caller cleans up).

Returns
-------
PropertyProofResult
    The parsed verdict.

Raises
------
RuntimeError
    If the formal tools are absent, the ``sby`` run errors out (a tool or
    setup failure, distinct from a ``FAIL`` disproof), or times out.

---

## Module `compiler.fpga_wrapper`

### Function `equation_to_fpga()`
One-liner: ODE string → (Python neuron, Verilog RTL).

---

## Module `compiler.guard_bits`

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

---

## Module `compiler.host_driver_gen`

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
live_update_spec : MMIOUpdateSpec, optional
    Live-control bank contract. When provided, generated drivers include
    CRC-guarded live-parameter update, trap check, and committed readback
    helpers for each named bank entry.

Returns
-------
str
    Complete driver source code.

---

## Module `compiler.intelligence.adiabatic_clocks`

### Class `AdiabaticPhase`
Adiabatic clock phase timing (ps).

Attributes
----------
name : str
rise_ps : float
hold_ps : float
fall_ps : float
sleep_ps : float


### Function `generate_adiabatic_clocks(phases, freq_mhz)`
Generate multi-phase clocking for adiabatic computing.

---

## Module `compiler.intelligence.aging_reliability`

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


### Function `predict_reliability()`
Predict MTTF from voltage, temperature, and technology node.

### Function `predict_aging(initial_fmax_mhz)`
Predict end-of-life Fmax after transistor aging.

---

## Module `compiler.intelligence.approximate_computing`

### Class `ApproximationConfig`
Approximate computing configuration.

Attributes
----------
populations : dict&#91;str, dict&#93;
total_energy_savings_pct : float
max_output_error_pct : float


### Function `configure_approximation(equations)`
Configure precision-energy tradeoff knobs per state variable.

---

## Module `compiler.intelligence.auto_quantization`

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

---

## Module `compiler.intelligence.bit_true_kernel`

### Function `generate_bittrue_kernel(module_name, equations)`
Generate a bit-true fixed-point kernel from a ``{var: derivative}`` mapping.

Each state variable advances by one unit-``dt`` explicit-Euler step using the
same wrap-truncate multiply and saturating accumulate as the RTL datapath.
Identifiers in the derivatives that are not state variables become step
arguments (the input current ``I`` maps to ``I_t`` when referenced), so the
kernel exercises the bit-true primitives without a per-instance I/O contract.
For a whole-neuron kernel proven bit-identical to the generated Verilog, use
:func:`generate_bittrue_kernel_from_neuron`.

Parameters
----------
module_name : str
    Base name for the generated struct / functions.
equations : dict
    Mapping from state-variable name to its derivative expression string.
data_width : int
    Fixed-point total width (default 16 → Q8.8).
fraction : int
    Fractional bits (default 8).
language : str
    ``"c"`` (default) or ``"rust"``.

Returns
-------
str
    Bit-true kernel source code.

### Function `generate_bittrue_kernel_from_neuron(neuron, module_name)`
Generate a whole-neuron kernel bit-identical to the compiled Verilog.

Mirrors :func:`sc_neurocore.compiler.verilog_compiler.compile_to_verilog`
exactly for explicit-Euler and discrete-map neurons — parameter/constant
Q-encoding, wrap-truncate arithmetic, the same overflow handling, and the
threshold / reset / spike sequencing of the RTL ``always`` block. Threshold
and reset expressions may read ``<state>_prev`` aliases for the pre-step
register while ordinary state names resolve to the integrated candidate.
Both state and output fields take the same post-reset value on a spike. The
resulting ``<module>_step`` therefore produces the identical per-cycle state
trace as the RTL, which the iverilog co-simulation proves.

Parameters
----------
neuron : EquationNeuron
    The neuron whose ODEs, parameters, threshold and reset rules are lowered.
module_name : str
    Base name for the generated struct / functions.
data_width, fraction : int
    Fixed-point format (default Q8.8).
signed : bool
    Signed two's complement (only ``True`` is supported; unsigned neuron state
    is not part of the RTL contract).
overflow : str
    ``"saturate"`` (default) or ``"wrap"``.
rounding : str
    ``"truncate"`` (default) or ``"nearest"``.
language : str
    ``"c"`` (default) or ``"rust"``.

Returns
-------
str
    Bit-true whole-neuron kernel source code.

---

## Module `compiler.intelligence.bitstream_encryption`

### Function `generate_bitstream_encryption(module_name)`
Generate bitstream encryption TCL/constraints for secure boot.

---

## Module `compiler.intelligence.bitstream_flow`

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

---

## Module `compiler.intelligence.bram_array`

### Function `generate_bram_array(module_name)`
Generate a time-multiplexed BRAM-backed neuron array in Verilog.

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
    State variables per neuron.

Returns
-------
str
    Verilog module source code.

---

## Module `compiler.intelligence.carbon_footprint`

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


### Function `estimate_carbon_footprint(profile_name)`
Estimate carbon footprint for a compilation target.

---

## Module `compiler.intelligence.cdc_analyzer`

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


### Function `analyze_cdc(equations)`
Analyze clock domain crossings in a neuron array.

---

## Module `compiler.intelligence.cdc_synchroniser`

### Function `generate_cdc_synchroniser(signal_name)`
Generate a CDC (Clock Domain Crossing) synchroniser in Verilog.

Uses a multi-stage register chain to safely transfer signals between
clock domains.

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

---

## Module `compiler.intelligence.checksum`

### Function `embed_model_checksum(verilog)`
Embed a SHA-256 checksum of the compiled model in the Verilog source.

---

## Module `compiler.intelligence.cognitive_bounds`

### Class `CognitiveBounds`
Clamping results for cognitive stability.

Attributes
----------
safe_equations : dict&#91;str, str&#93;
lyapunov_divergence_proxy : float
switches_inserted : int


### Function `enforce_cognitive_bounds(equations, state_bounds)`
Enforce safe operating ranges on cognitive models via RTL clamping.

---

## Module `compiler.intelligence.compilation_cache`

### Class `CompilationCache`
Memoized compilation result cache.

Keyed by ``(equations_hash, target, data_width, fraction)``.
Avoids redundant recompilation when re-targeting.

- **__init__**()
- **get**(equations, target, data_width, fraction)
  - Look up a cached compilation result.
- **put**(equations, target, data_width, fraction, result)
  - Store a compilation result in cache.
- **size**()
  - Number of cached entries.

---

## Module `compiler.intelligence.compilation_report`

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

---

## Module `compiler.intelligence.compilation_summary`

### Function `generate_compilation_summary(module_name, equations, target)`
Generate a comprehensive human-readable compilation summary.

Produces a markdown document summarising all aspects of a compilation.

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
    Lines of generated Verilog.

Returns
-------
str
    Markdown compilation summary.

---

## Module `compiler.intelligence.compliance_matrix`

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

---

## Module `compiler.intelligence.cxl_coherence`

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


### Function `advise_cxl_mapping(neuron_count, synapse_count)`
Advise on CXL.mem Type-3 device mapping for neuron state.

Plans the distribution of neuron state and synaptic weights
across CXL 3.0 Type-3 memory expander devices.

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

---

## Module `compiler.intelligence.debug_probes`

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

## Module `compiler.intelligence.dispatch_planner`

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

---

## Module `compiler.intelligence.drift_compensation`

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


### Function `generate_drift_compensator(module_name)`
Generate analog drift compensation controller.

---

## Module `compiler.intelligence.dvfs_controller`

### Function `generate_dvfs_controller(module_name)`
Generate a Verilog DVFS controller FSM.

---

## Module `compiler.intelligence.dvs_bridge`

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

---

## Module `compiler.intelligence.energy_harvesting`

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


### Class `EnergyHarvestBudget`
Energy harvesting feasibility analysis.

Attributes
----------
harvester_power_uw : float
design_power_uw : float
energy_positive : bool
recommended_duty_cycle : float
margin_pct : float


### Function `generate_energy_schedule(neuron_count)`
Generate energy-budget-aware neuron update schedule.

### Function `model_energy_harvest(design_power_uw)`
Model whether an energy harvester can sustain the neural workload.

---

## Module `compiler.intelligence.equivalence_sketch`

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

---

## Module `compiler.intelligence.fault_injection`

### Class `FaultCampaignResult`
Fault injection campaign result.

Attributes
----------
total_injections : int
sdc_count : int
sdc_rate : float
critical_bits : list&#91;int&#93;
recommended_tmr_bits : list&#91;int&#93;


### Function `run_fault_campaign(equations, data_width)`
Run a fault injection campaign on the state register.

---

## Module `compiler.intelligence.fault_tree`

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


### Function `generate_fault_tree(module_name, equations)`
Generate FTA/FMEA for DO-254 Level A certification.

---

## Module `compiler.intelligence.hil_calibration`

### Class `HILCalibration`
Hardware-in-the-loop calibration protocol.

Attributes
----------
protocol_steps : list&#91;str&#93;
num_parameters : int
sweep_ranges : dict&#91;str, tuple&#91;float, float&#93;&#93;


### Function `generate_hil_calibration(module_name, equations)`
Generate hardware-in-the-loop calibration protocol.

---

## Module `compiler.intelligence.hls_export`

### Function `generate_hls_cpp(module_name, equations)`
Translate compiled neuron equations to Vitis/Catapult HLS C++.

Generates a synthesisable ``ap_fixed`` C++ function with ``#pragma HLS``
directives for Xilinx Vitis HLS or Siemens Catapult. Each equation is a
derivative that is Euler-integrated (``<var>_next = <var> + dt * d<var>``);
the first state variable is treated as the membrane potential and reset by
subtracting the threshold when it spikes. Free identifiers in the equations
become function inputs so the generated unit is self-contained.

Parameters
----------
module_name : str
    Function/module name.
equations : dict
    Mapping ``state_var -> derivative expression`` (Python syntax).
data_width : int
    Fixed-point total width.
fraction : int
    Fractional bits.
hls_tool : str
    ``"vitis"`` or ``"catapult"``.
dt : float
    Euler integration timestep.
threshold : float
    Membrane spike threshold.

Returns
-------
str
    Complete HLS C++ source file.

---

## Module `compiler.intelligence.holographic_interconnect`

### Class `HolographicRouter`
Holographic interconnect router specification.

Attributes
----------
slm_grid_size : tuple&#91;int, int&#93;
diffraction_limit_nm : float
optical_fanout_per_beam : int
phase_array_complexity : float


### Function `route_holographic_interconnects(num_neurons, connections)`
Route 3D optical holographic interconnects using SLM phase arrays.

---

## Module `compiler.intelligence.ip_obfuscation`

### Class `ObfuscationResult`
IP obfuscation report.

Attributes
----------
techniques_applied : list&#91;str&#93;
key_bits : int
original_signals : int
obfuscated_signals : int


### Function `obfuscate_ip(module_name, equations)`
Apply logic locking and structural obfuscation for IP protection.

---

## Module `compiler.intelligence.learning_export`

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

---

## Module `compiler.intelligence.license_compliance`

### Class `LicenseCheck`
IP core license compatibility result.

Attributes
----------
compatible : bool
conflicts : list&#91;str&#93;
licenses_found : list&#91;str&#93;


### Function `check_license_compliance(project_license, dependencies)`
Verify IP core licensing compatibility.

---

## Module `compiler.intelligence.memory_map`

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


### Function `generate_memory_map(module_name, equations)`
Generate address decoder for multi-neuron SoC arrays.

Parameters
----------
module_name : str
    Module name.
equations : dict&#91;str, str&#93;
    ODE equations.
num_neurons : int
    Number of neuron instances.
data_width : int
    Register width in bits.
base_address : int
    Base address.

Returns
-------
MemoryMap

---

## Module `compiler.intelligence.model_complexity`

### Class `ModelComplexity`
Model compute-profile classification.

Attributes
----------
classification : str
    ``"compute_bound"``, ``"memory_bound"``, or ``"comm_bound"``.
compute_ops : int
    Total arithmetic operations.
memory_vars : int
    State variables.
comm_ratio : float
    Inter-variable coupling ratio.
recommended_paradigm : str
    Best platform class.


### Function `classify_model_complexity(equations)`
Classify a model's compute profile.

---

## Module `compiler.intelligence.morphology_synth`

### Class `Morphology`
Interconnect topology morphology.

Attributes
----------
topology : str
    ``"Hypercube"``, ``"3D Torus"``, or ``"2D Mesh"``.
bisection_bandwidth_gbps : float
routing_latency_ns : float
dimensions : int


### Function `synthesize_morphology(equations, max_generations)`
Auto-synthesizer for optimal interconnect topologies.

---

## Module `compiler.intelligence.multi_die_floorplan`

### Class `FloorplanResult`
Multi-die/chiplet floorplan assignment.

Attributes
----------
die_assignment : dict&#91;str, int&#93;
    Block name → die index.
die_utilization : dict&#91;int, float&#93;
    Die index → utilization (0-1).
total_dies : int


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

---

## Module `compiler.intelligence.mxfp_encoding`

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

---

## Module `compiler.intelligence.network_optimizer`

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

---

## Module `compiler.intelligence.nir_import`

### Class `NIRGraph`
Imported dict-form NIR graph representation.

Attributes
----------
nodes : dict&#91;str, dict&#93;
    Node name → its raw input parameters.
edges : list&#91;tuple&#91;str, str&#93;&#93;
    Directed edges ``(source, target)``.
equations : dict&#91;str, str&#93;
    Node name → the membrane (``dv/dt``) right-hand side, with parameters
    substituted to concrete values. Kept as a flat ``str`` per node for
    back-compatibility and stability analysis.
framework : str
    Source framework label.
node_types : dict&#91;str, str&#93;
    Node name → the canonical template type it resolved to.
state_equations : dict&#91;str, dict&#91;str, str&#93;&#93;
    Node name → ``{state_variable: right-hand side}`` for the full (possibly
    multi-compartment) model, so nothing is lost for multi-state neurons.
thresholds : dict&#91;str, str | None&#93;
    Node name → its spike threshold expression (``None`` if the type has no
    threshold, e.g. leaky/plain integrators).
resets : dict&#91;str, str | None&#93;
    Node name → its reset rule expression (``None`` if the type has none).
parameters : dict&#91;str, dict&#91;str, float&#93;&#93;
    Node name → the resolved numeric parameters (template defaults overlaid
    with the node's own values).


### Function `import_nir_graph(nir_data)`
Import a dict-form Neuromorphic Intermediate Representation graph.

Each node's ``type`` selects a canonical ODE template (shared with the FPGA
back-end); the node's parameters overlay the template defaults and are
substituted into concrete equations, thresholds and reset rules. Node types
outside the recognised set fall back to a leaky integrator.

Parameters
----------
nir_data : dict
    Graph as ``{"nodes": {name: {"type": ..., <params>}}, "edges": &#91;...&#93;}``.
    A node without a ``type`` defaults to ``LIF``.
framework : str
    Source framework label recorded on the result.

Returns
-------
NIRGraph
    Imported graph with per-node equations, state equations, thresholds,
    reset rules and resolved parameters. For the authoritative typed import
    use :func:`sc_neurocore.nir_bridge.from_nir`.

---

## Module `compiler.intelligence.ode_stability`

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


### Function `verify_ode_stability(equations)`
Verify numerical stability of discretized ODE system.

Uses eigenvalue analysis of the linearized system.

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

---

## Module `compiler.intelligence.omni_paradigm`

### Class `OmniDispatchMap`
Mapping of variables to heterogeneous computing backends.

Attributes
----------
cmos_variables : list&#91;str&#93;
    Standard digital/SRAM variables.
thermodynamic_variables : list&#91;str&#93;
    Stochastic/noise-driven variables (RRAM, PCM).
optical_variables : list&#91;str&#93;
    High-bandwidth weight/sum variables (Photonic).
quantum_variables : list&#91;str&#93;
    Entangled/superposition variables (Superconducting).


### Function `dispatch_omni_paradigm(equations)`
Dispatch ODE variables across heterogeneous computing paradigms.

---

## Module `compiler.intelligence.optical_encoding`

### Class `MZIWeightEncoding`
Encoded weights for a Mach-Zehnder interferometer photonic array.

Attributes
----------
phases_theta : list&#91;list&#91;float&#93;&#93;
    Phase-shift θ values (radians).
phases_phi : list&#91;list&#91;float&#93;&#93;
    Phase-shift φ values (radians).
transmission : list&#91;list&#91;float&#93;&#93;
    Effective transmission coefficients.
mesh_size : int
    Number of MZI columns.


### Function `encode_mzi_weights(weights)`
Encode a weight matrix as MZI phase-shift parameters.

### Function `generate_mzi_config(encoding)`
Generate a photonic chip configuration file from MZI weights.

---

## Module `compiler.intelligence.pareto_explorer`

### Class `ParetoPoint`
A single Pareto-optimal design point.

Attributes
----------
config : dict
power_mw : float
area_luts : int
latency_ns : float


### Function `explore_pareto(equations)`
Explore power/area/latency Pareto frontier.

---

## Module `compiler.intelligence.pim_layout`

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


### Function `plan_pim_layout(neuron_count, synapse_count)`
Plan data placement across PIM memory banks.

Distributes neuron state and synaptic weights across memory banks.

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

---

## Module `compiler.intelligence.pipeline_wrapper`

### Function `generate_pipeline_wrapper(module_name, equations)`
Generate a pipelined wrapper that inserts register stages.

Auto-computes the critical path depth and required pipeline stages.

Parameters
----------
module_name : str
    Inner neuron module name.
equations : dict&#91;str, str&#93;
    ODE equations.
data_width : int
    Data width.
target : str
    Target platform name.
stages : int, optional
    Override pipeline stages.

Returns
-------
str
    Synthesisable Verilog pipeline wrapper.

---

## Module `compiler.intelligence.portability_scorer`

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


### Function `score_portability(equations)`
Score how portable a model is across all profiles.

---

## Module `compiler.intelligence.posit_arithmetic`

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

---

## Module `compiler.intelligence.power_domain_wrapper`

### Function `generate_power_domain_wrapper(module_name)`
Generate a clock/power gating wrapper for always-on edge deployment.

Creates a wrapper module with ICG and power-down state retention.

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

---

## Module `compiler.intelligence.power_intent`

### Function `generate_power_intent(module_name)`
Generate IEEE 1801 UPF power intent for neuron arrays.

---

## Module `compiler.intelligence.power_state_machine`

### Function `generate_power_state_machine(module_name)`
Generate sleep/wake/hibernate FSM for ultra-low-power.

---

## Module `compiler.intelligence.pqc_protection`

### Class `PQCProtection`
Post-quantum cryptographic IP protection result.

Attributes
----------
algorithm : str
signature_hex : str
key_size_bits : int
quantum_safe : bool


### Function `protect_ip_pqc(module_name, equations)`
Apply post-quantum cryptographic protection to IP core.

---

## Module `compiler.intelligence.provenance_chain`

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


### Function `generate_provenance_chain(module_name, equations, verilog_source)`
Generate a cryptographic provenance chain for compilation.

### Function `format_provenance_json(chain)`
Format provenance chain as JSON manifest.

---

## Module `compiler.intelligence.reconfig_planner`

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

---

## Module `compiler.intelligence.regression_watchdog`

### Class `RegressionCheck`
Compilation regression check result.

Attributes
----------
metric : str
baseline : float
current : float
delta_pct : float
regression : bool


### Function `check_regression(baseline, current)`
Detect performance regressions between compilations.

---

## Module `compiler.intelligence.reversible_logic`

### Class `ReversibleNetlist`
Reversible logic netlist metrics.

Attributes
----------
toffoli_gates : int
fredkin_gates : int
ancilla_bits : int
landauer_dissipation_kt : float


### Function `synthesize_reversible_logic(equations, bits)`
Synthesize reversible (lossless) logic for zero-energy computation.

---

## Module `compiler.intelligence.sbom_gen`

### Class `SBOM`
Software/Hardware Bill of Materials.

Attributes
----------
format : str
components : list&#91;dict&#93;
total_components : int


### Function `generate_sbom(module_name, profile_name)`
Generate SBOM/HBOM for IP core compliance.

---

## Module `compiler.intelligence.seu_scrub_scheduler`

### Class `ScrubSchedule`
Configuration memory scrubbing schedule.

Attributes
----------
interval_ms : float
strategy : str
frames_per_cycle : int
expected_seu_rate : float


### Function `schedule_seu_scrubbing(config_bits)`
Generate scrubbing schedule for space-grade configuration memory.

---

## Module `compiler.intelligence.shadow_twin`

### Function `generate_digital_twin(module_name, equations, profile_name)`
Generate a Python digital twin that mirrors deployed hardware.

---

## Module `compiler.intelligence.side_channel_lint`

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


### Function `lint_side_channels(equations)`
Analyse equations for power/timing side-channel vulnerabilities.

---

## Module `compiler.intelligence.storage_recommendation`

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


### Function `storage_recommendation(neuron_count, state_bits_per_neuron)`
Determine optimal storage strategy for a neuron array.

Decides between registers (small), BRAM (medium), and URAM (large)
based on total state bits and target capabilities.

Parameters
----------
neuron_count : int
    Number of neurons in the array.
state_bits_per_neuron : int
    State bits per neuron.
has_uram : bool
    True if the target has UltraRAM.
register_threshold : int
    Max neurons for register-based storage.
uram_threshold : int
    Min neurons for URAM migration.

Returns
-------
StorageRecommendation
    Optimal storage strategy with resource estimates.

---

## Module `compiler.intelligence.supply_chain_risk`

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


### Function `score_supply_chain_risk(profile_name)`
Assess supply chain risk for a hardware profile.

---

## Module `compiler.intelligence.target_comparison`

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


### Function `compare_targets(equations, targets)`
Compare compilation results across multiple hardware targets.

### Function `format_comparison_report(results)`
Format a multi-target comparison as a markdown table.

---

## Module `compiler.intelligence.target_recommender`

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

---

## Module `compiler.intelligence.tcl_gen`

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

---

## Module `compiler.intelligence.telemetry_ingestion`

### Class `TelemetryResult`
Hardware telemetry comparison result.

Attributes
----------
samples : int
max_drift : float
mean_drift : float
alerts : list&#91;str&#93;
healthy : bool


### Function `ingest_telemetry(telemetry_data, twin_states)`
Ingest hardware telemetry and compare against digital twin.

---

## Module `compiler.intelligence.testbench_gen`

### Function `generate_testbench(module_name, equations)`
Generate verification testbench for compiled neuron.

---

## Module `compiler.intelligence.thermal_analysis`

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


### Function `thermal_analysis(estimated_power_mw, target_freq_mhz)`
Estimate thermal impact and frequency derating.

### Function `estimate_thermal_envelope()`
Predict junction temperature from power dissipation.

### Function `generate_thermal_constraints(module_name, analysis)`
Generate XDC constraints for thermal-aware DSP placement.

---

## Module `compiler.intelligence.timescale_partitioner`

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

---

## Module `compiler.intelligence.timing_closure`

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


### Function `verify_timing_closure(equations)`
Perform static timing analysis on the dataflow graph.

---

## Module `compiler.intelligence.tmr_wrapper`

### Function `generate_tmr_wrapper(module_name)`
Generate a Triple Modular Redundancy wrapper for any neuron module.

Instantiates three copies of the target module and a majority voter
to mask Single Event Upsets (SEUs).

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

---

## Module `compiler.intelligence.trojan_lint`

### Class `TrojanLintResult`
Hardware trojan lint analysis result.

Attributes
----------
suspicious_paths : list&#91;str&#93;
risk_level : str
total_checks : int


### Function `lint_hardware_trojans(equations)`
Detect suspicious combinational paths that could hide trojans.

---

## Module `compiler.intelligence.ucie_partitioning`

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


### Function `advise_ucie_partition(neuron_count, connectivity)`
Advise on neuron array partitioning across chiplet tiles.

Analyses a neuron array's connectivity to estimate inter-tile
spike traffic and UCIe bandwidth requirements.

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

---

## Module `compiler.intelligence.ucie_protocol_mapper`

### Class `UCIeMapping`
UCIe die-to-die protocol mapping result.

Attributes
----------
lanes : dict&#91;str, int&#93;
protocol_version : str
total_bandwidth_gbps : float


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

## Module `compiler.intelligence.vhdl_emitter`

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

---

## Module `compiler.intelligence.watermark`

### Class `WatermarkResult`
Netlist watermark embedding result.

Attributes
----------
watermark_hash : str
embedding_method : str
overhead_percent : float
verifiable : bool


### Function `embed_watermark(module_name, equations)`
Embed a verifiable watermark into the compiled netlist.

---

## Module `compiler.intelligence.weight_noise`

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

---

## Module `compiler.intelligence.weight_rom`

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

---

## Module `compiler.intelligence.wetware_mea`

### Class `MEAMapping`
Mapping of neuron populations to MEA electrodes.

Attributes
----------
electrode_count : int
stimulation_freq_hz : float
voltage_amplitude_mv : float
spatial_density : str


### Function `map_wetware_mea(populations, connectivity)`
Map neuron populations to wetware Multi-Electrode Array (MEA) sites.

---

## Module `compiler.ir_type_checker`

### Class `SignalType`
Signal domains accepted by the stochastic IR compatibility checker.


### Class `IRNode`
A typed node in the Stochastic IR graph.


### Class `IREdge`
Connection record from one typed IR node port to another.


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

## Module `compiler.layer_precision`

### Class `LayerPrecision`
Bitstream length assignment for one layer.

Parameters
----------
layer_index:
    Zero-based layer index in the adaptive-precision plan.
name:
    Non-empty layer name used in reports and manifests.
bitstream_length:
    Positive stochastic-computing bitstream length. Layer-level planners
    round this value to a power of two.
error_bound:
    Finite non-negative per-layer stochastic error bound.
sensitivity:
    Finite non-negative sensitivity score used for budget allocation.

- **__post_init__**()
  - Validate adaptive-precision row invariants.
- **to_dict**()
  - Return a JSON-serializable adaptive-precision manifest row.

---

## Module `compiler.length_planner`

### Function `assign_lengths(layer_weights, layer_names, total_budget, min_length, max_length, target_error, method)`
Assign per-layer bitstream lengths under a target error budget.

Parameters
----------
layer_weights:
    One- or two-dimensional finite weight tensors, one tensor per layer.
    One-dimensional tensors are treated as single-output layers.
layer_names:
    Optional non-empty layer names. When provided, the list length must match
    `layer_weights` exactly.
total_budget:
    Optional aggregate bitstream-length budget for sensitivity planning.
    When omitted, sensitivity planning uses `max_length * n_layers`.
min_length:
    Minimum bitstream length assigned to any layer.
max_length:
    Maximum bitstream length assigned to any layer.
target_error:
    Positive per-layer target error used by the Hoeffding planner.
method:
    Planning method. `hoeffding` uses analytic Hoeffding lengths;
    `sensitivity` and `proportional` allocate from sensitivity scores.

Returns
-------
list&#91;LayerPrecision&#93;
    Validated layer precision rows in input-layer order.

Raises
------
ValueError
    If planner bounds, method, names, or weight tensors are invalid.

---

## Module `compiler.live_control_ops`

### Class `MMIOWrite`
One deterministic memory-mapped write in a live-update transaction.

- **__post_init__**()

### Class `MMIORead`
One deterministic memory-mapped read in a live-control transaction.

- **__post_init__**()

---

## Module `compiler.live_control_specs`

### Class `TrapSpec`
Contract for overflow and saturation trap signalling.

- **__post_init__**()

### Class `ParameterBankSpec`
Immutable contract describing one writable bank mapped to MMIO.

- **__post_init__**()
- **entry_width_bits**()
  - Return storage width in bits for one bank entry.
- **entry_width_bytes**()
  - Return storage width in bytes for one bank entry.
- **span_bytes**()
  - Return total byte span from start to end for the bank.
- **end_address_bytes**()
  - Return first invalid byte address beyond the parameter bank.
- **encoded_word_max**()
  - Largest unsigned storage word accepted for one entry.
- **signed_code_min**()
  - Smallest signed two's-complement code accepted for convenience.
- **signed_code_max**()
  - Largest signed two's-complement code accepted for convenience.
- **normalise_encoded_word**(value)
  - Return an unsigned storage word after validating encoded range.
- **entry_index**(parameter)
  - Resolve a parameter name or numeric entry into a bank index.
- **entry_address**(parameter)
  - Return byte address for one parameter entry in this bank.
- **to_dict**()
  - Serialise to JSON-compatible mapping.
- **from_dict**(cls, payload)
  - Rehydrate immutable bank spec from serialized mapping.

### Class `MMIOUpdateSpec`
Contract for dynamic updates through bus-mapped control registers.

- **__post_init__**()
- **has_traps**()
  - Whether the contract requires overflow/saturation signalling.
- **total_address_space_bytes**()
  - Total MMIO span from min bank start to max bank end.
- **control_register_addresses**()
  - Return absolute addresses for the fixed live-control register map.
- **status_bits**()
  - Return host-visible status-bit assignments.
- **control_bits**()
  - Return host-writeable control-bit assignments.
- **trap_bits**()
  - Return deterministic trap-bit assignments for generated parameter banks.
- **effective_trap_width**()
  - Return trap-vector width needed by host-visible generated traps.
- **trap_clear_mask**()
  - Return the mask that clears all host-visible generated trap bits.
- **update_checksum**(bank_name, parameter, encoded_value)
  - Return deterministic IEEE CRC32 guard for one staged update.
- **bank_index**(bank_name)
  - Return deterministic bank-select index for one bank name.
- **bank_by_name**(bank_name)
  - Return a bank by name or fail closed.
- **build_update_sequence**(bank_name, parameter, encoded_value)
  - Build an atomic staged MMIO update sequence.
- **build_apply_sequence**()
  - Build the host-side write sequence that applies a loaded shadow word.
- **build_rollback_sequence**()
  - Build the host-side write sequence that restores shadow from active state.
- **build_selective_trap_clear_sequence**(trap_mask)
  - Build the host-side sequence for clearing selected sticky traps.
- **build_trap_clear_sequence**()
  - Build the host-side sequence for clearing all generated sticky traps.
- **build_readback_sequence**(bank_name, parameter)
  - Build the host-side select/readback sequence for one active entry.
- **to_dict**()
  - Serialise contract for manifest persistence.
- **from_dict**(cls, payload)
  - Rehydrate contract from serialized mapping.

---

## Module `compiler.mixed_dense_kernel`

### Class `MixedDenseBatchResult`
Per-element results of a batched mixed-precision dense contraction.

Each array has shape ``(n_batch, n_outputs)``.

Attributes
----------
outputs_q1616 : numpy.ndarray
    Saturated Q16.16 accumulator codes, ``int32``.
overflow : numpy.ndarray
    ``True`` where the accumulator left the Q16.16 range, ``bool_``.
underflow : numpy.ndarray
    ``True`` where a non-zero contraction rounded to zero without
    overflowing, ``bool_``.


### Function `mixed_dense_forward_batch_q88_q1616(weights_q88, inputs_q1616, n_outputs, n_inputs)`
Pure-Python batched mixed-precision dense MAC — the bit-true floor reference.

Parameters
----------
weights_q88 : array_like
    Row-major ``n_outputs * n_inputs`` Q8.8 weights (``int16`` range).
inputs_q1616 : array_like
    Row-major ``n_batch * n_inputs`` Q16.16 input codes (``int32`` range).
n_outputs : int
    Number of dense output channels.
n_inputs : int
    Number of dense input channels.

Returns
-------
MixedDenseBatchResult
    Per-batch, per-output saturated Q16.16 codes with overflow/underflow flags.

Raises
------
ValueError
    If shapes are inconsistent or the accumulation can exceed ``int64``.

### Function `available_backends()`
Probe which acceleration backends can run the mixed-dense kernel.

Returns
-------
dict
    Mapping of backend name to availability, in fastest-first order. The
    ``python`` floor is always ``True``.

### Function `mixed_dense_forward_batch(weights_q88, inputs_q1616, n_outputs, n_inputs)`
Run the batched mixed-precision dense MAC through the fastest available backend.

Parameters
----------
weights_q88, inputs_q1616, n_outputs, n_inputs
    See :func:`mixed_dense_forward_batch_q88_q1616`.
backend : str, optional
    ``"auto"`` (default) selects the fastest available backend in
    :data:`FASTEST_FIRST_BACKENDS` order. A specific name forces that backend.

Returns
-------
MixedDenseBatchResult
    Bit-identical to the Python floor for every backend.

Raises
------
ValueError
    If ``backend`` is not a known name.
ImportError
    If an explicitly requested accelerator backend is unavailable.

---

## Module `compiler.mixed_dense_quantization`

### Class `CompiledMixedDense`
Bit-true mixed fixed-point dense operator compiled from float weights.

- **__post_init__**()
- **output_size**()
  - Number of dense output channels.
- **input_size**()
  - Number of dense input channels.
- **accumulator_divisor**()
  - Raw-product divisor that converts Qw*Qa products into Qa codes.
- **manifest**()
  - Deterministic deployment metadata for host, Rust, and HDL emitters.
- **forward_with_overflow**(inputs)
  - Return saturated Q-accumulator codes and per-output overflow flags.
- **forward_accumulator_codes**(inputs)
  - Return saturated accumulator-format integer codes for dense outputs.
- **precision_trap_report**(inputs)
  - Return saturation telemetry suitable for a hardware trap register.
- **precision_envelope_report**(inputs)
  - Return a conservative absolute-output envelope for this workload.
- **forward_float**(inputs)
  - Return dense outputs reconstructed from saturated accumulator codes.

### Function `compile_dense_mixed_precision(weights, fmt)`
Compile a dense weight matrix into the mixed Q8.8/Q16.16 MAC contract.

---

## Module `compiler.mixed_precision_spec`

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
- **require_scalar_encoding**()
  - Reject variables whose precision needs detached block exponents.
- **summary**()
  - Return a human-readable summary of the precision allocation.
- **manifest**()
  - Return deterministic per-variable precision metadata.

---

## Module `compiler.mlir_emitter`

### Class `MLIRNode`
Operation record emitted into the dependency-free MLIR text builder.


### Class `MLIRBundle`
Generated MLIR file, optional lowered Verilog, and evidence manifest.

- **to_dict**()
  - Return a JSON-serialisable manifest representation.

### Class `MLIREmitter`
Translate sc-neurocore objects into MLIR text formatted for CIRCT.

- **__init__**(module_name)
- **get_wire**()
  - Allocate the next SSA wire name for emitted MLIR operations.
- **emit_and**(lhs, rhs)
  - Emit a comb.and operation for stochastic multiplication.
- **emit_lfsr**(width, seed)
  - Emit a clocked, parametric ``sc_lfsr`` instance for CIRCT lowering.
- **emit_xor**(lhs, rhs)
  - Emit a comb.xor operation.
- **emit_mux**(cond, true_val, false_val)
  - Emit a comb.mux operation for SC scaled addition.
- **generate**()
  - Generate CIRCT-consumable ``hw``/``comb`` dialect MLIR for the module.
- **write_bundle**(output_dir)
  - Write MLIR plus a manifest describing CIRCT lowering readiness.

### Function `generate_mlir_bundle(emitter, output_dir)`
Write a CIRCT-ready MLIR file, a manifest, and optionally lowered Verilog.

With ``run_circt=False`` (default) the bundle is evidence-first: it records
whether ``circt-opt`` is available but does not execute it. With
``run_circt=True`` it verifies the module and lowers it to Verilog through
``circt-opt``, writing ``<module>.v`` and recording genuine execution
evidence in the manifest; it raises :class:`SCCompilerError` if ``circt-opt``
is missing or rejects the MLIR, rather than claiming an un-run lowering.

---

## Module `compiler.multi_target`

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

## Module `compiler.operator_abstraction`

### Class `LiftedSignal`
One internal result abstracted to a free input port.

Attributes
----------
internal : str
    Name of the internal signal to abstract (e.g. a multiplier product wire).
port : str
    Name of the free input port to expose it as. It may differ from
    ``internal`` so two modules that name the product differently can present
    the same abstracted interface for the miter to share.
msb : str or None
    Most-significant bit-index expression of the port width, preserving a
    parameter-dependent width (``"2*DATA_WIDTH-1"``). ``None`` declares a
    scalar (1-bit) port.
signed : bool
    Whether the port is declared ``signed``.

- **declaration**()
  - Return the ``input wire`` port declaration for the module header.

### Function `abstract_to_free_inputs(verilog)`
Abstract each signal's driver away and expose it as a free input port.

For every :class:`LiftedSignal` the internal name is renamed to the target
port name, its declaration and continuous-assign driver are removed, and the
port is added to ``top``'s ANSI interface. The result over-approximates the
original: the abstracted result is unconstrained, so a miter proof that
survives it is sound for the concrete design (see the module docstring).

Parameters
----------
verilog : str
    Single-module Verilog source containing ``top``.
top : str
    Module to transform.
signals : list&#91;LiftedSignal&#93;
    The internal results to abstract. Must be non-empty with unique port
    names that do not collide with an existing port.

Returns
-------
str
    The transformed Verilog source.

Raises
------
ValueError
    If ``signals`` is empty, a port name is duplicated or already declared, or
    a signal's declaration / driver cannot be located.

---

## Module `compiler.overflow_proof`

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


### Class `FixedPointEnvelopeProof`
Static fixed-point width proof over conservative Q-code bounds.

The proof is intended for deployment artefacts that already compute
conservative absolute output bounds, such as mixed Q8.8/Q16.16 and
block-floating dense paths.  It does not use realised output cancellation:
a small output value remains unsafe when the absolute product envelope
exceeds the signed Q-format capacity.

- **manifest**()
  - Return a stable JSON-serialisable proof manifest.

### Function `prove_fixed_point_envelope(bound_codes)`
Prove whether conservative Q-code bounds fit a fixed-point format.

Parameters
----------
bound_codes : Sequence&#91;int&#93;
    Conservative absolute output bounds in integer Q-code units.  For
    signed formats the function accepts signed codes and proves against
    their absolute magnitudes.  Unsigned formats reject negative codes.
total_bits : int
    Total output width, including the sign bit for signed formats.
fractional_bits : int
    Fractional precision bits in the Q-format.
signed : bool
    Whether the target fixed-point format is signed.

Returns
-------
FixedPointEnvelopeProof
    Fail-closed width proof and saturation requirement.

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

---

## Module `compiler.pipeline`

### Class `CompilerPipeline`
Coordinate MLIR lowering, synthesis, and place-and-route tool steps.

The pipeline writes generated artifacts under ``work_dir`` and validates
artifact paths before invoking the external EDA toolchain.

- **__init__**(work_dir)
- **compile_mlir_to_verilog**(mlir_content, output_name)
  - Lower ``hw``/``comb`` dialect MLIR to Verilog with ``circt-opt``.
- **run_synthesis**(v_path, target_fpga)
  - Run Yosys synthesis and return the expected JSON netlist path.
- **run_pnr**(json_path, target_device)
  - Run nextpnr place-and-route and return the expected ASC path.

---

## Module `compiler.pipeline_analysis`

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

## Module `compiler.power_estimator`

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

## Module `compiler.precision_config`

### Class `BlockFloatingScalarEncodingError`
Raised when block-floating precision is used by a scalar-only encoder.


### Class `BlockFloatingPrecisionConfig`
Block-floating specification for a single variable.

Attributes
----------
mantissa_bits:
    Number of signed mantissa bits emitted on the fixed datapath.
exponent_bits:
    Number of bits stored for each shared block exponent code.
block_size:
    Number of flattened parameters that share one exponent code.
signed:
    Whether the mantissa stream uses signed two's-complement values.

- **__post_init__**()
  - Validate block-floating width and layout invariants.
- **data_width**()
  - Mantissa datapath width emitted for hardware and manifest payloads.
- **fraction**()
  - Conservative fractional width used by fixed-datapath fallbacks.
- **emit_fraction**()
  - Fractional width advertised to downstream fixed-datapath emitters.
- **kind**()
  - Stable manifest kind for block-floating precision contracts.
- **int_bits**()
  - Signed mantissa magnitude bits excluding shared exponent metadata.
- **exponent_bias**()
  - Bias that maps stored exponent codes to unbiased exponents.
- **exponent_code_min**()
  - Smallest encoded shared-exponent code.
- **exponent_code_max**()
  - Largest encoded shared-exponent code.
- **mantissa_abs_max**()
  - Largest signed mantissa magnitude before applying the block exponent.
- **max_exponent**()
  - Largest unbiased exponent represented by the exponent stream.
- **min_exponent**()
  - Smallest unbiased exponent represented by the exponent stream.
- **max_value**()
  - Largest positive value representable by mantissa and exponent fields.
- **min_value**()
  - Smallest signed value representable by mantissa and exponent fields.
- **resolution**()
  - Smallest positive exponent quantum available to the block stream.
- **q_label**()
  - Canonical block-floating label including mantissa, exponent, and block size.
- **is_block_floating**()
  - Whether this precision contract requires shared exponent metadata.
- **supports_scalar_encoding**()
  - Whether ``encode`` can produce a complete scalar storage word.
- **require_scalar_encoding**()
  - Reject scalar consumers before block exponent metadata is lost.
- **can_represent**(value)
  - Return whether ``value`` lies inside the coarse block-floating range.
- **encode**(value)
  - Reject scalar encoding when block exponent metadata is unavailable.
- **manifest**()
  - Return the parameter-count-independent block-floating manifest.
- **block_exponent_count**(parameter_count)
  - Return the number of shared exponents needed for ``parameter_count``.
- **block_exponent_layout**(parameter_count)
  - Return the flattened exponent-vector layout for ``parameter_count``.
- **validate_exponents**(exponents)
  - Validate and normalise encoded block exponents for a parameter payload.
- **manifest_for_parameter_count**(parameter_count)
  - Return a deterministic manifest with optional concrete layout metadata.

### Class `PrecisionConfig`
Fixed-point configuration for a single variable.

Attributes
----------
data_width:
    Total fixed-point storage width in bits.
fraction:
    Number of fractional bits below the binary point.
signed:
    Whether encoded values use signed two's-complement storage.

- **int_bits**()
  - Integer magnitude bits available above the configured fraction.
- **max_value**()
  - Largest fixed-point value representable by this configuration.
- **min_value**()
  - Smallest fixed-point value representable by this configuration.
- **resolution**()
  - Quantisation step represented by one least-significant bit.
- **q_label**()
  - Standard sign-inclusive Q-format label.
- **emit_fraction**()
  - Fractional width advertised to downstream fixed-point emitters.
- **kind**()
  - Stable manifest kind for fixed-point precision contracts.
- **is_block_floating**()
  - Whether this precision contract requires shared exponent metadata.
- **supports_scalar_encoding**()
  - Whether ``encode`` can produce a complete scalar storage word.
- **require_scalar_encoding**()
  - Validate that this fixed-point config is scalar-encodable.
- **manifest**()
  - Return a deterministic fixed-point manifest for compilers and telemetry.
- **can_represent**(value)
  - Return whether ``value`` lies inside the fixed-point dynamic range.
- **encode**(value)
  - Quantise ``value`` to the nearest clamped fixed-point integer code.

### Function `encode_scalar_value(config, value)`
Encode one scalar value or reject precision modes requiring exponent metadata.

---

## Module `compiler.precision_presets`

### Function `from_preset(var_presets)`
Create a MixedPrecisionSpec from named presets.

Set ``scalar_only`` when the downstream consumer cannot carry detached
block-exponent metadata. Block-floating selections then fail during preset
resolution instead of later scalar encoding.

---

## Module `compiler.precision_solver`

### Function `solve_precision(bounds)`
Solve a deterministic per-variable fixed-point precision assignment.

The solver derives integer bits from each value range and fractional bits
from the requested resolution. Optional total-bit budgets reduce the largest
fractional fields first until the budget is met or every reduced variable is
already at one fractional bit.

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
    Align each variable's data_width to this multiple.

Returns
-------
MixedPrecisionSpec
    Per-variable precision configuration derived from range, resolution,
    alignment, signedness, and optional total-bit budget constraints.

---

## Module `compiler.proof_transforms`

### Class `ProofTransform`
Metadata for one selectable formal-proof RTL transform.

Attributes
----------
kind : ProofTransformKind
    Stable dispatch key accepted by :func:`apply_proof_transform`.
module : str
    Import path that owns the concrete transform implementation.
entrypoint : str
    Public callable name inside ``module``.
purpose : str
    Human-readable reason the proof pipeline may select the transform.
default_enabled : bool
    Whether ordinary compiler emission enables the transform by default.
    Proof transforms remain opt-in, so current registry entries are false.


### Function `list_proof_transforms()`
Return the formal-proof transforms available to opt-in proof pipelines.

Returns
-------
tuple&#91;ProofTransform, ...&#93;
    Immutable registry entries. Every entry is disabled by default for
    ordinary compilation and must be selected explicitly by proof tooling.

### Function `get_proof_transform(kind)`
Return the registry entry for ``kind``.

Parameters
----------
kind : str
    Stable transform key, such as ``"whitebox_taps"`` or
    ``"operator_abstraction"``.

Returns
-------
ProofTransform
    Registry metadata for the requested transform.

Raises
------
KeyError
    If ``kind`` is not registered.

### Function `apply_proof_transform(kind, verilog)`
Apply one registered proof-only RTL transform.

Parameters
----------
kind : ProofTransformKind
    Transform dispatch key.
verilog : str
    Verilog source containing ``top``.
top : str
    Module name passed to the selected transform.
taps : sequence of StateTap, optional
    State taps required when ``kind`` is ``"whitebox_taps"``.
signals : sequence of LiftedSignal, optional
    Lifted signals required when ``kind`` is ``"operator_abstraction"``.

Returns
-------
str
    Transformed Verilog source.

Raises
------
KeyError
    If ``kind`` is not registered.
ValueError
    If the selected transform's required payload is missing.

---

## Module `compiler.q_format`

### Class `QFormat`
Signed fixed-point Q-format specification.

Attributes
----------
integer_bits:
    Number of signed integer bits, including the sign bit.
fraction_bits:
    Number of fractional bits below the binary point.

- **__post_init__**()
  - Validate the fixed-point width fields after construction.
- **total_bits**()
  - Total signed fixed-point storage width in bits.
- **scale**()
  - Integer scale factor used to encode real values.
- **min_val**()
  - Smallest representable signed fixed-point value.
- **max_val**()
  - Largest representable signed fixed-point value.
- **min_value**()
  - Minimum representable fixed-point value.
- **max_value**()
  - Maximum representable fixed-point value.
- **q_label**()
  - Canonical Q-format label.
- **from_string**(cls, fmt)
  - Parse 'Q8.8', 'Q4.12', etc.

### Class `QFormatMixed`
Mixed fixed-point contract for Q-format weights and wider accumulators.

Attributes
----------
weight_fmt:
    Stored weight format.
accum_fmt:
    Wider accumulator format used by mixed-precision dense kernels.
scale_per_tensor:
    Whether one scale is shared by the tensor instead of per-channel scale.
rounding:
    Rounding policy applied by the quantizer.

- **__post_init__**()
  - Validate mixed-format compatibility after construction.
- **accumulator_guard_bits**()
  - Extra accumulator bits available above the stored weight width.
- **metadata**()
  - Deterministic metadata for manifests and hardware telemetry.

---

## Module `compiler.quantization_reports`

### Class `PrecisionTrapReport`
Per-output fixed-point saturation report for deployment trap wiring.

- **__post_init__**()
- **output_count**()
  - Number of output channels covered by this report.
- **overflow_count**()
  - Number of outputs that saturated during the producing operation.
- **has_overflow**()
  - Whether any output channel saturated.
- **underflow_count**()
  - Number of nonzero outputs that collapsed below one output LSB.
- **has_underflow**()
  - Whether any nonzero output collapsed to the zero code.
- **saturated_min_count**()
  - Number of outputs clamped to the minimum representable code.
- **saturated_max_count**()
  - Number of outputs clamped to the maximum representable code.
- **manifest**()
  - Deterministic trap metadata for host and hardware telemetry.

### Class `PrecisionEnvelopeReport`
Conservative fixed-point output-envelope report for deployment checks.

- **__post_init__**()
- **output_count**()
  - Number of output channels covered by this report.
- **overflow_count**()
  - Number of outputs that saturated during the producing operation.
- **observed_overflow_free**()
  - Whether the realised workload avoided fixed-point saturation.
- **underflow_count**()
  - Number of nonzero outputs that collapsed below one output LSB.
- **observed_underflow_free**()
  - Whether the realised workload avoided sub-LSB output collapse.
- **conservative_safe_bound_code**()
  - Largest symmetric absolute code accepted as overflow-free.
- **max_abs_output_code**()
  - Maximum absolute saturated output code observed in the workload.
- **max_abs_bound_code**()
  - Maximum conservative absolute output bound for the workload.
- **min_headroom_code**()
  - Smallest conservative headroom, in fixed-point integer codes.
- **conservative_overflow_free**()
  - Whether the absolute envelope proves the workload is in range.
- **fixed_point_envelope_proof**()
  - Static Q-format width proof for the conservative envelope.
- **required_total_bits**()
  - Signed fixed-point width required by the conservative envelope.
- **required_integer_bits**()
  - Q-format integer bits, including sign, required by the envelope.
- **width_headroom_bits**()
  - Remaining signed fixed-point width after the conservative proof.
- **saturation_required**()
  - Whether the conservative proof requires a saturating output clamp.
- **static_overflow_proven_safe**()
  - Whether static width proof guarantees no Q-format overflow.
- **manifest**()
  - Deterministic envelope metadata for predeployment gates.

---

## Module `compiler.quantizer`

### Function `parse_precision_format(fmt)`
Parse fixed-point and block-floating precision labels.

---

## Module `compiler.resource_estimator`

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

---

## Module `compiler.riscv_driver`

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

---

## Module `compiler.sby_formal`

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

---

## Module `compiler.sensitivity_analysis`

### Function `analyze_sensitivity(layer_weights, lengths, n_trials, seed)`
Measure per-layer sensitivity to bitstream length reduction.

Parameters
----------
layer_weights:
    One-dimensional or two-dimensional layer weight arrays. Vector weights
    are treated as a single-output dense layer.
lengths:
    Candidate stochastic bitstream lengths to sample. When omitted, the
    estimator uses the default production planning ladder.
n_trials:
    Number of independent input-vector samples per layer.
seed:
    Deterministic NumPy random seed used for reproducible planning.

Returns
-------
list&#91;float&#93;
    One non-negative sensitivity score for each supplied layer.

Raises
------
ValueError
    If trial count, candidate lengths, or layer weight arrays are invalid.

---

## Module `compiler.slr_placement`

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

---

## Module `compiler.sva_gen`

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

---

## Module `compiler.synapse_planner`

### Function `assign_synapse_precisions(layer_weights, layer_names, sensitivity_maps, target_error, min_bits, max_bits, min_length, max_length, confidence)`
Assign per-synapse bit widths and SC lengths with error bounds.

Parameters
----------
layer_weights:
    One- or two-dimensional finite weight tensors, one tensor per layer.
    One-dimensional tensors are treated as single-output layers.
layer_names:
    Optional non-empty layer names. When provided, the list length must match
    `layer_weights` exactly.
sensitivity_maps:
    Optional finite non-negative sensitivity maps with shapes matching each
    corresponding weight tensor.
target_error:
    Positive aggregate target error fraction.
min_bits:
    Minimum fixed-point bit width assigned to any synapse.
max_bits:
    Maximum fixed-point bit width assigned to any synapse.
min_length:
    Minimum stochastic bitstream length assigned to any synapse.
max_length:
    Maximum stochastic bitstream length assigned to any synapse.
confidence:
    Hoeffding confidence in the open interval `(0, 1)`.

Returns
-------
list&#91;SynapsePrecision&#93;
    Validated per-synapse precision rows in layer/output/input order.

Raises
------
ValueError
    If planner bounds, names, sensitivity maps, or weight tensors are
    invalid.

---

## Module `compiler.synapse_precision`

### Class `SynapsePrecision`
Precision assignment and conservative error bound for one synapse.

Parameters
----------
layer_index:
    Zero-based layer index in the adaptive-precision plan.
layer_name:
    Non-empty layer name used in reports and manifests.
output_index:
    Zero-based output row index for the weight matrix.
input_index:
    Zero-based input column index for the weight matrix.
bit_width:
    Positive fixed-point bit width assigned to the synapse.
bitstream_length:
    Positive stochastic-computing bitstream length assigned to the synapse.
sensitivity:
    Finite non-negative synapse sensitivity score.
quantization_error_bound:
    Finite non-negative error bound from fixed-point quantization.
stochastic_error_bound:
    Finite non-negative Hoeffding-style stochastic error bound.
total_error_bound:
    Finite non-negative aggregate bound that must cover both components.

- **__post_init__**()
  - Validate per-synapse precision-row invariants.
- **to_dict**()
  - Return a JSON-serialisable precision-plan row.

---

## Module `compiler.testbench_gen`

### Function `generate_testbench(neuron, module_name, n_steps, input_current, data_width, fraction, cycles_per_step)`
Generate a Verilog testbench for a compiled equation neuron.

Drives the module with constant current for ``n_steps`` logical steps and
monitors spike_out and state outputs, producing a VCD waveform.

Parameters
----------
neuron : EquationNeuron
    The neuron (same one passed to compile_to_verilog).
module_name : str
    Must match the module name used in compile_to_verilog.
n_steps : int
    Number of logical integration steps to drive.
input_current : float
    Constant input current (Q-encoded internally).
data_width : int
    Bit width matching the compiled module.
fraction : int
    Fractional bits matching the compiled module.
cycles_per_step : int
    Clock cycles per logical step. Combinational modules advance one logical
    step per clock (``1``, the default). A pipelined module advances one
    logical step every ``latency + 1`` clocks and gates ``spike_out`` to pulse
    only on that valid cycle, so pass ``latency + 1`` here to drive ``n_steps``
    logical steps; the spike count over the run is unchanged by the padding.

Returns
-------
str
    Verilog testbench source code.

---

## Module `compiler.verilog_compiler`

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
pipeline_points : list&#91;str&#93;, optional
    Explicit list of intermediate signal names.

### Function `compile_to_datapath(neuron, module_name, data_width, fraction)`
Compile an EquationNeuron to a **combinational** processing element.

State is carried on ports — ``<var>_reg`` inputs and ``<var>_next_out``
outputs — instead of internal registers, so one PE can be time-multiplexed
across many neurons (the folded interconnect stores per-neuron state in BRAM
and streams it through this PE one neuron per cycle). ``spike_out`` and each
``<var>_next_out`` are the post-threshold, post-reset next values, computed by
the same fragments as :func:`compile_to_verilog` — so the folded datapath is
bit-for-bit identical to the per-instance module.

``param_ports`` names the parameters to carry on **input ports** instead of
baking them as module ``parameter`` defaults. A folded population whose neurons
have heterogeneous parameters drives these ports from a per-neuron parameter ROM
(one value per neuron, streamed by the sequencer), the parameter-space analogue of
the state BRAM. Because the arithmetic body references each parameter by the same
``P_<NAME>`` identifier whether it is a ``parameter`` or an ``input wire``, moving a
parameter to a port changes only its declaration — the datapath stays bit-for-bit
identical. Every name must be a real neuron parameter/constant; the rest stay baked.

Pipelining is not supported here (a combinational PE has no register stages);
the folded sequencer provides the one-cycle-per-neuron timing instead.

---

## Module `compiler.verilog_compiler_config`

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

---

## Module `compiler.whitebox_taps`

### Class `StateTap`
One observation tap exposing an internal signal as an output port.

Attributes
----------
port : str
    Name of the new ``output wire`` port.
source : str
    Verilog expression assigned to the port — typically an internal register
    name (``"v_reg"``) or a constant (``"32'd0"``) that pins a tap a peer
    module lacks.
msb : str or None
    The most-significant bit-index expression of the port width, so a
    parameter-dependent width is preserved (``"DATA_WIDTH-1"`` renders
    ``&#91;DATA_WIDTH-1:0&#93;``). ``None`` declares a scalar (1-bit) port.
signed : bool
    Whether the port is declared ``signed``.

- **declaration**()
  - Return the ``output wire`` port declaration for the module header.
- **assignment**()
  - Return the continuous ``assign`` that drives the tap from its source.

### Function `expose_state_taps(verilog)`
Instrument ``top`` to expose internal signals as observation output ports.

Adds each tap's ``output wire`` port to the module's ANSI port list and a
continuous ``assign`` before ``endmodule``. The result is behaviourally
identical to the original on its original ports; the new ports let a miter
assert the state-matching invariant an unbounded k-induction proof needs (see
the module docstring for why hierarchical references and ``bind`` do not work
with yosys 0.33).

Parameters
----------
verilog : str
    Verilog source containing ``top``.
top : str
    Module to instrument.
taps : list&#91;StateTap&#93;
    The taps to add. Must be non-empty; port names must be unique and must
    not collide with an existing port.

Returns
-------
str
    The instrumented Verilog source.

Raises
------
ValueError
    If ``taps`` is empty, a tap port is duplicated or already declared, or the
    module / its ``endmodule`` cannot be located.

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
  - Render a multi-line human-readable continual-learning report.

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

- **__post_init__**()
  - Validate the scalar learning-rule parameters.
- **positive_update**(weights, pre_spikes, post_spikes)
  - Hebbian update from positive (real) data.
- **negative_update**(weights, pre_spikes, post_spikes)
  - Anti-Hebbian update from negative (corrupted) data.
- **contrastive_step**(weights, pos_pre, pos_post, neg_pre, neg_post)
  - Apply one positive phase followed by one negative phase.
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
initial_membrane_fraction : float
    Fraction of each layer's threshold pre-loaded into the IF membrane
    potential before the first timestep. ``0.0`` reproduces the
    threshold-balancing route; ``0.5`` applies the QCFS optimal shift
    (Bu et al. 2022) that cancels the quantisation flooring bias.
n_layers : int
    Number of layers.

- **__post_init__**()
  - Derive the layer count from the converted weight stack.
- **run**(x)
  - Run the converted SNN for T timesteps on input x.
- **classify**(x)
  - Run SNN and return predicted class indices.

### Function `replace_relu_with_qcfs(model, T, theta, learn_theta)`
Swap every ReLU/ReLU6 in a model for a QCFS activation, in place.

This prepares a trained or fresh ANN for conversion-aware fine-tuning:
after substitution the network is retrained for a few epochs so the QCFS
thresholds settle, after which :func:`convert` produces a near-lossless
SNN (Bu et al. 2022).

Parameters
----------
model : nn.Module
    Model whose ReLU/ReLU6 activations are replaced. Mutated in place,
    recursing through every submodule.
T : int
    Quantisation step budget for each inserted QCFS layer.
theta : float
    Initial firing threshold for each inserted QCFS layer.
learn_theta : bool
    Whether each inserted threshold is a trainable parameter (the QCFS
    fine-tuning default).

Returns
-------
nn.Module
    The same ``model`` instance, returned for chaining.

### Function `convert(model, calibration_data, T, percentile)`
Convert a trained PyTorch ANN to a rate-coded SNN.

The conversion route is selected from the model's activations: a model
carrying :class:`QCFSActivation` layers takes the QCFS route (learned
thresholds, ``theta / 2`` membrane shift, no calibration); any other
model takes the threshold-balancing route (calibrated or unit
thresholds, rest-state membrane).

Parameters
----------
model : nn.Module
    Trained PyTorch model with Linear/Conv2d layers and either ReLU or
    QCFS activations.
calibration_data : Tensor, optional
    Sample input batch for threshold calibration on the ReLU route. If
    None, the ReLU route uses a default threshold of 1.0 per layer.
    Ignored on the QCFS route, whose thresholds are already learned.
T : int, optional
    Number of simulation timesteps (higher = more accurate, slower). If
    None, the QCFS route adopts the layers' trained step budget and the
    ReLU route defaults to 16.
percentile : float
    Activation percentile for threshold normalization on the ReLU route.

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
  - Quantise activations to the spike-rate grid with a straight-through gradient.
- **extra_repr**()
  - Return the compact PyTorch module representation.

---

## Module `core.bipolar`

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
Parser for the Mind Description Language (MDL).

A universal, substrate-independent format for archiving an agent's
architecture and state.

- **encode**(orchestrator, agent_name)
  - Export the orchestrator state to a YAML MDL string.
- **decode**(mdl_string)
  - Parse an MDL string back into a dictionary for reconstruction.

---

## Module `core.orchestrator`

### Class `CognitiveOrchestrator`
Central orchestrator that sequences registered modules into a pipeline.

Connects disparate processing modules and routes a :class:`TensorStream`
through them in execution order.

- **register_module**(name, module_obj)
  - Register a named module object for later pipeline execution.
- **set_attention**(module_name)
  - Focus orchestrator resources on a specific module.
- **execute_pipeline**(pipeline, initial_input)
  - Execute a sequence of modules, handling TensorStream conversions.

---

## Module `core.sc_correlation`

### Class `CorrelationDiagnostic`
Result of a two-stream SC correlation check.

Attributes
----------
value_a, value_b : float
    Decoded one-probabilities of the two streams.
scc : float
    Estimated stochastic cross-correlation in &#91;-1, 1&#93;.
predicted_and_bias : float
    AND bias the SCC implies via ``multiply_correlation_bias``.
observed_and_bias : float
    AND bias measured directly from the streams.
flagged : bool
    Whether ``abs(predicted_and_bias)`` exceeds the configured threshold.


### Function `estimate_scc(bits_a, bits_b)`
Estimate the stochastic cross-correlation of two equal-length bitstreams.

Parameters
----------
bits_a, bits_b : array-like of {0, 1}
    Equal-length observed SC bitstreams.

Returns
-------
float
    SCC in :math:`&#91;-1, 1&#93;`. Returns ``0.0`` for a degenerate stream (all-zero
    or all-one), where correlation is not identifiable.

Raises
------
ValueError
    If the streams are empty, non-binary, or of unequal length.

### Function `observed_and_bias(bits_a, bits_b)`
Return the measured ``AND`` bias :math:`a/N - p_a p_b` of two streams.

Parameters
----------
bits_a, bits_b : array-like of {0, 1}
    Equal-length observed SC bitstreams.

Returns
-------
float
    The signed deviation of the observed one-probability of ``AND`` from the
    independent product :math:`p_a p_b`.

### Function `correlation_diagnostic(bits_a, bits_b)`
Diagnose ``AND``-composition correlation bias between two SC bitstreams.

Estimates the SCC, converts it to the predicted AND bias through
:func:`sc_neurocore.core.sc_error_bounds.multiply_correlation_bias`, compares
it with the directly-measured bias, and flags streams whose predicted bias
exceeds ``bias_threshold``.

Parameters
----------
bits_a, bits_b : array-like of {0, 1}
    Equal-length observed SC bitstreams.
bias_threshold : float, optional
    Absolute predicted-bias magnitude above which the pair is flagged
    (default ``0.01``). Must be non-negative.

Returns
-------
CorrelationDiagnostic
    Decoded values, SCC, predicted and observed AND bias, and the flag.

---

## Module `core.sc_error_bounds`

### Class `SCErrorBound`
Summary of the accuracy of one encoded SC value.

Attributes
----------
value : float
    The encoded value.
length : int
    Bitstream length :math:`N`.
bipolar : bool
    Whether the value is in the bipolar domain.
variance : float
    Estimator variance.
std_error : float
    Estimator standard error :math:`\sqrt{\text{variance}}`.
ci95_halfwidth : float
    Half-width of the 95% normal-approximation confidence interval
    (:math:`1.959964 \times \text{std\_error}`).


### Function `bernoulli_variance(value, length)`
Variance of the decoded unipolar SC estimate :math:`\hat p = k/N`.

Parameters
----------
value : float
    The encoded probability :math:`p \in &#91;0, 1&#93;`.
length : int
    Bitstream length :math:`N` (positive).

Returns
-------
float
    :math:`p (1 - p) / N`.

### Function `bernoulli_std_error(value, length)`
Return the standard error :math:`\sqrt{p(1-p)/N}` of a unipolar SC estimate.

### Function `bipolar_variance(value, length)`
Variance of the decoded bipolar SC estimate for :math:`v \in &#91;-1, 1&#93;`.

With :math:`p = (v + 1)/2` and :math:`\hat v = 2 \hat p - 1`,
:math:`\operatorname{Var}(\hat v) = 4 \operatorname{Var}(\hat p)
= (1 - v^2)/N`.

Parameters
----------
value : float
    Bipolar value :math:`v \in &#91;-1, 1&#93;`.
length : int
    Bitstream length :math:`N` (positive).

Returns
-------
float
    :math:`(1 - v^2)/N`.

### Function `bipolar_std_error(value, length)`
Return the standard error :math:`\sqrt{(1 - v^2)/N}` of a bipolar SC estimate.

### Function `multiply_variance(value_a, value_b, length)`
Variance of a unipolar ``AND`` product of two *independent* streams.

The output stream is Bernoulli with value :math:`p_a p_b`, so its decoded
estimate has variance :math:`p_a p_b (1 - p_a p_b) / N`.

Parameters
----------
value_a, value_b : float
    Encoded probabilities in :math:`&#91;0, 1&#93;`.
length : int
    Bitstream length :math:`N` (positive).

Returns
-------
float
    :math:`p_a p_b (1 - p_a p_b) / N`.

### Function `multiply_correlation_bias(value_a, value_b, scc)`
Signed bias of a unipolar ``AND`` from stochastic cross-correlation.

For independent streams :math:`E&#91;\text{AND}&#93; = p_a p_b`. With stochastic
cross-correlation :math:`\rho \in &#91;-1, 1&#93;` (Alaghi & Hayes), the joint
one-probability moves linearly toward its comonotone bound
:math:`\min(p_a, p_b)` for :math:`\rho > 0` and its countermonotone bound
:math:`\max(0, p_a + p_b - 1)` for :math:`\rho < 0`:

.. math::
    E&#91;\text{AND}&#93; = p_a p_b + \begin{cases}
        \rho\,(\min(p_a, p_b) - p_a p_b) & \rho \ge 0 \\
        \rho\,(p_a p_b - \max(0, p_a + p_b - 1)) & \rho < 0.
    \end{cases}

Parameters
----------
value_a, value_b : float
    Encoded probabilities in :math:`&#91;0, 1&#93;`.
scc : float
    Stochastic cross-correlation :math:`\rho \in &#91;-1, 1&#93;`.

Returns
-------
float
    The signed bias :math:`E&#91;\text{AND}&#93; - p_a p_b`.

### Function `mux_add_variance(value_a, value_b, length)`
Variance of a fair-select ``MUX`` scaled adder of two streams.

A 2:1 multiplexer with select :math:`s \sim \mathrm{Bernoulli}(1/2)` emits a
stream of value :math:`q = (p_a + p_b)/2`, whose estimate has variance
:math:`q (1 - q) / N`.

Parameters
----------
value_a, value_b : float
    Encoded probabilities in :math:`&#91;0, 1&#93;`.
length : int
    Bitstream length :math:`N` (positive).

Returns
-------
float
    :math:`q (1 - q) / N` with :math:`q = (p_a + p_b)/2`.

### Function `dot_product_variance(values_a, values_b, length)`
Variance of a MUX-tree unipolar dot product of two independent vectors.

The scaled dot product :math:`\frac{1}{K}\sum_k \text{AND}(a_k, b_k)` of
length :math:`K` has variance
:math:`\frac{1}{K^2 N}\sum_k a_k b_k (1 - a_k b_k)`, whose standard error
grows as :math:`O(1/\sqrt{K N})` — the SC analogue of the
:math:`\sqrt{K}\,u` inner-product growth of Connolly & Higham (2025).

Parameters
----------
values_a, values_b : list of float
    Equal-length vectors of probabilities in :math:`&#91;0, 1&#93;`.
length : int
    Bitstream length :math:`N` (positive).

Returns
-------
float
    Variance of the scaled dot-product estimate.

### Function `low_discrepancy_error_bound(length)`
Deterministic error bound :math:`1/N` for a low-discrepancy SC source.

Sobol/Halton bitstreams are deterministic: a value representable at
resolution :math:`1/N` is encoded exactly, and any value is within
:math:`1/N`. This replaces the pseudo-random :math:`O(1/\sqrt N)` standard
error with a hard :math:`O(1/N)` bound (Najafi, Lilja & Riedel 2018).

Parameters
----------
length : int
    Bitstream length :math:`N` (positive).

Returns
-------
float
    :math:`1 / N`.

### Function `hoeffding_confidence(length, epsilon)`
Distribution-free confidence that :math:`|\hat p - p| < \epsilon`.

Hoeffding's inequality gives
:math:`P(|\hat p - p| \ge \epsilon) \le 2 e^{-2 N \epsilon^2}`, so the
confidence is :math:`\max(0, 1 - 2 e^{-2 N \epsilon^2})`.

Parameters
----------
length : int
    Bitstream length :math:`N` (positive).
epsilon : float
    Absolute error tolerance in :math:`(0, 1&#93;`.

Returns
-------
float
    The confidence lower bound in :math:`&#91;0, 1&#93;`.

### Function `hoeffding_min_length(epsilon, confidence)`
Smallest :math:`N` guaranteeing ``confidence`` at tolerance ``epsilon``.

Inverting Hoeffding's bound,
:math:`N \ge \lceil \ln(2 / (1 - c)) / (2 \epsilon^2) \rceil`.

Parameters
----------
epsilon : float
    Absolute error tolerance in :math:`(0, 1&#93;`.
confidence : float
    Target confidence :math:`c \in &#91;0, 1)`.

Returns
-------
int
    The minimum distribution-free bitstream length.

### Function `min_length_for_std_error(value, target_std)`
Smallest :math:`N` whose SC standard error is at most ``target_std``.

Unipolar: :math:`N \ge \lceil p(1-p) / \sigma^2 \rceil`; bipolar:
:math:`N \ge \lceil (1 - v^2) / \sigma^2 \rceil`.

Parameters
----------
value : float
    Encoded value: :math:`p \in &#91;0, 1&#93;` (unipolar) or :math:`v \in &#91;-1, 1&#93;`
    (bipolar).
target_std : float
    Target standard error :math:`\sigma > 0`.
bipolar : bool, optional
    Interpret ``value`` in the bipolar domain (default ``False``).

Returns
-------
int
    The minimum bitstream length (at least 1).

### Function `sc_error_bound(value, length)`
Summarise the SC accuracy of one encoded value as an :class:`SCErrorBound`.

Parameters
----------
value : float
    Encoded value: :math:`p \in &#91;0, 1&#93;` (unipolar) or :math:`v \in &#91;-1, 1&#93;`
    (bipolar).
length : int
    Bitstream length :math:`N` (positive).
bipolar : bool, optional
    Interpret ``value`` in the bipolar domain (default ``False``).

Returns
-------
SCErrorBound
    Variance, standard error, and 95% confidence half-width.

---

## Module `core.tensor_stream`

### Class `TensorStream`
Unified tensor container for sc-neurocore.

Handles automatic conversion between the probability, bitstream, and
quantum domains.

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
Simple CLI dashboard for monitoring SC simulation rates.

- **__init__**(n_neurons)
  - Create a dashboard with one rolling history per neuron.
- **update**(firing_rates, step)
  - Append one frame of firing rates and render the dashboard.

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

---

## Module `edge.sc_network`

### Class `SCLayer`
Single dense SC layer: weights × inputs via AND + popcount threshold.

- **__post_init__**()
  - Validate and normalise layer configuration after dataclass construction.
- **words_per_input**()
  - Return packed weight words required to cover all layer inputs.
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
  - Validate network-level stochastic-computing execution parameters.
- **add_layer**(layer)
  - Append a layer after validating stochastic-mode compatibility.
- **encode_inputs**(probabilities)
  - Encode float probabilities into per-input packed bitstreams.
- **run**(input_probabilities)
  - Full inference: encode → cascaded layer inference → spike output.
- **export_weights**()
  - Export all layer weights in serialization-ready format.
- **from_weights**(cls, layers_data, bit_length, lfsr_seed)
  - Construct network from deserialized weight data.
- **layer_count**()
  - Return the number of layers currently registered in the network.
- **total_neurons**()
  - Return the total output-neuron count across all registered layers.

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

## Module `energy.folded_estimator`

### Class `FoldedAreaEstimate`
Area, latency, and power estimate for a folded-interconnect network.

All counts are pre-synthesis estimates derived from Yosys-calibrated per-block
costs; treat them as ~20%-accurate architectural guidance, not synthesis truth.

- **__post_init__**()
- **as_dict**()
  - Return a plain mapping of the estimate (for JSON artefacts).
- **summary**()
  - Human-readable one-block summary.

### Function `estimate_folded_area(metrics)`
Estimate folded-interconnect FPGA resources, latency, and power.

Parameters
----------
metrics : FoldedResourceMetrics
    The architectural summary attached to a folded compile
    (``NetworkCompilationResult.folded_metrics``).
target : str
    FPGA target key in :data:`sc_neurocore.energy.fpga_models.TARGETS`
    (``'ice40'``, ``'ecp5'``, ``'artix7'``, ``'zynq'``).
data_width : int
    Fixed-point data width (the multiply and ROM-mux widths).
clock_mhz : float
    Target clock frequency, used to turn cycles into time and energy.
event_driven : bool
    Charge the event-driven neuron cost and AER infrastructure instead of the
    clock-driven LIF cost.
include_infra : bool
    Add the AXI-Lite register-file (and AER, when ``event_driven``) infrastructure
    LUTs.

Returns
-------
FoldedAreaEstimate
    The folded area, latency, and power estimate.

Raises
------
ValueError
    If ``target`` is not a known FPGA target.

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

### Function `compare_outputs(baseline, candidate, config, context)`
Compare outputs from baseline and candidate implementations.

---

## Module `experimental.builtins`

### Function `build_builtin_registry()`
Register all built-in experimental routes.

### Function `builtin_cases_for_route(route_name)`
Return the default case set for a built-in route.

---

## Module `experimental.examples`

### Function `make_demo_sigmoid_route()`
Create a self-contained demo route for benchmarking and comparison.

### Function `build_demo_registry()`
Create a registry with the built-in demo route.

---

## Module `experimental.memory_routes`

### Function `make_delayed_recall_shared_state_route()`
Route a bounded delayed-recall task against a shared-memory candidate.

The candidate is *quantum-inspired* only in the broad architectural sense:
it adds a non-local shared latent state to a real spiking baseline. It does
not claim to model ATP, Posner molecules, or validated in-vivo quantum
memory.

---

## Module `experimental.physics_routes`

### Function `make_heat_cosine_mode_route()`
Route Monte Carlo heat evolution against an exact Neumann cosine mode.

### Function `make_harmonic_symplectic_route()`
Route harmonic-oscillator integration against the symplectic solver.

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

### Function `run_demo()`
---

## Module `experiments.demo_swarm_control`

### Function `run_demo()`
---

## Module `experiments.demo_tcbo_consciousness`

### Function `run_demo()`
---

## Module `experiments.demonstration_convergence`

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
  - Render a human-readable report of the top attributed inputs.

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
- **to_dict**()

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
  - Allocate and bind the next SSA register for an SC-IR edge.
- **get**(edge_name)
  - Return an allocated register or validate an external input register.

### Class `ShapeInference`
Infers tensor shapes dynamically across the SNN graph.

- **__init__**(input_shapes)
- **infer**(node)
  - Infer and store the output shape for one supported SC-IR node.

### Class `CompilerExporter`
Export SC-IR graph-like objects to strict SSA MLIR text.

- **__init__**(target)
  - Create an exporter for a supported compiler backend target.
- **export_to_mlir**(ir_graph, input_shapes)
  - Emit strict SSA MLIR text via topological traversal.

---

## Module `export.onnx_export`

### Class `ONNXTensorType`
Tensor element type and static shape for the JSON ONNX model.

Parameters
----------
elem_type:
    ONNX tensor element type id.
shape:
    Static tensor dimensions.

- **to_dict**()
  - Return the ONNX tensor-type dictionary representation.

### Class `ONNXNode`
Custom-domain ONNX node for a lowered stochastic-computing operation.

Parameters
----------
op_type:
    ONNX operator type.
domain:
    Operator domain.
inputs:
    Input tensor names.
outputs:
    Output tensor names.
name:
    Stable node name.
attributes:
    Optional scalar operator attributes.

- **to_dict**()
  - Return the ONNX node dictionary representation.

### Class `ONNXGraph`
JSON-serializable ONNX model envelope.

Parameters
----------
name:
    ONNX graph name.
nodes:
    Lowered ONNX nodes.
inputs:
    Named graph inputs and tensor types.
outputs:
    Named graph outputs and tensor types.
metadata:
    String metadata entries attached to the model.

- **to_dict**()
  - Return the complete ONNX model dictionary representation.
- **to_json**(indent)
  - Return the complete ONNX model as formatted JSON.

### Class `ONNXExporter`
Export SC-NeuroCore IR graphs to ONNX-compatible dictionaries.

Parameters
----------
graph_name:
    Name assigned to the emitted ONNX graph.

- **__init__**(graph_name)
- **export**(ir_graph, input_shapes, metadata)
  - Convert an SC-IR graph to an ONNX graph representation.

---

## Module `export.onnx_exporter`

### Class `SCOnnxExporter`
Export SC networks to ONNX protobuf or legacy JSON.

- **export**(layers, filename)
  - Export *layers* to *filename*.

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
- **lower**(ir_graph, input_shapes, func_name)
  - Lower SC-IR graph to Relay IR text.
- **emit_build_script**(relay_text)
  - Generate a self-contained TVM build script for the lowered IR.

---

## Module `fault_injection.fault_injection`

### Class `FaultModel`

### Class `RadiationProfile`
Preset radiation environments with typical BER (bit error rate).

- **__post_init__**()
- **leo**(cls)
- **geo**(cls)
- **deep_space**(cls)
- **terrestrial**(cls)

### Class `FaultInjectionResult`
- **__post_init__**()
- **probability_original**()
- **probability_corrupted**()
- **absolute_error**()

### Class `ResilienceReport`
- **__post_init__**()
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

- **__post_init__**()
- **to_dict**()
  - Return a JSON-ready report.

### Class `ResilienceModeReport`
Full resilience-mode output for one layer and radiation profile.

- **__post_init__**()
- **requires_replay**()
  - Whether any fault model requires deterministic replay.
- **to_dict**()
  - Return a JSON-ready report.

### Class `FaultInjectionResilienceMode`
Run seeded fault-injection trials and degradation policy together.

- **__init__**(config)
- **run**(bitstreams)
  - Evaluate resilience for a binary ``(neurons, bits)`` layer.

---

## Module `fault_injection.resilience_policy`

### Class `DegradationAction`
Runtime action recommended after seeded fault diagnosis.


### Class `SeededFaultObservation`
Fault-injection observation linked to deterministic replay seed.

- **__post_init__**()

### Class `DegradationPlan`
Graceful-degradation decision derived from seeded diagnostics.

- **__post_init__**()
- **to_dict**()
  - Return a JSON-ready summary without expanding full bitstreams.

### Class `GracefulDegradationPolicy`
Combine seeded fault injection with stochastic-doctor diagnosis.

- **__post_init__**()
- **evaluate**(bitstreams)
  - Inject seeded faults, audit the layer, and recommend degradation.

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
  - Return whether the privacy budget is exhausted.

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
  - Seed the client RNG deterministically from the client id.
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
  - Append a per-round loss value to the client history.
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
  - Build a differential-privacy certificate from an accountant.
- **to_dict**()
  - Return the certificate as a serialisable dictionary.
- **is_compliant**()
  - Return whether the spent epsilon is within the target budget.

### Class `AdaptiveEpsilonScheduler`
Dynamically adjust ε per round based on convergence state.

- **__post_init__**()
  - Initialise the current epsilon to the base privacy budget.
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
  - Stamp the audit entry with the current wall-clock time.

### Class `AuditLog`
Per-round provenance record for regulatory compliance.

- **log_round**(round_number, num_active, epsilon_consumed, grad_norm)
  - Record one federated round in the audit log.
- **to_list**()
  - Return the audit log as a list of serialisable dictionaries.
- **total_rounds**()
  - Return the number of logged federated rounds.
- **max_epsilon**()
  - Return the maximum epsilon recorded across all rounds.

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

## Module `federation.attestation`

### Class `FpgaArtifact`
A content-addressed FPGA evidence artifact backing a hardware claim.

Parameters
----------
role
    What the artifact is — e.g. ``"synthesis-timing"``, ``"synthesis-utilisation"``,
    ``"cosim-transcript"`` (the bit-exactness proof), or ``"bitstream"``.
digest
    SHA-256 of the artifact (``"sha256:<hex>"``); the identity that binds the claim
    to the real file.
media_type
    The artifact's media type (e.g. ``"text/vivado-timing"``).

Raises
------
ValueError
    If any field is blank.

- **__post_init__**()
  - Reject blank fields — every artifact must be addressable.
- **to_dict**()
  - Return the JSON-serialisable mapping of the artifact.

### Function `regrade_sc_inference(unit)`
Recompute the sc-inference grade from the signed unit.

``reference-validated`` only when the accelerated backend is bit-identical to the
NumPy floor (``max_abs_error == 0``); otherwise ``bounded-model``. Derived from the
evidence, never read from a (forgeable) ``claim_status`` field.

### Function `seal_sc_inference(result)`
Seal an sc-inference result in **recompute** mode (no attestation).

Parameters
----------
result
    The path-free stochastic-computing inference result.
signer
    The studio's :class:`~scpn_studio_platform.seal.keys.Signer`.

Returns
-------
HonestyEnvelope
    A recompute-verifiable envelope: the WASM/NumPy kernel re-derives the result and
    its digest must match.

### Function `regrade_fpga(unit)`
Recompute the FPGA grade from the signed unit.

``reference-validated`` only when the co-simulation is bit-exact **and** a
``cosim-transcript`` artifact backs that claim; otherwise ``validation-gap``. A
bit-exact flag with no transcript to prove it does not earn the validated grade.

### Function `attest_fpga_deployment(result, artifacts)`
Seal an FPGA deployment in **attestation** mode with a signed result-pack.

The FPGA run cannot be recomputed client-side, so the claim is verified by signature
plus artifact-digest binding, not recompute. The result-pack reference content-
addresses the real Vivado/cosim/bitstream artifacts and is studio-self-attested (the
``provider_sig`` is the studio's detached signature over the pack digest).

Parameters
----------
result
    The path-free synthesis/deployment result.
artifacts
    The content-addressed artifacts the claim rests on; at least one is required, and
    a ``cosim-transcript`` must be present for the claim to grade as validated.
signer
    The studio's :class:`~scpn_studio_platform.seal.keys.Signer`.

Returns
-------
HonestyEnvelope
    An attestation-verifiable envelope on the ``fpga`` substrate.

Raises
------
ValueError
    If no artifacts are supplied.

### Function `verify_envelope(envelope, rendered_grade)`
Verify a SC-NeuroCore honesty envelope, dispatching the regrade by schema.

Parameters
----------
envelope
    The :meth:`HonestyEnvelope.to_dict` wire form on the page, or ``None`` when the
    page carries no seal.
rendered_grade
    The grade the page displays for the claim, or ``None``.
keyring
    The trust anchor mapping ``key_id`` → public verifier.

Returns
-------
Verdict
    :data:`~scpn_studio_platform.seal.verdict.Verdict.VERIFIED` only when the
    signature is valid and the rendered grade equals the grade recomputed from the
    signed unit; otherwise ``STRIPPED`` / ``FORGED`` / ``UNGRADED``.

---

## Module `federation.evidence`

### Class `ScInferenceResult`
A path-free stochastic-computing inference result.

Parameters
----------
active_backend
    The accelerated backend that produced the result (e.g. ``"rust"``).
reference_backend
    The bit-true reference backend (the NumPy floor).
max_abs_error
    Maximum absolute difference between the active and reference outputs for
    the fixed seed; ``0.0`` means bit-identical.
bitstream_length
    The stochastic bitstream length used.
input_digest
    SHA-256 of the weights/inputs/seed.
result_digest
    SHA-256 of the output firing rates.

Raises
------
ValueError
    If a count is non-positive, an error is negative, or a digest is empty.

- **__post_init__**()
  - Validate counts, error sign, and digests.

### Class `FpgaDeploymentResult`
A path-free FPGA synthesis/deployment result.

Parameters
----------
device
    The target device (e.g. ``"xc7z020-1clg400"``).
cosim_bit_exact
    Whether the synthesised RTL co-simulation matched the Q8.8 fixed-point
    reference bit-for-bit.
lut_used, ff_used
    Look-up tables and flip-flops consumed by the synthesised design.
worst_negative_slack_ns
    Worst negative slack at the target clock; ``>= 0`` means timing closed.
clock_mhz
    The synthesis target clock frequency, in MHz.
result_digest
    SHA-256 of the bitstream / synthesis report.

Raises
------
ValueError
    If a resource count is negative, the clock is non-positive, or the digest
    is empty.

- **__post_init__**()
  - Validate resource counts, clock, and digest.

### Function `sc_inference_evidence(result)`
Build the ``studio.sc-inference.v1`` bundle (measured software result).

Parameters
----------
result
    The path-free inference result.
operator
    Opaque identity of the operator/tenant.
studio_version
    Version of the SC-NeuroCore studio that produced the result.
started, ended
    ISO-8601 start/end timestamps (passed in; no hidden clock).
host
    Optional host descriptor the run executed on.

Returns
-------
EvidenceBundle
    A ``measured`` bundle that renders as validated only when the accelerated
    backend is bit-identical to the NumPy floor.

### Function `fpga_deployment_evidence(result)`
Build the ``studio.fpga-deployment.v1`` bundle (hardware-validated silicon).

The synthesised RTL is co-simulated against the Q8.8 fixed-point reference on
the ``fpga`` substrate. The claim is ``reference-validated`` only when that
co-simulation is bit-exact; a mismatch degrades to ``validation-gap``.

Parameters
----------
result
    The path-free synthesis/deployment result.
operator
    Opaque identity of the operator/tenant.
studio_version
    Version of the SC-NeuroCore studio that produced the result.
started, ended
    ISO-8601 start/end timestamps (passed in; no hidden clock).
host
    Optional host descriptor the synthesis ran on.

Returns
-------
EvidenceBundle
    A ``hardware-validated`` bundle on the ``fpga`` substrate.

---

## Module `federation.manifest`

### Function `declared_surface()`
Return the content-addressable declared surface of the SC-NeuroCore studio.

The surface is the canonical JSON of each advertised verb plus the evidence
schema list, keyed by a stable logical path. Hashing surface *content* (not git
state) is what makes the digest reproducible across checkouts.

Returns
-------
dict&#91;str, bytes&#93;
    Mapping of logical path to canonical-JSON bytes, suitable for
    :func:`scpn_studio_platform.manifest.content_digest`.

### Function `build_manifest()`
Build the SC-NeuroCore studio's capability manifest.

Parameters
----------
studio_version
    The studio version to stamp; defaults to :data:`STUDIO_VERSION` (the
    source ``sc-neurocore`` version).

Returns
-------
CapabilityManifest
    The schema-A manifest, with a content digest over :func:`declared_surface`.

---

## Module `federation.verbs`

### Function `core_verbs()`
Return the SC-NeuroCore verbs drawn from the shared core spine.

Returns
-------
tuple&#91;Verb, ...&#93;
    The subset of :data:`NEUROCORE_VERBS` whose names are in the platform
    :data:`scpn_studio_platform.verbs.CORE_VERBS`.

### Function `domain_verbs()`
Return the SC-NeuroCore verbs distinctive to a neuromorphic studio.

Returns
-------
tuple&#91;Verb, ...&#93;
    The subset of :data:`NEUROCORE_VERBS` not present in the platform core
    spine (``encode``, ``deploy``).

### Function `evidence_schemas()`
Return the distinct evidence schemas every SC-NeuroCore verb emits.

Returns
-------
tuple&#91;str, ...&#93;
    The sorted, de-duplicated union of each verb's ``produces`` schemas.

---

## Module `few_shot.haam`

### Class `HebbianFewShot`
Class-indexed Hebbian memory for few-shot spike episodes.

Parameters
----------
n_features : int
    Number of spike-rate features per pattern after temporal averaging.
n_classes : int
    Number of class slots stored by the associative memory.
lr_hebbian : float, default=0.1
    Non-negative multiplier applied when support patterns are accumulated
    into their class memory rows.

- **__init__**(n_features, n_classes, lr_hebbian)
- **store**(spike_pattern, label)
  - Store one support pattern in the class memory.
- **query_scores**(spike_pattern)
  - Return cosine scores for a query against every stored class.
- **query**(spike_pattern)
  - Classify one query pattern by nearest stored memory.
- **few_shot_episode**(support_x, support_y, query_x)
  - Run one reset-store-query few-shot episode.
- **export_weights**()
  - Return a defensive copy of the class memory matrix.
- **reset**()
  - Clear the memory matrix and support counts.

### Class `SpikePrototypeNet`
Nearest-prototype classifier for spike-rate few-shot episodes.

Parameters
----------
n_features : int
    Number of features per vector after temporal averaging.
metric : {"cosine", "euclidean", "hamming"}, default="cosine"
    Distance or similarity metric used to score queries against support-set
    prototypes. ``hamming`` thresholds vectors at zero and scores by negative
    normalised bit disagreement.

- **__post_init__**()
  - Validate the prototype classifier configuration after dataclass init.
- **classify**(support_x, support_y, query_x)
  - Classify query patterns from support-set prototypes.
- **export_prototypes**()
  - Return defensive copies of the most recently computed prototypes.

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

---

## Module `formal.lean_bridge`

### Class `FormalProofEngine`
Invokes Lean 4 `safety_bounds.lean` to formally verify mathematical parameters dynamically.

- **__init__**()
- **is_available**()
- **list_axioms**()
  - Return explicit Lean axiom names declared in the bundled proof file.
- **list_theorems**()
  - Return explicit top-level Lean theorem names declared in the proof file.
- **axiom_inventory_matches**()
  - Return True only when the proof file contains the reviewed axiom set.
- **theorem_inventory_matches**()
  - Return True only when the proof file contains the reviewed theorem set.
- **proof_inventory_matches**()
  - Return True only when reviewed theorem and axiom inventories match.
- **check_proofs**()
  - Invoke native Lean elaboration for the bundled proof boundary.

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
Enumeration of the supported sensor modalities.


### Class `EventStream`
Timestamped event stream from a single sensor modality.

- **num_events**()
  - Return the number of events in the stream.
- **duration_us**()
  - Return the stream duration in microseconds.
- **event_rate**()
  - Return the mean event rate in events per microsecond.
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
- **measure_scc**(a, b)
  - Compute stochastic cross-correlation between two bitstreams.

### Class `CrossModalAttention`
SC-domain cross-modal attention kernel.

Implements query-key-value attention using stochastic arithmetic:
  - Q·K similarity: SC-AND (bitwise AND = multiplication)
  - Weighted V: SC-MUX (bitwise multiplexer = scaled addition)

- **__init__**(num_channels, seed)
- **attend**(query_stream, key_stream, value_stream)
  - Compute cross-modal attention in SC domain.

### Class `FusionMetrics`
Metrics from a sensor fusion pass.


### Class `SensorFusionLayer`
Multi-stream sensor fusion with per-modality weighting.

- **__init__**(num_channels, bitstream_length, seed)
- **set_weight**(modality, weight)
  - Set the fusion weight for a modality, clipped to ``&#91;0, 1&#93;``.
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
  - Encode DVS camera events into an EventStream of AER addresses.

### Class `CochleaAdapter`
Adapter for silicon cochlea (frequency-to-channel mapping).

- **__init__**(num_channels, freq_min_hz, freq_max_hz)
- **freq_to_channel**(freq_hz)
  - Log-scale frequency to channel mapping (tonotopic).
- **encode_spikes**(timestamps, frequencies)
  - Encode cochlear spike timestamps and frequencies into an EventStream.

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
  - Emit configurable multi-modal fusion SystemVerilog as a string.

### Class `FusionEnergyEstimate`
Sub-mW energy estimate for fusion pipeline.

- **total_mw**()
  - Return the total estimated power in milliwatts.

### Class `FusionEnergyEstimator`
Estimates per-inference energy for SC fusion on FPGA.

- **__init__**(tech_node_nm, vdd_v)
- **estimate**(num_streams, num_channels, bitstream_length, use_attention, clock_mhz)
  - Estimate fusion energy from stream, channel, and timing parameters.

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
Hyperdimensional computing encoder.

Dimension D is usually >= 10,000.

- **generate_random_vector**()
  - Generate a random D-dimensional binary vector in {0, 1}.
- **bind**(v1, v2)
  - Bind two hypervectors via XOR.
- **bundle**(vectors)
  - Bundle hypervectors by majority superposition.
- **permute**(v, shifts)
  - Permute a hypervector by a cyclic shift.

### Class `AssociativeMemory`
Simple HDC associative clean-up memory.

Stores (key, value) pairs or bare prototypes for nearest-match retrieval.

- **store**(label, vector)
  - Store a labelled hypervector in the clean-up memory.
- **query**(query_vec)
  - Return the label of the closest stored vector by Hamming distance.

---

## Module `hdl.aer_priority_queue_reference`

### Class `AERPriorityEvent`
Address-event payload emitted by the hardware priority queue.


### Class `AERPriorityStep`
One cycle of queue input/output activity.


### Class `AERPriorityQueueReference`
Finite AER strict-priority queue with hardware-equivalent traps.

- **__init__**()
- **occupancy**()
- **ready**()
- **empty**()
- **full**()
- **enqueue**(event)
  - Accept an event if capacity is available, otherwise latch a drop.
- **peek**()
  - Return the next event without consuming it.
- **dequeue**()
  - Consume and return the highest-priority event.
- **step**(event)
  - Advance one hardware cycle with optional input and output ready.
- **drain**()
  - Drain the queue according to the strict-priority contract.
- **extend**(events)
  - Enqueue all events that fit and return the accepted count.

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

### Function `generate_live_parameter_bank(spec)`
Generate a live-parameter bank from an MMIO update spec.

The emitted RTL stores each parameter bank in distributed RAM or BRAM and
exposes the fixed live-control register map through either AXI4-Lite or a
PCIe MMIO register-window adapter.  The PCIe path intentionally models the
endpoint-adapter contract only: upstream PCIe hard IP must decode posted
writes and reads into the single-clock MMIO strobes exposed here.

Both protocol paths stage low/high data words, commit updates only after a
checksum-valid update/apply handshake, and export a flattened
``parameter_words`` bus for downstream dense, BFP, or phase-coupling RTL.

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
  - Initialize a bounded research Kuramoto HDL emitter configuration.
- **initial_phase_state_fixed**()
  - Return the emitted fixed-point reset state for each oscillator.
- **fixed_point_step**(phase_state)
  - Mirror one generated RTL phase step in integer fixed-point arithmetic.
- **fixed_point_error_summary**()
  - Characterise fixed-point drift against the float Kuramoto Euler step.
- **fixed_state_to_float**(phase_state)
  - Convert integer fixed-point phase state to radians.
- **generate**()
  - Emit deterministic Verilog for the configured research Kuramoto core.

---

## Module `hdl_gen.lfsr16_emitter`

### Class `Lfsr16Emitter`
Emit a synthesisable standalone LFSR-16 Verilog module.

- **__init__**(module_name, seed)
- **generate**()
  - Return the standalone LFSR-16 Verilog module.

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

---

## Module `homeostasis.regulator`

### Class `StabilityMetrics`
Network stability measurements.

- **summary**()
  - Render a multi-line human-readable network-stability report.

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

---

## Module `identity.decoder`

### Class `StateDecoder`
Extract cognitive state from spiking network for session priming.

- **__init__**(substrate)
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

---

## Module `identity.encoder`

### Class `TraceEncoder`
Encode reasoning traces as temporal spike patterns.

LSH maps text chunks to neuron groups. Poisson spike trains
weighted by chunk salience deliver the encoded signal.

- **__init__**(n_neurons, hash_dims, seed)
- **encode**(text, duration_ms, dt)
  - Convert text to spike pattern array (n_neurons, n_steps).
- **encode_key_value**(key, value)
  - Encode a key-value pair as a combined spike pattern.

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
Convert Dynamic Vision Sensor AER events into stochastic bitstreams.

Parameters
----------
height:
    Positive number of pixel rows in the event-camera frame.
width:
    Positive number of pixel columns in the event-camera frame.
decay_tau:
    Positive finite exponential-decay time constant in milliseconds.

- **__post_init__**()
  - Validate sensor geometry and allocate the internal event surface.
- **process_events**(events)
  - Integrate a timestamp-ordered batch of DVS events.
- **generate_bitstream_frame**(length)
  - Generate a stochastic bitstream cube from the current DVS surface.

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
  - Simulate publishing a velocity command to /cmd_vel; returns success.

---

## Module `interfaces.zenith_bci_loop`

### Class `ZenithBCILoopConfig`
Configuration for deterministic closed-loop stream processing.

Attributes
----------
n_channels:
    Number of neural acquisition channels in each waveform window.
sampling_rate_hz:
    Sample rate used to convert window length into ingest latency.
gpu_lanes:
    Parallel codec/decode lanes available for latency estimation.
latency_budget_ms:
    Maximum allowed end-to-end closed-loop latency in milliseconds.
threshold_sigma:
    Spike-detection threshold multiplier passed to the BCI template.
snippet_samples:
    Number of waveform samples captured around each detected event.
waveform_mode:
    Closed-loop waveform codec mode forwarded to the BCI template.
quantize_bits:
    Bit width used by the waveform quantizer.
timestamp_bits:
    Bit width reserved for encoded event timestamps.

- **__post_init__**()
  - Validate strictly positive loop sizing and latency parameters.

### Class `ZenithBCILoopResult`
Single-step closed-loop output with latency budget evidence.

Attributes
----------
command:
    Integer control action emitted for the processed waveform window.
feedback_active_channels:
    Number of channels that received active feedback.
spike_count:
    Total detected spikes in the processed window.
decoded_rates:
    Per-channel decoded spike-rate estimates.
latency_breakdown_ms:
    Stage-level latency ledger keyed by processing stage name.
total_latency_ms:
    Sum of all estimated stage latencies in milliseconds.
latency_budget_ms:
    Budget the closed-loop step was checked against.
latency_budget_met:
    Whether ``total_latency_ms`` stayed within ``latency_budget_ms``.
pathway_name:
    Human-readable identifier for the acquisition/control pathway.
schema_version:
    Stable serializer schema emitted by :meth:`to_dict`.

- **to_dict**()
  - Return the stable JSON-compatible result payload.

### Class `ZenithBCILoop`
Closed-loop primitive for continuous BCI streams with latency guarantees.

- **__init__**(config)
- **process_stream**(waveform)
  - Process one continuous stream window into a closed-loop control action.

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

### Function `export_scnir_from_nir(model_path)`
Read a NIR model, export SC-NIR metadata, and write it to JSON.

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
  - Build the lateral-inhibition kernel matrix.
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
Fuse multiple data modalities using stochastic multiplexing.

Parameters
----------
input_dims : Mapping&#91;str, int&#93;
    Declared feature count for each accepted modality.
fusion_weights : Mapping&#91;str, float&#93;
    Raw modality weights. Positive totals are normalised to one; non-positive
    totals fall back to equal weights across the weighted modalities, matching
    the Rust fusion layer contract.
length : int, default=LAYER_DEFAULT_LENGTH
    Stochastic bitstream length carried for layer-level configuration.

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
  - Validate modality metadata and normalise fusion weights.
- **forward**(inputs)
  - Return the weighted stochastic-fusion expectation.

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
  - Build the backing layer and inject stuck-at and variability defects.
- **forward**(input_values)
  - Run a forward pass through the defect-injected stochastic layer.
- **update_weights**(gradient, lr)
  - Update weights with gradient, respecting stuck-at mask.
- **weights**()
  - Return the current defect-affected weight matrix.
- **n_stuck**()
  - Return the number of stuck-at synapses.
- **stuck_fraction**()
  - Return the fraction of synapses that are stuck-at.

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
  - Initialise the prediction weights and recurrent state.
- **forward**(inputs)
  - Process one timestep.
- **reset**()
  - Re-initialise the prediction weights and clear the previous input.

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
  - Validate the dendrite geometry and initialise compartment state.
- **step**(branch_inputs)
  - Advance one timestep.
- **branch_voltages**()
  - Current compartment voltages, shape (n_branches, branch_length).
- **reset**()
  - Reset all compartment and soma voltages to zero.

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
  - Validate the layer configuration and initialise stochastic kernels.
- **forward**(input_image)
  - Apply the stochastic convolution to a channel-first image tensor.

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
  - Build the LIF neurons and their per-input STDP synapses.
- **run_epoch**(input_values)
  - Run one bitstream epoch of duration ``length`` and return the spikes.
- **get_weights**()
  - Return the dense weight matrix gathered from all synapses.

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
- **forward**(input_values)
  - Compute output firing rates for the layer.

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
- **step**(reward)
  - Apply reward-modulated weight update.

### Class `MetaLearner`
MAML-style meta-learning for spiking networks.

Finn et al. 2017. Inner loop: fast adaptation on a task.
Outer loop: meta-gradient across tasks.

- **__init__**(network, inner_lr, outer_lr)
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

---

## Module `learning.callbacks`

### Class `TrainingCallback`
Base class for training callbacks.

- **log**(metrics, step)
  - Record a mapping of metric names to values at the given step.
- **close**()
  - Flush and release any resources held by the callback.

### Class `TensorBoardCallback`
Log scalars to TensorBoard via ``torch.utils.tensorboard``.

- **__init__**(log_dir)
- **log**(metrics, step)
  - Write each metric as a TensorBoard scalar at the given step.
- **close**()
  - Close the underlying TensorBoard summary writer.

### Class `WandBCallback`
Log metrics to Weights & Biases.

- **__init__**(project)
- **log**(metrics, step)
  - Forward the metrics to the active Weights & Biases run.
- **close**()
  - Finish the active Weights & Biases run.

### Class `CSVCallback`
Log metrics to a CSV file (no dependencies).

- **__init__**(path)
- **log**(metrics, step)
  - Buffer one row of metrics for the given step in memory.
- **close**()
  - Write all buffered metric rows to the CSV file.

---

## Module `learning.federated`

### Class `FederatedAggregator`
Privacy-preserving federated learning using SC bitstreams.

- **aggregate_gradients**(client_gradients)
  - Aggregate gradient bitstreams from multiple clients by majority vote.
- **secure_sum_protocol**(client_gradients)
  - Sum client bitstreams as a secure-aggregation surrogate.

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
Genetic algorithm for evolving SNN weights and parameters.

- **__init__**(layer_factory, fitness_func)
- **evolve**(generations)
  - Run the GA for the given number of generations and return the best individual.

---

## Module `learning.online_o1`

### Class `OnlineO1Config`
Hardware-bounded configuration for local reward-modulated STDP.

- **__post_init__**()
  - Validate and normalize fixed-point fields after dataclass initialization.
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
  - Initialize bounded mutable synapse state from a validated configuration.
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

---

## Module `learning.schedulers`

### Class `StepScheduler`
Drop learning rate by *gamma* every *step_size* steps.

- **__init__**(lr_init, step_size, gamma)
- **step**()
  - Advance one scheduler step and return the current learning rate.
- **reset**()
  - Reset the internal step counter without changing the current rate.

### Class `ExponentialScheduler`
Multiply learning rate by *gamma* each step.

- **__init__**(lr_init, gamma)
- **step**()
  - Apply one exponential decay update and return the new rate.
- **reset**()
  - Leave the stateless exponential schedule unchanged.

### Class `CosineScheduler`
Cosine annealing from *lr_init* to *lr_min* over *total_steps*.

- **__init__**(lr_init, lr_min, total_steps)
- **step**()
  - Advance one cosine-annealing step and return the new rate.
- **reset**()
  - Restore the initial learning rate and restart the cosine schedule.

### Class `WarmupCosineScheduler`
Linear warmup followed by cosine decay.

- **__init__**(lr_init, lr_min, warmup_steps, total_steps)
- **step**()
  - Advance through warmup or cosine decay and return the current rate.
- **reset**()
  - Return to the pre-warmup state with zero current learning rate.

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

---

## Module `math.category_theory`

### Class `CategoryObject`
Domain-tagged value transported between computational categories.


### Class `Morphism`
Named structure-preserving map between two :class:`CategoryObject` domains.

- **__init__**(func, name)
- **__call__**(obj)
  - Apply the morphism, tagging the result with the morphism name.

### Class `CategoryTheoryBridge`
Functors mapping between the stochastic, quantum, and bio domains.

- **stochastic_to_quantum**(bitstream)
  - Map a bitstream probability ``p`` to the quantum amplitude pair.
- **quantum_to_bio**(state_vector)
  - Map quantum probability ``|beta|^2`` to a concentration in ``&#91;0, 10&#93;`` uM.
- **bio_to_stochastic**(concentration, length)
  - Map a concentration to a Bernoulli bitstream of the given length.
- **get_functor**(source, target)
  - Return the morphism mapping the ``source`` domain to ``target``.

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

### Function `ollivier_ricci_curvature(knm, i, j, backend)`
Compute Ollivier-Ricci curvature between nodes i and j on the coupling graph.

Ollivier (2009), "Ricci curvature of Markov chains on metric spaces."
The curvature kappa(i,j) measures how much the neighborhoods of i and j
overlap. Positive curvature = neighborhoods converge (community structure).
Negative curvature = neighborhoods diverge (bottleneck).

kappa(i,j) = 1 - W1(mu_i, mu_j) / d(i,j)
where mu_i is the lazy random walk distribution from node i,
and W1 is the Wasserstein-1 distance on the unweighted support graph
(an exact successive-shortest-path min-cost flow).

Parameters
----------
knm : np.ndarray, shape (N, N)
    Coupling matrix (non-negative, not necessarily symmetric).
i, j : int
    Node indices.
backend : {"auto", "rust", "julia", "go", "mojo", "python"}
    Acceleration backend selector. ``auto`` prefers Rust when the
    ``sc_neurocore_engine`` wheel is built, else the pure-NumPy path.
    The named backends force a specific path and raise ``RuntimeError``
    when that backend is unavailable. Every backend reproduces the
    NumPy reference to float64 round-off.

Returns
-------
float
    Ollivier-Ricci curvature. Returns 0.0 for self or disconnected pairs.

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
  - Fill any unset conductance parameters from the technology presets.
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
  - Return the total count of stuck-on and stuck-off devices.
- **fault_rate**()
  - Return the fraction of crossbar cells that are faulty.

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
  - Estimate read/write power, latency and area for a crossbar array.

### Class `CrossbarTopology`
Physical wiring topology of a memristor crossbar array.


### Class `CrossbarArray`
Physical crossbar array specification.

- **num_devices**()
  - Return the device count, doubling for differential topologies.
- **conductance_model**()
  - Return the conductance model for this array's technology.

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
Strategy for compensating memristor conductance non-idealities.


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
  - Return a state variable by name.
- **__setitem__**(key, value)
  - Assign a state variable by name.
- **copy**()
  - Return an independent copy of this neuron state.
- **as_dict**()
  - Return state variables as a plain dictionary.

### Class `PluginMeta`
Metadata carried by every neuron plugin.


### Class `NeuronPlugin`
Abstract base class for pluggable neuron models.

- **meta**()
  - Return metadata describing this neuron plugin.
- **default_state**()
  - Return the default state for a new neuron instance.
- **default_params**()
  - Return default parameters for the neuron dynamics.
- **ode_dynamics**(state, current, params, dt)
  - Advance the neuron state by one timestep *dt*.
- **threshold_check**(state, params)
  - Return True if the neuron has fired.
- **reset**(state, params)
  - Reset state after a spike.
- **simulate**(current_trace, dt, params)
  - Simulate the neuron response to a current trace.

### Class `LIFPlugin`
Leaky Integrate-and-Fire neuron model.

- **meta**()
  - Return LIF plugin metadata and parameter documentation.
- **default_state**()
  - Return the resting LIF membrane state.
- **default_params**()
  - Return default SI-valued LIF parameters.
- **ode_dynamics**(state, current, params, dt)
  - Advance LIF membrane voltage by one Euler step.
- **threshold_check**(state, params)
  - Return whether the LIF voltage crosses threshold.
- **reset**(state, params)
  - Reset LIF membrane voltage after a spike.

### Class `IzhikevichPlugin`
Izhikevich (2003) simple model of spiking neurons.

- **meta**()
  - Return Izhikevich plugin metadata and parameter documentation.
- **default_state**()
  - Return regular-spiking Izhikevich default state.
- **default_params**()
  - Return regular-spiking Izhikevich default parameters.
- **ode_dynamics**(state, current, params, dt)
  - Advance Izhikevich membrane and recovery variables.
- **threshold_check**(state, params)
  - Return whether the Izhikevich spike cutoff is crossed.
- **reset**(state, params)
  - Apply Izhikevich post-spike reset to voltage and recovery.

### Class `AdExPlugin`
Adaptive Exponential Integrate-and-Fire (Brette & Gerstner 2005).

- **meta**()
  - Return AdEx plugin metadata and parameter documentation.
- **default_state**()
  - Return the default AdEx voltage and adaptation state.
- **default_params**()
  - Return Brette-Gerstner style AdEx default parameters.
- **ode_dynamics**(state, current, params, dt)
  - Advance AdEx voltage and adaptation current by one Euler step.
- **threshold_check**(state, params)
  - Return whether the AdEx spike cutoff is crossed.
- **reset**(state, params)
  - Apply AdEx spike reset and adaptation increment.

### Class `HodgkinHuxleyPlugin`
Hodgkin–Huxley conductance-based model (1952).

- **meta**()
  - Return Hodgkin-Huxley plugin metadata and parameter documentation.
- **default_state**()
  - Return resting Hodgkin-Huxley voltage and gating variables.
- **default_params**()
  - Return canonical Hodgkin-Huxley conductance parameters.
- **ode_dynamics**(state, current, params, dt)
  - Advance Hodgkin-Huxley voltage and gating variables.
- **threshold_check**(state, params)
  - Return whether the Hodgkin-Huxley voltage threshold is crossed.
- **reset**(state, params)
  - Return an independent no-op reset copy for Hodgkin-Huxley.

### Class `PluginRegistry`
Discovers and manages neuron model plugins.

- **__init__**()
  - Create an empty plugin registry.
- **register**(plugin)
  - Register a neuron plugin under its metadata name.
- **get**(name)
  - Return a registered plugin by name, if present.
- **list_plugins**()
  - Return registered plugin names in deterministic order.
- **__len__**()
  - Return the number of registered plugins.
- **__contains__**(name)
  - Return whether a plugin name is registered.
- **with_builtins**(cls)
  - Create a registry pre-loaded with all built-in neuron models.

### Class `VerilogGenerator`
Generates synthesisable SystemVerilog from a NeuronPlugin.

- **__init__**(bit_width, frac_bits)
  - Configure fixed-point width for generated plugin modules.
- **generate**(plugin)
  - Produce a complete SystemVerilog module for the given plugin.

### Class `DocGenerator`
Generates markdown documentation from plugin metadata.

- **generate**(plugin)
  - Generate Markdown documentation for one plugin.
- **generate_index**(registry)
  - Generate a summary index for all registered plugins.

---

## Module `model_zoo.pretrained`

### Function `load_pretrained(name)`
Build a model-zoo network and load its shipped pretrained weights.

Parameters
----------
name:
    Registered pretrained model name. Supported values are ``"mnist"``,
    ``"shd"``, and ``"dvs_gesture"``.

Returns
-------
Network
    A freshly built network whose projection CSR arrays were replaced by
    validated arrays from the matching ``.npz`` archive.

Raises
------
ValueError
    If the name is unknown, the registry/schema wiring is inconsistent, the
    built network projection layout does not match the archive schema, or
    any archive member is malformed.
FileNotFoundError
    If the expected pretrained archive is missing.

---

## Module `models.zoo`

### Class `SCDigitClassifier`
Pre-configured SC Network for MNIST-like Digit Classification.
Uses: Conv Layer -> Vectorized Dense Layer

- **__init__**()
- **forward**(image)
  - Classify a 28x28 image.

### Class `SCKeywordSpotter`
Dense MFCC keyword spotter (e.g. "Yes"/"No").

Classifies a fixed-length 16-dimensional MFCC feature vector with a single
vectorized SC dense layer. This is the feed-forward baseline; a recurrent
variant for variable-length audio sequences is not yet implemented.

- **__init__**(n_keywords)
- **predict**(mfcc_features)

---

## Module `nas.darts_sc_nas`

### Class `BitstreamCandidate`
SC bitstream candidate that injects variance for one stream length.

- **__init__**(length, lut_cost, power_cost)
- **forward**(x)
  - Return the candidate output with training-time SC variance noise.

### Class `SCMixedOp`
Continuous relaxation over discrete SC bitstream configurations.

- **__init__**(c_in, c_out, kernel_size, stride, padding)
- **forward**(x)
  - Return the mixed convolution output under DARTS bitstream weights.
- **expected_resource_cost**()
  - Return expected LUT and power costs from architecture weights.
- **extract_optimal_config**()
  - Return the bitstream length with the largest architecture logit.

### Class `SCNASNetwork`
Small differentiable hardware-aware search network for SC-NAS.

- **__init__**()
- **forward**(x)
  - Return class logits from the differentiable SC-NAS network.
- **hardware_penalty**()
  - Return expected LUT and power penalties across search layers.

---

## Module `nas.equiv`

### Class `EquivResult`
Result of a formal equivalence check.

- **summary**()
  - Return a one-line verdict for the equivalence proof result.

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
Supported bitstream decorrelation generators for SC-NAS candidates.


### Class `NeuronType`
Neuron model families available to the hardware-aware NAS search.


### Class `FPGAResourceBudget`
Hardware resource constraints for the target FPGA.

- **utilisation**(luts, ffs, bram, dsp)
  - Return per-resource utilisation ratios for a candidate design.

### Class `NASObjective`
Search objectives and constraints.


### Class `LayerConfig`
Configuration for a single network layer.

- **lut_cost**()
  - Return estimated LUT cost for this layer.
- **ff_cost**()
  - Return estimated flip-flop cost for this layer.
- **dsp_cost**()
  - Return estimated DSP block cost for this layer.
- **bram_cost_kb**()
  - Return estimated BRAM storage cost in kibibytes.
- **power_cost**()
  - Return estimated dynamic power cost in milliwatts.

### Class `SCCandidate`
A candidate SC network architecture.

- **evaluate_resources**()
  - Update aggregate resource estimates from the candidate layers.
- **meets_budget**(budget)
  - Return whether this candidate fits within an FPGA resource budget.
- **fingerprint**()
  - Return a deterministic non-cryptographic architecture fingerprint.

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
- **search**()
  - Run the evolutionary search. Returns the final Pareto front.

### Class `NASReport`
Summary report from an SC-NAS search.

- **best_accuracy**()
  - Return the best accuracy in the Pareto front, or zero when empty.
- **most_efficient**()
  - Return the lowest-LUT candidate in the Pareto front, if present.
- **summary**()
  - Return a deterministic human-readable search summary.

### Class `NASVerilogEmitter`
Emits SystemVerilog for Pareto-optimal SC-NAS candidates.

- **emit**(candidate, module_name)
  - Generate SystemVerilog for a searched architecture.
- **emit_pareto**(front)
  - Emit Verilog for all Pareto-optimal candidates.

### Function `pareto_front(candidates, objectives)`
Extract the Pareto-optimal front (NSGA-II non-dominated sorting).

Maximises accuracy, minimises resource usage.

### Function `run_nas(objective, budget, population_size, num_generations, seed, convergence_patience, surrogate_optimizer)`
Run an SC-NAS search and return its report.

---

## Module `nas.search`

### Class `NASResult`
Result of a NAS run.

- **best_accuracy**()
  - Architecture with highest accuracy on the Pareto front.
- **best_efficiency**()
  - Architecture with lowest energy on the Pareto front.
- **summary**()
  - Return a line-oriented summary of the Pareto front.

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
  - Return the number of layers encoded by this architecture.
- **layer_sizes**()
  - Return adjacent layer dimensions for hardware cost estimation.
- **total_params**()
  - Return the dense connection count across all encoded layers.

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

---

## Module `network._torch_bridge`

### Class `NetworkTorchBridge`
Differentiable bridge for a bounded subset of declarative ``Network`` graphs.

- **__init__**(populations, projections, surrogate_fn)
- **forward**(inputs)
  - Run the differentiable bridge on ``(T, batch, input_dim)`` current input.
- **sync_to_network**()
  - Copy learned bridge weights back into the underlying CSR projections.

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
- **population_sizes**(scale)
  - Return Potjans population sizes at ``scale`` without building connectivity.
- **step**(dt)
  - Advance the network by one timestep `dt` (ms).
- **simulate**(duration_ms, dt)
  - Run the network for `duration_ms` ms.
- **population_rates**(rasters, dt, burn_in_ms)
  - Return mean firing rate (Hz) per population.
- **total_indegree**(target)
  - Return mean total synaptic in-degree of a target cell.
- **reset_state**()
  - Re-randomise voltages, currents and refractory state.
- **__repr__**()
  - Return a concise debug representation of the cortical column.
- **population_names**()
  - Return the ordered cortical population names.

---

## Module `network.export`

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
  - Validate the population sizes and initialise oscillator state.
- **step**(dt)
  - Advance one timestep (dt in ms); return (spikes_e, spikes_i).
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
  - Return the recorded snapshot timestep array.

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
- **run**(n_steps, dt)
  - Run the distributed simulation for *n_steps* timesteps.

---

## Module `network.network`

### Class `Network`
Declarative network: collects objects, runs the simulation loop.

- **__init__**()
- **add**(obj)
  - Register a simulation object by type.
- **run**(duration, dt, progress, backend, spike_gating)
  - Run the simulation for *duration* seconds at timestep *dt*.
- **to_torch**(surrogate_fn)
  - Build an explicit differentiable bridge without altering NumPy/Rust execution.

---

## Module `network.population`

### Class `Population`
A group of N identical neurons with vectorized state access.

- **__init__**(model, n, params, label)
  - Create *n* neurons of *model* (class or string name).
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
- **n_synapses**()
  - Number of synaptic connections.
- **delay_mode**()
  - Delay mode: 'none', 'uniform', or 'per_synapse'.
- **max_delay**()
  - Maximum delay in timesteps across all synapses.
- **propagate**(source_spikes)
  - Compute target currents from source spikes through CSR connectivity.
- **update_plasticity**(src_spikes, tgt_spikes, a_plus, a_minus, tau, directional_bias)
  - Trace-based STDP weight update.

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

### Function `random_connectivity(n_src, n_tgt, p, weight, seed)`
Erdos-Renyi random connectivity.

### Function `small_world(n, k, p_rewire, weight, seed)`
Watts-Strogatz small-world graph (n-by-n adjacency).

### Function `scale_free(n, m, weight, seed)`
Generate a Barabasi-Albert preferential-attachment graph.

Parameters
----------
n : int
    Number of source and target nodes. Must be at least two.
m : int
    Number of existing nodes sampled for each new node. Must satisfy
    ``1 <= m < n``.
weight : float
    Finite synaptic weight assigned to every emitted edge.
seed : int, default=42
    Seed for deterministic preferential-attachment sampling.

Returns
-------
tuple of ndarray
    CSR ``(indptr, indices, data)`` arrays for the symmetric adjacency.

Raises
------
ValueError
    If ``n``, ``m``, or ``weight`` falls outside the Barabasi-Albert domain.

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
  - Append a timestamped reasoning step to the trace.
- **length**()
  - Return the number of recorded reasoning steps.
- **mean_confidence**()
  - Return the mean confidence across all reasoning steps.
- **is_complete**()
  - Return whether the trace has been finalised with at least one step.
- **finalize**()
  - Stamp the trace end time to mark reasoning as complete.
- **to_dict**()
  - Serialise the trace and its steps into a plain dictionary.

### Class `Hypervector`
Packed binary hypervector (pure-Python mirror of the Rust Hypervector).

Uses ``np.uint64`` packed bitstream layout compatible with the
``neuro_symbolic`` crate's ``Vec<u64>`` representation.

- **__init__**(data, length)
- **zeros**(cls, dim)
  - Construct an all-zero hypervector of the given dimensionality.
- **random**(cls, seed, dim)
  - Construct a seeded pseudo-random binary hypervector.
- **bind**(other)
  - XOR binding (self-inverse, dimension-preserving).
- **permute**(shift)
  - Cyclic right rotation by *shift* bits.
- **hamming_distance**(other)
  - Normalised Hamming distance (0.0 = identical, 1.0 = opposite).
- **similarity**(other)
  - Cosine-like similarity: 1 − 2·hamming.
- **popcount**()
  - Return the number of set bits across the packed words.
- **density**()
  - Return the fraction of bits that are set (0.0–1.0).
- **threshold_bundle**(vectors)
  - Majority-vote bundle across N vectors.

### Class `SymbolEncoder`
Deterministic symbol → hypervector mapping (mirrors Rust SymbolEncoder).

- **__init__**(base_seed)
- **encode**(symbol)
  - Return the cached or freshly generated hypervector for a symbol.
- **encode_sequence**(symbols)
  - Bind a symbol sequence into one position-aware hypervector.
- **vocabulary_size**()
  - Return the number of distinct symbols encoded so far.

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
  - Return the mean absolute error over the last 50 updates.
- **converged**()
  - Return whether recent errors are stable below the threshold.

### Class `VerifiableInference`
Wraps prediction + HDC symbol matching with an auditable trace.

- **__init__**(encoder, layer, symbol_library)
- **register_symbol**(name)
  - Register a symbol into the lookup library.
- **register_symbols**(names)
  - Register several symbols into the lookup library.
- **num_symbols**()
  - Return the number of symbols in the lookup library.
- **infer**(observation, top_k)
  - Run inference: prediction → error → HDC symbol match.

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

### Function `build_self_verification_trace(result)`
Verify a high-level neuro-symbolic inference result against its observation.

---

## Module `neurons._units`

### Function `require_pint()`
### Function `is_quantity(value)`
### Function `require_quantity(value, label)`
### Function `quantity_to_base(value)`
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

## Module `neurons.behavior_taxonomy`

### Function `behavior_tag_definition(tag)`
Return the observable definition of a behaviour tag.

Raises
------
ValueError
    If ``tag`` is not in the controlled vocabulary.

### Function `validate_behavior_tags(tags)`
Validate a behaviour-tag collection and return a sorted, unique tuple.

A model is either ``excitable`` or ``quiescent`` but never both, and a
``quiescent`` model cannot also carry a firing-pattern tag — these
contradictions signal a corrupted or hand-edited tag set and are rejected.

Parameters
----------
tags:
    An iterable of tag strings (for example the ``behavior_tags`` field of a
    descriptor or the output of the probe).

Returns
-------
tuple&#91;str, ...&#93;
    The tags, de-duplicated and sorted, ready to store or compare.

Raises
------
ValueError
    If ``tags`` is not an iterable of strings, names a tag outside the
    vocabulary, or carries a contradictory combination.

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

## Module `neurons.descriptor_generator`

### Function `generate_descriptor_payload(class_name)`
Return a curatable v2 descriptor payload for a registered model.

Parameters
----------
class_name:
    Registered model class name (a key of the model registry).

Returns
-------
dict&#91;str, Any&#93;
    A descriptor payload using the v2 section layout, with introspected and
    carried-over values filled and curation-only fields left empty.

Raises
------
ValueError
    If ``class_name`` is not a public Python identifier.
KeyError
    If ``class_name`` is not registered.

### Function `generate_descriptor(class_name)`
Return a validated :class:`ModelDescriptor` skeleton for a model.

Parameters
----------
class_name:
    Registered public model class name to introspect.

Returns
-------
ModelDescriptor
    Parsed descriptor generated from the model code and any curated v1
    schema fields.

Raises
------
ValueError
    If ``class_name`` is not a public Python identifier.
KeyError
    If ``class_name`` is not registered.

### Function `merge_descriptor_payloads(curated, regenerated)`
Merge a curated descriptor onto a freshly regenerated one.

Structural fields (which parameters and state variables exist, their
defaults and initial values, the timestep) always follow the regenerated
payload, which is read from the model code — so the corpus can never drift
from the implementation. Curation fields (parameter units/ranges/meaning,
state semantics, taxonomy, the backend matrix, reproducibility, notes,
validation evidence, silicon evidence anchors, and any richer provenance,
dynamics, or display fields) are preserved from the curated payload. The
result is the regenerated payload with curation overlaid, ready to be
re-serialised.

Parameters
----------
curated:
    The existing on-disk descriptor payload (may carry curation).
regenerated:
    A freshly generated payload from :func:`generate_descriptor_payload`.

Returns
-------
dict&#91;str, Any&#93;
    The merged descriptor payload.

---

## Module `neurons.descriptor_tiers`

### Class `CompletenessTiers`
A model's readiness on both catalogue-to-silicon axes.

Parameters
----------
science:
    Science-axis tier in ``0``-``5`` (S0-S5).
silicon:
    Silicon-axis tier in ``0``-``5`` (H0-H5), or ``None`` when the model has
    no committed silicon evidence yet.

- **science_label**()
  - The science tier as an ``S<n>`` label.
- **silicon_label**()
  - The silicon tier as an ``H<n>`` label, or ``"none"`` when unattempted.

### Function `science_tier(descriptor)`
Return the science-axis tier (0-5) the descriptor reaches.

S0-S3 are the curation kernel from
:func:`~sc_neurocore.neurons.model_descriptor.descriptor_completeness_tier`.
A descriptor only climbs above S3 when its evidence facets carry the proof:

* **S4** — faithful dynamics: declared dynamics plus a confirmed three-way
  agreement with the publication (``validation.dynamics_faithful``).
* **S5** — class-validated: a non-trivial metric and committed evidence
  (``validation.is_class_validated``).

Parameters
----------
descriptor:
    The model descriptor to score.

Returns
-------
int
    The science tier in ``0``-``5``.

### Function `silicon_tier(descriptor)`
Return the silicon-axis tier (0-5) the descriptor reaches, or ``None``.

``None`` means the model has no compile-clean RTL yet (no silicon evidence).
Otherwise the ladder climbs one rung at a time, each rung credited only when
both its boolean flag and its proof anchor are present:

* **H0** — ``silicon.compiles`` (iverilog-valid RTL).
* **H1** — Python<->Verilog validated (``cosim_validated`` + ``cosim_evidence``).
* **H2** — synthesisable (``synthesised`` + ``synth_report``).
* **H3** — timing-closed and resource-characterised (``timing_closed`` +
  ``timing_report`` + ``clock_mhz``).
* **H4** — formally equivalent (``formally_equivalent`` + ``equivalence_proof``).
* **H5** — tool-level signed PPA (``ppa_signed`` + ``ppa_report``).

Parameters
----------
descriptor:
    The model descriptor to score.

Returns
-------
int | None
    The silicon tier in ``0``-``5``, or ``None`` when no RTL compiles.

### Function `completeness_tiers(descriptor)`
Return both axis tiers for a descriptor.

Parameters
----------
descriptor:
    The model descriptor to score.

Returns
-------
CompletenessTiers
    The science (0-5) and silicon (0-5 or ``None``) tiers together.

### Function `is_perfect(descriptor)`
Return whether a model is *perfect* by the master-plan acceptance contract.

A model is perfect when it reaches S5 on the science axis and meets the
terminal silicon tier its deployability class declares
(``silicon.target_tier``). A model whose terminal tier is undeclared cannot
be certified perfect: without the deployability class the required silicon
tier is unknown, so this returns ``False`` rather than guessing.

Parameters
----------
descriptor:
    The model descriptor to judge.

Returns
-------
bool
    ``True`` only when S5 is reached and the declared terminal H-tier is met.

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

- **__init__**(equations, parameters, state, threshold, reset, constants, dt, method, units, input_unit, detection, substeps)
  - Initialise an equation-defined neuron from ODE strings.
- **initial_threshold_active**()
  - Return whether the threshold condition holds on the INITIAL committed state.
- **step**(I)
  - Advance the neuron by one macro timestep; return 1 if it spikes.
- **get_state**()
  - Return current state, with units if in strict mode.
- **reset**()
  - Reset state to initial values.
- **__repr__**()
  - Human-readable representation of the neuron equations.

### Function `from_equations()`
Build an EquationNeuron from Brian2-style equation strings.

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

## Module `neurons.equation_namespace`

### Function `build_eval_namespace()`
Return the maths namespace exposed to compiled equation expressions.

The returned dict is fresh on every call so a neuron may own its namespace
without aliasing another's. The bindings are the exact functions the
fixed-point emitter mirrors; keep them identical to preserve co-simulation
bit-exactness (see the module docstring).

---

## Module `neurons.equation_safety`

### Class `ExpressionSafetyValidator`
Validate equation expression strings against the AST allowlist.

Holds the allowed node set, the blocked-name set, and the maximum AST depth,
and exposes :meth:`validate`, which raises :class:`ValueError` on any
disallowed construct. The same validator instance is reused for a neuron's
dynamics, threshold, reset rules, and (for exponential Euler) its symbolic
Jacobian, so every expression that reaches an ``eval`` site has passed the
same gate.

- **__init__**()
  - Create a validator with the given maximum AST depth.
- **validate**(expr)
  - Validate an expression against the AST whitelist.

---

## Module `neurons.equation_units_runtime`

### Class `StrictRuntime`
Base-unit runtime values and unit maps produced by strict validation.

``parameters``/``state``/``constants``/``dt`` are the base-unit floats the
integrator steps; ``runtime_units`` maps each name (and the input ``I``) to
its base unit for converting runtime inputs; ``base_state_units`` and
``display_state_units`` let :meth:`EquationNeuron.get_state` re-attach the
caller's display units to each state variable.


### Function `prepare_strict_runtime()`
Convert pint quantities to base-unit floats for runtime.

### Function `convert_runtime_value()`
Convert a runtime value from pint quantity to float.

---

## Module `neurons.expression_derivative`

### Class `exprel`
SymPy image of the DSL ``exprel(x) = (exp(x) - 1) / x`` (exprel(0) = 1).

- **fdiff**(argindex)
  - Return d/dx (exp(x) - 1)/x = (exp(x)·(x - 1) + 1) / x².

### Class `sigmoid`
SymPy image of the DSL logistic ``sigmoid(x) = 1/(1 + exp(-x))``.

- **fdiff**(argindex)
  - Return sigmoid(x)·(1 - sigmoid(x)).

### Class `ExpressionDifferentiationError`
Raised when an expression cannot be faithfully differentiated in-grammar.


### Function `differentiate(expr, wrt)`
Return ``∂expr/∂wrt`` as an equation string in the DSL grammar.

Parameters
----------
expr:
    The right-hand-side of ``d(wrt)/dt`` — an equation string already valid
    under the neuron-DSL grammar.
wrt:
    The state-variable name to differentiate with respect to.

Returns
-------
str
    The partial derivative as a new equation string using only DSL-grammar
    tokens. ``"0"`` when the expression does not depend on ``wrt``.

Raises
------
ExpressionDifferentiationError
    If the expression depends on ``wrt`` through a construct whose derivative
    is not expressible in the smooth grammar (``abs``/``clip``/``min``/
    ``max``, a comparison, a conditional, ``%``, ``//``, or an unknown
    function).

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

## Module `neurons.model_catalogue`

### Class `CatalogueCoverage`
Aggregate descriptor coverage across the registered model catalogue.

Parameters
----------
total_models:
    Number of registered models.
described:
    Number of models with a committed descriptor.
tier_counts:
    Count of described models at each science-kernel tier (keys ``0``-``3``);
    the legacy curation view, retained for backward compatibility.
science_tier_counts:
    Count of described models at each full science-axis tier (keys ``0``-``5``,
    S0-S5).
silicon_tier_counts:
    Count of described models at each silicon-axis tier, keyed by label
    (``"none"`` for no compile-clean RTL, then ``"H0"``-``"H5"``).
citeable:
    Number of described models with citeable provenance.
fully_curated_parameters:
    Number of described models whose every parameter has unit, range, and
    meaning.

- **to_public_dict**()
  - Return a JSON-compatible coverage summary.

### Function `descriptor_path(class_name)`
Return the committed descriptor path for a model class.

Parameters
----------
class_name:
    Public Python class identifier for the model descriptor.

Returns
-------
pathlib.Path
    Absolute path under ``neurons/model_descriptors``.

Raises
------
ValueError
    If ``class_name`` is empty, private, dotted, or path-like.

### Function `load_descriptor_payload(class_name)`
Return the raw committed descriptor payload.

Parameters
----------
class_name:
    Public Python class identifier for the model descriptor.

Returns
-------
dict&#91;str, Any&#93; | None
    Parsed TOML payload when the descriptor is committed, otherwise
    ``None``.

Raises
------
ValueError
    If ``class_name`` is empty, private, dotted, or path-like.

### Function `load_descriptor(class_name)`
Return the validated committed descriptor for a model.

Parameters
----------
class_name:
    Public Python class identifier for the model descriptor.

Returns
-------
ModelDescriptor | None
    Validated descriptor when the TOML file exists, otherwise ``None``.

Raises
------
ValueError
    If ``class_name`` is empty, private, dotted, or path-like.

### Function `catalogue_descriptor_coverage()`
Return descriptor coverage over every registered model.

Returns
-------
CatalogueCoverage
    Aggregate coverage over the current model registry and committed
    descriptor corpus.

---

## Module `neurons.model_descriptor`

### Class `ParameterSpec`
A single model parameter with its physical semantics.

Parameters
----------
name:
    Parameter identifier matching the model's constructor field.
default:
    Default numeric value.
unit:
    Physical unit (for example ``mV``, ``ms``, ``pA``); empty when uncurated.
value_range:
    Optional ``(min, max)`` admissible range.
biological_range:
    Optional ``(min, max)`` biologically plausible range.
meaning:
    Human-readable description; empty when uncurated.

- **is_curated**()
  - True when the parameter has a unit, a range, and a meaning.

### Class `StateVariableSpec`
A model state variable with its initial value and semantics.


### Class `Provenance`
Citation and licensing provenance for a model.

- **is_citeable**()
  - True when authors, a year, and a valid DOI are all present.

### Class `BackendSupport`
Implementation status and numeric parity for one compute backend.


### Class `Reproducibility`
Reference-run reproducibility anchors for a model.

- **is_reproducible**()
  - True when a reference config and a golden trace digest are present.

### Class `Validation`
Class-correct validation evidence for a model's dynamics.

Records the outcome of two checks (§3-§4 of the catalogue-to-silicon master
plan): whether the model's discretised dynamics were confirmed faithful to
the publication (``dynamics_faithful`` — the three-way schema/class/paper
agreement), and by which metric the model was validated for its class, with
committed evidence. Every field is a recorded outcome, never derived, so the
science tier can read them as ground truth.

Parameters
----------
dynamics_faithful:
    True when the schema-DSL equations, parameters, dt, threshold, and reset
    were confirmed to match the publication (and the hand class where one
    exists). Gates the faithful-dynamics tier S4.
metric:
    The class-appropriate validation metric from :data:`VALIDATION_METRICS`
    (``"none"`` until validated).
operating_point:
    Human-readable statement of the operating point the validation used
    (for example an input drive or a parameter regime).
tolerance:
    The honest agreement tolerance achieved (for example ``"0 spikes"`` or a
    distributional distance), as a citeable string.
evidence:
    A path, citation, or digest pointing at the committed validation evidence.
    Together with a non-``"none"`` metric this gates the validated tier S5.

- **is_class_validated**()
  - True when a non-trivial metric and committed evidence are both present.

### Class `Silicon`
Ladder of committed silicon-realisation evidence for a model.

Each rung of the silicon axis (H0-H5) is only credited when its evidence
anchor is recorded, so a silicon tier can never be claimed ahead of proof
(master plan invariant I7). ``compiles`` is the H0 anchor (iverilog-valid
RTL); each higher boolean requires its companion report to count.

Parameters
----------
compiles:
    RTL lowers to iverilog-valid Verilog (compile-clean). The H0 anchor.
cosim_validated:
    Python<->Verilog agreement by the class-correct metric was demonstrated;
    credited for H1 only alongside ``cosim_evidence``.
synthesised:
    Passes a real synthesis flow (for example Yosys); credited for H2 only
    alongside ``synth_report``.
timing_closed:
    Meets a stated clock on a target device with reported resources;
    credited for H3 only alongside ``timing_report`` and ``clock_mhz``.
formally_equivalent:
    Machine-checked Python-semantics<->RTL equivalence in CI; credited for H4
    only alongside ``equivalence_proof``.
ppa_signed:
    Tool-level RTL->GDSII signoff (open PDK) with clean DRC/LVS/STA; credited
    for H5 only alongside ``ppa_report``.
cosim_evidence, synth_report, timing_report, equivalence_proof, ppa_report:
    Paths, citations, or digests pointing at the committed proof for each rung.
target_device:
    The device the timing/resource numbers were characterised on.
clock_mhz:
    The clock the design closes at (MHz); required for the H3 credit.
target_tier:
    The terminal H-tier this model's deployability class is expected to reach,
    from :data:`SILICON_TARGET_TIERS` (``""`` until declared).
terminal_reason:
    Why the model terminates at ``target_tier`` (for example a research
    multicompartment model that need not reach signed PPA).


### Class `ModelDescriptor`
The full declarative descriptor for one neuron model.


### Class `ModelDescriptorError`
Raised when a descriptor payload violates the schema contract.


### Function `descriptor_completeness_tier(descriptor)`
Return the science-axis completeness kernel (0-3) a descriptor satisfies.

This is the S0-S3 base of the science axis — the discovery-and-curation
tiers that need no execution evidence. The full science axis (adding the
faithful-dynamics tier S4 and the class-validated tier S5) and the silicon
axis (H0-H5) are derived from this kernel plus the descriptor's evidence
facets by :mod:`sc_neurocore.neurons.descriptor_tiers`.

Tier 0 — exists and identifies a real model (class, module, params, state).
Tier 1 — discovery taxonomy declared (family and category). Behaviour tags
         are an optional measured facet, not a tier requirement, so a tier
         never depends on running a simulation.
Tier 2 — scientifically curated: citeable provenance (authors + year + DOI)
         and every parameter curated (unit + range + meaning).
Tier 3 — engineering-verified: at least two implemented backends and a
         reproducibility anchor (reference config + golden trace digest).

### Function `parse_model_descriptor(payload)`
Validate a descriptor payload and return a :class:`ModelDescriptor`.

Parameters
----------
payload:
    A loaded descriptor mapping (from TOML/JSON), using the v2 section
    layout: ``metadata``, ``provenance``, ``state``, ``parameters``,
    ``integration``, ``dynamics``, ``backends``, ``reproducibility``,
    ``documentation``.

Returns
-------
ModelDescriptor
    The validated descriptor.

Raises
------
ModelDescriptorError
    If any required identifier is missing or a controlled-vocabulary field
    carries an unknown value.

---

## Module `neurons.model_taxonomy`

### Function `model_family(class_name)`
Return ``(family, category_slug)`` for a model, or ``None`` if unclassified.

### Function `families()`
Return the family display name to category slug mapping.

### Function `classified_models()`
Return every class name that the taxonomy classifies.

---

## Module `neurons.models.adaptive_threshold_if`

### Class `AdaptiveThresholdIFNeuron`
Integrate-and-fire with dynamic threshold. Platkiewicz & Bhatt 2010.

C dV/dt = -g_L(V - V_rest) + I
dtheta/dt = -(theta - theta_rest) / tau_theta
On spike: V -> V_reset, theta += delta_theta

Reference: Platkiewicz, J. & Brette, R. (2010). PLoS Comput. Biol. 6(7):e1000850.

- **__post_init__**()
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

- **__post_init__**()
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

``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``
(prefer Rust when the factory-default Euler contract holds and the engine
wheel is present).

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` updates from the current state, returning ``(trace, spikes)``.
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

- **__post_init__**()
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

- **__post_init__**()
- **step**(exc_current, inh_current)
- **reset**()

---

## Module `neurons.models.alpha_motor_neuron`

### Class `AlphaMotorNeuron`
Alpha motor neuron with WB Na/K, PIC, and Ca-dependent AHP dynamics.

The integrator is still the documented explicit substep path, but mutable
runtime state is validated before integration and all substep candidates are
computed locally before committing. Invalid state, unstable exponentials,
or non-finite membrane/calcium candidates fail closed without poisoning the
neuron state.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.amari_field`

### Class `AmariNeuralField`
Amari 1977 — continuous neural field, discretized on N nodes.

tau du_i/dt = -u_i + sum_j w(|i-j|) f(u_j) * dx + I_i
w(x) = A * exp(-a*|x|) - B * exp(-b*|x|)    (Mexican hat)
f(u) = max(0, u)                              (Heaviside-linear)

Reference: Amari, S. (1977). Biol. Cybern. 27:77–87.

- **__post_init__**()
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

- **__post_init__**()
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
  - Initialise the wrapped astrocyte model and exposed pseudo-voltage.
- **step**(current)
  - Advance one timestep and return the thresholded release event.
- **ca**()
  - Current cytosolic calcium concentration in micromolar.
- **ip3**()
  - Current IP3 concentration in micromolar.
- **reset**()
  - Reset the wrapped astrocyte state and pseudo-voltage.

---

## Module `neurons.models.astrocyte_lif`

### Class `AstrocyteLIFNeuron`
Astrocyte-LIF hybrid with tripartite synapse feedback.

Runtime state is revalidated before every step. Calcium and membrane
candidates are computed locally and committed only after both are finite
and physiologically admissible, preventing corrupted glial state from
leaking into downstream membrane dynamics.

- **__post_init__**()
- **step_with_pre**(i_ext, pre_spike)
  - Step with external current and presynaptic spike indicator.
- **step**(current)
  - Step without presynaptic spike info (no glial calcium increment).
- **reset**()
  - Reset state to initial conditions.

---

## Module `neurons.models.atype_k_neuron`

### Class `ATypeKNeuron`
A-type K+ neuron with Wang-Buzsaki core and transient IA current.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.av_ron_cardiac`

### Class `AvRonCardiacNeuron`
Av-Ron, Parnas & Segel cardiac ganglion Type III burster.

The four-state conductance model uses instantaneous sodium activation,
voltage-dependent h/n/s gate relaxation, and candidate-first RK4 integration
so invalid states cannot partially mutate the neuron.

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
- **step**(current)
- **reset**()

---

## Module `neurons.models.bertram_phantom`

### Class `BertramPhantomBurster`
Bertram et al. 2000 — phantom burster with dual slow variables.

C dV/dt  = -(I_Ca + I_K + I_s1 + I_s2 + I_L) + I_ext
ds1/dt   = (s1_inf(V) - s1) / tau_s1
ds2/dt   = (s2_inf(V) - s2) / tau_s2

Two slow variables (s1, s2) with different timescales produce
bursting via a phantom slow manifold.

Reference: Bertram, R., Previte, J., Sherman, A., Kinard, T.A. & Satin, L.S.
(2000). Biophys. J. 79:2880–2892.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.bk_neuron`

### Class `BKNeuron`
BK calcium-activated K+ channel neuron.

Runtime calcium, gate, and membrane candidates are computed locally and
committed only after all finite/probability/bounds checks pass. This keeps
the documented explicit substep path while preventing non-finite calcium or
voltage state from being silently reset after poisoning downstream currents.

- **__post_init__**()
- **step**(current)
  - Advance one dt. Returns 1 if spike, 0 otherwise.
- **reset**()
  - Reset to default initial conditions.

---

## Module `neurons.models.booth_rinzel`

### Class `BoothRinzelNeuron`
Booth & Rinzel 1995 — bistable motoneuron, 2-compartment.

C dVs/dt = -I_Na(Vs) - I_K(Vs) - I_L(Vs) - gc*(Vs - Vd)/p + I/p
C dVd/dt = -I_Ca(Vd) - I_KCa(Vd) - I_L(Vd) - gc*(Vd - Vs)/(1-p)
dq/dt   = (q_inf(Vd) - q) / tau_q
dCa/dt  = -f * (alpha_Ca * I_Ca + k_Ca * Ca)

Reference: Booth, V. & Rinzel, J. (1995). J. Neurophysiol. 73:1934–1945.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.brainscales_adex`

### Class `BrainScaleSAdExNeuron`
BrainScaleS-2 — analog AdEx (1000x real-time). Schemmel 2010.

Reference: Schemmel, J. et al. (2010). Proc. ISCAS 2010: 1947–1950.

- **__post_init__**()
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
- **step**(i_ampa_ext, s_ampa_rec, s_nmda_rec, s_gaba)
  - Advance one timestep.
- **reset**()
- **get_state**()

---

## Module `neurons.models.butera_respiratory`

### Class `ButeraRespiratoryNeuron`
Butera, Rinzel & Smith 1999 pre-Botzinger respiratory neuron.

The maintained numerical contract is candidate-first RK4 over the published
three-state Model I ODE: membrane voltage ``v``, potassium activation ``n``,
and persistent sodium inactivation ``h_nap``. Invalid drive, parameters,
corrupted state, non-finite rates, or out-of-envelope candidates raise
before mutating state.

- **__post_init__**()
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

Reference: Cazelles, B., Courbage, M. & Rabinovich, M. (2001). Europhys. Lett. 56(4):504-509.

- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.
- **reset**()

---

## Module `neurons.models.cerebellar_basket_neuron`

### Class `CerebellarBasketNeuron`
Cerebellar basket cell with A-type and Ca-dependent K currents.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.cfc`

### Class `ClosedFormContinuousNeuron`
Hasani et al. 2022 — closed-form continuous-depth neuron (CfC).

Analytical solution of the LTC ODE between timesteps:
x(t+dt) = x(t)*exp(-dt/tau_eff) + f_target*(1 - exp(-dt/tau_eff))
where tau_eff and f_target depend on input.

Reference: Canavier, C.C. et al. (1993). Biophys. J. 65:2373–2382.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.chandelier_neuron`

### Class `ChandelierNeuron`
Axo-axonic fast-spiking interneuron with Kv1 and Kv3 currents.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.chay`

### Class `ChayNeuron`
Chay 1985 pancreatic beta-cell burster with guarded stiff integration.

Reference: Chay, T.R. (1985). Physica D 16:233-242.

- **step**(current)
  - Advance one timestep and return an upward-threshold spike flag.
- **reset**()

---

## Module `neurons.models.chay_keizer`

### Class `ChayKeizerNeuron`
Chay & Keizer 1983 pancreatic beta-cell minimal model (five-state burster).

The original five-dimensional model: membrane potential ``v``, the
Hodgkin-Huxley activation/inactivation gates ``m``/``h`` of the inward
calcium current, the delayed-rectifier potassium activation ``n``, and the
free cytosolic calcium concentration ``ca`` (the slow variable that packages
spikes into bursts). Calcium enters through the voltage-gated calcium channel
during the active phase, gradually activating a calcium-dependent potassium
conductance until it terminates the burst; calcium then decays through the
silent phase until the next burst begins. With the published parameters the
model produces square-wave bursts with a period of order ten to twenty
seconds and a cytosolic calcium oscillation of order one micromolar.

The conductances follow the Hodgkin-Huxley convention written as
``g (E_rev - V)`` (inward positive). The gate rate functions are the
Hodgkin-Huxley 1952 forms with the membrane potential shifted by ``v_prime``
for the calcium gates and ``v_star`` for the potassium gate, scaled by the
temperature factor ``phi``; calcium influx is the surface-to-volume scaled
calcium current minus a first-order pump removal.

Reference: Chay, T.R. & Keizer, J. (1983). Minimal model for membrane
oscillations in the pancreatic beta-cell. Biophys. J. 42:181-190.
DOI 10.1016/S0006-3495(83)84384-7. Parameters are the paper's Table I with
the burst calcium-removal rate of Fig. 1b; cross-checked against the Wolfram
Demonstrations reference implementation.

- **step**(current)
  - Advance one timestep and return an upward-threshold spike flag.
- **reset**()

---

## Module `neurons.models.chay_keizer_minimal`

### Class `ChayKeizerMinimalNeuron`
Reduced three-state Chay-Keizer pancreatic beta-cell burster.

The essential bursting dynamics of the original five-state Chay-Keizer model
reduced to three variables: membrane potential ``v``, the delayed-rectifier
potassium activation ``n``, and the free cytosolic calcium ``c`` (the slow
variable). The calcium-current activation is taken at its instantaneous
steady state, so the fast spikes are generated by the ``v``-``n`` subsystem
while calcium slowly accumulates through the active phase, activates a
calcium-dependent potassium conductance, and terminates the burst; calcium is
then pumped down through the silent phase. A constant ATP-sensitive potassium
conductance sets the excitability (the glucose handle).

Currents follow the convention ``g (V - E_rev)`` (outward positive) in the
physical units of the reference (picosiemens, femtofarads, femtoamperes,
millivolts, micromolar, milliseconds). With the published parameters the cell
bursts autonomously: fast spikes ride an active-phase plateau near -20 mV with
a silent phase near -65 mV, while cytosolic calcium traces a slow sawtooth.

Reference: Bertram, R., Marinelli, I., Fletcher, P.A., Satin, L.S. &
Sherman, A.S. (2023). Deconstructing the integrated oscillator model for
pancreatic beta-cells. Mathematical Biosciences 365:109085, Table 1
(DOI 10.1016/j.mbs.2023.109085); the reduced model of Chay & Keizer (1983),
Biophys. J. 42:181-189.

- **step**(current)
  - Advance one timestep and return an upward-threshold spike flag.
- **reset**()

---

## Module `neurons.models.chialvo_map`

### Class `ChialvoMapNeuron`
Chialvo 1995 — 2D discrete map neuron.

x&#91;n+1&#93; = x²·exp(y-x) + k + I
y&#91;n+1&#93; = a·y - b·x + c

Reference: Chialvo, D.R. (1995). Chaos, Solitons & Fractals 5:461–479.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.clif`

### Class `ComplementaryLIFNeuron`
Complementary LIF with separate positive and negative leaky paths.

The public spike contract is ternary: +1 for positive-threshold crossing,
-1 for negative-threshold crossing, and 0 for no spike.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.coba_lif`

### Class `COBALIFNeuron`
Conductance-based LIF with coupled RK4 synaptic conductance state.

C dV/dt = -g_L(V - E_L) - g_e(V - E_e) - g_i(V - E_i) + I
dg_e/dt = -g_e / tau_e, dg_i/dt = -g_i / tau_i.

Conductance injections are applied before integration. The full
``(v, g_e, g_i)`` candidate is advanced with RK4 and committed only after
finite-value and envelope checks pass.

- **__post_init__**()
- **step**(current, delta_ge, delta_gi)
  - Advance one candidate-first RK4 timestep.
- **reset**()
  - Restore the membrane to leak reversal and clear conductances.

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

- **__post_init__**()
- **p_open**(displacement)
  - Boltzmann activation of MET channels.
- **step**(displacement)
  - Step with basilar membrane displacement.
- **reset**()
  - Reset state to resting potential.
- **state**()
  - Return a compact state and parameter snapshot for reproducibility.

---

## Module `neurons.models.compte_wm`

### Class `CompteWMNeuron`
Compte et al. NMDA-based working-memory neuron with Mg2+ block.

- **__post_init__**()
- **step**(current, spike_in)
- **reset**()

---

## Module `neurons.models.connor_stevens`

### Class `ConnorStevensNeuron`
Connor-Stevens 1977 A-type potassium current model.

State variables are membrane voltage ``v`` plus sodium activation ``m``,
sodium inactivation ``h``, delayed-rectifier potassium activation ``n``,
A-type potassium activation ``a``, and A-type inactivation ``b``. One
public ``step`` advances the 1 ms macro-step with candidate-first RK4
sub-steps and commits only finite, physically bounded candidates.

- **__post_init__**()
- **step**(current)
  - Advance one macro-step and return an upward-threshold spike flag.
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` macro-steps, returning ``(v_trace, spikes)``.
- **reset**()

---

## Module `neurons.models.courage_nekorkin_map`

### Class `CourageNekorkinMapNeuron`
Courbage-Nekorkin-Vdovin 2007 discontinuous 2D spiking-bursting map.

x(n+1) = x(n) + F(x(n)) - y(n) - beta*H(x(n) - d) + I
y(n+1) = y(n) + eps*(x(n) - J)

with the piecewise-linear ``F`` and the Heaviside ``H`` of the module
docstring. Defaults sit in the published chaotic spiking-bursting regime.

Reference: Courbage, M., Nekorkin, V.I. & Vdovin, L.V. (2007).
"Chaotic oscillations in a map-based model of neural activity."
Chaos 17:043109 (arXiv:0712.2097), eqs. 3-6.

- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.
- **reset**()

---

## Module `neurons.models.dcn_neuron`

### Class `DCNNeuron`
Deep cerebellar nuclei neuron — main output of the cerebellum.

WB Na⁺/K⁺ core + T-type Ca²⁺ (rebound bursting), Ih (pacemaker),
persistent Na⁺ (subthreshold), Ca²⁺-dependent AHP. 7 currents total.

Reference: Llinás & Mühlethaler (1988) J Physiol 404:241;
Jahnsen (1986) J Physiol 372:129.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.de_schutter_purkinje`

### Class `DeSchutterPurkinjeNeuron`
Single-compartment Purkinje-cell conductance model after De Schutter & Bower.

The maintained Python model exposes the seven compact state variables
``(v, h_na, n_k, m_cap, h_cap, q_kca, ca)``. It is a compact point-neuron
approximation; use the audit index before treating it as the full
multi-compartment reconstruction from the original paper.

The production update is candidate-first RK4 over all seven states. Five
internal substeps are retained to preserve the existing model time base, but
the public step commits only after every substep candidate is finite. The
historical explicit Euler path remains available through
``integrator="baseline_euler"`` for regression comparisons.

Reference: De Schutter, E. & Bower, J.M. (1994). J. Neurophysiol. 71:375–400.

- **__post_init__**()
  - Validate the selected integration method before simulation.
- **step**(current)
  - Advance the compact conductance model and return a spike indicator.
- **reset**()
  - Restore voltage, gates, and calcium state to their defaults.

---

## Module `neurons.models.dendrify`

### Class `DendrifyNeuron`
Pagkalos, Chavlis & Poirazi 2023 — Dendrify active-dendrite 2-compartment model.

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
integrator : {"rk4", "baseline_euler"}
    Numerical integration path. ``"rk4"`` is the production default;
    ``"baseline_euler"`` preserves the historical dendrite-first Euler
    update as an explicit comparison path.

- **__post_init__**()
- **mg_block**(v)
  - Mg2+ block factor: B(V) = 1/(1 + &#91;Mg&#93;/3.57 * exp(-0.062*V)).
- **step**(i_soma, glutamate)
  - Step with somatic input current and dendritic glutamate.
- **reset**()
  - Reset state to resting potential.

---

## Module `neurons.models.destexhe_thalamic`

### Class `DestexheThalamicNeuron`
Destexhe et al. 1996 — thalamocortical relay with T-current and I_h.

6 ODEs: V, m_Na, h_Na, n_K, m_T, h_T (+ optional h-current).

Reference: Destexhe, Bal, McCormick & Sejnowski (1996). Ionic mechanisms
underlying synchronized oscillations and propagating waves in a model of
ferret thalamic slices. J Neurophysiol 76:2049-2070.

- **step**(current)
- **reset**()

---

## Module `neurons.models.direction_selective_rgc`

### Class `DirectionSelectiveRGC`
Direction-selective retinal ganglion cell.

Parameters are finite and physical: positive ``tau``, ``theta`` and ``dt``;
non-negative centre/surround weights; finite preferred direction and state.

- **__post_init__**()
- **new_on**(cls)
  - Create an On-centre cell.
- **new_off**(cls)
  - Create an Off-centre cell.
- **step_rf**(intensity, surround_mean)
  - Step with local intensity and surround mean intensity.
- **step**(current)
  - Simple step with no surround input.
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

``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``.

- **step**(i_syn)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` updates, returning ``(i_mem_trace, spikes)``.
- **reset**()

---

## Module `neurons.models.durstewitz_dopamine`

### Class `DurstewitzDopamineNeuron`
Durstewitz, Seamans & Sejnowski 2000 — PFC neuron with D1 modulation.

A minimal Hodgkin-Huxley prefrontal pyramidal cell with fast Na⁺
(instantaneous activation ``m_∞``, dynamic inactivation ``h_na``), a
delayed-rectifier K⁺ (``n_k``), an NMDA current with the Jahr & Stevens
(1990) Mg²⁺ block, and an ohmic leak — the three-state ``(V, h_na, n_k)``
system. D1 agonism (``d1_level`` in ``&#91;0, 1&#93;``) enhances NMDA
(``g_nmda_scale``), shifts the Na⁺ half-activation (``v_shift_na``), and
enhances K⁺ (``g_k_scale``), stabilising persistent up-states.

The production integrator is candidate-first RK4 over the three-state
system: each sub-step evaluates the full right-hand side from one consistent
state, forms the RK4 candidate, and commits it only once finite. The
historical hard-coded forward-Euler update — which advanced the gates from
the old voltage and then the voltage from the freshly updated gates, mixing
two inconsistent states — remains reachable only through the explicit
``integrator="baseline_euler"`` regression option, which now evaluates a
single consistent right-hand side.

Reference: Durstewitz, D., Seamans, J. K. & Sejnowski, T. J. (2000).
Dopamine-mediated stabilization of delay-period activity in a network model
of prefrontal cortex. J. Neurophysiol. 83:1733–1750.

- **__post_init__**()
- **mg_block**(v)
  - Jahr & Stevens (1990) Mg²⁺ block ``B(V) = 1/(1 + &#91;Mg&#93;/3.57 · exp(-0.062 V))``.
- **step**(current)
  - Advance the neuron by one ``dt`` step and report a threshold crossing.
- **reset**()
  - Restore the resting potential and gating defaults.

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

Reference: Fardet, T. & Levina, A. (2020). PLoS Comput. Biol. 16(12):e1008503.

- **__post_init__**()
  - Validate the energy-LIF state before first use.
- **step**(current)
  - Advance one exact-flow step and return `1` when a spike occurs.
- **reset**()
  - Restore membrane voltage and energy reserve to resting state.

---

## Module `neurons.models.ermentrout_kopell_map_neuron`

### Class `ErmentroutKopellMapNeuron`
Ermentrout-Kopell 1986 canonical Type I (theta neuron) map.

The canonical model for Type I (saddle-node) excitability. Phase
variable θ advances on a circle; spike occurs when θ crosses π.

θ(n+1) = θ(n) + dt · &#91;(1 - cos θ) + (1 + cos θ) · I&#93;

Reference: Ermentrout & Kopell (1986) SIAM J Appl Math 46:233–253.

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.
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

Membrane dynamics use the exact constant-current RC flow before evaluating
the finite-step escape hazard.

Reference: Gerstner, W. (2000). Neural Comput. 12:43–89.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.expif`

### Class `ExpIFNeuron`
Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003.

Reference: Fourcaud-Trocmé, N. et al. (2003). J. Neurosci. 23:11628–11640.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.fitzhugh_nagumo`

### Class `FitzHughNagumoNeuron`
FitzHugh-Nagumo 1961 two-state excitable-system model.

dv/dt = v - v^3 / 3 - w + I
dw/dt = epsilon * (v + a - b*w)

The production default is RK4 over the published two-state ODE. The
historical explicit-Euler path remains available only through the explicit
``baseline_euler`` integrator option for compatibility experiments.

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 steps from the current state, returning ``(trace, spikes)``.
- **reset**()

---

## Module `neurons.models.fitzhugh_rinzel`

### Class `FitzHughRinzelNeuron`
FitzHugh-Rinzel three-state qualitative bursting model.

dv/dt = v - v^3/3 - w + y + I
dw/dt = delta * (a + v - b*w)
dy/dt = mu * (c - v - d*y)

Runtime integration uses RK4 over the published three-state ODE with
current held constant for one step.

- **__post_init__**()
- **step**(current)
  - Advance the model by one RK4 step.
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 steps from the current state, returning ``(trace, spikes)``.
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

- **__post_init__**()
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

- **__post_init__**()
- **static_type**(cls)
  - Static gamma — bag2/chain intrafusal fibres (length-sensitive).
- **step**(drive)
- **reset**()

---

## Module `neurons.models.gamma_renewal`

### Class `GammaRenewalNeuron`
Gamma renewal process neuron. Keat et al. 2001.

ISI ~ Gamma(k, k/rate). Hazard h(t) evaluated at elapsed time
since last spike. P(spike in dt) = 1 - exp(-h(t)*dt).

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §7.4.

- **__post_init__**()
- **step**(rate_override)
- **reset**()

---

## Module `neurons.models.gated_lif`

### Class `GatedLIFNeuron`
Yao et al. 2022 NeurIPS — LIF with learnable gates.

Reference: Yao, M. et al. (2022). Proc. NeurIPS 35:19606–19618.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.gif_population`

### Class `GIFPopulationNeuron`
Generalized integrate-and-fire population neuron with escape-rate spiking.

The deterministic subthreshold flow solves the coupled linear membrane and
spike-history adaptation equations exactly over one fixed-current time step.
Spiking then follows the Mensi et al. escape-rate hazard with a bounded
Poisson interval probability.

Reference: Mensi, S. et al. (2012). J. Neurophysiol. 107:1756-1775.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.glif`

### Class `GLIFNeuron`
Allen Institute GLIF5 generalised leaky integrate-and-fire neuron.

The four dynamic states are advanced with candidate-first RK4 over the
continuous GLIF flow. Spike reset is applied only after the candidate is
finite and crosses the adaptive threshold.

Reference: Teeter, C. et al. (2018). Nat. Commun. 9:709.

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.
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

---

## Module `neurons.models.golomb_fs`

### Class `GolombFSNeuron`
Golomb et al. 2007 — fast-spiking cortical interneuron with a Kv3 current.

The membrane potential follows

``C dV/dt = -I_Na - I_Kd - I_Kv3 - I_L + I_ext``

with transient sodium ``I_Na = g_Na m_inf^3 h (V - E_Na)`` (instantaneous
activation), delayed-rectifier ``I_Kd = g_Kd n^4 (V - E_K)``, the fast
high-threshold ``I_Kv3 = g_Kv3 p^2 (V - E_K)`` that sharpens spikes and sustains
high-frequency firing, and an ohmic leak. All gating variables relax through
sigmoidal steady states, so the right-hand side has no removable singularities.

The production integrator is candidate-first RK4 over the four-state
``(V, h, n, p)`` system: each sub-step evaluates the full right-hand side from
one consistent state, forms the RK4 candidate, and commits it only once finite.
The historical hard-coded forward-Euler update — which staggered the gate and
membrane increments against mismatched states — remains reachable only through
the explicit ``integrator="baseline_euler"`` regression option.

Reference: Golomb, D., Donner, K., Shacham, L., Shlosberg, D., Amitai, Y. &
Hansel, D. (2007). Mechanisms of firing patterns in fast-spiking cortical
interneurons. J. Neurophysiol. 97:3831–3843.

- **__post_init__**()
- **step**(current)
  - Advance the neuron by one ``10 * dt`` step and report a threshold crossing.
- **reset**()
  - Restore the resting membrane potential and gating defaults.

---

## Module `neurons.models.granule_cell`

### Class `GranuleCell`
Cerebellar granule cell — D'Angelo et al. 2001 model.

Smallest and most numerous neuron in the brain. 7 ionic currents:
INa (m³h), IKdr (n⁴), IKA (a³b), ICaT (mT²s), IKCa (Hill), Ih (r),
plus tonic GABA. Ca²⁺ dynamics with KCa half-saturation.

Reference: D'Angelo et al. (2001) J Neurosci 21:759–770.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.gutkin_ermentrout`

### Class `GutkinErmentroutNeuron`
Gutkin-Ermentrout persistent-sodium conductance neuron.

The model keeps voltage ``v`` and delayed-rectifier activation ``n`` as
dynamic states. Persistent sodium activation is instantaneous through
``m_inf(v)``. The implementation advances the coupled ODE with a
candidate-first fourth-order Runge-Kutta step and commits the candidate
only when the complete numeric contract remains finite and biologically
bounded.

Parameters
----------
v:
    Membrane voltage state in the inherited normalized millivolt scale.
n:
    Delayed-rectifier potassium activation gate. Must remain in ``&#91;0, 1&#93;``.
g_na:
    Persistent sodium conductance. Must be non-negative.
g_k:
    Delayed-rectifier potassium conductance. Must be non-negative.
g_l:
    Leak conductance. Must be non-negative.
e_na:
    Sodium reversal potential.
e_k:
    Potassium reversal potential.
e_l:
    Leak reversal potential.
dt:
    Integration step. Must be positive and finite.
v_threshold:
    Upward voltage-crossing spike threshold.

Raises
------
ValueError
    If the initial parameters violate the finite-state, conductance, gate,
    or timestep contract.

Notes
-----
The historical SC-NeuroCore surface uses an implicit unit membrane
capacitance for this reduced model. Spike output is an event marker from
the threshold crossing; voltage is not reset by this model.

References
----------
Gutkin, B. S., & Ermentrout, G. B. (1998). Dynamics of membrane
excitability determine interspike interval variability: A link between
spike generation mechanisms and cortical spike train statistics. Neural
Computation, 10(5), 1047-1065.

- **__post_init__**()
  - Validate the initial state and parameters.
- **step**(current)
  - Advance one RK4 step under constant external current.
- **reset**()
  - Restore the documented default voltage and potassium gate.

---

## Module `neurons.models.hay_l5`

### Class `HayL5PyramidalNeuron`
Reduced Layer 5 thick-tufted pyramidal-cell model after Hay et al. 2011.

The maintained production surface is a compact three-compartment reduction
with soma, apical trunk, and apical tuft voltages plus six gates/calcium
state variables. It preserves the public dual-input API
``step(current_soma, current_tuft=0.0)`` while moving the default numerical
path to candidate-first RK4. The historical explicit Euler path remains
available through ``integrator="baseline_euler"`` for regression and
benchmark comparisons.

Reference: Hay, E. et al. (2011). PLoS Comput. Biol. 7:e1002107.

- **__post_init__**()
  - Validate the selected integrator and numeric configuration.
- **step**(current_soma, current_tuft)
  - Advance the model by one public step and return a spike indicator.
- **reset**()
  - Restore voltage, gate, and calcium state to documented defaults.

---

## Module `neurons.models.hill_tononi`

### Class `HillTononiNeuron`
Hill & Tononi 2005 — thalamocortical sleep/wake neuron.

A conductance-based cell with fast Na⁺ (instantaneous ``m_∞``, dynamic
inactivation ``h_na``), delayed-rectifier K⁺ (``n_k``), a hyperpolarisation-
activated ``Ih`` (``m_h``), a low-threshold T-type Ca²⁺ current (instantaneous
``m_T``, dynamic inactivation ``h_t``), a sodium-dependent K⁺ current
(``I_KNa`` gated by intracellular ``na_i`` through a Hill function), and an
ohmic leak — the six-state ``(V, h_na, n_k, m_h, h_t, na_i)`` system. The
intracellular sodium accumulates from the sodium current and is removed by a
saturating Na/K pump.

The production integrator is candidate-first RK4 over the six-state system:
each sub-step evaluates the full right-hand side from one consistent state,
forms the RK4 candidate, and commits it only once finite. The historical
hard-coded forward-Euler update — which advanced the gates from the old
voltage and then the voltage and sodium from the freshly updated gates,
mixing inconsistent states — remains reachable only through the explicit
``integrator="baseline_euler"`` regression option.

The sodium concentration is clamped to be non-negative once per committed
step, matching the original. The conductance powers use explicit
multiplication and the ``I_KNa`` Hill exponent ``3.5`` is evaluated as
``b·b·b·sqrt(b)`` (an IEEE-754 exact decomposition of ``b**3.5``) so the
Rust, Julia, Go, and Mojo kernels reproduce the trajectory bit-for-bit
rather than depending on a per-platform ``pow`` implementation.

Reference: Hill, S. & Tononi, G. (2005). Modeling sleep and wakefulness in
the thalamocortical system. J. Neurophysiol. 93:1671–1698.

- **__post_init__**()
- **step**(current)
  - Advance the neuron by one ``dt`` step and report a threshold crossing.
- **reset**()
  - Restore the resting potential, gating defaults, and sodium baseline.

---

## Module `neurons.models.hindmarsh_rose`

### Class `HindmarshRoseNeuron`
Hindmarsh-Rose 1984 — 3D chaotic bursting model.

dx/dt = y - x³ + bx² - z + I
dy/dt = 1 - 5x² - y
dz/dt = r(s(x - x_rest) - z)

Reference: Hindmarsh, J.L. & Rose, R.M. (1984). Proc. R. Soc. Lond. B 221:87–102.

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 steps from the current state, returning ``(trace, spikes)``.
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
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` updates, returning ``(v_trace, spikes)``.
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

- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.
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

- **__post_init__**()
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
- **step**(input_current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 steps from the current state, returning ``(trace, spikes)``.
- **reset_state**()
- **get_state**()

---

## Module `neurons.models.jansen_rit`

### Class `JansenRitUnit`
Jansen & Rit 1995 — neural mass model for EEG generation.

6 ODEs: 3 populations (pyramidal, excitatory, inhibitory) x 2 states.

Reference: Jansen, B.H. & Rit, V.G. (1995). Biol. Cybern. 73:357–366.

- **__post_init__**()
- **step**(p_ext)
- **reset**()

---

## Module `neurons.models.kilinc_bhatt_map_neuron`

### Class `KilincBhattMapNeuron`
Nagumo-Sato / Aihara sigmoid map neuron with a dynamic threshold.

Minimal 2D map with built-in spike frequency adaptation via a slow
threshold variable. Designed for efficient hardware implementation
while retaining biologically relevant dynamics.

x(n+1) = -x(n) + k · σ(4·(x(n) - θ(n))) + I
θ(n+1) = β · θ(n) + γ · H(x(n) - θ_spike)

where σ(z) = 1 / (1 + exp(-z)) and H() is Heaviside. This is the
Nagumo-Sato (1972) discrete-time neuron with an accumulated dynamic
threshold, using the Aihara, Takabe & Toyoda (1990) sigmoid firing in
place of the hard Heaviside so that spiking is graded rather than
all-or-nothing.

References: Nagumo & Sato (1972) Kybernetik 10:155-164;
Aihara, Takabe & Toyoda (1990) Phys Lett A 144:333-340.

- **step**(current)
- **reset**()

---

## Module `neurons.models.klif`

### Class `KLIFNeuron`
KLIF — LIF with learnable scaling factor k.

V&#91;t+1&#93; = alpha * V&#91;t&#93; + k * I; spike when V >= threshold.
The scaling factor k is a trainable parameter for SNN backprop.

Reference: Jiang, C. & Zhang, Y. (2024). Neural Comput. 36(8):1546–1564.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.lapicque`

### Class `LapicqueNeuron`
Lapicque 1907 — classical RC integrate-and-fire.

tau * dv/dt = -(v - v_rest) + R * I

Constant-current steps use the exact RC flow:
V(t + dt) = V_inf + (V(t) - V_inf) * exp(-dt / tau)
where V_inf = V_rest + R * I.

Reference: Lapicque, L. (1907). J. Physiol. Pathol. Gén. 9:620–635.

``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``.

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` updates, returning ``(v_trace, spikes)``.
- **reset**()

---

## Module `neurons.models.larter_breakspear`

### Class `LarterBreakspearNeuron`
Breakspear, Terry & Friston 2003 — neural mass with ion channels.

3 ODEs per node. Combines Wilson-Cowan population dynamics with
conductance-based ion channel kinetics for whole-brain modelling.
Used in The Virtual Brain (TVB) simulator.

Reference: Larter, R. et al. (1999). Chaos 9:795–804.; Breakspear, M. et al. (2003). Cereb. Cortex 13:189–202.

- **__post_init__**()
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
Fully parameterized learnable neuron.

V&#91;t+1&#93; = alpha * V&#91;t&#93; + beta * I&#91;t&#93; + gamma * f(V&#91;t&#93;)
where alpha, beta, gamma are trainable scalars and f is a
learnable activation (here sigmoid) — a trainable generalisation of
the simple threshold models fitted to cortical recordings by Jolivet et al.

Reference: Jolivet, Rauch, Lüscher & Gerstner (2006). Predicting spike
timing of neocortical pyramidal neurons by simple threshold models.
J Comput Neurosci 21:35-49.

- **__post_init__**()
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

Reference: Dieudonné & Dumoulin (2000). Serotonin-driven long-range
inhibitory connections in the cerebellar cortex. J Neurosci 20:1837-1848.

- **__post_init__**()
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

---

## Module `neurons.models.marder_stg`

### Class `MarderSTGNeuron`
Liu-Golowasch-Marder-Abbott 1998 stomatogastric ganglion neuron.

Single-compartment crustacean STG model with seven voltage-gated currents
(Na, CaT, CaS, A, KCa, Kd, H) plus a leak, in the Prinz/LGMA unit convention
(conductances in mS/cm², capacitance in µF/cm², calcium in µM, voltage in mV,
time in ms). All gates use the published voltage-dependent steady states and
time constants; the calcium reversal is computed from the Nernst equation,
and intracellular calcium relaxes towards rest with a 20 ms time constant
driven by the two calcium currents. The thirteen-state vector is integrated
with classical fourth-order Runge-Kutta.

Parameters
----------
v : float
    Membrane potential (mV).
m_na, h_na, m_cat, h_cat, m_cas, h_cas, m_a, h_a, m_kca, m_kd, m_h : float
    Gating variables in &#91;0, 1&#93;.
ca : float
    Intracellular calcium concentration (µM, ≥ 0).
cm : float
    Specific membrane capacitance (µF/cm²).
g_na, g_cat, g_cas, g_a, g_kca, g_kd, g_h, g_l : float
    Maximal conductances (mS/cm²).
e_na, e_k, e_h, e_l : float
    Reversal potentials (mV). The calcium reversal is Nernst-derived.
ca_out : float
    Extracellular calcium concentration (µM) for the Nernst equation.
ca_rest : float
    Resting calcium concentration (µM).
tau_ca : float
    Calcium relaxation time constant (ms).
f_ca : float
    Calcium-current-to-concentration coupling (µM·cm²/µA).
celsius : float
    Temperature (°C) for the Nernst equation.
dt : float
    Integration time step (ms).
v_threshold : float
    Voltage at which a spike is registered (mV).

References
----------
Liu, Z., Golowasch, J., Marder, E. & Abbott, L.F. (1998). A model neuron with
activity-dependent conductances regulated by multiple calcium sensors.
J. Neurosci. 18(7):2309–2320. Channel kinetics: ModelDB accession 93321.

- **__post_init__**()
- **step**(current)
  - Advance one ``dt`` and return 1 on an upward threshold crossing.
- **reset**()
  - Restore the resting initial condition.

---

## Module `neurons.models.martinotti_neuron`

### Class `MartinottiNeuron`
Martinotti cell — adapting interneuron targeting layer 1 apical dendrites.

Pospischil 2008 gating with fast Na⁺, delayed-rectifier K⁺, a very strong
M-current (Kv7) that gives the pronounced spike-frequency adaptation
characteristic of the type, a low-threshold T-type Ca²⁺ current for rebound,
and an ohmic leak — the six-state ``(V, m, h, n, p, s)`` system (T-type
activation ``m_T`` is instantaneous; unlike the SST cell there is no ``Ih``).

The sodium and potassium activation rates follow the Traub-Miles
``x/(exp(±x/k)-1)`` form and use the L'Hôpital limit at the removable
singularity. The β_m numerator is the published ``V - V_T - 40`` offset; an
earlier revision shared the ``-17`` offset of α_h, which lowered β_m enough to
drive the cell into depolarisation block (it fired two or three spikes then
stuck near threshold for any stimulus). The corrected kinetics restore a
monotone frequency-current relation.

The production integrator is candidate-first RK4 over the six-state system:
each sub-step evaluates the full right-hand side from one consistent state,
forms the RK4 candidate, and commits it only once finite. The historical
hard-coded forward-Euler update — which staggered the gate and membrane
increments against mismatched states — remains reachable only through the
explicit ``integrator="baseline_euler"`` regression option.

Reference: Silberberg, G. & Markram, H. (2007). Disynaptic inhibition between
neocortical pyramidal cells mediated by Martinotti cells. Neuron 53:735–746;
Toledo-Rodriguez et al. (2005); Pospischil et al. (2008) Biol. Cybern. 99:427.

- **__post_init__**()
- **step**(current)
  - Advance the neuron by one ``4 * dt`` step and report a threshold crossing.
- **reset**()
  - Restore the resting potential and gating defaults.

---

## Module `neurons.models.mat`

### Class `MATNeuron`
Kobayashi 2009 — Multi-timescale Adaptive Threshold.

Reference: Kobayashi, R. et al. (2009). Front. Comput. Neurosci. 3:9.

- **__post_init__**()
  - Validate the numerical contract before the first integration step.
- **step**(current)
  - Advance one MAT step and return `1` only when a spike occurs.
- **reset**()
  - Restore voltage and adaptive thresholds to the resting state.

---

## Module `neurons.models.mcculloch_pitts`

### Class `McCullochPittsNeuron`
McCulloch & Pitts 1943 — binary threshold neuron.

y = 1 if sum(w_i * x_i) >= theta, else 0.

Reference: McCulloch, W.S. & Pitts, W. (1943). Bull. Math. Biophys. 5:115–133.

- **__post_init__**()
- **step**(weighted_input)
- **reset**()

---

## Module `neurons.models.mckean`

### Class `McKeanNeuron`
McKean 1970 piecewise-linear FitzHugh-Nagumo caricature.

The model evolves the two-state ODE using candidate-first RK4 while
preserving the three piecewise-linear voltage branches of McKean's
analytically tractable Nagumo equation.

Reference: McKean, H.P. (1970). Advances in Mathematics, 4:209-223.

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.
- **reset**()

---

## Module `neurons.models.medvedev_map`

### Class `MedvedevMapNeuron`
Medvedev 2005 — 1D piecewise-monotone spiking map.

Reference: Medvedev, G.S. (2005). Physica D 202:37–59.

- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.
- **reset**()

---

## Module `neurons.models.mihalas_niebur`

### Class `MihalasNieburNeuron`
Mihalas-Niebur generalised integrate-and-fire neuron.

The continuous four-state flow is advanced with a candidate-first RK4
integrator. Spike reset is applied only after the continuous candidate is
finite and crosses the adaptive threshold.

Reference: Mihalas, S. & Niebur, E. (2009). Neural Comput. 21:704-718.

- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.
- **reset**()

---

## Module `neurons.models.morris_lecar`

### Class `MorrisLecarNeuron`
Morris-Lecar 1981 — calcium-potassium oscillator.

C dv/dt = -g_Ca m_∞(v)(v-E_Ca) - g_K w(v-E_K) - g_L(v-E_L) + I
dw/dt = λ(v)(w_∞(v) - w)

Reference: Morris, C. & Lecar, H. (1981). Biophys. J. 35:193–213.

Integrator options:
- ``rk4`` is the maintained default fourth-order path over the Morris-Lecar ODEs
- ``baseline_euler`` preserves the historical explicit-Euler path for comparison
- ``rosenbrock`` is a linearly implicit stiff-system path over the same
  conductance equations

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` updates, returning ``(v_trace, spikes)``.
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

The production update is candidate-first RK4 over the three coupled state
variables ``(u, v_basal, v_apical)``. Basal, apical, and somatic drives are
held constant across each RK4 stage; the stage apical voltage gates the
stage basal-to-soma coupling, so all derivatives are evaluated from one
consistent state. The historical explicit Euler path remains available only
through ``integrator="baseline_euler"`` for regression comparisons.

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
integrator : {"rk4", "baseline_euler"}
    Numerical integration path. Default: "rk4".

- **__post_init__**()
- **step_compartments**(x_basal, x_apical, i_soma)
  - Advance one step with basal, apical, and direct somatic drives.
- **step**(current)
  - Advance one step with ``current`` delivered to the basal dendrite.
- **reset**()
  - Reset soma, basal, and apical state to initial conditions.

---

## Module `neurons.models.neurogrid`

### Class `NeuroGridNeuron`
Boahen-style reduced Neurogrid two-compartment analog neuron.

The model couples a passive dendritic integrator to an EIF-like soma. The
production path uses candidate-first RK4 for the continuous two-state flow
and applies the discrete Neurogrid spike/reset rule once to the accepted
public-step candidate. The historical dendrite-first explicit Euler update
remains available through ``integrator="baseline_euler"`` for regression
comparisons.

Reference: Benjamin, B.V. et al. (2014). Proc. IEEE 102:699-716.

- **__post_init__**()
  - Validate the integrator choice and initial numeric configuration.
- **step**(current)
  - Advance the neuron by one public step and return a spike indicator.
- **reset**()
  - Restore soma and dendrite voltages to rest.

---

## Module `neurons.models.nlif`

### Class `NonlinearLIFNeuron`
Quadratic nonlinear LIF neuron with slow adaptation.

The membrane follows

``c_m dV/dt = a(V - v_rest)(V - v_crit) - w + I``

and the adaptation current follows

``tau_w dw/dt = b(V - v_rest) - w``.

The parameter validation is intentionally fail-closed: invalid geometry,
non-finite state, or unstable integration constants are rejected before any
state mutation can occur.

- **__post_init__**()
- **step**(current)
  - Advance one candidate-first RK4 step and return ``1`` on spike.
- **reset**()
  - Restore dynamic state without changing model parameters.

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

---

## Module `neurons.models.non_resetting_lif`

### Class `NonResettingLIFNeuron`
Adaptive multi-timescale threshold (aMAT) variant — non-resetting LIF.

tau_m dV/dt = -(V - V_rest) + R*I
On spike: threshold rises by delta_theta, V does NOT reset.
dtheta/dt  = -(theta - theta_rest) / tau_theta

Kobayashi et al. 2009, Jolivet et al. 2004.

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §1.3.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.perfect_integrator`

### Class `PerfectIntegratorNeuron`
Non-leaky integrate-and-fire. Lapicque 1907 (no leak).

dV/dt = I / C

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §1.3.

``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``
(prefer Rust under the factory-default contract when the engine is present).

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` updates, returning ``(v_trace, spikes)``.
- **reset**()

---

## Module `neurons.models.pernarowski`

### Class `PernarowskiNeuron`
Pernarowski 1994 pancreatic beta-cell burster.

Three coupled ODEs over ``(v, w, z)`` with one fast cubic state and
two slower recovery/adaptation variables. The public implementation uses
candidate-first RK4 integration and preserves continuous threshold-crossing
semantics without an artificial reset during normal evolution.

Reference: Pernarowski, M. (1994). SIAM J. Appl. Math. 54:814–832.

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.
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

---

## Module `neurons.models.pinsky_rinzel`

### Class `PinskyRinzelNeuron`
Pinsky-Rinzel 1994 two-compartment CA3 pyramidal cell.

Reduction of the 19-compartment Traub CA3 model to a soma compartment
carrying the fast Na⁺/delayed-rectifier K⁺ spike currents and a dendrite
compartment carrying the Ca²⁺, Ca-dependent K⁺ afterhyperpolarisation, and
voltage/Ca-dependent K⁺ currents, coupled by ``gc``. Eight states are
integrated with a fixed-step fourth-order Runge-Kutta scheme; the somatic
sodium activation ``m`` is taken at its instantaneous steady state ``m∞``.

The voltages use the physiological convention (rest ≈ −60 mV); reversal
potentials and gating rates equal the original rest=0 mV formulation shifted
by −60 mV. Parameters and kinetics follow the published model and the
ModelDB 35358 reference channels.

Parameters
----------
v_s, v_d : float
    Somatic and dendritic membrane potential (mV).
h, n, s, c, q : float
    Gating variables in &#91;0, 1&#93;: Na⁺ inactivation ``h``, delayed-rectifier
    activation ``n``, Ca²⁺ activation ``s``, voltage/Ca-dependent K⁺
    activation ``c``, and Ca-dependent afterhyperpolarisation ``q``.
ca : float
    Dimensionless dendritic calcium concentration (≥ 0).
cm : float
    Membrane capacitance (µF/cm²); default 3.0 per Pinsky & Rinzel (1994).
gc : float
    Soma-dendrite coupling conductance (mS/cm²).
p : float
    Somatic membrane-area fraction in (0, 1).
g_na, g_kdr, g_ca, g_kahp, g_kc, g_l : float
    Maximal conductances (mS/cm²) for the Na⁺, K-DR, Ca²⁺, K-AHP, K-C, and
    leak currents.
e_na, e_k, e_ca, e_l : float
    Reversal potentials (mV).
dt : float
    Integration time step (ms).
v_threshold : float
    Somatic voltage at which a spike is registered (mV).

References
----------
Pinsky, P.F. & Rinzel, J. (1994). Intrinsic and network rhythmogenesis in a
reduced Traub model for CA3 neurons. J. Comput. Neurosci. 1:39–60.
doi:10.1007/BF00962717. Reference channel kinetics: ModelDB accession 35358.

- **__post_init__**()
- **step**(current_soma, current_dend)
  - Advance one ``dt`` and return 1 on a rising somatic threshold crossing.
- **reset**()
  - Restore the published resting initial condition.

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

- **__post_init__**()
- **alpha**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.poisson`

### Class `PoissonNeuron`
Poisson spike generator — stochastic firing at rate λ.

P(spike in dt) = 1 - exp(-λ · dt). Essential for input layer generation.

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §7.2.

- **__post_init__**()
- **step**(rate_override)
- **reset**()

---

## Module `neurons.models.pospischil`

### Class `PospischilNeuron`
Pospischil et al. 2008 — minimal Hodgkin-Huxley cortical/thalamic neuron.

The membrane potential follows

``C dV/dt = -I_Na - I_Kd - I_M - I_L + I_ext``

with transient sodium ``I_Na = g_Na m^3 h (V - E_Na)``, delayed-rectifier
potassium ``I_Kd = g_Kd n^4 (V - E_K)``, the slow voltage-gated potassium
adaptation current ``I_M = g_M p (V - E_K)``, and an ohmic leak. The Traub-Miles
activation kinetics are written against the shifted potential ``V - V_T`` and the
``M`` current uses the Yamada ``p_inf``/``tau_p`` relaxation. The slow ``I_M``
conductance selects the firing class: ``g_M = 0.07`` regular spiking (default),
``g_M = 0`` fast spiking, ``g_M = 0.03`` intrinsically bursting.

The production integrator is candidate-first RK4 over the five-state
``(V, m, h, n, p)`` system: every sub-step evaluates the full right-hand side
from one consistent state, forms the RK4 candidate, and commits it only once it
is finite. The historical hard-coded forward-Euler update — which evaluated the
gates and the membrane against mismatched states — remains reachable only through
the explicit ``integrator="baseline_euler"`` option for regression comparison.

Reference: Pospischil, M. et al. (2008). Minimal Hodgkin-Huxley type models for
different classes of cortical and thalamic neurons. Biol. Cybern. 99:427–441.

- **__post_init__**()
- **step**(current)
  - Advance the neuron by one ``4 * dt`` step and report a threshold crossing.
- **reset**()
  - Restore the resting membrane potential and gating defaults.

---

## Module `neurons.models.prescott`

### Class `PrescottNeuron`
Prescott 2008 two-state excitability model with M-current tuning.

Reference: Prescott, S.A. et al. (2008). PLoS Comput. Biol. 4:e1000198.

- **__post_init__**()
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
PV+ (parvalbumin) fast-spiking interneuron (Wang-Buzsáki 1996 + Kv3.1).

The membrane potential follows

``C dV/dt = -I_Na - I_K - I_Kv3 - I_L + I_ext``

with transient sodium ``I_Na = g_Na m_inf^3 h (V - E_Na)`` (instantaneous
activation), delayed-rectifier ``I_K = g_K n^4 (V - E_K)``, the fast Kv3.1
current ``I_Kv3 = g_Kv3 p (V - E_K)`` that narrows the action potential and
sustains >200 Hz firing without adaptation, and an ohmic leak. The ``h`` and
``n`` gates carry the Wang-Buzsáki temperature factor ``phi``.

The production integrator is candidate-first RK4 over the four-state
``(V, h, n, p)`` system: each sub-step evaluates the full right-hand side from
one consistent state, forms the RK4 candidate, and commits it only once finite.
The historical hard-coded forward-Euler update — which staggered the gate and
membrane increments against mismatched states — remains reachable only through
the explicit ``integrator="baseline_euler"`` regression option.

Reference: Wang, X.-J. & Buzsáki, G. (1996). Gamma oscillation by synaptic
inhibition in a hippocampal interneuronal network model. J. Neurosci.
16:6402–6413.

- **__post_init__**()
- **step**(current)
  - Advance the neuron by one 0.5 ms step and report a threshold crossing.
- **reset**()
  - Restore the resting membrane potential and gating defaults.

---

## Module `neurons.models.quadratic_if`

### Class `QuadraticIFNeuron`
Quadratic Integrate-and-Fire — canonical Type-I excitability.

dv/dt = v² + I
Reset when v >= v_peak.

Reference: Latham, P.E. et al. (2000). J. Neurophysiol. 83:808–827.

``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``
(prefer Rust under the factory-default contract when the engine is present).

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` updates, returning ``(v_trace, spikes)``.
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
- **step_complex**(i_re, i_im)
  - Step with real and imaginary current components.
- **step**(current)
  - Step with real-only current (imaginary = 0).
- **reset**()
  - Reset state to initial conditions.

---

## Module `neurons.models.rall_cable`

### Class `RallCableNeuron`
Rall 1962 N-compartment passive cable with an implicit step.

The step solves the sealed-end passive cable operator as a tridiagonal
backward-Euler system with distal current held constant over ``dt``.
State is committed only after the finite candidate solve succeeds.

Reference: Rall, W. (1959). Exp. Neurol. 1:491–527.

- **__post_init__**()
- **step**(current)
  - Advance one implicit cable step and return the somatic spike flag.
- **reset**()
  - Reset all compartments to the leak reversal potential.

---

## Module `neurons.models.renshaw_cell`

### Class `RenshawCell`
Renshaw cell — spinal inhibitory interneuron for recurrent inhibition.

WB gating core with strong adaptation to produce burst-then-decay
response to motor axon collateral input.

Reference: Renshaw (1941); Windhorst (1996) Prog Neurobiol 46(5).

- **step**(current)
- **reset**()

---

## Module `neurons.models.resonate_and_fire`

### Class `ResonateAndFireNeuron`
- **__post_init__**()
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

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.
- **reset**()

---

## Module `neurons.models.sfa`

### Class `SFANeuron`
Benda-Herz spike-frequency adaptation IF with RK4 candidates.

Reference: Benda, J. & Herz, A.V.M. (2003). Neural Comput. 15:2523–2564.

The membrane and adaptation conductance state is advanced as a coupled
``(v, g_sfa)`` RK4 candidate. The candidate is committed only after finite
and envelope checks pass. A spike resets voltage and adds ``delta_g`` to
the RK4 adaptation candidate.

- **__post_init__**()
- **step**(current)
  - Advance one candidate-first RK4 timestep.
- **reset**()
  - Restore voltage to rest and clear adaptation conductance.

---

## Module `neurons.models.sherman_rinzel_keizer`

### Class `ShermanRinzelKeizerNeuron`
Sherman, Rinzel & Keizer 1988 reduced pancreatic beta-cell burster.

- **step**(current)
  - Advance one constant-current RK4 step and return threshold crossing.
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

- **__post_init__**()
- **step**(current)
  - Return instantaneous firing rate (Hz) for given mean input current.
- **reset**()

---

## Module `neurons.models.sigma_delta`

### Class `SigmaDeltaNeuron`
Yoon 2017 — event-driven sigma-delta encoding.

Reference: Yoon, Y. C. (2017). LIF and simplified SRM neurons encode
signals into spikes via a form of asynchronous pulse sigma-delta
modulation. IEEE Trans. Neural Netw. Learn. Syst. (DOI 10.1109/tnnls.2016.2526029).

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.sigmoid_rate`

### Class `SigmoidRateNeuron`
Continuous rate model with sigmoidal transfer. Wilson & Cowan 1972 style.

tau dr/dt = -r + sigma(beta * (input - theta))

Reference: Wilson, H.R. & Cowan, J.D. (1972). Biophys. J. 12:1–24.

- **__post_init__**()
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

---

## Module `neurons.models.spike_response`

### Class `SpikeResponseNeuron`
Spike Response Model (SRM0) — kernel-based, no ODEs.

v(t) = η(t - t_last) + Σ κ(t - t_in) · w
Spike when v(t) ≥ threshold.
Gerstner 1995.

Reference: Gerstner, W. (1995). Phys. Rev. E 51:738–758.

- **__post_init__**()
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
SpiNNaker LIF with exact constant-current membrane flow.

The SpiNNaker software LIF model evolves one membrane state ``v`` under a
constant input current plus tonic ``i_offset`` during each integration
interval. SC-NeuroCore evaluates that linear ODE analytically instead of
using forward Euler, while retaining the documented hard threshold, reset,
and absolute refractory timer semantics.

References
----------
Furber, S. B. et al. (2014). The SpiNNaker Project. Proceedings of the
IEEE, 102(5), 652-665.

- **__post_init__**()
  - Validate and normalise public scalar parameters.
- **step**(current)
  - Advance one exact-flow step and return a binary spike indicator.
- **reset**()
  - Restore voltage and refractory state to the documented rest state.

---

## Module `neurons.models.srm0`

### Class `SRM0Neuron`
Spike Response Model with exact constant-current kernel flow.

The zeroth-order Spike Response Model keeps membrane potential ``v`` and a
refractory afterhyperpolarisation kernel ``eta``. During one step, external
current is held constant and the coupled linear system for ``(v, eta)`` is
integrated analytically. This avoids the first-order membrane Euler error
while preserving the SRM0 threshold/reset semantics.

Parameters
----------
v:
    Membrane potential state.
v_rest:
    Rest potential and post-spike voltage reset.
v_threshold:
    Spike threshold. A spike is emitted when the exact-flow candidate
    reaches or exceeds this value.
tau_m:
    Membrane time constant. Must be positive and finite.
tau_eta:
    Refractory-kernel decay time constant. Must be positive and finite.
eta_reset:
    Positive refractory-kernel amplitude. On spike, ``eta`` is set to
    ``-eta_reset``.
resistance:
    Current-to-voltage gain.
dt:
    Integration step. Must be positive and finite.

Raises
------
TypeError
    If a scalar parameter is not numeric.
ValueError
    If a parameter is non-finite or a positive contract is violated.

References
----------
Gerstner, W., & Kistler, W. M. (2002). Spiking Neuron Models:
Single Neurons, Populations, Plasticity. Cambridge University Press,
chapter 4.

- **__post_init__**()
  - Initialise private kernel state after validating public parameters.
- **step**(current)
  - Advance one exact-flow SRM0 step.
- **reset**()
  - Restore voltage, refractory kernel, and internal clock state.
- **get_state**()
  - Return the current diagnostic SRM0 state.

---

## Module `neurons.models.sst_neuron`

### Class `SSTNeuron`
SST+ (somatostatin) low-threshold spiking interneuron.

Pospischil et al. 2008 LTS parameterisation: fast Na⁺ and delayed-rectifier
K⁺ for the spike, an M-current (Kv7) for spike-frequency adaptation, a
low-threshold T-type Ca²⁺ current for rebound bursting, ``Ih`` for the
hyperpolarisation sag, and an ohmic leak — the seven-state
``(V, m, h, n, p, s, r)`` system (T-type activation ``m_T`` is instantaneous).

The sodium and potassium activation rates follow the Traub-Miles
``x/(exp(±x/k)-1)`` form and use the L'Hôpital limit at the removable
singularity. The β_m numerator is the published ``V - V_T - 40`` offset; an
earlier revision shared the ``-17`` offset of α_h, which lowered β_m enough to
drive the cell into depolarisation block (it fired three spikes then stuck near
threshold for any stimulus). The corrected kinetics restore a monotone
frequency-current relation.

The production integrator is candidate-first RK4 over the seven-state system:
each sub-step evaluates the full right-hand side from one consistent state,
forms the RK4 candidate, and commits it only once finite. The historical
hard-coded forward-Euler update — which staggered the gate and membrane
increments against mismatched states — remains reachable only through the
explicit ``integrator="baseline_euler"`` regression option.

Reference: Pospischil, M. et al. (2008). Minimal Hodgkin-Huxley type models for
different classes of cortical and thalamic neurons. Biol. Cybern. 99:427–441.

- **__post_init__**()
- **step**(current)
  - Advance the neuron by one ``4 * dt`` step and report a threshold crossing.
- **reset**()
  - Restore the resting potential and gating defaults.

---

## Module `neurons.models.stellate_cell`

### Class `StellateCell`
Cerebellar stellate cell — fast-spiking molecular layer interneuron.

WB Na⁺/K⁺ core + Kv3.1 for narrow APs. Feedforward inhibition onto
Purkinje cell dendrites. Smaller than basket cells.

Reference: Sultan & Bower (1999) J Comp Neurol 409:63;
Häusser & Clark (1997) Neuron 19:665.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.stochastic_if`

### Class `StochasticIFNeuron`
Brunel & Hakim 1999 — Ornstein-Uhlenbeck driven IF.

Reference: Tuckwell, H.C. (1988). Introduction to Theoretical Neurobiology, Vol. 2. Cambridge Univ. Press.

- **__post_init__**()
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
Terman & Wang 1995 relaxation oscillator for LEGION networks.

The model evolves a fast excitatory variable ``v`` and slow recovery
variable ``w`` under the published cubic/sigmoid ODE. Runtime integration
uses candidate-first RK4 so invalid derivatives or candidates cannot poison
state.

Reference: Terman, D. & Wang, D.L. (1995). Physica D 81:148-176.

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.
- **reset**()

---

## Module `neurons.models.theta`

### Class `ThetaNeuron`
Theta neuron — canonical Type-I on the unit circle.

dθ/dt = (1 - cos θ) + (1 + cos θ) · I
Spike when θ crosses π.
Ermentrout & Kopell 1986.

Reference: Ermentrout, G.B. & Kopell, N. (1986). SIAM J. Appl. Math. 46:233–253.

``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``
(prefer Rust under the factory-default contract when the engine is present).

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` updates, returning ``(theta_trace, spikes)``.
- **reset**()

---

## Module `neurons.models.threshold_linear_rate`

### Class `ThresholdLinearRateNeuron`
Threshold-linear (ReLU) rate neuron. Dayan & Abbott 2001.

r = gain * max(0, input - theta)

Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §15.2.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.traub_miles`

### Class `TraubMilesNeuron`
Traub & Miles 1991 — reduced hippocampal CA3 pyramidal.

Reference: Traub, R.D. & Miles, R. (1991). Neuronal Networks of the Hippocampus. Cambridge Univ. Press.

- **__post_init__**()
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

---

## Module `neurons.models.unipolar_brush_cell`

### Class `UnipolarBrushCell`
Unipolar brush cell (UBC) — excitatory vestibular cerebellum interneuron.

LIF with slow NMDA-like persistent current that prolongs mossy fibre
bursts into sustained granule cell activation. Giant 1:1 synapse.

Reference: Mugnaini & Floris (1994) J Comp Neurol 339:174-180
(discovery/anatomy); Diana et al. (2007) J Neurosci 27:3823-3838
(bimodal firing physiology).

- **__post_init__**()
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

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.models.vip_neuron`

### Class `VIPNeuron`
VIP (vasoactive intestinal peptide) irregular-spiking interneuron.

The membrane potential follows

``C dV/dt = -I_Na - I_K - I_A - I_L + I_ext``

with transient sodium ``I_Na = g_Na m_inf^3 h (V - E_Na)`` (instantaneous
activation), delayed-rectifier ``I_K = g_K n^4 (V - E_K)``, a transient A-type
``I_A = g_A a^3 b (V - E_K)`` (Kv4) whose slow inactivation produces the firing
accommodation typical of the small high-input-resistance VIP soma, and an ohmic
leak. Every gating variable relaxes through a sigmoidal steady state, so the
right-hand side has no removable singularities.

The production integrator is candidate-first RK4 over the five-state
``(V, h, n, a, b)`` system: each sub-step evaluates the full right-hand side
from one consistent state, forms the RK4 candidate, and commits it only once
finite. The historical hard-coded forward-Euler update — which staggered the
gate and membrane increments against mismatched states — remains reachable only
through the explicit ``integrator="baseline_euler"`` regression option.

Reference: Porter, J.T., Cauli, B., Tsuzuki, K., Lambolez, B., Rossier, J. &
Audinat, E. (1998). Selective excitation of subtypes of neocortical interneurons
by nicotinic receptors. J. Neurosci. 19:5228–5235; Bhatt et al. (2019).

- **__post_init__**()
- **step**(current)
  - Advance the neuron by one ``4 * dt`` step and report a threshold crossing.
- **reset**()
  - Restore the resting membrane potential and gating defaults.

---

## Module `neurons.models.wang_buzsaki`

### Class `WangBuzsakiNeuron`
Wang-Buzsáki 1996 — fast-spiking GABAergic interneuron.

3 ODEs. Simplified HH with only Na + K delayed rectifier.
Designed for gamma (30-80 Hz) oscillation modelling.

Reference: Wang, X.-J. & Buzsáki, G. (1996). J. Neurosci. 16:6402–6413.

- **__post_init__**()
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

- **__post_init__**()
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

- **__post_init__**()
- **step**(ext_input)
- **reset**()

---

## Module `neurons.models.wilson_hr`

### Class `WilsonHRNeuron`
Wilson 1999 polynomial cortical model.

dV/dt = -(17.81 + 47.71*V + 32.63*V^2)*(V - 0.55) - 26*R*(V + 0.92) + I
dR/dt = (-R + 1.35*V + 1.03) / tau_R

The maintained production path advances the coupled (V, R) state with
candidate-first RK4 and commits only finite candidates. Spike detection is
a threshold event followed by Wilson-HR hard voltage reset.

- **__post_init__**()
- **step**(current)
- **simulate**(n_steps, current, backend)
  - Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.
- **reset**()

---

## Module `neurons.models.wong_wang`

### Class `WongWangUnit`
Wong & Wang 2006 — reduced decision-making attractor model.

Reference: Wong, K.-F. & Wang, X.-J. (2006). J. Neurosci. 26:1314–1328.

- **__post_init__**()
- **step**(stim1, stim2)
- **reset**()

---

## Module `neurons.models.yamada`

### Class `YamadaNeuron`
Yamada, Kashimori & Kambara 1989 — subcritical Hopf burster.

3 ODEs: V, n (fast K recovery), q (slow variable for bursting).
Exhibits square-wave bursting via slow modulation of a Hopf bifurcation.

Reference: Yamada, W.M. et al. (1989). In: Methods in Neuronal Modeling. MIT Press, pp. 97–133.

- **__post_init__**()
- **step**(current)
- **reset**()

---

## Module `neurons.reference_trace_contracts`

### Class `FeatureTolerance`
Absolute and relative tolerance for one scalar trace feature.

Parameters
----------
absolute:
    Absolute error allowed between simulated and reference feature values.
relative:
    Relative error allowed as a fraction of the reference value magnitude.

- **accepts**(actual, expected)
  - Return whether ``actual`` lies inside the configured tolerance.

### Class `ReferenceTraceProvenance`
Provenance attached to one reference trace.

Parameters
----------
kind:
    Reference class, for example ``analytic_closed_form``.
source:
    Human-readable source for the reference values.
equation:
    Equation or feature-generation statement used to derive the values.
citation:
    Optional publication or DOI string when the trace is external.


### Class `ReferenceTraceProtocol`
Simulation protocol for one reference trace.

Parameters
----------
dt:
    Simulation timestep in the units used by the model schema.
steps:
    Number of timesteps to execute.
inputs:
    Constant keyword inputs passed to ``UniversalNeuron.step``.
state_variables:
    State variables to record after each timestep.


### Class `ReferenceTraceSpec`
Committed reference-trace specification.

Parameters
----------
name:
    Stable corpus identifier.
schema_name:
    Bundled UniversalNeuron schema name.
runner:
    Production runner name. The v1 corpus supports ``universal_dsl``.
protocol:
    Deterministic simulation protocol.
provenance:
    Reference source and derivation metadata.
expected_features:
    Scalar reference features keyed by stable feature names.
tolerances:
    Per-feature tolerance map.


### Class `TraceSimulationResult`
Trace and feature values produced by the current implementation.

Parameters
----------
name:
    Reference-trace name that was simulated.
steps:
    Number of executed timesteps.
trace:
    Recorded state values after each timestep.
spikes:
    Spike indicator emitted by each timestep.
features:
    Scalar feature map extracted from ``trace`` and ``spikes``.


### Class `FeatureMismatch`
One failed scalar-feature comparison.

Parameters
----------
feature:
    Feature name whose value drifted.
expected:
    Reference feature value.
actual:
    Current simulation feature value.
tolerance:
    Tolerance that was applied.
absolute_error:
    Absolute difference between ``actual`` and ``expected``.


### Class `TraceValidationReport`
Validation report for one reference trace.

Parameters
----------
name:
    Reference-trace name.
passed:
    Whether every expected feature matched within tolerance.
simulation:
    Trace produced by the current implementation.
mismatches:
    Failed feature comparisons, empty on success.


---

## Module `neurons.reference_trace_io`

### Function `reference_trace_spec_from_payload(payload)`
Parse and validate a reference-trace corpus payload.

Parameters
----------
payload:
    JSON-compatible mapping using schema
    ``sc-neurocore.reference-trace.v1``.

Returns
-------
ReferenceTraceSpec
    Validated immutable reference-trace specification.

Raises
------
ValueError
    If any required field is missing, malformed, non-finite, or references
    an unsupported runner or bundled neuron schema.

### Function `list_reference_trace_specs()`
Return all committed reference-trace names.

Returns
-------
tuple&#91;str, ...&#93;
    Sorted stable identifiers for every JSON payload in the corpus.

### Function `load_reference_trace_spec(name)`
Load one committed reference-trace specification.

Parameters
----------
name:
    Stable corpus identifier.

Returns
-------
ReferenceTraceSpec
    Validated reference-trace specification.

Raises
------
ValueError
    If ``name`` is not present in the committed corpus.

---

## Module `neurons.reference_trace_runner`

### Function `simulate_reference_trace(spec_or_name)`
Run a reference-trace protocol through ``UniversalNeuron``.

Parameters
----------
spec_or_name:
    Reference-trace specification or committed corpus name.

Returns
-------
TraceSimulationResult
    Recorded state trace, spike sequence, and extracted feature map.

### Function `validate_reference_trace(name)`
Validate one committed reference trace by name.

Parameters
----------
name:
    Stable corpus identifier.

Returns
-------
TraceValidationReport
    Feature-level validation report.

### Function `validate_reference_trace_spec(spec)`
Validate one in-memory reference-trace specification.

Parameters
----------
spec:
    Reference-trace specification to simulate and compare.

Returns
-------
TraceValidationReport
    Feature-level validation report.

### Function `validate_all_reference_traces()`
Validate every committed reference trace.

Returns
-------
tuple&#91;TraceValidationReport, ...&#93;
    Sorted reports for the current corpus.

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
- **step**(input_current)
- **reset_state**()
- **get_state**()

---

## Module `neurons.schema_module_aliases`

### Function `schema_for_module(module)`
Return the schema stem for a models/ module stem (identity if unmapped).

### Function `module_for_schema(schema)`
Return the models/ module stem for a schema stem (identity if unmapped).

### Function `class_for_schema(schema)`
Return the descriptor class name for a schema stem, if known.

### Function `resolve_schema_join(schema)`
Return ``(module, class_name_or_none)`` for a schema stem.

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
- **science**()
  - Authored science layer (schema v2), empty for v1 schemas.
- **validation**()
  - Authored validation layer (schema v2), empty for v1 schemas.
- **provenance**()
  - Authored provenance layer (schema v2), empty for v1 schemas.
- **hints**()
  - Optional engineering hints (schema v2), empty for v1 schemas.
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

### Class `FoldedResourceMetrics`
Architectural resource summary of a folded (time-multiplexed) interconnect.

Quantifies what the shared-datapath fold buys versus the direct interconnect's
one-module-instance-per-neuron unrolling: one processing element per distinct
neuron type is reused across every neuron of that type (a per-type PE pool), with
per-neuron state held in BRAM, at the cost of ``cycles_per_tick`` cycles to advance
the whole network by one timestep.

Attributes
----------
neurons : int
    Total neurons sharing the datapath across all folded populations.
state_vars_per_neuron : int
    Widest per-neuron state-variable count across the folded types (the largest
    BRAM word = ``state_vars_per_neuron`` × data width). Equal to the single type's
    count for a homogeneous network.
pe_instances : int
    Physical processing elements instantiated: one per distinct neuron type. A
    single population, or several populations all of one type, share one PE.
shared_multipliers : int
    Multipliers in the shared weighted fan-in, summed over external-source columns
    across all populations and reused across each population's neurons. Spiking
    fan-in (recurrent or inter-population) is spike-gated and uses none.
state_ram_bits : int
    Total BRAM-backed neuron-state storage, in bits, summed over populations
    (each population contributes ``neurons`` × its type's state-var count × data width).
cycles_per_tick : int
    Clock cycles to advance the whole network by one timestep
    (``neurons`` process cycles + 1 commit cycle).
direct_neuron_instances : int
    Neuron module instances the direct interconnect would unroll (= ``neurons``);
    the count the fold collapses to ``pe_instances``.
populations : int
    Number of folded populations sharing the one sequencer and the global spike bus.
param_rom_bits : int
    Total per-neuron parameter-ROM storage, in bits, for heterogeneous populations
    (each contributes ``neurons`` × its count of per-neuron-varying parameters × data
    width). Zero for a network whose populations all have uniform parameters (the PE
    bakes them). The parameter-space analogue of ``state_ram_bits``.

- **as_dict**()
  - Return a deterministic plain-``int`` mapping for manifests/JSON.

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
    ``"direct"``, ``"aer"``, or ``"folded"`` (the time-multiplexed shared datapath).
folded_metrics : FoldedResourceMetrics | None
    Architectural fold resource summary when ``interconnect == "folded"``; ``None``
    for the direct/AER paths.
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
interconnect : str | None
    ``None`` (default) auto-selects direct (small) or AER (large) wiring;
    ``"direct"`` forces direct; ``"folded"`` opts into the time-multiplexed
    shared-datapath interconnect (one PE per neuron type + per-population BRAM
    state, swept by a single sequencer), which supports the :func:`_can_fold`
    subset: any number of populations with external-weighted, recurrent, or
    inter-population spiking fan-in.

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
params : dict&#91;str, np.ndarray&#91;Any, Any&#93;&#93;
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
weights : np.ndarray&#91;Any, Any&#93;
    Weight matrix of shape ``(n_dst, n_src)`` in float32.
    Row *i* contains the weights from all source neurons to
    destination neuron *i*.
bias : np.ndarray&#91;Any, Any&#93; | None
    Optional bias vector of shape ``(n_dst,)``.
delay_steps : int | tuple&#91;int, ...&#93;
    Number of explicit unit-delay timesteps on this connection.  A scalar
    applies to all source columns; a tuple carries one delay per source
    column for heterogeneous NIR ``Delay`` vectors.
source_threshold : np.ndarray&#91;Any, Any&#93; | None
    Optional threshold vector applied to source signals before the weight
    matrix.  Represents NIR ``Threshold`` on the source side.
destination_threshold : np.ndarray&#91;Any, Any&#93; | None
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
- **forward**(x)
- **reset**()

### Class `SCIFNode`
IF neuron — integrator with threshold, no leak.

NIR IF: dv/dt = R*I, spike when v > v_threshold
Euler: v += R*I*dt

- **from_nir**(cls, name, node, dt, reset_mode)
- **__post_init__**()
- **forward**(x)
- **reset**()

### Class `SCLINode`
Leaky integrator — LIF without threshold.

NIR LI: tau*dv/dt = (v_leak - v) + R*I

- **from_nir**(cls, name, node, dt)
- **__post_init__**()
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
- **forward**(x)
- **reset**()

### Class `SCCubaLINode`
Current-based leaky integrator (CubaLIF without threshold).

NIR CubaLI: tau_syn * dI_syn/dt = -I_syn + w_in * I
            tau_mem * dv/dt = (v_leak - v) + R * I_syn

- **from_nir**(cls, name, node, dt)
- **__post_init__**()
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

### Function `map_node(name, node)`
Convert a single NIR node to its SC-NeuroCore equivalent.

---

## Module `nir_bridge.parser`

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

### Function `build_silicon_mapping_report(source, config)`
Build a deterministic target-mapping report for a parsed NIR network.

### Function `write_silicon_mapping_report(output_dir, source, config)`
Write `nir_silicon_mapping_report.json` in deterministic form.

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
- **emit_lumerical_netlist**(ir_graph)
  - Emit a Lumerical-compatible photonic netlist from an IR graph.

### Class `OpticalModulation`
Optical modulation scheme.


### Class `PhotonicTarget`
Hardware target specification for a photonic backend.

- **lightmatter**(cls)
  - Return a Lightmatter-style photonic target profile.
- **silicon_photonics**(cls)
  - Return a generic silicon-photonics target profile.
- **two_d_waveguide**(cls)
  - Return a two-dimensional-material waveguide target profile.

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
  - Append one waveguide pair to the analyzer batch.
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

---

## Module `optimizer.resource_optimizer`

### Class `OptimizationStep`
One step in the optimization process.


### Class `OptimizationResult`
Result of the resource optimization process.

- **summary**()
  - Render a multi-line human-readable report of the optimization outcome.

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
- **optimize**(network)
  - Greedy knapsack optimization maximizing weighted accuracy.
- **optimize_annealing**(network)
  - Simulated annealing for larger design spaces.

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

### Class `SurrogateSCOptimizer`
Compiler optimiser using a learned surrogate over SC design points.

- **__init__**(target)
- **optimise**(network)
  - Select budgeted layer settings for ``network``.

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

- **__post_init__**()
- **evolve**(steps)
  - Applies a rewrite rule.
- **dimension_estimate**()
  - Estimate effective dimension via BFS neighborhood growth.

---

## Module `pipeline.ingestion`

### Class `MultimodalDataset`
Validated multimodal training dataset.

Parameters
----------
data:
    Mapping from modality names to normalized arrays. The first axis is the
    sample axis and must have the same length for every modality.
labels:
    Label array whose first axis matches the modality sample count.

- **__post_init__**()
  - Validate dataset shape invariants after construction.
- **get_sample**(idx)
  - Return the per-modality arrays for one sample index.

### Class `DataIngestor`
Normalize raw multimodal arrays into a `MultimodalDataset`.

Parameters
----------
label_key:
    Reserved key used to extract labels from the raw input mapping.

- **__init__**(label_key)
  - Initialize the ingestor with the reserved label key.
- **prepare_dataset**(raw_data)
  - Normalize and package raw multimodal data.

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
  - Validate consent identity, legal basis, telemetry flag, and token fields.
- **to_dict**()
  - Return this consent boundary as a deterministic mapping.
- **from_dict**(cls, data)
  - Build a consent boundary from a manifest section.

### Class `RetentionPolicy`
Retention windows for neural telemetry and artefacts.

- **__post_init__**()
  - Validate retention windows and enforce the maximum retention horizon.
- **to_dict**()
  - Return this retention policy as a deterministic mapping.
- **from_dict**(cls, data)
  - Build a retention policy from a manifest section.

### Class `RedactionPolicy`
Field-level redaction policy for protected telemetry and logs.

- **__post_init__**()
  - Validate redaction activation, field list, and replacement marker.
- **to_dict**()
  - Return this redaction policy as a deterministic mapping.
- **from_dict**(cls, data)
  - Build a redaction policy from a manifest section.

### Class `TelemetryPolicy`
Telemetry sink and sampling policy.

- **__post_init__**()
  - Validate telemetry activation, sink name, and sampling interval.
- **to_dict**()
  - Return this telemetry policy as a deterministic mapping.
- **from_dict**(cls, data)
  - Build a telemetry policy from a manifest section.

### Class `ProvenanceRecord`
Cryptographic provenance record for model and dataset artefacts.

- **__post_init__**()
  - Validate provenance artefact identity, hash, and source-system fields.
- **to_dict**()
  - Return this provenance record as a deterministic mapping.
- **from_dict**(cls, data)
  - Build a provenance record from one manifest list item.

### Class `IntegratorResponsibility`
Operational responsibilities for an integrator in a deployment pipeline.

- **__post_init__**()
  - Validate integrator contact details and approval responsibility fields.
- **to_dict**()
  - Return this integrator responsibility record as a deterministic mapping.
- **from_dict**(cls, data)
  - Build an integrator responsibility record from a manifest section.

### Class `PrivacyFeatureFlags`
Feature activation and audit flags for a governed workflow.

- **__post_init__**()
  - Validate feature toggles and the audit-flag activation contract.
- **to_dict**()
  - Return this privacy feature flag set as a deterministic mapping.
- **from_dict**(cls, data)
  - Build privacy feature flags from a manifest section.

### Class `GovernanceContract`
Full privacy governance contract for BCI/neural workflows.

- **__post_init__**()
  - Enforce cross-section privacy governance invariants.
- **audit_required_features**()
  - Return sorted feature keys that require audit flags.
- **active_features**()
  - Return names of enabled privacy features in deterministic order.
- **to_dict**()
  - Return a deterministic JSON-serialisable representation.
- **from_dict**(cls, data)
  - Build a full governance contract from a manifest mapping.

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

---

## Module `qat.lsq`

### Class `LSQQuantizer`
Learned-step-size fake quantiser for signed weights.

Parameters
----------
n_bits : int
    Quantiser bit width (``>= 2``). The signed grid is
    ``&#91;-2**(n_bits-1), 2**(n_bits-1) - 1&#93;``.
per_channel : bool
    Learn one step per channel along ``ch_axis`` instead of a single
    scalar step.
ch_axis : int
    Channel axis used when ``per_channel`` is set.
num_channels : int, optional
    Channel count; required when ``per_channel`` is set.

Attributes
----------
step : torch.nn.Parameter
    The learned step size(s). Lazily initialised from the first input.

- **__init__**(n_bits)
- **forward**(x)
  - Fake-quantise ``x`` at the learned step, with the LSQ gradient.
- **integer_weights**(x)
  - Return integer codes and the step(s) for hardware export.

### Class `LSQLinear`
Linear layer whose weights are quantised by a learned step size.

Per-channel (per output neuron) quantisation is the default, matching the
granularity that recovers most of the accuracy lost to low-bit weights.

Parameters
----------
in_features, out_features : int
    Layer dimensions.
n_bits : int
    Weight quantiser bit width.
per_channel : bool
    Learn one step per output neuron (default) or a single scalar step.
bias : bool
    Whether to include a full-precision bias.

- **__init__**(in_features, out_features)
- **forward**(x)
  - Apply the layer with LSQ-quantised weights.
- **export_quantized**()
  - Export integer weights, the learned step(s), and the bias.

---

## Module `qat.observers`

### Class `MinMaxObserver`
Per-tensor running min/max range observer.

Parameters
----------
n_bits : int
    Quantiser bit width the derived scale targets.
symmetric : bool
    Use a symmetric (zero-centred) mapping — the default for weights.
unsigned : bool
    Target an unsigned integer grid (e.g. non-negative activations).
eps : float
    Scale floor guarding against a zero-width observed range.

- **__init__**(n_bits)
- **observe**(x)
  - Fold ``x`` into the running range and return it unchanged.
- **calculate_qparams**()
  - Return the ``(scale, zero_point)`` for the observed range.
- **quantize**(x)
  - Fake-quantise ``x`` with the currently observed scale.

### Class `PerChannelMinMaxObserver`
Per-channel running min/max range observer.

Tracks an independent min/max — and therefore an independent scale — for
every slice along ``ch_axis``. For a weight tensor shaped
``(out_features, in_features)`` the default ``ch_axis=0`` yields one scale
per output neuron.

Parameters
----------
n_bits : int
    Quantiser bit width the derived scales target.
ch_axis : int
    Axis whose length is the channel count.
symmetric : bool
    Use a symmetric (zero-centred) mapping — the default for weights.
unsigned : bool
    Target an unsigned integer grid.
eps : float
    Scale floor guarding against a zero-width observed range.

- **__init__**(n_bits)
- **observe**(x)
  - Fold ``x`` into the running per-channel range and return it unchanged.
- **calculate_qparams**()
  - Return the per-channel ``(scale, zero_point)`` vectors.
- **quantize**(x)
  - Fake-quantise ``x`` with the observed per-channel scales.

### Function `fake_quantize(x, scale, zero_point)`
Quantise then de-quantise ``x`` (simulated quantisation, no STE).

This is the inference-time / calibration-time fake-quant used to evaluate
an observer's scales; for training use the learned-step quantisers in
:mod:`sc_neurocore.qat.lsq`. ``scale`` and ``zero_point`` broadcast against
``x``, so per-channel parameters must already be reshaped onto the channel
axis by the caller.

Parameters
----------
x : torch.Tensor
    Tensor to fake-quantise.
scale, zero_point : torch.Tensor
    Quantiser parameters, broadcastable to ``x``.
n_bits : int
    Quantiser bit width.
unsigned : bool
    Whether the integer grid is unsigned.

Returns
-------
torch.Tensor
    The de-quantised approximation of ``x`` on the integer grid.

---

## Module `qat.pact`

### Class `PACTActivation`
Parameterised clipping activation with uniform quantisation.

Parameters
----------
n_bits : int
    Activation quantiser bit width (``>= 2``). The activation grid has
    ``2**n_bits - 1`` positive levels over ``&#91;0, alpha&#93;``.
alpha_init : float
    Initial clipping bound.

Attributes
----------
alpha : torch.nn.Parameter
    The learned clipping bound.

- **__init__**(n_bits, alpha_init)
- **forward**(x)
  - Clip ``x`` to ``&#91;0, alpha&#93;`` and quantise to ``n_bits`` levels.
- **quantize**(x)
  - Return integer activation codes and the scale for export.
- **extra_repr**()
  - Return the compact module representation.

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

### Class `LSQPACTLIFNet`
Feedforward SNN with LSQ weights and a PACT-quantised analogue input.

Combines the learned-step per-channel weight quantiser
(:class:`~sc_neurocore.qat.lsq.LSQLinear`) with the parameterised-clipping
activation quantiser (:class:`~sc_neurocore.qat.pact.PACTActivation`): the
continuous input current is clipped and quantised by a learned bound before
the first layer, and every dense layer's weights carry an independently
learned per-output-neuron step size. Inter-layer signals are binary spikes,
so only the input needs activation quantisation.

Parameters
----------
n_input, n_hidden, n_output : int
    Layer dimensions.
n_layers : int
    Number of hidden layers.
weight_bits : int
    Bit width of the LSQ weight quantiser.
act_bits : int
    Bit width of the PACT input-activation quantiser.
beta : float
    LIF membrane decay.
surrogate_fn : Callable
    Surrogate-gradient spike function.
per_channel : bool
    Learn a per-output-neuron weight step (default) or a single scalar step.

Example
-------
>>> net = LSQPACTLIFNet(784, 128, 10, weight_bits=4, act_bits=4)
>>> x = torch.randn(25, 32, 784)  # (T, batch, features)
>>> spikes, mem = net(x)
>>> spikes.shape
torch.Size(&#91;32, 10&#93;)

- **__init__**(n_input, n_hidden, n_output, n_layers, weight_bits, act_bits, beta, surrogate_fn, per_channel)
- **forward**(x)
  - x: (T, batch, n_input). Returns (spike_counts, membrane_acc).
- **export_quantized**()
  - Export LSQ integer weights per layer plus the PACT input scale.

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
- **encode**(bitstream)
  - Encode logical bitstream into surface code physical qubits.
- **measure_syndrome**(physical_bits)
  - Measure X and Z stabilizer syndromes.
- **decode**(physical_bits)
  - Decode surface code: measure syndromes, correct single-qubit errors, majority vote.
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

### Function `cmd_learn(args)`
Run one bounded learning pass over a repository.

Parameters
----------
args
    Parsed ``learn`` sub-command arguments, including the repository path,
    state-file path, model override, and SNN stimulus directory.

Returns
-------
int
    Process-style exit code, ``0`` after the learning pass and state write
    complete.

### Function `cmd_daemon(args)`
Run the continuous repository-learning daemon.

Parameters
----------
args
    Parsed ``daemon`` sub-command arguments, including cycle size, sleep
    interval, state-file path, optional dashboard flag, and stimulus directory.

Returns
-------
int
    Process-style exit code, ``0`` after graceful shutdown and final state
    persistence.

### Function `cmd_status(args)`
Print a saved learning-state summary.

Parameters
----------
args
    Parsed ``status`` sub-command arguments containing the state-file path.

Returns
-------
int
    ``0`` when a readable state file is summarised, ``1`` when the file is
    absent, empty, or unreadable.

### Function `main(argv)`
Dispatch the quantum-cognition CLI.

Parameters
----------
argv
    Optional argument vector. ``None`` uses ``sys.argv`` through
    :mod:`argparse`.

Returns
-------
int
    Process-style exit code from the selected sub-command, or ``0`` after
    printing help when no sub-command is provided.

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
- **backend**()
  - Active backend name.
- **execute_non_local_sync**(entangle_pairs)
  - Execute non-local synchronisation via entanglement.
- **execute_posner_circuit**(shots)
  - Dispatch an actual 8q Posner Hamiltonian circuit to IBM QPU.
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
np.ndarray&#91;Any, Any&#93;
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
tuple&#91;np.ndarray&#91;Any, Any&#93;, dict&#91;str, int&#93;&#93;
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
  - Return a concise representation of the dashboard window size.

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
  - Assign a finite membrane voltage through the Population alias.
- **__repr__**()
  - Return compact debug telemetry for the coupled neuron.

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
  - Return the inner neuron's membrane voltage.
- **v**(value)
  - Assign a finite membrane voltage through the wrapper alias.
- **v_threshold**()
  - Return the inner neuron's spike threshold.
- **v_rest**()
  - Return the inner neuron's resting membrane potential.
- **get_state**()
  - Return the inner neuron's checkpointable state.
- **reset**()
  - Reset the wrapped neuron state for NeuronProtocol compatibility.
- **reset_state**()
  - Reset the wrapped neuron state to rest and full ATP.
- **__repr__**()
  - Return compact debug telemetry for the population wrapper.

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
  - Return a concise representation of the brain learning state.

---

## Module `quantum_cognition.kane_mapper`

### Class `KaneRegisterLayout`
Physical layout of a Si:P qubit register.

Attributes
----------
n_qubits : int
    Number of ³¹P donor qubits.
qubit_positions : np.ndarray&#91;Any, Any&#93;
    Shape ``(n_qubits, 2)`` — (x, y) coordinates in nanometres.
coupling_matrix : np.ndarray&#91;Any, Any&#93;
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
- **get_constraints**(n_sites)
  - Return design constraints for a register of given size.
- **__repr__**()
  - Return a concise constructor-style mapper description.

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
hyperfine_tensors_1, hyperfine_tensors_2 : list&#91;np.ndarray&#91;Any, Any&#93;&#93;
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
- **from_hyperfine_tensors**(cls)
  - Construct an exact RPM model from explicit 3×3 hyperfine tensors.
- **singlet_yield**(b_local)
  - Compute singlet yield Φ_S for the current parameters.
- **singlet_yield_field_sweep**(b_range)
  - Compute singlet yield over a range of magnetic fields.
- **atp_efficiency**(b_local, entanglement_boost)
  - Compute ATP hydrolysis efficiency from singlet yield.
- **get_state**()
  - Return parameters as a dict for serialisation.
- **__repr__**()
  - Return a concise representation of the active RPM parameters.

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

Simulates entangled :sup:`31`\\ P nuclear spins in Posner molecules.
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
- **to_statevector**()
  - Return the exact statevector represented by this MPS.
- **set_statevector**(statevector)
  - Load a statevector into MPS form without silent truncation.
- **evolve_exact**(couplings, time_us)
  - Evolve under explicit two-spin coupling tensors.
- **apply_measurement**(site_idx, intensity)
  - Apply a Born-rule projective measurement at one spin site.
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
  - Return a concise diagnostic summary for logs and dashboards.

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
  - Return debug representation with observed pool and bridge handles.

---

## Module `quantum_cognition.tests.test_dashboard`

### Class `FakeNeuron`
Minimal neuron telemetry record consumed by the dashboard renderer.


### Class `FakePool`
Minimal spin-pool telemetry record consumed by the dashboard renderer.

- **__init__**(entanglement_map)
  - Store a deterministic entanglement map for dashboard rendering.

### Class `FakeBrain`
Small dashboard-compatible brain fixture exercising the public draw path.

- **__init__**(history)
  - Initialise deterministic dashboard telemetry.
- **get_learning_state**()
  - Return the state dictionary read by the dashboard header.
- **get_history**()
  - Return spike and directive history entries for raster rendering.

### Function `test_dashboard_draw_covers_terminal_fallback_and_history_bands(monkeypatch, capsys)`
Draw uses fallback terminal size and renders all spike-raster bands.

### Function `test_dashboard_draw_reports_hidden_neuron_count(monkeypatch, capsys)`
Draw reports hidden neuron counts when the terminal is narrow.

### Function `test_bar_handles_non_positive_scale()`
Bar renderer returns an empty-scale bar for non-positive maxima.

### Function `test_dashboard_repr_reports_raster_window()`
The dashboard repr includes the configured raster window.

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

## Module `quantum_cognition.tests.test_gotm_brain_llm_contracts`

### Function `test_gotm_brain_import_detects_local_llm(monkeypatch)`
Importing with a local llm module enables the LLM directive path.

### Function `test_gotm_brain_invalid_llm_directive_falls_back(monkeypatch)`
Unknown local LLM replies fall back to the stable directive.

### Function `test_gotm_brain_llm_exception_falls_back(monkeypatch)`
Local LLM runtime errors fall back to the stable directive.

---

## Module `quantum_cognition.tests.test_kane_mapper`

### Class `TestKaneSiliconMapper`
Contract tests for the Kane silicon register mapper.

- **test_linear_positions**()
  - Linear topology should place qubits in a line.
- **test_grid_positions**()
  - Grid topology should place qubits in a 2D arrangement.
- **test_triangular_positions_stagger_odd_rows**()
  - Triangular topology staggers every odd row by half a spacing.
- **test_hexagonal_positions_stop_after_requested_site_count**()
  - Hexagonal topology fills only the requested number of donor sites.
- **test_coupling_matrix_symmetry**()
  - Coupling matrix must be symmetric with zero diagonal.
- **test_coupling_decay**()
  - Coupling should decay with distance.
- **test_coupling_positive**()
  - All coupling values should be non-negative.
- **test_exchange_coupling_returns_prefactor_at_zero_distance**()
  - Co-located donors use the Kane exchange prefactor directly.
- **test_t2_budget**()
  - T₂ budget must be positive.
- **test_single_qubit**()
  - Single qubit register should work.
- **test_zero_sites_are_rejected**()
  - Zero-site registers are invalid because no donor can be placed.
- **test_constraints**()
  - Constraints dict should contain all expected fields.
- **test_constraints_infeasible**()
  - Wide spacing should be infeasible (coupling too weak).
- **test_serialisation**()
  - to_dict should produce JSON-compatible output.
- **test_invalid_spacing**()
  - Non-positive donor spacing is rejected.
- **test_invalid_topology**()
  - Unknown lattice topology names are rejected.
- **test_repr**()
  - The repr reports spacing and topology for diagnostics.

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

## Module `quantum_cognition.tests.test_radical_pair_contracts`

### Function `test_rejects_invalid_quadrature_order()`
Constructor rejects quadrature rules that cannot integrate an interval.

### Function `test_explicit_hyperfine_tensor_state_counts()`
Explicit tensor construction records nuclei on both radicals.

### Function `test_invalid_hyperfine_tensor_shape_is_rejected()`
Tensor validation reports the exact radical-side tensor index.

### Function `test_zero_nucleus_singlet_density_has_electron_dimension()`
The no-bath helper returns pure electron singlet density matrices.

### Function `test_dense_hamiltonian_rejects_oversized_nuclear_bath()`
Dense exact evolution fail-closes before allocating huge spin operators.

### Function `test_singlet_yield_rejects_non_positive_rates(params, message)`
Singlet-yield validation rejects non-positive kinetic parameters.

---

## Module `quantum_cognition.tests.test_spin_pool`

### Class `TestSpinPoolMPS`
Contract tests for the exact spin-pool MPS publication path.

- **test_init_defaults**()
  - Default construction creates an eight-site product-state pool.
- **test_entanglement_map_normalised**()
  - Entanglement map should sum to 1 after init.
- **test_measurement_updates_map**()
  - Measurement at a site should shift entanglement towards that site.
- **test_normalisation_preserved**()
  - Entanglement map should remain normalised after measurements.
- **test_atp_efficiency_range**()
  - ATP efficiency is a singlet probability in &#91;0, 1&#93;.
- **test_invalid_site_index**()
  - Measurement rejects negative and out-of-range spin-site indices.
- **test_negative_intensity**()
  - Measurement intensity must be non-negative.
- **test_reset**()
  - Reset restores the product state and clears measurement metadata.
- **test_state_roundtrip**()
  - get_state → set_state should preserve state.
- **test_scpn_payload**()
  - SCPN payload exposes entanglement and ATP observable arrays.
- **test_invalid_params**()
  - Constructor rejects invalid site, bond, and update parameters.
- **test_singlet_statevector_roundtrip_preserves_atp_observable**()
  - Singlet import/export preserves the ATP singlet observable.
- **test_statevector_import_rejects_invalid_public_contracts**()
  - Statevector import/export rejects invalid public contracts.
- **test_statevector_import_rejects_silent_bond_truncation**()
  - Statevector import rejects states exceeding configured bond rank.
- **test_checkpoint_without_map_recomputes_quantum_diagnostics**()
  - Checkpoint restore recomputes diagnostics when the map is absent.
- **test_corrupted_checkpoint_zero_norm_fails_on_statevector_export**()
  - Corrupted zero-norm checkpoint tensors fail on exact export.
- **test_single_site_pool_rejects_two_site_atp_observable**()
  - A one-site pool cannot expose the two-site ATP observable.
- **test_atp_observable_rejects_out_of_range_public_site**()
  - ATP observable rejects public site indices outside the spin pool.
- **test_exact_evolution_rejects_invalid_coupling_contracts**()
  - Exact dense evolution rejects invalid time, size, and couplings.
- **test_exact_evolution_preserves_norm_for_valid_public_coupling**()
  - Exact dense evolution accepts a valid coupling tensor and stays unitary.
- **test_status_and_repr_report_public_telemetry**()
  - Status and repr expose stable telemetry fields for dashboards.
- **test_internal_heisenberg_same_site_is_noop**()
  - Internal same-site Heisenberg routing leaves the state unchanged.
- **test_internal_tebd_gate_rejects_non_adjacent_sites**()
  - Internal TEBD gate rejects non-adjacent site pairs explicitly.
- **test_internal_heisenberg_adjacent_gate_preserves_state_norm**()
  - Internal adjacent Heisenberg routing preserves state norm.
- **test_internal_heisenberg_nonadjacent_swap_network_preserves_state_norm**()
  - Internal non-adjacent Heisenberg routing preserves state norm.

---

## Module `recorders.spike_recorder`

### Class `BitstreamSpikeRecorder`
Record a binary spike train and compute basic spike statistics.

Parameters
----------
dt_ms:
    Duration represented by one recorded sample in milliseconds. A value of
    ``0.0`` is accepted for legacy zero-duration dry runs and produces a
    firing rate of ``0.0``.
spikes:
    Existing spike samples to seed the recorder with. Values must be binary
    integers where ``1`` means a spike occurred at that sample.

- **__post_init__**()
  - Validate seeded recorder state.
- **record**(spike)
  - Append one binary spike sample.
- **reset**()
  - Remove all recorded spike samples while preserving ``dt_ms``.
- **as_array**()
  - Return the recorded spike train as a NumPy ``uint8`` array.
- **total_spikes**()
  - Return the number of recorded spike samples equal to ``1``.
- **firing_rate_hz**()
  - Return the mean firing rate in hertz.
- **isi_histogram**(bins)
  - Compute a histogram of inter-spike intervals in milliseconds.

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
  - Reset the block membrane potential to zero.

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
  - Reset the block membrane potential to zero.

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
  - Propagate the input through every residual block in sequence.
- **reset**()
  - Reset the membrane potential of every block in the stack.
- **n_blocks**()
  - Return the number of residual blocks in the stack.
- **depth**()
  - Return the effective depth (two weight layers per block).

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
Synchronize two learning agents by mutually attracting their weights.

- **synchronize**(agent_a, agent_b)
  - Adjust both agents toward a shared weight configuration.

---

## Module `safety_cert.safety_cert`

### Class `SafetyStandard`

### Class `SILLevel`

### Class `ASILLevel`

### Class `Requirement`
One safety requirement with traceability links.

- **__post_init__**()

### Class `TraceabilityMatrix`
Requirement → implementation → verification traceability.

Each requirement links to:
- Implementation artifacts (RTL files, Python modules)
- Verification artifacts (formal proofs, UVM results, tests)

- **__init__**()
- **add_requirement**(req)
- **link_implementation**(req_id, impl_ref)
- **link_verification**(req_id, verif_ref)
- **coverage**()
- **open_count**()
- **generate_report**()
  - Generate text traceability report.

### Class `FailureCategory`

### Class `FailureMode`
One failure mode in the FMEDA.

- **__post_init__**()
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

- **__post_init__**()

### Class `FormalProofCertificate`
Certificate packaging formal verification evidence.

- **__post_init__**()
- **add_property**(prop)
- **proven_count**()
- **total_count**()
- **pass_rate**()
- **compute_hash**()
- **generate_report**()

### Class `WCETPath`
Worst-case execution time for one SC computation path.

- **__post_init__**()
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

- **__post_init__**()

### Class `ComplianceChecklist`
Standard-specific compliance checklist generator.

- **generate**(cls, standard)

### Class `CertificationPackage`
Complete certification package output.

- **__post_init__**()
- **checklist_coverage**()

### Class `CertificationGenerator`
Top-level generator for safety certification packages.

- **generate**(standard, target_sil, modules, formal_properties, network_config)
  - Generate a complete certification package.

### Class `CCFDefence`
One defence against common cause failures.

- **__post_init__**()

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

- **__post_init__**()
- **required_hft**()
  - Determine required HFT from SFF and target SIL.
- **is_simplex_ok**()

### Class `ChangeRecord`
One tracked change affecting safety.

- **__post_init__**()

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

- **__post_init__**()
- **requires_unit_testing**()
- **requires_architectural_design**()
- **from_sil**(sil)

### Class `ReliabilityMetrics`
System-level reliability from FMEDA data.

- **__post_init__**()
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

- **__post_init__**()

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

- **__post_init__**()
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

### Function `generate_scpn_datastream_payload()`
Generate a JSON-compatible stream payload in one call.

---

## Module `scpn.dcls_tent_kernel`

### Class `DclsForwardResult`
Result of one DCLS-max tent contraction.

Attributes
----------
output_q88 : int
    Saturated Q8.8 synapse output.
accumulator_q16_16 : int
    Saturated Q16.16 accumulator before the output shift.
overflow : bool
    ``True`` when accumulator or output saturation occurred.
active_tap_count : int
    Number of non-zero spike taps consumed by the contraction.
max_gate_q88 : int
    Largest Q8.8 tent gate applied to an active spike tap.


### Class `DclsBatchResult`
Per-channel results of a batched DCLS-max tent contraction.

Each array is indexed by output channel and has length ``B`` (the number of
``centre``/``sigma`` pairs supplied).

Attributes
----------
outputs_q88 : numpy.ndarray
    Saturated Q8.8 outputs, ``int16``.
accumulators_q16_16 : numpy.ndarray
    Saturated Q16.16 accumulators, ``int32``.
overflow : numpy.ndarray
    Saturation flags, ``bool_``.
active_tap_counts : numpy.ndarray
    Active spike-tap counts per channel, ``int64``.
max_gates_q88 : numpy.ndarray
    Largest applied tent gate per channel, ``int16``.


### Function `tent_gate_q88(tap_index, centre_q88, sigma_q88)`
Return the Q8.8 triangular tent gate for a delay tap.

The gate is ``max(0, 1 - |delay - centre| / sigma)`` evaluated in Q8.8 with
truncating integer division, so it matches the synthesisable RTL exactly.

Parameters
----------
tap_index : int
    Zero-based delay tap; the delay is ``tap_index`` in Q8.8 whole units.
centre_q88 : int
    Learnable tent centre in Q8.8.
sigma_q88 : int
    Tent half-width in Q8.8; must be positive.

Returns
-------
int
    Tent gate in ``&#91;0, 256&#93;`` Q8.8 (``256`` is ``1.0``).

Raises
------
ValueError
    If ``sigma_q88`` is not positive or ``tap_index`` is negative.

### Function `dcls_max_forward_q88(spikes, weights_q88, centre_q88, sigma_q88)`
Run one DCLS-max tent contraction in bit-true Q8.8 arithmetic.

Parameters
----------
spikes : array_like
    Per-tap spike flags; any non-zero entry marks an active tap.
weights_q88 : array_like
    Per-tap synaptic weights in Q8.8, same length as ``spikes``.
centre_q88 : int
    Learnable tent centre in Q8.8.
sigma_q88 : int
    Tent half-width in Q8.8; must be positive.

Returns
-------
DclsForwardResult
    Saturated Q8.8 output, Q16.16 accumulator and saturation diagnostics.

Raises
------
ValueError
    If the inputs are empty, length-mismatched or ``sigma_q88`` is not
    positive.

### Function `dcls_max_forward_batch_q88(spikes, weights_q88, centres_q88, sigmas_q88, n_taps)`
Pure-Python batched DCLS-max contraction — the bit-true floor reference.

Each output channel ``b`` contracts its own ``n_taps``-long spike/weight row
through a tent kernel with channel-specific learnable ``centre``/``sigma``.

Parameters
----------
spikes : array_like
    Flattened ``n_channels * n_taps`` spike flags (row-major per channel).
weights_q88 : array_like
    Flattened ``n_channels * n_taps`` Q8.8 weights (row-major per channel).
centres_q88 : array_like
    Per-channel learnable tent centres in Q8.8, length ``n_channels``.
sigmas_q88 : array_like
    Per-channel tent half-widths in Q8.8, length ``n_channels``; positive.
n_taps : int
    Number of delay taps per channel.

Returns
-------
DclsBatchResult
    Per-channel saturated outputs, accumulators and diagnostics.

Raises
------
ValueError
    If shapes are inconsistent or any ``sigma`` is non-positive.

### Function `available_backends()`
Probe which acceleration backends can run the DCLS batch kernel.

Returns
-------
dict
    Mapping of backend name to availability, in fastest-first order. The
    ``python`` floor is always ``True``.

### Function `dcls_max_forward_batch(spikes, weights_q88, centres_q88, sigmas_q88, n_taps)`
Run the batched DCLS-max tent kernel through the fastest available backend.

Parameters
----------
spikes, weights_q88, centres_q88, sigmas_q88, n_taps
    See :func:`dcls_max_forward_batch_q88`.
backend : str, optional
    ``"auto"`` (default) picks the fastest available backend in
    :data:`FASTEST_FIRST_BACKENDS` order. A specific name (``"rust"``,
    ``"mojo"``, ``"julia"``, ``"go"``, ``"python"``) forces that backend.

Returns
-------
DclsBatchResult
    Identical to the pure-Python floor for every backend (bit-exact).

Raises
------
ValueError
    If ``backend`` is not a known name.
ImportError
    If an explicitly requested accelerator backend is unavailable.

---

## Module `scpn.layers.l10_boundary`

### Class `L10_StochasticParameters`
Stochastic configuration parameters for the L10 boundary firewall layer.


### Class `L10_BoundaryLayer`
Topological firewall with dissonance rejection.

- **__init__**(params)
- **step**(dt, l9_input, external_noise)
  - Advance the firewall one timestep and return its boundary state.
- **get_global_metric**()
  - Return the scalar boundary-integrity metric for this layer.

---

## Module `scpn.layers.l11_morphic`

### Class `L11_StochasticParameters`
Stochastic configuration parameters for the SCPN morphic resonance / noospheric layer.


### Class `L11_MorphicLayer`
Noospheric spin-glass with memetic spreading dynamics.

- **__init__**(params)
- **step**(dt, l10_input)
  - Advance the morphic resonance / noospheric layer one timestep and return its output state.
- **get_global_metric**()
  - Return the scalar global metric summarising this layer's state.

---

## Module `scpn.layers.l12_quantum_info`

### Class `L12_StochasticParameters`

### Class `L12_QuantumInfoLayer`
ENAQT-inspired ecological coherence transport.

- **__init__**(params)
- **step**(dt, l11_input)
- **get_global_metric**()

---

## Module `scpn.layers.l13_temporal`

### Class `L13_StochasticParameters`
Stochastic configuration parameters for the SCPN source / temporal binding layer.


### Class `L13_TemporalLayer`
Temporal binding via cross-correlation within a sliding window.

- **__init__**(params)
- **step**(dt, l12_input)
  - Advance the source / temporal binding layer one timestep and return its output state.
- **get_global_metric**()
  - Return the scalar global metric summarising this layer's state.

---

## Module `scpn.layers.l14_integration`

### Class `L14_StochasticParameters`
Stochastic configuration parameters for the SCPN transdimensional integration layer.

- **__post_init__**()
  - Validate and finalise the layer parameters after construction.

### Class `L14_IntegrationLayer`
Weighted integration across SCPN layer metrics.

- **__init__**(params)
- **step**(dt, layer_metrics, l13_input)
  - Advance the transdimensional integration layer one timestep and return its output state.
- **get_global_metric**()
  - Return the scalar global metric summarising this layer's state.

---

## Module `scpn.layers.l15_meta`

### Class `L15_StochasticParameters`

### Class `L15_MetaLayer`
Self-monitoring meta-cognitive layer with GCI computation.

- **__init__**(params)
- **step**(dt, l14_input)
- **get_global_metric**()

---

## Module `scpn.layers.l16_director`

### Class `L16_StochasticParameters`

### Class `L16_DirectorLayer`
Cybernetic closure with PI control and Lyapunov monitoring.

- **__init__**(params)
- **step**(dt, l15_input)
- **get_global_metric**()

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

---

## Module `scpn.layers.l3_genomic`

### Class `L3_StochasticParameters`
Parameters for the Stochastic L3 Genomic Layer.


### Class `L3_GenomicLayer`
Stochastic implementation of the Genomic-Epigenomic Layer.

Models gene expression, epigenetic modifications, and bioelectric
pattern formation using bitstream representations.

- **__init__**(params)
- **step**(dt, l2_input, bioelectric_signal)
  - Advance the layer by one time step.
- **get_global_metric**()
  - Return the global genomic activity metric.
- **get_ciss_coherence**()
  - Return CISS spin coherence metric.

---

## Module `scpn.layers.l4_cellular`

### Class `L4_StochasticParameters`
Parameters for the Stochastic L4 Cellular Layer.


### Class `L4_CellularLayer`
Stochastic implementation of the Cellular-Tissue Synchronization Layer.

Models collective cellular behavior, gap junction coupling, and
tissue-level pattern formation using bitstream representations.

- **__init__**(params)
- **step**(dt, l3_input, external_stimulus)
  - Advance the layer by one time step.
- **get_global_metric**()
  - Return the global synchronization metric (Kuramoto order parameter).
- **get_tissue_pattern**()
  - Return 2D tissue activity pattern.

---

## Module `scpn.layers.l5_organismal`

### Class `L5_StochasticParameters`
Parameters for the Stochastic L5 Organismal Layer.


### Class `L5_OrganismalLayer`
Stochastic implementation of the Organismal-Psychoemotional Layer.

Models whole-organism integration, autonomic regulation, and
emotional dynamics using bitstream representations.

- **__init__**(params)
- **step**(dt, l4_input, external_event)
  - Advance the layer by one time step.
- **get_global_metric**()
  - Return the global organismal coherence metric.
- **get_emotional_valence**()
  - Return current emotional valence.

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

---

## Module `scpn.layers.l8_phase_field`

### Class `L8_StochasticParameters`
Stochastic configuration parameters for the SCPN cosmic phase-locking layer.

- **__post_init__**()
  - Validate and finalise the layer parameters after construction.

### Class `L8_PhaseFieldLayer`
Stochastic cosmic phase-locking via Kuramoto-coupled PTA oscillators.

- **__init__**(params)
- **step**(dt, l7_input)
  - Advance the cosmic phase-locking layer one timestep and return its output state.
- **get_global_metric**()
  - Return the scalar global metric summarising this layer's state.

---

## Module `scpn.layers.l9_memory`

### Class `L9_StochasticParameters`
Stochastic configuration parameters for the SCPN holographic memory layer.


### Class `L9_MemoryLayer`
Hopfield associative memory with stochastic bitstream encoding.

- **__init__**(params)
- **store**(pattern)
  - Hebbian imprint: W += pattern ⊗ pattern.
- **step**(dt, l8_input, boundary_cue, ebs_context)
  - Advance the holographic memory layer one timestep and return its output state.
- **get_global_metric**()
  - Return the scalar global metric summarising this layer's state.

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

---

## Module `security.side_channel_benchmark`

### Class `SideChannelBenchmarkError`
Raised when side-channel benchmark inputs or outputs are invalid.


### Class `SideChannelBenchmarkArm`
One benchmark arm with class-activity leakage proxy evidence.

- **__post_init__**()

### Class `SideChannelBenchmarkRecord`
Per-sample benchmark record with realised protected probability.

- **__post_init__**()

### Class `SideChannelDeployManifest`
Deploy/evidence manifest for an analytic side-channel benchmark.

- **__post_init__**()

### Class `SideChannelBenchmarkReport`
Analytic baseline-versus-protected side-channel benchmark report.

- **__post_init__**()

### Function `run_side_channel_leakage_benchmark()`
Compare correlated baseline streams against activity-balanced streams.

### Function `write_side_channel_benchmark_report(output_path)`
Run the analytic benchmark and write a canonical JSON artifact.

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

## Module `sensors.adc_to_spike_kernel`

### Class `ADCSpikeWindowConfig`
Fixed-point and decimation contract for the ADC-to-spike encoder.

Attributes
----------
adc_width : int
    Raw ADC sample width in bits (must exceed one).
q_int : int
    Q-format integer bits (must be positive).
q_frac : int
    Q-format fractional bits (must be non-negative).
decimation : int
    Number of ADC samples averaged into one spike window (must be positive).
signed_input : bool
    ``True`` if the ADC delivers two's-complement samples, ``False`` for
    offset-binary samples centred at mid-scale.
threshold_q : int
    Q-format magnitude that emits one spike (must be positive).

- **q_total**()
  - Total Q-format bit width.
- **q_min**()
  - Most negative representable Q-format code.
- **q_max**()
  - Most positive representable Q-format code.
- **validate**()
  - Raise :class:`ValueError` if any field is out of contract.

### Class `ADCSpikeWindowResult`
Per-window outputs of the ADC-to-spike encoder.

Each array is indexed by completed decimation window.

Attributes
----------
window_values_q : numpy.ndarray
    Sign-aware averaged Q-format window codes, ``int32``.
spike_counts : numpy.ndarray
    Deterministic per-window spike counts (``|window| // threshold``),
    ``int32``.
polarities : numpy.ndarray
    ``True`` where the window code is negative, ``bool_``.


### Function `quantise_adc(sample, config)`
Centre and quantise one raw ADC sample to a Q-format code.

Mirrors ``ADCToSpikeReference.quantise_adc``: two's-complement or offset-binary
centring, Q-format up-shift or sign-aware round-down, then saturation.

Parameters
----------
sample : int
    Raw ADC sample.
config : ADCSpikeWindowConfig
    Fixed-point contract.

Returns
-------
int
    Saturated Q-format code.

### Function `adc_to_spike_windows_q(samples, config)`
Pure-Python ADC-to-spike window encoder — the bit-true floor reference.

Parameters
----------
samples : array_like
    Raw ADC samples; the first ``n_windows * decimation`` are consumed.
config : ADCSpikeWindowConfig, optional
    Fixed-point/decimation contract (defaults to Q8.8, decimation 8).

Returns
-------
ADCSpikeWindowResult
    Per-window averaged codes, spike counts and polarities.

Raises
------
ValueError
    If the config is invalid or fewer than ``decimation`` samples are given.

### Function `available_backends()`
Probe which acceleration backends can run the ADC-to-spike kernel.

Returns
-------
dict
    Mapping of backend name to availability, in fastest-first order. The
    ``python`` floor is always ``True``.

### Function `adc_to_spike_windows(samples, config)`
Encode ADC samples into spike windows through the fastest available backend.

Parameters
----------
samples : array_like
    Raw ADC samples.
config : ADCSpikeWindowConfig, optional
    Fixed-point/decimation contract.
backend : str, optional
    ``"auto"`` (default) selects the fastest available backend in
    :data:`FASTEST_FIRST_BACKENDS` order; a specific name forces that backend.

Returns
-------
ADCSpikeWindowResult
    Bit-identical to the Python floor for every backend.

Raises
------
ValueError
    If ``backend`` is not a known name.
ImportError
    If an explicitly requested accelerator backend is unavailable.

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
  - Return the total number of pixels in the DVS frame.
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

- **__post_init__**()
  - Validate and normalise the membrane parameters.
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
  - Initialise random spins and their bipolar ``{-1, 1}`` representation.
- **step**()
  - Perform one parallel Metropolis-Hastings update and return the energy.
- **get_energy**()
  - Calculate global energy.
- **get_config**()
  - Return the current spin configuration as a ``0/1`` int8 array.

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
  - Advance one forward-Euler step; return ``(y_new, dt_used)``.

### Class `HeunSolver`
Heun's method (improved Euler / explicit trapezoidal) — O(h²).

k1 = f(t_n, y_n)
k2 = f(t_n + h, y_n + h*k1)
y_{n+1} = y_n + h/2 * (k1 + k2)

- **step**(f, y, t, dt)
  - Advance one Heun (improved-Euler) step; return ``(y_new, dt_used)``.

### Class `RK4Solver`
Classical Runge-Kutta — O(h⁴).

k1 = f(t, y)
k2 = f(t + h/2, y + h/2 * k1)
k3 = f(t + h/2, y + h/2 * k2)
k4 = f(t + h, y + h * k3)
y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)

Reference: Kutta, W. (1901). Z. Math. Phys. 46:435–453.

- **step**(f, y, t, dt)
  - Advance one classical RK4 step; return ``(y_new, dt_used)``.

### Class `DormandPrinceSolver`
Dormand-Prince adaptive RK45 — embedded pair for step-size control.

Uses the 7-stage, 5th-order solution with a 4th-order embedded error
estimate. Step size is adapted to maintain local truncation error
below the specified tolerance.

Reference: Dormand, J.R. & Prince, P.J. (1980). J. Comput. Appl. Math. 6:19–26.

- **__init__**(atol, rtol, max_factor, min_factor, safety)
- **step**(f, y, t, dt)
  - Advance one adaptive RK45 step; return ``(y_new, dt_used, dt_next)``.
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
  - Advance one exponential-Euler step; return ``(y_new, dt_used)``.

### Function `get_solver(name)`
Return an ODE solver instance selected by name.

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
  - Advance one Rosenbrock-Euler step; return ``(y_new, dt_used)``.

### Class `ImplicitEuler`
Backward (implicit) Euler — L-stable, 1st order.

y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

Solved via fixed-point iteration: for stiff neuron ODEs,
3–5 iterations typically suffice.

Reference: Hairer, E. & Wanner, G. (1996). Solving ODEs II. Springer.

- **__init__**(max_iterations, tol)
- **step**(f, y, t, dt)
  - Advance one backward-Euler step; return ``(y_new, dt_used)``.

### Class `TrapezoidalRule`
Trapezoidal rule (Crank-Nicolson) — A-stable, 2nd order.

y_{n+1} = y_n + h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))

Higher-order than implicit Euler while retaining A-stability.
Solved via fixed-point iteration.

Reference: Crank, J. & Nicolson, P. (1947). Proc. Camb. Phil. Soc. 43:50–67.

- **__init__**(max_iterations, tol)
- **step**(f, y, t, dt)
  - Advance one trapezoidal-rule step; return ``(y_new, dt_used)``.

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
  - Advance one Störmer-Verlet step; return ``(y_new, dt_used)``.

### Class `LeapfrogSolver`
Leapfrog (kick-drift-kick) — symplectic 2nd order.

Equivalent to Störmer-Verlet but staggered:
    p_{n+1/2} = p_n + (h/2) * f_p(q_n)
    q_{n+1} = q_n + h * f_q(p_{n+1/2})
    p_{n+1} = p_{n+1/2} + (h/2) * f_p(q_{n+1})

State vector: y = &#91;q₀..q_{n-1}, p₀..p_{n-1}&#93;

Reference: Yoshida, H. (1990). Phys. Lett. A 150:262–268.

- **step**(f, y, t, dt)
  - Advance one leapfrog (kick-drift-kick) step; return ``(y_new, dt_used)``.

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
  - Validate input/weight lengths and derive the input count.
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
Simulated quantum-measurement entropy source.

Injects simulated quantum indeterminacy into neural models by maintaining
a qubit state ``|psi>``, applying Hadamard superposition and phase
rotations, and measuring (collapsing) the state to generate noise.

- **__post_init__**()
  - Initialise the RNG and reset the qubit register to ``|0>``.
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
  - Return a human-readable one-line summary of the codec statistics.

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

---

## Module `spike_codec.predictive_codec`

### Class `PredictiveCompressionResult`
Compression result with predictive coding metrics.


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
    Must be finite and positive. Typical: 4.0-5.0 (4 sigma catches
    ~99.99% of noise).
snippet_samples : int
    Waveform samples to extract around each spike (before + after peak).
    Must fit the one-byte wire header: 1-255 samples.
max_templates : int
    Maximum number of spike waveform templates to maintain.
    Must fit the two-byte wire header: 1-65535 templates.
template_threshold : float
    Correlation threshold for template matching, inclusive range 0-1.
quantize_bits : int
    Background signal quantization, inclusive range 1-8. Fewer bits
    increase compression.
mode : str
    Compression mode controlling what is preserved:
    - ``"full"``: spike timing + waveform templates + background LFP (~137x)
    - ``"waveform"``: spike timing + waveform templates, no background (~1700x)
    - ``"spike"``: spike timing only, Neuralink-equivalent (~4500x)

- **__init__**(threshold_sigma, snippet_samples, max_templates, template_threshold, quantize_bits, mode)
- **compress**(waveform)
  - Compress raw electrode waveform.

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
  - Build the per-layer spiking graph-convolution stack.
- **forward**(node_features, adjacency)
  - Forward pass through all layers.
- **graph_classify**(node_features, adjacency)
  - Classify a graph by global readout (sum pooling + argmax).
- **n_layers**()
  - Return the number of graph-convolution layers.

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

## Module `studio.analysis_manifest`

### Class `StudioAnalysisResultManifest`
Path-free metadata for one Studio analysis response.

Parameters
----------
analysis_type:
    Stable analysis endpoint identifier such as ``"fi_curve"``.
source:
    Input surface used to produce the analysis result.
input_sha256:
    SHA-256 digest of the canonical request payload.
result_sha256:
    SHA-256 digest of the canonical result payload without this manifest.
output_keys:
    Sorted top-level keys present in the result payload before metadata.
evidence_classification:
    Stable evidence lane label for analysis results.
status:
    Terminal status for this analysis evidence object.

- **to_public_dict**()
  - Return the public, path-free analysis manifest.

### Function `build_analysis_result_manifest()`
Build digest-backed reproducibility metadata for an analysis result.

Parameters
----------
analysis_type:
    Stable analysis endpoint identifier.
source:
    Input surface used to produce the result.
request_payload:
    Request body used to run the analysis.
result_payload:
    Result payload before or after metadata insertion.

Returns
-------
StudioAnalysisResultManifest
    Path-free metadata suitable for UI display, exports, and evidence
    bundles.

Raises
------
ValueError
    If request or result payloads cannot be encoded as portable JSON.

### Function `attach_analysis_result_manifest()`
Return an analysis result with a path-free metadata manifest attached.

Parameters
----------
analysis_type:
    Stable analysis endpoint identifier.
source:
    Input surface used to produce the result.
request_payload:
    Request body used to run the analysis.
result_payload:
    Mutable analysis result payload.

Returns
-------
dict&#91;str, Any&#93;
    The same result object with ``analysis_metadata`` set.

### Function `infer_analysis_source(request_payload)`
Infer the Studio input surface from an analysis request payload.

Parameters
----------
request_payload:
    Request body submitted to an analysis endpoint.

Returns
-------
AnalysisSource
    ``"model"`` when a model name is present, ``"ode"`` when equations are
    present, ``"mixed"`` for comparison payloads, otherwise ``"unknown"``.

---

## Module `studio.app`

### Class `SimulateRequest`
Request body for direct ODE simulation in Studio.


### Class `ModelSimulateRequest`
Request body for model-catalogue simulation in Studio.


### Class `FICurveRequest`
Request body for firing-rate versus current sweeps.


### Class `DclsEvaluateRequest`
Request body for DCLS kernel parity evaluation.


### Class `BenchmarkRunRequest`
Request body for local Studio benchmark runs.


### Class `BenchmarkContributeRequest`
Request body for benchmark-databank contribution uploads.


### Class `CompileRequest`
Request body for equation-to-SystemVerilog compilation.


### Class `BifurcationRequest`
Request body for one-parameter bifurcation sweeps.


### Class `SensitivityRequest`
Request body for Studio model sensitivity analysis.


### Class `NullclineRequest`
Request body for two-dimensional nullcline analysis.


### Class `PrecisionRequest`
Request body for float versus fixed-point precision comparisons.


### Class `AdaptivePrecisionAutoTuneRequest`
Request body for adaptive synapse-precision auto-tuning.


### Class `AdaptivePrecisionFormalBundleRequest`
Request body for adaptive-precision formal evidence bundles.


### Class `PresetActionResolveRequest`
Request body for resolving one preset action template.


### Class `PresetActionsExecuteAllRequest`
Request body for executing all actions attached to a preset.


### Class `PresetDefaultFlowRunRequest`
Request body for running a preset's default action flow.


### Class `PresetDefaultFlowVerifyRequest`
Request body for verifying a preset default-flow plan.


### Class `PresetDefaultFlowGuardedRunRequest`
Request body for running a preset flow with fingerprint guards.


### Class `PresetDefaultFlowRunFromContractRequest`
Request body for running a preset flow from a stored contract.


### Class `PresetDefaultFlowAttestRequest`
Request body for attesting a completed preset flow run.


### Class `StudioIdentityServiceAccountUpdateRequest`
Request body for admin service-account metadata updates.


### Class `StudioBrowserUserUpdateRequest`
Request body for admin browser-user metadata updates.


### Class `StudioBrowserUserCreateRequest`
Request body for admin browser-user creation.


### Class `StudioBrowserUserPasswordRotateRequest`
Request body for admin browser-user password rotation.


### Class `StudioBrowserLoginRequest`
Request body for browser-user login.


### Class `StudioEvidenceBundleRequest`
Request body for admin evidence bundle export.


### Class `StudioAuditQuarantineArchiveRequest`
Request body for admin audit quarantine archive creation.


### Class `StudioAuditQuarantineArchiveValidateRequest`
Request body for admin audit quarantine archive validation.


### Class `StudioAuditQuarantineArchiveRestoreRequest`
Request body for admin audit quarantine archive restore materialization.


### Class `StudioAuditQuarantineArchivePurgeRequest`
Request body for admin audit quarantine archive retention purges.


### Class `StudioTrainingWeightRestoreRequest`
Request body for admin training weight-restore materialization.


### Class `StudioTrainingWeightAttachRequest`
Request body for admin training weight-restore warm-start attach.


### Class `StudioTrainingWeightLiveAttachRequest`
Request body for admin training weight-restore live attach.


### Class `PresetDefaultFlowAttestationVerifyRequest`
Request body for verifying a preset flow attestation.


### Class `CompareRequest`
Request body for comparing two Studio simulation configurations.


### Class `FreqResponseRequest`
Request body for frequency-response analysis.


### Class `HeatmapRequest`
Request body for two-parameter response heatmap analysis.


### Class `NetworkRequest`
Request body for balanced excitatory-inhibitory network simulation.


### Class `CodegenRequest`
Request body for Studio script and one-liner generation.


### Function `create_app(runtime_settings)`
Create and configure the Visual SNN Studio FastAPI application.

---

## Module `studio.behavior_probe`

### Class `BehaviorObservation`
One reproducibility-checked observation at a single drive current.

- **to_public_dict**()
  - Return a JSON-compatible observation.

### Class `ModelBehaviorProfile`
The measured behavioural envelope of one model over the current sweep.

- **to_public_dict**()
  - Return a JSON-compatible behaviour profile.

### Function `derive_behavior_tags(observations)`
Derive the behaviour tags from a model's sweep observations.

Pure and side-effect free, so the tag logic is testable without running any
simulation. Excitability is read from the sign of the response (robust even
for stochastic models); the firing-pattern tags are read only from
reproducible observations, and are withheld entirely for a stochastic model.

Parameters
----------
observations:
    The per-current observations, in any order.
stochastic:
    Whether the model's spike train failed to reproduce at some drive.

Returns
-------
tuple&#91;str, ...&#93;
    The validated, sorted behaviour tags. Empty when no current could be
    driven (an honest "no measured behaviour").

### Function `probe_model_behavior(name)`
Probe one model across the current sweep and derive its behaviour tags.

Never raises for a model that cannot be driven: each failed drive is
recorded as an error observation and the profile is still returned.

### Function `probe_all_models()`
Probe every catalogue model and return a recordable evidence manifest.

The manifest is the source the fast catalogue gate compares descriptors
against: a per-model tag set plus the sweep configuration and digests, so a
committed ``behavior_tags`` field can be checked for equality with the
measurement without re-running any simulation.

### Function `load_behavior_evidence()`
Load the recorded behaviour evidence manifest.

Raises
------
FileNotFoundError
    If the manifest has not been generated.

### Function `behavior_tags_for(name, evidence)`
Return the recorded behaviour tags for a model (empty if unrecorded).

---

## Module `studio.benchmark_contribution`

### Function `safe_environment()`
Collect only the privacy-safe, aggregatable host facts.

### Function `run_local_benchmark(n_channels, n_taps, repeats, seed)`
Time the DCLS kernel across the in-process-safe backends on this machine.

Returns a fully-formed submission (schema ``scpn.benchmark.submission.v1``)
that the caller may inspect and, opt-in, hand to :func:`store_contribution`.
Julia is skipped in-process (the torch/juliacall segfault) and reported as
parity-verified offline rather than timed live.

### Function `validate_submission(payload)`
Return a list of schema/privacy violations; empty means the payload is OK.

### Function `store_contribution(payload, handle)`
Validate and append a submission to the local databank (opt-in path).

Raises ``ValueError`` with the joined violations if the payload fails schema
or privacy validation, so an invalid or identifying submission never lands.

### Function `load_databank()`
Return every stored contribution (already free of identifying fields).

### Function `databank_leaderboard()`
Aggregate the databank into a per-CPU, per-backend speed-up leaderboard.

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

## Module `studio.compile_traceability`

### Class `StudioCompileTraceability`
Path-free source-to-RTL provenance for a Studio compile result.

Parameters
----------
equations:
    ODE equation strings submitted by the operator.
threshold:
    Optional spike-threshold expression.
reset:
    Optional reset expression.
params:
    Numeric parameter overrides used by the compiler.
init:
    Initial state values supplied with the request.
module_name:
    RTL module name requested by the operator.
verilog:
    Generated Verilog/SystemVerilog source.
source:
    Stable source-kind label for the compile request.
output_language:
    RTL language emitted by the backend compiler.
evidence_classification:
    Evidence lane label consumed by Studio evidence bundles.
status:
    Terminal status for this compile traceability object.

- **to_public_dict**()
  - Return the public, path-free traceability payload.

### Function `build_compile_traceability()`
Build path-free traceability for an equation-to-RTL compile result.

Parameters
----------
equations:
    ODE equation strings submitted by the operator.
threshold:
    Optional spike-threshold expression.
reset:
    Optional reset expression.
params:
    Numeric parameter overrides used by the compiler.
init:
    Initial state values supplied with the request.
module_name:
    RTL module name requested by the operator.
verilog:
    Generated Verilog/SystemVerilog source.

Returns
-------
StudioCompileTraceability
    Immutable traceability record with stable public JSON conversion.

Raises
------
ValueError
    If no equations are supplied.

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

## Module `studio.dcls`

### Function `probe_backends()`
Report each backend's in-process status without ever crashing.

Each entry is ``{backend, available, live}``: ``live`` is ``False`` for a
backend we decline to run in this process (Julia under torch), in which case
``available`` reflects its declared support rather than a live probe.

### Function `dcls_benchmark()`
Return the recorded multi-backend throughput benchmark, or ``None``.

These are pre-measured numbers from ``benchmarks/results`` — a CPU-shielded
run on a known host — not a live timing, so the Studio reports honest,
reproducible figures with their measurement context rather than noisy
per-request samples. Backends are ordered fastest-measured-first; Python is
the 1x reference floor.

### Function `dcls_kernel_info()`
Describe the DCLS-max kernel: provenance, fixed-point contract, evidence.

### Function `dcls_tent_profile(centre_q88, sigma_q88, n_taps)`
Return the per-tap triangular gate profile of a learnable tent kernel.

Each delay tap ``t`` receives a Q8.8 gate from :func:`tent_gate_q88`; the
profile is what the learnable ``centre``/``sigma`` shape and what the synapse
convolves the spike taps against.

### Function `dcls_forward_parity(spikes, weights_q88, centre_q88, sigma_q88)`
Run the contraction on every available backend and report the parity.

The Python floor is the reference; each accelerated backend must reproduce
its Q8.8 output bit-for-bit. The returned ``bit_exact`` flag is the evidence
that the learnable-delay kernel is hardware-faithful across the whole stack.

---

## Module `studio.evidence_classification`

### Function `validate_studio_evidence_classification(value)`
Return a controlled Studio evidence class or fail closed.

Parameters
----------
value:
    Candidate evidence class supplied by a Studio manifest.

Returns
-------
StudioEvidenceClassification
    The validated evidence class.

Raises
------
ValueError
    If ``value`` is not one of the supported Studio evidence classes.

### Function `validate_studio_evidence_status(value)`
Return a controlled terminal evidence status or fail closed.

Parameters
----------
value:
    Candidate terminal status supplied by a Studio manifest.

Returns
-------
StudioEvidenceStatus
    The validated terminal evidence status.

Raises
------
ValueError
    If ``value`` is not a supported terminal evidence status.

---

## Module `studio.model_scan`

### Class `ModelScanEntry`
Path-free firing-pattern classification for one Studio model.

A model that could not be driven by the scan's constant current carries the
``error`` pattern and a non-empty ``error_type`` so the failure is visible in
the result rather than aborting the whole scan.

- **is_error**()
  - True when this entry records a simulation failure.
- **to_public_dict**()
  - Return a JSON-compatible model scan entry.

### Class `StudioModelScanManifest`
Path-free metadata for one complete Studio model scan.

Parameters
----------
current:
    Constant input current used for each model simulation.
duration:
    Simulation duration used for each model scan run.
model_count:
    Number of successfully classified models in the result.
pattern_counts:
    Count of each detected firing-pattern label.
input_sha256:
    SHA-256 digest of the scan configuration.
result_sha256:
    SHA-256 digest of the returned model classifications.
evidence_classification:
    Controlled evidence class for the scan workflow.
status:
    Controlled terminal status for the scan workflow.

- **to_public_dict**()
  - Return the JSON-compatible model scan metadata.

### Function `scan_all_models(current, duration)`
Simulate every model at a given current and classify its firing pattern.

Results are cached per ``(current, duration)`` pair so a scan for one
configuration cannot be served as evidence for another configuration.

---

## Module `studio.models`

### Class `RustStudioBackendUnavailable`
Raised when the Studio Rust batch-simulation path is unavailable.


### Class `RustStudioBackendError`
Raised when the Studio Rust batch-simulation path fails at runtime.


### Class `ModelMetadataError`
Raised when Studio model metadata loading fails for a known model.


### Function `list_models()`
Return declared metadata for every registered neuron model.

Each entry is built from the model's committed descriptor (family, category,
maturity, provenance, parameter and state counts). Models without a descriptor
fall back to code introspection with an ``inferred`` category. Results are
cached after the first call.

### Function `get_model_detail(name)`
Return the full declared metadata view for a single model.

### Function `model_facets()`
Return the catalogue facet taxonomy and counts for discovery UX.

### Function `model_documentation(name)`
Return the rendered reference documentation for a model, or ``None``.

The per-model reference page lives at ``docs/api/models/<module>.md``; the
Studio serves its Markdown so the documentation is browsable inline next to
the live model rather than only in the built docs site.

### Function `simulate_model(name, param_overrides, dt, duration, current, protocol, frequency_hz, use_fast_path)`
Simulate a named model. Uses Rust engine when model has default params.

Set ``use_fast_path=False`` to force the Python reference model and bypass the
Rust accelerator. The behaviour probe relies on this so its characterisation
is the canonical model's, independent of whether the Rust extension happens to
be loaded (the two backends can differ for models with an internal RNG).

---

## Module `studio.network`

### Function `simulate_ei_network(n_exc, n_inh, w_ee, w_ei, w_ie, w_ii, p_conn, ext_rate, duration, dt)`
Simulate a balanced E-I network. Uses Rust engine when available.

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

## Module `studio.nir_compile`

### Function `compile_nir_graph(graph)`
Lower a parsed NIR graph to synthesisable Verilog and return the artefacts.

### Function `compile_nir_file_bytes(data)`
Compile a standard ``.nir`` (HDF5) document supplied as raw bytes.

Options are forwarded to :func:`compile_nir_graph`.

---

## Module `studio.platform.action_evidence`

### Class `StudioActionEvidence`
Path-free manifest describing one Studio worker-backed action.

- **to_public_dict**()
  - Return the path-free evidence payload.

### Function `write_studio_action_evidence_manifest(context)`
Write a normalised evidence manifest for a worker-backed action.

Parameters
----------
context:
    Job sandbox used to write the path-confined evidence artifact.
action_kind:
    Stable Studio action identifier, such as ``studio.compile``.
result:
    Portable JSON result payload whose canonical SHA-256 digest is recorded.
result_artifact:
    Manifest entry for the result artifact produced by the same job.
evidence_artifact_path:
    Relative path for the evidence manifest artifact.
evidence_classification:
    Controlled evidence class used by bundle and operator views.
replay_route:
    HTTP method and route template used to reproduce the action.
status:
    Terminal action status.
request_id:
    Optional request correlation identifier.
principal_id:
    Optional authenticated principal identifier.
error_message:
    Optional bounded terminal error message.

Returns
-------
StudioActionEvidence
    Path-free evidence payload plus its artifact manifest entry.

Raises
------
ValueError
    If any controlled field is invalid or ``result`` is not portable JSON.

---

## Module `studio.platform.analysis_limits`

### Class `AnalysisBudgetError`
Raised when a synchronous analysis request exceeds its execution budget.

Parameters
----------
limit:
    Which budget dimension was violated.
projected:
    Projected value for the violated dimension (integration steps or
    simulation count) computed from the request.
allowed:
    Maximum value permitted for the violated dimension.
message:
    Operator-facing summary that must not include local paths or secrets.

- **__init__**()
- **to_public_detail**()
  - Return a JSON-serializable, path-free HTTP error detail payload.

### Class `AnalysisBudget`
Ceilings that bound synchronous Studio analysis execution.

Parameters
----------
max_steps_per_simulation:
    Maximum integration steps (``ceil(duration / dt)``) for any single
    simulation in a request.
max_total_steps:
    Maximum summed integration steps across every simulation a request
    drives (``simulation_count * steps_per_simulation`` for a sweep).
max_simulations:
    Maximum number of simulations a single request may drive, independent
    of per-simulation cost.

- **__post_init__**()
  - Validate that every budget ceiling is a positive integer.
- **to_public_dict**()
  - Return a JSON-serializable, path-free budget payload.

### Class `AnalysisCost`
Projected synchronous integration cost of an analysis request.

Parameters
----------
simulation_count:
    Number of simulations the request drives.
steps_per_simulation:
    Largest single-simulation integration-step count among the request's
    simulations.
total_steps:
    Summed integration steps across every simulation in the request.

- **to_public_dict**()
  - Return a JSON-serializable, path-free cost payload.

### Function `simulation_step_count(duration, dt)`
Return the integration-step count for one simulation.

Parameters
----------
duration:
    Simulated time span in milliseconds. Must be finite and positive.
dt:
    Integration timestep in milliseconds. Must be finite and positive.

Returns
-------
int
    ``ceil(duration / dt)`` integration steps.

Raises
------
AnalysisBudgetError
    If ``duration`` or ``dt`` is non-finite or non-positive. The error uses
    the ``"timestep"`` limit so callers can surface a path-free 422.

### Function `resolve_request_timestep(dt)`
Return the timestep used to project a request's cost.

Parameters
----------
dt:
    Caller-supplied timestep in milliseconds, or ``None`` when the request
    defers to the named model's default timestep.

Returns
-------
float
    ``dt`` when supplied, otherwise
    :data:`STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS`. A supplied non-positive
    ``dt`` is returned unchanged so the timestep gate rejects it.

### Function `evaluate_analysis_cost()`
Project the cost of a request whose simulations share one duration/dt.

Parameters
----------
simulation_count:
    Number of simulations the request drives. Must be positive.
duration:
    Shared simulated time span in milliseconds.
dt:
    Shared integration timestep in milliseconds.

Returns
-------
AnalysisCost
    Projected per-simulation and total integration-step counts.

Raises
------
AnalysisBudgetError
    If ``simulation_count`` is non-positive (``"simulations"`` limit) or the
    timestep is invalid (``"timestep"`` limit).

### Function `evaluate_multi_config_cost(configs)`
Project the cost of a request whose simulations have distinct duration/dt.

Parameters
----------
configs:
    One ``(duration, dt)`` pair per simulation the request drives.

Returns
-------
AnalysisCost
    ``steps_per_simulation`` is the largest single-simulation cost and
    ``total_steps`` is the summed cost across ``configs``.

Raises
------
AnalysisBudgetError
    If ``configs`` is empty (``"simulations"`` limit) or any pair has an
    invalid timestep (``"timestep"`` limit).

### Function `evaluate_nullcline_grid_cost()`
Project the synchronous point-evaluation cost of a nullcline grid.

Parameters
----------
grid_size:
    Number of points per axis in the square nullcline grid.
equation_count:
    Number of ODE right-hand-side expressions evaluated at each grid point.

Returns
-------
AnalysisCost
    Cost projection where each grid point is treated as one synchronous
    unit of work and ``steps_per_simulation`` records the equation
    evaluations performed at that point.

Raises
------
AnalysisBudgetError
    If either count is non-positive. The ``"simulations"`` limit is used
    because the grid point count is checked against the same synchronous
    request fan-out ceiling as parameter sweeps.

### Function `evaluate_model_scan_cost()`
Project the synchronous simulation cost of a Studio model scan.

Parameters
----------
model_count:
    Number of catalogue models the scan will simulate.
duration:
    Shared simulated time span in milliseconds.
dt:
    Timestep in milliseconds used for budget projection. Model-scan calls
    currently defer to model defaults at execution time, so the route uses
    :data:`STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS` for this projection.

Returns
-------
AnalysisCost
    Cost projection where each catalogue model contributes one synchronous
    simulation.

Raises
------
AnalysisBudgetError
    If ``model_count`` is non-positive (``"simulations"`` limit) or the
    timestep is invalid (``"timestep"`` limit).

### Function `enforce_analysis_budget(cost, budget)`
Reject a projected analysis cost that exceeds the configured budget.

Parameters
----------
cost:
    Projected request cost from :func:`evaluate_analysis_cost` or
    :func:`evaluate_multi_config_cost`.
budget:
    Active synchronous analysis ceilings.

Raises
------
AnalysisBudgetError
    If the simulation count, per-simulation steps, or total steps exceed the
    budget. The first violated dimension is reported, in
    simulations -> per-simulation -> total order.

---

## Module `studio.platform.audit_quarantine_archive`

### Class `StudioAuditQuarantineArchiveResult`
Path-free result returned after writing a quarantine archive.

Parameters
----------
archive_id:
    Stable archive identifier derived from the evidence job ID.
manifest:
    JSON manifest describing the archive artifacts written by the job.
summary:
    Path-free aggregate counts for operator review.
artifact_paths:
    Archive-relative artifact paths written through the Studio job context.

- **to_public_dict**()
  - Return the path-free quarantine archive result.

### Class `StudioAuditQuarantineArchiveValidation`
Path-free validation result for one quarantine archive import candidate.

Parameters
----------
valid:
    Whether the supplied archive and optional manifest satisfy the import
    contract.
archive_id:
    Archive identifier when it can be read safely.
summary:
    Recomputed path-free summary for a valid archive candidate.
errors:
    Stable validation error codes for operator remediation.
warnings:
    Stable validation warning codes for non-blocking operator review.

- **to_public_dict**()
  - Return the path-free validation result.

### Class `StudioAuditQuarantineArchiveRetentionEntry`
Path-free retention disposition for one quarantine archive job.

Parameters
----------
archive_id:
    Stable archive identifier returned by the archive job.
job_id:
    Studio job identifier that owns the archive artifacts.
created_at_utc:
    Job creation timestamp from the job manager record.
finished_at_utc:
    Terminal job timestamp when available.
event_count:
    Number of quarantined audit rows captured in the archive.
retained_event_count:
    Number of retained audit rows visible to the source quarantine export.
artifact_paths:
    Path-free artifact identifiers declared by the archive job.
disposition:
    Operator retention decision for this archive.
summary:
    Archive summary copied from the validated job result.

- **to_public_dict**()
  - Return this retention entry as a path-free JSON object.

### Class `StudioAuditQuarantineArchiveRetentionPlan`
Path-free retention inventory for quarantine archive jobs.

Parameters
----------
entries:
    Archive job entries sorted newest first.
retain_latest:
    Number of newest archives marked for retention.
skipped_record_count:
    Number of archive-owner job records that were incomplete or malformed.

- **to_public_dict**()
  - Return the path-free retention plan for operator APIs.

### Class `StudioAuditQuarantineArchiveRestoreResult`
Path-free result returned after materializing a restore artifact.

Parameters
----------
archive_id:
    Validated archive identifier restored into job artifacts.
manifest:
    JSON manifest describing the restore artifacts written by the job.
summary:
    Path-free aggregate counts for operator review.
artifact_paths:
    Restore-relative artifact paths written through the Studio job context.

- **to_public_dict**()
  - Return the path-free quarantine archive restore result.

### Class `StudioAuditQuarantineArchivePurgeResult`
Path-free result returned after purging archive prune candidates.

Parameters
----------
purged_entries:
    Archive entries removed from the Studio job manager.
retained_entries:
    Archive entries kept according to the retention policy.
skipped_record_count:
    Number of archive-owner job records that were incomplete or malformed.
retain_latest:
    Number of newest valid archives retained before purging older entries.

- **to_public_dict**()
  - Return the path-free purge result for operator APIs.

### Function `write_studio_audit_quarantine_archive(context)`
Write quarantined audit evidence into a confined Studio job archive.

Parameters
----------
context:
    Studio job context that owns the archive artifacts and enforces path
    confinement, byte ceilings, and SHA-256 manifests.
quarantine_export:
    Path-free payload returned by ``JsonlAuditSink.export_quarantine``.
clock:
    Optional UTC clock for deterministic tests.

Returns
-------
StudioAuditQuarantineArchiveResult
    Path-free manifest and artifact list for the generated archive.

Raises
------
ValueError
    If the export payload is malformed or not the quarantine export schema.

### Function `validate_studio_audit_quarantine_archive(archive_payload)`
Validate one quarantine archive before import or restore handling.

Parameters
----------
archive_payload:
    Candidate archive JSON object, normally loaded from
    ``evidence/audit-quarantine/archive.json``.
manifest_payload:
    Optional candidate manifest JSON object, normally loaded from
    ``evidence/audit-quarantine/manifest.json``.

Returns
-------
StudioAuditQuarantineArchiveValidation
    Path-free validation verdict with stable error codes.

### Function `build_studio_audit_quarantine_archive_retention_plan(records)`
Build a non-destructive retention plan for quarantine archive jobs.

Parameters
----------
records:
    Studio job records from the local job manager.
retain_latest:
    Number of newest valid quarantine archives that should be retained.

Returns
-------
StudioAuditQuarantineArchiveRetentionPlan
    Path-free inventory marking older valid archives as prune candidates.

Raises
------
ValueError
    If ``retain_latest`` is not positive.

### Function `write_studio_audit_quarantine_restore(context)`
Materialize validated quarantine archive rows into restore artifacts.

Parameters
----------
context:
    Studio job context that owns the restore artifacts and enforces path
    confinement, byte ceilings, and SHA-256 manifests.
archive_payload:
    Candidate archive JSON object from
    ``evidence/audit-quarantine/archive.json``.
manifest_payload:
    Optional companion manifest JSON object from
    ``evidence/audit-quarantine/manifest.json``.
clock:
    Optional UTC clock for deterministic tests.

Returns
-------
StudioAuditQuarantineArchiveRestoreResult
    Path-free restore manifest and artifact list.

Raises
------
ValueError
    If archive validation fails before restore materialization.

### Function `purge_studio_audit_quarantine_archive_prune_candidates(records)`
Purge archive jobs marked as retention prune candidates.

Parameters
----------
records:
    Studio job records from the local job manager before deletion.
purge_job:
    Callable that performs the path-confined job purge for one job ID.
retain_latest:
    Number of newest valid quarantine archives to keep.

Returns
-------
StudioAuditQuarantineArchivePurgeResult
    Path-free summary of purged and retained archive entries.

Raises
------
ValueError
    If ``retain_latest`` is not positive.

---

## Module `studio.platform.auth_throttle`

### Class `StudioLoginThrottleDecision`
Decision returned before a browser login attempt is evaluated.

Parameters
----------
allowed:
    Whether the login attempt may proceed to password verification.
reason:
    Stable denial reason for audit rows and API responses.
retry_after_seconds:
    Whole-second cooldown remaining when the attempt is denied.


### Class `StudioLoginThrottleSnapshot`
Secret-free aggregate state for browser-login throttling.

Parameters
----------
active_bucket_count:
    Number of normalized login keys with live failure or lockout state.
locked_bucket_count:
    Number of live buckets currently locked out.
max_retry_after_seconds:
    Largest remaining retry interval across locked buckets, or ``0`` when
    no bucket is locked.

- **to_public_dict**()
  - Return a secret-free throttle aggregate for operator APIs.

### Class `StudioBrowserLoginThrottle`
Windowed lockout guard for Studio browser-login attempts.

The guard stores only normalized login keys and failure timestamps in memory.
It does not persist or inspect password material. A successful login clears
the bucket for the normalized key.

- **__init__**()
- **check**(username)
  - Return whether a browser-login attempt may proceed.
- **record_failure**(username)
  - Record a failed browser-login attempt and return the new state.
- **record_success**(username)
  - Clear the failure bucket after a successful browser login.
- **snapshot**()
  - Return aggregate lockout state without exposing login keys.

---

## Module `studio.platform.backup_plan`

### Class `StudioBackupPlanItem`
One durable Studio state target that must be backed up.

Parameters
----------
item_id:
    Stable item identifier for operator automation.
target_kind:
    Expected filesystem object kind.
description:
    Operator-facing state description.
source_label:
    Path-free source label, normally an environment variable name or a
    documented Studio default.
configured:
    Whether the target is configured for this runtime.
exists:
    Whether the current target exists.
required:
    Whether this target is required by the active deployment profile.
backup_actions:
    Path-free actions for capturing the target.
restore_actions:
    Path-free actions for restoring the target.
local_path:
    Optional resolved path. This is emitted only when the operator requests
    local-path disclosure for an internal handoff.

- **to_public_dict**()
  - Return a JSON-serializable item payload.

### Class `StudioBackupPlan`
Machine-readable Studio backup and restore plan.

Parameters
----------
deployment_profile:
    Active Studio deployment profile.
items:
    Durable state targets that an operator backup must capture.
include_local_paths:
    Whether public serialization should include resolved local paths.
schema_version:
    Stable schema identifier.

- **missing_required_count**()
  - Return the number of required targets that are not configured.
- **missing_existing_count**()
  - Return the number of configured targets that do not exist yet.
- **ready_for_restore_drill**()
  - Return whether all required targets are configured and present.
- **to_public_dict**()
  - Return a JSON-serializable backup-plan payload.

### Function `build_studio_backup_plan(settings)`
Build the Studio durable-state backup and restore plan.

Parameters
----------
settings:
    Runtime settings that define identity, audit, and job state locations.
    When omitted, settings are read from the current environment.
include_local_paths:
    Include resolved local paths in the serialized plan. Keep this disabled
    for deployment logs that may leave the host.
project_root:
    Optional Studio project workspace root for tests and embedded tools.

Returns
-------
StudioBackupPlan
    Backup and restore manifest for the active Studio runtime profile.

---

## Module `studio.platform.bootstrap`

### Class `StudioIdentityBootstrapResult`
Result returned after creating a Studio service-account identity file.

Parameters
----------
identity_file_path:
    Destination JSON identity file path.
principal_id:
    Service-account principal written to the identity file.
roles:
    Roles granted to the bootstrap service account.
bearer_token:
    One-time bearer token returned to the operator. It is never written to
    disk by the bootstrap routine.
token_sha256:
    SHA-256 digest persisted in the identity file.
expires_at_utc:
    Optional UTC expiry timestamp written to the identity file.
file_permissions_hardened:
    Whether the routine successfully applied owner-only file permissions on
    platforms that expose POSIX file modes.
parent_directory_created:
    Whether the routine created the identity file parent directory.

- **to_public_dict**()
  - Return bootstrap metadata without the bearer token.

### Function `bootstrap_studio_admin_identity(identity_file_path)`
Create a local Studio admin identity file for first deployment.

Parameters
----------
identity_file_path:
    Destination for the JSON identity file consumed by
    ``SC_NEUROCORE_STUDIO_IDENTITY_FILE``.
principal_id:
    Stable service-account principal recorded in audit rows.
roles:
    Non-empty role set granted to the service account.
token_bytes:
    Entropy bytes requested from ``token_factory``. Values below
    ``MIN_BOOTSTRAP_TOKEN_BYTES`` are rejected.
expires_at_utc:
    Optional ISO-8601 timestamp. Values are normalised to UTC with a ``Z``
    suffix before being written.
overwrite:
    Whether an existing identity file may be replaced atomically.
token_factory:
    Token generator hook. Production callers use ``secrets.token_urlsafe``;
    tests may inject a deterministic generator.

Returns
-------
StudioIdentityBootstrapResult
    Bootstrap metadata plus the one-time bearer token.

Raises
------
FileExistsError
    If the destination exists and ``overwrite`` is false.
ValueError
    If identity metadata is malformed.
OSError
    If the destination cannot be written.

---

## Module `studio.platform.capabilities`

### Class `CapabilityStatus`
Runtime status for a Studio capability.


### Class `EvidenceClass`
Evidence class attached to a Studio capability.


### Class `CapabilityRequirement`
Requirement needed for a capability to be usable.

- **to_public_dict**()
  - Return a public, non-secret representation.

### Class `CapabilityDescriptor`
Static descriptor for a Studio capability.


### Class `CapabilityHealth`
Runtime health projection for a Studio capability.

- **to_public_dict**()
  - Return a public, non-secret API representation.

### Class `CapabilityRegistry`
In-memory registry for Studio capabilities.

- **__init__**()
- **register**(descriptor)
  - Register a capability descriptor.
- **health**(capability_id)
  - Return runtime health for one capability.
- **health_all**()
  - Return runtime health for all capabilities sorted by stable ID.

### Function `build_default_studio_capability_registry()`
Build the default capability registry for the current Studio backend.

---

## Module `studio.platform.compile_process`

### Function `run_compile_process_task(context, payload)`
Compile one Studio ODE payload in an isolated process.

Parameters
----------
context:
    Job sandbox used to write path-confined result and evidence artifacts.
payload:
    JSON object matching the public ``/api/compile`` request contract.

Returns
-------
dict&#91;str, object&#93;
    Path-free compile response payload.

Raises
------
ValueError
    If the payload does not match the JSON-serializable compile contract.

---

## Module `studio.platform.deployment_profiles`

### Class `StudioDeploymentProfilePackage`
Machine-readable Studio deployment profile package.

Parameters
----------
name:
    Operator-facing deployment package name.
runtime_profile:
    Studio runtime profile value exported through
    ``SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE``.
summary:
    Short profile summary for operator dashboards and release notes.
environment:
    Environment variables required by this package. Values are placeholders
    or concrete safe defaults and must not contain secrets.
required_operator_inputs:
    Secret or path values that the operator must provide outside the
    repository before launching Studio.
security_controls:
    Controls that are active or required for the profile.
backup_items:
    Durable state that must be included in local backup/restore plans.
preflight_command:
    Command that validates the profile from the target launch environment.
launch_command:
    Command used to launch the Studio process after environment setup.
schema_version:
    Stable JSON schema identifier.

- **to_public_dict**()
  - Return a JSON-serializable deployment package manifest.
- **to_env_lines**()
  - Return sorted shell ``export`` lines for non-secret profile values.

### Function `build_studio_deployment_profile_package(name)`
Build one Studio deployment-profile package.

Parameters
----------
name:
    Deployment package name. ``local`` targets single-operator loopback
    use, ``lab`` targets private LAN or VPN-hosted research workstations,
    and ``server`` targets reverse-proxied service deployment.

Returns
-------
StudioDeploymentProfilePackage
    Path-placeholder manifest with environment, preflight, launch, and
    backup guidance.

Raises
------
ValueError
    If ``name`` is not a supported Studio deployment package.

### Function `list_studio_deployment_profile_packages()`
Return all supported Studio deployment-profile packages.

---

## Module `studio.platform.evidence_bundle`

### Class `StudioEvidenceBundleResult`
Path-free result returned after writing a Studio evidence bundle.

Parameters
----------
bundle_id:
    Stable bundle identifier derived from the evidence job ID.
manifest:
    JSON manifest describing every file written into the bundle.
summary:
    Path-free aggregate counts for operator review and UI rendering.
artifact_paths:
    Bundle-relative artifact paths written through the Studio job context.

- **to_public_dict**()
  - Return the path-free evidence bundle result.

### Function `write_studio_evidence_bundle(context)`
Write a path-confined Studio evidence bundle into a job context.

Parameters
----------
context:
    Studio job context that owns the bundle files and enforces artifact
    path confinement, byte ceilings, and SHA-256 manifests.
project_payload:
    Optional saved Studio project payload from ``load_project``.
simulation_payloads:
    Optional Studio simulation responses carrying ``studio.simulation-run.v1``
    run metadata.
analysis_payloads:
    Optional Studio analysis responses carrying ``studio.analysis-result.v1``
    analysis metadata.
model_scan_payloads:
    Optional Studio model-scan responses carrying ``studio.model-scan.v1``
    scan metadata classified as analysis evidence.
weight_restore_payloads:
    Optional Studio training weight-restore responses carrying
    ``studio.training.weight-restore.v1`` materialization evidence classified
    as training evidence.
weight_restore_attach_payloads:
    Optional Studio training weight-restore attach responses carrying
    ``studio.training.weight-restore-attach.v1`` evidence classified as
    training evidence.
default_flow_runs:
    Optional guided default-flow run responses carrying reproducibility
    fingerprints.
default_flow_attestations:
    Optional guided default-flow attestations for the supplied run
    responses.
job_records:
    Completed or failed Studio job records to preserve with their declared
    artifacts. Artifacts ending in ``evidence.json`` must carry the
    ``studio.action-evidence.v1`` contract and are classified as
    first-class action evidence in the bundle manifest.
artifact_reader:
    Reader used to fetch verified job artifact bytes. Required when
    ``job_records`` contains artifacts.
audit_export:
    Optional path-free audit export payload.
command_replay:
    Optional JSON replay metadata, such as API method, route, request
    digest, and operator note.
clock:
    Optional UTC clock for deterministic tests.

Returns
-------
StudioEvidenceBundleResult
    Path-free manifest and artifact list for the generated bundle.

Raises
------
ValueError
    If replay metadata is not JSON-safe or job artifacts are supplied
    without an artifact reader.

---

## Module `studio.platform.identity`

### Class `StudioIdentityLifecycleError`
Raised when an identity mutation would break lifecycle invariants.


### Class `StudioIdentityRecord`
Persistent Studio service-account identity.

Parameters
----------
principal_id:
    Stable service-account identifier recorded in policy audit events.
roles:
    Role names granted to the service account.
token_sha256:
    Lowercase SHA-256 hex digest of the bearer token. Raw tokens are never
    stored in the identity file.
expires_at_utc:
    Optional UTC expiry instant. Expired records fail authentication.
active:
    Whether the identity is currently allowed to authenticate.

- **to_public_record**()
  - Return a token-free operator representation of this identity.

### Class `StudioIdentityPublicRecord`
Path-free and token-free service-account record for operators.

Parameters
----------
principal_id:
    Stable service-account identifier.
roles:
    Roles granted to the service account.
expires_at_utc:
    Optional UTC expiry instant formatted as an ISO timestamp.
active:
    Whether the service account can authenticate.

- **to_public_dict**()
  - Return an API payload without token hashes or local paths.

### Class `StudioBrowserUserRecord`
Persistent browser-login identity for Studio operators.

Parameters
----------
username:
    Stable browser-login username.
principal_id:
    Principal identifier recorded in audit events after login.
roles:
    Role names granted to the browser user.
password_pbkdf2_sha256:
    Encoded PBKDF2-HMAC-SHA256 password verifier.
expires_at_utc:
    Optional UTC expiry instant. Expired users cannot log in.
active:
    Whether the user can currently authenticate.

- **to_public_record**()
  - Return a password-free operator representation of this user.

### Class `StudioBrowserUserPublicRecord`
Path-free and password-free browser-user record for operators.

- **to_public_dict**()
  - Return an API payload without password verifier material.

### Class `StudioIdentityStore`
Validated Studio identity store loaded from a local JSON file.

- **public_records**()
  - Return token-free service-account records for admin APIs.
- **public_browser_users**()
  - Return password-free browser-user records for admin APIs.

### Class `StudioIdentityResult`
Result of authenticating one Studio authorization header.


### Class `StudioIdentityAuthenticator`
Authenticate Studio bearer tokens against a validated identity store.

- **__init__**(identity_store, clock)
- **authenticate_authorization_header**(authorization)
  - Authenticate an HTTP ``Authorization`` header.
- **authenticate_browser_user**(username, password)
  - Authenticate one browser user with username and password.

### Function `load_studio_identity_store(path)`
Load and validate a Studio identity store from JSON.

Parameters
----------
path:
    JSON file containing ``sc-neurocore.studio.identity.v1`` service
    account records.

Returns
-------
StudioIdentityStore
    Validated immutable service-account records.

Raises
------
ValueError
    If the file cannot be parsed into the supported identity schema.

### Function `list_studio_identity_public_records(path)`
Load token-free Studio service-account records from an identity file.

Parameters
----------
path:
    Persistent identity JSON file.

Returns
-------
tuple&#91;StudioIdentityPublicRecord, ...&#93;
    Public service-account records sorted by principal identifier.

### Function `list_studio_browser_user_public_records(path)`
Load password-free Studio browser-user records from an identity file.

Parameters
----------
path:
    Persistent identity JSON file.

Returns
-------
tuple&#91;StudioBrowserUserPublicRecord, ...&#93;
    Public browser-user records sorted by username.

### Function `update_studio_identity_record(path)`
Atomically update mutable service-account metadata.

The raw bearer-token SHA-256 digest is preserved and never returned.

Parameters
----------
path:
    Persistent identity JSON file.
principal_id:
    Existing service-account identifier to update.
roles:
    Replacement role set. The set must be non-empty.
active:
    Replacement active flag.
expires_at_utc:
    Optional replacement UTC expiry timestamp.

Returns
-------
StudioIdentityPublicRecord
    Updated token-free service-account record.

Raises
------
KeyError
    If ``principal_id`` is not present in the store.
ValueError
    If replacement metadata is malformed.

### Function `update_studio_browser_user_record(path)`
Atomically update mutable browser-user metadata.

The stored PBKDF2-HMAC-SHA256 password verifier is preserved and never
returned.

Parameters
----------
path:
    Persistent identity JSON file.
username:
    Existing browser-login username to update.
roles:
    Replacement role set. The set must be non-empty.
active:
    Replacement active flag.
expires_at_utc:
    Optional replacement UTC expiry timestamp.

Returns
-------
StudioBrowserUserPublicRecord
    Updated password-free browser-user record.

Raises
------
KeyError
    If ``username`` is not present in the store.
ValueError
    If replacement metadata is malformed.

### Function `rotate_studio_browser_user_password(path)`
Atomically rotate one browser user's password verifier.

Parameters
----------
path:
    Persistent identity JSON file.
username:
    Existing browser-login username to update.
password:
    New raw password supplied through an authenticated admin request.

Returns
-------
StudioBrowserUserPublicRecord
    Password-free browser-user record after verifier rotation.

Raises
------
KeyError
    If ``username`` is not present in the store.
ValueError
    If the username or replacement password is malformed.

### Function `add_studio_browser_user_record(path)`
Atomically add one persistent browser-login user to an identity file.

Parameters
----------
path:
    Persistent identity JSON file.
username:
    Unique browser-login username.
principal_id:
    Stable principal identifier recorded in policy audit events.
roles:
    Non-empty role set granted after login.
password:
    Raw password read from an operator-controlled secret channel.
active:
    Whether the new browser user can authenticate immediately.
expires_at_utc:
    Optional UTC expiry timestamp.

Returns
-------
StudioBrowserUserPublicRecord
    Password-free representation of the new browser user.

Raises
------
ValueError
    If the new user metadata is malformed or conflicts with an existing
    browser username.

### Function `make_browser_user_password_verifier(password)`
Create an encoded PBKDF2-HMAC-SHA256 password verifier.

Parameters
----------
password:
    Raw browser-user password.

Returns
-------
str
    Encoded verifier containing algorithm, iteration count, salt, and hash.

### Function `verify_browser_user_password(password, encoded_verifier)`
Verify a raw browser-user password against an encoded verifier.

---

## Module `studio.platform.jobs`

### Class `StudioJobRejected`
Raised when a Studio job request violates the local sandbox policy.


### Class `StudioJobCancelled`
Raised inside a cooperative Studio job when cancellation is requested.


### Class `StudioJobArtifactUnavailable`
Raised when a declared Studio job artifact cannot be safely served.


### Class `StudioJobArtifact`
Path-free manifest entry for one Studio job artifact.

Parameters
----------
relative_path:
    Artifact path relative to the job directory.
size_bytes:
    Number of bytes written to the artifact.
sha256:
    SHA-256 digest of the artifact payload.

- **to_public_dict**()
  - Return a path-free JSON representation of this artifact.

### Class `StudioJobRecord`
Immutable public state for one local Studio job.

- **to_public_dict**()
  - Return path-free job state suitable for operator APIs.

### Class `StudioJobResourceProfile`
Path-free execution limits for one Studio job kind.

Parameters
----------
kind:
    Job kind covered by this profile.
default_timeout_seconds:
    Default wall-clock timeout applied when a job omits an override.
max_artifact_bytes:
    Maximum size for each artifact written through ``StudioJobContext``.
execution_models:
    Supported manager execution models for this job kind.

- **to_public_dict**()
  - Return a JSON-serializable, path-free resource profile.

### Class `StudioJobStatusSnapshot`
Path-free aggregate health for the local Studio job manager.

- **to_public_dict**()
  - Return a JSON-serializable, path-free status snapshot.

### Class `StudioJobListSnapshot`
Path-free list payload for Studio job operator views.

- **to_public_dict**()
  - Return JSON-serializable job records without filesystem paths.

### Class `StudioJobArtifactPayload`
Verified payload for one declared Studio job artifact.


### Class `StudioJobContext`
Execution context passed to one local Studio job task.

- **__init__**()
- **cancelled**()
  - Return whether the manager requested cooperative cancellation.
- **artifacts**()
  - Return artifacts written through this context.
- **check_cancelled**()
  - Raise when the manager requested cooperative cancellation.
- **write_artifact**(relative_path, payload)
  - Write a confined artifact into the job directory.
- **append_artifact_event**(relative_path, payload)
  - Append one JSON event to a confined live artifact log.
- **publish_existing_artifact**(relative_path)
  - Declare an already-written confined artifact in the manifest.
- **read_seed_input**(relative_path)
  - Read one confined seed-input payload provided at job submission.
- **poll_control_command**()
  - Consume one pending control command delivered to a running job.
- **read_control_seed**(relative_path)
  - Read one confined seed payload delivered with a control command.

### Class `StudioJobManager`
Manage local Studio jobs inside per-job sandbox directories.

- **__init__**()
- **submit**()
  - Submit one local job to the bounded worker supervisor.
- **submit_process_task**()
  - Submit an importable Studio job task to an isolated Python process.
- **send_control_command**(job_id)
  - Deliver a control command and confined seeds to a running process job.
- **cancel**(job_id)
  - Request cooperative cancellation for one job.
- **wait**(job_id, timeout_seconds)
  - Wait for one job to finish and return its latest record.
- **record**(job_id)
  - Return the latest immutable record for one job.
- **list_records**()
  - Return all known jobs in creation order.
- **list_snapshot**()
  - Return a path-free snapshot of all known jobs.
- **purge_terminal_record**(job_id)
  - Delete one terminal job directory and remove its in-memory record.
- **read_artifact**(job_id, relative_path)
  - Return a verified payload for one declared job artifact.
- **read_live_artifact_bytes**(job_id, relative_path)
  - Return newly appended bytes from a confined live artifact.
- **status**()
  - Return aggregate, path-free job manager health.

---

## Module `studio.platform.operator`

### Class `StudioOperatorCapabilityStatus`
Aggregate health for the Studio capability registry.

- **to_public_dict**()
  - Return a public capability-health aggregate.

### Class `StudioOperatorIdentityStatus`
Path-free identity posture for Studio operator APIs.

- **to_public_dict**()
  - Return public identity posture without service-account material.

### Class `StudioOperatorRoutePolicyStatus`
Route-policy enforcement posture for Studio operator APIs.

- **to_public_dict**()
  - Return public route-policy posture.

### Class `StudioOperatorResourceLimitStatus`
Path-free runtime resource limits relevant to Studio operators.

- **to_public_dict**()
  - Return configured resource ceilings without host paths.

### Class `StudioOperatorBrowserLoginStatus`
Path-free browser-login lockout posture for Studio operators.

- **to_public_dict**()
  - Return browser-login lockout limits without identity material.

### Class `StudioOperatorStatus`
Path-free aggregate status for the Studio operator control plane.

- **to_public_dict**()
  - Return the path-free operator status API payload.

### Function `build_studio_operator_status()`
Build the aggregate operator status from live Studio platform components.

---

## Module `studio.platform.pipeline_process`

### Function `run_pipeline_process_task(context, payload)`
Run one Studio graph-to-synthesis pipeline in an isolated process.

Parameters
----------
context:
    Job sandbox used to write path-confined result and evidence artifacts.
payload:
    JSON object matching the public ``/api/pipeline/run`` request contract.

Returns
-------
dict&#91;str, object&#93;
    Path-free pipeline result payload.

Raises
------
ValueError
    If the payload does not match the JSON-serializable pipeline contract.

---

## Module `studio.platform.policy`

### Class `AuditSinkError`
Raised when a Studio audit sink cannot persist an event.


### Class `AuditSinkStatus`
Operator-safe status for a Studio audit sink.

- **to_public_dict**()
  - Return an operator-safe status dictionary without local paths.

### Class `AuditExport`
Operator-safe export of persisted Studio audit rows.

- **to_public_dict**()
  - Return a path-free JSON export payload for admin operators.

### Class `AuditQuarantineExport`
Operator-safe export of quarantined retained Studio audit rows.

- **to_public_dict**()
  - Return path-free quarantined audit rows for incident handoff.

### Class `RouteVisibility`
Visibility class for a Studio API route.


### Class `Principal`
Authenticated Studio caller identity.


### Class `RoutePolicy`
Authorization policy attached to one Studio route.

- **__post_init__**()
  - Validate fail-closed policy metadata.

### Class `PolicyDecision`
Authorization decision returned by the Studio policy gateway.


### Class `AuditEvent`
Audit event emitted for Studio policy decisions.

- **to_json_dict**()
  - Return a JSON-serializable representation of the audit event.

### Class `AuditSink`
Append-only sink for Studio policy audit events.

- **record**(event)
  - Record a Studio policy audit event.
- **status**()
  - Return operator-safe audit sink status.

### Class `InMemoryAuditSink`
Append-only in-memory audit sink for local Studio policy tests.

- **__init__**()
- **events**()
  - Return recorded audit events in insertion order.
- **record**(event)
  - Record a Studio policy audit event.
- **status**()
  - Return status for the non-persistent in-memory audit sink.

### Class `JsonlAuditSink`
Append-only JSONL audit sink for Studio policy decisions.

- **__init__**(path)
- **path**()
  - Return the configured JSONL audit log path.
- **record**(event)
  - Append a Studio policy audit event as one JSON object.
- **status**()
  - Return status for the persistent JSONL audit sink.
- **export_recent**(limit)
  - Export the most recent persisted audit rows without exposing paths.
- **export_quarantine**(limit)
  - Export quarantined retained audit rows without exposing local paths.

### Class `RoutePolicyRegistry`
Registry of Studio route policies keyed by HTTP method and path.

- **__init__**()
- **register**(method, path_template, policy)
  - Register one Studio route policy.
- **policy_for**(method, path_template)
  - Return the policy for one HTTP method and path template.
- **policies**()
  - Return registered route policies in stable method/path order.
- **missing_policies**(routes)
  - Return route signatures that have no registered Studio policy.

### Class `PolicyGateway`
Fail-closed Studio route authorization gateway.

- **__init__**(audit_sink, clock)
- **authorize**(policy)
  - Authorize a caller against a Studio route policy.

### Function `build_default_studio_route_policy_registry()`
Build route policies for the current Studio platform API surface.

---

## Module `studio.platform.preflight`

### Class `StudioPreflightCheck`
One secret-free Studio release-preflight check result.

Parameters
----------
check_id:
    Stable machine-readable identifier for the checked release invariant.
status:
    ``"pass"`` when the invariant holds, ``"fail"`` when it is violated and
    blocks release, or ``"warn"`` for a non-blocking advisory the operator
    should resolve before production use.
message:
    Operator-facing summary that does not include local paths or secrets.
evidence:
    Small scalar evidence fields suitable for JSON reports.
remediation:
    Operator actions that can resolve a failed or warned check. Entries
    must not expose local filesystem paths or secret material.

- **to_public_dict**()
  - Return a JSON-serializable, path-free check payload.

### Class `StudioPreflightReport`
Machine-readable Studio release-preflight report.

Parameters
----------
checks:
    Ordered release-readiness checks.
deployment_profile:
    Parsed Studio deployment profile when runtime settings were valid.
schema_version:
    Stable report schema identifier.

- **passed**()
  - Return whether no preflight check failed (warnings do not block).
- **warned**()
  - Return whether any check raised a non-blocking advisory warning.
- **to_public_dict**()
  - Return a JSON-serializable, secret-free preflight payload.

### Function `run_studio_preflight(env)`
Run Studio release-readiness checks from environment-style settings.

Parameters
----------
env:
    Optional environment mapping. When omitted, ``os.environ`` is used by
    the runtime settings builder.
clock:
    Optional UTC timestamp used for expiry-sensitive identity checks.

Returns
-------
StudioPreflightReport
    Ordered path-free report. The report fails closed if settings cannot
    be parsed or any release invariant is missing.

---

## Module `studio.platform.process_worker`

### Function `main(argv)`
Run one importable Studio process task and persist a JSON result.

Parameters
----------
argv:
    Optional command-line argument sequence. ``None`` reads process
    arguments from ``sys.argv`` through ``argparse``.

Returns
-------
int
    ``0`` when the imported task completed and wrote a result; ``1`` when
    the task failed and the result file contains the public error string.

---

## Module `studio.platform.sessions`

### Class `StudioBrowserSessionIssue`
Issued Studio browser session.

Parameters
----------
bearer_token:
    Raw bearer token returned once to the browser.
principal:
    Principal bound to the issued session.
expires_at_utc:
    UTC expiry timestamp for the session.

- **to_public_dict**()
  - Return the login response payload.

### Class `StudioBrowserSessionRecord`
Server-side browser session record stored without raw token material.


### Class `StudioBrowserSessionResult`
Result of authenticating a browser bearer-session token.


### Class `StudioBrowserSessionManager`
Issue, authenticate, and revoke ephemeral browser bearer sessions.

- **__init__**()
- **issue**(principal)
  - Issue a new browser session for an authenticated principal.
- **authenticate_authorization_header**(authorization)
  - Authenticate a bearer session from an HTTP ``Authorization`` header.
- **revoke_authorization_header**(authorization)
  - Revoke a browser bearer session if the header contains one.
- **revoke_principal**(principal_id)
  - Revoke all browser sessions for one principal identifier.
- **public_session**(authorization)
  - Return the current session payload without token material.

---

## Module `studio.platform.settings`

### Class `StudioRuntimeSettings`
Runtime settings consumed by the Studio FastAPI application.

- **__post_init__**()
  - Validate settings that affect Studio security boundaries.

### Function `build_default_studio_runtime_settings(env)`
Build Studio runtime settings from environment-style values.

---

## Module `studio.platform.synthesis_process`

### Function `run_synthesis_process_task(context, payload)`
Run one Studio synthesis request in an isolated process.

Parameters
----------
context:
    Job sandbox used to write path-confined result and evidence artifacts.
payload:
    JSON object matching the public ``/api/synth/run`` request contract.

Returns
-------
dict&#91;str, object&#93;
    Path-free synthesis result payload.

Raises
------
ValueError
    If the payload does not match the JSON-serializable synthesis contract.

### Function `run_multi_target_synthesis_process_task(context, payload)`
Run one Studio multi-target synthesis request in an isolated process.

Parameters
----------
context:
    Job sandbox used to write path-confined result and evidence artifacts.
payload:
    JSON object matching the public ``/api/synth/multi-target`` contract.

Returns
-------
dict&#91;str, object&#93;
    Path-free multi-target synthesis result payload.

Raises
------
ValueError
    If the payload does not match the JSON-serializable synthesis contract.

### Function `run_pnr_process_task(context, payload)`
Run one Studio PnR request in an isolated process.

Parameters
----------
context:
    Job sandbox used to write path-confined result and evidence artifacts.
payload:
    JSON object matching the public ``/api/synth/pnr`` request contract.

Returns
-------
dict&#91;str, object&#93;
    Path-free PnR result payload.

Raises
------
ValueError
    If the payload does not match the JSON-serializable PnR contract.

---

## Module `studio.platform.training_checkpoint`

### Class `StudioTrainingCheckpoint`
Path-free manifest for restoring a Studio Training Monitor run.

Parameters
----------
job_id:
    Source Training Monitor job ID.
config:
    JSON-serializable training configuration that can seed another Studio
    training run.
status:
    Source job status at export time.
final_metrics:
    Terminal metric map, when the source job reached a terminal state.
evidence_summary:
    Optional path-free terminal evidence summary from the source job.
weight_checkpoint:
    Optional path-free metadata for a job-managed binary weight artifact.
generated_at_utc:
    UTC timestamp for the export operation.
config_sha256:
    SHA-256 digest of the canonical configuration payload.
checkpoint_sha256:
    SHA-256 digest of the checkpoint payload excluding this digest field.

- **to_public_dict**()
  - Return the JSON-serializable checkpoint payload.

### Function `build_training_checkpoint()`
Build a portable Training Monitor checkpoint manifest.

Parameters
----------
job_id:
    Source Training Monitor job ID.
config:
    Training configuration to preserve.
status:
    Source job status at export time.
final_metrics:
    Optional terminal metrics from the training status response.
evidence_summary:
    Optional path-free terminal evidence summary.
weight_checkpoint:
    Optional path-free metadata for a job-managed binary weight artifact.
clock:
    Optional UTC timestamp override for deterministic tests.

Returns
-------
StudioTrainingCheckpoint
    Digest-backed checkpoint manifest suitable for API export.

Raises
------
ValueError
    If any supplied payload cannot be represented as portable JSON.

### Function `import_training_checkpoint_payload(payload)`
Validate a Training Monitor checkpoint import payload.

Parameters
----------
payload:
    JSON object supplied to ``POST /api/training/checkpoint/import``.

Returns
-------
dict&#91;str, JsonValue&#93;
    Path-free import result containing the restored training config and
    source checkpoint metadata.

Raises
------
ValueError
    If the checkpoint schema, hashes, or config payload are invalid.

---

## Module `studio.platform.training_evidence`

### Class `TrainingEvidenceSummary`
Path-free operator summary of one Training Monitor evidence artifact.

- **to_public_dict**()
  - Return the JSON-serializable evidence summary.

### Function `build_training_evidence_summary(record, artifact_reader)`
Return a verified Training Monitor evidence summary when available.

Parameters
----------
record:
    Path-free Studio job record that may declare the Training Monitor
    evidence artifact.
artifact_reader:
    Verified artifact reader, normally ``StudioJobManager.read_artifact``.

Returns
-------
dict&#91;str, object&#93; | None
    Public evidence summary for terminal training records, or ``None`` when
    the record does not declare a Training Monitor evidence artifact.

### Function `validate_training_evidence_summary(payload)`
Validate a portable Training Monitor evidence summary.

Parameters
----------
payload:
    Candidate ``studio.training.evidence-summary.v1`` object, normally
    embedded in a portable Training Monitor checkpoint.

Returns
-------
dict&#91;str, JsonValue&#93;
    JSON-compatible, validated evidence summary.

Raises
------
ValueError
    If the summary is not verified training evidence, uses an unsupported
    status or classification, or contains malformed artifact metadata.

---

## Module `studio.platform.training_process`

### Function `run_training_process_task(context, payload)`
Run one Studio training job in an isolated process.

Parameters
----------
context:
    Job sandbox used to write path-confined status and evidence artifacts.
payload:
    JSON object matching the public ``/api/training/start`` configuration
    contract.

Returns
-------
dict&#91;str, object&#93;
    Path-free terminal training metadata.

Raises
------
ValueError
    If ``payload`` is not a JSON object suitable for a training
    configuration.

### Function `run_training_attach_process_task(context, payload)`
Run a warm-start training job seeded with restored, verified weights.

The worker reads the confined seed weight and metadata inputs, verifies and
materializes them through the shared materializer, then runs a training job
that loads the restored state dictionary at the epoch-zero checkpoint
boundary before training forward. On success it writes a path-free
``studio.training.weight-restore-attach.v1`` evidence artifact recording the
verified digests, the resolved target architecture, and the architecture
fingerprint that gated compatibility. A strict load of an incompatible
architecture fails the job before training begins.

Parameters
----------
context:
    Job sandbox holding the confined seed inputs and artifact outputs.
payload:
    JSON object with ``config`` (the target training configuration),
    ``restore_plan`` (the path-free restore plan), and
    ``architecture_fingerprint``.

Returns
-------
dict&#91;str, object&#93;
    Path-free terminal training metadata augmented with the attach evidence.

Raises
------
ValueError
    If the payload is malformed or the restored weights are incompatible
    with the target architecture.

---

## Module `studio.platform.training_weight_loader`

### Function `load_training_weight_state_dict(payload)`
Deserialize a verified torch checkpoint payload into a state dictionary.

The loader is the trusted boundary that runs only after
:func:`sc_neurocore.studio.platform.training_weights.materialize_training_weight_payload`
has verified the payload digest and byte size against the restore plan. It
restricts unpickling to tensor storages and primitive containers through
``weights_only=True`` and rejects any payload that does not carry the
expected portable training checkpoint schema or a string-keyed model state
dictionary.

Parameters
----------
payload:
    Raw bytes of a ``studio.training.torch-state-dict.v1`` checkpoint that
    was produced by the Studio Training Monitor and already passed digest
    verification.

Returns
-------
Mapping&#91;str, object&#93;
    The string-keyed model state dictionary extracted from the checkpoint.

Raises
------
ValueError
    If the payload cannot be safely deserialized, does not carry the
    expected schema, or does not contain a string-keyed model state
    dictionary.

---

## Module `studio.platform.training_weights`

### Class `StudioTrainingWeightCheckpoint`
Path-free metadata for a terminal Training Monitor weight artifact.

Parameters
----------
framework:
    Framework used to serialize the weight payload.
format:
    Serialized payload format.
architecture:
    Human-readable model architecture summary.
parameter_count:
    Number of serialized model parameters.
config_sha256:
    SHA-256 digest of the canonical training configuration.
weights_artifact:
    Manifest entry for the binary weight artifact.
metadata_artifact:
    Manifest entry for the JSON metadata artifact.
final_metrics:
    Terminal metric payload associated with the weights.
schema_version:
    Schema identifier for this metadata contract.

- **to_public_dict**()
  - Return a path-free JSON-compatible checkpoint summary.

### Class `StudioTrainingWeightRestorePlan`
Path-free contract for reloading a Training Monitor weight artifact.

Parameters
----------
source_job_id:
    Studio job ID that owns the published weight artifacts.
source_status:
    Source training job status reported by the checkpoint import.
config_sha256:
    SHA-256 digest of the training configuration associated with weights.
framework:
    Framework expected by the weight payload.
format:
    Serialized payload format.
architecture:
    Human-readable model architecture summary.
parameter_count:
    Number of serialized model parameters.
weights_artifact:
    Path-free manifest entry for the binary weight artifact.
metadata_artifact:
    Path-free manifest entry for the JSON metadata artifact.
artifact_route_template:
    Authenticated artifact download route template for this Studio API.
loader_policy:
    Required loader trust boundary for clients that materialize weights.
schema_version:
    Schema identifier for this restore-plan contract.

- **to_public_dict**()
  - Return a JSON-compatible restore plan without local paths.

### Class `StudioTrainingWeightMaterialization`
Trusted in-memory materialization of verified training weights.

Parameters
----------
source_job_id:
    Studio job ID that owns the source weight artifacts.
config_sha256:
    SHA-256 digest of the training configuration associated with weights.
framework:
    Framework used by the trusted loader.
format:
    Serialized payload format consumed by the trusted loader.
architecture:
    Human-readable model architecture summary.
parameter_count:
    Number of parameters declared by the checkpoint metadata.
state_dict:
    In-memory state dictionary returned by the trusted loader.
weights_sha256:
    Verified SHA-256 digest of the binary weight payload.
metadata_sha256:
    Verified SHA-256 digest of the metadata payload.
schema_version:
    Schema identifier for this materialization contract.

- **to_public_dict**()
  - Return path-free materialization metadata without tensor payloads.

### Function `write_training_weight_checkpoint(context)`
Write binary training weights and path-free metadata artifacts.

Parameters
----------
context:
    Confined Studio job context that owns artifact publication.
weights_payload:
    Serialized weight payload. The context enforces the byte ceiling.
config:
    Training configuration used to produce the weights.
architecture:
    Human-readable architecture summary.
parameter_count:
    Number of model parameters represented by ``weights_payload``.
final_metrics:
    Terminal training metrics attached to the checkpoint metadata.

Returns
-------
StudioTrainingWeightCheckpoint
    Path-free summary suitable for status and checkpoint API payloads.

Raises
------
ValueError
    If the payload is empty, metadata is not portable JSON, or the context
    rejects either artifact.

### Function `build_training_weight_restore_plan()`
Build a safe restore plan from validated weight metadata.

Parameters
----------
source_job_id:
    Studio job ID that owns the published training weight artifacts.
source_status:
    Source training job status associated with the checkpoint.
weight_checkpoint:
    Path-free weight metadata from a portable training checkpoint.
expected_config_sha256:
    Optional training config digest that the metadata must match.

Returns
-------
StudioTrainingWeightRestorePlan
    Path-free restore plan that identifies the authenticated artifact route
    and digest checks required before a client materializes weights.

Raises
------
ValueError
    If source metadata is missing or the weight checkpoint is invalid.

### Function `materialize_training_weight_payload()`
Validate and materialize a Training Monitor weight payload in memory.

Parameters
----------
restore_plan:
    ``studio.training.weight-restore-plan.v1`` object produced by a
    validated checkpoint import.
metadata_payload:
    Raw bytes fetched from the authenticated metadata artifact route.
weights_payload:
    Raw bytes fetched from the authenticated weight artifact route.
trusted_loader:
    Loader that deserializes ``weights_payload`` after all schema, size,
    and digest checks pass. Production PyTorch integrations should use a
    loader that restricts deserialization to state dictionaries.

Returns
-------
StudioTrainingWeightMaterialization
    Verified, path-free in-memory materialization metadata plus the loaded
    state dictionary.

Raises
------
ValueError
    If the restore plan, metadata payload, artifact digests, artifact
    sizes, or loader output is invalid.

### Function `build_training_weight_restore_evidence(materialization)`
Wrap a verified weight materialization as path-free restore evidence.

The returned object is the ``studio.training.weight-restore.v1`` payload that
a restore job persists as a confined evidence artifact. It carries only the
verified digests, parameter counts, and loaded-key totals from the
materialization; the in-memory tensor state dictionary is never serialized.

Parameters
----------
materialization:
    Verified, path-free materialization returned by
    :func:`materialize_training_weight_payload`.
source_status:
    Terminal status of the source training job that owns the restored
    weight artifacts.

Returns
-------
dict&#91;str, JsonValue&#93;
    JSON-compatible restore evidence classified as training evidence with a
    completed terminal status.

Raises
------
ValueError
    If ``source_status`` is empty.

### Function `validate_training_weight_restore_evidence(payload)`
Validate a path-free ``studio.training.weight-restore.v1`` evidence object.

Parameters
----------
payload:
    Candidate restore evidence object from a Studio API response or import.

Returns
-------
dict&#91;str, JsonValue&#93;
    The JSON-compatible, validated restore evidence object.

Raises
------
ValueError
    If the schema, classification, status, source identifiers, or embedded
    materialization summary are invalid.

### Function `training_architecture_fingerprint(config)`
Return a SHA-256 fingerprint of a training config's architecture fields.

The fingerprint folds only the configuration fields that determine the model
state-dictionary shape (dataset, hidden layer widths, and the learnable
beta/threshold flags). Two configurations whose fingerprints match produce
architecturally compatible models, so the fingerprint identifies whether
restored weights can be attached to a target training configuration.

Parameters
----------
config:
    Training configuration following the ``/api/training/start`` contract.

Returns
-------
str
    SHA-256 hex digest of the canonical architecture field projection.

### Function `build_training_weight_restore_attach_evidence(materialization)`
Wrap a verified materialization as path-free attach evidence.

The returned ``studio.training.weight-restore-attach.v1`` object records that
verified weights were attached to a target training job. It carries only the
verified digests from the materialization, the target job identity, the
resolved target architecture, and the architecture fingerprint that gated
compatibility; the in-memory tensor state dictionary is never serialized.

Parameters
----------
materialization:
    Verified, path-free materialization returned by
    :func:`materialize_training_weight_payload`.
mode:
    Attach delivery mode. One of ``warm_start`` or ``live``.
target_job_id:
    Studio job ID that received the attached weights.
target_architecture:
    Resolved architecture summary of the target model.
target_parameter_count:
    Number of parameters in the target model.
architecture_fingerprint:
    SHA-256 architecture fingerprint that gated the attach.

Returns
-------
dict&#91;str, JsonValue&#93;
    JSON-compatible attach evidence classified as training evidence with a
    completed terminal status.

Raises
------
ValueError
    If the mode, target identifiers, or fingerprint are invalid.

### Function `validate_training_weight_restore_attach_evidence(payload)`
Validate a ``studio.training.weight-restore-attach.v1`` evidence object.

Parameters
----------
payload:
    Candidate attach evidence object from a Studio API response or import.

Returns
-------
dict&#91;str, JsonValue&#93;
    The JSON-compatible, validated attach evidence object.

Raises
------
ValueError
    If the schema, classification, status, mode, target identifiers,
    fingerprint, or embedded materialization summary are invalid.

### Function `validate_training_weight_checkpoint_metadata(payload)`
Validate imported Training Monitor weight metadata.

Parameters
----------
payload:
    Path-free weight metadata from a ``studio.training.checkpoint.v1``
    payload.
expected_config_sha256:
    Optional checkpoint configuration digest that must match the weight
    metadata configuration digest.

Returns
-------
dict&#91;str, JsonValue&#93;
    JSON-compatible, validated weight metadata.

Raises
------
ValueError
    If the schema, framework, format, config digest, artifact paths,
    artifact sizes, artifact hashes, or metadata payload are invalid.

---

## Module `studio.presets`

### Function `list_presets()`
Return public metadata for all curated Studio presets.

### Function `get_preset(preset_id)`
Return one curated Studio preset by identifier.

### Function `get_preset_actions(preset_id)`
Return validated action definitions for one curated Studio preset.

### Function `get_preset_action(preset_id, action_id)`
Return one action definition for a curated Studio preset.

### Function `list_preset_action_catalog()`
Return the route-facing catalogue of preset action identifiers.

---

## Module `studio.project`

### Function `save_project(name, state)`
Save full Studio state and return path-free evidence metadata.

### Function `load_project(name)`
Load a saved project by name.

### Function `list_projects()`
List all saved projects.

### Function `delete_project(name)`
Delete a saved project.

### Function `run_pipeline(graph, target)`
Run the Studio graph-to-synthesis pipeline.

If the graph has ODE equations in population params, the pipeline compiles
them to SystemVerilog and runs synthesis. Otherwise it uses a default LIF
compile path.

Parameters
----------
graph:
    Studio network graph payload.
target:
    Studio synthesis target identifier.
process_limits:
    Optional host-supported CPU and address-space ceilings for the
    downstream synthesis child process.

Returns
-------
dict&#91;str, Any&#93;
    Pipeline result containing validation, simulation, compile, and
    synthesis step payloads, or a bounded failure payload.

---

## Module `studio.project_manifest`

### Class `StudioProjectSaveManifest`
Path-free metadata returned after persisting a Studio project.

Parameters
----------
name:
    Sanitized project name.
saved_at:
    Unix timestamp stored in the project payload.
version:
    Studio project payload version.
state_sha256:
    SHA-256 digest of the canonical project state JSON.
project_sha256:
    SHA-256 digest of the canonical full project payload JSON.
evidence_classification:
    Stable evidence lane label for saved project workspaces.
status:
    Terminal status for this project-save evidence object.

- **to_public_dict**()
  - Return the public, path-free project save response.

### Function `build_project_save_manifest()`
Build digest-backed metadata for a persisted Studio project.

Parameters
----------
name:
    Sanitized project name.
saved_at:
    Unix timestamp stored in the project payload.
version:
    Studio project payload version.
state:
    Project state object persisted under ``state``.
project_payload:
    Full persisted project payload.

Returns
-------
StudioProjectSaveManifest
    Path-free metadata suitable for API responses, logs, and evidence
    manifests.

Raises
------
ValueError
    If the state or payload cannot be encoded as portable JSON.

### Function `dump_project_payload(payload)`
Return the durable JSON representation for a saved project payload.

The writer keeps the human-readable indentation used by existing Studio
project files, while rejecting non-standard JSON values such as NaN and
Infinity so saved projects remain portable across runtimes.

---

## Module `studio.simulation`

### Function `simulate(equations, threshold, reset, params, init, dt, duration, current, protocol, frequency_hz)`
Run an ODE neuron simulation and return time series data.

### Function `fi_curve(equations, threshold, reset, params, init, dt, duration, i_min, i_max, i_steps)`
Sweep current and compute firing rate at each level.

---

## Module `studio.simulation_manifest`

### Class `StudioSimulationRunManifest`
Path-free metadata for one Studio simulation result.

Parameters
----------
source:
    Simulation surface that produced the result.
input_sha256:
    SHA-256 digest of the canonical request payload.
result_sha256:
    SHA-256 digest of the canonical result payload without this manifest.
dt:
    Effective simulation time step in milliseconds.
n_steps:
    Number of executed simulation steps before plotting downsampling.
sample_count:
    Number of samples returned to the UI after plotting downsampling.
spike_count:
    Number of spikes detected during the simulation.
state_variables:
    Sorted state variable names returned in the result payload.
evidence_classification:
    Stable evidence lane label for simulation runs.
status:
    Terminal status for this simulation evidence object.

- **to_public_dict**()
  - Return the public, path-free simulation run manifest.

### Function `build_simulation_run_manifest()`
Build digest-backed reproducibility metadata for a simulation result.

Parameters
----------
source:
    Simulation surface that produced the result.
request_payload:
    Request body used to run the simulation.
result_payload:
    Result payload after classification, without relying on local paths.

Returns
-------
StudioSimulationRunManifest
    Path-free metadata suitable for UI display, exported JSON, and evidence
    bundles.

Raises
------
ValueError
    If request or result payloads cannot be encoded as portable JSON.

---

## Module `studio.svg_export`

### Function `traces_to_svg(time, states, spikes, model_name, dt, width, height)`
Generate publication-quality SVG from simulation traces.

Returns a complete SVG string with axes, grid, legend, spike markers,
and colour-coded polylines. Traces are downsampled to <=2000 points.

---

## Module `studio.synthesis`

### Class `EdaProcessLimits`
Optional process resource limits for external Studio EDA commands.

Parameters
----------
cpu_seconds:
    Maximum CPU seconds allowed for the child process on hosts that expose
    POSIX ``RLIMIT_CPU``. ``None`` leaves CPU accounting to the existing
    wall-clock timeout.
address_space_bytes:
    Maximum address space bytes allowed for the child process on hosts that
    expose POSIX ``RLIMIT_AS``. ``None`` leaves memory unconstrained by this
    helper.

- **__post_init__**()
  - Validate positive resource ceilings when they are configured.

### Function `check_tools()`
Detect which EDA tools are installed.

### Function `supported_targets()`
Return synthesis targets accepted by the Studio EDA routes.

### Function `run_synthesis(verilog_source, target)`
Run Yosys synthesis and return resource usage.

Parameters
----------
verilog_source:
    SystemVerilog or Verilog source text to synthesise.
target:
    Studio synthesis target identifier.
process_limits:
    Optional host-supported CPU and address-space ceilings for the Yosys
    child process.
tool_status:
    Optional path-free EDA tool status snapshot. When omitted, the backend
    captures a fresh snapshot for this result.

Returns
-------
dict&#91;str, Any&#93;
    Path-free synthesis result with success state, target, resource counts,
    capacity metadata, utilisation, or a bounded error message.

### Function `estimate_resources(ir_op_count, target)`
Quick resource estimate from IR operation count, no Yosys needed.

Heuristic: each IR op maps to ~2 LUTs + 1 FF on average.
LIF step op maps to ~12 LUTs + 8 FFs + 1 DSP (multiplier).

### Function `multi_target_synthesis(verilog_source)`
Run synthesis on all supported targets and return a comparison.

Parameters
----------
verilog_source:
    SystemVerilog or Verilog source text to synthesise.
process_limits:
    Optional host-supported CPU and address-space ceilings applied to every
    Yosys child process.

Returns
-------
dict&#91;str, Any&#93;
    Mapping with per-target synthesis results and the supported target list.

### Function `run_pnr(json_path, target)`
Run nextpnr place-and-route and return timing report.

Parameters
----------
json_path:
    Path to a Yosys JSON netlist. The path must point to a regular JSON
    file and must not be a symlink.
target:
    Studio target identifier with nextpnr support.
process_limits:
    Optional host-supported CPU and address-space ceilings for the nextpnr
    child process.

Returns
-------
dict&#91;str, Any&#93;
    Path-free PnR result with success state, timing metadata, log excerpt,
    or a bounded error message.

---

## Module `studio.synthesis_provenance`

### Class `StudioSynthesisToolProvenance`
Version and availability evidence for one synthesis-related tool.

Parameters
----------
key:
    Stable API key for the tool status entry.
executable:
    Command name used by the Studio backend.
role:
    Tool role in the synthesis workflow.
available:
    Whether the tool was detected by the backend.
version:
    First version line returned by the tool, when available.

- **to_public_dict**()
  - Return a path-free public tool provenance payload.

### Class `StudioSynthesisTargetProvenance`
Path-free provenance for one Studio synthesis target.

Parameters
----------
target:
    Studio target identifier.
capacity:
    Static capacity metadata used for resource utilisation.
synthesis_command:
    Yosys synthesis command selected for the target.
pnr_tool:
    Place-and-route command for the target, when one is configured.
device:
    Device selector passed to the PnR tool, when configured.
tools:
    Required tool provenance records for this target.
evidence_classification:
    Stable evidence lane label for synthesis target provenance.
status:
    Terminal status for this synthesis target provenance object.

- **provenance_grade**()
  - Return the claim grade for this target's synthesis provenance.
- **to_public_dict**()
  - Return the public, path-free target provenance payload.

### Function `build_synthesis_target_provenance(target)`
Build provenance metadata for one Studio synthesis target.

Parameters
----------
target:
    Studio target identifier.
target_config:
    Target configuration containing synthesis, PnR, and device selectors.
capacity:
    Static target capacity metadata.
tool_status:
    Path-free tool detection payload from ``check_tools``.

Returns
-------
StudioSynthesisTargetProvenance
    Path-free target provenance record.

Raises
------
ValueError
    If the target configuration lacks a synthesis command.

### Function `build_synthesis_target_provenance_matrix()`
Build a path-free provenance matrix for all Studio synthesis targets.

Parameters
----------
targets:
    Mapping of target identifiers to backend target configuration.
capacities:
    Mapping of target identifiers to static capacity metadata.
tool_status:
    Path-free tool detection payload from ``check_tools``.

Returns
-------
dict&#91;str, JsonValue&#93;
    Matrix payload keyed by target with a stable SHA-256 digest.

---

## Module `studio.templates`

### Function `list_templates()`
Return all curated Studio equation templates.

### Function `get_template(name)`
Return one curated Studio equation template by name.

---

## Module `studio.training`

### Class `TrainingJob`
Manage one Studio training run for thread or process execution.

- **__init__**(config)
- **start**()
  - Start this training job in its legacy background thread.
- **stop**()
  - Request cooperative cancellation for this training job.
- **run_blocking**(context)
  - Run this training job inside a bounded Studio job context.

### Function `list_surrogates()`
Return available surrogate-gradient functions for the Studio UI.

### Function `list_cell_types()`
Return available training cell types for the Studio UI.

### Function `start_training(config, job_manager)`
Start a Studio training job.

When ``job_manager`` is supplied, execution is delegated to the bounded
Studio job sandbox. Without it, the legacy in-process training thread is
used for direct module tests and backward compatibility.

### Function `start_training_attach(source_job_id, config, job_manager)`
Start a warm-start training job seeded with restored, verified weights.

Builds the canonical restore plan from the source training job's stored
checkpoint metadata, fetches the integrity-checked weight and metadata
artifacts, and submits a bounded process job that materializes and verifies
the weights before loading them into the target model at the epoch-zero
checkpoint boundary. The verified weights are delivered to the worker as
confined seed inputs, never through the API response.

Parameters
----------
source_job_id:
    Studio job ID of a completed training job that published weights.
config:
    Target training configuration for the warm-started job.
job_manager:
    Bounded Studio job manager that owns artifact reads and job submission.
expected_config_sha256:
    Optional source configuration digest that the checkpoint must match.

Returns
-------
dict&#91;str, Any&#93;
    ``{"job_id", "status"}`` for the warm-started job, or ``{"error"}`` when
    the source job is unknown or published no usable weight checkpoint.

Raises
------
ValueError
    If the restore plan cannot be built from the source checkpoint.

### Function `request_live_training_weight_attach(target_job_id, source_job_id, job_manager)`
Deliver verified weights to a running training job for a live attach.

Validates that the target job is running, that the source job published a
weight checkpoint, and that the source and target architectures are
compatible, then delivers the integrity-checked weight artifacts to the
running worker as a confined control command. The worker applies the attach
at its next epoch boundary; an incompatible attach is rejected without
interrupting the running job.

Parameters
----------
target_job_id:
    Studio job ID of the running training job to attach weights into.
source_job_id:
    Studio job ID of a completed training job that published weights.
job_manager:
    Bounded Studio job manager that owns artifact reads and control delivery.
expected_config_sha256:
    Optional source configuration digest that the checkpoint must match.

Returns
-------
dict&#91;str, Any&#93;
    ``{"target_job_id", "status", "architecture_fingerprint"}`` when the
    attach command is delivered, or ``{"error"}`` for a precondition failure.

Raises
------
ValueError
    If the restore plan cannot be built from the source checkpoint.

### Function `stop_training(job_id, job_manager)`
Request cooperative stop for a Studio training job.

### Function `get_training_status(job_id, job_manager)`
Return path-free status for one Studio training job.

### Function `stream_metrics(job_id, job_manager)`
Generator that yields SSE-formatted metric events.

### Function `list_jobs()`
Return path-free summaries for known Studio training jobs.

### Function `export_training_checkpoint(job_id, job_manager)`
Return a portable checkpoint for one Studio training job.

Parameters
----------
job_id:
    Training Monitor job identifier.
job_manager:
    Optional Studio job manager used to attach terminal worker evidence
    metadata when the job has reached a terminal state.

Returns
-------
dict&#91;str, Any&#93;
    `studio.training.checkpoint.v1` payload, or an error payload when the
    training job is unknown to the parent-process Training Monitor
    registry.

### Function `import_training_checkpoint(data)`
Validate a portable checkpoint and return its training config.

Parameters
----------
data:
    JSON object submitted to `/api/training/checkpoint/import`.

Returns
-------
dict&#91;str, Any&#93;
    Validated checkpoint import payload containing restored training
    configuration and source-job metadata.

Raises
------
ValueError
    If the checkpoint schema, config digest, or checkpoint digest is
    invalid.

---

## Module `swarm.agent`

### Class `AgentConfig`
Hyper-parameters for a single swarm agent.

The sensory layout reserves 20 channels for neighbour, obstacle, target,
field, emotional, and chemical inputs. Motor output must include at least
speed and turn channels.

- **__post_init__**()
  - Validate neural dimensions, dynamics, actuation, and seed domains.

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
  - Return the flat trainable-weight vector length.
- **weights**()
  - Return all trainable weights as a flat 1-D vector.
- **weights**(flat)
  - Replace trainable weights from a finite exact-size vector.
- **think**(sensory)
  - Run one SNN tick and return ``(speed, turn_angle)``.
- **act**(speed, turn)
  - Update position and heading given finite motor commands.
- **reset**(rng, width, height)
  - Reset kinematic and neural state while preserving trainable weights.

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
Neuroevolution hyper-parameters for homogeneous swarm agents.

- **__post_init__**()
  - Validate population, selection, mutation, evaluation, and seed domains.

### Class `SwarmEvolver`
Genetic algorithm that evolves SNN weights for swarm control.

Parameters
----------
cfg : EvolverConfig
    Evolution and evaluation parameters.

- **__init__**(cfg)
- **evaluate_individual**(weights)
  - Create environment, inject *weights* into every agent, run, score.
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

- **__post_init__**()
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

- **__post_init__**()
  - Validate the conductance and rectification parameters.
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

- **__post_init__**()
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

- **__post_init__**()
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
- **train**(x_batch, y_batch)
  - Train on a mini-batch using the EP two-phase protocol.
- **predict**(x, n_settle)
  - Predict output by settling in free phase.
- **get_params**()
  - Return serialisable parameter dict.

---

## Module `training.loops`

### Function `auto_device()`
Select a usable supported device in priority order: CUDA, MPS, CPU.

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


### Function `relaxed_sc_linear(input_value, weight, bias, sc_config)`
Linear layer whose multiply-accumulate path uses relaxed SC products.

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

### Function `write_stochastic_backprop_export_manifest(path, benchmark_report, sc_config)`
Write a canonical stochastic backpropagation SC-NIR export manifest.

### Function `write_stochastic_backprop_handoff_bundle(export_manifest, output_dir)`
Materialise an auditable SC-NIR HDL handoff bundle from an export manifest.

---

## Module `training.surrogate`

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
- **get**(name)
  - Get recorded spikes for a named module. Returns (T, *shape) or None.
- **layer_names**()
  - Return names of modules that currently have spike hooks attached.
- **reset**()
  - Clear recorded spike tensors while keeping hooks attached.
- **remove**()
  - Remove forward hooks and clear all recorded spike tensors.

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
Decode spike counts with a population-vector weighted average.

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
Complete dense-weight SNN checkpoint for transfer workflows.

Parameters
----------
weights:
    One two-dimensional weight matrix per layer. A matrix shape must be
    ``(output_features, input_features)`` for the matching ``layer_sizes``
    entry ``(input_features, output_features)``.
layer_names:
    Unique layer names in forward order.
layer_sizes:
    ``(input_features, output_features)`` pairs for each layer.
neuron_types:
    Optional neuron-model labels, either empty or one label per layer.
metadata:
    JSON-serializable provenance or training metadata.
frozen_layers:
    Layer names currently marked non-trainable.

- **__post_init__**()
  - Normalize arrays and reject inconsistent checkpoint state.
- **n_layers**()
  - Return the number of serialized layers.
- **total_params**()
  - Return the total number of scalar weight parameters.

### Function `save_checkpoint(checkpoint, path)`
Save an SNN checkpoint to ``path.npz`` plus ``path.json``.

Parameters
----------
checkpoint:
    Validated checkpoint to serialize.
path:
    Base path without extension. Parent directories are created.

### Function `load_checkpoint(path)`
Load and validate an SNN checkpoint from ``path.npz`` and ``path.json``.

Parameters
----------
path:
    Base path without extension.

Returns
-------
SNNCheckpoint:
    Reconstructed checkpoint with finite ``float64`` weight arrays.

---

## Module `transfer.fine_tune`

### Class `TransferConfig`
Configuration for checkpoint-based SNN transfer learning.

Parameters
----------
freeze_until:
    Freeze all layers up to and including this layer name or index. ``-1``
    means do not add frozen layers.
lr_backbone:
    Learning rate for frozen backbone layers, usually zero or a small value.
lr_head:
    Learning rate for unfrozen task-head layers.

- **__post_init__**()
  - Reject invalid freeze targets and learning rates.

### Function `freeze_layers(checkpoint, layer_names, until_index)`
Mark checkpoint layers as frozen.

Parameters
----------
checkpoint:
    Checkpoint to mutate.
layer_names:
    Specific layer names to freeze.
until_index:
    Freeze every layer with index less than or equal to this value.

Returns
-------
SNNCheckpoint:
    The same checkpoint object with ``frozen_layers`` updated.

### Function `unfreeze_layers(checkpoint, layer_names, all_layers)`
Mark checkpoint layers as trainable.

Parameters
----------
checkpoint:
    Checkpoint to mutate.
layer_names:
    Specific layer names to unfreeze.
all_layers:
    When true, clear every frozen-layer marker.

Returns
-------
SNNCheckpoint:
    The same checkpoint object with ``frozen_layers`` updated.

### Function `apply_transfer_config(checkpoint, config)`
Apply a transfer config and return per-layer learning rates.

Parameters
----------
checkpoint:
    Checkpoint to mutate according to ``config``.
config:
    Validated transfer schedule.

Returns
-------
tuple&#91;SNNCheckpoint, list&#91;float&#93;&#93;:
    The mutated checkpoint and one learning rate per layer.

---

## Module `transformers.block`

### Class `StochasticTransformerBlock`
Spiking transformer block (S-Former).

Structure: input -> multi-head attention -> add & norm -> feed forward
-> add & norm -> output.

- **__post_init__**()
  - Validate the block dimensions and build the attention heads and FFN.
- **forward**(x)
  - Run the block on input of shape (d_model,) or (seq_len, d_model).

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
  - Derive the head dimension and initialise the projection weights.
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
  - Initialise the discretised state-space matrices.
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
  - Sample the central-pattern-generator oscillator frequencies.
- **encode**(seq_len)
  - Generate positional encoding.
- **encode_spikes**(seq_len, rng)
  - Generate spike-encoded positional encoding.

---

## Module `utils.adapter_discovery`

### Function `discover_adapters()`
Discover adapter classes and register them in the global registry.

Parameters
----------
include_first_party : bool, default=True
    Register the built-in adapter importers declared by
    :data:`FIRST_PARTY_ADAPTERS`.  This source-level path keeps editable
    checkouts wired even before packaging metadata is installed.
include_entry_points : bool, default=True
    Load installed third-party plugins from
    :data:`ADAPTER_ENTRY_POINT_GROUP` through ``importlib.metadata``.

Returns
-------
dict&#91;str, type&#93;
    Mapping from registry name to the discovered adapter class. Duplicate
    registry entries are tolerated so repeated discovery is idempotent.

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
  - Encode one scalar into a stochastic bitstream.
- **decode**(bitstream)
  - Decode a stochastic bitstream back into the configured value range.

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
  - Add one binary sample to the sliding window.
- **estimate**()
  - Return the current sliding-window probability estimate.
- **reset**()
  - Clear all buffered samples and reset the estimator state.

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
probabilities : np.ndarray&#91;Any, Any&#93;
    Probability matrix, any shape. Values in &#91;0, 1&#93;.
length : int
    Bitstream length per element.
method : str
    "sobol" or "halton".
seed : int or None
    Random seed for scrambling.

Returns
-------
np.ndarray&#91;Any, Any&#93;
    Shape (*probabilities.shape, length), dtype uint8.

### Function `star_discrepancy_estimate(samples, n_test)`
Estimate star discrepancy of a sample set (quality metric for LDS).

Lower discrepancy → more uniform coverage → better SC precision.

Parameters
----------
samples : np.ndarray&#91;Any, Any&#93;
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
Deterministic wrapper around NumPy's per-instance random generator.

The wrapper is intentionally small: it exposes the distributions used by
stochastic-computing utilities while enforcing scalar parameter domains
before NumPy mutates generator state. Scalar draws return Python scalars;
shaped draws return dtype-stable NumPy arrays.

Example
-------
>>> rng = RNG(seed=42)
>>> vals = rng.random(5)
>>> vals.shape
(5,)
>>> RNG(seed=42).random(5) == vals  # deterministic
array(&#91; True,  True,  True,  True,  True&#93;)

- **__init__**(seed)
  - Create an independent stream from a non-negative integer seed.
- **normal**(mean, std, size)
  - Draw samples from a normal distribution.
- **uniform**(low, high, size)
  - Draw samples from a bounded uniform distribution.
- **bernoulli**(p, size)
  - Draw Bernoulli samples from a validated probability.
- **random**(size)
  - Draw uniform samples from the half-open interval ``&#91;0, 1)``.
- **shuffle**(x)
  - Shuffle an array in place using this instance's random stream.

---

## Module `uvm_gen.uvm_gen`

### Class `PortDirection`
SystemVerilog port direction tokens accepted by the UVM generator.


### Class `PortType`
SystemVerilog net/data type tokens emitted for module ports.


### Class `ModulePort`
Parsed RTL port metadata used by generated UVM components.

- **sv_decl**()
  - Return the SystemVerilog declaration for this parsed port.
- **is_clock**()
  - Return whether the port name matches a supported clock convention.
- **is_reset**()
  - Return whether the port name matches a supported reset convention.

### Class `ModuleParam`
RTL module parameter.


### Class `RTLModule`
Parsed RTL module specification.

- **from_verilog_source**(cls, source)
  - Parse a Verilog/SystemVerilog module header.
- **input_ports**()
  - Input ports excluding generated clock and reset controls.
- **output_ports**()
  - Output ports monitored by the generated scoreboard and coverage.
- **clock_port**()
  - First parsed clock-like port, if the RTL module declares one.
- **reset_port**()
  - First parsed reset-like port, if the RTL module declares one.
- **total_input_bits**()
  - Total data-input width excluding clock and reset ports.
- **total_output_bits**()
  - Total output width observed by generated verification components.

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
  - Return generated artefacts keyed by their output filenames.

### Class `UVMGenerator`
Generates complete UVM verification IP from an RTL module spec.

- **__init__**(stimulus, coverage, scoreboard)
- **generate**(rtl)
  - Generate the complete UVM testbench for an RTL module.
- **generate_multi**(modules)
  - Generate UVM testbenches for multiple modules.
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
common dangerous patterns but cannot prove semantic safety, model data
flow, or reason about values assembled before screening. Do not use as a
security boundary without additional sandboxing.

- **verify_code_safety**(source_code)
  - Return whether ``source_code`` passes the static blocklist.
- **verify_logic_invariant**(func, input_sample, expected_condition)
  - Return whether a dynamic invariant holds for one input sample.

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

### Function `publication_grade_snn_standard_profile()`
Return the default formal SNN verification standard profile.

### Function `assess_snn_verification_standard(evidence, profile)`
Assess evidence against the default or supplied SNN verification profile.

---

## Module `verification.temporal_properties`

### Class `PropertyResult`
Outcome states returned by temporal property verification checks.


### Class `Counterexample`
Input that violates a property.


### Class `VerificationResult`
Result of a temporal property check.

- **summary**()
  - Format the verification outcome and optional counterexample.

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
trains : list of np.ndarray&#91;Any, Any&#93;
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
  - Return the latent state dimension.
- **obs_dim**()
  - Return the observation dimension.
- **control_dim**()
  - Return the control-input dimension.
- **__post_init__**()
  - Validate the state-space matrix dimensions.
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
  - Run RTS smoothing over a forward-filter result.

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
  - Initialise the underlying linear-Gaussian state-space model.
- **reset**()
  - Reset the running belief state to the prior mean.
- **predict_next_state**(current_state, action)
  - Predict E&#91;x_{t+1} | x_t, u_t&#93; under the SSM dynamics.
- **predict_next_state_with_cov**(current_state, current_cov, action)
  - Predict mean + covariance of x_{t+1} given (x_t, Σ_t, u_t).
- **forecast**(initial_state, actions)
  - Multi-step deterministic forecast (mean trajectory).
- **forecast_with_cov**(initial_state, initial_cov, actions)
  - Multi-step probabilistic forecast (mean + cov trajectory).

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
  - Seed the RNG and initialise the predictor weights.
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
