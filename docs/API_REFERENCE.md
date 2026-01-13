# SC-NeuroCore API Reference

## 1. Neurons (`sc_neurocore.neurons`)

### `StochasticLIFNeuron`
**Path:** `sc_neurocore.neurons.stochastic_lif`

A stochastic Leaky Integrate-and-Fire neuron model.

**Parameters:**
- `v_rest` (float): Resting potential (default: 0.0)
- `v_reset` (float): Reset potential (default: 0.0)
- `v_threshold` (float): Firing threshold (default: 1.0)
- `tau_mem` (float): Membrane time constant (default: 20.0)
- `dt` (float): Time step (default: 1.0)
- `noise_std` (float): Standard deviation of Gaussian noise (default: 0.0)
- `resistance` (float): Membrane resistance (default: 1.0)
- `seed` (int | None): Random seed

**Methods:**
- `step(input_current: float) -> int`: Updates state and returns 1 if spike, 0 otherwise.
- `reset_state() -> None`: Resets membrane potential to `v_rest`.
- `get_state() -> Dict`: Returns current state (e.g., `{'v': ...}`).

## 2. Synapses (`sc_neurocore.synapses`)

### `BitstreamSynapse`
**Path:** `sc_neurocore.synapses.sc_synapse`

Implements stochastic multiplication via bitstream ANDing.

**Parameters:**
- `w_min` (float): Minimum representable weight.
- `w_max` (float): Maximum representable weight.
- `length` (int): Bitstream length (default: 256).
- `w` (float): Initial weight value.

**Methods:**
- `update_weight(new_w: float) -> None`: Updates weight and regenerates bitstream.
- `apply(pre_bits: np.ndarray) -> np.ndarray`: Performs element-wise AND with input bitstream.
- `effective_weight_probability() -> float`: Returns the empirical probability of the weight bitstream.

### `BitstreamDotProduct`
**Path:** `sc_neurocore.synapses.dot_product`

Computes the dot product of inputs and weights in the stochastic domain.

**Methods:**
- `apply(pre_matrix, y_min, y_max) -> (post_matrix, y_scalar)`: Applies synapses to a matrix of inputs and estimates the scalar result.

## 3. Sources (`sc_neurocore.sources`)

### `BitstreamCurrentSource`
**Path:** `sc_neurocore.sources.bitstream_current_source`

Generates input currents from multiple stochastic channels.

**Parameters:**
- `x_inputs` (List[float]): Input scalar values.
- `weight_values` (List[float]): Synaptic weights.
- `length` (int): Bitstream length.

**Methods:**
- `step() -> float`: Returns the instantaneous current estimate for the current time step.

## 4. Utilities (`sc_neurocore.utils`)

### `BitstreamEncoder`
**Path:** `sc_neurocore.utils.bitstreams`

Encodes scalars into Bernoulli bitstreams.

**Methods:**
- `encode(x: float) -> np.ndarray`: Returns a bitstream.
- `decode(bitstream: np.ndarray) -> float`: Reconstructs the scalar value.

### `BitstreamAverager`
**Path:** `sc_neurocore.utils.bitstreams`

Rolling window average for bitstreams.

**Methods:**
- `push(bit: int)`: Add bit to buffer.
- `estimate() -> float`: Get current moving average.

## 5. Recorders (`sc_neurocore.recorders`)

### `BitstreamSpikeRecorder`
**Path:** `sc_neurocore.recorders.spike_recorder`

Records spike events.

**Methods:**
- `record(spike: int)`: Log a spike.
- `firing_rate_hz() -> float`: Calculate firing rate.
- `isi_histogram(bins) -> (hist, edges)`: Compute Inter-Spike Interval histogram.
