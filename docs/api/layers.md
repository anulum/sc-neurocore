# Layers

Pre-built layer compositions combining neurons, synapses, encoders,
and recorders into reusable building blocks.

| Class | Architecture | Backend |
|-------|-------------|---------|
| `SCDenseLayer` | Fully-connected LIF | NumPy (loop-based) |
| `VectorizedSCLayer` | Fully-connected, packed bitwise | NumPy / CuPy GPU |
| `SCConv2DLayer` | 2D convolution | NumPy |
| `SCRecurrentLayer` | Echo state / reservoir | NumPy |
| `SCLearningLayer` | Dense + online STDP | NumPy |
| `SCFusionLayer` | Multi-modal MUX fusion | NumPy |
| `StochasticAttention` | SC attention mechanism | NumPy |
| `MemristiveDenseLayer` | Memristive device model | NumPy |
| `JaxSCDenseLayer` | Fully-connected LIF | JAX (JIT, GPU/TPU) |
| `HardwareAwareSCLayer` | Dense + memristive defects | NumPy |
| `PredictiveCodingSCLayer` | XOR error, zero-multiplication | NumPy |
| `RallDendrite` | Compartmental dendritic tree | NumPy |
| `LateralInhibition` | Gaussian surround suppression | NumPy |
| `WinnerTakeAll` | k-WTA competitive layer | NumPy |

## Dense Layer

`SCDenseLayer` accepts either a 1-D `weight_values` vector, which creates one
shared stochastic-computing current source for all neurons, or a 2-D
`(n_neurons, n_inputs)` weight matrix, which creates one SC dot-product source
per output neuron.

::: sc_neurocore.layers.sc_dense_layer.SCDenseLayer

## Vectorized Layer

::: sc_neurocore.layers.vectorized_layer.VectorizedSCLayer

## Convolutional Layer

`SCConv2DLayer` applies deterministic NumPy accumulation over stochastic
probability encodings. Unipolar mode accepts finite channel-first tensors in
`[0, 1]`; bipolar mode accepts finite tensors in `[-1, 1]`. Construction and
runtime validation fail closed for invalid channel counts, kernel geometry,
stride, padding, bitstream length, input rank, input domain, and empty output
geometry before the im2col convolution path runs.

::: sc_neurocore.layers.sc_conv_layer.SCConv2DLayer

## Recurrent / Reservoir Layer

::: sc_neurocore.layers.recurrent.SCRecurrentLayer

## Learning Layer

::: sc_neurocore.layers.sc_learning_layer.SCLearningLayer

## Fusion Layer

::: sc_neurocore.layers.fusion.SCFusionLayer

## Attention Layer

`StochasticAttention` exposes three attention paths:

- `forward(...)` uses SC-native row-sum normalisation without exponentials;
- `forward_softmax(...)` uses temperature-scaled softmax with stable
  max-subtraction;
- `forward_bitstream(...)` encodes `Q`, `K`, and `V` as unipolar
  probabilities, computes dot products with bitstream AND gates, and decodes
  the row-normalised result by popcount.

Bitstream mode accepts Bernoulli streams by default. Passing
`use_sobol=True` uses Sobol low-discrepancy streams for lower deterministic
variance at the same bitstream length. All bitstream inputs must be finite
probabilities in `[0, 1]`, and `length` must be positive.

The public contract is tested as follows:

- invalid `dim_k`, temperature, and unsupported `sc_mode` fail closed;
- 1-D and 2-D `Q`, `K`, and `V` inputs produce finite outputs with the
  expected `(n_queries, value_dim)` shape;
- single-key attention returns the value row for both row-sum and softmax
  modes;
- softmax remains finite on large scores and sharp temperatures select the
  best-matching key;
- bitstream mode rejects invalid lengths and out-of-range probabilities;
- Bernoulli bitstream attention approximates row-sum attention within its
  stochastic tolerance;
- Sobol bitstream attention returns finite bounded probabilities.

::: sc_neurocore.layers.attention.StochasticAttention

## Memristive Layer

::: sc_neurocore.layers.memristive.MemristiveDenseLayer

## JAX Dense Layer

`JaxSCDenseLayer` accepts dense inputs shaped `(n_inputs,)` or `(T, n_inputs)`
and projects them through a validated `(n_neurons, n_inputs)` weight matrix.
It also accepts direct current vectors shaped `(n_neurons,)` or
`(T, n_neurons)` for low-level LIF experiments. Constructor and runtime inputs
validate dimensions, finite values, dense weight shape, known neuron parameter
keys, and JAX PRNG seed range before backend execution.

The class is exported through the lazy package facades as
`sc_neurocore.JaxSCDenseLayer` and `sc_neurocore.layers.JaxSCDenseLayer`.
Importing the symbol does not construct backend state; constructing the class
still requires the `jax` optional dependency group.

::: sc_neurocore.layers.jax_dense_layer.JaxSCDenseLayer

## Hardware-Aware SC Layer

`HardwareAwareSCLayer` wraps `VectorizedSCLayer` with deterministic
memristive-defect injection. A seeded `stuck_mask` selects defective
synapses, each defective synapse is forced to a sampled stuck-at value
(`0.0` or `1.0`), and `update_weights(...)` masks gradients at those
locations so training can only adapt the remaining synapses.

The public contract is tested as follows:

- `forward(...)` preserves the vectorised layer output shape;
- non-zero `stuck_rate` creates observable stuck synapses;
- stuck weights remain unchanged after gradient updates;
- non-stuck weights update when gradients are applied;
- all weights stay clipped to `[0, 1]`;
- `stuck_rate=0.0` produces no stuck synapses.

::: sc_neurocore.layers.hardware_aware.HardwareAwareSCLayer

## Predictive Coding SC Layer (Conjecture C9)

`PredictiveCodingSCLayer` models prediction error in the stochastic-computing
domain: predicted and actual Bernoulli bitstreams are compared with XOR, and
the popcount of the error stream gives the per-neuron surprise magnitude.
Learning moves each prediction weight toward the observed input probability,
so repeated exposure to a stable pattern should lower mean prediction error
while a switched pattern should raise surprise.

The public contract is tested as follows:

- `forward(...)` returns `prediction_error`, per-neuron `surprises`, and
  `(n_neurons, n_inputs)` `predictions`;
- prediction error and surprise values remain in `[0, 1]`;
- learned weights remain clipped to `[0, 1]`;
- repeated inputs reduce prediction error on average;
- novel inputs after training produce larger surprise than familiar inputs;
- `reset()` restores the seeded initial weights.

::: sc_neurocore.layers.predictive_coding.PredictiveCodingSCLayer

## Rall Branching Dendrite

`RallDendrite` represents each branch as a fixed-length compartment chain.
Inputs are injected at the distal compartment, coupled toward the proximal
compartment, and accumulated at the soma through attenuation factors derived
from Rall's 3/2 branch-diameter rule.

The public contract is tested as follows:

- the initial and reset states have zero compartment and soma voltages;
- distal input reaches the soma after repeated steps;
- activating more branches increases somatic voltage relative to a single
  active branch under the same parameters;
- soma voltage decays when input is removed;
- `branch_voltages` returns a copy shaped `(n_branches, branch_length)`;
- Rall attenuation factors remain normalised;
- distal compartments remain higher than proximal compartments during
  sustained distal injection.

::: sc_neurocore.layers.rall_dendrite.RallDendrite

## Lateral Inhibition

::: sc_neurocore.layers.circuit_primitives.LateralInhibition

## Winner-Take-All

::: sc_neurocore.layers.circuit_primitives.WinnerTakeAll
