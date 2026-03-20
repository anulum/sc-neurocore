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

::: sc_neurocore.layers.sc_dense_layer.SCDenseLayer

## Vectorized Layer

::: sc_neurocore.layers.vectorized_layer.VectorizedSCLayer

## Convolutional Layer

::: sc_neurocore.layers.sc_conv_layer.SCConv2DLayer

## Recurrent / Reservoir Layer

::: sc_neurocore.layers.recurrent.SCRecurrentLayer

## Learning Layer

::: sc_neurocore.layers.sc_learning_layer.SCLearningLayer

## Fusion Layer

::: sc_neurocore.layers.fusion.SCFusionLayer

## Attention Layer

::: sc_neurocore.layers.attention.StochasticAttention

## Memristive Layer

::: sc_neurocore.layers.memristive.MemristiveDenseLayer

## JAX Dense Layer

::: sc_neurocore.layers.jax_dense_layer.JaxSCDenseLayer

## Hardware-Aware SC Layer

Trains around memristive defects (stuck-at faults) by masking gradients
on defective synapses.

::: sc_neurocore.layers.hardware_aware.HardwareAwareSCLayer

## Predictive Coding SC Layer (Conjecture C9)

Zero-multiplication predictive coding: XOR = error, popcount = magnitude,
STDP = precision. First SC implementation of Bayesian prediction error minimization.

::: sc_neurocore.layers.predictive_coding.PredictiveCodingSCLayer

## Rall Branching Dendrite

Compartmental dendritic tree with Rall's 3/2 power rule for impedance matching.
Distal-to-proximal propagation with inter-compartment coupling.

::: sc_neurocore.layers.rall_dendrite.RallDendrite

## Lateral Inhibition

::: sc_neurocore.layers.circuit_primitives.LateralInhibition

## Winner-Take-All

::: sc_neurocore.layers.circuit_primitives.WinnerTakeAll
