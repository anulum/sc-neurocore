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
| `StochasticAttentionLayer` | SC attention mechanism | NumPy |
| `MemristiveSCLayer` | Memristive device model | NumPy |
| `JaxSCDenseLayer` | Fully-connected LIF | JAX (JIT, GPU/TPU) |

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

::: sc_neurocore.layers.memristive.MemristiveSCLayer

## JAX Dense Layer

::: sc_neurocore.layers.jax_dense_layer.JaxSCDenseLayer
