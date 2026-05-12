# Transformers

SC-native transformer blocks built on stochastic attention.

- `StochasticTransformerBlock` — S-Former: spiking transformer with
  per-head stochastic attention over disjoint feature subspaces.
  Architecture: Input -> SC Multi-Head Attention -> Add & Norm -> SC
  Dense FF -> Add & Norm -> Output. `d_model` must be divisible by
  `n_heads`; each head owns `d_model / n_heads` contiguous channels.
  Inputs must be finite one- or two-dimensional arrays with trailing
  dimension `d_model`.

```python
from sc_neurocore.transformers import StochasticTransformerBlock

block = StochasticTransformerBlock(d_model=64, n_heads=4, length=256)
output = block.forward(input_sequence)
```

See [Tutorial 54: Spiking Transformers](../tutorials/54_spikformer.md).

::: sc_neurocore.transformers.block
    options:
      show_root_heading: true
