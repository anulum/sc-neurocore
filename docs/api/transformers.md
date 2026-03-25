# Transformers

SC-native transformer blocks built on stochastic attention.

- `StochasticTransformerBlock` — S-Former: spiking transformer with bitstream multi-head attention. Architecture: Input -> SC Multi-Head Attention -> Add & Norm -> SC Dense FF -> Add & Norm -> Output. Softmax approximated via CORDIV.

```python
from sc_neurocore.transformers import StochasticTransformerBlock

block = StochasticTransformerBlock(d_model=64, n_heads=4, length=256)
output = block.forward(input_sequence)
```

See [Tutorial 54: Spiking Transformers](../tutorials/54_spikformer.md).

::: sc_neurocore.transformers.block
    options:
      show_root_heading: true
