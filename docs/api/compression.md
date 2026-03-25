# SNN Model Compression

Weight pruning, structural pruning, stochastic-aware pruning, and quantization for FPGA cost reduction.

## Pruning

Three pruning strategies:

- `prune_weights` — Magnitude-based: zero out weights with |w| below threshold. Standard approach.
- `prune_neurons` — Structural: remove entire neurons with low firing rates, reducing layer width (not just sparsity).
- `prune_stochastic` — **SC-specific**: score weights by bitstream contribution. Weights near 0 or 1 produce near-deterministic bitstreams (low entropy) and can be replaced with constant gates. Importance = `min(p, 1-p) * bitstream_length`.

```python
from sc_neurocore.compression import prune_stochastic

# Prune weights contributing <1 popcount bit per inference
pruned, report = prune_stochastic(weights, bitstream_length=256, min_popcount_bits=1.0)
print(f"Sparsity: {report.sparsity:.1%}")
```

::: sc_neurocore.compression.pruning
    options:
      show_root_heading: true
      members:
        - prune_weights
        - prune_neurons
        - prune_stochastic
        - PruningReport

## Quantization

::: sc_neurocore.compression.quantization
    options:
      show_root_heading: true
      members:
        - quantize_weights
        - quantize_delays
