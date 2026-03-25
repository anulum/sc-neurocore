# Adaptive Precision

Per-layer adaptive bitstream length for mixed-precision SC networks.

- `AdaptivePrecisionManager` — Auto-select bitstream length per layer (Hoeffding/Chebyshev/sensitivity bounds). Layers needing high precision get longer bitstreams; tolerant layers get shorter ones.

```python
from sc_neurocore.compiler.adaptive_precision import AdaptivePrecisionManager
```

::: sc_neurocore.compiler.adaptive_precision
    options:
      show_root_heading: true
