# Differential Privacy

Spike-level differential privacy for training and inference.

- `DPSNNTrainer` — Train SNNs with differential privacy guarantees (epsilon-delta). Clips per-sample gradients, adds calibrated noise. Privacy budget tracking via moments accountant.
- `DPInferenceWrapper` — Add noise to spike outputs for private inference.

Spike-domain DP is cheaper than ANN DP because spikes are already binary — clipping is free.

```python
from sc_neurocore.privacy import DPSNNTrainer
```

See [Tutorial 62: Differential Privacy](../tutorials/62_privacy.md).

::: sc_neurocore.privacy
    options:
      show_root_heading: true
