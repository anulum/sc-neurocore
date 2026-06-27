# Tutorial 81: SNN Transfer Learning

Pretrain, save, load, freeze, and fine-tune an SNN checkpoint. The workflow is
designed for reproducible transfer experiments where the checkpoint carries both
weights and enough architecture metadata to reject incompatible reloads.

## Save and Load Checkpoints

```python
import numpy as np
from sc_neurocore.transfer import SNNCheckpoint, load_checkpoint, save_checkpoint

rng = np.random.default_rng(42)

model_weights = [
    rng.standard_normal((256, 784)).astype(np.float64) * 0.05,
    rng.standard_normal((10, 256)).astype(np.float64) * 0.1,
]

checkpoint = SNNCheckpoint(
    weights=model_weights,
    layer_names=["hidden", "output"],
    layer_sizes=[(784, 256), (256, 10)],
    neuron_types=["LIF", "LIF"],
    metadata={
        "task": "mnist",
        "accuracy": 0.972,
        "timesteps": 25,
        "beta": 0.9,
    },
)
save_checkpoint(checkpoint, "mnist_snn")

loaded = load_checkpoint("mnist_snn")
print(loaded.n_layers, loaded.total_params, loaded.metadata["task"])
```

The loader checks the JSON sidecar, rejects unexpected archive entries, disables
pickle while opening the `.npz`, and verifies that each weight matrix shape is
`(output_features, input_features)` for the stored layer-size pair.

## Freeze and Fine-Tune

Freeze feature-extraction layers and keep the output head trainable:

```python
from sc_neurocore.transfer import TransferConfig, apply_transfer_config

config = TransferConfig(
    freeze_until=0,
    lr_backbone=0.0,
    lr_head=0.001,
)

loaded, per_layer_lr = apply_transfer_config(loaded, config)
print(loaded.frozen_layers)
print(per_layer_lr)
```

`freeze_until` accepts either a zero-based layer index or a layer name. Unknown
layer names and out-of-range indices raise `ValueError` instead of silently
leaving the schedule unchanged.

## Gradual Unfreezing

```python
from sc_neurocore.transfer import TransferConfig, apply_transfer_config, unfreeze_layers

phase_1 = TransferConfig(freeze_until=0, lr_backbone=0.0, lr_head=0.001)
loaded, phase_1_rates = apply_transfer_config(loaded, phase_1)

unfreeze_layers(loaded, layer_names=["hidden"])
phase_2 = TransferConfig(freeze_until=-1, lr_backbone=0.0001, lr_head=0.001)
loaded, phase_2_rates = apply_transfer_config(loaded, phase_2)
```

The helpers mutate the checkpoint and return it for call chaining. Save the
checkpoint again after each phase when you need an auditable recovery point.

## Transfer Workflow

1. Pretrain on the source task.
2. Save a validated `SNNCheckpoint`.
3. Load the checkpoint for the target task.
4. Replace or reinitialize task-specific head weights outside the checkpoint
   helper.
5. Freeze the transferred backbone and assign per-layer learning rates.
6. Fine-tune, unfreeze gradually when needed, then save the adapted checkpoint.

## What Transfers in SNNs

| Component | What Transfers | Why It Helps |
| --- | --- | --- |
| Synaptic weights | Feature detectors and temporal filters | Preserves learned spike-response structure. |
| Layer metadata | Layer order, sizes, and neuron labels | Prevents incompatible checkpoint reuse. |
| Frozen-layer markers | Fine-tuning schedule state | Makes transfer phases reproducible. |
| Experiment metadata | Task, accuracy, timestep, or provenance fields | Keeps downstream analysis tied to the saved state. |

## Evidence

`benchmarks/results/bench_transfer.json` records local, non-isolated regression
evidence for the Python API and polyglot mirrors. The 2026-06-27 run reports:

| Check | Result |
| --- | --- |
| Python checkpoint roundtrip | 100 calls in 0.154779 s, 646.081 calls/s |
| Python `apply_transfer_config` | 100 calls in 0.009153 s, 10925.32 calls/s |
| Rust checkpoint and fine-tune mirrors | compile and unit tests pass |
| Julia transfer validation | pass |
| Mojo checkpoint and fine-tune kernels | pass |

The benchmark artifact is regression evidence only and sets
`production_benchmark_claim` to `false`.

## References

- Yosinski et al. (2014). "How transferable are features in deep neural
  networks?" NeurIPS 2014.
- Dampfhoffer et al. (2022). "Are SNNs really more energy-efficient than ANNs?
  An in-depth hardware-aware study." IEEE TETCI.
