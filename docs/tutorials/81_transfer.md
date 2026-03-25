# Tutorial 81: SNN Transfer Learning

Pretrain, save, load, freeze, fine-tune. The standard ML workflow adapted
for spiking neural networks.

## Save and Load Checkpoints

```python
from sc_neurocore.transfer import (
    save_checkpoint, load_checkpoint, SNNCheckpoint, TransferConfig,
)

# Save trained model
ckpt = SNNCheckpoint(
    weights=model_weights,
    layer_names=["h1", "out"],
    layer_sizes=[(784, 256), (256, 10)],
)
save_checkpoint(ckpt, "mnist_snn")

# Load checkpoint
ckpt = load_checkpoint("mnist_snn")
```

## Freeze and Fine-Tune

```python
from sc_neurocore.transfer.fine_tune import apply_transfer_config

# Freeze all layers except the last (readout)
config = TransferConfig(freeze_until=0, lr_head=0.001)
ckpt, per_layer_lr = apply_transfer_config(ckpt, config)

# per_layer_lr: [0.0, 0.001] — first layer frozen, second trains
```

## Transfer Learning Workflow

1. **Pretrain** on large dataset (e.g., MNIST 60K examples)
2. **Save** checkpoint with `save_checkpoint()`
3. **Load** on new task with `load_checkpoint()`
4. **Freeze** feature extraction layers
5. **Fine-tune** readout layer on new task (few examples suffice)

SNN transfer learning preserves learned temporal dynamics — spike timing
patterns transfer across tasks, not just weight magnitudes.

## API Reference

::: sc_neurocore.transfer
    options:
      show_root_heading: true
