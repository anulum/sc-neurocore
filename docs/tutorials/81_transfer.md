# Tutorial 81: SNN Transfer Learning

Pretrain, save, load, freeze, fine-tune. The standard ML transfer
learning workflow adapted for spiking neural networks — where temporal
dynamics transfer across tasks, not just weight magnitudes.

## Why Transfer Learning for SNNs

SNN training is expensive: surrogate gradients through T timesteps,
slow convergence, GPU-hungry. Transfer learning amortises this cost:

1. Train once on a large dataset (MNIST, ImageNet-derived spikes)
2. Save the trained weights
3. Fine-tune on your specific task with few examples

The key SNN-specific insight: learned *temporal dynamics* (membrane
decay rates, threshold adaptation patterns) transfer across tasks.
A network trained on MNIST has learned general spike-timing features
that help with any visual task.

## Save and Load Checkpoints

```python
import numpy as np
from sc_neurocore.transfer import (
    save_checkpoint, load_checkpoint, SNNCheckpoint, TransferConfig,
)

rng = np.random.default_rng(42)

# Simulate trained weights
model_weights = [
    rng.standard_normal((784, 256)).astype(np.float32) * 0.05,
    rng.standard_normal((256, 10)).astype(np.float32) * 0.1,
]

# Save trained model
ckpt = SNNCheckpoint(
    weights=model_weights,
    layer_names=["hidden", "output"],
    layer_sizes=[(784, 256), (256, 10)],
    metadata={
        "task": "mnist",
        "accuracy": 0.972,
        "timesteps": 25,
        "beta": 0.9,
    },
)
save_checkpoint(ckpt, "mnist_snn")
print(f"Saved: {len(ckpt.weights)} layers, {sum(w.size for w in ckpt.weights):,} params")

# Load checkpoint
loaded = load_checkpoint("mnist_snn")
print(f"Loaded: task={loaded.metadata.get('task')}, accuracy={loaded.metadata.get('accuracy')}")
```

## Freeze and Fine-Tune

Freeze feature extraction layers, train only the readout:

```python
from sc_neurocore.transfer.fine_tune import apply_transfer_config

# Freeze all layers except the last
config = TransferConfig(
    freeze_until=0,      # freeze layers 0..0 (hidden layer)
    lr_head=0.001,       # learning rate for unfrozen head
    lr_frozen=0.0,       # frozen layers don't train
)

ckpt, per_layer_lr = apply_transfer_config(loaded, config)
print(f"Per-layer LR: {per_layer_lr}")
# [0.0, 0.001] — hidden frozen, output trains
```

### Gradual Unfreezing

For better fine-tuning, gradually unfreeze layers:

```python
# Phase 1: only readout (10 epochs)
config1 = TransferConfig(freeze_until=0, lr_head=0.001)

# Phase 2: unfreeze hidden layer with small LR (10 more epochs)
config2 = TransferConfig(freeze_until=-1, lr_head=0.001, lr_frozen=0.0001)
```

## Transfer Learning Workflow

```
Step 1: Pretrain on large dataset
  └─ MNIST 60K examples, 25 timesteps, 50 epochs → 97.2% accuracy

Step 2: Save checkpoint
  └─ save_checkpoint(ckpt, "mnist_pretrained")

Step 3: Load on new task
  └─ ckpt = load_checkpoint("mnist_pretrained")

Step 4: Modify architecture for new task
  └─ Replace output layer: 10 → 5 classes (new task)
  └─ Keep hidden layer weights from MNIST

Step 5: Freeze + fine-tune
  └─ Freeze hidden, train output on 100 examples of new task
  └─ Result: 90%+ accuracy with only 100 examples
```

## What Transfers in SNNs

| Component | What Transfers | Why It Helps |
|-----------|---------------|-------------|
| Synaptic weights | Learned feature detectors | Same edges/textures in different tasks |
| Beta (leak rate) | Temporal integration timescale | Similar input dynamics |
| Threshold | Activity level calibration | Network excitability |
| Weight structure | Sparse connectivity pattern | Efficient information routing |

## Comparison

| Feature | SC-NeuroCore | snnTorch | Norse |
|---------|:-----------:|:--------:|:-----:|
| Save/load checkpoints | Yes | Manual (PyTorch) | Manual (PyTorch) |
| TransferConfig | Yes | No | No |
| Freeze/unfreeze API | Yes | Manual | Manual |
| Metadata preservation | Yes | No | No |
| SC weight export | Yes | No | No |

## References

- Yosinski et al. (2014). "How transferable are features in deep
  neural networks?" NeurIPS 2014.
- Dampfhoffer et al. (2022). "Are SNNs really more energy-efficient
  than ANNs? An in-depth hardware-aware study." IEEE TETCI.
