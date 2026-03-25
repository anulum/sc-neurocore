# Transfer Learning

Save, load, freeze, fine-tune SNN models.

```python
from sc_neurocore.transfer import save_checkpoint, load_checkpoint, freeze_layers

save_checkpoint(model, "model_v1.npz")
model = load_checkpoint("model_v1.npz")
freeze_layers(model, layers=["conv1", "conv2"])
```

See [Tutorial 81: Transfer Learning](../tutorials/81_transfer.md).

::: sc_neurocore.transfer
    options:
      show_root_heading: true
