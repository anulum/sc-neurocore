# Transfer Learning

Save, load, freeze, fine-tune SNN models.

```python
from sc_neurocore.transfer import save_checkpoint, load_checkpoint, freeze_layers

save_checkpoint(model, "model_v1")
model = load_checkpoint("model_v1")
freeze_layers(model, layers=["conv1", "conv2"])
```

Checkpoints are stored as a `model_v1.npz` weight archive plus a `model_v1.json`
metadata file. Loading validates the JSON metadata schema, rejects unexpected
archive members, and opens `.npz` weights with pickle disabled.

See [Tutorial 81: Transfer Learning](../tutorials/81_transfer.md).

::: sc_neurocore.transfer
    options:
      show_root_heading: true
