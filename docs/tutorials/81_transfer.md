# Tutorial 81: SNN Transfer Learning

Pretrain, save, load, freeze, fine-tune.

```python
from sc_neurocore.transfer import save_checkpoint, load_checkpoint, SNNCheckpoint, TransferConfig
from sc_neurocore.transfer.fine_tune import apply_transfer_config

# Save trained model
ckpt = SNNCheckpoint(weights=model_weights, layer_names=['h1','out'], layer_sizes=[(784,256),(256,10)])
save_checkpoint(ckpt, 'mnist_snn')

# Load and fine-tune
ckpt = load_checkpoint('mnist_snn')
config = TransferConfig(freeze_until=0, lr_head=0.001)
ckpt, per_layer_lr = apply_transfer_config(ckpt, config)
```
