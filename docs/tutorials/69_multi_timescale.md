# Tutorial 69: Multi-Timescale SNN

Different layers run at different temporal resolutions.

## HetSyn Layer

Per-synapse time constants (log-normal, matching Allen Institute data):

```python
from sc_neurocore.temporal_hierarchy import HetSynLayer

layer = HetSynLayer(n_inputs=64, n_neurons=32, tau_mean=5.0, tau_std=1.0)
print(layer.tau_stats)
# {'mean': 5.2, 'std': 4.1, 'min': 0.5, 'max': 42.3}
```

## Multi-Clock Network

Fast sensory (1x), medium cognitive (5x), slow decision (10x):

```python
from sc_neurocore.temporal_hierarchy import MultiClockSNN, HetSynLayer

net = MultiClockSNN(
    layers=[
        HetSynLayer(64, 32, tau_mean=2.0),   # fast
        HetSynLayer(32, 16, tau_mean=10.0),   # medium
        HetSynLayer(16, 4, tau_mean=50.0),    # slow
    ],
    layer_names=["sensory", "cognitive", "decision"],
    clock_intervals=[1, 5, 10],  # ticks per step
)

outputs = net.run(inputs)  # (T, 4)
```
