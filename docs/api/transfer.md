# Transfer Learning

The transfer package provides validated checkpoint serialization plus layer
freezing helpers for SNN transfer-learning workflows.

```python
import numpy as np
from sc_neurocore.transfer import (
    SNNCheckpoint,
    TransferConfig,
    apply_transfer_config,
    freeze_layers,
    load_checkpoint,
    save_checkpoint,
)

checkpoint = SNNCheckpoint(
    weights=[
        np.ones((32, 64), dtype=np.float64),
        np.ones((10, 32), dtype=np.float64),
    ],
    layer_names=["hidden", "output"],
    layer_sizes=[(64, 32), (32, 10)],
    neuron_types=["LIF", "LIF"],
    metadata={"task": "mnist"},
)

save_checkpoint(checkpoint, "model_v1")
checkpoint = load_checkpoint("model_v1")
freeze_layers(checkpoint, layer_names=["hidden"])
checkpoint, learning_rates = apply_transfer_config(
    checkpoint,
    TransferConfig(freeze_until=0, lr_backbone=0.0, lr_head=0.01),
)
```

Checkpoints are stored as a `model_v1.npz` weight archive plus a `model_v1.json`
metadata file. Loading validates the JSON metadata schema, rejects unexpected
archive members, opens `.npz` weights with pickle disabled, rejects non-finite
weights, and checks every matrix against its `(input_features, output_features)`
layer-size contract.

## Validation Surface

| Surface | Contract |
| --- | --- |
| Python | Constructor and loader reject duplicate layer names, shape mismatches, non-finite weights, unknown frozen layers, invalid learning rates, and non-JSON metadata. |
| Rust | `src/sc_neurocore/accel/rust/safety/checkpoint.rs` and `fine_tune.rs` compile as standalone safety mirrors with unit tests. |
| Julia | `src/sc_neurocore/accel/julia/transfer/checkpoint.jl` and `fine_tune.jl` validate the same in-memory checkpoint and transfer schedule invariants. |
| Mojo | `src/sc_neurocore/accel/mojo/kernels/checkpoint.mojo` and `fine_tune.mojo` run deterministic validation kernels. |

## Local Evidence

`benchmarks/results/bench_transfer.json` records local, non-isolated regression
evidence. The 2026-06-27 run reports:

| Check | Result |
| --- | --- |
| Python checkpoint roundtrip | 100 calls in 0.154779 s, 646.081 calls/s |
| Python `apply_transfer_config` | 100 calls in 0.009153 s, 10925.32 calls/s |
| Rust checkpoint compile/tests | pass |
| Rust fine-tune compile/tests | pass |
| Julia validation | pass |
| Mojo checkpoint/fine-tune validation | pass |

These timings are regression evidence only; the artifact marks
`production_benchmark_claim` as `false`.

See [Tutorial 81: Transfer Learning](../tutorials/81_transfer.md).

::: sc_neurocore.transfer
    options:
      show_root_heading: true
