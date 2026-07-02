# Pipeline

Data ingestion and training orchestration for SNN workflows.

- `DataIngestor` — Validated multimodal dataset preparation: min-max
  normalizes each modality to `[0, 1]`, preserves the reserved `labels` field
  as labels, and rejects empty, scalar, non-finite, or mismatched sample axes.
- `SCTrainingLoop` — Standard and RL training orchestration with logging, checkpointing, and early stopping

```python
from sc_neurocore.pipeline import DataIngestor, SCTrainingLoop

dataset = DataIngestor().prepare_dataset(
    {"vision": [[0.0, 1.0], [2.0, 3.0]], "labels": [0, 1]}
)
sample = dataset.get_sample(0)
```

::: sc_neurocore.pipeline
    options:
      show_root_heading: true
