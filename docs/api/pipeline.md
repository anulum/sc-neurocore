# Pipeline

Data ingestion and training orchestration for SNN workflows.

- `DataIngestor` — Multimodal dataset preparation: spike encoding, batching, augmentation
- `SCTrainingLoop` — Standard and RL training orchestration with logging, checkpointing, and early stopping

```python
from sc_neurocore.pipeline import DataIngestor, SCTrainingLoop
```

::: sc_neurocore.pipeline
    options:
      show_root_heading: true
