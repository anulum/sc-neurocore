# Sensors and DVS Pipeline

Event camera (DVS) data loading, preprocessing, and spike encoding. Supports AEDAT, HDF5, NumPy.

```python
from sc_neurocore.sensors import DVSPipeline

pipeline = DVSPipeline(resolution=(240, 180))
events = pipeline.load("recording.aedat")
```

See [Tutorial 45: DVS Pipeline](../tutorials/45_dvs_pipeline.md).

::: sc_neurocore.sensors
    options:
      show_root_heading: true
