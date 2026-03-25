# Platform Profiler

Cross-platform SNN performance profiler (CPU, GPU, Rust, simulated FPGA).

```python
from sc_neurocore.profiler import PlatformProfiler

profiler = PlatformProfiler()
report = profiler.profile(model, inputs, backends=["python", "rust"])
```

See [Tutorial 43: Platform Profiler](../tutorials/43_platform_profiler.md).

::: sc_neurocore.profiler
    options:
      show_root_heading: true
