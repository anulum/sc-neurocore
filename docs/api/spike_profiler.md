# Spike-Level Training Profiler

Live training diagnostics: dead neurons, gradient pathology, saturated layers,
energy bottlenecks. The first automated SNN training profiler in any framework.

## Profiler

::: sc_neurocore.profiling.spike_profiler
    options:
      show_root_heading: true
      members:
        - SpikeProfiler
        - ProfileReport
        - LayerStats
        - Pathology
        - Severity
