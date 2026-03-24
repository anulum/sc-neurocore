# Spike-Level Debugger

Temporal spike debugger: trace execution, find divergence, analyze causality.

## Tracer

::: sc_neurocore.debug.tracer
    options:
      show_root_heading: true
      members:
        - SpikeTracer
        - ExecutionTrace

## Analyzer

::: sc_neurocore.debug.analyzer
    options:
      show_root_heading: true
      members:
        - find_divergence
        - causal_chain
        - spike_diff
        - DivergencePoint
        - CausalEvent
