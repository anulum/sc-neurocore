# Event-Driven Asynchronous Simulation

Priority-queue-based simulation: only neurons with pending events are updated.
10,000x speedup for sparse networks.

::: sc_neurocore.event_driven.simulator
    options:
      show_root_heading: true
      members:
        - EventDrivenSimulator
        - SpikeEvent
        - EventStats
