# Evolutionary Substrate

Self-replicating evolutionary SC substrate. Networks emit mutated child
networks as NIR or Verilog, deployed to FPGA tiles for hardware-speed
open-ended evolution with formal safety guards.

20 industrial features: genome mutation/crossover, speciation, novelty search,
island model, CPPN developmental encoding, Pareto front, co-evolution,
FPGA tile deployment, extinction events, bloat control, and more.

## Quick Start

```python
from sc_neurocore.evo_substrate.evo_substrate import (
    Genome, MutationEngine, CrossoverEngine, FitnessEvaluator,
    ReplicationEngine, OrganismEmitter, SafetyBounds,
    TileDeploymentTracker, HallOfFame, IslandModel,
    NoveltyArchive, FormalSafetyGuard, ParetoFront,
    CPPNGenome, ComplexityTracker,
)
```

::: sc_neurocore.evo_substrate.evo_substrate
