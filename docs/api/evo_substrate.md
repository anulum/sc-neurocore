# Evolutionary Substrate

Self-replicating evolutionary substrate for stochastic computing neural
networks. Organisms are compact ``Genome`` structures (topology +
neuron kinetics + plasticity parameters) that mutate, recombine,
speciate, migrate between islands, and deploy onto FPGA tiles. Fitness
evaluation can be hooked directly into wet-lab MEA recordings via
:func:`sc_neurocore.bioware.bioware.mea_fitness_hook`.

```python
from sc_neurocore.evo_substrate.evo_substrate import (
    Genome, NeuronGene, TopologyGene, PlasticityGene,
    MutationEngine, CrossoverEngine, FitnessEvaluator,
    ReplicationEngine, OrganismEmitter, SafetyBounds,
    TileDeploymentTracker, HallOfFame, IslandModel,
    NoveltyArchive, FormalSafetyGuard, ParetoFront,
    CPPNGenome, ComplexityTracker, BloatPenalizer,
    ExtinctionDetector, LineageTracker,
)
```

---

## 1. Genome

A ``Genome`` is the evolving individual. It serialises to a fixed-length
float vector (see :meth:`Genome.to_vector` / :meth:`Genome.from_vector`)
so downstream optimisers can operate on a flat search space.

| Field           | Type             | Meaning                                                        |
| --------------- | ---------------- | -------------------------------------------------------------- |
| `genome_id`     | `str`            | 12-char SHA-256 fingerprint via :meth:`compute_id`             |
| `parent_id`     | `str`            | genome_id of the parent (empty at generation 0)                |
| `generation`    | `int`            | generation counter                                             |
| `topology`      | `TopologyGene`   | connection probability + topology scalars                      |
| `neuron`        | `NeuronGene`     | kinetic parameters (``tau_fast``, ``tau_work``, ``tau_deep``…) |
| `plasticity`    | `PlasticityGene` | plasticity scalars (STDP / BCM / ELIGENT)                      |
| `weight_seed`   | `int`            | deterministic PRNG seed for weight initialisation              |
| `identity_deep` | `float`          | snapshot of the parent's ``v_deep`` at birth                   |

The three gene sub-dataclasses each expose ``to_vector`` / ``from_vector``
round-trips; concatenated they form the genome vector whose length
equals :attr:`Genome.vector_dim`.

---

## 2. Mutation + crossover

- :class:`MutationType` — enum: `POINT`, `STRUCTURAL`, `DUPLICATION`,
  `SWAP`, `IDENTITY`.
- :class:`MutationConfig` — per-type rates, Gaussian σ, structural
  intensity bounds.
- :class:`MutationEngine` — applies a :class:`MutationConfig` to a
  :class:`Genome`; deterministic under a fixed ``seed``.
- :class:`CrossoverEngine` — crossover of two genomes; respects
  gene-block boundaries (topology | neuron | plasticity).

Helper functions:

- :func:`genomic_distance(a, b)` — Euclidean distance in vector space.
- :func:`assign_species(population, threshold)` — greedy species
  partitioning by pairwise distance (NEAT-style).
- :func:`population_diversity(population)` — mean pairwise distance as a
  population-level novelty proxy.

---

## 3. Fitness

- :class:`FitnessType` — enum: `ACCURACY`, `NOVELTY`, `MULTI_OBJECTIVE`,
  `HARDWARE_COST`.
- :class:`FitnessResult` — typed score record with auxiliary metrics.
- :class:`FitnessEvaluator` — scorer over the population; accepts a
  ``metrics_fn`` callable. Plug
  :func:`sc_neurocore.bioware.bioware.mea_fitness_hook` for closed-loop
  wet-lab scoring, or a pure-software proxy for simulation-only runs.

:class:`ReplicationEngine` orchestrates the full generation: selection
from the current population via the evaluator, parent crossover,
mutation, safety gating, tile allocation. See :class:`ReplicationEngine`
for the step API.

---

## 4. Population control

| Class                      | Responsibility                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| :class:`IslandModel`       | N isolated sub-populations with periodic migration; counteracts premature convergence.    |
| :class:`NoveltyArchive`    | Keeps a sparse archive of behaviourally distinct genomes; feeds novelty-based fitness.    |
| :class:`HallOfFame`        | Top-K elites retained across generations, used as crossover parents.                      |
| :class:`ParetoFront`       | Non-dominated front for multi-objective scoring (``MULTI_OBJECTIVE`` fitness).            |
| :class:`LineageTracker`    | Records parent → child edges as :class:`LineageRecord` nodes for post-hoc phylogeny.      |
| :class:`BloatPenalizer`    | Penalises genomes whose effective complexity grows without accuracy improvement.          |
| :class:`ComplexityTracker` | Measures structural complexity (active connections, unique neuron types).                 |
| :class:`ExtinctionDetector`| Triggers mass-extinction events when population diversity collapses below a threshold.    |
| :class:`FormalSafetyGuard` | Validates every proposed genome against hard invariants (tau positivity, connection-rate bounds, Lyapunov-bounded plasticity) before replication. |

---

## 5. Developmental encoding — `CPPNGenome`

:class:`CPPNGenome` encodes a genome as a compositional pattern-producing
network (CPPN) that generates per-connection weights from coordinate
inputs, rather than storing them as a flat vector. Useful for
symmetry-exploiting topology searches — the CPPN is mutated / crossed
over directly; the expanded weight matrix is produced on demand.

---

## 6. Hardware deployment

- :class:`SafetyBounds` — hardware-side limits (max voltage, max
  current, max routing length) paired with :class:`FormalSafetyGuard`
  (genome-side pre-check).
- :class:`TileAllocation` — which FPGA tile a genome occupies, plus its
  resource fingerprint.
- :class:`TileDeploymentTracker` — live map of tile occupancy; vacates
  on extinction and re-allocates on replication.
- :class:`OrganismEmitter` — serialises a :class:`Genome` to NIR or
  Verilog for hardware push; the inverse of :class:`GenomeSerializer`.
- :class:`GenomeSerializer` — JSON / binary round-trip of a genome.
- :class:`ResourceBudget` — tracks (power_mw, area_um2, latency_ns) for
  the evolving population; prevents hardware budget overruns.

---

## 7. Bridges into other subsystems

- :func:`sc_neurocore.bioware.bioware.mea_fitness_hook` — wraps MEA
  spike recordings into a :class:`ReplicationEngine` metrics callable.
- :meth:`sc_neurocore.arcane_zenith.ArcaneZenithCognitiveCore.step_from_genome`
  — seeds the cognitive core's ``tau_fast`` + ``tau_work`` from a
  :class:`NeuronGene` and steps it with ``topology.connectivity`` as
  drive current.

---

## Reference

- Source: `src/sc_neurocore/evo_substrate/evo_substrate.py`.
- Tests: `tests/test_evo_substrate/test_evo_substrate.py`.
- Demo: `examples/16_evo_substrate_demo.py`.

::: sc_neurocore.evo_substrate.evo_substrate
    options:
      show_root_heading: true
