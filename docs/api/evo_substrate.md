# Evolutionary Substrate

Open-ended evolution of stochastic-computing neural networks: self-replicating
organisms whose **genomes** encode topology, neuron kinetics, and plasticity
parameters, mutate and recombine under safety invariants, speciate by
genomic distance, migrate between islands, and deploy onto FPGA tiles. The
fitness function accepts either a pure-software proxy or the closed-loop
wet-lab MEA hook exposed by
:func:`sc_neurocore.bioware.bioware.mea_fitness_hook`.

```python
from sc_neurocore.evo_substrate.evo_substrate import (
    Genome, NeuronGene, TopologyGene, PlasticityGene,
    MutationEngine, CrossoverEngine, FitnessEvaluator,
    ReplicationEngine, OrganismEmitter, SafetyBounds,
    TileDeploymentTracker, HallOfFame, IslandModel,
    NoveltyArchive, FormalSafetyGuard, ParetoFront,
    CPPNGenome, ComplexityTracker, BloatPenalizer,
    ExtinctionDetector, LineageTracker, AgeRegulator,
    TournamentSelector, EvoStatisticsTracker,
    HWFitnessCollector, CoevolutionArena,
    assign_species, population_diversity, genomic_distance,
    dominates, shared_fitness, compute_bloat, genome_complexity,
    genome_diff,
)
```

---

## 1. Mathematical formalism

### 1.1 Genome as a fixed-length vector

A :class:`Genome` serialises to a vector
$\mathbf{g} \in \mathbb{R}^{19}$ via

$$
\mathbf{g}
  = \bigl[\;\mathbf{t}\;\|\;\mathbf{n}\;\|\;\mathbf{p}\;\bigr],
$$

where $\mathbf{t} \in \mathbb{R}^{5}$ is the
:class:`TopologyGene` block
$(N_{\text{neurons}},\, N_{\text{layers}},\, c,\, r_{\text{rec}},\, L_{\text{bits}})$,
$\mathbf{n} \in \mathbb{R}^{8}$ is the
:class:`NeuronGene` block
$(\tau_{\text{fast}},\, \tau_{\text{work}},\, \tau_{\text{deep}},\,
\theta,\, \gamma,\, \delta_{\text{conf}},\, \kappa,\, w_{\text{inh}})$,
and $\mathbf{p} \in \mathbb{R}^{6}$ is the
:class:`PlasticityGene` block
$(\eta_{\text{STDP}},\, \tau_{+},\, \tau_{-},\, U_{\text{STP}},\,
\eta_{\text{hom}},\, s_{\text{meta}})$.

The :meth:`Genome.compute_id` fingerprint is the first 12 hex digits of
SHA-256 over the raw bytes of $\mathbf{g}$, giving a collision-safe
content-addressable id. Round-trip is exact via
:meth:`Genome.from_vector`.

### 1.2 Point mutation (Gaussian, multiplicative)

For each coordinate $i$, with probability $p_{\text{point}}$ (default
0.2),

$$
g_i \leftarrow g_i + \mathcal{N}(0,\, \sigma_{\text{point}}^{2})
                       \cdot \bigl(|g_i| + \varepsilon\bigr),
$$

where $\sigma_{\text{point}} = 0.05$ and $\varepsilon = 10^{-8}$.
The multiplicative coupling keeps the relative step size constant across
parameters with very different magnitudes (e.g. $\tau_{\text{deep}}=10^{4}$
vs $\gamma=0.2$).

### 1.3 Structural / duplication / swap mutations

- **Structural.** $N_{\text{neurons}} \leftarrow N_{\text{neurons}} +
  \delta$ with $\delta \in \{-2, -1, 1, 2\}$, clamped to
  $[N_{\min},\,N_{\max}] = [4,\,1024]$; connectivity $c$ receives
  a small Gaussian kick and is clamped to $[0.01,\,1]$.
- **Duplication.** Layer count increases by 1 (capped at 10),
  neuron count scaled by $1.5$ (capped at $N_{\max}$). This models
  whole-gene duplication — the dominant driver of complexity growth in
  biological evolution.
- **Swap.** $\tau_{\text{fast}}$ and $\tau_{\text{work}}$ are swapped,
  a simple inversion-like operator that probes time-scale re-assignment
  without changing the vector's L2 norm.

Mutation type is drawn via cumulative-probability selection using the
rates in :class:`MutationConfig` (structural 0.05, duplication 0.01,
swap 0.02, else point).

### 1.4 Uniform crossover

Two parents $\mathbf{a},\mathbf{b} \in \mathbb{R}^{19}$ produce a
child $\mathbf{c}$ by coordinate-wise Bernoulli selection:

$$
c_i = \begin{cases}
a_i & \text{if } u_i < 0.5 \\
b_i & \text{otherwise}
\end{cases},
\quad u_i \sim \mathcal{U}(0,1).
$$

This is the standard Syswerda uniform operator (Syswerda, 1989); gene-block
boundaries (topology | neuron | plasticity) are respected because each
block occupies a contiguous slice of the vector.

### 1.5 Genomic distance (Adam-like normalised L1)

$$
d(\mathbf{a},\mathbf{b})
= \frac{1}{D}
  \sum_{i=1}^{D}
  \frac{|a_i - b_i|}{|a_i| + |b_i| + \varepsilon},
\qquad D = 19.
$$

This normalised metric is scale-invariant, which is crucial because
$\tau_{\text{deep}}$ and $\gamma$ differ by five orders of magnitude.
$d=0$ means clones; $d$ approaches 1 for maximally different genomes.

### 1.6 NEAT-style speciation

:func:`assign_species` partitions the population greedily:

$$
\text{species}(o) =
\begin{cases}
k, & \min_k d\bigl(\mathbf{g}_o,\, \mathbf{g}_{r_k}\bigr) < \theta_{\text{sp}} \\
k_{\text{new}}, & \text{otherwise}
\end{cases}
$$

where $r_k$ is the representative genome of species $k$ and
$\theta_{\text{sp}}$ is the speciation threshold (default 0.3). The
first organism placed in a species becomes its representative — this
matches Stanley & Miikkulainen's NEAT algorithm (Stanley, 2002).

### 1.7 Composite fitness

:meth:`FitnessResult.compute_composite` combines three terms:

$$
F = w_{\text{acc}} \cdot A
   \;+\; w_{\text{en}} \cdot E
   \;+\; w_{\text{lat}} \cdot L,
$$

with default weights $(0.5,\,0.3,\,0.2)$, where $A$ is the metrics-fn
accuracy score and $E$, $L$ are hardware-cost proxies derived from the
topology gene:

$$
E = \max\!\left(0,\; 1 - 0.5 \cdot \tfrac{N_{\text{neurons}}}{1024}
                     - 0.5 \cdot \tfrac{L_{\text{bits}}}{1024}\right),
\qquad
L = \max\!\left(0,\; 1 - \tfrac{N_{\text{layers}}}{10}\right).
$$

### 1.8 Pareto dominance for multi-objective selection

One fitness result dominates another iff it is at least as good on every
objective and strictly better on at least one:

$$
\mathbf{f}_a \succ \mathbf{f}_b
\;\Leftrightarrow\;
\bigl(\forall i:\, f_{a,i} \geq f_{b,i}\bigr)
\wedge
\bigl(\exists j:\, f_{a,j} > f_{b,j}\bigr).
$$

:class:`ParetoFront` maintains the set of non-dominated organisms across
generations, exposed through :func:`dominates`.

### 1.9 Bloat-aware fitness penalty

Complexity is measured by
:func:`genome_complexity`$(g) = 0.7\,\tfrac{N_{\text{neurons}}}{N_{\max}}
+ 0.3\,\tfrac{N_{\text{layers}}}{10}$. :class:`BloatPenalizer`
subtracts a parsimony term from the composite score:

$$
F_{\text{penalised}} = F - \lambda \cdot
  \max\!\left(0,\; \mathrm{complexity}(g) - \mathrm{complexity}_{\text{baseline}}\right),
$$

defaulting to $\lambda = 0.01$.

### 1.10 Fitness sharing (niche preservation)

:func:`shared_fitness` divides an organism's raw fitness by a niche count,
yielding:

$$
F_{\text{shared}}(o_i) =
  \frac{F(o_i)}
       {\sum_j \max\!\bigl(0,\;1 - d(g_i, g_j)/\sigma_{\text{share}}\bigr)}.
$$

This is the Goldberg & Richardson (1987) fitness-sharing operator; it
prevents a single dominant lineage from erasing weaker but diverse niches.

### 1.11 CPPN developmental encoding

Instead of storing weights directly, :class:`CPPNGenome` stores a small
network of CPPN nodes with activations drawn from
$\{\sin,\, \tanh,\, \text{Gaussian},\, \text{sigmoid}\}$. The connection
weight between post-synaptic neuron at coordinate $\mathbf{x}$ and
pre-synaptic neuron at coordinate $\mathbf{y}$ is obtained by a forward
pass $w = \mathrm{CPPN}(\mathbf{x}, \mathbf{y})$. This matches Stanley's
HyperNEAT formulation (Stanley et al., 2009) and exploits spatial
symmetries — a mutation in one CPPN edge reshapes the entire weight
matrix coherently.

---

## 2. Theory (why this particular design)

### 2.1 Genotype–phenotype map is lossy but hardware-closed

The 19-D genome does **not** encode specific weights — those are
deterministic from ``weight_seed`` — nor specific spike trains. Instead,
the genome encodes *control points* of the phenotype (time constants,
connection probability, plasticity rates) that stay inside the envelope
the hardware FPGA tile can realise. This is deliberate: the evolutionary
search operates in a space where **every point is constructible** on the
target substrate, so crossing a fitness gradient cannot yield an organism
that fails to instantiate.

### 2.2 Why 19 coordinates, not 100s

Open-ended evolution typically gets more powerful with higher-dimensional
genomes, but each added dimension multiplies the search volume. SC-NeuroCore
fixes a small, physically motivated 19-D genome and lets
:class:`CPPNGenome` provide the escape hatch for high-dimensional weight
searches when needed. This follows the same principle as PicBreeder
(Secretan, 2011) — small active genome, large effective phenotype via
developmental indirection.

### 2.3 Formal safety as a hard filter

Every proposed genome passes through
:class:`FormalSafetyGuard` *before* it enters the population. The guard
checks three invariants:

1. **Time-constant positivity.**
   $\tau_{\text{fast}},\tau_{\text{work}},\tau_{\text{deep}} > 0$ and
   $\tau_{\text{fast}} < \tau_{\text{work}} < \tau_{\text{deep}}$.
2. **Connectivity bounds.** $c \in [0.01,\,1]$ and
   $N_{\text{neurons}} \in [4,\,1024]$.
3. **Lyapunov-bounded plasticity.**
   $\eta_{\text{STDP}} \cdot \max(\tau_{+}, \tau_{-}) < C_{\text{lyap}}$
   where $C_{\text{lyap}}$ is a pre-computed bound that keeps the STDP
   update map contractive under worst-case rate inputs.

Invariant 3 is checked by
:meth:`FormalSafetyGuard.check` rather than proved on-the-fly; see
`docs/api/formal.md` §6 for the matching Lean 4 theorem (axiomatised,
with a Mathlib proof roadmap).

### 2.4 Extinction as a diversity reset

Real evolution periodically resets via mass-extinction events (Raup, 1991).
:class:`ExtinctionDetector` mirrors this: when the best fitness has not
improved for `stagnation_gens=10` generations, a fraction
(`kill_fraction=0.9`) of the population is culled and reseeded from the
:class:`HallOfFame`. This is not just ergodicity theatre — it breaks local
maxima that incremental mutation cannot escape.

### 2.5 Island model with periodic migration

:class:`IslandModel` runs N independent sub-populations with migration
every $M$ generations. Each island has its own RNG seed and mutation
pressure; diverse islands explore different basins. Migration copies the
top-$k$ organisms between adjacent islands, propagating discoveries
without erasing sub-population identity. This is the textbook Whitley
distributed GA (Whitley, 1999) adapted to hardware-aware selection.

---

## 3. Position in the pipeline

```
+------------------+    +-------------------+    +------------------+
|  ArcaneZenith    |    |    evo_substrate  |    |     bioware      |
|  cognitive core  |<---|  (this module)    |--->|  MEA closed-loop |
+------------------+    +-------------------+    +------------------+
                            ^      |     ^
                            |      |     |
                      seeds |      |     | deploys
                            |      v     |
                      +----------+ +------------+
                      | hdl_gen  | | FPGA tile  |
                      | verilog  | | allocation |
                      +----------+ +------------+
```

- **Upstream inputs.** `ArcaneZenith.step_from_genome` seeds its
  time-constants from :class:`NeuronGene`; `sc_scope` and `sc_doctor`
  (see `debug.md`) observe phenotype behaviour.
- **Outputs.** :class:`OrganismEmitter` converts winning genomes to
  NIR graph or Verilog for `hdl_gen/verilog_generator.py`; resource
  budget checks go through :class:`SafetyBounds` and
  :class:`TileDeploymentTracker`.
- **Closed loop.** Fitness metrics come from either a software proxy
  or :func:`bioware.mea_fitness_hook`; the loop does not leave the
  substrate.

---

## 4. Features

- Content-addressable genomes (SHA-256 ids, 12 hex chars).
- 5 mutation operators (point, structural, duplication, swap, identity).
- Uniform crossover with gene-block alignment.
- Normalised L1 genomic distance (scale-invariant).
- NEAT-style greedy speciation, fitness sharing, novelty archive.
- Tournament + elitist + age-regulated selection
  (:class:`TournamentSelector` + :class:`AgeRegulator`).
- Industrial-mode :class:`ReplicationEngine` with 9 co-operating guards.
- Multi-objective Pareto front (:class:`ParetoFront`,
  :func:`dominates`).
- Bloat penalty, complexity tracking, extinction detector, hall of fame.
- CPPN developmental encoding (:class:`CPPNGenome`) for high-dim weight
  searches.
- Island model (:class:`IslandModel`) with migration.
- Hardware-side gating (:class:`SafetyBounds`, :class:`ResourceBudget`,
  :class:`TileDeploymentTracker`).
- NIR + Verilog emission (:class:`OrganismEmitter`).
- Co-evolution arena (:class:`CoevolutionArena`) for predator/prey or
  critic/actor dynamics.
- Full lineage graph (:class:`LineageTracker`) — every child records
  its parent and mutation type, giving a reconstructable phylogeny.

---

## 5. Usage — end-to-end generation

```python
from sc_neurocore.evo_substrate.evo_substrate import (
    Genome, ReplicationEngine, MutationEngine, MutationConfig,
)

def metrics_fn(genome):
    # Plug your closed-loop MEA hook, or a software proxy:
    return {"accuracy": 0.5 + 0.01 * genome.topology.num_neurons / 32}

cfg = MutationConfig(
    point_rate=0.2,
    point_sigma=0.05,
    structural_rate=0.05,
    duplication_rate=0.01,
    swap_rate=0.02,
)
engine = ReplicationEngine(
    mutation_engine=MutationEngine(cfg, rng_seed=7),
    max_population=32,
    elitism=1,
    industrial_mode=True,
)

for i in range(16):
    g = Genome()
    g.compute_id()
    engine.seed(g)

engine.evaluate_all(metrics_fn)

for gen in range(20):
    stats = engine.evolve_generation(metrics_fn)
    print(
        f"gen {stats['generation']:>3}  "
        f"pop={stats['population_size']:>2}  "
        f"best={stats['best_fitness']:.3f}  "
        f"diversity={stats['diversity']:.3f}"
    )
```

Sample output from a real run (`industrial_mode=True`, 16-organism
seed, 20 generations, `rng_seed=7`):

```
gen   1  pop=16  best=0.042  diversity=0.006
gen   5  pop=16  best=0.044  diversity=0.008
gen  10  pop=16  best=0.044  diversity=0.008
gen  20  pop=16  best=0.044  diversity=0.007
```

Low `best_fitness` values reflect the toy `metrics_fn` above (returns
`0.5 + 0.01 · num_neurons/32`, penalised by hardware-cost and
bloat terms); replace with a real evaluator to see non-trivial
selection pressure. Population stays at 16 in this run because
tournament selection + safety-guard rejection keep new organisms
below the `max_population=32` cap when the selected parents are
similar.

---

## 6. API reference

### 6.1 Gene blocks

| Class               | Fields (with defaults)                                                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| :class:`TopologyGene`   | `num_neurons=16`, `num_layers=2`, `connectivity=0.3`, `recurrent_fraction=0.1`, `bitstream_length=256`                                             |
| :class:`NeuronGene`     | `tau_fast=5`, `tau_work=200`, `tau_deep=10000`, `theta=1`, `gamma=0.2`, `delta_conf=0.3`, `kappa=5`, `w_inh=0.3`                                   |
| :class:`PlasticityGene` | `stdp_lr=0.01`, `stdp_tau_plus=20`, `stdp_tau_minus=20`, `stp_u_base=0.5`, `homeostatic_rate=0.001`, `meta_sensitivity=1`                          |

### 6.2 Mutation + crossover

| Symbol                      | Purpose                                                            |
| --------------------------- | ------------------------------------------------------------------ |
| :class:`MutationType`       | enum: `POINT`, `STRUCTURAL`, `DUPLICATION`, `SWAP`, `IDENTITY`     |
| :class:`MutationConfig`     | per-type rates, Gaussian σ, structural intensity bounds            |
| :class:`MutationEngine`     | deterministic under `rng_seed`; `mutate(g)` returns `(child, op)`  |
| :class:`CrossoverEngine`    | uniform crossover, gene-block-aligned                              |
| :func:`genomic_distance`    | scale-invariant L1                                                 |
| :func:`assign_species`      | NEAT-style speciation                                              |
| :func:`population_diversity`| mean pairwise distance                                             |

### 6.3 Fitness + selection

| Symbol                         | Purpose                                                              |
| ------------------------------ | -------------------------------------------------------------------- |
| :class:`FitnessType`           | `ACCURACY`, `ENERGY`, `LATENCY`, `COMPOSITE`                         |
| :class:`FitnessResult`         | `(accuracy, energy_score, latency_score, composite)`                 |
| :class:`FitnessEvaluator`      | scorer over population; accepts `metrics_fn`                         |
| :class:`TournamentSelector`    | $k$-way tournament with optional elitism                             |
| :class:`AgeRegulator`          | ages out organisms past `max_age`                                    |
| :class:`ParetoFront`           | non-dominated front                                                  |
| :func:`dominates`              | Pareto relation $\succ$                                              |
| :func:`shared_fitness`         | Goldberg–Richardson niching                                          |

### 6.4 Population control

| Symbol                          | Purpose                                                             |
| ------------------------------- | ------------------------------------------------------------------- |
| :class:`IslandModel`            | N sub-populations + periodic migration                              |
| :class:`NoveltyArchive`         | sparse archive of behaviourally distinct genomes                    |
| :class:`HallOfFame`             | top-K elites across generations                                     |
| :class:`BloatPenalizer`         | parsimony penalty on composite fitness                              |
| :class:`ComplexityTracker`      | structural complexity over time                                     |
| :class:`ExtinctionDetector`     | mass-extinction trigger on stagnation                               |
| :class:`CoevolutionArena`       | predator/prey or critic/actor co-evolution                          |
| :class:`EvoStatisticsTracker`   | per-generation :class:`GenerationStats` log                         |

### 6.5 Safety + hardware

| Symbol                          | Purpose                                                              |
| ------------------------------- | -------------------------------------------------------------------- |
| :class:`FormalSafetyGuard`      | genome-side invariants (tau positivity, c bounds, Lyapunov plast.)   |
| :class:`SafetyBounds`           | hardware-side limits (V, I, routing length)                          |
| :class:`ResourceBudget`         | tracks `(power_mw, area_um2, latency_ns)`                            |
| :class:`TileAllocation`         | which FPGA tile a genome occupies                                    |
| :class:`TileDeploymentTracker`  | live map of tile occupancy; handles replication + extinction         |
| :class:`HWFitnessReport`        | post-silicon metrics feedback                                        |
| :class:`HWFitnessCollector`     | aggregates :class:`HWFitnessReport` into a fitness proxy             |

### 6.6 Indirect encoding (CPPN)

| Symbol                  | Purpose                                                               |
| ----------------------- | --------------------------------------------------------------------- |
| :class:`ActivationFunc` | `SINE`, `TANH`, `GAUSSIAN`, `SIGMOID`                                 |
| :class:`CPPNNode`       | one activation node                                                   |
| :class:`CPPNEdge`       | one weighted edge                                                     |
| :class:`CPPNGenome`     | NEAT-like CPPN; expands to weight matrix via forward pass             |

### 6.7 Lineage + diff

| Symbol                  | Purpose                                                               |
| ----------------------- | --------------------------------------------------------------------- |
| :class:`LineageRecord`  | `(genome_id, parent_id, generation, mutation_type, fitness)`          |
| :class:`LineageTracker` | records all records; walk ancestry via `get_ancestors(genome_id)`     |
| :class:`GenomeDiff`     | `(topology_delta, neuron_delta, plasticity_delta)`                    |
| :func:`genome_diff`     | per-block L2 delta between two genomes                                |
| :func:`genome_complexity` | scalar $0.7 N/N_{\max} + 0.3 L/10$                                  |

### 6.8 Emission

| Symbol                   | Purpose                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| :class:`OrganismEmitter` | genome → NIR graph or Verilog                                        |
| :class:`GenomeSerializer`| JSON / binary round-trip                                             |

---

## 7. Verified benchmarks

Measured on Ubuntu 24.04 / CPython 3.12.3 / Intel i5-11600K @ 3.90 GHz,
single-thread. All figures produced by
`benchmarks/bench_evo_substrate.py` (committed) and reproducible with
`python benchmarks/bench_evo_substrate.py`.

| Operation                                    | Throughput            | Latency   |
| -------------------------------------------- | --------------------- | --------- |
| `MutationEngine.mutate`                      | 15 223 ops/s          | 65.69 µs  |
| `CrossoverEngine.crossover`                  | 29 631 ops/s          | 33.75 µs  |
| `genomic_distance` (19-D)                    | 92 891 ops/s          | 10.77 µs  |
| `FormalSafetyGuard.check`                    | 1 376 152 ops/s       |  0.73 µs  |
| `assign_species` (n=64, θ=0.3)               |  1 430 ops/s          |  0.70 ms  |
| `ReplicationEngine.evolve_generation` (pop=32, industrial_mode=True) | 161 gen/s | 6.20 ms |

Raw JSON at `benchmarks/results/bench_evo_substrate.json` is written by
the same script every run, so any doc regression (drift, rename, hidden
simplification) can be caught by diffing the JSON rather than re-reading
the markdown.

### 7.1 Determinism + reproducibility

All RNGs in the module are ``numpy.random.default_rng`` seeded through
explicit constructor arguments (``MutationEngine(rng_seed=…)``,
``CrossoverEngine(rng_seed=…)``, :class:`ReplicationEngine`'s internal
``self.mutator.rng``). Two consequences:

1. A given ``(config, seeds, metrics_fn)`` triple is bit-reproducible:
   re-running the 20-generation demo above yields the same lineage tree,
   same :class:`HallOfFame` entries, and the same :class:`ParetoFront`.
2. Islands in :class:`IslandModel` take independent seeds derived from a
   master seed, so experiments can be re-run with different master seeds
   to bound Monte-Carlo noise on any reported figure.

The lineage tracker (:class:`LineageTracker`) also lets you replay any
subtree: given a surviving ``genome_id``, :meth:`get_ancestors` returns
the exact mutation chain from seed to present, which is what
:class:`OrganismEmitter` serialises alongside the Verilog blob for
audit trails on the FPGA tile side.

### 7.2 Rust coverage gap (honest accounting)

`evo_substrate.py` has **no Rust counterpart** today. The hot paths
(mutation 64 µs, crossover 33 µs, genomic distance 10 µs,
evolve_generation 6 ms for pop=32) are dominated by NumPy array
allocation, not pure arithmetic, so a naive port to Rust would yield
~2–3× at best — not the 5–10× the multi-language rule usually buys
on compute-heavy inner loops. Bigger gains come from amortising the
per-call allocation (reusing buffers across generations) on the
Python side. A Rust path into `crates/evo_substrate_core/` is on the
roadmap but is currently below the bar that justifies the FFI
surface; the decision will be revisited when population sizes exceed
~1 000 organisms or generation counts exceed ~10 k.

**Interpretation.** Safety checks and distance computations are cheap
enough (<1 µs and ~10 µs respectively) that they do not dominate the
generation cost. Mutation and crossover are the slower inner ops
(NumPy array creation per call); bulk speedup when moving to a Rust
inner loop is achievable but not necessary for current population
sizes — at 32 organisms × 20 generations, a full run completes in
$\approx 0.12$ s.

Figures above are `time.perf_counter` deltas from
`benchmarks/bench_evo_substrate.py`.

---

## 8. Citations

1. Stanley K.O., Miikkulainen R. (2002). *Evolving Neural Networks
   through Augmenting Topologies*. Evolutionary Computation 10(2):99–127.
2. Stanley K.O., D'Ambrosio D.B., Gauci J. (2009). *A Hypercube-Based
   Encoding for Evolving Large-Scale Neural Networks*. Artif. Life
   15(2):185–212. (HyperNEAT / CPPN.)
3. Syswerda G. (1989). *Uniform Crossover in Genetic Algorithms*.
   Proc. 3rd Int. Conf. on Genetic Algorithms, 2–9.
4. Goldberg D.E., Richardson J. (1987). *Genetic Algorithms with Sharing
   for Multimodal Function Optimization*. ICGA-87, 41–49. (Fitness
   sharing.)
5. Deb K., Pratap A., Agarwal S., Meyarivan T. (2002). *A Fast and
   Elitist Multiobjective Genetic Algorithm: NSGA-II*. IEEE TEC
   6(2):182–197. (Pareto dominance.)
6. Whitley D. (1999). *An overview of evolutionary algorithms:
   practical issues and common pitfalls*. Information and Software
   Technology 43(14):817–831. (Island model.)
7. Secretan J. et al. (2011). *Picbreeder: A Case Study in Collaborative
   Evolutionary Exploration of Design Space*. Evolutionary Computation
   19(3):373–403.
8. Raup D.M. (1991). *Extinction: Bad Genes or Bad Luck?* W. W. Norton.
   (Mass-extinction dynamics.)
9. Lehman J., Stanley K.O. (2011). *Abandoning Objectives: Evolution
   Through the Search for Novelty Alone*. Evolutionary Computation
   19(2):189–223. (Novelty archive.)
10. Šotek M. (2026). *SC-NeuroCore: Self-replicating neuromorphic
    substrate*. Internal report, ANULUM.

---

## Reference

- Source: `src/sc_neurocore/evo_substrate/evo_substrate.py` (1594 LOC).
- Tests: `tests/test_evo_substrate/test_evo_substrate.py` (908 LOC).
- Demo: `examples/16_evo_substrate_demo.py`.
- Benchmark: `benchmarks/bench_evo_substrate.py`.

::: sc_neurocore.evo_substrate.evo_substrate
    options:
      show_root_heading: true
