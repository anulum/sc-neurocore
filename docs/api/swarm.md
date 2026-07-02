# Neuromorphic Swarm Control

**Module:** `sc_neurocore.swarm`
**Source:** `src/sc_neurocore/swarm/` — 5 files, 1098 LOC
**Status (v3.14.0):** all 9 public symbols wired; 87 tests pass across 4
focused swarm files; pure-Python (no Rust path) — performance falls steeply
beyond ~100 agents (§9). The agent's "soft-LIF" naming overstates the
mechanism; see §3.2.

This page covers the entire swarm subpackage:
`SwarmAgent` (sigmoid-pseudo-spike SNN per agent), `SwarmEnvironment`
(2-D arena with obstacles + targets), `CollectiveFields` (chemical /
emotional / symbolic field layers), `SwarmFitness` (5 objectives plus a
weighted composite), and `SwarmEvolver` (truncation-elite GA over flat
weight vectors).

---

## 1. Public surface

`sc_neurocore.swarm.__init__` re-exports 9 symbols from the 5 modules:

| Symbol | Source file | Role |
|--------|-------------|------|
| `AgentConfig` | `agent.py` | Hyper-parameters for one agent |
| `SwarmAgent` | `agent.py` | Per-agent SNN + kinematic state |
| `EnvConfig` | `swarm_env.py` | Arena hyper-parameters |
| `SwarmEnvironment` | `swarm_env.py` | 2-D continuous arena + step loop |
| `FieldConfig` | `collective_fields.py` | Field-layer hyper-parameters |
| `CollectiveFields` | `collective_fields.py` | Chemical / emotional / symbolic fields |
| `SwarmFitness` | `fitness.py` | Static objective + composite scoring |
| `EvolverConfig` | `neuroevolution_swarm.py` | GA hyper-parameters |
| `SwarmEvolver` | `neuroevolution_swarm.py` | GA over flat weight vectors |

All 9 are dataclass-style or stateful classes; none are utility
functions. The package has no module-level globals.

---

## 2. Architecture

```
                ┌─────────────────────────────┐
                │ SwarmEvolver (GA, weights)  │
                │   pop_size × n_eval_steps   │
                └────────────┬────────────────┘
                             │ inject weights into every agent
                             ▼
   ┌────────────────────────────────────────────────────────┐
   │ SwarmEnvironment.step(dt, fields=None) — per timestep  │
   │                                                         │
   │   for each agent:                                       │
   │     build 20-channel sensory vector                     │
   │       (8 nbrs · 3 obs · 2 tgt · 2 chem · 2 sym · 2 emo) │
   │     speed, turn = agent.think(sensory)                  │
   │     agent.act(speed * dt, turn * dt)                    │
   │     boundary wrap / clamp                               │
   │     fields.deposit_chemical(agent.position, ...)        │
   │                                                         │
   │   target capture (Euclidean within capture_radius)      │
   │   fields.update(...) → diffuse, sync emotions, decay sym│
   └────────────────────────────────────────────────────────┘
                             │
                             ▼
                ┌─────────────────────────────┐
                │ SwarmFitness.composite(env) │
                │   0.30 cov + 0.20 coh +     │
                │   0.10 aln + 0.30 tgt -     │
                │   0.10 obstacle_penalty     │
                └─────────────────────────────┘
```

Agent has no awareness of fields directly; the environment passes
chemical gradients, symbolic field reads, and the agent's own emotion
vector into the sensory channel slice on each step. Weight vectors
flow from `SwarmEvolver` into `SwarmAgent.weights` (setter). Fitness
flows back from `SwarmFitness.composite(env)` into `SwarmEvolver`.

---

## 3. `SwarmAgent` and `AgentConfig`

```python
@dataclass
class AgentConfig:
    n_sensory: int = 20
    n_hidden: int = 16
    n_motor: int = 2
    membrane_decay: float = 0.9
    threshold: float = 1.0
    max_speed: float = 2.0
    seed: Optional[int] = None
```

Per-agent SNN dimensions (defaults): 20-channel sensory in, 16 hidden
units, 2 motor channels (speed, turn). Total trainable weights per
agent:

```
n_weights = n_hidden * n_sensory     # W_in   = 16 × 20 = 320
          + n_hidden * n_hidden      # W_rec  = 16 × 16 = 256
          + n_motor  * n_hidden      # W_out  = 2  × 16 =  32
                                     # total = 608
```

Weights are initialised with Xavier-style scaling
`σ = sqrt(2 / (n_in + n_out))` per matrix.

The `weights` property exposes a flat 1-D vector of length `n_weights`
for the GA; the setter validates that the candidate is one-dimensional,
exact-size, numeric, and finite before it splits the vector into
`W_in`, `W_rec`, `W_out`. Invalid candidates leave the previous weights
unchanged.

### 3.1 Sensory layout

The 20-channel sensory vector is filled by `SwarmEnvironment.step`:

| Channels | Source | Range |
|----------|--------|-------|
| 0..7 | nearest-neighbour distances (8 nearest) | normalised by `max(width, height)` |
| 8..10 | nearest-obstacle surface distances (3 nearest) | clipped to `[-1, 1]` after `/ 50` |
| 11..12 | nearest-target distances (2 nearest) | normalised by `max(width, height)` |
| 13..14 | chemical gradient `(dx, dy)` | unit vector if fields present, zero otherwise |
| 15..16 | symbolic field at agent position | 2-channel raw field value |
| 17..18 | agent's own emotion `(valence, arousal)` | first 2 of 8-D emotion vector |
| 19 | the agent's own previous chemical output | `[0, 1]` |

If `fields` is `None` at step time, channels 13..19 are zero.

### 3.2 The "soft-LIF" naming overstates the mechanism

`SwarmAgent` is documented as a "Spiking-neural-network agent with
soft-LIF dynamics". The actual `think()` body does **not** implement
an LIF cell. There is no
hard threshold, no spike emission as a binary event, and no membrane
reset. Instead:

```python
# Membrane integration (LIF-shaped, fine so far)
self.membrane = (
    c.membrane_decay * self.membrane
    + self.W_in @ inp
    + self.W_rec @ self.firing_rate
)

# This is NOT a spike — it is a sigmoid pseudo-rate
spike_prob = 1.0 / (1.0 + np.exp(-(self.membrane - c.threshold)))

# EMA over rate, not over spikes
self.firing_rate = 0.8 * self.firing_rate + 0.2 * spike_prob

# Soft "reset" — proportional, not the hard reset of LIF
self.membrane *= 1.0 - spike_prob
```

This is closer to a **mean-field rate model** with a sigmoid
non-linearity than to LIF. There is no spike vector at any point;
`firing_rate` is a smoothed sigmoid output. The motor readout
`W_out @ firing_rate` then runs through `tanh` and a linear scale.

**Why it matters:** anyone expecting LIF dynamics — sparse binary
spikes, hard threshold, integer spike counts — will not find them
here. The fitness landscape under this surrogate may differ
qualitatively from a true SNN. For a true LIF surrogate gradient,
see `sc_neurocore.neurons.stochastic_lif` or the model registry.

A more honest name would be `SoftSigmoidRecurrentNet` or
`ReLU-ish-sigmoid SNN surrogate`. Tracked as task #25 (rename or
implement).

### 3.3 `think` and `act`

`think(sensory) -> (speed, turn)` first validates that `sensory` is a
finite one-dimensional vector with exactly `n_sensory` entries. It then
runs one tick of the recurrent sigmoid network and returns motor
commands:
- `speed = (tanh(W_out[0] @ firing_rate) + 1) * 0.5 * max_speed`
  — clipped to `[0, max_speed]`
- `turn = tanh(W_out[1] @ firing_rate) * π` — clipped to `[-π, π]`

The update is candidate-first: malformed sensory vectors, overflowed
membrane states, non-finite firing rates, or non-finite motor commands
raise `ValueError` without mutating membrane, firing-rate, or chemical
state. The chemical-output side channel is read from sensory channel 19
and clamped to `[0, 1]`.

`act(speed, turn)` advances the agent's `position` and `heading`:
```python
heading = (heading + turn) % (2π)
position += speed * (cos heading, sin heading)
```
Both motor commands must be finite.

`reset(rng=None, width, height)` zeroes the membrane / firing rate
and re-randomises position and heading. Weights are not touched —
useful for episodic resets within a training run. The arena width and
height must be positive finite values, and a caller-supplied `rng` must
be a `numpy.random.Generator`; invalid reset inputs leave the existing
kinematic state intact.

---

## 4. `SwarmEnvironment` and `EnvConfig`

```python
@dataclass
class EnvConfig:
    width: float = 100.0
    height: float = 100.0
    n_agents: int = 20
    n_obstacles: int = 5
    n_targets: int = 3
    boundary_mode: str = "wrap"  # "wrap" or "clamp"
    capture_radius: float = 3.0
    respawn_targets: bool = True
    agent_config: Optional[AgentConfig] = None
    seed: Optional[int] = None
```

A 2-D continuous arena. Obstacles are circles `(x, y, radius)`,
targets are points `(x, y)`. Boundary mode decides what happens at
the edges (wrap-around toroidal vs hard clamp).

### 4.1 Neighbour / obstacle / target queries

| Method | Returns | Cost |
|--------|---------|------|
| `get_positions()` | `(n_agents, 2)` ndarray | O(n) |
| `get_headings()` | `(n_agents,)` ndarray | O(n) |
| `get_pairwise_distances()` | `(n_agents, n_agents)` Euclidean matrix | O(n²) |
| `get_neighbor_distances(i, k=8)` | sorted distances to k nearest others | O(n log n) per agent |
| `get_obstacle_distances(i, k=3)` | sorted surface distances to k nearest obstacles | O(n_obs log n_obs) |
| `get_target_distances(i, k=2)` | sorted distances to k nearest targets | O(n_tgt log n_tgt) |

The per-step cost is **dominated by the per-agent neighbour query
called inside the step loop**: `n × O(n log n)` = `O(n² log n)`. No
KD-tree, no spatial hashing — see §9 for the resulting performance
falloff.

### 4.2 `step(dt, fields=None)`

For each agent:
1. Build the 20-channel sensory vector (§3.1).
2. Call `agent.think(sensory)` → `(speed, turn)`.
3. Call `agent.act(speed * dt, turn * dt)`.
4. Apply boundary mode (wrap or clamp).
5. If `fields` is not None, deposit `agent.chemical_output * dt` at
   the agent's position.

Then once per step:
6. Target capture — for each target, find the nearest agent
   distance; if `< capture_radius`, increment `targets_captured` and
   (optionally) respawn the target at a fresh random position.
7. `fields.update(agents, env, dt)` if fields are active.

`step_count` increments at the end of each call.

### 4.3 `get_state()`

Returns a JSON-serialisable snapshot:
```python
{"step": step_count,
 "positions": [[x, y], ...],
 "headings": [θ, ...],
 "obstacles": [[x, y, r], ...],
 "targets": [[x, y], ...],
 "targets_captured": int}
```

There is no inverse `set_state` — environments are reconstructed by
re-seeding the same `EnvConfig`.

---

## 5. `CollectiveFields` and `FieldConfig`

```python
@dataclass
class FieldConfig:
    grid_size: int = 50
    diffusion_rate: float = 0.1
    decay_rate: float = 0.05
    emotional_coupling: float = 0.1
    symbolic_decay: float = 0.02
    seed: int | None = None
```

Three coupled field layers on a `grid_size × grid_size` lattice:

| Field | Storage | Update |
|-------|---------|--------|
| `chemical_field` | `(grid_size, grid_size)` float | Laplacian diffusion + exponential decay |
| `emotional_field` | `(n_agents, 8)` float | mean-field coupling toward swarm mean |
| `symbolic_field` | `(grid_size, grid_size, 2)` float | exponential decay, no diffusion |

### 5.1 Chemical Laplacian diffusion

```
∂C/∂t = D · ∇²C - λ · C
```

Implemented as `_apply_laplacian` (`collective_fields.py:54`) — a
manual 3×3 stencil convolution with zero-padded edges, written as a
double Python loop over `(di, dj) ∈ {-1, 0, 1}²` (Rustification
candidate, see §10). After the Laplacian, the field is multiplied by
`(1 - decay_rate * dt)` and clipped at 0 (no negative concentrations).

`get_chemical_gradient(x, y)` returns a unit-vector `(dx, dy)` of
central-difference partials, mapped from grid to world coordinates.
The norm includes a `+1e-12` guard against zero-gradient division.

### 5.2 Emotional mean-field coupling

```python
mean = emotional_field.mean(axis=0)              # (8,)
emotional_field += coupling * (mean - emotional_field)
```

Each agent's 8-D emotion vector is pulled toward the swarm mean by a
fraction `coupling` per call. Default `coupling = 0.1`, so the
half-life of any deviation from the mean is `log(2) / log(1 / 0.9)
≈ 6.6` steps.

### 5.3 Symbolic field

A 2-channel raster with no diffusion — only exponential decay
(`* (1 - symbolic_decay * dt)`). Agents read with
`get_symbolic_at(x, y)`, write with `deposit_symbolic(x, y, channel,
amount)`. There is no spatial coupling beyond grid-cell granularity.

---

## 6. `SwarmFitness`

Static class with five per-objective scorers and one weighted
composite. All inputs are NumPy arrays from `SwarmEnvironment`.

| Method | What it rewards | Range |
|--------|-----------------|-------|
| `coverage_score(positions, area)` | fraction of a 10×10 bin grid that contains ≥1 agent | `[0, 1]` |
| `cohesion_score(positions)` | mean pairwise distance close to bbox-diagonal × 0.25 | `[0, 1]` (Gaussian) |
| `alignment_score(headings)` | Rayleigh statistic — mean resultant length of heading angles | `[0, 1]` |
| `target_score(positions, targets)` | inverse mean distance from each agent to its nearest target | `[0, 1]` (`1/(1+d/10)`) |
| `obstacle_penalty(positions, obstacles)` | fraction of agents inside any obstacle | `[0, 1]` |

The composite at `fitness.py:114-137`:

```python
0.30 * coverage
+ 0.20 * cohesion
+ 0.10 * alignment
+ 0.30 * target_proximity
- 0.10 * obstacle_penalty
```

Weights are hard-coded; there is no `composite(env, weights=...)`
overload. The score is unbounded below (because the obstacle penalty
is subtracted) but bounded at `+1.0` above when all positive
objectives saturate and no agent is inside an obstacle.

---

## 7. `SwarmEvolver` and `EvolverConfig`

```python
@dataclass
class EvolverConfig:
    pop_size: int = 20
    n_elite: int = 4
    mutation_rate: float = 0.1
    mutation_std: float = 0.3
    n_eval_steps: int = 200
    use_fields: bool = False
    env_config: Optional[EnvConfig] = None
    agent_config: Optional[AgentConfig] = None
    seed: Optional[int] = None
```

A textbook genetic algorithm over flat weight vectors:

1. **Population** — `pop_size` flat vectors, each of length
   `n_weights = template_agent.n_weights` (608 for the default
   AgentConfig). Initialised from `N(0, 0.5)`.
2. **Evaluation** — for each individual, build a fresh
   `SwarmEnvironment`, inject the same weights into all agents
   (homogeneous swarm), run `n_eval_steps`, score with
   `SwarmFitness.composite(env)`.
3. **Selection** — truncation: take the top `n_elite` individuals by
   fitness.
4. **Reproduction** — elite survive unchanged; remainder produced
   by uniform crossover of two random elite parents plus Gaussian
   mutation (`mutation_rate` of genes perturbed by `N(0, mutation_std)`).
5. **Iterate** for `n_generations`.

`run(n_generations) -> list[best_fitness_per_generation]`.
`get_best_weights()` returns the highest-fitness vector after the last
evaluation.

`EvolverConfig` rejects invalid population, selection, mutation,
evaluation-step, and seed domains at construction. The default
`n_elite=4` is capped to `pop_size` for small populations so the
default configuration remains usable when `pop_size < 4`. Public
weight ingress (`evaluate_individual`, crossover, mutation) reuses the
same exact-size finite-vector contract as `SwarmAgent.weights`; a
non-finite mutation result fails closed before replacing the population.

The evolver re-seeds the environment per individual via
`int(self.rng.integers(0, 2**31))`, so each evaluation sees a
different obstacle/target layout — fitness becomes an average over
seeds within a generation. There is no fixed test-set evaluation.

---

## 8. Performance — measured (this workstation)

Hardware: Intel i5-11600K, 32 GB DDR4, Python 3.12.3.

### 8.1 `SwarmEnvironment.step` (100 timesteps, dt=1.0)

| n_agents | n_obstacles | env-only wall | env+fields wall | env steps/s | env+fields steps/s |
|---------:|------------:|--------------:|----------------:|------------:|-------------------:|
| 20 | 5 | 380.2 ms | 476.6 ms | 263.0 | 209.8 |
| 100 | 5 | 1 815.8 ms | 2 459.8 ms | 55.1 | 40.7 |
| 500 | 5 | 17 347.7 ms | 21 005.2 ms | 5.8 | 4.8 |

Step rate falls super-linearly with `n_agents` (~5× drop per 5×
population, suggesting the per-agent neighbour query
dominates — `n × O(n log n)` ≈ `O(n² log n)`). The chemical-field
Laplacian adds ~20 % overhead at small populations and shrinks in
relative cost as the agent loop grows.

### 8.2 `SwarmEvolver.evolve_generation` (one generation, end-to-end)

| pop_size | n_eval_steps | best_fitness | wall |
|---------:|-------------:|-------------:|-----:|
| 20 | 50 | 0.3300 | 3 022.8 ms |
| 20 | 200 | 0.3063 | 11 845.1 ms |
| 50 | 100 | 0.3524 | 18 783.4 ms |

Per-generation cost is `pop_size × n_eval_steps × per-agent-step`,
plus a fresh-environment construction overhead per individual. The
"agent-step throughput" is consistent at ~270–340 agent-steps/s,
matching §8.1.

### 8.3 No Rust path

`agent.py`, `swarm_env.py`, `collective_fields.py`, `fitness.py`, and
`neuroevolution_swarm.py` are all pure Python / NumPy. The two
hottest kernels — `_apply_laplacian` (Python double loop) and
`get_pairwise_distances` (NumPy O(N²)) — are Rustification
candidates. Tracked under task #13 (network/topology Rustification);
the same effort would extend naturally to swarm.

---

## 9. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.swarm import SwarmAgent, ...` | `swarm/__init__.py:23-27` | `tests/test_swarm.py` |
| `SwarmEvolver` injects weights into every agent | `evaluate_individual` (`neuroevolution_swarm.py`) | `tests/test_swarm_control.py::test_evolver_*` |
| `SwarmEnvironment.step` builds sensory vector + calls agent.think + boundary | `swarm_env.py` | `tests/test_swarm.py` |
| `CollectiveFields.update` runs diffusion + emotion sync + symbolic decay | `collective_fields.py` | `tests/test_swarm_fitness_contracts.py` |
| `SwarmFitness.composite` reads positions / headings / targets / obstacles | `fitness.py` | `tests/test_swarm_fitness_contracts.py` |

Every public symbol terminates in a tested call site; no orphan
helpers.

---

## 10. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 9 symbols re-exported and used by the tests |
| 2 | Multi-angle tests | ✅ PASS | 87 tests across 4 focused files, including strict public contract tests for agent/evolver validation |
| 3 | Rust path | ❌ FAIL | Pure Python; `_apply_laplacian` and `get_pairwise_distances` dominate at n ≥ 100 (§8) |
| 4 | Benchmarks | ✅ PASS | §8.1 + §8.2 measured this session |
| 5 | Performance docs | ✅ PASS | §8 |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ⚠️ WARN | SPDX header on every file ✅. **"Soft-LIF" naming is misleading** (§3.2) — the implementation is a sigmoid pseudo-rate model, not LIF. The previous `agent.py` `type: ignore` suppressions are removed; one scoped `swarm_env.py` suppression remains (§11.3). British English mostly clean (`vectorise`/`synchronise` consistent in docstrings). |

Net: **1 WARN, 1 FAIL.** The FAIL is performance-driven, not
correctness-driven; the WARN is honesty-driven (the naming claim
overstates what the code does).

---

## 11. Known issues

### 11.1 "Soft-LIF" naming overstates the mechanism

See §3.2. Either rename `SwarmAgent` to something that describes the
sigmoid-pseudo-rate model honestly (e.g. `SoftSigmoidRecurrentAgent`)
or replace `think()` with a real LIF (hard threshold + binary spike +
hard reset, with surrogate gradient if differentiability is needed).
Tracked as task #25.

### 11.2 No spatial index → O(n² log n) per step

`get_neighbor_distances` and `get_obstacle_distances` recompute the
full pair-distance vector for every query. Adding a KD-tree
(`scipy.spatial.cKDTree`) or a uniform spatial hash would drop the
per-step cost from `O(n² log n)` to `O(n × k)` for `k`-nearest
queries. At n=500 this would be ~20× faster. Tracked as task #26.

### 11.3 Remaining `swarm_env.py` `# type: ignore`

The previous `agent.py` assignment suppressions are gone. The remaining
swarm suppression is:

- `swarm_env.py:155` — `# type: ignore[no-untyped-def]` on
  `step(self, dt, fields=None)` (because `fields` is a forward
  reference)

The follow-up fix is to use a `TYPE_CHECKING` import of
`CollectiveFields` and type the optional field parameter directly.

### 11.4 Composite fitness weights are hard-coded

`SwarmFitness.composite` does not accept user weights; the 0.30 / 0.20
/ 0.10 / 0.30 / -0.10 mix is in the source. Useful for ranking
within a single experiment, less so for multi-objective sweeps.
Adding `composite(env, weights: dict | None = None)` would close the
gap.

### 11.5 No fixed-seed evaluation in `SwarmEvolver`

Per §7, each individual sees a different environment seed. Fitness
is therefore an average over seeds, not a deterministic test. To
compare two runs you must either pin the evolver seed AND the
generation count, or add a fixed-seed evaluation pass at the end.

---

## 12. Tests

```bash
PYTHONPATH=src python3 -m pytest \
    tests/test_swarm.py \
    tests/test_swarm_control.py \
    tests/test_swarm_fitness_contracts.py \
    tests/test_swarm_agent_evolver_contracts.py -q
# 87 passed in 5.58s (verified 2026-07-02)
```

What the existing tests cover:

- **`test_swarm.py`** (3 test classes): basic agent construction,
  weight setter / getter round-trip, env step shape, evolver one-gen
  best-fitness > 0
- **`test_swarm_control.py`** (291L, large): full GA loop, target
  capture, boundary modes (wrap + clamp), field deposit / decay,
  emotional sync, symbolic field, `evolve_generation` return type,
  evolver `get_best_weights` correctness, fitness composite weights
- **`test_swarm_fitness_contracts.py`** (184L): individual fitness scorer
  edge cases (empty positions, single agent, all-same heading, etc.),
  symbolic field deposit + decay, obstacle penalty for agent on
  exact boundary
- **`test_swarm_agent_evolver_contracts.py`**: public fail-closed
  contracts for `AgentConfig`, `SwarmAgent.weights`, `think`, `act`,
  `reset`, `EvolverConfig`, `evaluate_individual`, mutation, `run`, and
  `get_best_weights`.

What is NOT tested:

- **Spatial-index correctness** — there is no spatial index to test;
  task #26 would need new tests for `cKDTree` parity.
- **Long-horizon evolver convergence** — every test uses ≤ 3
  generations; nobody asserts that fitness improves over 100+ gens.
- **Field stability under high deposit rate** — the chemical clip at
  0 is tested, but not whether high-rate deposits ever overflow the
  `float64` range. (Unlikely in practice, but undocumented.)

---

## 13. References

The swarm code does not cite specific publications, so the references
below are the standard literature for each component:

- **Genetic algorithms over neural network weights** — Stanley K. O.,
  Miikkulainen R. "Evolving Neural Networks through Augmenting
  Topologies." *Evolutionary Computation* 10(2):99-127 (2002). NEAT,
  the canonical reference; the implementation here is much simpler
  (fixed topology, truncation selection, uniform crossover).
- **Sigmoid-pseudo-rate "soft-LIF"** — closer to Wong K.-F., Wang X.-J.
  "A Recurrent Network Mechanism of Time Integration in Perceptual
  Decisions." *J Neurosci* 26(4):1314-1328 (2006), or any
  rate-coded RNN. For true LIF surrogate gradients see Neftci E.,
  Mostafa H., Zenke F. "Surrogate Gradient Learning in Spiking
  Neural Networks." *IEEE Signal Processing Magazine* 36(6):51-63
  (2019).
- **Chemical-field swarm communication** — Bonabeau E., Dorigo M.,
  Theraulaz G. *Swarm Intelligence: From Natural to Artificial Systems*
  (Oxford UP, 1999). The pheromone-deposit-and-diffuse pattern.
- **Rayleigh statistic for circular alignment** — Mardia K. V., Jupp
  P. E. *Directional Statistics* (Wiley, 2000).

Internal:

- Network simulation engine: [`api/network.md`](network.md)
- Neuron model registry (true LIF variants): [`api/neurons.md`](neurons.md)

---

## 14. Auto-rendered API

::: sc_neurocore.swarm
    options:
      show_root_heading: true
      show_source: true
      members:
        - AgentConfig
        - SwarmAgent
        - EnvConfig
        - SwarmEnvironment
        - FieldConfig
        - CollectiveFields
        - SwarmFitness
        - EvolverConfig
        - SwarmEvolver
