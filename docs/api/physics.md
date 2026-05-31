# SPDX-License-Identifier: AGPL-3.0-or-later
# Physics — Stochastic PDE Solvers + Wolfram Hypergraph

Physics simulation using stochastic methods: heat equation via reflected
Brownian Feynman-Kac paths and Wolfram Physics Project hypergraph evolution.

## FeynmanKacHeatSolver — 1D Heat Equation via Reflected Brownian Motion

Solves the 1D heat equation using the Feynman-Kac connection between diffusion
PDEs and Brownian motion. Walkers follow Euler-Maruyama Brownian increments
with variance `2 * diffusivity * dt` and exact reflective Neumann boundaries on
`[0, length]`.

This is not a clipped lattice random walk. Boundary reflection uses
triangle-wave folding with period `2 * length`, so arbitrarily large stochastic
increments remain inside the physical domain without changing the reflected
transition kernel.

| Parameter | Meaning |
|-----------|---------|
| `length` | Positive domain length `L` |
| `diffusivity` | Non-negative heat-equation coefficient `alpha` |
| `num_walkers` | Positive number of Monte Carlo walkers |
| `dt` | Positive Euler-Maruyama timestep |
| `seed` | Integer seed for reproducible trajectories |

Methods:

- `set_initial_delta(x_0)` — initialize all walkers at a point in `[0, L]`
- `set_initial_distribution(f, n_grid)` — sample a non-negative finite density
- `step(n_substeps)` — advance reflected Brownian paths
- `evolve_to(T)` — advance monotonically to a finite target time
- `get_density(n_bins)` — probability density histogram integrating to one
- `expectation(observable)` — Monte Carlo Feynman-Kac expectation

`StochasticHeatSolver` is retained as a backwards-compatible alias for
`FeynmanKacHeatSolver`.

## WolframHypergraph — Discrete Space-Time Evolution

Simulates the Wolfram Physics Project hypergraph rewrite system. The universe is a set of hyperedges (relations between nodes). Evolution applies a rewrite rule:

`{{x, y}, {y, z}} → {{x, z}, {x, w}, {y, w}}`

This is triangle completion with a new node w. The graph grows at each step, and the effective spatial dimension emerges from the topology.

| Parameter | Meaning |
|-----------|---------|
| `edges` | Initial hyperedge list (list of int tuples) |
| `max_node_id` | Highest existing node ID |

Methods:

- `evolve(steps)` — Apply rewrite rule for N steps
- `dimension_estimate()` — Estimate effective dimension via BFS neighborhood growth: measures how |B(r)| scales with r, fits d from V(r) ~ r^d

## Usage

```python
from sc_neurocore.physics.heat import FeynmanKacHeatSolver
from sc_neurocore.physics.wolfram_hypergraph import WolframHypergraph
import numpy as np

# Heat equation
solver = FeynmanKacHeatSolver(
    length=1.0,
    diffusivity=0.1,
    num_walkers=10000,
    dt=1e-3,
    seed=42,
)
solver.set_initial_delta(0.5)
solver.evolve_to(0.2)
profile = solver.get_density(n_bins=64)
u_cos = solver.expectation(lambda x: np.cos(np.pi * x))

# Wolfram hypergraph
wh = WolframHypergraph(
    edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
    max_node_id=4,
)
wh.evolve(steps=10)
print(f"Edges: {len(wh.edges)}, Nodes: {wh.max_node_id}")
print(f"Effective dimension: {wh.dimension_estimate():.2f}")
```

::: sc_neurocore.physics
    options:
      show_root_heading: true
