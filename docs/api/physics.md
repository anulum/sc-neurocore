# Physics — Stochastic PDE Solvers + Wolfram Hypergraph

Physics simulation using stochastic methods: heat equation via random walks (Feynman-Kac) and Wolfram Physics Project hypergraph evolution.

## StochasticHeatSolver — 1D Heat Equation via Random Walks

Solves the 1D heat equation using the Feynman-Kac connection between diffusion PDEs and Brownian motion. N random walkers perform discrete random walks on a 1D lattice; their density at time t approximates the temperature profile u(x,t).

Walker dynamics: at each step, move -1, 0, or +1 with probabilities [0.25, 0.5, 0.25]. Reflective boundary conditions (walkers clipped to [0, length-1]).

| Parameter | Meaning |
|-----------|---------|
| `length` | Lattice size (spatial resolution) |
| `num_walkers` | Number of random walkers (more = smoother profile) |
| `alpha` | Diffusion coefficient (reserved for future dt scaling) |

Methods: `step()` (advance one timestep), `get_temperature_profile()` → normalized density histogram.

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
from sc_neurocore.physics.heat import StochasticHeatSolver
from sc_neurocore.physics.wolfram_hypergraph import WolframHypergraph
import numpy as np

# Heat equation
solver = StochasticHeatSolver(length=100, num_walkers=10000, alpha=0.1)
solver.walkers[:] = 50  # point source at center
for _ in range(200):
    solver.step()
profile = solver.get_temperature_profile()  # diffused Gaussian-like

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
