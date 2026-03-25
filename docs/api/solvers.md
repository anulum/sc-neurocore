# Solvers

Combinatorial optimization via SC-native Ising machines.

- `StochasticIsingGraph` — Quantum-inspired Ising solver. Spins S_i in {-1, +1} (mapped to 0/1 for SC). Energy: `E = -Sum(J_ij * S_i * S_j) - Sum(h_i * S_i)`. Finds minimum-energy configuration via simulated annealing with SC arithmetic.

Maps to SC hardware: spin products = AND gates, energy accumulation = popcount.

```python
import numpy as np
from sc_neurocore.solvers import StochasticIsingGraph

J = np.random.randn(10, 10)
J = (J + J.T) / 2
np.fill_diagonal(J, 0)
solver = StochasticIsingGraph(num_spins=10, J=J)
solution = solver.solve(n_steps=1000)
```

::: sc_neurocore.solvers.ising
    options:
      show_root_heading: true
