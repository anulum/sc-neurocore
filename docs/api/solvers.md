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

## Exact-current LIF profile

`ExactCurrentLIFProfile` is the versioned semantic contract for the existing
`ExactLIFSolver` exact-flow, hard-reset current-based LIF model. Its canonical JSON and
SHA-256 bind the equation, model-source bytes, parameters, normalized units,
piecewise-constant input family, analytical crossing solver, inclusive
threshold, hard reset, zero refractory interval, explicit shot reset,
binary64 behavior, and absence of stochastic state. Unknown fields, schema
versions, unit changes, source drift, and digest mismatches fail closed.

`ExactCurrentLIFSession` preserves voltage and shot-relative time across calls.
Each call returns an immutable packet containing the exact producer commit,
profile digest, input ticks, initial and final state, exact spike events, and an
ordered state trace. Threshold and reset samples share the analytical crossing
timestamp, so downstream consumers do not have to infer event ordering from a
grid-aligned spike count.

```python
from sc_neurocore.solvers import (
    CurrentDriveTick,
    ExactCurrentLIFProfile,
    ExactCurrentLIFSession,
)

profile = ExactCurrentLIFProfile(tau_ms=10.0)
session = ExactCurrentLIFSession(
    profile,
    producer_commit="0123456789abcdef0123456789abcdef01234567",
    shot_id="mif-shot-17",
)
packet = session.execute(
    [
        CurrentDriveTick(duration_ms=2.0, currents=(10.0, 20.0)),
        CurrentDriveTick(duration_ms=3.0, currents=(30.0,)),
    ]
)
assert packet.profile_digest == profile.digest
checkpoint = session.serialize_state()
validated = type(packet).from_json(
    packet.to_json(),
    profile=profile,
    expected_producer_commit="0123456789abcdef0123456789abcdef01234567",
)
assert validated == packet
```

The canonical default profile and a four-tick complete-state receipt are shipped
as `exact_current_lif_profile_v1.json` and
`exact_current_lif_multitick_v1.json` under
`src/sc_neurocore/neurons/reference_trace_data/`. The receipt binds its profile
digest and the exact implementation commit and includes three off-grid events.

The profile is a precise software/reference contract. It does not by itself
claim fixed-point equivalence, generated RTL parity, synthesis timing, PPA,
board/HIL evidence, or biological fidelity. Those remain separate gates.

::: sc_neurocore.solvers.exact_lif_profile
    options:
      show_root_heading: true
