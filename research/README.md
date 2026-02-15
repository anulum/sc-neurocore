# Research — Speculative & Theoretical Modules

This directory contains experimental and speculative modules that explore
theoretical frontiers of neuromorphic computing. These are **not production
code** and are not part of the main `sc_neurocore` package.

## Packages

| Package | Description |
|---------|-------------|
| `eschaton/` | Entropy-survival computing, heat-death layers, nested universe simulation |
| `exotic/` | Anyon computing, chemical reaction-diffusion, fungal networks, space radiation hardening |
| `meta/` | DAO governance, Fermi game theory, omega integration, singularity, time crystals, time travel |
| `post_silicon/` | Claytronics, femto-scale computing, reversible computing, synthetic cells |
| `transcendent/` | Multiverse solvers, noetic semiotics, spacetime lattice, vacuum decay |
| `speculative/` | Individual modules extracted from core packages (consciousness, qualia, immortality, etc.) |

## Usage

These modules are not importable from the main package. To use them:

```python
import sys
sys.path.insert(0, "research")

from eschaton.heat_death import HeatDeathLayer
from speculative.analysis_consciousness import PhiEvaluator
```

## Tests

```bash
PYTHONPATH=src:bridge:research python -m pytest research/tests/ -x -q
```

## Contributing

Contributions to these theoretical modules are welcome. Please open an issue
to discuss your idea before submitting a PR, as these modules explore territory
that may not have established scientific consensus.
