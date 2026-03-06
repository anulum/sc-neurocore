# Speculative Modules (research/)

The `research/` directory contains theoretical explorations and speculative modules
that are **not part of the installable `sc_neurocore` package**.

These modules are maintained separately for academic interest and future research
directions. They are not covered by the core test suite's 97% coverage threshold.

## Packages

| Package | Description |
|---------|-------------|
| `eschaton/` | Terminal-horizon computing explorations |
| `exotic/` | Non-standard computing substrates (anyon, chemical, fungal, mechanical) |
| `meta/` | Meta-computing paradigms (hyper-Turing, time-crystal, vacuum) |
| `post_silicon/` | Post-silicon computing architectures |
| `transcendent/` | Transcendent computing models (noetic, cosmic) |
| `speculative/` | Individual speculative modules extracted from core packages |

## Usage

Speculative modules are not auto-imported. To use them, add `research/` to your
Python path:

```bash
PYTHONPATH=src:bridge:research python your_script.py
```

```python
from exotic.space import SpaceComputingSubstrate
from meta.time_crystal import TimeCrystalComputer
```

## Running Speculative Tests

```bash
PYTHONPATH=src:bridge:research python -m pytest research/tests/ -v
```

See the `research/` directory in the repository root for full details.
