<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Installation, extras and environments
-->

# Installation, Extras and Environments

This text was part of the repository README until the README became a short landing page.

## Install Extras

Start with the base package. It installs the Python package plus `numpy` and
`scipy`; it does not install PyTorch, JAX, Qiskit, PennyLane, Lava, FastAPI, or
hardware toolchains.

```bash
pip install sc-neurocore              # base package: core simulation, compiler, HDL scaffold
pip install sc-neurocore[minimal]     # explicit v4 minimal profile
pip install sc-neurocore[core]        # explicit base profile
pip install sc-neurocore[training]    # PyTorch-backed training
pip install sc-neurocore[nir]         # NIR import/export
pip install sc-neurocore[studio]      # local web studio
pip install sc-neurocore[bioware]     # biological closed-loop prototypes
```

Run the dependency-light smoke path with `python examples/minimal_smoke_demo.py`.
It exercises root public imports, stochastic bitstreams, one LIF bitstream pass,
and packaged Verilog primitive discovery without quantum, Lava, gdsfactory,
PyTorch, JAX, or hardware-only dependencies.

Acceleration and research extras are intentionally opt-in:

```bash
pip install sc-neurocore[accel]       # Numba JIT experiments
pip install sc-neurocore[gpu]         # CuPy CUDA experiments
pip install sc-neurocore[jax]         # JAX-backed experiments
pip install sc-neurocore[quantum]     # research-grade Qiskit/PennyLane bridges
pip install sc-neurocore[annealing]   # dimod/neal parity for the Ising/QUBO bridge
pip install sc-neurocore[research]    # plotting, graph, ONNX, and torch research stack
pip install sc-neurocore[full]        # local research environment only; pulls heavy extras
```

See [Install Profiles](install_profiles.md) before using `full`.
The default package and FPGA scaffold flow do not require those heavy extras.

For development (includes all modules and source-only research code):

```bash
pip install -e ".[dev]"               # editable install with pytest, mypy, ruff, hypothesis
```

Pinned dependency files for reproducible environments:

```bash
pip install -r requirements.txt       # runtime only
pip install -r requirements-dev.txt   # runtime + dev tools
```

## Development Setup

```bash
git clone https://github.com/anulum/sc-neurocore.git
cd sc-neurocore
pip install -e ".[dev]"    # editable install with all dev tools
make preflight             # verify setup (the gates for what you changed)
```

If you are changing the Rust bridge locally, install `bridge/` in the same
environment or run source-tree commands with `PYTHONPATH=src:bridge`.

Build docs locally:
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve
```

## Docker

Docker builds can include the optional Rust acceleration engine when the build
profile installs it. Published speed evidence is scoped to committed benchmark
artefacts such as `benchmarks/results/rust_scaling_benchmark.json`:

```bash
# Build
make docker-build
# or: docker build -f deploy/Dockerfile -t sc-neurocore:latest .

# Build the offline HDL shipping profile with packaged baseline RTL primitives
# and the hash-locked HDL dependency set. Vivado is not required for this image.
docker build -f deploy/Dockerfile --build-arg INSTALL_EXTRAS=hdl -t sc-neurocore:hdl .

# Run interactive Python shell
make docker-run
# or: docker run --rm -it sc-neurocore:latest

# Smoke test via docker compose
docker compose -f deploy/docker-compose.yml up
```

Pre-built images are published to GHCR on every release:

```bash
docker pull ghcr.io/anulum/sc-neurocore:latest
docker run --rm -it ghcr.io/anulum/sc-neurocore:latest
```
