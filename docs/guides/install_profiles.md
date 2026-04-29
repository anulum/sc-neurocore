<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Install profiles and dependency boundaries
-->

# Install Profiles

SC-NeuroCore keeps the default install small. Start with the base package, then
add extras only for the workflows you actually run.

## Base install

```bash
pip install sc-neurocore
```

The base wheel installs the public Python package surface and the core numeric
dependencies declared in `pyproject.toml`: `numpy` and `scipy`. It does not
install PyTorch, JAX, Qiskit, PennyLane, Lava, FastAPI, MPI, Vivado, Yosys, or
other hardware toolchains.

The base path is enough for:

- core stochastic-computing layers and bitstream utilities;
- Python simulation and many model-zoo examples;
- HDL generation and project scaffolding;
- CPU-side report parsing and JSON planning tools.

## Rust engine

The Rust engine is optional acceleration. If an engine wheel or local source
build is present, SC-NeuroCore detects `sc_neurocore_engine` at import time and
uses it for supported hot paths. If it is absent, Python/NumPy fallbacks remain
available.

Use a source checkout when building the engine locally:

```bash
git clone https://github.com/anulum/sc-neurocore.git
cd sc-neurocore
maturin develop
```

Check what the current environment can use:

```bash
sc-neurocore info
```

## Optional extras

| Install command | Use when | Adds |
| --- | --- | --- |
| `pip install "sc-neurocore[training]"` | Training PyTorch-backed models | `torch` |
| `pip install "sc-neurocore[nir]"` | Importing/exporting Neuromorphic Intermediate Representation graphs | `nir` |
| `pip install "sc-neurocore[gpu]"` | CuPy CUDA experiments | `cupy-cuda12x` |
| `pip install "sc-neurocore[jax]"` | JAX-backed experiments | `jax`, `jaxlib` |
| `pip install "sc-neurocore[quantum]"` | Quantum-circuit experiments | `qiskit`, `pennylane`, `qiskit-aer` |
| `pip install "sc-neurocore[studio]"` | Web studio / local design UI | `fastapi`, `uvicorn`, `httpx` |
| `pip install "sc-neurocore[bioware]"` | Biological closed-loop and spike-sorting prototypes | `scikit-learn` |
| `pip install "sc-neurocore[docs]"` | Building this documentation locally | `mkdocs`, `mkdocs-material`, `mkdocstrings` |
| `pip install "sc-neurocore[dev]"` | Contributor lint/test work | `pytest`, `ruff`, `mypy`, `bandit`, docs helpers |

Use `full` only for local research environments where large optional packages
are acceptable:

```bash
pip install "sc-neurocore[full]"
```

## Research-only polyglot layer

The source tree contains Julia, Go, Mojo, and WGSL implementations for selected
kernels. Treat this as a research and benchmarking layer, not as the shipped
runtime shape.

Rules of thumb:

- A base `pip install sc-neurocore` does not require Julia, Go, Mojo, or WGSL
  tooling.
- FPGA deployment emits SystemVerilog and tool scripts; it does not ship the
  polyglot benchmark matrix to the device.
- A polyglot file is authoritative only when a maintained Python loader uses it
  and tests cover that path.
- Mirror or transcript files under `src/sc_neurocore/accel/` are not evidence
  that a feature is shipped.

See [Acceleration Mirror Authority](accel_mirror_authority.md) for the
authoritative-entrypoint list.

## Hardware toolchains

FPGA and ASIC tools are external to Python packaging:

- Yosys / nextpnr are needed only when you want open-source synthesis or
  place-and-route.
- Vivado is needed only for Xilinx implementation and power/timing reports.
- Quartus is needed only for Intel FPGA implementation and reports.

The deploy command can still scaffold a project without those tools. Synthesis
and power reports appear only after an external toolchain has produced them.

### Version evidence for hardware reports

Hardware reports must name the EDA tool version that produced them. Current
release evidence uses:

| Tool | Version status | Scope |
| --- | --- | --- |
| AMD Vivado | `v2025.2` | SHD / PYNQ-Z2 synthesis evidence in `docs/CHANGELOG.md`. |
| OpenROAD | Not release-pinned yet | The ASIC flow can generate OpenROAD-compatible decks, but no committed OpenROAD place-and-route report is release evidence yet. |
| Yosys | Report-specific | Existing reports name the Yosys version beside the numbers because generic-cell counts vary by release. |

Do not quote OpenROAD area, power, timing, or GDS results until the exact
OpenROAD binary or container digest and PDK revision are recorded with the
report.
