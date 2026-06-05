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

For scripts, Dockerfiles, or lab SOPs that want every install command to name a
profile explicitly, `core` is a stable alias for the same base dependency set:

```bash
pip install "sc-neurocore[core]"
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

### Base install evidence

The install-profile audit in `benchmarks/results/install_profile_audit.json` records the base boundary used for release evidence. The latest measured run installed the local
`sc-neurocore` wheel with its declared base dependencies, imported the public
package, and verified that PyTorch, JAX, Qiskit, PennyLane, Lava, FastAPI, MPI,
NIR, HTTP client, plotting, GPU, and Studio stacks were not pulled into the base
environment.

Current measured evidence:

| Field | Value |
| --- | --- |
| Command | `python -m pip install <repo>` |
| Elapsed time | 12.942 s |
| Installed packages | `defusedxml`, `numpy`, `pip`, `sc-neurocore`, `scipy` |
| Heavy optional packages installed | None |
| Public import smoke | `sc_neurocore.__version__ == "3.15.18"` and 44 public exports |

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
| `pip install "sc-neurocore[core]"` | Explicit base install for reproducible scripts and Dockerfiles | No additional packages beyond base `numpy` / `scipy` |
| `pip install "sc-neurocore[training]"` | Training PyTorch-backed models | `torch` |
| `pip install "sc-neurocore[nir]"` | Importing/exporting Neuromorphic Intermediate Representation graphs | `nir` |
| `pip install "sc-neurocore[hdl]"` | Equation-to-HDL workflows, unit-checked equations, packaged HDL primitives | `pint`; bundled `.v` / `.sv` / OpenROAD helper artefacts |
| `pip install "sc-neurocore[gpu]"` | Research-grade CuPy CUDA experiments; requires local CUDA compatibility | `cupy-cuda12x` |
| `pip install "sc-neurocore[jax]"` | Research-grade JAX experiments; requires local accelerator/runtime compatibility | `jax`, `jaxlib` |
| `pip install "sc-neurocore[quantum]"` | Research-grade quantum-circuit experiments; Qiskit/PennyLane are optional and never installed by default | `qiskit`, `pennylane`, `qiskit-aer` |
| `pip install "sc-neurocore[studio]"` | Web studio / local design UI | `fastapi`, `uvicorn`, `httpx` |
| `pip install "sc-neurocore[bioware]"` | Biological closed-loop and spike-sorting prototypes | `scikit-learn` |
| `pip install "sc-neurocore[docs]"` | Building this documentation locally | `mkdocs`, `mkdocs-material`, `mkdocstrings` |
| `pip install "sc-neurocore[dev]"` | Contributor lint/test work | `pytest`, `ruff`, `mypy`, `bandit`, docs helpers |

Use `full` only for local research environments where large optional packages
are acceptable:

```bash
pip install "sc-neurocore[full]"
```

The `hdl` and `full` profiles use the same wheel artefact set as the base
package: source RTL primitives under `hardware/`, baseline offline HDL
primitives under `hdl/primitives/`, safety SystemVerilog under
`hdl_gen/safety/`, and OpenROAD helper scripts under `hdl_gen/openroad_flow/`.
Those files are bundled with the wheel so install-time extras only decide which
Python dependencies are added; external synthesis tools still produce
device-specific bitstreams or routed reports locally.

## Offline HDL shipping profile

The `hdl-offline` shipping profile is the audited baseline for machines where
Vivado is unavailable or intentionally excluded from the install image. It
ships the static primitive RTL used by the baseline stochastic-computing FPGA
path:

- `sc_bitstream_encoder.v`
- `sc_bitstream_synapse.v`
- `sc_dense_layer_core.v`
- `sc_dotproduct_to_current.v`
- `sc_firing_rate_bank.v`
- `sc_lif_neuron.v`

Python consumers can enumerate or read those files from the installed wheel:

```python
from sc_neurocore.hdl.resources import baseline_primitive_text, list_baseline_primitive_rtl

for name in list_baseline_primitive_rtl():
    rtl = baseline_primitive_text(name)
```

Docker and conda consume the same wheel-contained resources:

```bash
docker build -f deploy/Dockerfile --build-arg INSTALL_EXTRAS=hdl -t sc-neurocore:hdl .
conda build conda/
```

The install-profile audit fails if the package-data pattern, static primitive
files, Docker HDL profile hook, or conda base runtime dependencies drift from
the release contract. Current audit evidence is recorded in
`benchmarks/results/install_profile_audit.json`.

The `full` profile is the CPU-side union for training, NIR, Studio, HDL, codec,
bioware, and quantum workflows. It is a local research environment profile, not
the standard user install. It deliberately does not pull GPU-, MPI-, Lava-,
Julia-, or JAX-specific stacks because those depend on local hardware, drivers,
or external runtimes.

Heavy backends are always opt-in. Qiskit, PennyLane, CuPy, JAX, Lava, MPI,
Julia, Go, Mojo, and WGSL tooling are research-grade integration surfaces until
their target environment, install profile, and verification artefacts are named
explicitly in the relevant docs. A default installation must not require them.

## Research-only polyglot layer

The source tree contains Julia, Go, Mojo, and WGSL implementations for selected
kernels. Treat this as a research and benchmarking layer, not as the shipped
runtime shape.

This layer exists to compare implementation strategies under controlled parity
and timing harnesses. It supports internal acceleration research only; it is
not a user-facing install profile, not part of the default wheel contract, and
not a prerequisite for standard simulation, training, NIR export, HDL
generation, or FPGA deployment.

Rules of thumb:

- A base `pip install sc-neurocore` does not require Julia, Go, Mojo, or WGSL
  tooling.
- `pip install "sc-neurocore[full]"` still does not imply a Julia, Go, Mojo, or
  WGSL toolchain; those runtimes remain source-checkout research dependencies.
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

Before publishing a hardware report, capture the local toolchain inventory:

```bash
python tools/eda_toolchain_versions.py --pretty --out build/eda-toolchain.json
```

Release gates can also require specific tools and version substrings:

```bash
python tools/eda_toolchain_versions.py \
  --require vivado --expect vivado=v2025.2 \
  --require yosys --expect yosys=0.63
```

The JSON inventory records Vivado, OpenROAD, Yosys, nextpnr, IceStorm,
Trellis, Quartus, Lattice Diamond/Radiant, PYNQ, and OpenROAD/PDK pin fields.
