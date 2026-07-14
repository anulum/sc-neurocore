# Command-Line Interface (`sc-neurocore`)

The SC-NeuroCore command-line interface is an `argparse` package rooted at
`src/sc_neurocore/cli/`. The installed entry point remains
`sc_neurocore.cli:main`, so existing console-script integrations do not change.
The equivalent module invocation is:

```bash
python -m sc_neurocore.cli
```

The CLI is a Python control plane. It validates input, dispatches work, writes
artefacts, and reports downstream status. Numerical kernels and hardware
generators retain their own Python/Rust/Julia/Mojo/Go parity obligations; the
argument parser itself has no compute-kernel counterpart.

## First successful path

Start by inspecting the installed runtime:

```bash
sc-neurocore info
```

The command reports the package and Python versions, the optional Rust engine
version and SIMD tier, and installed NumPy/JAX versions without importing
optional dependencies merely to discover their metadata.

Top-level help groups the product into four modes:

```text
Model     info, compile, compile-nir, serve, map-nir
Hardware  deploy, collect-synthesis, scnir, formal, hub-init
Studio    studio and studio-* operator commands
Maintain  benchmark, preflight
```

Only top-level command names are shown initially. Use progressive disclosure
for command-specific options:

```bash
sc-neurocore compile-nir --help
sc-neurocore formal --help
sc-neurocore studio-bootstrap-admin --help
```

## Installation and entry-point contract

`pyproject.toml` registers:

```toml
[project.scripts]
sc-neurocore = "sc_neurocore.cli:main"
```

The three supported invocation forms are equivalent:

```bash
sc-neurocore --version
python -m sc_neurocore.cli --version
python -c 'from sc_neurocore.cli import main; raise SystemExit(main(["--version"]))'
```

`main()` returns an integer status and accepts an optional explicit argument
sequence for embedding and tests. With no sequence it reads `sys.argv`,
preserving the original console-script contract.

## Command reference

### Model mode

| Command | Purpose | Primary downstream surface |
| --- | --- | --- |
| `info` | Report runtime and optional-engine status | package metadata and `sc_neurocore_engine.simd_tier()` |
| `compile` | Lower one ODE into Verilog and optional HLS C++ | equation builder/compiler, HLS exporter, optional Yosys adapter |
| `compile-nir` | Lower NIR/ONNX into RTL, SC-NIR metadata, manifests, and optional handoff audit | NIR bridge and SC-NIR writers |
| `serve` | Serve a NIR graph over the spike-stream protocol | `sc_neurocore.serve.SpikeServer` |
| `map-nir` | Emit deterministic target-mapping evidence | NIR silicon-mapping writer |

Compile an equation:

```bash
sc-neurocore compile \
  'dv/dt = -(v - E_L) / tau_m + I / C' \
  --threshold 'v > -50' \
  --reset 'v = -65' \
  --params 'E_L=-65,tau_m=10,C=1' \
  --init 'v=-65' \
  --module-name lif_demo \
  --testbench \
  --output build/lif
```

`compile` supports three explicit datapath modes:

- Standard RTL (default).
- Pipelined RTL via `--pipeline auto|N` or named
  `--pipeline-points _mul0,_mul1`.
- Dual-datapath adaptive precision via `--adaptive-precision` and LP/HP
  width/fraction or named precision options.

Optional `--emit-hls` produces `MODULE.hls.cpp` beside the Verilog source.
`--synthesize` runs Yosys only for open-source targets and reports that Vivado
targets require their vendor flow.

Compile a NIR graph with an explicit fixed-point format:

```bash
sc-neurocore compile-nir model.nir \
  --module-name cortical_network \
  --data-width 16 \
  --fraction 8 \
  --interconnect auto \
  --source-kind lfsr \
  --T 512 \
  --audit-handoff \
  --output build/cortical_network
```

`compile-nir` accepts `.nir` and `.onnx`. It rejects `data-width <= 1`, negative
fractions, and `fraction >= data-width` before loading the graph. Generated
bundles include the top module, neuron modules, weight ROM, stochastic source
modules, hierarchy boundary modules when present, `scnir_document.json`, and
`scnir_source_manifest.json`. Folded builds additionally write
`folded_metrics.json`.

Serve and map a graph:

```bash
sc-neurocore serve model.nir --port 8123 --dt 1.0
sc-neurocore map-nir model.nir \
  --hardware-targets loihi2,spinnaker2,akida \
  --T 256 \
  --output build/silicon-map
```

### Hardware mode

| Command | Purpose | Primary downstream surface |
| --- | --- | --- |
| `deploy` | Build FPGA or browser deployment artefacts | NIR bridge, trusted checkpoint loader, ANN-to-SNN conversion, web deployer |
| `collect-synthesis` | Convert vendor reports into optimiser evidence | optimiser report parsers and evidence writer |
| `scnir` | Validate, upgrade, export, or audit SC-NIR evidence | `sc_neurocore.ir` schema/conversion/audit APIs |
| `formal verify-network` | Emit RTL/SVA/SBY/report evidence and replay traces | network formal contracts and optional SymbiYosys |
| `hub-init` | Generate a self-hosted hub Compose bundle | hub bundle generator |

Deploy a NIR graph:

```bash
sc-neurocore deploy model.nir --target ice40 --output build/deploy
```

PyTorch checkpoints are treated as untrusted input. `.pt`/`.pth` deployment
requires an explicit SHA-256 digest and a state-dict-like payload containing a
bounded, finite, floating-point, composition-compatible dense weight chain:

```bash
sc-neurocore deploy model.pt \
  --checkpoint-sha256 "$EXPECTED_SHA256" \
  --target artix7 \
  --output build/deploy
```

Supported targets are `ice40`, `ecp5`, `artix7`, `zynq`, and `web`. The first
two emit Yosys/nextpnr build files, Artix-7/Zynq emit a Vivado Tcl project, and
`web` delegates to the browser-deployment builder.

Collect synthesis evidence:

```bash
sc-neurocore collect-synthesis \
  --design build/design.json \
  --utilisation build/utilisation.rpt \
  --power build/power.rpt \
  --timing build/timing.rpt \
  --accuracy-score 0.99 \
  --out build/synthesis_evidence.json
```

The collector parses existing reports; it never launches vendor tools.

SC-NIR operations:

```bash
sc-neurocore scnir validate model.scnir.json
sc-neurocore scnir upgrade legacy.scnir.json --output current.scnir.json
sc-neurocore scnir export model.nir --output model.scnir.json
sc-neurocore scnir audit-hdl build/network --output build/scnir_handoff_audit.json
sc-neurocore scnir compatibility . --output build/scnir_compatibility.json
sc-neurocore scnir closure-audit . --output build/scnir_closure_audit.json
```

Upgrade and export require an explicit `--output`; neither command overwrites
or invents a destination implicitly. Compatibility and closure audits default
their evidence root to the current working directory.

Formal verification:

```bash
sc-neurocore formal verify-network \
  --module-name dense_lif_frontier \
  --input-width 3 \
  --output-width 2 \
  --window-cycles 16 \
  --max-spikes 3 \
  --refractory-cycles 2 \
  --antagonistic-pair 0,1 \
  --coactivation-cap 1 \
  --output build/formal
```

Optional constraints cover refractory intervals, antagonistic outputs,
temporal separation, population coactivation, silence after coactivation, and
maximum population inactivity. `--spike-trace` replays a JSON trace against
the same contracts. `--run-symbiyosys` records passed, failed, or unavailable
tool status in the evidence report.

Generate a hub bundle:

```bash
sc-neurocore hub-init --host 127.0.0.1 --port 8765 --output build/hub
```

### Studio mode

Launch the local application:

```bash
sc-neurocore studio --port 8001
```

Operator commands are deliberately separate from the launcher:

| Command | Purpose |
| --- | --- |
| `studio-backup-plan` | Emit a durable-state backup/restore manifest |
| `studio-deployment-profile` | Emit `local`, `lab`, or `server` settings as JSON or env lines |
| `studio-preflight` | Evaluate release-readiness posture |
| `studio-bootstrap-admin` | Create the first service-account identity file |
| `studio-add-browser-user` | Add a password-authenticated browser user from stdin |

Identity commands require an explicit identity path. Browser-user creation also
requires a username, at least one role, and `--password-stdin`; passwords are
never accepted as command-line arguments.

### Maintenance mode

```bash
sc-neurocore benchmark
sc-neurocore preflight
```

These commands delegate to the repository benchmark and preflight tools and
return their subprocess status unchanged.

## Package architecture

`src/sc_neurocore/cli/parser.py` is the composition root. It creates the
top-level parser, asks each command module to register its parser, and dispatches
the selected handler. Command modules import heavyweight optional dependencies
inside handlers, keeping top-level help and `--version` lightweight.

| Module | Responsibility |
| --- | --- |
| `parser.py` | top-level help, version handling, parser composition, dispatch |
| `commands/info.py` | runtime and optional-engine status |
| `commands/compile.py` | ODE and NIR compilation |
| `commands/serve.py` | spike-stream server launch |
| `commands/mapping.py` | NIR silicon-mapping report |
| `commands/deploy.py` | trusted model deployment and target project generation |
| `commands/synthesis.py` | synthesis-report evidence collection |
| `commands/scnir.py` | SC-NIR document and audit operations |
| `commands/formal.py` | network formal contracts, replay, and SymbiYosys adapter |
| `commands/hub.py` | self-hosted hub bundle generation |
| `commands/studio.py` | Studio launch and local operator state commands |
| `commands/maintenance.py` | benchmark and preflight delegation |

The parser imports command registrars, while command handlers do not import the
parser. The only command-to-command dependency is compilation's reuse of the
deployment target table and Yosys adapter. No cyclic import is permitted.

The former generated files named `cli.rs`, `cli.jl`, and `cli.mojo` were not
runtime implementations: they embedded Python as comments or returned constant
values, had no callable entry point, and were not referenced by builds or tests.
They were removed rather than presented as false parity. Real downstream
compute surfaces retain their language-specific implementations and benchmarks.

## Wiring and regression ownership

| Public path | Downstream contract | Focused regression surface |
| --- | --- | --- |
| parser/help/version | all command registrars and `sc_neurocore.cli:main` | `tests/test_cli_dispatch.py`, `tests/test_cli_architecture.py` |
| equation compilation | equation builder/compiler, HLS, pipeline/adaptive precision, Yosys | `tests/test_cli_equation.py`, focused equation-compiler CLI tests |
| NIR compilation | NIR bridge, FPGA compiler, SC-NIR writers and handoff audit | `tests/test_cli_nir.py`, `tests/cli_nir_test_support.py` |
| deployment | trusted checkpoint loader, conversion, web/FPGA project adapters | `tests/test_cli_deploy.py` |
| formal verification | network contracts, replay, SBY runner | `tests/test_cli_formal.py` |
| SC-NIR operations | schema, conversion, compatibility, closure, HDL audit | `tests/test_cli_scnir.py` and focused `tests/test_scnir_*.py` CLI cases |
| Studio | app factory and platform state APIs | `tests/test_cli_studio.py` and focused Studio CLI cases |
| synthesis evidence | optimiser report reader/writer | `tests/test_cli_synthesis.py`, `tests/test_optimizer/test_synthesis_evidence_cli.py` |
| serve/map/hub | server, mapping writer, hub generator | `tests/test_cli_serve.py`, `tests/test_cli_mapping.py`, `tests/test_cli_hub.py` |

The NIR co-simulation oracle records post-threshold, post-reset membrane output,
matching both generated RTL non-blocking assignments and the bit-true fixed-point
Python model. Direct, recurrent, AER, and mixed-signal networks are compared
against that reference through real Icarus Verilog simulations.

## Error and security boundaries

- Missing required command inputs return status `1` with actionable usage.
- Parser syntax and unsupported option choices remain `argparse` errors.
- NIR/ONNX extensions and fixed-point formats are validated before graph work.
- PyTorch deployment requires an exact SHA-256 and bounded tensor payload.
- Dense tensors must be finite, floating-point, non-empty, and
  composition-compatible.
- Studio identity paths and password-stdin intent must be explicit.
- SC-NIR export/upgrade outputs must be explicit.
- Formal compound constraints reject missing, non-integer, negative, or
  self-conflicting fields.
- External tool failures are converted to bounded status/report evidence; no
  `shell=True` command construction is used.

## Performance evidence

`benchmarks/bench_cli_startup.py` measures two source trees in fresh Python
processes. Each child imports `sc_neurocore.cli`, dispatches `--version`, and
reports import time and maximum RSS. The harness also records end-to-end process
wall time, raw samples, medians, source hashes, Git state, Python/platform data,
CPU affinity, governor/frequency, and load averages.

The 2026-07-12 local result is
`benchmarks/results/local_python_2026-07-12_cli_startup.json`:

| Metric (median, 30 runs after 5 warmups) | Parent flat module | CLI package | Delta |
| --- | ---: | ---: | ---: |
| CLI import | 114.633 ms | 89.100 ms | -22.27% |
| fresh-process wall time | 198.215 ms | 198.189 ms | -0.013% |
| maximum RSS | 20,568 KiB | 20,568 KiB | 0% |

The child processes were pinned to CPU 0 with `taskset`, but the CPU was not
exclusively isolated. The workstation used the `powersave` governor and had load
averages near 11–12 during capture. These values are local regression context,
not publication-grade throughput claims. Rerun on reserved isolated cores before
citing performance externally.

Rust, Julia, Mojo, and Go are marked `not_applicable` in this artefact because
argument parsing and process dispatch are Python-only. This does not waive
polyglot parity for any numerical kernel reached after dispatch.

## Tests, typing, and coverage

The modular production package and its command tests are checked with Ruff and
strict mypy. The focused CLI cohort is module-scoped; the repository-wide local
suite is not required or recommended for this change.

```bash
python -m pytest -q tests/test_cli_*.py

python -m mypy --strict \
  src/sc_neurocore/cli \
  tests/cli_test_support.py \
  tests/cli_nir_test_support.py \
  tests/test_cli_*.py
```

Combined CLI and linked command-contract coverage reaches 100% statement and
branch coverage: 1,172 statements and 294 branches, with no CLI coverage omit,
test skip, or coverage suppression.

```bash
COVERAGE_FILE=/tmp/sc_neurocore_cli.coverage \
python -m coverage run --branch --source=sc_neurocore.cli \
  -m pytest -q tests/test_cli_*.py

COVERAGE_FILE=/tmp/sc_neurocore_cli.coverage \
python -m coverage run --append --branch --source=sc_neurocore.cli \
  -m pytest -q \
  tests/test_scnir_schema.py \
  tests/test_nir_bridge/test_scnir_export.py \
  tests/test_scnir_handoff_audit.py \
  tests/test_scnir_compatibility.py \
  tests/test_studio_backup_plan.py \
  tests/test_studio_deployment_profiles.py \
  tests/test_studio_identity_bootstrap.py \
  tests/test_optimizer/test_synthesis_evidence_cli.py \
  tests/test_equation_compiler_cli.py \
  -k cli

COVERAGE_FILE=/tmp/sc_neurocore_cli.coverage \
python -m coverage report -m
```

The benchmark harness has its own real-subprocess contract test:

```bash
python -m pytest -q tests/test_bench_cli_startup.py
```

## Auto-rendered API

The generated symbol reference in `docs/API_REFERENCE.md` is produced by
`scripts/generate_docs.py`. Public command handlers and parser functions carry
complete parameter and return documentation; command-specific details belong in
this guide and `COMMAND --help` rather than in the top-level help screen.
