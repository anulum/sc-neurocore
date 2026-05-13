# Command-Line Interface (`sc-neurocore`)

**Module:** `sc_neurocore.cli`
**Entry point:** `sc-neurocore` (declared in `pyproject.toml` `[project.scripts]`)
**Source:** `src/sc_neurocore/cli.py` — `argparse`-based, single-`main()` dispatch
**Status (v3.14.0):** deployment, serving, hub bundle generation, compilation, and synthesis-evidence collection have focused tests.

---

## 1. Installation & Entry Point

The CLI ships with the PyPI package `sc-neurocore`. Installing the
package registers the console script `sc-neurocore`:

```bash
pip install sc-neurocore
sc-neurocore --version
# sc-neurocore 3.14.0
```

Without installation (development checkout):

```bash
PYTHONPATH=src python3 -m sc_neurocore.cli --version
```

The entry point is wired through `pyproject.toml`:

```toml
[project.scripts]
sc-neurocore = "sc_neurocore.cli:main"
```

`main()` returns an integer exit code; the module entrypoint calls
`sys.exit(main())`.

---

## 2. Command Reference

The CLI accepts a single positional `command` token chosen from:

```text
{info, benchmark, preflight, deploy, serve, map-nir, hub-init, compile, compile-nir, scnir, studio, collect-synthesis}
```

with an optional positional `model` argument (file path or ODE string,
depending on the command). All other parameters are keyword flags; running
`sc-neurocore -h` prints the full argparse help.

| Command | Purpose | Required positional | Returns |
|---------|---------|---------------------|---------|
| `info` | Print version, Python, Rust engine status, optional deps | — | `0` |
| `benchmark` | Run `pytest benchmarks/benchmark_suite.py --benchmark-only` | — | pytest exit code |
| `preflight` | Run `tools/preflight.py` | — | preflight exit code |
| `compile` | Equation string → SystemVerilog RTL (+ optional TB + Yosys) | ODE string | `0` on success, `1` on missing model |
| `deploy` | NIR/PyTorch model → SC-NeuroCore HDL project for FPGA, or static web scaffold with `--target web` | model file path | `0` on success, `1` on bad format |
| `serve` | Start streaming spike inference server (`SpikeServer`) | `.nir` file path | `0` while running |
| `map-nir` | Generate deterministic silicon-mapping reports for neuromorphic targets | `.nir` file path | `0` on success, `1` on bad input |
| `hub-init` | Generate an offline-first self-hosted Docker Compose hub bundle | — | `0` on success, `1` on invalid config |
| `compile-nir` | Compile NIR/ONNX network files to FPGA artefacts | `.nir` or `.onnx` path | `0` on success, `1` on bad input |
| `scnir` | Validate, upgrade, or export SC-aware NIR metadata documents | `validate model.scnir.json`, `upgrade model.scnir.json --output upgraded.scnir.json`, or `export model.nir --output model.scnir.json` | `0` on success, `1` on invalid input |
| `studio` | Launch Visual SNN Design Studio (FastAPI + Uvicorn) | — | `0` on clean exit, `1` if FastAPI missing |
| `collect-synthesis` | Convert real utilisation, timing, and power reports into optimiser evidence JSON | — | `0` on success, `1` on missing or invalid input |

### 2.1 `info`

Prints package version, Python interpreter version, Rust engine status
(version + SIMD tier from `sc_neurocore_engine.simd_tier()`, or
`not available` if the engine wheel isn't installed), and versions of the
optional `numpy` and `jax` imports if importable.

Verified output on this workstation (no Rust engine wheel installed):

```text
sc-neurocore 3.14.0
Python 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]
Rust engine: not available
NumPy: 2.2.6
```

The Rust-engine status line additionally reports a version mismatch when the
engine wheel reports a different `__version__` than the Python package
(handled by `_format_engine_status`).

### 2.2 `compile-nir`

Compiles `.nir` or `.onnx` network files through NIR import,
`NeuronGraph` lowering, fixed-point quantisation, SC-NIR metadata export, and
FPGA RTL generation:

```bash
sc-neurocore compile-nir model.nir \
  --module-name my_snn \
  --T 1024 \
  --source-kind sobol \
  --base-seed 66 \
  -o build/
```

The output directory contains the top module, per-neuron modules, combined
weight ROM, one SC-NIR stochastic source module per stream, and
`scnir_source_manifest.json`. `--source-kind` currently accepts `lfsr` and
`sobol`; both map to source modules with the `threshold[15:0]`/`bit_out`
interface. `--base-seed` controls deterministic stream seed allocation.

### 2.3 `scnir`

Validates SC-NIR JSON metadata with the fail-closed validator in
`sc_neurocore.ir.scnir_schema`:

```bash
sc-neurocore scnir validate model.scnir.json
```

The validator rejects unknown fields, duplicate stream identifiers, invalid
bitstream lengths, unsupported encodings, invalid fixed-point precision,
under-specified random sources, and correlation constraints that reference
missing streams. The reference schema is `schemas/scnir/scnir.schema.json`.

Canonicalises a supported SC-NIR document through the versioned migration
surface:

```bash
sc-neurocore scnir upgrade model.scnir.json --output upgraded.scnir.json
```

The current schema is `sc-neurocore.scnir.v0.3`. The upgrade command migrates
`sc-neurocore.scnir.v0.1` documents by adding explicit `delay_steps=0` to legacy
streams and migrates every pre-`v0.3` document by adding explicit
`signal_kind` metadata before running the typed validator and deterministic
writer. Unsupported schema versions are rejected instead of being guessed.

Exports deterministic SC-NIR metadata from a NIR graph by using the existing
NIR import path and `NeuronGraph` lowering:

```bash
sc-neurocore scnir export model.nir --output model.scnir.json --T 1024
```

The export records spiking population streams, analogue-state population
streams, weight streams, fixed-point precision, explicit recurrent delay steps,
unique source seeds, and max-correlation constraints. It fails closed on
invalid bitstream length or fixed-point precision configuration.

### 2.4 `benchmark`

Delegates to the project's pytest-benchmark suite via `subprocess.run`:

```python
subprocess.run(
    [sys.executable, "-m", "pytest",
     "benchmarks/benchmark_suite.py", "--benchmark-only"]
).returncode
```

The CLI itself is not benchmarked (see [Section 7](#7-performance)). The exit
code is the pytest exit code; CI consumers should treat any non-zero value as
failure.

### 2.5 `preflight`

Delegates to `tools/preflight.py`. Used by the pre-push policy (see
`feedback_preflight_no_block` memory: never let the pre-push hook run the full
suite — `preflight.py` is the gated subset).

### 2.5 `hub-init`

Writes a deterministic local hub bundle containing:

- `docker-compose.yml`
- `.env.example`
- `hub_manifest.json`
- `model_zoo_index.json`
- `benchmark_plan.json`
- `README.md`
- local `cache/`, `models/`, and `benchmarks/results/` directories

Default generation is offline-first and loopback-bound:

```bash
sc-neurocore hub-init --output build/hub --port 8001
docker compose -f build/hub/docker-compose.yml up studio
```

Operational flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--bind-host` | `127.0.0.1` | Host address used for Studio port publishing |
| `--port` | `8001` | Studio service port |
| `--hub-image` | `sc-neurocore-hub:local` | Compose image tag |
| `--online` | unset | Clears generated offline environment flags |

The generated Compose services use the checked-in `deploy/Dockerfile`, run
with a read-only root filesystem, mount only cache/model/result directories,
set `no-new-privileges`, and include a Studio readiness check against
`/api/health`. The benchmark runner is opt-in via the `benchmark` Compose
profile; it is not started with the Studio service.

### 2.6 `compile`

Compiles a free-form ODE description into synthesisable SystemVerilog using
`sc_neurocore.compiler.equation_compiler.equation_to_fpga`. Optionally emits a
testbench and runs Yosys (open-source FPGA targets only).

Verified four-step output on `dv/dt = -(v-E_L)/tau_m + I/C`:

```bash
PYTHONPATH=src python3 -m sc_neurocore.cli compile \
    "dv/dt = -(v-E_L)/tau_m + I/C" \
    --threshold "v > -50" \
    --reset "v = -65" \
    --params "E_L=-65,tau_m=10,C=1" \
    --init "v=-65" \
    --dt 1.0 \
    --output build/lif \
    --module-name lif_demo
# [1/4] Parsing ODE: dv/dt = -(v-E_L)/tau_m + I/C
#   State variables: ['v']
#   Parameters: ['E_L', 'tau_m', 'C']
# [2/4] Verilog written: build/lif/lif_demo.v
# [3/4] Testbench skipped (use --testbench to generate)
# [4/4] Synthesis skipped (use --synthesize to run Yosys)
```

The generated `lif_demo.v` is 45 lines (Q8.8 fixed-point, 16-bit signed
parameters) and synthesises with Yosys for ICE40/ECP5 targets when
`--synthesize` is passed.

> **Default changed to `--dt 1.0`.** See [Section 9](#9-known-issues)
> for the history. The compiler now rejects values that quantise to 0
> in Q8.8 with an actionable `ValueError`.

### 2.7 `deploy`

FPGA targets run a five-step pipeline (six with auto-synthesis):

1. **Load model** — `.nir` via `nir_lib.read` + `from_nir`, or `.pt`/`.pth`
   via `torch.load(weights_only=True)` with automatic per-`Linear` ReLU
   stitching for ANN-to-SNN conversion.
2. **Quantise** — `Q88` configuration (8 integer + 8 fraction bits = 16-bit
   signed).
3. **Generate Verilog** — calls `equation_to_fpga` for the canonical LIF
   `dv/dt = (-v + I)/tau` with `tau=20.0`.
4. **Copy HDL library** — recursively copies `hdl/` into the output directory,
   excluding testbench (`tb_*`) and `formal/` files.
5. **Generate project files** — `Makefile` for Yosys targets (`ice40`,
   `ecp5`) or `project.tcl` for Vivado targets (`artix7`, `zynq`).
6. **Auto-synthesise** (optional) — runs Yosys + nextpnr + bitstream packing
   when the open-source toolchain is on `$PATH`.

`_TARGET_CONFIGS`:

| Target | Family | Device | Package | Tool |
|--------|--------|--------|---------|------|
| `ice40` | `ice40` | `hx8k` | `ct256` | Yosys |
| `ecp5` | `ecp5` | `85k` | `CABGA381` | Yosys |
| `artix7` | `xc7a` | `xc7a100t` | `csg324` | Vivado |
| `zynq` | `xc7z` | `xc7z020` | `clg400` | Vivado |

`--target web` generates a static browser bundle instead of an FPGA project:

```bash
sc-neurocore deploy model.nir --target web --output build/web --dt 1.0 --T 256
```

Generated files:

- `manifest.json` — deterministic model/runtime contract.
- `index.html` — browser entry point.
- `runtime/sc_neurocore_web.js` — manifest loader and WebGPU capability check.
- `runtime/sc_neurocore_webgpu.wgsl` — minimal SC probability shader scaffold.
- `model/<name>` — copied source model artefact.

The web target accepts `.nir`, `.pt`, `.pth`, and `.json` inputs. It does not
invoke PyTorch, NIR import, Node.js, or a native WASM build during generation,
so packaging can be tested in CI without browser drivers or hardware
accelerators.

### 2.8 `serve`

Loads a `.nir` graph and starts `sc_neurocore.serve.SpikeServer` in blocking
mode on the configured port. Other formats are rejected with exit code 1.

```bash
sc-neurocore serve model.nir --port 8001 --dt 1.0
```

### 2.9 `collect-synthesis`

Collects FPGA synthesis reports into the strict optimiser observation format.
The command requires explicit compiler-design metadata and measured model
accuracy so it cannot invent missing benchmark evidence.

Required flags:

- `--design` — JSON compiler-design metadata for the synthesised model.
- `--utilisation` / `--utilization` — Vivado utilisation or Quartus fitter report.
- `--power` — Vivado or Quartus power report.
- `--accuracy-score` — measured model accuracy or parity score for this design.

Optional flags:

- `--timing` — timing report when latency appears outside the utilisation report.
- `--latency-cycles` — explicit latency when vendor reports do not carry it.
- `--clock-mhz` and `--inferences-per-run` — both required together for
  workload-normalised energy calculation.
- `--out` — output JSON path; without it, JSON is written to stdout.

Example:

```bash
sc-neurocore collect-synthesis \
    --design build/network_design.json \
    --utilisation build/vivado_utilisation.rpt \
    --power build/vivado_power.rpt \
    --timing build/vivado_timing.rpt \
    --accuracy-score 0.991 \
    --clock-mhz 100 \
    --inferences-per-run 1 \
    --out build/synthesis_observations.json
```

The output is accepted by `sc_neurocore.optimizer.load_observations()` and by
`tools/optimise_sc_design.py --evidence`.

### 2.10 `studio`

Launches the Visual SNN Design Studio (FastAPI + Uvicorn) and opens
`http://127.0.0.1:{port}` in the default browser. Requires the `studio`
extra:

```bash
pip install "sc-neurocore[studio]"
sc-neurocore studio --port 8001
```

If FastAPI/Uvicorn is missing, the command exits with code 1 and prints the
install hint.

---

## 3. Architecture

```
                ┌──────────────────────┐
                │ sc-neurocore (entry) │
                └──────────┬───────────┘
                           │ argparse
        ┌──────────────┬──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
   _cmd_info      _cmd_compile   _cmd_deploy  _cmd_collect_synthesis
        │              │              │              │
        ▼              ▼              ▼              ▼
 sc_neurocore   equation_      nir_bridge     optimiser
 _engine        compiler       conversion     synthesis_evidence
 (Rust status)  (Python)       (Python)       (Python)
                       │              │              │
                       ▼              ▼              ▼
                   Verilog        hdl/ +       evidence JSON
                   RTL            Makefile/.tcl
```

`main()` is a single linear dispatcher — no subparsers, no
command classes, no plugin registry. Each `_cmd_*` helper performs its own
imports lazily so that the CLI cold-start cost is bounded by the dispatcher
itself, not by every command's transitive dependency tree.

### Private helpers

| Symbol | Purpose |
|--------|---------|
| `_cmd_info` | Print runtime/engine status |
| `_cmd_compile` | ODE → Verilog dispatcher |
| `_cmd_serve` | Streaming server launcher |
| `_cmd_benchmark` | pytest-benchmark delegate |
| `_cmd_preflight` | preflight.py delegate |
| `_cmd_deploy` | NIR/PyTorch → FPGA project |
| `_cmd_collect_synthesis` | Report files → optimiser evidence JSON |
| `_cmd_studio` | FastAPI Studio launcher |
| `_auto_synthesize` | Yosys + nextpnr + packing |
| `_generate_project` | Makefile or project.tcl emitter |
| `_format_engine_status` | Rust engine status line |
| `_safe_simd_tier` | Defensive SIMD-tier accessor |
| `_print_optional_dependency_version` | numpy/jax version line |
| `_TARGET_CONFIGS` | FPGA target metadata table |

---

## 4. Rust Engine Integration

The CLI does **not** call the Rust engine for compute — `cli.py` is dispatch
only. It does query the engine for status reporting in `_cmd_info`:

```python
import sc_neurocore_engine as engine
version = getattr(engine, "__version__", "unknown")
simd_tier = engine.simd_tier()  # "scalar" | "sse2" | "avx2" | "avx512" | …
```

If the engine wheel is missing, `_format_engine_status` returns
`"Rust engine: not available"` rather than raising. This makes the CLI usable
in pure-Python environments (no maturin build, no Rust toolchain).

If the engine wheel reports a `__version__` different from the Python
package's `__version__`, the status line includes the mismatch — useful when
debugging mixed-installation issues (e.g. `pip install -e .` against an older
wheel still on the `sys.path`).

There is no Rust path for the CLI itself; the dispatcher is intentionally
pure Python so it imports in <300 ms with no Rust toolchain present.

---

## 5. FPGA Targets

Two backend tools are supported:

- **Yosys + nextpnr** (open-source) — `ice40`, `ecp5`. Auto-synthesised by
  `--synthesize` and wrapped in a `Makefile` for repeat builds.
  Bitstream packing uses `icepack` (ice40) or `ecppack` (ecp5).
- **Vivado** (proprietary) — `artix7`, `zynq`. Not auto-run; `project.tcl` is
  emitted for the user to invoke `vivado -mode batch -source project.tcl`.

The `_auto_synthesize` helper is a best-effort path: it returns `False`
silently if `yosys` is not on `$PATH`, and it never raises on
synthesis/PnR failure — instead it prints the last 5 lines of `stderr`. This
is a deliberate degradation strategy: the user gets the Verilog output even
when the toolchain is broken or absent.

---

## 6. Examples

All examples below were executed on this workstation (Linux, Python 3.12.3,
sc-neurocore 3.14.0) before being committed. Each block is reproducible.

### 6.1 Print version & engine status

```bash
sc-neurocore info
# sc-neurocore 3.14.0
# Python 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]
# Rust engine: not available
# NumPy: 2.2.6
```

### 6.2 Compile a custom LIF to Verilog

```bash
sc-neurocore compile \
    "dv/dt = -(v-E_L)/tau_m + I/C" \
    --threshold "v > -50" \
    --reset "v = -65" \
    --params "E_L=-65,tau_m=10,C=1" \
    --init "v=-65" \
    --dt 1.0 \
    --output build/lif \
    --module-name lif_demo \
    --testbench
```

Generates:

- `build/lif/lif_demo.v` — 45-line synthesisable RTL
- `build/lif/tb_lif_demo.v` — Icarus-runnable testbench

Then simulate:

```bash
iverilog -o sim build/lif/lif_demo.v build/lif/tb_lif_demo.v && vvp sim
```

### 6.3 Compile with synthesis (open-source toolchain)

```bash
sc-neurocore compile "dv/dt = -(v-E_L)/tau_m + I/C" \
    --threshold "v > -50" --reset "v = -65" \
    --params "E_L=-65,tau_m=10,C=1" --init "v=-65" --dt 1.0 \
    --target ice40 --synthesize
```

If `yosys` is on `$PATH` the synthesis runs in-process and prints cell/wire
counts plus the path to the JSON netlist. If `nextpnr-ice40` is also present,
place-and-route runs and emits `*.asc`; if `icepack` is present, the
bitstream `*.bin` is produced and its size logged.

### 6.4 Deploy a NIR graph to ICE40

```bash
sc-neurocore deploy model.nir --target ice40 --output build/deploy
cd build/deploy && make synth
```

### 6.5 Collect synthesis evidence

After an external FPGA tool has produced utilisation, timing, and power
reports:

```bash
sc-neurocore collect-synthesis \
    --design build/network_design.json \
    --utilisation build/vivado_utilisation.rpt \
    --power build/vivado_power.rpt \
    --timing build/vivado_timing.rpt \
    --accuracy-score 0.991 \
    --clock-mhz 100 \
    --inferences-per-run 1 \
    --out build/synthesis_observations.json
```

This produces evidence JSON with one `observations` record plus optional
workload-normalised energy fields. It does not run Vivado, Quartus, Yosys, or
nextpnr.

### 6.6 Launch the Studio

```bash
pip install "sc-neurocore[studio]"
sc-neurocore studio --port 8001
# SC-NeuroCore Studio starting at http://127.0.0.1:8001
```

---

## 7. Performance

### 7.1 Cold-start (this workstation, 2026-04-17)

Measured with `time.perf_counter()` around `from sc_neurocore.cli import main`,
five fresh interpreter starts, hot disk cache:

| Run | Import time | Max RSS |
|-----|-------------|---------|
| 1 | 195.2 ms | 28.4 MB |
| 2 | 298.6 ms | 28.3 MB |
| 3 | 182.7 ms | 28.3 MB |
| 4 | 215.7 ms | 28.5 MB |
| 5 | 202.9 ms | 28.5 MB |
| **Median** | **~203 ms** | **28.4 MB** |

Hardware: Intel i5-11600K, 32 GB DDR4, root ext4 SSD. Python 3.12.3
(system). Run from `/media/anulum/.../SC-NEUROCORE/` with `PYTHONPATH=src`.

The cold-start cost is dominated by `argparse` and the lazy bootstrap of
`sc_neurocore`'s top-level `__init__.py`. None of the `_cmd_*` helpers'
imports are paid until that command is dispatched.

### 7.2 Per-command latency

`_cmd_info` walks the optional `numpy` / `jax` / `sc_neurocore_engine`
imports — each adds 50–200 ms when present. `_cmd_compile` invokes the
equation compiler (single ODE: <30 ms). `_cmd_deploy` performs disk I/O for
the entire `hdl/` tree copy (~200 KB) plus optional Yosys (multi-second).

### 7.3 Rust path

**N/A** — the CLI is intentionally pure-Python dispatch. The Rust engine is
only queried for status. Compute hot paths are reached only via downstream
modules (`compiler`, `serve`, `nir_bridge`, etc.), which carry their own
Rust paths.

### 7.4 Benchmarks

**`sc-neurocore benchmark`** delegates to the project's pytest-benchmark
suite under `benchmarks/benchmark_suite.py`. The CLI itself has no
dedicated benchmark suite — cold-start is documented above instead.

---

## 8. Pipeline Wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| Console script | `pyproject.toml [project.scripts] sc-neurocore = "sc_neurocore.cli:main"` | `pip install` registers it |
| Package main | `python -m sc_neurocore.cli` works via the module entrypoint | Manual invocation |
| Compile path | `_cmd_compile` → `sc_neurocore.compiler.equation_compiler.equation_to_fpga` + `generate_testbench` | `tests/test_equation_compiler.py` |
| Deploy NIR path | `_cmd_deploy` → `sc_neurocore.nir_bridge.from_nir` | `tests/test_nir_bridge*.py` |
| Deploy PyTorch path | `_cmd_deploy` → `sc_neurocore.conversion.convert` | `tests/test_conversion*.py` |
| Serve path | `_cmd_serve` → `sc_neurocore.serve.SpikeServer` | `tests/test_serve_server.py` |
| Synthesis evidence path | `_cmd_collect_synthesis` → `sc_neurocore.optimizer.build_payload_from_reports` | `tests/test_optimizer/test_synthesis_evidence_cli.py` |
| Studio path | `_cmd_studio` → `sc_neurocore.studio.app.create_app` | `tests/test_cli.py::test_studio_*` |
| Status path | `_cmd_info` → `sc_neurocore_engine.simd_tier` | `tests/test_cli.py::test_info_*` |

Every dispatched command terminates in either a registered subprocess or a
public symbol of another sc-neurocore subpackage. There are no orphan
helpers.

---

## 9. Known Issues

### 9.1 `--dt 0.001` (was: silent dead Verilog, now: fail-fast)

**Discovered:** 2026-04-17 while writing this doc.
**Severity:** HIGH — silent correctness bug in v3.14.0.
**Status:** **fixed** by task #7. The compiler now raises `ValueError`
on Q8.8 dt underflow and the CLI default has been changed from
`--dt 0.001` to `--dt 1.0`.

**Original behaviour (v3.14.0):** `--dt 0.001` (1 ms) was encoded into
Q8.8 fixed-point as `0.001 * 256 = 0.256`, which truncated to `0`. The
generated Verilog multiplied the `dv` update by zero on every cycle, so
the membrane voltage never changed. The bug was silent — no warning, no
error.

**Current behaviour:**

- The CLI default is `--dt 1.0` (one timestep per Q8.8 LSB → `16'sd256`
  in the generated multiplier).
- Any `dt` that quantises to 0 in the chosen fixed-point format raises
  `ValueError` from `compile_to_verilog` with an actionable message:
  ```
  ValueError: dt=0.001 underflows in Q8.8: smallest representable
  non-zero value is 0.00390625 (neuron.dt * 2**8 = 0.256 → 0).
  Use dt >= 0.00390625 (e.g. dt=1.0 for 1-step intervals), or pass
  a wider fraction (e.g. Q4.12 via fraction=12) to the compiler.
  ```
- `dt=0.0` is still accepted (degenerate but legal — produces a
  non-advancing model, useful for certain test patterns).
- The `fraction` argument to `compile_to_verilog` lets callers widen the
  fixed-point format (e.g. `fraction=12` for Q4.12 accepts `dt=0.001`).

**Reproduce the new behaviour:**

```bash
# default dt=1.0 succeeds
sc-neurocore compile "dv/dt=-(v-E_L)/tau_m" \
    --threshold "v>-50" --reset "v=-65" \
    --params "tau_m=10,E_L=-65" --init "v=-65"
grep _dt_mul_v build/sc_equation_neuron.v
# wire signed [31:0] _dt_mul_v = (...) * 16'sd256;   ← non-zero

# explicit dt=0.001 raises with actionable message
sc-neurocore compile "dv/dt=-v/tau" \
    --threshold "v>-50" --reset "v=-65" \
    --params "tau=10" --init "v=-65" --dt 0.001
# ValueError: dt=0.001 underflows in Q8.8: ...
```

Regression tests: `tests/test_equation_compiler.py::TestDtUnderflowGuard`
(7 cases covering raise, message content, boundary at `1/256`,
`dt=0.0` legality, wider-fraction acceptance, CLI default success,
CLI explicit-dt raise).

### 9.2 `compile` / `deploy` / `serve` test coverage (closed: task #8)

`tests/test_cli.py` now covers all three commands:

- **deploy** (`TestDeployCommand`, 5 tests): missing-arg exit code,
  unsupported extension exit code, full PyTorch happy path (writes
  `sc_deploy_lif.sv` + `Makefile`), Vivado target emits `project.tcl`,
  end-to-end via `main()`. The 3 PyTorch-using tests skip cleanly
  when torch is not installed (CI has torch).
- **serve** (`TestServeCommand`, 4 tests): missing-arg exit code, non-`.nir`
  rejection, full happy path with mocked `nir.read` + `from_nir` +
  `SpikeServer`, dispatch via `main()`.
- **compile** end-to-end coverage was already in
  `tests/test_equation_compiler.py::TestCompileCLI`; the new
  `TestDtUnderflowGuard` adds 2 more CLI cases (default-dt success,
  explicit-dt-0.001 raise).

### 9.3 `collect-synthesis` does not run vendor tools

`collect-synthesis` intentionally parses reports that already exist. It does
not invoke Vivado, Quartus, Yosys, nextpnr, or board programming utilities. Use
`deploy` to scaffold the FPGA project, run the external implementation flow,
then pass the generated reports into `collect-synthesis`.

---

## 10. Tests & Coverage

`tests/test_cli.py` and the focused optimiser CLI tests cover the public
dispatcher surface:

```text
test_version_flag                              PASS
test_info_command                              PASS
test_no_command_prints_help                    PASS
test_info_without_rust_engine                  PASS
test_info_reports_engine_version_mismatch      PASS
test_info_ignores_broken_optional_jax_import   PASS
test_info_ignores_broken_optional_numpy_import PASS
test_format_engine_status_without_simd_tier    PASS
test_format_engine_status_with_broken_simd_tier PASS
test_benchmark_delegates_to_subprocess         PASS
test_preflight_delegates_to_subprocess         PASS
test_studio_launches_uvicorn                   PASS
test_studio_missing_fastapi                    PASS
test_studio_command_via_main                   PASS
test_collect_synthesis_command_writes_optimizer_evidence PASS
test_collect_synthesis_command_reports_missing_required_args PASS
```

Run locally:

```bash
PYTHONPATH=src python3 -m pytest \
    tests/test_cli.py \
    tests/test_optimizer/test_synthesis_evidence_cli.py -q
```

Multi-angle dimensions covered:

- happy paths (info, version)
- absence of optional deps (no numpy / no jax / no Rust engine / no FastAPI)
- broken optional deps (`__version__`-less modules, `simd_tier` raising,
  `simd_tier` not callable)
- subprocess delegation (`benchmark`, `preflight`)
- end-to-end command dispatch via `main()` (`studio`)
- report evidence success and missing-required-argument errors

---

## 11. Audit Status (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | Console script registered; every `_cmd_*` reaches a downstream public symbol |
| 2 | Multi-angle tests | ✅ PASS | CLI tests cover info, benchmark, preflight, studio, deploy, serve, compile dispatch cases, and synthesis-evidence collection. |
| 3 | Rust path | N/A | Dispatch-only; engine is queried for status only |
| 4 | Benchmarks | N/A | CLI cold-start measured (Section 7); no pytest-benchmark suite for the dispatcher itself |
| 5 | Performance docs | ✅ PASS | Section 7 (this page) with measured numbers |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX header present; no undocumented type-check suppressions remain in `cli.py`. |

Net status: **0 WARN, 0 FAIL.** Tasks #7 and #8 are both closed, and
`collect-synthesis` is wired, documented, and tested.

---

## 12. References

- argparse — Python standard library, [docs.python.org/3/library/argparse.html](https://docs.python.org/3/library/argparse.html)
- Yosys Open SYnthesis Suite — [yosyshq.net/yosys](https://yosyshq.net/yosys/)
- nextpnr — [github.com/YosysHQ/nextpnr](https://github.com/YosysHQ/nextpnr)
- Project IceStorm (icepack) — [clifford.at/icestorm](http://www.clifford.at/icestorm/)
- Project Trellis (ecppack) — [github.com/YosysHQ/prjtrellis](https://github.com/YosysHQ/prjtrellis)
- NIR (Neuromorphic Intermediate Representation) — [neuroir.org](https://neuroir.org/)
- AMD Vivado — [www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado.html](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado.html)

Internal:

- Equation compiler: [`api/compiler.md`](compiler.md)
- NIR bridge: [`api/nir_bridge.md`](nir_bridge.md)
- ANN-to-SNN conversion: [`api/conversion.md`](conversion.md)
- Streaming server: [`api/serve.md`](serve.md)
- Studio: [`guides/studio.md`](../guides/studio.md)

---

## 13. Auto-rendered API

::: sc_neurocore.cli
    options:
      show_root_heading: true
      show_source: true
      members:
        - main
