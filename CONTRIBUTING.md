# Contributing to SC-NeuroCore

© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li

Thank you for your interest in SC-NeuroCore. Contributions are welcome under the following guidelines.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork and create a feature branch:
   ```bash
   git clone https://github.com/<your-user>/sc-neurocore.git
   cd sc-neurocore
   git checkout -b feature/your-feature
   ```
3. **Install dev dependencies and git hooks:**
   ```bash
   pip install -e ".[dev]"
   make install-hooks
   ```
4. **Build** the Rust engine (optional, for engine work):
   ```bash
   cd engine && maturin develop --release && cd ..
   ```
5. **Run the preflight gate** to verify your setup:
   ```bash
   make preflight
   ```

## Preflight Gate

Every push is guarded by `tools/preflight.py`, which runs the same checks as CI:

| Gate | What it checks |
|------|----------------|
| **ruff format** | Python formatting (`src/` and `tests/`) |
| **ruff check** | Code quality and import hygiene |
| **bandit** | Security static analysis |
| **spdx-guard** | SPDX license headers on all source files |
| **pytest** | 3 376+ tests with 100% coverage gate |

```bash
make preflight          # full gate (lint + tests)
make preflight-fast     # lint-only (~5s)
```

The `.githooks/pre-push` hook runs `preflight-fast` automatically before every `git push`. Install it with `make install-hooks`.

## Development Guidelines

### Code Style

- **Rust**: `cargo fmt` before committing.
- **Python**: `ruff format` + `ruff check` (both enforced in CI and preflight). Annotate new and changed code fully and run strict mypy.
- **SPDX header**: Every `.py`, `.rs`, `.v` file must start with `# SPDX-License-Identifier: AGPL-3.0-or-later`.

### Language-native API documentation

Every new or materially changed public API must be documented in its native
documentation system in the same change. This forward ratchet applies across
the project; existing gaps remain explicit backfill work and are not silently
declared compliant.

- **Python:** module/class/function docstrings, including parameters, returns,
  failure behaviour, state mutation, and scientific or project provenance.
- **Rust:** `//!` module docs and `///` on public items; new public modules
  enable `#![warn(missing_docs)]`, and `cargo doc --no-deps` must pass.
- **Go:** GoDoc comments beginning with the exported identifier; `go test` and
  `go vet` must pass for the touched package.
- **Julia:** docstrings on exported modules, types, and functions, accessible
  through Julia's help/documentation system.
- **Mojo:** module and exported ABI comments recording parameter, ownership,
  error, and mutation contracts until a stable generated-doc gate is available.
- **RTL:** module, parameter, port, fixed-point-format, latency, reset, and
  event-semantics comments beside the maintained Verilog/SystemVerilog source.

Narrative Markdown complements these native surfaces but does not replace
them. Compatibility aliases must point to the canonical documented identity
and must not create duplicate scientific provenance.

### Testing

- All new Rust code must include tests in `engine/tests/` or inline `#[cfg(test)]` modules.
- All new Python APIs must have pytest coverage in `tests/`.
- SIMD paths must include portable fallback — never assume AVX-512 or AVX2 availability.

Python coverage includes isolated worker processes through the configured
`subprocess` patch and source directory, including scripts that run without
importing the package. The `_exit` patch saves watchdog observations before an
immediate process exit. Each process writes separate data; pytest-cov combines
it into the current batch, and `--cov-append` retains earlier batches. Keep both
patches and the parallel setting when changing the coverage runner.

Hosted CI partitions the complete test-module list into eight shards for each
Python interpreter, then runs each shard in memory-reclaiming batches. Every
module runs once per interpreter; interpreter-independent formal proofs retain
their single coverage run. Focused coverage and backend gates are distributed
among the Python 3.12 shards. The required `test (3.11)` through `test (3.14)`
checks require all test shards to succeed. Python 3.12 additionally combines all
eight indexed coverage artifacts and native Windows refusal data before applying
the unchanged 100% gate. A failed, absent or misidentified shard cannot pass.

Inspect a shard without running tests or replacing existing coverage data:

```bash
bash tools/run_full_cov.sh --shard=1/8 --list-files | tr '\0' '\n'
```

`--list-files` emits NUL-delimited repository-relative paths. Shard assignment
does not depend on `--batch-size`; choose batch size for process memory limits.

CI builds the release engine wheel once per Python ABI and shares that run's
artifact with its test shards. Ready-to-install wheels use exact source,
dependency, toolchain and platform cache keys. Local installer commands still
build the engine unless `SC_NEUROCORE_CI_ENGINE_WHEEL_DIR` names the downloaded
directory; a missing, duplicate or ABI-incompatible wheel fails admission.

One shared timing snapshot drives every shard in a run. With previous JUnit
durations, the runner assigns the longest modules to the least-loaded shard.
New modules receive the median recorded duration and always join discovery.
Without history, deterministic round-robin assignment still runs every module.
The next run retains actual measured durations, including completed tests in
failed batches. Timing data controls order and placement, never test membership.

```bash
bash tools/run_full_cov.sh --shard=1/8 --timings=pytest-durations.json --list-files
python tools/pytest_shards.py record --reports=test-results --output=pytest-durations.json
```

`SCN_CI_SHARD_COUNT`, the matrix indices and displayed shard count must agree;
the workflow contract tests enforce this. Increase capacity based on successful
end-to-end measurements, including queueing and setup. More queued shards can
cost more time than they save, and one indivisible long test sets a lower bound.

Installed-wheel CI runs `tests/test_engine_global_pickle.py` from an external
consumer directory. It round-trips every public native class/function global
through every supported pickle protocol and verifies object identity. Binding
tests separately exercise instance persistence or refusal, inputs and mutation.

### Frontend toolchain

Use the Node version pinned in the workflows and npm 12.2.1, as declared by
`studio/frontend/package.json`. When the installed npm version differs, run
`npx --yes npm@12.2.1 ci` from `studio/frontend`; the same prefix can run the
package scripts without changing the shared machine's global npm installation.
The producer and federation test host use matching React singleton versions.

### Hardware catalogue coverage

The hardware catalogue uses three separate metadata modules: current synthesis
profile definitions, their hardware/NIR projection, and the public HTTP route.
Their focused gate retains branch and subprocess tracing and requires 100%:

```bash
python -m pytest tests/test_studio_hardware_catalogue.py \
  tests/test_studio_synthesis_target_profiles.py \
  tests/test_studio_hardware_catalogue_routes.py -q \
  --cov --cov-config=tests/studio_hardware_catalogue.coveragerc \
  --cov-report=term-missing --cov-fail-under=100
```

The frontend CI job runs the catalogue/modal DOM and shared HTTP transport tests with
per-file statement, branch, function and line thresholds of 100%. The real
backend browser contract is `e2e/hardware-catalogue-live.spec.ts`, selected by
`playwright.live.config.ts` in the native Studio lane. It checks registration
conservation, target selection and refusal/retry without synthesising or
programming hardware. These checks attest the software catalogue, not a vendor
SDK or physical device. Shared-host local checks remain module-specific and
serial; the commands above describe the hosted gate.

### Candidate dimensional admission coverage

Candidate dimensional admission has a separate branch-coverage gate over its
four static unit modules. It exercises the actual candidate validation,
simulation and review API as well as expression and unit algebra:

```bash
python -m pytest tests/test_studio_candidate_units.py \
  tests/test_studio_candidate_unit_expressions.py \
  tests/test_studio_candidate_unit_functions.py \
  tests/test_studio_candidate_unit_values.py \
  tests/test_equation_units_runtime.py \
  tests/test_studio_candidate_api_refusals.py -q \
  --cov --cov-config=tests/studio_candidate_units.coveragerc \
  --cov-report=term-missing --cov-fail-under=100
```

The real browser contract in `e2e/candidate-authoring-live.spec.ts` retains a
valid AdEx candidate through authoring, simulation, review, export and workspace
reopen, and checks located refusals for a parseable voltage-as-time declaration.
These gates check dimensional admission and unchanged numeric execution, not
independent scientific validation or catalogue promotion.

### Candidate process jobs and storage custody

Catalogue-to-candidate preparation is enrolled in the same exact Python and
frontend coverage gates. `tests/test_studio_candidate_catalogue.py` reads every
bound canonical schema through the actual HTTP route, checks declaration and
source conservation, then runs a completed proposal through durable simulation
and review. The native frontend runner also covers proposal preview, explicit
replacement and delayed/disconnected HTTP response ownership. The real browser
case in `e2e/candidate-authoring-live.spec.ts` continues through validation,
diff, jobs, downloads and workspace reopen; it also runs in installed-wheel
acceptance. These checks establish proposal custody, not independent scientific
acceptance or catalogue promotion.

The native candidate frontend cohort uses the actual FastAPI server and
supervised Python workers. From `studio/frontend`, run:

```bash
npx vitest run --config vitest.candidates.config.ts
```

It retains per-file statement, branch, function and line thresholds of 100%
for the request client, record parser, observation hook, history and panel.
The tests forward real backend responses through controlled loopback sockets
to check delayed replies and disconnections without replacing fetch or the
scientific runtime. Set `SC_NEUROCORE_STUDIO_CANDIDATE_PYTHON` to select an
existing Studio interpreter and `SC_NEUROCORE_STUDIO_CANDIDATE_COVERAGE_DIR`
for reports. The real Chromium contracts are
`e2e/candidate-authoring-live.spec.ts` and `e2e/candidate-jobs-live.spec.ts`.

The Python 3.12 candidate gate combines branch and subprocess observations
from the candidate workflow and the owning storage protocol/client tests,
then applies `tests/studio_candidate_jobs.coveragerc` at 100%. Preserve both
the candidate and storage modules in that report; admission, actor custody,
cancellation and artifact reads cross the isolated authority boundary.
Use a short, test-owned `TMPDIR` for its Unix socket names. Hosted CI uses
`RUNNER_TEMP`; shared-machine checks remain serial, module-specific and bounded.

`tests/test_studio_distribution.py` also compares real installed-wheel jobs
with source results for Euler, declared-order Gauss–Seidel and seeded
diffusion through both direct wheel and sdist-to-wheel builds. Its missing
installed worker control must fail rather than import the checkout; a corrupt
installed parent schema must fail the review job without leaking its contents. These
checks establish software result and artifact custody, not independent
scientific acceptance or OS isolation.

### Equation compiler frontend coverage

From `studio/frontend`, `npm run test:compiler-coverage` checks the compiler
controls, store ownership and project/session custody with per-file statement,
branch, function and line thresholds of 100%. Its Inspector integration starts
the real FastAPI server and native compiler workers; the Python environment
must provide the Studio runtime and the matching native engine. Set
`SC_NEUROCORE_STUDIO_COMPILER_PYTHON` to select that interpreter and
`SC_NEUROCORE_STUDIO_COMPILER_API_ORIGIN` to select a test-owned IPv4 loopback
port (default `http://127.0.0.1:24022`). CI runs it in the primary native lane.
The run keeps API logs and jobs below `TMPDIR`; set
`SC_NEUROCORE_STUDIO_COMPILER_COVERAGE_DIR` for its coverage reports.
The DOM test observes download payloads; the real browser cases separately
verify file delivery and evidence manifest digests.

### Installed Studio browser acceptance

The publish and release workflows install the built wheel into a fresh
virtual environment, check its CLI, UI assets and catalogue, then run
`studio/frontend/playwright.installed.config.ts` against the installed server.
The browser checks Poisson constructor inputs and replay, and an exact 63-bit
seed through both a catalogue model and the equation playground, including
project saves, local sessions, downloads and a separate replay
interpreter. An editable source import is refused before the browser starts.
The same acceptance also compiles authored equations through the installed
native engine, checks source replay and RTL, and downloads compiler evidence.
Candidate authoring and process-job cases also run through the installed UI:
they retain original request bytes after draft changes and reloads, export
the review packet, and cancel a long reference job through its supervisor.
Both wheels come from the same checkout: publishing consumes the matching
Linux engine artifact; release validation builds it with the locked Cargo graph.

This configuration requires `SC_NEUROCORE_STUDIO_INSTALLED_PYTHON` and
`SC_NEUROCORE_STUDIO_INSTALLED_TEST_ROOT` as absolute paths. The selected
interpreter also runs the downloaded pack's independent replay.
The test root holds jobs, audit output, browser evidence and an isolated project
store. CI runs with a disposable user filesystem; shared-machine checks retain
the same explicit project-store namespace.
An existing browser may be selected explicitly with
`SC_NEUROCORE_STUDIO_BROWSER_EXECUTABLE`; CI uses Playwright's locked Chromium.
These checks do not attest federation-host adoption or hardware execution.

### Commit Messages

Follow conventional commit format:
```
feat(scope): short description
fix(scope): short description
docs(scope): short description
```

Examples:
```
feat(bitstream): add rotate_left method for HDC permutation
fix(simd): correct AVX2 popcount for non-aligned buffers
docs(readme): update benchmark table for v3.8
```

## What to Contribute

**High-value contributions:**
- New SIMD kernels (ARM SVE, RISC-V Vector)
- Additional HDC similarity metrics
- Jupyter notebook tutorials
- Performance benchmarks on new hardware
- Bug reports with reproducible test cases

**Please discuss first** (open an issue) before:
- Changing the public Python API
- Modifying the IR op set
- Adding new crate dependencies

## Submitting a Pull Request

1. Run `make preflight` — all gates must pass
2. Add a changelog entry if the change is user-visible
3. Ensure SPDX headers are present on new files
4. Open a PR against `main` with a clear description
5. Reference any related issues

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make test` | Python tests with coverage |
| `make test-rust` | Rust engine tests |
| `make test-all` | Both Python and Rust |
| `make lint` | ruff format + ruff check |
| `make fmt` | Auto-format Python + Rust |
| `make bandit` | Security static analysis |
| `make sast` | Alias for bandit |
| `make preflight` | Full CI-equivalent gate |
| `make preflight-fast` | Lint-only (~5s) |
| `make install-hooks` | Install git pre-push hook |
| `make bench` | Python benchmarks |
| `make bench-rust` | Rust Criterion benchmarks |
| `make bridge` | Build Rust bridge via maturin |
| `make docs` | Live docs preview |
| `make docs-build` | Build docs (strict mode) |
| `make build` | Build sdist + wheel |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker image interactively |
| `make clean` | Remove build artifacts |

## Licence

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](LICENSE). For commercial licensing enquiries, contact protoscience@anulum.li.
