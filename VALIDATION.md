<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Validation and coverage scope -->

# Validation

## Test Matrix

| Suite | Count | Scope |
|-------|------:|-------|
| Python unit/integration | — | `pytest tests/` over every tracked test module; the README's generated capability table gives the current test-file count |
| Rust engine | CI-listed | `cargo test --manifest-path engine/Cargo.toml` |
| Bridge (PyO3) | — | Maturin build + Python import smoke test |
| HDL formal verification | 92 proof jobs | SymbiYosys `.sby` jobs under `hdl/formal/` (catalogue model checks and legacy SC cores; the README's count is test-bound) |

CI runs tests on Python 3.11–3.14 (Ubuntu) and the Rust engine tests on Ubuntu, Windows and macOS.

## CI Validation Gates

All gates must pass before merge.

| Gate | Workflow | What it enforces |
|------|----------|------------------|
| ruff format | `ci.yml` | `ruff format --check` |
| ruff | `ci.yml` | `ruff check` (import hygiene, code quality) |
| bandit | `ci.yml` | Security static analysis (SAST) |
| test + coverage | `ci.yml` | `pytest --cov-fail-under=100` on Python 3.11, 3.12, 3.13, 3.14 |
| installed NIR interoperability | `ci.yml` | Separate hash-pinned snnTorch and Sinabs profiles, real conversions, nonempty JUnit results and zero supported-profile skips |
| spdx-guard | `ci.yml` | SPDX license headers on all `.py`, `.rs`, `.v` files |
| build | `ci.yml` | `python -m build` + smoke import |
| rust-lint | `v3-engine.yml` | `cargo fmt --check` + `cargo clippy -D warnings` |
| rust-test | `v3-engine.yml` | `cargo test` on Ubuntu, Windows and macOS |
| bridge-build | `v3-engine.yml` | Maturin build + v3 integration tests |
| wheels | `v3-wheels.yml` | Cross-platform wheel builds (Linux, macOS, Windows) |
| installed engine ABI | `v3-wheels.yml` | Capture facade/extension identities from site-packages outside the checkout and retain the measured inventory |
| default ABI reference | `v3-wheels.yml` | Compare every interface field with the qualified Linux x86-64 / CPython 3.12 / NumPy 2.2.3 default-feature reference |
| ABI proof tool coverage | `ci.yml` | 100% statement and branch coverage of inventory/comparison tools, including actual CLI subprocesses |
| pre-commit | `pre-commit.yml` | Trailing whitespace, YAML/TOML, typos, ruff format |
| codeql | `codeql.yml` | GitHub CodeQL security analysis |
| scorecard | `scorecard.yml` | OpenSSF Scorecard supply-chain audit |
| docs | `docs.yml` | MkDocs build verification |

## Studio optional engine typing

The E-I and model simulator getter annotations match the native bridge callable
contract in both dependency profiles. Focused HTTP tests exercise model and
network requests; a separate isolated interpreter without the engine SDK verifies
health, real NumPy E-I simulation and adaptive-threshold model simulation.
The annotation change preserves simulation bodies and numerical behavior.

## Studio input identity

Model-run contracts check exact Python and NumPy constructor integers,
including seeds above 2^53, and reject fractional, boolean and non-finite
inputs. HTTP experiment checks exercise both explicit and legacy parameter
seeds against direct model steps, cache receipts and replay packs.
The installed-wheel browser gate checks Poisson controls and full-domain
seeds through both catalogue and equation workspaces, saved sessions and
a separate replay interpreter. See the [installed browser gate](CONTRIBUTING.md#installed-studio-browser-acceptance).

## Coverage Policy

The installed snnTorch job measures `execution_metadata.py`, `graph_identity.py`
and `snntorch_bridge.py` with 100% statement and branch coverage. It uses an
independent coverage configuration, so the package-wide NIR omission below
does not exclude these modules from that required gate. The separate Sinabs
profile exercises its compatible legacy NIR reader and exporter.

The desktop configuration checks also exercise an actual reader-only FIFO in
a separate interpreter and retain regular-file symlink compatibility. Profile
coverage includes that child through the configured subprocess and immediate-exit
patches; these checks do not qualify the full desktop installation.

`tests/test_studio_desktop_paths.py` and
`tests/test_studio_desktop_configuration.py` exercise the desktop profile
contract on real filesystems: XDG defaults and overrides, private new
directories, legacy project preservation, symlink cycles, bounded schema
refusal and atomic configuration replacement. These focused checks cover
the profile modules; desktop window, installer and lifecycle qualification
remain separate.

Engine interface capture and comparison have dedicated tests under
`tests/test_tools/test_engine_abi_inventory*.py` and
`tests/test_tools/test_engine_abi_contracts.py`. These exercise the real CLI,
live global pickle identities, editable-installation refusal and damaged
inventory rejection. [Engine bridge contracts](docs/guides/engine_bridge_contracts.md)
describe the evidence scope and the additional per-binding behavioral proof.

`tests/test_mckean_source_engine_binding.py` exercises the canonical installed
McKean class and complete native batch: numerical traces, array dtype/layout,
input preservation, atomic failure, configured reset, instance-pickle refusal
and the public Rust dispatcher. `tests/test_mckean_engine_binding.py` covers the
distinct retained SC triangular profile. These contracts keep both model
identities explicit; they do not substitute for the other binding owners.

- Threshold: 100% (enforced by `pytest --cov-fail-under=100`) of the measured
  files. The scope below is generated from `pyproject.toml
  [tool.coverage.run] omit` and checked on every commit; the reason for each
  pattern stands next to it in that file.

<!-- coverage-scope:begin (generated by tools/coverage_scope.py; do not edit) -->

The coverage threshold applies to 1,317 of 1,889 Python files under
`src/sc_neurocore` (243,826 of 381,470 source lines, 63.9 %). The remaining
572 files (137,644 lines) are omitted by the patterns below and are not part of
the percentage. An omitted file may still have tests; the gate does not measure them.

| Omission pattern | Files | Source lines |
| --- | ---: | ---: |
| `*/quantum/hardware_bridge.py` | 1 | 135 |
| `*/quantum_cognition/bridge_adapter.py` | 1 | 551 |
| `*/integrations/lava_bridge.py` | 1 | 164 |
| `*/training/*` | 14 | 4,117 |
| `*/conversion/*` | 31 | 4,782 |
| `*/qat/torch_qat.py` | 1 | 367 |
| `*/qat/lsq.py` | 1 | 298 |
| `*/qat/pact.py` | 1 | 123 |
| `*/qat/observers.py` | 1 | 349 |
| `*/adapters/sonata.py` | 1 | 375 |
| `*/nir_bridge/*` | 32 | 10,003 |
| `*/accel/jax_backend.py` | 1 | 663 |
| `*/layers/jax_dense_layer.py` | 1 | 240 |
| `*/accel/julia/neurons/__init__.py` | 1 | 643 |
| `*/accel/julia/solvers/__init__.py` | 1 | 65 |
| `*/accel/julia/scpn/__init__.py` | 1 | 115 |
| `*/accel/julia/mixed_dense/__init__.py` | 1 | 94 |
| `*/accel/julia/adc_to_spike/__init__.py` | 1 | 103 |
| `*/accel/go/rk4_neurons/__init__.py` | 1 | 163 |
| `*/accel/go/dcls_tent/__init__.py` | 1 | 134 |
| `*/accel/go/mixed_dense/__init__.py` | 1 | 118 |
| `*/accel/go/adc_to_spike/__init__.py` | 1 | 124 |
| `*/accel/mojo/rk4_neurons/__init__.py` | 1 | 164 |
| `*/accel/mojo/dcls_tent/__init__.py` | 1 | 137 |
| `*/accel/mojo/mixed_dense/__init__.py` | 1 | 119 |
| `*/accel/mojo/adc_to_spike/__init__.py` | 1 | 125 |
| `*/neurons/models/adaptive_threshold_if.py` | 1 | 213 |
| `*/neurons/models/adaptive_threshold_moe.py` | 1 | 140 |
| `*/neurons/models/adex.py` | 1 | 708 |
| `*/neurons/models/aihara_map_neuron.py` | 1 | 224 |
| `*/neurons/models/alpha_motor_neuron.py` | 1 | 209 |
| `*/neurons/models/amari_field.py` | 1 | 176 |
| `*/neurons/models/astrocyte.py` | 1 | 108 |
| `*/neurons/models/atype_k_neuron.py` | 1 | 165 |
| `*/neurons/models/balanced_resonate_and_fire.py` | 1 | 153 |
| `*/neurons/models/benda_herz.py` | 1 | 103 |
| `*/neurons/models/bertram_phantom.py` | 1 | 214 |
| `*/neurons/models/bk_neuron.py` | 1 | 178 |
| `*/neurons/models/booth_rinzel.py` | 1 | 200 |
| `*/neurons/models/brainscales_adex.py` | 1 | 84 |
| `*/neurons/models/brunel_wang.py` | 1 | 293 |
| `*/neurons/models/butera_respiratory.py` | 1 | 218 |
| `*/neurons/models/cazelles_map.py` | 1 | 370 |
| `*/neurons/models/cerebellar_basket_neuron.py` | 1 | 168 |
| `*/neurons/models/chandelier_neuron.py` | 1 | 155 |
| `*/neurons/models/chay.py` | 1 | 160 |
| `*/neurons/models/chay_keizer.py` | 1 | 233 |
| `*/neurons/models/chay_keizer_minimal.py` | 1 | 189 |
| `*/neurons/models/chialvo_map.py` | 1 | 512 |
| `*/neurons/models/clif.py` | 1 | 102 |
| `*/neurons/models/coba_lif.py` | 1 | 430 |
| `*/neurons/models/cochlear_hair_cell.py` | 1 | 145 |
| `*/neurons/models/compte_wm.py` | 1 | 374 |
| `*/neurons/models/connor_stevens.py` | 1 | 469 |
| `*/neurons/models/courage_nekorkin_map.py` | 1 | 387 |
| `*/neurons/models/dcn_neuron.py` | 1 | 253 |
| `*/neurons/models/de_schutter_purkinje.py` | 1 | 362 |
| `*/neurons/models/dendritic_nmda.py` | 1 | 208 |
| `*/neurons/models/direction_selective_rgc.py` | 1 | 140 |
| `*/neurons/models/dpi_neuron.py` | 1 | 464 |
| `*/neurons/models/durstewitz_dopamine.py` | 1 | 320 |
| `*/neurons/models/ermentrout_kopell_map_neuron.py` | 1 | 360 |
| `*/neurons/models/escape_rate.py` | 1 | 230 |
| `*/neurons/models/expif.py` | 1 | 648 |
| `*/neurons/models/fitzhugh_nagumo.py` | 1 | 429 |
| `*/neurons/models/fitzhugh_rinzel.py` | 1 | 482 |
| `*/neurons/models/gamma_motor_neuron.py` | 1 | 101 |
| `*/neurons/models/gamma_renewal.py` | 1 | 89 |
| `*/neurons/models/glif.py` | 1 | 585 |
| `*/neurons/models/golgi_cell.py` | 1 | 401 |
| `*/neurons/models/golomb_fs.py` | 1 | 289 |
| `*/neurons/models/granule_cell.py` | 1 | 269 |
| `*/neurons/models/gutkin_ermentrout.py` | 1 | 196 |
| `*/neurons/models/hay_l5.py` | 1 | 437 |
| `*/neurons/models/hill_tononi.py` | 1 | 287 |
| `*/neurons/models/hindmarsh_rose.py` | 1 | 476 |
| `*/neurons/models/hodgkin_huxley.py` | 1 | 397 |
| `*/neurons/models/ibarz_tanaka_map.py` | 1 | 427 |
| `*/neurons/models/izhikevich2007.py` | 1 | 464 |
| `*/neurons/models/jansen_rit.py` | 1 | 242 |
| `*/neurons/models/lapicque.py` | 1 | 428 |
| `*/neurons/models/larter_breakspear.py` | 1 | 186 |
| `*/neurons/models/lugaro_cell.py` | 1 | 103 |
| `*/neurons/models/marder_stg.py` | 1 | 323 |
| `*/neurons/models/martinotti_neuron.py` | 1 | 370 |
| `*/neurons/models/mat.py` | 1 | 166 |
| `*/neurons/models/mcculloch_pitts.py` | 1 | 221 |
| `*/neurons/models/mckean.py` | 1 | 103 |
| `*/neurons/models/medvedev_map.py` | 1 | 615 |
| `*/neurons/models/mihalas_niebur.py` | 1 | 440 |
| `*/neurons/models/morris_lecar.py` | 1 | 301 |
| `*/neurons/models/motor_unit.py` | 1 | 308 |
| `*/neurons/models/multicompartment_mcn.py` | 1 | 390 |
| `*/neurons/models/neurogrid.py` | 1 | 202 |
| `*/neurons/models/nlif.py` | 1 | 134 |
| `*/neurons/models/non_resetting_lif.py` | 1 | 137 |
| `*/neurons/models/perfect_integrator.py` | 1 | 298 |
| `*/neurons/models/pernarowski.py` | 1 | 491 |
| `*/neurons/models/pinsky_rinzel.py` | 1 | 310 |
| `*/neurons/models/pospischil.py` | 1 | 341 |
| `*/neurons/models/pv_fast_spiking_neuron.py` | 1 | 327 |
| `*/neurons/models/quadratic_if.py` | 1 | 286 |
| `*/neurons/models/rall_cable.py` | 1 | 141 |
| `*/neurons/models/renshaw_cell.py` | 1 | 265 |
| `*/neurons/models/rulkov_map.py` | 1 | 425 |
| `*/neurons/models/sc_chaotic_map_neuron.py` | 1 | 112 |
| `*/neurons/models/sc_clipped_logistic_bursting_map.py` | 1 | 486 |
| `*/neurons/models/sc_clipped_rational_recovery_map.py` | 1 | 347 |
| `*/neurons/models/sc_decoupled_adaptation_ion_mass.py` | 1 | 112 |
| `*/neurons/models/sc_four_state_glif.py` | 1 | 508 |
| `*/neurons/models/sc_non_resetting_adaptive_lif.py` | 1 | 98 |
| `*/neurons/models/sc_resetting_wilson_hr.py` | 1 | 456 |
| `*/neurons/models/sc_scaled_reset_adaptive_if.py` | 1 | 377 |
| `*/neurons/models/sc_sigma_delta_accumulator.py` | 1 | 55 |
| `*/neurons/models/sc_six_state_thalamocortical.py` | 1 | 404 |
| `*/neurons/models/sc_stochastic_rate_adaptation.py` | 1 | 116 |
| `*/neurons/models/sc_triangular_mckean.py` | 1 | 424 |
| `*/neurons/models/sc_upward_crossing_rulkov_map.py` | 1 | 385 |
| `*/neurons/models/sc_wb_nmda_magnesium_block.py` | 1 | 153 |
| `*/neurons/models/sfa.py` | 1 | 156 |
| `*/neurons/models/sherman_rinzel_keizer.py` | 1 | 134 |
| `*/neurons/models/siegert.py` | 1 | 105 |
| `*/neurons/models/sigma_delta.py` | 1 | 91 |
| `*/neurons/models/sigmoid_rate.py` | 1 | 183 |
| `*/neurons/models/spinnaker_lif.py` | 1 | 116 |
| `*/neurons/models/srm0.py` | 1 | 177 |
| `*/neurons/models/sst_neuron.py` | 1 | 386 |
| `*/neurons/models/stellate_cell.py` | 1 | 199 |
| `*/neurons/models/terman_wang.py` | 1 | 454 |
| `*/neurons/models/theta.py` | 1 | 234 |
| `*/neurons/models/threshold_linear_rate.py` | 1 | 155 |
| `*/neurons/models/unipolar_brush_cell.py` | 1 | 118 |
| `*/neurons/models/upper_motor_neuron.py` | 1 | 185 |
| `*/neurons/models/vip_neuron.py` | 1 | 312 |
| `*/neurons/models/wang_buzsaki.py` | 1 | 131 |
| `*/neurons/models/wendling.py` | 1 | 172 |
| `*/neurons/models/wilson_cowan.py` | 1 | 245 |
| `*/neurons/models/wilson_hr.py` | 1 | 436 |
| `*/neurons/models/wong_wang.py` | 1 | 294 |
| `*/neurons/models/yamada.py` | 1 | 159 |
| `*/network/mpi_runner.py` | 1 | 293 |
| `*/neurons/__main__.py` | 1 | 16 |
| `*/neurons/dsl_cli.py` | 1 | 383 |
| `*/quantum_cognition/__main__.py` | 1 | 463 |
| `*/experiments/*` | 27 | 3,084 |
| `*/quantum_cognition/fs_watcher.py` | 1 | 264 |
| `*/viz/plots.py` | 1 | 285 |
| `*/studio/*` | 320 | 76,043 |
| `*/federation/*` | 5 | 1,120 |

<!-- coverage-scope:end -->

- Excluded lines: `pragma: no cover`, `if __name__`, `raise NotImplementedError`,
  conditional imports (`HAS_MPI`, `HAS_CUPY`, `HAS_NUMBA`)

## Installed codec catalogue

The wheel test job runs the catalogue consumer from outside the checkout in
isolated Python mode. It checks the base distribution by itself and alongside
the matching engine wheel. The receipt binds each imported public codec and
the compiled extension to bytes in the installed wheel. It exercises all 18
catalogue variants through 28 workloads and requires exact packet equality
and cross-decoding for each available Python and Rust pair.

The same consumer accepts a kernel-only engine with `--profile legacy` for
compatibility checks. This profile exercises the remaining Python or mixed
paths and checks that an explicit Rust request fails when the full container
is unavailable. See the [reproduction commands](benchmarks/README.md#installed-codec-catalogue).

The wheel test job also launches the installed `sc-neurocore codec` console
script and exercises the file commands against the base and native wheels.
The tests cross-read encoded variants, reject invalid metadata and truncated
inputs, and check that a failed decode preserves the previous output.

## Recorded spike-event stream

The installed Python and Rust codecs must produce identical packets and
cross-decode events, sample offsets and band-power values on the original
mono challenge recordings and simultaneous multichannel EEG. The API tests
cover partial bins, original ADC and layout refusal, malformed packet state
and restart. Resetting both endpoints starts a separate stream at sample zero.

The [benchmark reproduction steps](benchmarks/README.md#recorded-spike-event-and-band-power-streams)
use the same packet sizes and recorded samples for both backends.

## Kuramoto Coupling Correctness

The UPDE solver implements `dθ_n/dt = ω_n + Σ_m K_nm sin(θ_m − θ_n)` with
phase-difference coupling. Tests verify:

- Two identical oscillators with K > 0 converge to phase lock
- N oscillators reach order parameter R → 1 for strong coupling
- Coupling term is zero when all phases are equal

## Holonomic Layer Adapters (L1–L16)

Each adapter in `src/sc_neurocore/adapters/holonomic/` has a corresponding
test file. Adapters implement the `HolonomicAdapter` protocol with:

- `adapt()` — transform state through the layer
- `inverse()` — reverse transform (where applicable)
- Round-trip property: `inverse(adapt(x)) ≈ x` within tolerance

## HDL Verification

Verilog modules in `hdl/` are verified via:

- Formal assertions (SystemVerilog `assert property`)
- Testbenches in `tb/` with golden-vector comparison
- Co-simulation parity checks against Python reference


## Typed scalar native IR

Local Linux qualification exercises the default-feature Rust graph API, normal
release-wheel Python calls, persisted text, direct RTL and pinned CIRCT 1.152.0.
The focused native regression cohort has 69 passing tests. The installed graph
and tool cohort has 113 passing tests, and the retained CPU codec cohort has 67.
The numerical CIRCT corpus contains 94 circuits and 9,095 exact observations.
Direct RTL and CIRCT are each compared with independent encoded-integer references.
Signed/unsigned widths through 128 bits, division by zero, signed minimum,
oversized shifts, rounding ties, casts and saturation bounds are included.
Unused malformed declarations, bounded vector nesting and parser ID exhaustion
are refused before hardware dimensions or wrapped identifiers can be emitted.
Six seeded arithmetic/type/admission faults are caught by public numerical or refusal
gates. Source-leaf instrumentation is separate from default-crate execution;
LLVM branch counts of zero are not branch coverage evidence.

The public API and dedicated tests are documented in [the native IR contract](docs/api/sc-ir.md).
These results cover scalar graph circuits. Equation/state/integrator lowering,
full compiler acceptance and physical board measurements have separate evidence.

## Function-call numeric contracts

The [per-call contract](docs/guides/function_call_domains.md) reports conservative
encoded argument hulls and separate numerical measurements. Tests execute twenty
expressions over every signed input pair at six and eight bits through generated
C with undefined-behaviour checks and Rust with overflow checks. Critical
32-bit Q3.29 operands exercise wide products and narrowing. Clocked RTL checks
banker rounding over 4,096 input pairs, including exact singleton bounds.

Mathematical-domain exclusions and empty domains are explicit. Numerical tests
compare real unit outputs with the model's functions or independent Decimal power
references and reject corrupted bounds. The three numeric-analysis modules have
complete measured statement and branch coverage. The source-bound benchmark
separates a fresh interpreter's first collector invocation from cached calls,
records runtime dependencies and loaded-host context, and checks complete
contract equality outside timing.

These checks establish bounded regression evidence. Conservative hulls do not
prove state reachability, operand correlations or model-level error propagation;
analysis timings do not establish hardware throughput or board behaviour.
