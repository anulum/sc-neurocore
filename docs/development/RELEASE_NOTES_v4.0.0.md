<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Release notes 4.0.0 -->

# SC-NeuroCore 4.0.0

These notes are the GitHub Release text for `v4.0.0`. The complete list of changes is the `[4.0.0]` section
of [CHANGELOG.md](https://github.com/anulum/sc-neurocore/blob/main/CHANGELOG.md).

## Highlights

- **Equation compiler.** Supported equation models are lowered through a typed native IR to SystemVerilog,
  fixed-point C and Rust, and portable C++, and a saved circuit can be replayed against its source. The opt-in v2
  function set adds `rsqrt`, `hypot`, `round`, `heaviside` and general real powers. The default `lut-v1` lowering
  is unchanged.
- **Model catalogue.** <!-- count:python_model_classes -->183<!-- /count --> lazy-loaded Python model classes and
  <!-- count:rust_pyo3_model_wrappers -->212<!-- /count --> Rust PyO3 model wrappers. Agreement across the Python,
  Rust, Julia, Go and Mojo implementations is tracked per model; the README states how many models meet the strict
  five-runtime bar. That is agreement between implementations, not with the source paper, RTL or a device.
- **Readiness evidence.** Per-model facet receipts (co-simulation, formal, hardware) are bound to the exact sources
  they checked and go stale when those sources change.
- **Supply chain.** Hash-locked requirements and blocking audits of the committed locks for Python, Rust, npm, Go,
  Julia and Pixi.
- **Framework interchange.** NIR files now keep graph and node metadata, ordering and the timestep and reset
  convention through real round trips. New `from_snntorch()` and `to_snntorch()` exchange supported CPU float64
  inference networks with snnTorch and refuse what they cannot preserve, including stochastic-bitstream contracts.
- **Standalone function units.** `emit_function_unit_verilog()` emits one combinational fixed-point function as its
  own Verilog module, reusing the compiler's existing arithmetic.

## SNN Studio: development preview

SNN Studio ships in this release as a **development preview**, labelled experimental. It is not the Studio 1.0
release; Studio 1.0 will be released separately after its own acceptance.

Changes in this preview:

- **Breaking for Studio HTTP clients.** The candidate simulation and reference-review routes return `202` with a
  durable process-job receipt instead of the result. Fetch the result from the actor-owned status route; history and
  artifacts survive a reconnect or an API restart, and cancellation goes through the supervisor. Direct numerical
  Python calls return what they returned before.
- **Candidates from the catalogue.** A workspace candidate can be started from a catalogue model's canonical schema,
  previewed before it replaces an existing draft, and then validated, run, reviewed and exported through the
  existing candidate workflow. Canonical models and their evidence are unchanged.

## Breaking changes and migration

- **Python 3.11 is the minimum.** Python 3.10 is no longer tested, built or published for.
- **The `lava` optional extra is removed.** It installed nothing on the supported Pythons: `lava-nc` 0.10.0
  requires Python `<3.11`.
- **Equation evaluation.** `evaluate_scalar` and `evaluate_condition` take a program and its bindings. The
  evaluation-globals argument and `EVAL_GLOBALS` are removed.
- **Studio NIR envelope functions renamed.** `sc_neurocore.studio.network_graph.graph_to_nir` and `nir_to_graph`
  are now `graph_to_envelope` and `envelope_to_graph` in the same module. The Studio export route returns the
  envelope shape.
- **Moved public names.** Import them from their new modules:

  | Name | Old module | New module |
  |---|---|---|
  | `CONTROL_*`, `STATUS_*` and `TRAP_*` constants (16) | `sc_neurocore.hdl_gen.bus_interface` | `sc_neurocore.compiler.live_control_types` |
  | `predict_and_xor_world_model`, `xor_and_recover_world_model` | `sc_neurocore.spike_codec.predictive_codec` | `sc_neurocore.world_model.spike_predictor` |
  | `simulate_ei_network` | `sc_neurocore.studio.network_graph` | `sc_neurocore.studio.network` |
  | `classify_firing_pattern` | `sc_neurocore.studio.codegen` | `sc_neurocore.studio.firing_pattern` |
  | `ModelDescriptor`, `descriptor_completeness_tier` | `sc_neurocore.studio.models` | `sc_neurocore.neurons.model_descriptor` |
  | `load_descriptor` | `sc_neurocore.studio.models` | `sc_neurocore.neurons.model_catalogue` |
  | `dump_project_payload` | `sc_neurocore.studio.project` | `sc_neurocore.studio.project_manifest` |

- **Names that are no longer public.**
  - The unit helpers of `sc_neurocore.neurons.equation_builder` (`build_quantity_namespace`, `is_quantity`,
    `quantity_to_base`, `require_pint`, `require_quantity`, `validate_quantity_expression`) moved to the internal
    module `sc_neurocore.neurons._units`, and `UNIT_REGISTRY` is gone.
  - `DELAY_E_SIGMA` and `DELAY_I_SIGMA` of `sc_neurocore.network.cortical_column` moved to the internal module
    `sc_neurocore.network._cortical_column_parameters`.
  - `generate_model_script`, `generate_ode_script` (`sc_neurocore.studio.codegen`) and `build_ir_from_equation`,
    `emit_systemverilog`, `verify_ir` (`sc_neurocore.studio.compiler`) are removed from those modules.
- **Removed modules.** The generated Go and Julia mirrors `sc_neurocore.accel.go.services.{asic_flow, bioware,
  chiplet_gen, dna_mapper, equation_compiler, hierarchical_partitioner, quantum_annealing, safety_cert}` and
  `sc_neurocore.accel.julia.bioware` are removed. They returned constants or did not build; the maintained Python
  and native paths are unchanged.
- **Refusals.** `NotImplementedError` from a public entry point means that the operation exists but this release
  does not support the requested configuration. The message names the unsupported option and, where one exists,
  the supported alternative.

## Known limits

- Linux engine wheels are tagged `manylinux_2_39`. Systems with an older glibc (for example Ubuntu 22.04,
  Debian 12 or RHEL 9) get no prebuilt engine wheel; build the engine locally with the Rust toolchain
  (`python -m maturin develop --release` in `bridge/`).
- The conda recipe in `conda/` is a draft. Its source digest is filled in only after the sdist is published.
- MNIST 99.49 % is the committed 2026-03-26 run, not re-measured for 4.0.0.
- Benchmark timings are recorded on named hosts and dates and are not portable speed claims.

## Artifacts

The SHA-256 of every file attached to this release follows below, written by the release workflow.
