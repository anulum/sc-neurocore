<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Compiler surface policy -->

# Compiler Surface Policy

This page defines the root-level `sc_neurocore.compiler` API boundary. The
package facade exports the stable symbols used by ordinary compiler clients.
Some modules remain direct public modules because existing workflows import
them by module path. The remaining modules are internal build tools used by the
compiler facade, deployment facade, or hardware evidence pipeline.

Use `from sc_neurocore.compiler import ...` for facade symbols. Import direct
public modules only when their module path is shown below. Internal build tools
can change without a deprecation window.

## Root Module Decisions

| Module | Status | Decision |
| --- | --- | --- |
| `_sby_runner` | internal build tool | Shared SymbiYosys task runner behind the equivalence and property proof runners. |
| `_verilog_folded_datapath` | internal build tool | Combinational processing-element emitter reached through the stable Verilog compiler facade. |
| `_verilog_integrators` | internal build tool | Fixed-point lowering for Euler, Gauss-Seidel, RK4, exponential-Euler, and map recurrences. |
| `_verilog_neuron_core` | internal build tool | Shared next-state, event, stochastic-threshold, and reset fragment builder for both RTL forms. |
| `_verilog_registered_module` | internal build tool | Clocked state-owning module emitter reached through the stable Verilog compiler facade. |
| `adaptive_precision` | public facade | Package-level exports provide the adaptive precision planning API. |
| `auto_tune` | internal build tool | Internal heuristic helper used by precision planning workflows. |
| `block_floating` | direct public module | Shared-exponent format types are imported directly by quantization workflows. |
| `block_floating_quantization` | direct public module | Block-floating dense compilation helpers are direct quantization surfaces. |
| `c_expr_emitter` | internal build tool | C/C++ (`ap_fixed`) expression emitter behind the HLS exporter. |
| `c_fixed_emitter` | internal build tool | Bit-exact integer C/Rust expression emitter behind the bit-true kernel. |
| `certification_gen` | internal build tool | Deployment evidence generator reached through `deployment`. |
| `cocotb_gen` | internal build tool | Testbench generator reached through `deployment`. |
| `compiler_impl` | internal build tool | Legacy implementation module behind documented compiler entry points. |
| `constraint_gen` | direct public module | Timing constraint generator has direct tests and documentation. |
| `deployment` | direct public module | Deployment facade aggregates constraints, drivers, evidence, and multi-target helpers. |
| `equation_compiler` | public facade | Package-level exports provide equation-to-FPGA helpers. |
| `equivalence_check` | internal build tool | SymbiYosys equivalence runner behind the hardware evidence pipeline. |
| `equivalence_miter` | internal build tool | Sequential-equivalence miter builder feeding the equivalence runner. |
| `expr_lut_tables` | internal build tool | Shared transcendental look-up tables for the expression emitters. |
| `fixed_point_quantization` | direct public module | Fixed-point quantization primitives back direct quantization workflows. |
| `formal_evidence` | internal build tool | Evidence writer used by hardware audit pipelines. |
| `formal_property_check` | internal build tool | SymbiYosys RTL-property runner behind the hardware evidence pipeline. |
| `fpga_wrapper` | internal build tool | Wrapper generator behind compiler and deployment paths. |
| `guard_bits` | internal build tool | Static-analysis primitive reached through `static_analysis`. |
| `host_driver_gen` | internal build tool | Host driver generator reached through `deployment`. |
| `ir_type_checker` | public facade | Package-level exports provide stochastic IR validation helpers. |
| `layer_precision` | internal build tool | Validated row model behind adaptive precision layer manifests. |
| `length_planner` | internal build tool | Validated layer-length planner behind adaptive precision manifests. |
| `live_control` | public facade | Package-level exports provide live MMIO control specs. |
| `live_control_ops` | internal build tool | Implementation module for live-control operations. |
| `live_control_specs` | internal build tool | Implementation module for live-control spec records. |
| `live_control_types` | internal build tool | Implementation module for live-control type aliases. |
| `manifest_gen` | internal build tool | Manifest helper used by validation and compiler evidence paths. |
| `mixed_dense_kernel` | internal build tool | Numeric kernel behind mixed dense compilation. |
| `mixed_dense_quantization` | direct public module | Mixed dense compiler helpers are direct quantization surfaces. |
| `mixed_precision` | compatibility facade | Backward-compatible facade for mixed-precision solver imports. |
| `mixed_precision_spec` | internal build tool | Data model behind the mixed-precision facade. |
| `mlir_emitter` | public facade | Package-level exports provide MLIR bundle emission. |
| `multi_target` | internal build tool | Multi-target comparison helper reached through `deployment`. |
| `operator_abstraction` | internal build tool | Lifts an internal result to a free input to abstract a multiplier for tractable unbounded proofs. |
| `overflow_proof` | internal build tool | Static-analysis proof primitive reached through `static_analysis`. |
| `pipeline` | public facade | Package-level exports provide the compiler pipeline. |
| `pipeline_analysis` | internal build tool | Static-analysis primitive reached through `static_analysis`. |
| `power_estimator` | direct public module | Power estimator has direct tests and documentation. |
| `precision_config` | direct public module | Precision configuration types are direct mixed-precision surfaces. |
| `precision_pairs` | internal build tool | Preset/configuration helper behind precision planning. |
| `precision_presets` | internal build tool | Preset registry behind mixed-precision facade. |
| `precision_solver` | direct public module | Constraint solver has direct tests and documented usage. |
| `proof_transforms` | public facade | Package-level exports provide the opt-in formal-proof transform registry. |
| `q_format` | direct public module | Q-format types are direct quantization surfaces. |
| `quantization_reports` | internal build tool | Report records are re-exported through quantization surfaces. |
| `quantize_core` | compatibility facade | Backward-compatible import surface for `quantizer`. |
| `quantizer` | public facade | Package-level exports provide canonical quantization helpers. |
| `resource_estimator` | internal build tool | Deployment estimator reached through `deployment`. |
| `riscv_driver` | direct public module | RISC-V driver generator has direct tests and documentation. |
| `sby_formal` | internal build tool | SymbiYosys script generator reached through `deployment`. |
| `sensitivity_analysis` | direct public module | Sensitivity analysis has direct tests and adaptive-precision usage. |
| `slr_placement` | internal build tool | Placement helper reached through `deployment`. |
| `static_analysis` | direct public module | Static-analysis facade aggregates guard, proof, pipeline, power, and SVA helpers. |
| `sva_gen` | direct public module | SVA generator has direct tests and documentation. |
| `synapse_planner` | internal build tool | Validated synapse planner behind adaptive precision manifests and Studio auto-tuning. |
| `synapse_precision` | internal build tool | Validated row model behind adaptive precision synapse manifests. |
| `testbench_gen` | direct public module | Verilog testbench generator has direct tests and documentation. |
| `validation` | direct public module | Adaptive-runtime validation is re-exported by a compatibility facade. |
| `verilog_compiler` | compatibility facade | Historical two-function import surface delegating to the registered and folded emitters. |
| `verilog_compiler_config` | internal build tool | Fixed-point geometry, range diagnostics, and backend policy record for Verilog emission. |
| `verilog_expr_emitter` | internal build tool | Fixed-point AST lowerer with deterministic product rounding, shared LUT geometry, and fail-closed stochastic rounding. |
| `whitebox_taps` | internal build tool | Instruments a module to expose internal state as taps for unbounded equivalence proofs. |

## Enforcement

`tests/test_compiler_surface_policy.py` reads this table and the filesystem on
each run. A new root compiler module must either join the package facade, become
a direct public or compatibility module, or appear here as an internal build
tool with a clear reason.

`operator_abstraction` and `whitebox_taps` are intentionally proof-only build
tools, not production compiler flags. Proof pipelines select them through
`sc_neurocore.compiler.proof_transforms` or the package-level
`apply_proof_transform` / `list_proof_transforms` facade. `quantize_core` remains
a compatibility facade for legacy quantizer imports and is pinned by
`tests/test_compiler_compat_facades.py`.
