<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Zynq UltraScale+ target contract -->
# Zynq UltraScale+ target contract

SC-NeuroCore now has an explicit Zynq UltraScale+ MPSoC target contract for the
SystemVerilog emitter and Vivado project generator. The contract is deliberately
fail-closed: it emits target metadata, estimates conservative resources, and
rejects unsupported SKU names instead of fabricating board-level timing or pin
claims.

## Supported SKU baseline

The public target baseline covers the currently planned ZU3EG and ZU9EG
surfaces. Device resources are recorded as compiler budgets, not as timing
closure evidence. Timing closure still requires Vivado on a board-specific
self-hosted runner with verified XDC pin assignments.

| SKU | Part | DSP primitive | LUT budget | FF budget | DSP budget | 36K BRAM budget | URAM budget |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ZU3EG | `xczu3eg-sbva484-1-e` | `DSP48E2` | 70,560 | 141,120 | 360 | 216 | 0 |
| ZU9EG | `xczu9eg-ffvb1156-2-e` | `DSP48E2` | 274,080 | 548,160 | 2,520 | 912 | 0 |

The primitive is `DSP48E2` for Zynq UltraScale+ MPSoC. SC-NeuroCore must not
claim a newer-family DSP primitive for this target.

## Compiler behaviour

The Rust IR target surface is `SvTarget::zynq_ultrascale_plus(SkuKind, clock_mhz)`.
When this target is selected, `emit_systemverilog_with_target` returns both the
generated SystemVerilog and a `ResourceReport`.

The target-aware emitter currently:

- adds a target header comment containing the SKU, part, clock, and DSP primitive;
- annotates arithmetic kernels with `use_dsp=yes` and `sc_target_dsp=DSP48E2`;
- adds RAM-style metadata for constant vectors so larger tables can be steered toward block memory when appropriate;
- reports conservative LUT, DSP, BRAM, URAM, and critical-path estimates;
- reports boolean fit decisions against the selected SKU budgets.

The legacy `emit(graph)` API remains target-neutral and emits the generic
SystemVerilog contract.

## Vivado project generator

`tools/gen_vivado_project.py` converts a JSON manifest into a deterministic
Vivado batch Tcl project. A manifest must define:

- `top`: top-level module name;
- `sku`: `zu3eg` or `zu9eg`;
- `clock_mhz`: positive integer target clock;
- `sources`: one or more existing SystemVerilog source files;
- `xdc`: one or more existing constraint files;
- `output_dir`: optional project output directory.

The checked-in XDC files under `hdl/targets/ultrascale_plus/` intentionally
contain only clock/timing constraints. They do not contain `PACKAGE_PIN` or
`LOC` placement constraints because no board-revision pin manifest has been
verified in this repository. Adding fabricated pins would create false hardware
evidence.

## Resource finding from the 2026-06-04 benchmark

The isolated 2026-06-04 comparison benchmark measured the target contract across
Python/Vivado-Tcl and Rust surfaces.

| Surface | Artefact | Median | Isolation evidence | Key contract result |
| --- | --- | ---: | --- | --- |
| Python + Vivado Tcl | `benchmarks/results/local_python_2026-06-04_ultrascale_plus_target.json` | 122.678 us/manifest | `cgroup_effective_cpuset=10-11`, `runtime_cpuset_shield_claimed=true` | deterministic ZU3EG/ZU9EG Tcl generation with `DSP48E2` baseline |
| Rust | `benchmarks/results/local_rust_2026-06-04_ultrascale_plus_target.json` | 130.836 us/emit | `cpu_affinity=10-11`, `cgroup_effective_cpuset=10-11`, `runtime_cpuset_shield_claimed=true` | 64x32 dense graph estimates 2,048 DSPs, exceeding the ZU3EG budget of 360 |

The 64x32 dense result is not a failure of the benchmark. It is the intended
fail-closed hardware conclusion: a one-DSP-per-MAC implementation of that graph
cannot be honestly claimed to fit ZU3EG. ZU3EG deployment for that workload
requires a folded or time-multiplexed dense implementation, a smaller layer, or
a larger target.

## Dense folding contract

The follow-up dense-folding contract provides that missing resource-safe path.
`SvTarget::dense_fold_plan(64, 32)` and `tools/ultrascale_dense_folding.py`
both compute the same ZU3EG plan:

| Field | Value |
| --- | ---: |
| Unfurled MACs | 2,048 |
| ZU3EG DSP budget | 360 |
| Output rows per cycle | 5 |
| Input lanes per output row | 64 |
| DSPs per compute cycle | 320 |
| Output fold factor | 7 |
| Input fold factor | 1 |
| Compute cycles | 7 |

The SystemVerilog emitter now annotates over-budget UltraScale+ dense
instances with this fold plan so generated RTL carries the resource remedy next
to the unfurled dense instance. The standalone
`hdl/sc_dense_folded_q88_core.v` implements the folded Q8.8-weight/Q16.16-MAC
execution contract and is covered by Icarus simulation. The benchmark also runs
bounded Yosys elaboration on an 8x8 parameterisation, reporting 240 generic
cells. That Yosys number validates HDL elaboration only; it is not a ZU3EG
Vivado timing or utilisation report.

The folded core is deterministic fixed-point dense logic. It does not silently
replace the existing stochastic dense layer in the generic emitter path.
Deployment code must select it deliberately when the target resource contract
requires folding.

## Vivado gate

`tests/test_ultrascale_plus_flow.py` includes an opt-in Vivado CI gate. It runs
only when `MIF_VIVADO_CI=1` and `vivado` is available. Until that self-hosted
runner and a verified board-specific pin map exist, SC-NeuroCore claims target
contract evidence and synthesis-flow readiness, not board-level timing closure.

Reference: AMD publishes the Zynq UltraScale+ MPSoC device-family overview and
product tables in DS891 at
<https://docs.amd.com/go/en-US/ds891-zynq-ultrascale-plus-overview>.
