<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Cross-framework benchmark evidence
-->

# Cross-Framework Benchmark Evidence

This page is an evidence index for framework comparisons across Python, Rust, and external SNN frameworks. It separates committed measurements from planned comparisons so the public benchmark surface does not imply numbers that are not in the repository. Polyglot comparison claims must name the language/runtime, command, raw artefact, and environment.

## Evidence Rules

- A comparison is published only when a committed artefact under
  `benchmarks/results/` or `hdl/reports/` backs it.
- Wall-time claims must identify the framework, mode, scale, and local artefact.
- Spike-count, spike-timing, rate, or accuracy-parity claims must cite the
  artefact that stores the reference and SC-NeuroCore outputs.
- FPGA resource, timing, power, and energy claims are separate categories.
  Resource or timing reports are not power or energy evidence.
- Missing comparisons stay visible as gaps. They are not filled with estimates.

## Current Matrix

| Target | Committed artefact | Covered metrics | Status | Gap |
| --- | --- | --- | --- | --- |
| Brian2 parity | `benchmarks/results/brian2_parity_results.json` | single-LIF spike count/timing, population spike count/rate, wall time | Covered for parity microbench | Network API path failed in this artefact and remains excluded from parity claims |
| Brian2 translator suite | `benchmarks/results/snn_translator_20v.json` | Brunel-style variant wall time, spike counts, rate ratios | Covered for local translator variants | Not a replacement for Brian2 C++ standalone energy or hardware measurements |
| Brian2 head-to-head rerun | `benchmarks/results/upcloud_p4_rerun_20260310/brian2_headtohead.json` | cloud rerun wall-time comparison | Covered as archived cloud artefact | Use with adjacent `system_info.json` for environment context |
| snnTorch | `benchmarks/results/snntorch_vs_sc_microbench.json` | per-step wall time, spike totals, mean rates across four scales | Covered for CPU microbench | No committed snnTorch dataset-level accuracy parity or energy comparison |
| Norse | `benchmarks/results/cross_framework_1k.json` | 1k-neuron wall time, peak memory, spike totals, rates | Covered as CPU run | No committed Norse dataset-level accuracy parity or energy comparison |
| NEST | No committed artefact | None | Runner available, measurement gap | Run the opt-in NEST row and commit artefact before publishing NEST numbers |
| SpikingJelly | No committed artefact | None | Runner available, measurement gap | Run the opt-in SpikingJelly row and commit artefact before publishing SpikingJelly numbers |
| FPGA resource/timing | `hdl/reports/vivado_util_xc7z020_100mhz.rpt`; `hdl/reports/vivado_timing_xc7z020_100mhz.rpt`; `benchmarks/results/yosys_synth.json` | utilisation, timing, generic synthesis counts | Covered for resource/timing | These are not power or energy reports |
| FPGA power/energy | No committed measurement artefact | Parser available via `sc-neurocore collect-synthesis` | Capture path available, measurement gap | Commit real Vivado/Quartus power reports plus workload-normalised energy output before publishing energy numbers |

## Published Reference Points

The committed Brian2 parity artefact records:

- single LIF spike count: Brian2 `20`, SC-NeuroCore `20`;
- maximum single-LIF spike-time difference: `0.000 ms`;
- population mean rate: Brian2 `69.12 Hz`, SC-NeuroCore `70.48 Hz`;
- population wall time: Brian2 `0.507 s`, SC-NeuroCore `0.069 s`.

The committed snnTorch microbench artefact records the CPU comparison rows used
in the main benchmark page:

- single neuron, 1000 steps;
- dense 100→50, 500 steps;
- scale 500→500, 100 steps;
- scale 1000→1000, 50 steps.

The committed cross-framework 1k artefact records one CPU run covering
SC-NeuroCore NumPy, SC-NeuroCore Rust, Brian2 runtime mode, snnTorch, and Norse.
It stores wall time, peak memory, spike totals, and rates for the shared
1k-neuron setup.

## Required Next Measurements

1. Run the opt-in NEST and SpikingJelly rows and commit the resulting artefact:

   ```bash
   python benchmarks/cross_framework_benchmark.py \
       --scales 1000 \
       --skip-standalone \
       --include-gap-frameworks \
       --json benchmarks/results/cross_framework_1k_next.json
   ```

2. Capture checked optional dependency versions alongside future
   cross-framework artefacts. The harness now writes `dependency_versions` into
   JSON output.
3. Capture FPGA power reports from Vivado or Quartus for the same deployed
   network used in wall-time comparisons, then convert them to optimiser
   evidence:

   ```bash
   sc-neurocore collect-synthesis \
       --design build/network_design.json \
       --utilisation build/vivado_utilisation.rpt \
       --power build/vivado_power.rpt \
       --timing build/vivado_timing.rpt \
       --accuracy-score 0.991 \
       --clock-mhz 100 \
       --inferences-per-run 1 \
       --out benchmarks/results/fpga_power_observations.json
   ```

4. Commit only tool-generated reports and workload-normalised energy fields
   before publishing energy-per-inference claims.
5. Keep accuracy or spike-parity claims separate from resource and energy
   claims.
