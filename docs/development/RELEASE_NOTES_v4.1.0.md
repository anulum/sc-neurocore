<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Release notes 4.1.0 -->

# SC-NeuroCore 4.1.0

These notes are the GitHub Release text for `v4.1.0`. The complete list of changes is the `[4.1.0]` section
of [CHANGELOG.md](https://github.com/anulum/sc-neurocore/blob/main/CHANGELOG.md).

## Highlights

- **Bit-exact integer SHD core.** `hdl/sc_shd_int_core.v` runs an integer spiking network for the Spiking
  Heidelberg Digits task bit for bit against its integer Python reference (`tools/shd_int_reference.py`). The
  network takes 4-bit counts and uses int8 weights, integer axonal delays from 0 to 30 steps and a LIF neuron with
  an arithmetic-shift leak. The core has an event mode and a full-scan mode with identical results. The trained
  integer network ships as `hdl/fixtures/shd_int_network.json`, together with its checkpoint and export
  provenance and the tool that reproduces it from the checkpoint.
- **Device resource reports and a real place-and-route check.** `ScGraph.emit_sv_with_report(target, clock_mhz)`
  emits SystemVerilog with a report for the target device. For the iCE40 UltraPlus UP5K it includes a static bound
  on routed core logic cells and critical path for one calibrated block (dense layer, LIF neuron or power-of-two
  synapse stream) inside its measured domain and operand form. The bound is fitted on recorded Yosys and nextpnr
  runs, and every recorded design is checked against it. Any other graph, including one with port operands,
  carries the reason instead of a number. `sc-neurocore resource-check` synthesises, places and routes a graph,
  and judges package I/O, logic capacity, timing and the estimate as pass, violation or undetermined.
- **From a catalogue model to synthesis in one command.** `sc-neurocore hardware-path MODEL -o DIR` checks the
  model's capability matrix and writes the RTL. It co-simulates the RTL cycle by cycle against the bit-true C
  kernel, writes a guarantee summary and synthesises the design; on iCE40 and ECP5 with nextpnr installed it also
  places and routes. Every stage refuses by name. The record states what the run does and does not establish.
- **Guarantee summaries and evidence packets.** `sc-neurocore compile-ir` writes, next to the emitted top level,
  for each operation whether its module's formal job proves this instance or only other parameters, or whether the
  module is only co-simulated, a stochastic estimate or inline logic without a separate proof.
  `sc-neurocore evidence-packet export` writes a hash-bound packet for one generated dense layer design: IR,
  SystemVerilog, a cycle-by-cycle co-simulation against a bit-true reference, formal and synthesis results with
  their gaps, and a SHA-256 manifest. `evidence-packet verify` refuses any altered, missing or extra member.
- **FPGA flows that place and route.** ECP5 designs now place and route in `compile --synthesize` and `deploy`.
  `compile --synthesize` synthesises the module it wrote, and hierarchical designs from `deploy` no longer fail in
  nextpnr.

## Supply chain

- Engine wheels are published to PyPI through trusted publishing, so every uploaded engine file carries a
  provenance attestation. Before the upload, each engine wheel is installed and imported on a runner of its own
  platform and Python.
- Every GitHub Release file and the release container image carry a signed SLSA build provenance attestation;
  `SECURITY.md` shows how to verify them with `gh attestation verify`.
- The container image that is pushed is the image the vulnerability scan passed, not a second build.
- The REUSE lint now blocks the security packet and the release sweep.
- The dependency locks are aligned with current upstream releases, with the recorded holds unchanged.

## SNN Studio: development preview

SNN Studio remains a **development preview**, labelled experimental. Changes in this release:

- Network-canvas simulations run as durable supervised process jobs: submission, history, cancellation and result
  downloads survive reconnects and API restarts. Graph job recovery is saved with workspaces and named sessions.
- CSV export and spike-triggered averaging are refused, with the server's reason, when full-resolution samples
  were omitted.
- Studio place and route names a design that needs more I/O pins or more logic than its device has.

## Behaviour changes

- **Spike codec limits.** ISI varint rasters with 16,777,216 or more time steps, and delta groups outside 1–256, are
  refused before encoding. Valid packets keep their bytes. Data that 4.0.0 encoded under those conditions
  (including auto-selected varint) may not decode to the original; re-encode it from its source.
- **Equation input limits.** An equation expression longer than 16,384 characters is refused before it is parsed,
  and a text that overflows Python's parser is refused as too deep to parse instead of failing with an internal
  error.
- **Synthesis exit status.** A Yosys failure in `compile --synthesize` is reported as a failure with exit status 1;
  before, any unsuccessful run was reported as a missing Yosys.
- **Zynq reports.** The resource reports for Zynq targets are labelled uncalibrated.
- **AXI-Lite hardware.** The dense-layer configuration registers are initialised on reset, completion is held until
  the next accepted start, and register updates, byte strobes and responses are correct under backpressure.

## Corrections

- The earlier SHD hardware figures described an out-of-context synthesis without weights or delays. They are not the
  cost of the network with its weights and are no longer quoted. The claims of an end-to-end silicon run and of
  fitting in under 2 % of the device are withdrawn.

## Known limits

- Linux engine wheels are tagged `manylinux_2_39`. Systems with an older glibc (for example Ubuntu 22.04,
  Debian 12 or RHEL 9) get no prebuilt engine wheel; build the engine locally with the Rust toolchain
  (`python -m maturin develop --release` in `bridge/`).
- The conda recipe in `conda/` is a draft. Its source digest is filled in only after the sdist is published.
- The iCE40 bound covers only the calibrated blocks inside their measured domain, measured with the recorded Yosys
  and nextpnr versions. It is a structural estimate, not a guarantee for other tool versions or designs.
- Benchmark timings are recorded on named hosts and dates and are not portable speed claims.

## Artifacts

The SHA-256 of every file attached to this release follows below, written by the release workflow.
