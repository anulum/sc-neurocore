<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SNN Studio user manual -->

# Hardware, compilation and target catalogue

## Start with the claim you need

| Evidence | Establishes for its recorded scope | Does not establish by itself |
|---|---|---|
| Float simulation | The numerical run returned these states/spikes | Fixed-point equivalence or hardware execution |
| Precision comparison | Differences between float and a supported fixed-point kernel | RTL execution or physical timing |
| Compile output and traceability | Which inputs produced particular source bytes | Correct translation of every equation or synthesis success |
| RTL / bit-true C co-simulation | Agreement for the compared signals and bounded stimuli | Scientific model fidelity, unbounded proof or board behaviour |
| Yosys synthesis | A netlist and its counted resources | Placement, routing, timing closure or power |
| Place and route | The selected device/tool/constraint result | Programming a board or measuring latency/power |
| Physical measurement | What the specified device/operator/instrumentation observed | Other boards, tool versions, stimuli or deployments |

Keep model, parameters, initial state, method, timestep, Q-format, generated
source digest, tool versions and target together. Changing any prerequisite
can withdraw downstream qualification.

## Compile a catalogue model

1. Select **Models** and a model with a canonical schema and compiler capability.
2. Read the supported integrators and signed Q-formats offered for that model.
3. Set parameters, timestep, method and format deliberately.
4. Choose **Generate RTL** and inspect the traceability strip and emitted source.
5. If **Run RTL co-sim** is available, run it and inspect the actual parity result.
6. For supported targets, continue with **Synthesise + Route** only after the
   current RTL and current parity report meet the panel's prerequisites.
7. Export the compile/parity/synthesis artifacts for that exact configuration.

Models without canonical schemas are refused rather than matched by name to
another equation. Euler/map parity support is narrower than compile-only
integrators; read the model's capability message. The parity report compares
RTL with the generated bit-true C kernel, not directly with a biological source.
See [Model readiness](../model-readiness.md) and
[Numeric contract](../numeric-contract.md).

## Custom equations and the compiler inspector

ODE mode exposes a direct Python equation compiler and a separate Rust IR
inspector. They are distinct paths with distinct supported operations.

**Current limitation:** the Studio Rust **Build IR** adapter constructs a fixed
LIF graph; it does not translate arbitrary submitted equations, threshold and
reset into equivalent IR. A structurally valid graph or green verifier badge
therefore does not establish fidelity to the equations you typed. Do not use
this path as evidence of general equation-to-hardware correctness.

The direct **Generate RTL** / Python compiler path carries source-to-RTL
traceability and its own supported expression/numeric contract. Inspect that
output and qualify its behaviour independently. Direct output is not repaired
or certified by a successful fixed-template IR inspection. Keep failed emission
or tool diagnostics with the result; generated text is not synthesis evidence.
[Compiler Inspector](../compiler-inspector.md) describes both routes.

## Read fixed-point settings

A signed Q-format sets width, fractional precision and representable range.
For example Q8.8 here uses 16 signed bits with 8 fractional bits: resolution
`1/256`, range `-128` through `127.99609375`. Values outside that range and
operations near its limits depend on the declared overflow and rounding policy.

Use precision comparisons with the actual format and inspect error traces,
spike differences, resets and stress conditions. A lower numerical error on
one trace does not establish all-input equivalence. See the
[numeric contract](../numeric-contract.md) for supported operations and stress
coverage; formats/integrators are not interchangeable just because they emit RTL.

## FPGA target profiles

This table is a catalogue of the **synthesis profiles exposed by Studio**.
It is not a list of qualified physical boards or a guarantee that every model
can be deployed on them.

| Target key | Capacity device used by the profile | Synthesis | Routing offered here | Evidence boundary |
|---|---|---|---|---|
| `ice40` | iCE40 UP5K | Yosys `synth_ice40` | nextpnr-ice40 | Tool/profile result; inspect fit, package, I/O and timing |
| `ecp5` | ECP5 LFE5U-25F | Yosys `synth_ecp5` | nextpnr-ecp5 | Tool/profile result; inspect fit, package, I/O and timing |
| `gowin` | Gowin GW2A-18 | Yosys `synth_gowin` | Not offered by this profile | Synthesis counts against the named capacity device |
| `xilinx` | Artix-7 XC7A35T | Yosys `synth_xilinx` | Not offered by this profile | Synthesis counts against the named capacity device |

Read the [Synthesis Dashboard](../synthesis-dashboard.md#supported-fpga-targets)
for resource capacities, primitive counting and provenance. The active
deployment's tool status decides whether an operation can actually run.

Use these terms when evaluating an additional hardware target:

- **Supported for an operation:** the named model/numeric/toolchain/version
  combination has current evidence for that operation. State the operation.
- **Experimental:** a profile or adapter exists, but the required target
  qualification is incomplete. Preserve its reported limitations.
- **Planned/unavailable:** no executable supported path is established in this
  deployment. Do not infer one from a vendor name or framework integration.

Studio does not currently provide a complete physically qualified device
catalogue. A driver, framework bridge or four synthesis choices are insufficient
to claim supported Loihi, SpiNNaker, GPU or FPGA board execution in this UI.
Require model/Q-format/toolchain/version and device evidence for that claim.

## Synthesise and inspect

1. Confirm that the source belongs to your current experiment.
2. Open **FPGA synthesis**, inspect tool availability and choose a target.
3. Use **Estimate** for a heuristic when the IR path is appropriate. Label it
   as an estimate; it is not measured resource use or hardware power.
4. Use **Synthesise** for the admitted equation source, or the catalogue model's
   qualified **Synthesise + Route** path.
5. Read success, timeout, capacity and uncounted-cell information. A netlist
   that exceeds capacity does not fit the device.
6. Preserve the report, exact source and job artifacts in an evidence bundle.

**All Targets** compares synthesis profiles. It does not run one model on four
boards. Resource fit by count is necessary but does not ensure placement or
routing. Missing tools, unsupported targets, I/O limits and real synthesis
failures should remain visible, not be replaced by the heuristic estimate.

## Network pipeline

The canvas **Pipeline** validates the graph, simulates it, lowers supported
networks, co-simulates and synthesises. Its supported hardware subset is narrower
than graph simulation, including supported LIF/perfect-integrator populations
and constant drives under the lowering contract. Unsupported models, connectivity
or numeric configurations are refused with reasons; there is no licence to
replace the graph by a convenient template.

Read the step where the pipeline stopped. A failed simulation, unsupported
lowering, failed parity, overflow or capacity result prevents a successful
end-to-end claim. Keep the full graph and pipeline report. The
[integration reference](../integration.md#full-pipeline) describes the sequence.

Physical programming, calibrated power/latency measurements and a device
operator remain separate from this workflow. Record device revision, bitstream,
clock, stimulus, instrumentation and conditions when those measurements exist.

Next: [Troubleshooting](troubleshooting.md).
