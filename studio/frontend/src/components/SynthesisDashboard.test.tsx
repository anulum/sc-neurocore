// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { SiliconTerminalResult, SynthesisTargetProvenanceMatrix } from "../api/client";
import { ProvenanceMatrixSummary, SiliconTerminalSummary } from "./SynthesisDashboard";
import SynthesisEvidenceControls from "./SynthesisEvidenceControls";

const provenanceMatrix: SynthesisTargetProvenanceMatrix = {
  evidence_classification: "synthesis",
  matrix_sha256: "a".repeat(64),
  provenance_grade: "unverified",
  schema_version: "studio.synthesis-target-provenance-matrix.v1",
  status: "completed",
  targets: {
    ice40: {
      capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
      device: "up5k",
      evidence_classification: "synthesis",
      pnr_ready: false,
      pnr_tool: "nextpnr-ice40",
      provenance_grade: "unverified",
      schema_version: "studio.synthesis-target-provenance.v1",
      status: "completed",
      synthesis_command: "synth_ice40",
      synthesis_ready: true,
      target: "ice40",
      tools: [
        {
          available: true,
          executable: "yosys",
          key: "yosys",
          role: "synthesis",
          version: "Yosys 0.test",
        },
        {
          available: false,
          executable: "nextpnr-ice40",
          key: "nextpnr_ice40",
          role: "place_and_route",
          version: null,
        },
      ],
    },
    gowin: {
      capacity: { brams: 46, dsps: 48, ffs: 15552, luts: 20736 },
      device: null,
      evidence_classification: "synthesis",
      pnr_ready: true,
      pnr_tool: null,
      provenance_grade: "tool_backed",
      schema_version: "studio.synthesis-target-provenance.v1",
      status: "completed",
      synthesis_command: "synth_gowin",
      synthesis_ready: true,
      target: "gowin",
      tools: [
        {
          available: true,
          executable: "yosys",
          key: "yosys",
          role: "synthesis",
          version: "Yosys 0.test",
        },
      ],
    },
  },
};

describe("ProvenanceMatrixSummary", () => {
  it("renders target readiness, tool provenance, evidence class, and digest", () => {
    const html = renderToStaticMarkup(
      <ProvenanceMatrixSummary matrix={provenanceMatrix} />,
    );

    expect(html).toContain("Target provenance matrix");
    expect(html).toContain("synthesis / completed / aaaaaaaaaaaa");
    expect(html).toContain("aaaaaaaaaaaa");
    expect(html).toContain("GOWIN");
    expect(html).toContain("ICE40");
    expect(html).toContain("up5k");
    expect(html).toContain("ready - yosys ready Yosys 0.test");
    expect(html).toContain("missing - nextpnr-ice40 missing");
    expect(html).toContain("ready - not required");
    expect(html).toContain("synthesis");
    expect(html).toContain("completed");
  });
});

describe("SynthesisEvidenceControls", () => {
  it("renders job, bundle artifact metadata, and download actions", () => {
    const html = renderToStaticMarkup(
      <SynthesisEvidenceControls
        bundle={{
          artifact_paths: ["evidence/jobs/sj_synth/artifacts/synthesis/multi-target-evidence.json"],
          artifacts: [
            {
              relative_path: "evidence/jobs/sj_synth/artifacts/synthesis/multi-target-evidence.json",
              sha256: "b".repeat(64),
              size_bytes: 512,
            },
          ],
          bundle_id: "seb_synthesis",
          job_id: "sj_bundle",
          manifest: {},
          schema_version: "studio.evidence-bundle.v1",
          summary: {
            artifact_path_count: 1,
            entry_count: 1,
            entry_type_counts: { action_evidence: 1 },
            evidence_classification_counts: { synthesis: 1 },
            source_job_count: 1,
            source_job_kind_counts: { synthesis: 1 },
            source_job_owner_counts: { browser: 1 },
          },
        }}
        error={null}
        jobId="sj_synth"
        loading={false}
        onDownloadArtifact={() => undefined}
        onExport={() => undefined}
      />,
    );

    expect(html).toContain("Synthesis evidence");
    expect(html).toContain("job sj_synth");
    expect(html).toContain("bundle seb_synthesis");
    expect(html).toContain("evidence/jobs/sj_synth/artifacts/synthesis/multi-target-evidence.json");
    expect(html).toContain("512 B - sha bbbbbbbbbbbb");
    expect(html).toContain(
      "Download synthesis evidence artefact evidence/jobs/sj_synth/artifacts/synthesis/multi-target-evidence.json",
    );
  });
});

describe("SiliconTerminalSummary", () => {
  it("renders digest-bound route and timing evidence", () => {
    // `targets` is keyed by target name, so the lookup is `| undefined`. The
    // fixture above defines `ice40`; saying so here means a fixture that stops
    // defining it fails on that fact rather than on a downstream shape.
    const ice40 = provenanceMatrix.targets.ice40;
    if (ice40 === undefined) throw new Error("the provenance fixture must define ice40");
    const synthesis = {
      capacity: { brams: 56, dsps: 28, ffs: 24288, luts: 24288 },
      log_excerpt: "complete",
      resources: { brams: 0, cells: 20, dsps: 0, ffs: 8, luts: 12, wires: 30 },
      success: true,
      target: "ecp5",
      target_provenance: ice40,
      utilisation: { brams: 0, dsps: 0, ffs: 0.1, luts: 0.1 },
    };
    const terminal: SiliconTerminalResult = {
      artifacts: {
        netlist_sha256: "c".repeat(64),
        routed_design_sha256: "d".repeat(64),
      },
      evidence_classification: "synthesis",
      place_and_route: {
        critical_path: "clk to q",
        log_excerpt: "routed",
        max_freq_mhz: 37.08,
        success: true,
      },
      schema_version: "studio.silicon-terminal.v1",
      source_chain: {
        compile_input_sha256: "1".repeat(64),
        compile_traceability_sha256: "2".repeat(64),
        cosim_reference_trace_sha256: "3".repeat(64),
        cosim_rtl_trace_sha256: "3".repeat(64),
        model_name: "AdaptiveThresholdIFNeuron",
        module_name: "sc_adaptive_threshold_if_neuron",
        rtl_sha256: "b".repeat(64),
      },
      status: "completed",
      success: true,
      synthesis,
      target: "ecp5",
      target_provenance: synthesis.target_provenance,
    };

    const html = renderToStaticMarkup(<SiliconTerminalSummary terminal={terminal} />);

    expect(html).toContain("Selected RTL synthesis/PnR terminal");
    expect(html).toContain("AdaptiveThresholdIFNeuron");
    expect(html).toContain("sc_adaptive_threshold_if_neuron");
    expect(html).toContain("bbbbbbbbbbbb");
    expect(html).toContain("cccccccccccc");
    expect(html).toContain("dddddddddddd");
    expect(html).toContain("37.08 MHz");
  });
});

describe("ResourceBar and FitVerdict", () => {
  it("states the real share of an over-full device instead of capping it at 100 %", async () => {
    const { ResourceBar } = await import("./SynthesisDashboard");
    // The 20-neuron network measured at 6237 LUTs on the 5280-LUT UP5K.
    const html = renderToStaticMarkup(<ResourceBar label="LUTs" used={6237} total={5280} color="#4fc3f7" />);
    expect(html).toContain("6237 / 5280 (118.1%)");
    expect(html).toContain("over capacity");
    expect(html).not.toContain("(100.0%)");
  });

  it("says whether the named device holds the design, naming what it lacks", async () => {
    const { FitVerdict } = await import("./SynthesisDashboard");
    const over = renderToStaticMarkup(
      <FitVerdict verdict={{
        target: "ice40", fits_device: false, capacity_device: "iCE40 UP5K",
        exceeds_capacity: { luts: { needed: 6237, available: 5280 } },
      }} />,
    );
    expect(over).toContain("Does not fit the iCE40 UP5K: 6237 LUTs needed, 5280 available.");
    const fits = renderToStaticMarkup(
      <FitVerdict verdict={{ target: "xilinx", fits_device: true, capacity_device: "Artix-7 XC7A35T", exceeds_capacity: {} }} />,
    );
    // Synthesis for Xilinx is not bound to a device: the verdict names the one it judged against.
    expect(fits).toContain("Fits the Artix-7 XC7A35T by count; placement and routing decide the rest.");
    expect(renderToStaticMarkup(<FitVerdict verdict={{ target: "gowin" }} />)).toBe("");
  });

  it("leaves a design with cells of unknown cost unjudged", async () => {
    const { FitVerdict } = await import("./SynthesisDashboard");
    const html = renderToStaticMarkup(
      <FitVerdict verdict={{
        target: "gowin", fits_device: null, capacity_device: "Gowin GW2A-18",
        exceeds_capacity: {}, uncounted_cells: { RAM16SDP4: 64 },
      }} />,
    );
    expect(html).toContain("Not judged against the Gowin GW2A-18: 64 RAM16SDP4 cells have no counted cost, so these counts are a floor.");
    expect(html).not.toContain("Fits");
    const one = renderToStaticMarkup(
      <FitVerdict verdict={{ target: "ecp5", fits_device: null, uncounted_cells: { TRELLIS_DPR16X4: 1 } }} />,
    );
    expect(one).toContain("Not judged against the ECP5 device: 1 TRELLIS_DPR16X4 cell has no counted cost");
  });

  it("says an estimate is an estimate", async () => {
    const { FitVerdict } = await import("./SynthesisDashboard");
    const fits = renderToStaticMarkup(
      <FitVerdict estimate verdict={{ target: "ice40", fits_device: true, capacity_device: "iCE40 UP5K" }} />,
    );
    const over = renderToStaticMarkup(
      <FitVerdict estimate verdict={{
        target: "ice40", fits_device: false, capacity_device: "iCE40 UP5K",
        exceeds_capacity: { luts: { needed: 20012, available: 5280 } },
      }} />,
    );
    expect(fits).toContain("The estimate fits the iCE40 UP5K; synthesis counts the real design.");
    expect(over).toContain("The estimate does not fit the iCE40 UP5K: 20012 LUTs needed, 5280 available.");
    expect(over).not.toContain("Synthesis produced");
  });
});
