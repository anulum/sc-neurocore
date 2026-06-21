import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { SynthesisTargetProvenanceMatrix } from "../api/client";
import { ProvenanceMatrixSummary } from "./SynthesisDashboard";
import SynthesisEvidenceControls from "./SynthesisEvidenceControls";

const provenanceMatrix: SynthesisTargetProvenanceMatrix = {
  matrix_sha256: "a".repeat(64),
  schema_version: "studio.synthesis-target-provenance-matrix.v1",
  targets: {
    ice40: {
      capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
      device: "up5k",
      evidence_classification: "synthesis",
      pnr_ready: false,
      pnr_tool: "nextpnr-ice40",
      schema_version: "studio.synthesis-target-provenance.v1",
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
      capacity: { brams: 41, dsps: 0, ffs: 20736, luts: 20736 },
      device: null,
      evidence_classification: "synthesis",
      pnr_ready: true,
      pnr_tool: null,
      schema_version: "studio.synthesis-target-provenance.v1",
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
    expect(html).toContain("aaaaaaaaaaaa");
    expect(html).toContain("GOWIN");
    expect(html).toContain("ICE40");
    expect(html).toContain("up5k");
    expect(html).toContain("ready - yosys ready Yosys 0.test");
    expect(html).toContain("missing - nextpnr-ice40 missing");
    expect(html).toContain("ready - not required");
    expect(html).toContain("synthesis");
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
