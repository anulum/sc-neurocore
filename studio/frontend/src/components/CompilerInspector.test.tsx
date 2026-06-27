import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { CompileTraceability } from "../api/client";
import CompilerInspector from "./CompilerInspector";

const compileTraceability: CompileTraceability = {
  evidence_classification: "compile",
  input_sha256: "1".repeat(64),
  output: {
    language: "systemverilog",
    module_name: "lif_neuron",
    rtl_chars: 128,
    rtl_sha256: "2".repeat(64),
  },
  schema_version: "studio.compile-traceability.v1",
  source: "ode",
  source_payload: {
    equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
    init: { v: -65 },
    params: { C: 1, E_L: -65, tau_m: 10 },
    reset: "v = -65",
    threshold: "v > -50",
  },
  status: "completed",
  traceability_sha256: "3".repeat(64),
};

vi.mock("../stores/studio", () => ({
  useStudioStore: () => ({
    compileEvidenceBundle: {
      artifact_paths: ["evidence/replay.json"],
      artifacts: [
        {
          relative_path: "evidence/replay.json",
          sha256: "c".repeat(64),
          size_bytes: 128,
        },
      ],
      bundle_id: "seb_compile",
      job_id: "sj_compile",
      manifest: { entries: [{ type: "command_replay" }] },
      schema_version: "studio.evidence-bundle.v1",
      summary: {
        artifact_path_count: 1,
        entry_count: 1,
        entry_type_counts: { command_replay: 1 },
        evidence_classification_counts: {},
        source_job_count: 0,
        source_job_kind_counts: {},
        source_job_owner_counts: {},
      },
    },
    compileEvidenceBundleError: null,
    compileEvidenceBundleLoading: false,
    compileTraceability,
    createEvidenceBundleForSurface: async () => undefined,
    downloadEvidenceBundleArtifactForSurface: async () => undefined,
    irErrors: [],
    irText: "%0 = input clk",
    isSimulating: false,
    svSource: "module lif_neuron; endmodule",
    verilogSrc: "",
  }),
}));

describe("CompilerInspector", () => {
  it("renders compile traceability export controls", () => {
    const html = renderToStaticMarkup(<CompilerInspector />);

    expect(html).toContain("schema studio.compile-traceability.v1");
    expect(html).toContain("class compile");
    expect(html).toContain("status completed");
    expect(html).toContain("module lif_neuron");
    expect(html).toContain("input 111111111111");
    expect(html).toContain("rtl 222222222222");
    expect(html).toContain("trace 333333333333");
    expect(html).toContain("Export compile evidence bundle");
    expect(html).toContain("bundle seb_compile");
    expect(html).toContain("evidence/replay.json");
    expect(html).toContain("Download compile evidence artifact evidence/replay.json");
  });
});
