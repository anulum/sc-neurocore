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
  traceability_sha256: "3".repeat(64),
};

vi.mock("../stores/studio", () => ({
  useStudioStore: () => ({
    compileEvidenceBundle: null,
    compileEvidenceBundleError: null,
    compileEvidenceBundleLoading: false,
    compileTraceability,
    createEvidenceBundleForSurface: async () => undefined,
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
    expect(html).toContain("module lif_neuron");
    expect(html).toContain("input 111111111111");
    expect(html).toContain("rtl 222222222222");
    expect(html).toContain("trace 333333333333");
    expect(html).toContain("Export compile evidence bundle");
  });
});
