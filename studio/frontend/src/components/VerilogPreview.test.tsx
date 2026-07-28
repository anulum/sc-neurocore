// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio RTL preview model-mode tests

import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockStore } = vi.hoisted(() => ({
  mockStore: { current: {} as Record<string, unknown> },
}));

vi.mock("@monaco-editor/react", () => ({ default: () => <div>editor</div> }));
vi.mock("../stores/studio", () => ({
  useStudioStore: () => mockStore.current,
}));

describe("VerilogPreview", () => {
  beforeEach(() => {
    mockStore.current = {
      cosimResult: null,
      compileTraceability: null,
      isSimulating: false,
      modelDetail: null,
      modelIntegrator: "",
      runCosim: vi.fn(),
      sourceMode: "model",
      verilogSrc: "",
    };
  });

  it("invites compilation of the selected catalogue model", async () => {
    const { default: VerilogPreview } = await import("./VerilogPreview");
    const html = renderToStaticMarkup(<VerilogPreview />);

    expect(html).toContain("selected model");
    expect(html).not.toContain("Switch to ODE mode");
  });

  it("renders the consumed bit-exact real-tool parity report", async () => {
    mockStore.current = {
      cosimResult: {
        bit_exact: true,
        configuration: { integrator: "map", q_format: "Q8.8" },
        first_mismatch: null,
        rtl: { source_sha256: "b".repeat(64), trace_sha256: "a".repeat(64) },
        sample_count: 128,
        signals: ["spike_out", "v_out"],
        tools: { gcc: "gcc 13", iverilog: "Icarus 12", vvp: "VVP 12" },
      },
      compileTraceability: { output: { rtl_sha256: "b".repeat(64) } },
      isSimulating: false,
      modelDetail: { compile_configuration: { cosim_integrators: ["map"] } },
      modelIntegrator: "map",
      runCosim: vi.fn(),
      sourceMode: "model",
      verilogSrc: "module sc_model; endmodule",
    };
    const { default: VerilogPreview } = await import("./VerilogPreview");
    const html = renderToStaticMarkup(<VerilogPreview />);

    expect(html).toContain("BIT-EXACT PASS");
    expect(html).toContain("128 cycles");
    expect(html).toContain("GCC + Icarus/VVP");
    expect(html).toContain("aaaaaaaaaaaa");
    expect(html).toContain("compiled RTL match");
  });
});
