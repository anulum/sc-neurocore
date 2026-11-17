// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — When the Studio refuses to route or synthesise, and how it estimates and checks tools, against a mocked server

import { createStore } from "zustand/vanilla";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  fetchSynthEstimate,
  fetchSynthTools,
  runMultiTargetSynthesis,
  runSynthesis,
  runSynthesisTerminal,
} from "../api/client";
import type { SynthEstimate, SynthToolInfo } from "../api/types";
import { studioInitialData } from "./studioInitialState";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    fetchSynthEstimate: vi.fn(),
    fetchSynthTools: vi.fn(),
    runMultiTargetSynthesis: vi.fn(),
    runSynthesis: vi.fn(),
    runSynthesisTerminal: vi.fn(),
  };
});

afterEach(() => {
  vi.resetAllMocks();
});

const RTL_SHA = "a".repeat(64);

/** A selected catalogue model with Verilog, compiled and co-simulated bit-exactly against that Verilog. */
const ROUTABLE = {
  sourceMode: "model",
  svSource: "module lif(input clk); endmodule",
  synthTarget: "ice40",
  compileTraceability: { output: { rtl_sha256: RTL_SHA } } as unknown as StudioState["compileTraceability"],
  cosimResult: { bit_exact: true, rtl: { source_sha256: RTL_SHA } } as unknown as StudioState["cosimResult"],
} satisfies Partial<StudioState>;

/** A store with every real action over the initial data. */
function hardwareStore(overrides: Partial<StudioState> = {}) {
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
  store.setState(overrides);
  return store;
}

describe("choosing a synthesis target", () => {
  it("records the target and voids every synthesis result of the earlier one", () => {
    const store = hardwareStore({
      synthResult: {} as StudioState["synthResult"], multiTargetResult: {} as StudioState["multiTargetResult"],
      synthEstimate: {} as StudioState["synthEstimate"], latestSynthesisJobId: "sj_1",
    });
    store.getState().setSynthTarget("ecp5");
    expect(store.getState()).toMatchObject({
      synthTarget: "ecp5", synthResult: null, multiTargetResult: null, synthEstimate: null, latestSynthesisJobId: null,
    });
  });
});

describe("routing a selected catalogue model", () => {
  it.each<[string, Partial<StudioState>, string]>([
    ["it was never compiled", { compileTraceability: null },
      "Compile and bit-exact co-simulate the selected model before routing."],
    ["it was never co-simulated", { cosimResult: null },
      "Compile and bit-exact co-simulate the selected model before routing."],
    ["its co-simulation was not bit-exact",
      { cosimResult: { bit_exact: false, rtl: { source_sha256: RTL_SHA } } as unknown as StudioState["cosimResult"] },
      "Selected RTL does not have current bit-exact co-simulation parity."],
    ["its co-simulation ran on other RTL",
      { cosimResult: { bit_exact: true, rtl: { source_sha256: "b".repeat(64) } } as unknown as StudioState["cosimResult"] },
      "Selected RTL does not have current bit-exact co-simulation parity."],
    ["the target has no place-and-route terminal", { synthTarget: "gowin" },
      "Target gowin has no selected-RTL place-and-route terminal."],
  ])("is refused when %s, and nothing is sent", async (_, change, message) => {
    const store = hardwareStore({ ...ROUTABLE, ...change });
    await store.getState().runSynthesis();
    expect(store.getState().error).toBe(message);
    expect(store.getState().isSimulating).toBe(false);
    expect(runSynthesisTerminal).not.toHaveBeenCalled();
    expect(runSynthesis).not.toHaveBeenCalled();
  });

  it("is refused for several targets at once, which only ODE equations may use", async () => {
    const store = hardwareStore({ ...ROUTABLE });
    await store.getState().runMultiTargetSynthesis();
    expect(store.getState().error).toBe("Selected models use the digest-bound single-target synthesis/PnR terminal.");
    expect(store.getState().isSimulating).toBe(false);
    expect(runMultiTargetSynthesis).not.toHaveBeenCalled();
  });
});

describe("estimating resources", () => {
  it.each([
    ["no IR was built", ""],
    ["the IR holds no operation", "module lif\n  // no operations"],
  ])("asks for a built IR when %s", async (_, irText) => {
    const store = hardwareStore({ irText });
    await store.getState().runSynthEstimate();
    expect(store.getState().error).toBe("Build IR first to estimate resources");
    expect(fetchSynthEstimate).not.toHaveBeenCalled();
  });

  it("counts the IR's operations and loads the estimate for the chosen target", async () => {
    const store = hardwareStore({ synthTarget: "ecp5", irText: "func @lif {\n  %0 = add\n  %1 = mul\n  return %1\n}" });
    const estimate = { luts: 120 } as unknown as SynthEstimate;
    vi.mocked(fetchSynthEstimate).mockResolvedValue(estimate);
    await store.getState().runSynthEstimate();
    expect(fetchSynthEstimate).toHaveBeenCalledWith(2, "ecp5");
    expect(store.getState().synthEstimate).toBe(estimate);
  });

  it("reports an estimate the server refuses", async () => {
    const store = hardwareStore({ irText: "%0 = add" });
    vi.mocked(fetchSynthEstimate).mockRejectedValue(new Error(""));
    await store.getState().runSynthEstimate();
    expect(store.getState().error).toBe("Synthesis estimate failed");
  });
});

describe("checking the synthesis tools", () => {
  it("records which tools the server has, and treats a failed check as not critical", async () => {
    const store = hardwareStore();
    const tools = { yosys: { available: true } } as unknown as Record<string, SynthToolInfo>;
    vi.mocked(fetchSynthTools).mockResolvedValueOnce(tools).mockRejectedValueOnce(new Error("offline"));
    await store.getState().checkSynthTools();
    expect(store.getState().toolsAvailable).toBe(tools);
    await store.getState().checkSynthTools();
    expect(store.getState().toolsAvailable).toBe(tools);
    expect(store.getState().error).toBeNull();
  });
});
