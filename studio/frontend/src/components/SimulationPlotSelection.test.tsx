// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Heatmap selection through mounted UI
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { useStudioStore } from "../stores/studio";
import { studioExperimentKey } from "../studioExperimentKey";
import { studioSimulationConfigInput } from "../studioSimulationConfigInput";
import SimulationPlot from "./SimulationPlot";

const initial = useStudioStore.getState();
let root: Root | undefined;
afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  useStudioStore.setState(initial, true);
  document.body.replaceChildren();
  vi.restoreAllMocks(); vi.unstubAllGlobals();
});

it.each((["model", "ode"] as const).flatMap((sourceMode) =>
  (["current", "stale", "busy", "invalid"] as const).map((condition) => ({ sourceMode, condition }))))("handles $sourceMode heatmap click with $condition context", async ({ sourceMode, condition }) => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  const fetch = vi.fn<typeof globalThis.fetch>().mockRejectedValue(new Error("Controlled simulation failure"));
  vi.stubGlobal("fetch", fetch);
  useStudioStore.setState({ sourceMode, activeTab: "heatmap", modelParams: { x: 1, y: 2 }, odeParams: { x: 1, y: 2 },
    verilogSrc: "module previous; endmodule",
    heatmapResult: { param_x: "x", param_y: "y", x_values: [3], y_values: [4], rates: [[1]], rate_min: 1, rate_max: 1,
      analysis_metadata: { analysis_type: "heatmap", evidence_classification: "analysis", input_sha256: "a".repeat(64),
        result_sha256: "b".repeat(64), output_keys: ["rates"], schema_version: "studio.analysis-result.v1", source: sourceMode, status: "completed" } },
  });
  const key = studioExperimentKey(studioSimulationConfigInput(useStudioStore.getState()));
  // Seed a completed-analysis context; lifecycle production is tested separately.
  useStudioStore.setState({ analysisExperimentKey: key, heatmapExperimentKey: key });
  if (condition === "stale") {
    useStudioStore.setState({ duration: 200 });
    useStudioStore.setState({ analysisExperimentKey: studioExperimentKey(studioSimulationConfigInput(useStudioStore.getState())) });
  }
  if (condition === "busy") useStudioStore.setState({ isSimulating: true });
  if (condition === "invalid") {
    const heatmap = useStudioStore.getState().heatmapResult;
    if (!heatmap) throw new Error("Missing seeded heatmap");
    useStudioStore.setState({ heatmapResult: { ...heatmap, x_values: [Number.NaN] } });
  }
  const container = document.createElement("div"); document.body.append(container);
  root = createRoot(container);
  await act(async () => { root?.render(<SimulationPlot />); await Promise.resolve(); });
  const canvas = container.querySelector("canvas");
  if (!canvas) throw new Error("Heatmap canvas missing");
  canvas.width = 300; canvas.height = 150;
  await act(async () => { canvas.dispatchEvent(new MouseEvent("click", { bubbles: true, clientX: 100, clientY: 40 }));
    await Promise.resolve(); });
  if (condition === "current") {
    expect(useStudioStore.getState()[sourceMode === "model" ? "modelParams" : "odeParams"]).toEqual({ x: 3, y: 4 });
    expect(useStudioStore.getState().verilogSrc).toBe("");
    expect(fetch).toHaveBeenCalledOnce();
    expect(useStudioStore.getState().error).toBe("Controlled simulation failure");
  } else {
    expect(useStudioStore.getState()[sourceMode === "model" ? "modelParams" : "odeParams"]).toEqual({ x: 1, y: 2 });
    expect(useStudioStore.getState().verilogSrc).toBe("module previous; endmodule");
    expect(fetch).not.toHaveBeenCalled();
    if (condition === "stale") expect(useStudioStore.getState().error).toContain("another experiment");
    if (condition === "invalid") expect(useStudioStore.getState().error).toContain("finite values");
    if (condition === "busy") expect(useStudioStore.getState().isSimulating).toBe(true);
  }
});
