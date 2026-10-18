// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Heatmap selection preserves exact inactive seed inputs

import { afterEach, expect, it, vi } from "vitest";
import type { HeatmapResponse } from "./api/client";
import { studioExperimentKey } from "./studioExperimentKey";
import { selectStudioHeatmapPoint } from "./studioHeatmapSelection";
import { studioSimulationConfigInput } from "./studioSimulationConfigInput";
import { useStudioStore } from "./stores/studio";

const initial = useStudioStore.getState();
const heatmap: HeatmapResponse = {
  param_x: "tau", param_y: "gain", x_values: [20], y_values: [2],
  rates: [[10]], rate_min: 10, rate_max: 10,
  analysis_metadata: {
    analysis_type: "heatmap", evidence_classification: "analysis",
    input_sha256: "a".repeat(64), result_sha256: "b".repeat(64),
    output_keys: ["rates"], schema_version: "studio.analysis-result.v1",
    source: "model", status: "completed",
  },
};

afterEach(() => { useStudioStore.setState(initial, true); });

it.each(["model", "ode"] as const)("adopts a current %s cell without converting its stored seed", async (sourceMode) => {
  const runSimulation = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
  useStudioStore.setState({
    sourceMode, selectedModelName: "lif", runSimulation,
    modelParams: { tau: 10, gain: 1, seed: 9223372036854775807n },
    odeParams: { tau: 5, gain: 3 }, heatmapResult: heatmap,
  });
  useStudioStore.setState({
    heatmapExperimentKey: studioExperimentKey(studioSimulationConfigInput(useStudioStore.getState())),
  });
  await selectStudioHeatmapPoint(heatmap, 0, 0, useStudioStore.getState, useStudioStore.setState);
  const state = useStudioStore.getState();
  expect(state.modelParams.seed).toBe(9223372036854775807n);
  expect(sourceMode === "model" ? state.modelParams : state.odeParams).toMatchObject({ tau: 20, gain: 2 });
  expect(sourceMode === "model" ? state.odeParams.tau : state.modelParams.tau).toBe(sourceMode === "model" ? 5 : 10);
  expect(state.activeTab).toBe("trace");
  expect(runSimulation).toHaveBeenCalledOnce();
});

it("refuses a stale cell without changing the model's exact seed", async () => {
  const runSimulation = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
  const params = { tau: 10, gain: 1, seed: 9223372036854775807n };
  useStudioStore.setState({ modelParams: params, heatmapResult: heatmap, heatmapExperimentKey: null, runSimulation });
  await selectStudioHeatmapPoint(heatmap, 0, 0, useStudioStore.getState, useStudioStore.setState);
  expect(useStudioStore.getState().modelParams).toBe(params);
  expect(useStudioStore.getState().error).toContain("another experiment");
  expect(runSimulation).not.toHaveBeenCalled();
});
