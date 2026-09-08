// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Heatmap parameter selection ownership

import type { HeatmapResponse } from "./api/client";
import { compilerConfigurationInvalidatedState } from "./compilerStoreState";
import { studioExperimentKey } from "./studioExperimentKey";
import { studioSimulationConfigInput } from "./studioSimulationConfigInput";
import type { StudioState } from "./stores/studioTypes";

/**
 * Adopt a cell only from the current heatmap and invalidate compiled derivatives.
 *
 * @param heatmap - Exact result rendered by the clicked canvas.
 * @param xi - Horizontal cell index.
 * @param yi - Vertical cell index.
 * @param get - Read current experiment and request state.
 * @param set - Apply the parameter pair and invalidation atomically.
 */
export async function selectStudioHeatmapPoint(
  heatmap: HeatmapResponse, xi: number, yi: number,
  get: () => StudioState, set: (patch: Partial<StudioState>) => void,
): Promise<void> {
  const state = get();
  if (state.isSimulating) return;
  try {
    if (state.heatmapResult !== heatmap || state.heatmapExperimentKey === null
      || state.heatmapExperimentKey !== studioExperimentKey(studioSimulationConfigInput(state))) {
      throw new Error("Heatmap belongs to another experiment; run the sweep again before selecting a point.");
    }
    const params = state.sourceMode === "model" ? state.modelParams : state.odeParams;
    const x = heatmap.x_values[xi], y = heatmap.y_values[yi];
    if (!Number.isInteger(xi) || !Number.isInteger(yi) || x === undefined || y === undefined
      || !Number.isFinite(x) || !Number.isFinite(y) || heatmap.param_x === heatmap.param_y
      || !Object.hasOwn(params, heatmap.param_x) || !Object.hasOwn(params, heatmap.param_y)) {
      throw new Error("Heatmap point requires finite values for two existing parameters.");
    }
    const selected = { ...params, [heatmap.param_x]: x, [heatmap.param_y]: y };
    set({ ...compilerConfigurationInvalidatedState(), activeTab: "trace",
      ...(state.sourceMode === "model" ? { modelParams: selected } : { odeParams: selected }) });
    await get().runSimulation();
  } catch (error: unknown) {
    set({ error: error instanceof Error ? error.message : String(error) });
  }
}
