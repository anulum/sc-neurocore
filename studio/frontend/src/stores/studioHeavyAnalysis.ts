// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Async heavy-analysis orchestration for the Studio store (W12-D path).

import type { AnalysisJobKind } from "../api/client";
import { buildStudioAnalysisJobSelection } from "../studioAnalysisJobSelection";
import { runStudioAnalysisJob } from "../studioAnalysisJobRunner";
import { studioAnalysisFailureState } from "../studioAnalysisState";
import type { StudioSimulationConfigInput } from "../studioSimulationConfig";
import type { StudioState } from "./studioTypes";

export function simulationConfigInput(s: StudioState): StudioSimulationConfigInput {
  return {
    sourceMode: s.sourceMode,
    selectedModelName: s.selectedModelName,
    modelParams: s.modelParams,
    equations: s.equations,
    threshold: s.threshold,
    reset: s.reset,
    odeParams: s.odeParams,
    odeInit: s.odeInit,
    dt: s.dt,
    duration: s.duration,
    current: s.current,
    protocol: s.protocol,
  };
}

/** W12-D: heavy analyses via async job runner (stable method names, sweep guards). */
export async function runStoreHeavyAnalysis(
  kind: AnalysisJobKind,
  get: () => StudioState,
  set: (partial: Partial<StudioState>) => void,
): Promise<void> {
  const s = get();
  if (s.isSimulating) return;
  if (kind === "bifurcation" && !s.sweepParam) return;
  if (kind === "heatmap" && (!s.sweepParam || !s.sweepParamY)) return;
  const selection = buildStudioAnalysisJobSelection({
    analysis: kind,
    sourceMode: s.sourceMode,
    modelParams: s.modelParams,
    odeParams: s.odeParams,
    sweepParam: s.sweepParam,
    sweepParamY: s.sweepParamY,
  });
  if (!selection.ok) {
    set(studioAnalysisFailureState(selection.error));
    return;
  }
  const outcome = await runStudioAnalysisJob(
    { simulation: simulationConfigInput(s), selection: selection.selection },
    { applyPatch: (patch) => set(patch) },
  );
  if (!outcome.ok && outcome.stage === "request") {
    set(studioAnalysisFailureState(outcome.error));
  }
}

