// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio experiment export actions

import { fetchCodegen, fetchReplayNotebook, fetchReplayPack } from "../api/client";
import { studioExperimentExportRequest } from "../studioSimulationConfig";
import {
  studioAnalysisErrorState,
  studioCodegenResultState,
  studioCodegenStartState,
} from "../studioAnalysisState";
import { replayPackExport } from "../simulationExports";
import { exportStudioTrace } from "./studioTraceActions";
import { downloadBrowserArtefact } from "../browserArtefactDownload";
import { simulationConfigInput } from "./studioHeavyAnalysis";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/**
 * The experiment export actions: generating the Python script that reproduces the experiment, with
 * its one-line form, its replay script and its digest; downloading the replay pack and the replay
 * notebook; and exporting the trace as JSON, CSV or SVG.
 */
export type StudioExportActions = Pick<StudioStoreActions,
  "runCodegen" | "exportReplayPack" | "exportReplayNotebook" | "exportData" | "exportCSV" | "exportSVG">;

/**
 * Bind these actions to the Studio store.
 *
 * The experiment export actions: generating the Python script that reproduces the experiment, with
 * its one-line form, its replay script and its digest; downloading the replay pack and the replay
 * notebook; and exporting the trace as JSON, CSV or SVG. For a catalogue model, the exports that go
 * to the server wait until the selected model's details have arrived, so an export never describes
 * a model the Studio is still loading.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @returns The actions.
 */
export function createStudioExportActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): StudioExportActions {
  return {
  runCodegen: async () => {
    const s = get();
    set(studioCodegenStartState());
    try {
      if (s.sourceMode === "model" && s.modelDetail?.name !== s.selectedModelName) {
        throw new Error("Selected model is still loading");
      }
      const res = await fetchCodegen(studioExperimentExportRequest(simulationConfigInput(s)));
      set(studioCodegenResultState(res.script, res.oneliner, res.replay_script, res.experiment_sha256));
    } catch (e) { set(studioAnalysisErrorState(e instanceof Error ? e.message : String(e))); }
  },

  exportReplayPack: async () => {
    const s = get();
    try {
      if (s.sourceMode === "model" && s.modelDetail?.name !== s.selectedModelName) {
        throw new Error("Selected model is still loading");
      }
      const pack = await fetchReplayPack(studioExperimentExportRequest(simulationConfigInput(s)));
      const artefact = replayPackExport(pack);
      downloadBrowserArtefact(artefact.blob, artefact.filename);
    } catch (e) { set(studioAnalysisErrorState(e instanceof Error ? e.message : String(e))); }
  },

  exportReplayNotebook: async () => {
    const s = get();
    try {
      if (s.sourceMode === "model" && s.modelDetail?.name !== s.selectedModelName) {
        throw new Error("Selected model is still loading");
      }
      const notebook = await fetchReplayNotebook(studioExperimentExportRequest(simulationConfigInput(s)));
      const stem = s.sourceMode === "model" ? s.selectedModelName : "equations";
      downloadBrowserArtefact(
        new Blob([JSON.stringify(notebook, null, 1)], { type: "application/x-ipynb+json" }),
        `${stem}-replay.ipynb`,
      );
    } catch (e) { set(studioAnalysisErrorState(e instanceof Error ? e.message : String(e))); }
  },

  exportData: () => { exportStudioTrace("json", get, set); },
  exportCSV: () => { exportStudioTrace("csv", get, set); },
  exportSVG: () => { exportStudioTrace("svg", get, set); },
  };
}
