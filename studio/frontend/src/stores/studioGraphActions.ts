// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio graph-editing actions

import {
  graphEditRecorded,
  graphRedone,
  graphSnapshotOf,
  graphUndone,
  isLayoutOnlyUpdate,
  type StudioGraphHistory,
  type StudioGraphHistoryStep,
  type StudioGraphSnapshot,
} from "../studioGraphHistory";
import {
  type PopulationNode,
  type ProjectionEdge,
  fetchGraphModels as apiFetchGraphModels,
  createPopulation as apiCreatePop,
  createProjection as apiCreateProj,
  simulateGraph as apiSimGraph,
  graphModelContract as apiGraphModelContract,
  validateGraph as apiValidateGraph,
  exportNIR as apiExportNIR,
  fetchNetworkNotebook,
  importNIR as apiImportNIR,
  runPipeline as apiRunPipeline,
} from "../api/client";
import {
  studioDefaultPopulationRequest,
  studioDefaultProjectionRequest,
  studioGraphFailureState,
  studioGraphImportedState,
  studioGraphModelsLoadedState,
  studioGraphRequest,
  studioGraphSimulationCompletedState,
  studioGraphSimulationStartState,
  studioGraphWithoutPopulation,
  studioPipelineCompletedState,
  studioPipelineStartState,
  studioPopulationAddedState,
  studioPopulationUpdatedState,
  studioProjectionAddedState,
  studioProjectionRemovedState,
  studioProjectionUpdatedState,
} from "../studioGraphRequests";
import { studioGraphValidatedState, studioGraphValidationLocatedState } from "../studioGraphValidation";
import { studioDuplicatePlan, studioDuplicateSummary } from "../studioGraphDuplicate";
import { downloadBrowserArtefact } from "../browserArtefactDownload";
import {
  networkNirExportNotice,
  networkNirExportPlan,
  networkNirImportNotice,
  networkNirImportRequest,
} from "../networkNirExport";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/**
 * The graph the store currently holds, as a history snapshot.
 *
 * @param state - The store's graph fields.
 * @returns The snapshot.
 */
function studioGraphSnapshotFrom(state: {
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
}): StudioGraphSnapshot {
  return graphSnapshotOf(state.graphPopulations, state.graphProjections);
}

/**
 * The history after recording the graph as it stands before an edit.
 *
 * The snapshot is taken *before* the edit, which is what makes undo restore
 * the state the reader was looking at rather than the one they just created.
 *
 * @param state - The store's graph and history fields.
 * @returns The history with that snapshot recorded.
 */
function studioGraphEditRecordedFrom(state: {
  graphHistory: StudioGraphHistory;
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
}): StudioGraphHistory {
  return graphEditRecorded(state.graphHistory, studioGraphSnapshotFrom(state));
}

/**
 * The state patch that reinstates one recorded graph.
 *
 * @param step - The history step to reinstate.
 * @returns The patch: the graph as it was, and the history that produced it.
 */
function studioGraphRestoredState(step: StudioGraphHistoryStep): {
  graphHistory: StudioGraphHistory;
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
} {
  return {
    graphHistory: step.history,
    graphPopulations: step.snapshot.populations,
    graphProjections: step.snapshot.projections,
  };
}

/**
 * The graph-editing actions: the network's populations and projections, their history, its files, and
 * its run through the hardware pipeline.
 */
export type StudioGraphActions = Pick<StudioStoreActions,
  "loadGraphModels" | "addPopulation" | "removePopulation" | "updatePopulation" | "undoGraphEdit" | "redoGraphEdit" | "addProjection" | "removeProjection" | "updateProjection" | "selectProjection" | "selectPopulations" | "duplicateSelection" | "selectPopulation" | "loadPopulationModelContract" | "validateGraphAction" | "simulateGraphAction" | "exportGraphNIR" | "exportGraphNotebook" | "importGraphNIR" | "runPipelineAction">;

/**
 * Bind the graph-editing actions to the Studio store.
 *
 * The network graph is edited, undone and redone here, validated and simulated
 * through the server, run through the hardware pipeline for the chosen synthesis
 * target, and written to and read from NIR and notebook files. The
 * history of edits lives in the store; the only state kept in this closure is
 * the chain that runs population adds one after another.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @returns The graph-editing actions.
 */
export function createStudioGraphActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): StudioGraphActions {
  // The adds of populations still waiting for the server, run one after another.
  let populationAdds: Promise<void> = Promise.resolve();

  return {
  loadGraphModels: async () => {
    try {
      const graphModels = await apiFetchGraphModels();
      set(studioGraphModelsLoadedState(graphModels));
    } catch { /* non-critical */ }
  },

  addPopulation: (neuronType) => {
    // The label and position come from the population count, so an add reads
    // that count only after every earlier add has been answered and applied:
    // two adds made before the server answers would otherwise share both.
    const add = async () => {
      try {
        const s = get();
        const pop = await apiCreatePop(studioDefaultPopulationRequest(neuronType, s.graphPopulations.length));
        set((prev) => ({
          ...studioPopulationAddedState(prev.graphPopulations, pop),
          graphHistory: studioGraphEditRecordedFrom(prev),
        }));
      } catch (e) { set(studioGraphFailureState(e, "Population creation failed")); }
    };
    // `add` settles without rejecting, since everything it does before the
    // failure report is inside the `try`, so the chain never stops.
    populationAdds = populationAdds.then(add);
    return populationAdds;
  },

  removePopulation: (id) => {
    set((s) => {
      const graph = studioGraphWithoutPopulation({
        populations: s.graphPopulations,
        projections: s.graphProjections,
      }, id);
      if (graph.populations.length === s.graphPopulations.length) {
        return {};
      }
      const survives = graph.projections.some(
        (projection) => projection.id === s.selectedProjectionId,
      );
      return {
        graphHistory: studioGraphEditRecordedFrom(s),
        graphPopulations: graph.populations,
        graphProjections: graph.projections,
        // Deleting a population takes its projections with it, and one of
        // them may be the projection the editor is editing.
        selectedPopulationId: s.selectedPopulationId === id ? null : s.selectedPopulationId,
        selectedProjectionId: survives ? s.selectedProjectionId : null,
      };
    });
  },

  updatePopulation: (id, updates) => {
    set((s) => {
      const patch = studioPopulationUpdatedState(s.graphPopulations, id, updates);
      // A drag writes a position on every frame; those are not edits to undo.
      if (isLayoutOnlyUpdate(updates)) {
        return patch;
      }
      return { ...patch, graphHistory: studioGraphEditRecordedFrom(s) };
    });
  },

  undoGraphEdit: () => {
    set((s) => {
      const step = graphUndone(s.graphHistory, studioGraphSnapshotFrom(s));
      return step === null ? {} : studioGraphRestoredState(step);
    });
  },

  redoGraphEdit: () => {
    set((s) => {
      const step = graphRedone(s.graphHistory, studioGraphSnapshotFrom(s));
      return step === null ? {} : studioGraphRestoredState(step);
    });
  },

  addProjection: async (sourceId, targetId) => {
    const source = get().graphPopulations.find((population) => population.id === sourceId);
    if (!source) {
      set(studioGraphFailureState(new Error(`Source population ${sourceId} not found`), "Projection creation failed"));
      return;
    }
    try {
      const proj = await apiCreateProj(studioDefaultProjectionRequest(sourceId, targetId, source.neuron_type, get().dt));
      set((prev) => ({
        ...studioProjectionAddedState(prev.graphProjections, proj),
        graphHistory: studioGraphEditRecordedFrom(prev),
      }));
    } catch (e) { set(studioGraphFailureState(e, "Projection creation failed")); }
  },

  removeProjection: (id) => {
    set((s) => {
      const patch = studioProjectionRemovedState(s.graphProjections, id);
      if (patch.graphProjections.length === s.graphProjections.length) {
        return {};
      }
      return {
        ...patch,
        graphHistory: studioGraphEditRecordedFrom(s),
        // Editing a projection that is no longer in the graph would write
        // fields into nothing.
        selectedProjectionId: s.selectedProjectionId === id ? null : s.selectedProjectionId,
      };
    });
  },

  updateProjection: (id, updates) => {
    set((s) => ({
      ...studioProjectionUpdatedState(s.graphProjections, id, updates),
      graphHistory: studioGraphEditRecordedFrom(s),
    }));
  },

  selectProjection: (id) => {
    set({ selectedProjectionId: id, selectedPopulationId: null });
  },

  selectPopulations: (ids) => {
    set({ selectedPopulationIds: ids });
  },

  duplicateSelection: async () => {
    const s = get();
    const plan = studioDuplicatePlan(s.graphPopulations, s.graphProjections, s.selectedPopulationIds);
    if (plan.populations.length === 0) {
      set({ graphNotice: null });
      return;
    }
    const before = studioGraphEditRecordedFrom(s);
    try {
      // Identity is the server's: a client that minted ids would be a second
      // implementation of it, free to collide with the first.
      const created = new Map<string, string>();
      const populations: PopulationNode[] = [];
      for (const copy of plan.populations) {
        const made = await apiCreatePop({
          count: copy.count,
          drive: copy.drive,
          label: copy.label,
          model: copy.model,
          neuron_type: copy.neuron_type,
          params: copy.params,
          x: copy.x,
          y: copy.y,
        });
        created.set(copy.sourceId, made.id);
        populations.push(made);
      }
      const projections: ProjectionEdge[] = [];
      for (const copy of plan.projections) {
        const source = created.get(copy.sourceId);
        const target = created.get(copy.targetId);
        if (source === undefined || target === undefined) {
          // The plan only ever carries projections whose both endpoints were
          // copied, so this cannot happen — and if it ever did, dropping the
          // projection would hand back a graph quietly smaller than the one
          // the user asked for, which reads as success.
          throw new Error(
            `Duplicate produced a projection whose endpoint was not copied: ${copy.sourceId} → ${copy.targetId}`,
          );
        }
        const made = await apiCreateProj({
          delay: copy.delay,
          probability: copy.probability,
          rule: copy.rule,
          source_id: source,
          target_id: target,
          weight: copy.weight,
        });
        // The creation route carries neither seed nor autapses, and both are
        // executed; they are applied here so the copy runs as its original does.
        projections.push({
          ...made,
          ...(copy.autapses === undefined ? {} : { autapses: copy.autapses }),
          ...(copy.seed === undefined ? {} : { seed: copy.seed }),
        });
      }
      set((prev) => ({
        graphHistory: before,
        graphNotice: studioDuplicateSummary(plan),
        graphPopulations: [...prev.graphPopulations, ...populations],
        graphProjections: [...prev.graphProjections, ...projections],
        selectedPopulationIds: populations.map((population) => population.id),
      }));
    } catch (e) {
      set(studioGraphFailureState(e, "Duplicating the selection failed"));
    }
  },

  selectPopulation: (id) => {
    set({ selectedPopulationId: id, selectedProjectionId: null });
    if (id === null) return;
    const population = get().graphPopulations.find((one) => one.id === id);
    if (population !== undefined) {
      void get().loadPopulationModelContract(population.model);
    }
  },

  loadPopulationModelContract: async (model) => {
    if (get().populationModelContract?.model === model) return;
    try {
      set({ populationModelContract: await apiGraphModelContract(model) });
    } catch (e) {
      // The parameters simply are not offered without their contract; the
      // identity and drive fields still are, and the failure is reported
      // rather than shown as a model with no parameters.
      set({
        ...studioGraphFailureState(e, `Model contract for ${model} could not be loaded`),
        populationModelContract: null,
      });
    }
  },

  validateGraphAction: async () => {
    const s = get();
    try {
      const graph = studioGraphRequest(
        s.graphPopulations,
        s.graphProjections,
        s.duration,
        s.dt,
        s.seed,
      );
      const validation = await apiValidateGraph(graph);
      set(studioGraphValidatedState(validation, s.graphPopulations, s.graphProjections));
    } catch (e) {
      set(studioGraphFailureState(e, "Graph validation failed"));
    }
  },

  simulateGraphAction: async () => {
    const s = get();
    if (s.isSimulating) return;
    set(studioGraphSimulationStartState());
    try {
      const graph = studioGraphRequest(s.graphPopulations, s.graphProjections, s.duration, s.dt, s.seed);
      const validation = await apiValidateGraph(graph);
      if (!validation.valid) {
        set(
          studioGraphValidationLocatedState(validation, s.graphPopulations, s.graphProjections),
        );
        return;
      }
      const graphSimResult = await apiSimGraph(graph);
      set(studioGraphSimulationCompletedState(graphSimResult));
    } catch (e) { set(studioGraphFailureState(e, "Graph simulation failed", { clearBusy: true })); }
  },

  exportGraphNIR: async () => {
    const s = get();
    try {
      const exported = await apiExportNIR(
        studioGraphRequest(s.graphPopulations, s.graphProjections, s.duration, s.dt, s.seed),
      );
      networkNirExportPlan(exported).writeArtefact();
      set({ graphNotice: networkNirExportNotice(exported) });
    } catch (e) { set(studioGraphFailureState(e, "Graph NIR export failed")); }
  },

  exportGraphNotebook: async () => {
    const s = get();
    try {
      const notebook = await fetchNetworkNotebook(
        studioGraphRequest(s.graphPopulations, s.graphProjections, s.duration, s.dt, s.seed),
      );
      downloadBrowserArtefact(
        new Blob([JSON.stringify(notebook, null, 1)], { type: "application/x-ipynb+json" }),
        "network-tutorial.ipynb",
      );
      set({ graphNotice: "Notebook written: it rebuilds this network step by step and checks its spikes against this run." });
    } catch (e) { set(studioGraphFailureState(e, "Network notebook export failed")); }
  },

  importGraphNIR: async (file) => {
    try {
      const imported = await apiImportNIR(await networkNirImportRequest(file));
      set({ ...studioGraphImportedState(imported.graph), graphNotice: networkNirImportNotice(imported) });
    } catch (e) { set(studioGraphFailureState(e, "Graph NIR import failed")); }
  },

  runPipelineAction: async () => {
    const s = get();
    if (s.isSimulating || s.graphPopulations.length === 0) return;
    set(studioPipelineStartState());
    try {
      const graph = studioGraphRequest(s.graphPopulations, s.graphProjections, s.duration, s.dt, s.seed);
      const pipelineResult = await apiRunPipeline(graph, s.synthTarget);
      set(studioPipelineCompletedState(pipelineResult));
    } catch (e) {
      set({ ...studioGraphFailureState(e, "Pipeline run failed", { clearBusy: true }), pipelineRunning: false });
    }
  },
  };
}
