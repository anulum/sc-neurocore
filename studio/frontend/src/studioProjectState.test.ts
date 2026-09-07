// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio project state snapshot helper tests

import { describe, expect, it } from "vitest";

import { StudioRequestError } from "./api/client";
import {
  studioProjectDeletedListedState,
  studioProjectExpectedRevision,
  studioProjectFailureState,
  studioProjectListLoadedState,
  studioProjectRestoreState,
  studioProjectRevisionFromLoadResponse,
  studioProjectSaveFailureState,
  studioProjectSaveState,
  studioProjectSavedState,
  studioProjectStateFromLoadResponse,
  type StudioProjectSnapshotInput,
  type StudioProjectStateSnapshot,
  type StudioProjectTrainingConfig,
} from "./studioProjectState";

const fallbackTrainingConfig: StudioProjectTrainingConfig = {
  dataset: "synthetic",
  epochs: 10,
  batch_size: 64,
  lr: 0.001,
  hidden: [128],
  timesteps: 25,
  surrogate: "atan_surrogate",
  learn_beta: false,
  learn_threshold: false,
};

/**
 * A stored workspace.
 *
 * @returns The fixture.
 */
function snapshot(): StudioProjectStateSnapshot {
  return {
    sourceMode: "ode",
    equations: ["dv/dt = -v / tau"],
    threshold: "v > 1",
    reset: "v = 0",
    odeParams: { tau: 10 },
    odeInit: { v: 0 },
    selectedModelName: "lif",
    modelParams: { tau: 20 },
    dt: 0.05,
    duration: 250,
    current: 0,
    protocol: "step",
    graphPopulations: [{
      id: "p1",
      type: "population",
      label: "Exc",
      model: "LIFNeuron",
      count: 80,
      neuron_type: "excitatory",
      position: { x: 10, y: 20 },
      params: { tau: 10 },
    }],
    graphProjections: [{
      id: "e1",
      source: "p1",
      target: "p1",
      weight: 0.1,
      delay: 1,
      probability: 0.2,
    }],
    synthTarget: "ecp5",
    trainingConfig: {
      dataset: "mnist",
      epochs: 4,
      batch_size: 32,
      lr: 0.002,
      hidden: [64, 32],
      timesteps: 50,
      surrogate: "fast_sigmoid",
      learn_beta: true,
      learn_threshold: true,
    },
  };
}

describe("Studio project state helpers", () => {
  it("builds the complete project save snapshot", () => {
    expect(studioProjectSaveState(snapshot())).toEqual(snapshot());
  });

  it("builds a project save snapshot from a wider store state object", () => {
    const widerState: StudioProjectSnapshotInput & { error: string | null; isSimulating: boolean } = {
      ...snapshot(),
      error: "ignored",
      isSimulating: true,
    };

    expect(studioProjectSaveState(widerState)).toEqual(snapshot());
  });

  it("restores project state and preserves finite zero current", () => {
    const restored = studioProjectStateFromLoadResponse(
      { state: snapshot() },
      fallbackTrainingConfig,
    );

    expect(restored.sourceMode).toBe("ode");
    expect(restored.dt).toBe(0.05);
    expect(restored.current).toBe(0);
    expect(restored.trainingConfig).toEqual(snapshot().trainingConfig);
    expect(restored.graphPopulations).toHaveLength(1);
    expect(restored.graphProjections).toHaveLength(1);
  });

  it("defaults malformed loaded state fields without accepting invalid numeric records", () => {
    const restored = studioProjectStateFromLoadResponse({
      state: {
        sourceMode: "bad",
        equations: ["dv/dt = -v", 5],
        odeParams: { tau: 10, bad: Number.NaN, text: "x" },
        odeInit: null,
        dt: 0,
        duration: -100,
        current: 0,
        graphPopulations: [{ id: "p1" }, null],
        graphProjections: "bad",
        trainingConfig: {
          epochs: Number.POSITIVE_INFINITY,
          hidden: [64, "bad"],
          learn_beta: true,
        },
      },
    }, fallbackTrainingConfig);

    expect(restored.sourceMode).toBe("model");
    expect(restored.equations).toEqual([]);
    expect(restored.odeParams).toEqual({ tau: 10 });
    expect(restored.odeInit).toEqual({});
    expect(restored.dt).toBe(0.1);
    expect(restored.duration).toBe(100);
    expect(restored.current).toBe(0);
    expect(restored.graphPopulations).toEqual([{ id: "p1" }]);
    expect(restored.graphProjections).toEqual([]);
    expect(restored.trainingConfig).toEqual({
      ...fallbackTrainingConfig,
      learn_beta: true,
    });
  });

  it("returns a coherent default snapshot when the load response has no state object", () => {
    expect(studioProjectStateFromLoadResponse({}, fallbackTrainingConfig)).toMatchObject({
      sourceMode: "model",
      equations: [],
      threshold: "",
      reset: "",
      odeParams: {},
      odeInit: {},
      selectedModelName: "",
      modelParams: {},
      dt: 0.1,
      duration: 100,
      current: 10,
      protocol: "constant",
      graphPopulations: [],
      graphProjections: [],
      synthTarget: "ice40",
      trainingConfig: fallbackTrainingConfig,
    });
  });

  it("builds project persistence store patches", () => {
    const saved = {
      evidence_classification: "project_workspace" as const,
      name: "demo",
      project_sha256: "a".repeat(64),
      parent_revision: 2,
      revision: 3,
      saved_at: 1782028800,
      schema_version: "studio.project-save.v1" as const,
      state_sha256: "b".repeat(64),
      version: "studio.project.v1",
    };
    const summaries = [{
      name: "demo",
      revision: 3,
      revision_count: 3,
      saved_at: 1782028800,
      version: "studio.project.v1",
    }];

    expect(studioProjectSavedState(saved)).toEqual({
      projectRevision: { name: saved.name, revision: 3 },
      projectSaveResult: saved,
    });
    expect(studioProjectListLoadedState(summaries)).toEqual({ serverProjects: summaries });
    expect(studioProjectRestoreState(snapshot())).toEqual(snapshot());
    expect(studioProjectFailureState(new Error("storage offline"), "fallback")).toEqual({
      error: "storage offline",
    });
    expect(studioProjectFailureState("bad", "fallback")).toEqual({ error: "fallback" });
  });
});

describe("the revision an editor saves from", () => {
  it("is the one it loaded, for the workspace it loaded", () => {
    expect(studioProjectExpectedRevision({ name: "demo", revision: 4 }, "demo")).toBe(4);
  });

  it("is absent under a different name, so a save-as claims a new workspace", () => {
    // Sending revision 4 under "copy" would refuse a legitimate first save of
    // a workspace that does not exist yet.
    expect(studioProjectExpectedRevision({ name: "demo", revision: 4 }, "copy")).toBeNull();
  });

  it("is absent before anything is loaded", () => {
    expect(studioProjectExpectedRevision(null, "demo")).toBeNull();
  });

  it("comes back from a load response", () => {
    expect(studioProjectRevisionFromLoadResponse({ revision: 7, state: {} }, "demo")).toEqual({
      name: "demo",
      revision: 7,
    });
  });

  it("is null when the response carries no usable revision", () => {
    // A legacy payload, an error body, or a nonsense number: the editor then
    // saves as if the workspace were new, and the server decides.
    expect(studioProjectRevisionFromLoadResponse({ state: {} }, "demo")).toBeNull();
    expect(studioProjectRevisionFromLoadResponse({ revision: 0 }, "demo")).toBeNull();
    expect(studioProjectRevisionFromLoadResponse({ revision: 1.5 }, "demo")).toBeNull();
    expect(studioProjectRevisionFromLoadResponse("not a document", "demo")).toBeNull();
  });
});

describe("a refused save", () => {
  it("reports the conflict in words and says nothing was overwritten", () => {
    const conflict = new StudioRequestError(
      "the workspace moved to revision 5 while you were editing revision 3; reload and reapply your change.",
      409,
      { actual_revision: 5, error: "workspace_conflict", expected_revision: 3 },
    );

    expect(studioProjectSaveFailureState(conflict)).toEqual({
      error:
        "the workspace moved to revision 5 while you were editing revision 3; "
        + "reload and reapply your change. Nothing was overwritten.",
    });
  });

  it("tells the editor a busy workspace can simply be saved again", () => {
    const busy = new StudioRequestError(
      "another writer held workspace 'demo' for longer than 10.0 seconds; "
      + "nothing was written, so the same save can be retried.",
      503,
      { error: "workspace_busy", name: "demo", reason: "busy", timeout_seconds: 10 },
    );

    expect(studioProjectSaveFailureState(busy)).toEqual({
      error:
        "another writer held workspace 'demo' for longer than 10.0 seconds; "
        + "nothing was written, so the same save can be retried. "
        + "Nothing was written; save again.",
    });
  });

  it("falls back to the ordinary failure message for anything else", () => {
    expect(studioProjectSaveFailureState(new Error("network down"))).toEqual({
      error: "network down",
    });
    expect(studioProjectSaveFailureState("nonsense")).toEqual({ error: "Project save failed" });
  });
});

describe("the recoverable trash", () => {
  it("is carried into state as the server listed it", () => {
    const deleted = [{ deleted_at: 1782028800, name: "demo", token: "demo.1782028800000" }];

    expect(studioProjectDeletedListedState(deleted)).toEqual({ deletedProjects: deleted });
  });

  it("is empty rather than absent when nothing is deleted", () => {
    expect(studioProjectDeletedListedState([])).toEqual({ deletedProjects: [] });
  });
});
