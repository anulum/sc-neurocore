// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio graph request builder tests

import { describe, expect, it } from "vitest";

import type { GraphSimResult, NIRFormat, PipelineResult, PopulationNode, ProjectionEdge } from "./api/client";
import {
  STUDIO_DEFAULT_EXCITATORY_DRIVE,
  STUDIO_DEFAULT_POPULATION_MODEL,
  STUDIO_DEFAULT_PROJECTION_WEIGHT,
  studioGraphFailureState,
  studioGraphImportedState,
  studioGraphModelsLoadedState,
  studioDefaultPopulationRequest,
  studioDefaultProjectionRequest,
  studioGraphSimulationCompletedState,
  studioGraphSimulationStartState,
  studioGraphRequest,
  studioGraphWithoutPopulation,
  studioPipelineCompletedState,
  studioPipelineStartState,
  studioPopulationAddedState,
  studioPopulationDriveLabel,
  studioProjectionLabel,
  studioPopulationUpdatedState,
  studioProjectionAddedState,
  studioProjectionRemovedState,
  studioProjectionUpdatedState,
} from "./studioGraphRequests";

const population: PopulationNode = {
  id: "p1",
  type: "population",
  label: "Exc",
  model: "LIFNeuron",
  count: 80,
  neuron_type: "excitatory",
  position: { x: 100, y: 100 },
  params: {},
  drive: { kind: "constant", current: 1.2 },
};

const projection: ProjectionEdge = {
  id: "e1",
  source: "p1",
  target: "p2",
  weight: 40,
  delay: 1,
  rule: "random",
  probability: 0.2,
};

describe("Studio graph request builders", () => {
  it("builds the graph simulation request from current canvas state", () => {
    expect(studioGraphRequest([population], [projection], 250, 0.05)).toEqual({
      populations: [population],
      projections: [projection],
      duration: 250,
      dt: 0.05,
    });
    expect(studioGraphRequest([population], [projection], 250, 0.05, 7)).toEqual({
      populations: [population],
      projections: [projection],
      duration: 250,
      dt: 0.05,
      seed: 7,
    });
  });

  it("builds default population create requests on a catalogue model with an explicit drive", () => {
    expect(STUDIO_DEFAULT_POPULATION_MODEL).toBe("SCLapicqueLIFNeuron");
    expect(studioDefaultPopulationRequest("excitatory", 0)).toEqual({
      label: "Exc 0",
      model: "SCLapicqueLIFNeuron",
      count: 80,
      neuron_type: "excitatory",
      x: 100,
      y: 100,
      drive: STUDIO_DEFAULT_EXCITATORY_DRIVE,
    });
    expect(studioDefaultPopulationRequest("inhibitory", 2)).toEqual({
      label: "Inh 2",
      model: "SCLapicqueLIFNeuron",
      count: 20,
      neuron_type: "inhibitory",
      x: 500,
      y: 300,
      drive: { kind: "none" },
    });
  });

  it("signs the default projection weight by the source population type", () => {
    expect(studioDefaultProjectionRequest("p1", "p2", "excitatory")).toEqual({
      source_id: "p1",
      target_id: "p2",
      weight: STUDIO_DEFAULT_PROJECTION_WEIGHT,
      delay: 0,
      rule: "random",
      probability: 0.2,
    });
    expect(studioDefaultProjectionRequest("p2", "p1", "inhibitory").weight)
      .toBe(-STUDIO_DEFAULT_PROJECTION_WEIGHT);
  });

  it("labels drives and projections with their executed semantics", () => {
    expect(studioPopulationDriveLabel(undefined)).toBe("no input");
    expect(studioPopulationDriveLabel({ kind: "none" })).toBe("no input");
    expect(studioPopulationDriveLabel({ kind: "constant", current: 1.2 })).toBe("I = 1.2");
    expect(studioPopulationDriveLabel({ kind: "poisson", rate_hz: 50, weight: 2 }))
      .toBe("Poisson 50 Hz × 2");
    expect(studioProjectionLabel(projection)).toBe("w=40 p=0.2 d=1ms");
    expect(studioProjectionLabel({ ...projection, rule: "all_to_all", probability: undefined, delay: 0 }))
      .toBe("w=40 all");
  });

  it("removes a population and all incident projections from local graph state", () => {
    const unrelatedProjection = { ...projection, id: "e2", source: "p3", target: "p4" };

    expect(studioGraphWithoutPopulation({
      populations: [population, { ...population, id: "p2" }],
      projections: [projection, unrelatedProjection],
    }, "p1")).toEqual({
      populations: [{ ...population, id: "p2" }],
      projections: [unrelatedProjection],
    });
  });

  it("builds pipeline and graph simulation lifecycle patches", () => {
    const pipelineResult: PipelineResult = { success: true, target: "ice40" };
    const simResult: GraphSimResult = { n_spikes: 10, success: true };

    expect(studioPipelineStartState()).toEqual({
      error: null,
      isSimulating: true,
      pipelineResult: null,
    });
    expect(studioPipelineCompletedState(pipelineResult)).toEqual({
      isSimulating: false,
      pipelineResult,
    });
    expect(studioGraphSimulationStartState()).toEqual({
      error: null,
      graphErrors: [],
      graphIssues: [],
      isSimulating: true,
    });
    expect(studioGraphSimulationCompletedState(simResult)).toEqual({
      graphSimResult: simResult,
      isSimulating: false,
    });
  });

  it("builds graph model and element mutation patches", () => {
    const replacement = { ...population, label: "Renamed" };
    const updatedProjection = { ...projection, weight: 0.3 };
    const retainedProjection = { ...projection, id: "e2", weight: 0.5 };

    expect(studioGraphModelsLoadedState(["LIFNeuron"])).toEqual({ graphModels: ["LIFNeuron"] });
    expect(studioPopulationAddedState([], population)).toEqual({ graphPopulations: [population] });
    expect(studioPopulationUpdatedState([population], population.id, { label: "Renamed" }))
      .toEqual({ graphPopulations: [replacement] });
    expect(studioProjectionAddedState([], projection)).toEqual({ graphProjections: [projection] });
    expect(studioProjectionUpdatedState([projection], projection.id, { weight: 0.3 }))
      .toEqual({ graphProjections: [updatedProjection] });
    expect(studioProjectionRemovedState([projection, retainedProjection], projection.id))
      .toEqual({ graphProjections: [retainedProjection] });
  });

  it("builds graph import and failure patches", () => {
    const imported = {
      populations: [population],
      projections: [projection],
    };
    const nir: NIRFormat = {
      edges: [],
      format: "nir",
      nodes: {},
      version: "1.0",
    };

    expect(studioGraphImportedState(imported)).toEqual({
      activeTab: "canvas",
      graphPopulations: [population],
      graphProjections: [projection],
    });
    expect(nir.format).toBe("nir");
    expect(studioGraphFailureState(new Error("graph offline"), "fallback")).toEqual({
      error: "graph offline",
    });
    expect(studioGraphFailureState("bad", "fallback", { clearBusy: true })).toEqual({
      error: "fallback",
      isSimulating: false,
    });
  });
});
