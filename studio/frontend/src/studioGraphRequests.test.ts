// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio graph request builder tests

import { describe, expect, it } from "vitest";

import type { PopulationNode, ProjectionEdge } from "./api/client";
import {
  studioDefaultPopulationRequest,
  studioDefaultProjectionRequest,
  studioGraphRequest,
  studioGraphWithoutPopulation,
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
};

const projection: ProjectionEdge = {
  id: "e1",
  source: "p1",
  target: "p2",
  weight: 0.1,
  delay: 1,
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
  });

  it("builds default excitatory and inhibitory population create requests", () => {
    expect(studioDefaultPopulationRequest("excitatory", 0)).toEqual({
      label: "Exc 0",
      model: "LIFNeuron",
      count: 80,
      neuron_type: "excitatory",
      x: 100,
      y: 100,
    });
    expect(studioDefaultPopulationRequest("inhibitory", 2)).toEqual({
      label: "Inh 2",
      model: "LIFNeuron",
      count: 20,
      neuron_type: "inhibitory",
      x: 500,
      y: 300,
    });
  });

  it("builds the default projection create request", () => {
    expect(studioDefaultProjectionRequest("p1", "p2")).toEqual({
      source_id: "p1",
      target_id: "p2",
      weight: 0.1,
      probability: 0.2,
    });
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
});
