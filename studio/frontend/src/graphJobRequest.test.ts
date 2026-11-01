// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scientific graph edit fencing and exact retry snapshots

import { expect, it } from "vitest";
import { at } from "./arrayAt";
import type { NetworkGraph } from "./api/types";
import { graphJobInputKey, snapshotGraphJobRequest } from "./graphJobRequest";

/** Supply ordinary UI-shaped inputs without supplying a numerical result oracle. */
function graph(): NetworkGraph {
  return {
    populations: [
      { id: "e", type: "population", label: "Exc", model: "SCLapicqueLIFNeuron", count: 2,
        neuron_type: "excitatory", position: { x: 10, y: 20 }, params: { tau: 10 },
        drive: { kind: "constant", current: 2 } },
      { id: "i", type: "population", label: "Inh", model: "SCLapicqueLIFNeuron", count: 1,
        neuron_type: "inhibitory", position: { x: 30, y: 40 }, params: {},
        drive: { kind: "poisson", rate_hz: 100, weight: 1, seed: 9 } },
    ],
    projections: [{ id: "ei", source: "e", target: "i", weight: 40, delay: 0,
      rule: "random", probability: 0.25, seed: 123, autapses: false }],
    duration: 20, dt: 0.1, seed: 42,
  };
}

it("captures immutable wire JSON before later edits reach any nested input", () => {
  const request = graph();
  const snapshot = snapshotGraphJobRequest(request);
  const bytes = snapshot.json;
  at(request.populations, 0).params.tau = 11;
  at(request.populations, 0).drive = { kind: "constant", current: 3 };
  at(request.populations, 0).position.x = 200;
  at(request.projections, 0).weight = 41;
  expect(snapshot.json).toBe(bytes);
  expect(snapshot.json).toContain('"tau":10');
  expect(snapshot.inputKey).not.toBe(graphJobInputKey(request));
});

it("preserves scientific identity across moves while retaining the actual submitted layout bytes", () => {
  const request = graph();
  const before = snapshotGraphJobRequest(request);
  at(request.populations, 0).position = { x: 900, y: -50 };
  const moved = snapshotGraphJobRequest(request);
  expect(moved.inputKey).toBe(before.inputKey);
  expect(moved.json).not.toBe(before.json);
});

it("does not invent defaults for legacy omitted optional fields", () => {
  const request = graph();
  request.populations.forEach((population) => { delete population.drive; });
  const projection = at(request.projections, 0);
  delete projection.rule;
  delete projection.seed;
  delete projection.autapses;
  delete projection.probability;
  delete request.duration;
  delete request.dt;
  delete request.seed;
  const key = graphJobInputKey(request);
  for (const field of ["drive", "rule", "probability", "autapses", "seed", "duration", "dt"]) {
    expect(key).not.toContain(`"${field}":`);
  }
});

it.each([
  { field: "label", change: (input: NetworkGraph) => { at(input.populations, 0).label = "Edited"; } },
  { field: "model", change: (input: NetworkGraph) => { at(input.populations, 0).model = "OtherModel"; } },
  { field: "count", change: (input: NetworkGraph) => { at(input.populations, 0).count = 3; } },
  { field: "neuron type", change: (input: NetworkGraph) => { at(input.populations, 0).neuron_type = "inhibitory"; } },
  { field: "parameters", change: (input: NetworkGraph) => { at(input.populations, 0).params = { tau: 11 }; } },
  { field: "drive", change: (input: NetworkGraph) => { at(input.populations, 0).drive = { kind: "none" }; } },
  { field: "weight", change: (input: NetworkGraph) => { at(input.projections, 0).weight = 41; } },
  { field: "delay", change: (input: NetworkGraph) => { at(input.projections, 0).delay = 0.2; } },
  { field: "rule", change: (input: NetworkGraph) => { at(input.projections, 0).rule = "all_to_all"; } },
  { field: "probability", change: (input: NetworkGraph) => { at(input.projections, 0).probability = 0.5; } },
  { field: "projection seed", change: (input: NetworkGraph) => { at(input.projections, 0).seed = 124; } },
  { field: "autapses", change: (input: NetworkGraph) => { at(input.projections, 0).autapses = true; } },
  { field: "source", change: (input: NetworkGraph) => { at(input.projections, 0).source = "i"; } },
  { field: "target", change: (input: NetworkGraph) => { at(input.projections, 0).target = "e"; } },
  { field: "duration", change: (input: NetworkGraph) => { input.duration = 21; } },
  { field: "dt", change: (input: NetworkGraph) => { input.dt = 0.2; } },
  { field: "graph seed", change: (input: NetworkGraph) => { input.seed = 43; } },
])("withdraws the local edit fence after changing $field", ({ change }) => {
  const input = graph();
  const before = graphJobInputKey(input);
  change(input);
  expect(graphJobInputKey(input)).not.toBe(before);
});

it("refuses non-finite scientific input before JSON could turn it into null", () => {
  const input = graph();
  at(input.populations, 0).params.tau = Infinity;
  expect(() => snapshotGraphJobRequest(input)).toThrow();
});
