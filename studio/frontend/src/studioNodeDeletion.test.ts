// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Deleting a population removes it from the graph, not the view

/**
 * Deleting a node on the canvas used to change nothing.
 *
 * `onNodesChange` computed the new node array from the change batch and then
 * wrote back positions only, so a `remove` was computed and discarded: the
 * population and every projection touching it stayed in the graph and
 * reappeared on the next render. `removePopulation` existed in the store, with
 * a type entry and a correct implementation, and was called from nowhere in the
 * frontend.
 *
 * These cases drive the real change batches through the canvas library's own
 * `applyNodeChanges`, then through the store's graph reducer, and finally
 * through the request the server would receive — so a deleted node cannot come
 * back and cannot be referenced by a run.
 */

import { applyNodeChanges, type NodeChange } from "@xyflow/react";
import { afterEach, describe, expect, it } from "vitest";

import type { PopulationNode, ProjectionEdge } from "./api/client";
import {
  studioGraphRequest,
  studioGraphWithoutPopulation,
  studioNodeChangePlan,
} from "./studioGraphRequests";
import { useStudioStore } from "./stores/studio";

/** One population at a given position, so a move can be told from an edit. */
function population(id: string, x: number, y: number): PopulationNode {
  return {
    count: 4,
    drive: { kind: "none" },
    id,
    label: id.toUpperCase(),
    model: "SCLapicqueLIFNeuron",
    neuron_type: "excitatory",
    params: {},
    position: { x, y },
    type: "population",
  };
}

/** One projection between two named populations. */
function projection(id: string, source: string, target: string): ProjectionEdge {
  return {
    autapses: false,
    delay: 0,
    id,
    probability: 1,
    rule: "all_to_all",
    source,
    target,
    weight: 40,
  };
}

const POPULATIONS = [population("p1", 0, 0), population("p2", 200, 0), population("p3", 400, 0)];
const PROJECTIONS = [
  projection("e1", "p1", "p2"),
  projection("e2", "p2", "p3"),
  projection("e3", "p3", "p1"),
];

/** The canvas nodes as the component derives them from the graph. */
const NODES = POPULATIONS.map((p) => ({ id: p.id, position: p.position, data: {} }));

/**
 * Run a change batch through the canvas library's own `applyNodeChanges`.
 *
 * The plan has to be derived from what the library actually produces, not from
 * a batch hand-written to look like it.
 */
function planFor(changes: NodeChange[]): ReturnType<typeof studioNodeChangePlan> {
  return studioNodeChangePlan(changes, applyNodeChanges(changes, NODES), POPULATIONS);
}

describe("reading a canvas change batch", () => {
  it("reports a removal that used to be computed and dropped", () => {
    const plan = planFor([{ id: "p2", type: "remove" }]);

    expect(plan.removed).toEqual(["p2"]);
    expect(plan.moved).toEqual([]);
  });

  it("reports a drag as a move and never as a removal", () => {
    const plan = planFor([
      { id: "p1", type: "position", position: { x: 10, y: 20 }, dragging: false },
    ]);

    expect(plan.removed).toEqual([]);
    expect(plan.moved).toEqual([{ id: "p1", position: { x: 10, y: 20 } }]);
  });

  it("writes nothing when a render reasserts the same layout", () => {
    const plan = planFor([
      { id: "p1", type: "position", position: { x: 0, y: 0 }, dragging: false },
    ]);

    expect(plan).toEqual({ moved: [], removed: [] });
  });

  it("does not report a position for a node in the same batch that removes it", () => {
    const plan = planFor([
      { id: "p3", type: "position", position: { x: 999, y: 999 }, dragging: false },
      { id: "p3", type: "remove" },
    ]);

    expect(plan.removed).toEqual(["p3"]);
    expect(plan.moved).toEqual([]);
  });

  it("reads several removals in one batch", () => {
    const plan = planFor([
      { id: "p1", type: "remove" },
      { id: "p3", type: "remove" },
    ]);

    expect(plan.removed).toEqual(["p1", "p3"]);
  });

  it("ignores selection and dimension changes", () => {
    const plan = planFor([
      { id: "p1", type: "select", selected: true },
      { id: "p2", type: "dimensions", dimensions: { width: 10, height: 10 } },
    ]);

    expect(plan).toEqual({ moved: [], removed: [] });
  });
});

describe("what the graph looks like afterwards", () => {
  it("drops the population and every projection that touched it", () => {
    const plan = planFor([{ id: "p2", type: "remove" }]);

    const graph = studioGraphWithoutPopulation(
      { populations: POPULATIONS, projections: PROJECTIONS },
      plan.removed[0],
    );

    expect(graph.populations.map((p) => p.id)).toEqual(["p1", "p3"]);
    expect(graph.projections.map((e) => e.id)).toEqual(["e3"]);
  });

  it("leaves no stale reference for a run to resolve", () => {
    const graph = studioGraphWithoutPopulation(
      { populations: POPULATIONS, projections: PROJECTIONS },
      "p2",
    );

    const request = studioGraphRequest(graph.populations, graph.projections, 100, 0.1);
    const identifiers = new Set(request.populations.map((p) => p.id));

    expect(identifiers.has("p2")).toBe(false);
    for (const edge of request.projections) {
      expect(identifiers.has(edge.source)).toBe(true);
      expect(identifiers.has(edge.target)).toBe(true);
    }
  });

  it("keeps the graph unchanged when only the layout moved", () => {
    const before = studioGraphRequest(POPULATIONS, PROJECTIONS, 100, 0.1);
    const moved = POPULATIONS.map((p) =>
      p.id === "p1" ? { ...p, position: { x: 77, y: 88 } } : p,
    );

    const after = studioGraphRequest(moved, PROJECTIONS, 100, 0.1);

    expect(after.populations.map((p) => p.id)).toEqual(before.populations.map((p) => p.id));
    expect(after.projections).toEqual(before.projections);
    expect(after.populations.find((p) => p.id === "p1")?.position).toEqual({ x: 77, y: 88 });
  });
});

describe("the store action the canvas now calls", () => {
  const pristine = useStudioStore.getState();

  afterEach(() => {
    useStudioStore.setState(pristine, true);
  });

  it("removes the population and its incident projections from the live store", () => {
    useStudioStore.setState({
      graphPopulations: POPULATIONS,
      graphProjections: PROJECTIONS,
    });

    useStudioStore.getState().removePopulation("p2");

    const state = useStudioStore.getState();
    expect(state.graphPopulations.map((p) => p.id)).toEqual(["p1", "p3"]);
    expect(state.graphProjections.map((e) => e.id)).toEqual(["e3"]);
  });

  it("a deleted population does not come back on the next read", () => {
    useStudioStore.setState({
      graphPopulations: POPULATIONS,
      graphProjections: PROJECTIONS,
    });

    for (const id of planFor([{ id: "p1", type: "remove" }]).removed) {
      useStudioStore.getState().removePopulation(id);
    }

    const request = studioGraphRequest(
      useStudioStore.getState().graphPopulations,
      useStudioStore.getState().graphProjections,
      100,
      0.1,
    );
    expect(request.populations.map((p) => p.id)).toEqual(["p2", "p3"]);
    expect(request.projections.map((e) => e.id)).toEqual(["e2"]);
  });

  it("removing an identifier the graph does not have changes nothing", () => {
    useStudioStore.setState({
      graphPopulations: POPULATIONS,
      graphProjections: PROJECTIONS,
    });

    useStudioStore.getState().removePopulation("absent");

    const state = useStudioStore.getState();
    expect(state.graphPopulations).toHaveLength(3);
    expect(state.graphProjections).toHaveLength(3);
  });
});
