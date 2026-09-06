// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Undo and redo for Network Canvas graph edits

/**
 * A deleted population comes back without losing everything since.
 *
 * Deleting a population removes it and every projection touching it. That is
 * correct, and it is also the edit a user most wants back; the only way back
 * used to be a saved workspace revision, which discards every change made
 * after it.
 *
 * These cases drive the live store, so they exercise the same actions the
 * canvas calls. A move is deliberately not an edit: dragging writes a position
 * on every frame, and recording those would bury the edits worth undoing.
 */

import { afterEach, describe, expect, it } from "vitest";

import type { PopulationNode, ProjectionEdge } from "./api/client";
import {
  STUDIO_GRAPH_HISTORY_LIMIT,
  emptyGraphHistory,
  graphEditRecorded,
  graphRedone,
  graphSnapshotOf,
  graphSnapshotsEqual,
  graphUndone,
  isLayoutOnlyUpdate,
} from "./studioGraphHistory";
import { studioGraphRequest } from "./studioGraphRequests";
import { useStudioStore } from "./stores/studio";

function population(id: string): PopulationNode {
  return {
    count: 4,
    drive: { kind: "none" },
    id,
    label: id.toUpperCase(),
    model: "SCLapicqueLIFNeuron",
    neuron_type: "excitatory",
    params: {},
    position: { x: 0, y: 0 },
    type: "population",
  };
}

function projection(id: string, source: string, target: string): ProjectionEdge {
  return { delay: 0, id, probability: 1, rule: "all_to_all", source, target, weight: 40 };
}

const POPULATIONS = [population("p1"), population("p2"), population("p3")];
const PROJECTIONS = [projection("e1", "p1", "p2"), projection("e2", "p2", "p3")];

describe("the history structure", () => {
  it("starts empty, so nothing can be stepped either way", () => {
    const history = emptyGraphHistory();
    const now = graphSnapshotOf(POPULATIONS, PROJECTIONS);

    expect(graphUndone(history, now)).toBeNull();
    expect(graphRedone(history, now)).toBeNull();
  });

  it("keeps the graph a step reinstates and the graph it stepped away from", () => {
    const before = graphSnapshotOf(POPULATIONS, PROJECTIONS);
    const after = graphSnapshotOf(POPULATIONS.slice(1), PROJECTIONS.slice(1));
    const history = graphEditRecorded(emptyGraphHistory(), before);

    const undone = graphUndone(history, after);

    expect(undone).not.toBeNull();
    expect(graphSnapshotsEqual(undone!.snapshot, before)).toBe(true);
    expect(graphSnapshotsEqual(graphRedone(undone!.history, before)!.snapshot, after)).toBe(true);
  });

  it("discards the redo branch when an edit follows an undo", () => {
    const first = graphSnapshotOf(POPULATIONS, PROJECTIONS);
    const second = graphSnapshotOf(POPULATIONS.slice(1), PROJECTIONS);
    const undone = graphUndone(graphEditRecorded(emptyGraphHistory(), first), second)!;
    expect(undone.history.future).toHaveLength(1);

    const edited = graphEditRecorded(undone.history, first);

    expect(edited.future).toEqual([]);
  });

  it("keeps a bounded number of edits", () => {
    let history = emptyGraphHistory();
    for (let index = 0; index < STUDIO_GRAPH_HISTORY_LIMIT + 10; index += 1) {
      history = graphEditRecorded(history, graphSnapshotOf([population(`p${index}`)], []));
    }

    expect(history.past).toHaveLength(STUDIO_GRAPH_HISTORY_LIMIT);
    expect(history.past[0].populations[0].id).toBe("p10");
  });

  it("copies the arrays it records", () => {
    const populations = [...POPULATIONS];
    const snapshot = graphSnapshotOf(populations, PROJECTIONS);

    populations.pop();

    expect(snapshot.populations).toHaveLength(3);
  });

  it("recognises a layout-only update, and nothing else", () => {
    expect(isLayoutOnlyUpdate({ position: { x: 1, y: 2 } })).toBe(true);
    expect(isLayoutOnlyUpdate({ count: 8 })).toBe(false);
    expect(isLayoutOnlyUpdate({ count: 8, position: { x: 1, y: 2 } })).toBe(false);
    expect(isLayoutOnlyUpdate({})).toBe(false);
  });
});

describe("undo through the live store", () => {
  const pristine = useStudioStore.getState();

  afterEach(() => {
    useStudioStore.setState(pristine, true);
  });

  function seed(): void {
    useStudioStore.setState({
      graphHistory: emptyGraphHistory(),
      graphPopulations: POPULATIONS,
      graphProjections: PROJECTIONS,
    });
  }

  it("brings back a deleted population and the projections it took with it", () => {
    seed();

    useStudioStore.getState().removePopulation("p2");
    expect(useStudioStore.getState().graphPopulations.map((p) => p.id)).toEqual(["p1", "p3"]);
    expect(useStudioStore.getState().graphProjections).toEqual([]);

    useStudioStore.getState().undoGraphEdit();

    const state = useStudioStore.getState();
    expect(state.graphPopulations.map((p) => p.id)).toEqual(["p1", "p2", "p3"]);
    expect(state.graphProjections.map((e) => e.id)).toEqual(["e1", "e2"]);
  });

  it("redoes the deletion it just undid", () => {
    seed();
    useStudioStore.getState().removePopulation("p2");
    useStudioStore.getState().undoGraphEdit();

    useStudioStore.getState().redoGraphEdit();

    expect(useStudioStore.getState().graphPopulations.map((p) => p.id)).toEqual(["p1", "p3"]);
  });

  it("steps back through several edits in order", () => {
    seed();
    useStudioStore.getState().removePopulation("p1");
    useStudioStore.getState().removePopulation("p3");
    expect(useStudioStore.getState().graphPopulations.map((p) => p.id)).toEqual(["p2"]);

    useStudioStore.getState().undoGraphEdit();
    expect(useStudioStore.getState().graphPopulations.map((p) => p.id)).toEqual(["p2", "p3"]);

    useStudioStore.getState().undoGraphEdit();
    expect(useStudioStore.getState().graphPopulations.map((p) => p.id)).toEqual(["p1", "p2", "p3"]);
  });

  it("does nothing when there is nothing to step", () => {
    seed();

    useStudioStore.getState().undoGraphEdit();
    useStudioStore.getState().redoGraphEdit();

    expect(useStudioStore.getState().graphPopulations.map((p) => p.id)).toEqual([
      "p1",
      "p2",
      "p3",
    ]);
  });

  it("does not record a drag, so undo reaches the last real edit", () => {
    seed();
    useStudioStore.getState().removePopulation("p3");

    useStudioStore.getState().updatePopulation("p1", { position: { x: 400, y: 400 } });
    useStudioStore.getState().updatePopulation("p1", { position: { x: 800, y: 800 } });
    useStudioStore.getState().undoGraphEdit();

    const state = useStudioStore.getState();
    expect(state.graphPopulations.map((p) => p.id)).toEqual(["p1", "p2", "p3"]);
    expect(state.graphHistory.past).toEqual([]);
  });

  it("records an edit that is not a move", () => {
    seed();

    useStudioStore.getState().updatePopulation("p1", { count: 99 });
    expect(useStudioStore.getState().graphPopulations[0].count).toBe(99);

    useStudioStore.getState().undoGraphEdit();

    expect(useStudioStore.getState().graphPopulations[0].count).toBe(4);
  });

  it("records a projection removal and brings it back", () => {
    seed();

    useStudioStore.getState().removeProjection("e1");
    expect(useStudioStore.getState().graphProjections.map((e) => e.id)).toEqual(["e2"]);

    useStudioStore.getState().undoGraphEdit();

    expect(useStudioStore.getState().graphProjections.map((e) => e.id)).toEqual(["e1", "e2"]);
  });

  it("records nothing for a removal that removed nothing", () => {
    seed();

    useStudioStore.getState().removePopulation("absent");
    useStudioStore.getState().removeProjection("absent");

    expect(useStudioStore.getState().graphHistory.past).toEqual([]);
  });

  it("leaves a runnable graph after an undo", () => {
    seed();
    useStudioStore.getState().removePopulation("p2");
    useStudioStore.getState().undoGraphEdit();

    const state = useStudioStore.getState();
    const request = studioGraphRequest(state.graphPopulations, state.graphProjections, 100, 0.1);
    const identifiers = new Set(request.populations.map((p) => p.id));

    for (const edge of request.projections) {
      expect(identifiers.has(edge.source)).toBe(true);
      expect(identifiers.has(edge.target)).toBe(true);
    }
  });
});
