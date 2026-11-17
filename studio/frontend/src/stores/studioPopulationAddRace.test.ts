// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio population adds made before the server answers

import { createStore } from "zustand/vanilla";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PopulationNode } from "../api/client";
import { createPopulation } from "../api/client";
import type { PopulationCreateRequest } from "../api/types";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, createPopulation: vi.fn() };
});

afterEach(() => {
  vi.mocked(createPopulation).mockReset();
});

/** A store with the real actions over the initial data. */
function studioStore() {
  return createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
}

/** The node the server would return for a request, under the given id. */
function created(id: string, request: PopulationCreateRequest): PopulationNode {
  return {
    id,
    type: "population",
    label: request.label,
    model: request.model,
    count: request.count,
    neuron_type: request.neuron_type,
    position: { x: request.x, y: request.y },
    params: {},
    drive: request.drive,
  };
}

/**
 * A server whose answers the test releases one at a time, in order.
 *
 * Each request is recorded when it is sent; `answer` resolves the oldest
 * unanswered one, and `refuse` rejects it.
 */
function heldServer() {
  interface Held {
    request: PopulationCreateRequest;
    resolve: (node: PopulationNode) => void;
    reject: (error: Error) => void;
  }
  const sent: PopulationCreateRequest[] = [];
  const pending: Held[] = [];
  vi.mocked(createPopulation).mockImplementation((request: PopulationCreateRequest) => {
    sent.push(request);
    return new Promise<PopulationNode>((resolve, reject) => { pending.push({ request, resolve, reject }); });
  });
  /** The oldest request still waiting for its answer. */
  const oldest = (): Held => {
    const held = pending.shift();
    if (held === undefined) throw new Error("no request is waiting for an answer");
    return held;
  };
  return {
    sent,
    answer(id: string) { const held = oldest(); held.resolve(created(id, held.request)); },
    refuse() { oldest().reject(new Error("refused")); },
  };
}

describe("Studio population adds", () => {
  it("gives two adds made before the first answer their own labels and positions", async () => {
    const store = studioStore();
    const server = heldServer();
    const first = store.getState().addPopulation("excitatory");
    const second = store.getState().addPopulation("excitatory");
    await vi.waitFor(() => { expect(server.sent).toHaveLength(1); });
    server.answer("p0");
    await vi.waitFor(() => { expect(server.sent).toHaveLength(2); });
    server.answer("p1");
    await Promise.all([first, second]);
    expect(server.sent.map((request) => request.label)).toEqual(["Exc 0", "Exc 1"]);
    expect(server.sent.map((request) => request.x)).toEqual([100, 300]);
    expect(store.getState().graphPopulations.map((population) => population.label)).toEqual(["Exc 0", "Exc 1"]);
  });

  it("does not spend a label on an add the server refuses", async () => {
    const store = studioStore();
    const server = heldServer();
    const refused = store.getState().addPopulation("inhibitory");
    const next = store.getState().addPopulation("inhibitory");
    await vi.waitFor(() => { expect(server.sent).toHaveLength(1); });
    server.refuse();
    await vi.waitFor(() => { expect(server.sent).toHaveLength(2); });
    server.answer("p0");
    await Promise.all([refused, next]);
    expect(server.sent.map((request) => request.label)).toEqual(["Inh 0", "Inh 0"]);
    expect(store.getState().graphPopulations.map((population) => population.label)).toEqual(["Inh 0"]);
  });
});
