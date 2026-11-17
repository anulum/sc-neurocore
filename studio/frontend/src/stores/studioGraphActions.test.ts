// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio graph-editing actions against a mocked server

import { createStore } from "zustand/vanilla";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PopulationNode, ProjectionEdge } from "../api/client";
import {
  createPopulation,
  createProjection,
  exportNIR,
  fetchNetworkNotebook,
  graphModelContract,
  importNIR,
  simulateGraph,
  validateGraph,
} from "../api/client";
import type { GraphSimResult, NIRExportResult, NIRImportResult, PopulationModelContract } from "../api/types";
import { downloadBrowserArtefact } from "../browserArtefactDownload";
import { networkNirExportPlan, networkNirImportRequest } from "../networkNirExport";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    createPopulation: vi.fn(),
    createProjection: vi.fn(),
    exportNIR: vi.fn(),
    fetchNetworkNotebook: vi.fn(),
    graphModelContract: vi.fn(),
    importNIR: vi.fn(),
    simulateGraph: vi.fn(),
    validateGraph: vi.fn(),
  };
});
vi.mock("../browserArtefactDownload", () => ({ downloadBrowserArtefact: vi.fn() }));
vi.mock("../networkNirExport", async () => {
  const actual = await vi.importActual<typeof import("../networkNirExport")>("../networkNirExport");
  return { ...actual, networkNirExportPlan: vi.fn(), networkNirImportRequest: vi.fn() };
});

afterEach(() => {
  vi.resetAllMocks();
});

/** A store with the real actions over the initial data. */
function studioStore() {
  return createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
}

/** A population as the server returns it. */
function population(id: string, neuronType: "excitatory" | "inhibitory" = "excitatory"): PopulationNode {
  return {
    id,
    type: "population",
    label: id,
    model: "LIF",
    count: 10,
    neuron_type: neuronType,
    position: { x: 0, y: 0 },
    params: {},
  };
}

/** A projection as the server returns it. */
function projection(id: string, source: string, target: string, extra: Partial<ProjectionEdge> = {}): ProjectionEdge {
  return { id, source, target, weight: 0.5, delay: 0, ...extra };
}

const VALID = { valid: true, errors: [], issues: [] };

describe("projections", () => {
  it("refuses a projection whose source is not in the graph, without asking the server", async () => {
    const store = studioStore();
    await store.getState().addProjection("missing", "p1");
    expect(createProjection).not.toHaveBeenCalled();
    expect(store.getState().error).toBe("Source population missing not found");
  });

  it("adds the projection the server creates and records the edit for undo", async () => {
    const store = studioStore();
    store.setState({ graphPopulations: [population("p0"), population("p1")] });
    vi.mocked(createProjection).mockResolvedValue(projection("e0", "p0", "p1"));
    await store.getState().addProjection("p0", "p1");
    expect(vi.mocked(createProjection).mock.calls[0]?.[0]).toMatchObject({ source_id: "p0", target_id: "p1" });
    expect(store.getState().graphProjections.map((edge) => edge.id)).toEqual(["e0"]);
    store.getState().undoGraphEdit();
    expect(store.getState().graphProjections).toEqual([]);
  });

  it("reports a projection the server refuses", async () => {
    const store = studioStore();
    store.setState({ graphPopulations: [population("p0"), population("p1")] });
    vi.mocked(createProjection).mockRejectedValue(new Error("weight sign disagrees"));
    await store.getState().addProjection("p0", "p1");
    expect(store.getState().error).toBe("weight sign disagrees");
    expect(store.getState().graphProjections).toEqual([]);
  });
});

describe("populations", () => {
  it("keeps the projection being edited when the deleted population is not one of its ends", () => {
    const store = studioStore();
    store.setState({
      graphPopulations: [population("p0"), population("p1"), population("p2")],
      graphProjections: [projection("e0", "p0", "p1")],
      selectedProjectionId: "e0",
    });
    store.getState().removePopulation("p2");
    expect(store.getState().graphPopulations.map((node) => node.id)).toEqual(["p0", "p1"]);
    expect(store.getState().selectedProjectionId).toBe("e0");
  });

  it("selects nothing and loads no contract when the selection is cleared", () => {
    const store = studioStore();
    store.setState({ graphPopulations: [population("p0")], selectedProjectionId: "e0" });
    store.getState().selectPopulation(null);
    expect(store.getState().selectedPopulationId).toBeNull();
    expect(store.getState().selectedProjectionId).toBeNull();
    expect(graphModelContract).not.toHaveBeenCalled();
  });

  it("loads no contract for an id the graph does not hold", () => {
    const store = studioStore();
    store.getState().selectPopulation("gone");
    expect(store.getState().selectedPopulationId).toBe("gone");
    expect(graphModelContract).not.toHaveBeenCalled();
  });

  it("does not ask again for the contract it already holds", async () => {
    const store = studioStore();
    store.setState({ populationModelContract: { model: "LIF" } as unknown as PopulationModelContract });
    await store.getState().loadPopulationModelContract("LIF");
    expect(graphModelContract).not.toHaveBeenCalled();
  });

  it("offers no parameters, and says why, when a model's contract cannot be loaded", async () => {
    const store = studioStore();
    store.setState({ populationModelContract: { model: "Izhikevich" } as unknown as PopulationModelContract });
    vi.mocked(graphModelContract).mockRejectedValue(new Error(""));
    await store.getState().loadPopulationModelContract("LIF");
    expect(store.getState().populationModelContract).toBeNull();
    expect(store.getState().error).toBe("Model contract for LIF could not be loaded");
  });

  it("copies a projection's seed and autapses, which the creation route does not carry", async () => {
    const store = studioStore();
    store.setState({
      graphPopulations: [population("p0"), population("p1")],
      graphProjections: [projection("e0", "p0", "p1", { seed: 7, autapses: true })],
      selectedPopulationIds: ["p0", "p1"],
    });
    let made = 0;
    vi.mocked(createPopulation).mockImplementation(() => Promise.resolve(population(`c${String(made++)}`)));
    vi.mocked(createProjection).mockResolvedValue(projection("e1", "c0", "c1"));
    await store.getState().duplicateSelection();
    const copied = store.getState().graphProjections.find((edge) => edge.id === "e1");
    expect(copied).toMatchObject({ seed: 7, autapses: true });
  });
});

describe("validation and simulation", () => {
  it("reports a validation request that fails", async () => {
    const store = studioStore();
    vi.mocked(validateGraph).mockRejectedValue(new Error("server unreachable"));
    await store.getState().validateGraphAction();
    expect(store.getState().error).toBe("server unreachable");
  });

  it("starts no second simulation while one is running", async () => {
    const store = studioStore();
    store.setState({ isSimulating: true });
    await store.getState().simulateGraphAction();
    expect(validateGraph).not.toHaveBeenCalled();
  });

  it("does not simulate a graph the server finds invalid, and locates its errors", async () => {
    const store = studioStore();
    vi.mocked(validateGraph).mockResolvedValue({ valid: false, errors: ["no populations"], issues: [] });
    await store.getState().simulateGraphAction();
    expect(simulateGraph).not.toHaveBeenCalled();
    expect(store.getState().graphErrors).toEqual(["no populations"]);
    expect(store.getState().isSimulating).toBe(false);
  });

  it("reports a simulation the server fails, and is no longer busy", async () => {
    const store = studioStore();
    vi.mocked(validateGraph).mockResolvedValue(VALID);
    vi.mocked(simulateGraph).mockRejectedValue(new Error("worker crashed"));
    await store.getState().simulateGraphAction();
    expect(store.getState().error).toBe("worker crashed");
    expect(store.getState().isSimulating).toBe(false);
  });

  it("runs a valid graph and keeps the server's result", async () => {
    const store = studioStore();
    const result = { success: true } as GraphSimResult;
    vi.mocked(validateGraph).mockResolvedValue(VALID);
    vi.mocked(simulateGraph).mockResolvedValue(result);
    await store.getState().simulateGraphAction();
    expect(store.getState().graphSimResult).toBe(result);
    expect(store.getState().isSimulating).toBe(false);
  });
});

describe("files", () => {
  it("writes the NIR file the server exports and states its notes", async () => {
    const store = studioStore();
    const writeArtefact = vi.fn();
    vi.mocked(exportNIR).mockResolvedValue({ notes: [] } as unknown as NIRExportResult);
    vi.mocked(networkNirExportPlan).mockReturnValue({ writeArtefact } as unknown as ReturnType<typeof networkNirExportPlan>);
    await store.getState().exportGraphNIR();
    expect(writeArtefact).toHaveBeenCalledOnce();
    expect(store.getState().graphNotice).not.toBeNull();
  });

  it("reports an NIR export the server refuses", async () => {
    const store = studioStore();
    vi.mocked(exportNIR).mockRejectedValue(new Error("no NIR for a delay"));
    await store.getState().exportGraphNIR();
    expect(store.getState().error).toBe("no NIR for a delay");
  });

  it("downloads the network notebook the server writes", async () => {
    const store = studioStore();
    vi.mocked(fetchNetworkNotebook).mockResolvedValue({ cells: [] });
    await store.getState().exportGraphNotebook();
    expect(vi.mocked(downloadBrowserArtefact).mock.calls[0]?.[1]).toBe("network-tutorial.ipynb");
    expect(store.getState().graphNotice).toMatch(/^Notebook written/);
  });

  it("reports a notebook the server cannot write", async () => {
    const store = studioStore();
    vi.mocked(fetchNetworkNotebook).mockRejectedValue(new Error("notebook failed"));
    await store.getState().exportGraphNotebook();
    expect(store.getState().error).toBe("notebook failed");
    expect(downloadBrowserArtefact).not.toHaveBeenCalled();
  });

  it("replaces the graph with the one an NIR file holds", async () => {
    const store = studioStore();
    const imported = {
      graph: { populations: [population("n0")], projections: [] },
      origin: "foreign",
      notes: [],
    } as NIRImportResult;
    vi.mocked(networkNirImportRequest).mockResolvedValue({} as Awaited<ReturnType<typeof networkNirImportRequest>>);
    vi.mocked(importNIR).mockResolvedValue(imported);
    await store.getState().importGraphNIR(new File(["{}"], "net.nir"));
    expect(store.getState().graphPopulations.map((node) => node.id)).toEqual(["n0"]);
    expect(store.getState().graphNotice).not.toBeNull();
  });

  it("reports an NIR file the server cannot read", async () => {
    const store = studioStore();
    vi.mocked(networkNirImportRequest).mockRejectedValue(new Error("not an NIR file"));
    await store.getState().importGraphNIR(new File(["x"], "bad.nir"));
    expect(store.getState().error).toBe("not an NIR file");
  });
});
