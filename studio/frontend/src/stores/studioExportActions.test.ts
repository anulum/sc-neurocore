// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio experiment exports: what each sends, what it downloads, and when it waits for the model

import { createStore } from "zustand/vanilla";
import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchCodegen, fetchReplayNotebook, fetchReplayPack } from "../api/client";
import { downloadBrowserArtefact } from "../browserArtefactDownload";
import { replayPackExport } from "../simulationExports";
import { studioExperimentExportRequest } from "../studioSimulationConfig";
import { simulationConfigInput } from "./studioHeavyAnalysis";
import { studioInitialData } from "./studioInitialState";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioStoreActions } from "./studioStoreActions";
import { exportStudioTrace } from "./studioTraceActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, fetchCodegen: vi.fn(), fetchReplayNotebook: vi.fn(), fetchReplayPack: vi.fn() };
});
vi.mock("../browserArtefactDownload", async () => {
  const actual = await vi.importActual<typeof import("../browserArtefactDownload")>("../browserArtefactDownload");
  return { ...actual, downloadBrowserArtefact: vi.fn() };
});
vi.mock("../simulationExports", async () => {
  const actual = await vi.importActual<typeof import("../simulationExports")>("../simulationExports");
  return { ...actual, replayPackExport: vi.fn() };
});
vi.mock("./studioTraceActions", async () => {
  const actual = await vi.importActual<typeof import("./studioTraceActions")>("./studioTraceActions");
  return { ...actual, exportStudioTrace: vi.fn() };
});

afterEach(() => {
  vi.resetAllMocks();
});

const LIF = { name: "LIF" } as unknown as StudioState["modelDetail"];

/** A store with every real action over the initial data. */
function exportStore(overrides: Partial<StudioState> = {}) {
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
  store.setState(overrides);
  return store;
}

/** The export request the store's current inputs describe. */
const requestOf = (state: StudioState) => studioExperimentExportRequest(simulationConfigInput(state));

/** The three server exports, each with the call that starts it and the server call it makes. */
const SERVER_EXPORTS = [
  ["the Python script", (s: StudioState) => s.runCodegen(), fetchCodegen],
  ["the replay pack", (s: StudioState) => s.exportReplayPack(), fetchReplayPack],
  ["the replay notebook", (s: StudioState) => s.exportReplayNotebook(), fetchReplayNotebook],
] as const;

describe("exports that go to the server", () => {
  it.each(SERVER_EXPORTS)("%s waits until the selected model's details have arrived", async (_, start, call) => {
    const store = exportStore({ sourceMode: "model", selectedModelName: "LIF", modelDetail: null });
    await start(store.getState());
    expect(store.getState().error).toBe("Selected model is still loading");
    expect(call).not.toHaveBeenCalled();
  });

  it.each(SERVER_EXPORTS)("%s names a refusal, by its message or by its text", async (_, start, call) => {
    const store = exportStore({ sourceMode: "ode" });
    vi.mocked(call).mockRejectedValueOnce(new Error("export service down"));
    await start(store.getState());
    expect(store.getState().error).toBe("export service down");
    vi.mocked(call).mockRejectedValueOnce("refused as text");
    await start(store.getState());
    expect(store.getState().error).toBe("refused as text");
  });
});

describe("the Python script", () => {
  it("sends the current experiment and shows the script, its one-liner, its replay script and its digest", async () => {
    const store = exportStore({ sourceMode: "model", selectedModelName: "LIF", modelDetail: LIF, duration: 250 });
    vi.mocked(fetchCodegen).mockResolvedValue({
      script: "import sc_neurocore", oneliner: "python -c ...", replay_script: "replay()", experiment_sha256: "c".repeat(64),
    } as Awaited<ReturnType<typeof fetchCodegen>>);
    await store.getState().runCodegen();
    expect(fetchCodegen).toHaveBeenCalledWith(requestOf(store.getState()));
    expect(store.getState()).toMatchObject({
      activeTab: "code", codeScript: "import sc_neurocore", codeOneliner: "python -c ...", codeReplayScript: "replay()",
      codeExperimentSha256: "c".repeat(64),
    });
  });
});

describe("the replay pack and notebook", () => {
  it("downloads the replay pack the server builds for the current experiment", async () => {
    const store = exportStore({ sourceMode: "ode", duration: 250 });
    const pack = { files: [] } as unknown as Awaited<ReturnType<typeof fetchReplayPack>>;
    const blob = new Blob(["pack"]);
    vi.mocked(fetchReplayPack).mockResolvedValue(pack);
    vi.mocked(replayPackExport).mockReturnValue({ blob, filename: "experiment-replay.zip" });
    await store.getState().exportReplayPack();
    expect(fetchReplayPack).toHaveBeenCalledWith(requestOf(store.getState()));
    expect(replayPackExport).toHaveBeenCalledWith(pack);
    expect(downloadBrowserArtefact).toHaveBeenCalledWith(blob, "experiment-replay.zip");
  });

  it.each([
    ["a catalogue model", { sourceMode: "model", selectedModelName: "LIF", modelDetail: LIF }, "LIF-replay.ipynb"],
    ["ODE equations", { sourceMode: "ode" }, "equations-replay.ipynb"],
  ] as const)("downloads the replay notebook of %s under its own name, as notebook JSON", async (_, inputs, filename) => {
    const store = exportStore(inputs);
    const notebook = { cells: [], nbformat: 4 } as unknown as Awaited<ReturnType<typeof fetchReplayNotebook>>;
    vi.mocked(fetchReplayNotebook).mockResolvedValue(notebook);
    await store.getState().exportReplayNotebook();
    expect(fetchReplayNotebook).toHaveBeenCalledWith(requestOf(store.getState()));
    const [blob, name] = vi.mocked(downloadBrowserArtefact).mock.calls[0] ?? [];
    if (blob === undefined) throw new Error("nothing was downloaded");
    expect(name).toBe(filename);
    expect(blob.type).toBe("application/x-ipynb+json");
    expect(JSON.parse(await blob.text())).toStrictEqual(notebook);
  });
});

describe("the trace", () => {
  it.each([
    ["JSON", (s: StudioState) => { s.exportData(); }, "json"],
    ["CSV", (s: StudioState) => { s.exportCSV(); }, "csv"],
    ["SVG", (s: StudioState) => { s.exportSVG(); }, "svg"],
  ] as const)("is exported as %s", (_, start, format) => {
    const store = exportStore();
    start(store.getState());
    expect(exportStudioTrace).toHaveBeenCalledOnce();
    expect(vi.mocked(exportStudioTrace).mock.calls[0]?.[0]).toBe(format);
  });
});
