// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio compiler request state custody tests

import type { ModelSummary } from "../api/client";
import { at } from "../arrayAt";
import { afterEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./studio";

const initialState = useStudioStore.getState();

afterEach(() => {
  useStudioStore.setState(initialState, true);
  vi.unstubAllGlobals();
});

it.each([
  ["runCompile", "/api/compile"],
  ["runEmitSV", "/api/ir/emit-sv-direct"],
] as const)("%s sends every ODE initial state to %s", async (action, route) => {
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(
    new Response(JSON.stringify({ verilog: "module sc_neuron; endmodule", chars: 30 }), {
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fetch);
  useStudioStore.setState({
    sourceMode: "ode",
    equations: ["dv/dt = -v + w + I", "dw/dt = -w"],
    odeInit: { v: 2, w: 3 },
  });
  await useStudioStore.getState()[action]();
  expect(fetch).toHaveBeenCalledOnce();
  const [url, options] = at(fetch.mock.calls, 0);
  expect(url).toBe(route);
  const body = options?.body;
  expect(JSON.parse(typeof body === "string" ? body : JSON.stringify(body))).toMatchObject({
    equations: ["dv/dt = -v + w + I", "dw/dt = -w"],
    init: { v: 2, w: 3 },
  });
});

const catalogueModel: ModelSummary = {
  name: "AdExNeuron", module: "adex", category: "point", category_slug: "point",
  category_source: "declared", metadata_state: "available", metadata_error: null,
  identity_kind: "source-literature", counts_in_source_catalogue: true,
  public_label: "", aliases: [], tier: 0, evidence_kind: "transport fixture",
  science_tier: 0, science_label: "S0", silicon_tier: null, silicon_label: "none",
  verified_science_tier: 0, verified_science_label: "S0",
  verified_silicon_tier: null, verified_silicon_label: "none", verified_profile: null,
  is_perfect_verified: false, validation_metric: "none", integration_method: "euler",
  terminal_silicon_tier: "", terminal_reason: "No hardware qualification in this transport test.",
  family: "point", maturity: "experimental", biophysical_detail: "point",
  n_params: 0, n_state_vars: 2, state_var_names: ["v", "w"], dt: 0.1,
  description: "Catalogue loading transport fixture", intended_use: [], hardware_fit: [],
  behavior_tags: [], provenance: null,
};
const preset = { id: "saved", title: "Saved experiment", description: "Preset loading transport fixture", suggested_view: "trace" };
const libraries = [
  { action: "loadModels", field: "models", error: "modelsError", entries: [catalogueModel] },
  { action: "loadPresets", field: "presets", error: "presetsError", entries: [preset] },
] as const;

it.each(libraries)("$action preserves loaded entries during an outage and clears its own error after retry", async ({ action, field, error, entries }) => {
  const transport = vi.fn<typeof fetch>()
    .mockResolvedValueOnce(Response.json(entries))
    .mockResolvedValueOnce(Response.json({ detail: "Library backend unavailable" }, { status: 503 }))
    .mockResolvedValueOnce(Response.json([]));
  vi.stubGlobal("fetch", transport);
  useStudioStore.setState({ sourceMode: "ode", equations: ["dv/dt = -v"], odeParams: { tau: 12 } });
  await useStudioStore.getState()[action]();
  await expect(useStudioStore.getState()[action]()).resolves.toBeUndefined();
  expect(useStudioStore.getState()[field]).toEqual(entries);
  expect(useStudioStore.getState()[error]).toBe("Library backend unavailable");
  expect(useStudioStore.getState().odeParams).toEqual({ tau: 12 });
  expect(useStudioStore.getState().sourceMode).toBe("ode");
  await useStudioStore.getState()[action]();
  expect(useStudioStore.getState()[field]).toEqual([]);
  expect(useStudioStore.getState()[error]).toBeNull();
  expect(transport).toHaveBeenCalledTimes(3);
});

it.each(libraries.flatMap((library) => [
  { ...library, olderStatus: 200 }, { ...library, olderStatus: 503 },
]))("$action ignores an older $olderStatus response after a newer successful load", async ({ action, field, error, entries, olderStatus }) => {
  let reply: (response: Response) => void = () => { throw new Error("No pending transport"); };
  const transport = vi.fn<typeof fetch>()
    .mockImplementationOnce(() => new Promise<Response>((resolve) => { reply = resolve; }))
    .mockResolvedValueOnce(Response.json(entries));
  vi.stubGlobal("fetch", transport);
  useStudioStore.setState({ sourceMode: "ode" });
  const older = useStudioStore.getState()[action]();
  await useStudioStore.getState()[action]();
  reply(Response.json(olderStatus === 200 ? [] : { detail: "Stale failure" }, { status: olderStatus }));
  await older;
  expect(useStudioStore.getState()[field]).toEqual(entries);
  expect(useStudioStore.getState()[error]).toBeNull();
});

it("reports initial model-detail failure and retries that same model without an unhandled rejection", async () => {
  const transport = vi.fn<typeof fetch>(async (input) => input === "/api/models"
    ? Response.json([catalogueModel])
    : Response.json({ detail: "Model detail unavailable" }, { status: 503 }));
  vi.stubGlobal("fetch", transport);
  await expect(useStudioStore.getState().loadModels()).resolves.toBeUndefined();
  expect(useStudioStore.getState().models).toEqual([catalogueModel]);
  expect(useStudioStore.getState().selectedModelName).toBe("AdExNeuron");
  expect(useStudioStore.getState().modelDetail).toBeNull();
  expect(useStudioStore.getState().modelsError).toBe("Model detail unavailable");
  await useStudioStore.getState().loadModels();
  expect(transport.mock.calls.filter(([url]) => url === "/api/models/AdExNeuron")).toHaveLength(2);
  expect(useStudioStore.getState().modelsError).toBe("Model detail unavailable");
});

it.each(["refresh", "equations"] as const)("initial model-detail failure after %s follows the current selection", async (nextAction) => {
  vi.useFakeTimers();
  let answer: (response: Response) => void = () => { throw new Error("No pending model detail"); };
  const transport = vi.fn<typeof fetch>(async (input) => {
    if (input === "/api/models") return Response.json([catalogueModel]);
    if (input === "/api/models/AdExNeuron") {
      return new Promise<Response>((resolve) => { answer = resolve; });
    }
    return Response.json({ detail: "Simulation unavailable" }, { status: 503 });
  });
  vi.stubGlobal("fetch", transport);
  try {
    const initial = useStudioStore.getState().loadModels();
    await vi.waitFor(() => {
      expect(transport.mock.calls.some(([url]) => url === "/api/models/AdExNeuron")).toBe(true);
    });
    if (nextAction === "refresh") await useStudioStore.getState().loadModels();
    else useStudioStore.getState().setSourceMode("ode");
    answer(Response.json({ detail: "Initial model detail unavailable" }, { status: 503 }));
    await initial;
    await vi.runAllTimersAsync();
    expect(useStudioStore.getState().modelsError).toBe(nextAction === "refresh"
      ? "Initial model detail unavailable" : null);
    expect(useStudioStore.getState().sourceMode).toBe(nextAction === "refresh" ? "model" : "ode");
    expect(transport.mock.calls.filter(([url]) => url === "/api/models/AdExNeuron")).toHaveLength(1);
  } finally {
    vi.useRealTimers();
  }
});
