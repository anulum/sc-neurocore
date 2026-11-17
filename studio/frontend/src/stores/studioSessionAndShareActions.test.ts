// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio sessions saved in the browser and share URLs, against real storage and a mocked clipboard
// @vitest-environment happy-dom

import { createStore } from "zustand/vanilla";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { STUDIO_SAVED_SESSIONS_KEY } from "../studioSavedSessions";
import { copyStudioShareUrlInRuntime, scheduleStudioShareStatusClear } from "../studioShareRuntime";
import { createStudioSessionAndShareActions } from "./studioSessionAndShareActions";
import { studioInitialData } from "./studioInitialState";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

vi.mock("../studioShareRuntime", async () => {
  const actual = await vi.importActual<typeof import("../studioShareRuntime")>("../studioShareRuntime");
  return { ...actual, copyStudioShareUrlInRuntime: vi.fn(), scheduleStudioShareStatusClear: vi.fn() };
});

beforeEach(() => { localStorage.clear(); });
afterEach(() => {
  vi.resetAllMocks();
  localStorage.clear();
});

/** A store with the real actions over the initial data; its session actions show a restore by a recorded call. */
function sessionStore() {
  const runRestoredExperiment = vi.fn();
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    ...createStudioSessionAndShareActions(set, get, runRestoredExperiment),
  }));
  return { store, runRestoredExperiment };
}

/** The session names the browser's storage holds. */
function storedNames(): string[] {
  const raw = localStorage.getItem(STUDIO_SAVED_SESSIONS_KEY);
  return raw === null ? [] : (JSON.parse(raw) as { name: string }[]).map((session) => session.name);
}

describe("sessions saved in the browser", () => {
  it("saves the workspace under a name, in the store and in storage, and reports the write", () => {
    const { store } = sessionStore();
    store.setState({ duration: 250 });
    const result = store.getState().saveSession("long run");
    expect(result.status).toBe("ok");
    expect(store.getState().sessionWriteResult).toEqual(result);
    expect(store.getState().savedSessions.map((session) => session.name)).toEqual(["long run"]);
    expect(store.getState().savedSessions[0]?.state.duration).toBe(250);
    expect(storedNames()).toEqual(["long run"]);
  });

  it("restores a saved session and shows the restored experiment", () => {
    const { store, runRestoredExperiment } = sessionStore();
    store.setState({ duration: 250 });
    store.getState().saveSession("long run");
    store.setState({ duration: 50 });
    store.getState().loadSession("long run");
    expect(store.getState().duration).toBe(250);
    expect(runRestoredExperiment).toHaveBeenCalledOnce();
  });

  it("does nothing for a name it does not hold", () => {
    const { store, runRestoredExperiment } = sessionStore();
    store.setState({ duration: 50 });
    store.getState().loadSession("missing");
    expect(store.getState().duration).toBe(50);
    expect(store.getState().error).toBeNull();
    expect(runRestoredExperiment).not.toHaveBeenCalled();
  });

  it("reports a saved session it cannot restore, and changes nothing", () => {
    const { store, runRestoredExperiment } = sessionStore();
    store.setState({ duration: 50, savedSessions: [{ name: "bad seed", state: { duration: 250, seed: -1 } }] });
    store.getState().loadSession("bad seed");
    expect(store.getState().error).toBe("Saved simulation seed must be a non-negative integer below 2^63");
    expect(store.getState().duration).toBe(50);
    expect(runRestoredExperiment).not.toHaveBeenCalled();
    store.setState({ savedSessions: [{ name: "thrown", state: { get seed(): never { throw new Error(""); } } }] });
    store.getState().loadSession("thrown");
    expect(store.getState().error).toBe("Failed to restore the saved session");
  });

  it("deletes a saved session from the store and from storage, and reports the write", () => {
    const { store } = sessionStore();
    store.getState().saveSession("first");
    store.getState().saveSession("second");
    const result = store.getState().deleteSession("first");
    expect(result.status).toBe("ok");
    expect(store.getState().sessionWriteResult).toEqual(result);
    expect(store.getState().savedSessions.map((session) => session.name)).toEqual(["second"]);
    expect(storedNames()).toEqual(["second"]);
  });
});

describe("share URLs", () => {
  it("copies all twelve run settings as a URL, says so, and clears the notice when the timer fires", async () => {
    const { store } = sessionStore();
    // Every field holds a value of its own, so a payload that drops or swaps one cannot match.
    const settings = {
      sourceMode: "ode", selectedModelName: "LIF", equations: ["dv/dt = (-v + I) / tau"], threshold: "v > 1",
      reset: "v = 0", modelParams: { tau_m: 7 }, odeParams: { tau: 20 }, odeInit: { v: 0.25 }, dt: 0.05,
      duration: 250, current: 4, protocol: "constant",
    } satisfies Partial<StudioState>;
    store.setState(settings);
    vi.mocked(copyStudioShareUrlInRuntime).mockResolvedValue({ ok: true } as Awaited<ReturnType<typeof copyStudioShareUrlInRuntime>>);
    store.getState().shareURL();
    await vi.waitFor(() => { expect(store.getState().error).toBe("URL copied to clipboard"); });
    expect(vi.mocked(copyStudioShareUrlInRuntime).mock.calls[0]?.[0]).toStrictEqual(settings);
    const clear = vi.mocked(scheduleStudioShareStatusClear).mock.calls[0]?.[0];
    expect(clear).toBeTypeOf("function");
    clear?.();
    expect(store.getState().error).toBeNull();
  });

  it("shows why a URL could not be copied", async () => {
    const { store } = sessionStore();
    vi.mocked(copyStudioShareUrlInRuntime).mockResolvedValue({ ok: false, message: "Clipboard access was denied." });
    store.getState().shareURL();
    await vi.waitFor(() => { expect(store.getState().error).toBe("Clipboard access was denied."); });
  });
});
