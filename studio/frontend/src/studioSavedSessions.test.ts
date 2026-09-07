// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio saved-session persistence tests

import { describe, expect, it } from "vitest";

import {
  readStudioSessionsResult,
  STUDIO_SAVED_SESSIONS_KEY,
  readStoredStudioSessions,
  removeStudioSavedSession,
  studioSavedSessionRemovedState,
  studioSavedSessionRestoreState,
  studioSavedSessionState,
  studioSavedSessionUpsertState,
  upsertStudioSavedSession,
  writeStoredStudioSessions,
  type StudioSavedSessionInput,
  type StudioSavedSession,
  type StudioSavedSessionStorage,
} from "./studioSavedSessions";

/**
 * Build a storage holding one raw stored value.
 *
 * @param initialValue - The stored text, or `null` for an empty cache.
 * @returns The storage.
 */
function createStorage(initialValue: string | null = null): StudioSavedSessionStorage {
  const values = new Map<string, string>();
  if (initialValue !== null) {
    values.set(STUDIO_SAVED_SESSIONS_KEY, initialValue);
  }
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
  };
}

describe("Studio saved-session persistence", () => {
  const demoSession: StudioSavedSession = {
    name: "demo",
    state: { sourceMode: "model", selectedModelName: "lif" },
  };
  const demoInput: StudioSavedSessionInput = {
    current: 10,
    dt: 0.1,
    duration: 100,
    equations: ["dv/dt = -v / tau"],
    modelParams: { tau: 10 },
    odeInit: { v: -65 },
    odeParams: { tau: 20 },
    protocol: "constant",
    reset: "v = -65",
    selectedModelName: "lif",
    sourceMode: "model",
    threshold: "v > -50",
    frequencyHz: 10,
    seed: 42,
    trial: "replay",
  };

  it("returns an empty list when browser storage is unavailable or invalid", () => {
    expect(readStoredStudioSessions(null)).toEqual([]);
    expect(readStoredStudioSessions(createStorage("not-json"))).toEqual([]);
    expect(readStoredStudioSessions(createStorage("{\"name\":\"demo\"}"))).toEqual([]);
  });

  it("filters malformed saved-session records", () => {
    const stored = JSON.stringify([
      demoSession,
      { name: "missing-state" },
      { name: 12, state: {} },
      { name: "array-state", state: [] },
    ]);

    expect(readStoredStudioSessions(createStorage(stored))).toEqual([demoSession]);
  });

  it("writes saved sessions with the canonical storage key", () => {
    const storage = createStorage();

    writeStoredStudioSessions([demoSession], storage);

    expect(readStoredStudioSessions(storage)).toEqual([demoSession]);
  });

  it("upserts saved sessions by newest name and preserves unrelated sessions", () => {
    const existing: StudioSavedSession = { name: "other", state: { dt: 0.1 } };
    const replaced: StudioSavedSession = { name: "demo", state: { dt: 1 } };

    expect(upsertStudioSavedSession([demoSession, existing], replaced)).toEqual([
      replaced,
      existing,
    ]);
  });

  it("builds saved-session upsert state patches for store consumers", () => {
    const existing: StudioSavedSession = { name: "other", state: { dt: 0.1 } };
    const replaced: StudioSavedSession = { name: "demo", state: { dt: 1 } };

    expect(studioSavedSessionUpsertState([demoSession, existing], replaced)).toEqual({
      savedSessions: [replaced, existing],
    });
  });

  it("removes saved sessions by name", () => {
    const existing: StudioSavedSession = { name: "other", state: { dt: 0.1 } };

    expect(removeStudioSavedSession([demoSession, existing], "demo")).toEqual([existing]);
  });

  it("builds saved-session removal state patches for store consumers", () => {
    const existing: StudioSavedSession = { name: "other", state: { dt: 0.1 } };

    expect(studioSavedSessionRemovedState([demoSession, existing], "demo")).toEqual({
      savedSessions: [existing],
    });
  });

  it("builds the persisted state snapshot from the active Studio state", () => {
    expect(studioSavedSessionState(demoInput)).toEqual({
      current: 10,
      dt: 0.1,
      duration: 100,
      equations: ["dv/dt = -v / tau"],
      modelParams: { tau: 10 },
      odeInit: { v: -65 },
      odeParams: { tau: 20 },
      protocol: "constant",
      reset: "v = -65",
      selectedModelName: "lif",
      sourceMode: "model",
      threshold: "v > -50",
      frequencyHz: 10,
      seed: 42,
      trial: "replay",
    });
  });

  it("restores valid persisted state with finite numeric records and zero current", () => {
    expect(studioSavedSessionRestoreState({
      ...studioSavedSessionState(demoInput),
      current: 0,
      equations: ["ok", 4],
      modelParams: { keep: 1, drop: Number.NaN, text: "x" },
      odeInit: { v: -64, invalid: Number.POSITIVE_INFINITY },
      sourceMode: "ode",
    })).toEqual({
      ...demoInput,
      current: 0,
      equations: ["ok"],
      modelParams: { keep: 1 },
      odeInit: { v: -64 },
      sourceMode: "ode",
    });
  });

  it("falls back safely for malformed restored session fields", () => {
    expect(studioSavedSessionRestoreState({
      current: Number.NaN,
      dt: 0,
      duration: Number.POSITIVE_INFINITY,
      equations: "not-array",
      modelParams: [],
      odeInit: null,
      odeParams: { keep: 2, bad: "x" },
      protocol: 4,
      reset: null,
      selectedModelName: false,
      sourceMode: "invalid",
      threshold: undefined,
      frequencyHz: -3,
      seed: 1.5,
      trial: "random",
    })).toEqual({
      current: 10,
      dt: 0.1,
      duration: 100,
      equations: [],
      modelParams: {},
      odeInit: {},
      odeParams: { keep: 2 },
      protocol: "constant",
      reset: "",
      selectedModelName: "",
      sourceMode: "model",
      threshold: "",
      frequencyHz: 10,
      seed: null,
      trial: "replay",
    });
  });

  it("restores an explicit fresh trial and a missing seed as the model default", () => {
    expect(studioSavedSessionRestoreState({
      ...studioSavedSessionState(demoInput),
      seed: undefined,
      trial: "fresh",
      frequencyHz: 40,
    })).toEqual({ ...demoInput, seed: null, trial: "fresh", frequencyHz: 40 });
  });
});

describe("browser cache status", () => {
  const key = "sc-studio-sessions";

  /**
   * Build a storage that keeps its entries in a map.
   *
   * It carried a `removeItem` the storage contract does not declare and no
   * case called; a map holds the entries now and that method is gone.
   *
   * @param initial - The entries it starts with.
   * @param onSet - Called with each value written, when a case wants to see
   *   what was stored.
   * @returns The storage.
   */
  function storage(
    initial: Record<string, string> = {},
    onSet?: (value: string) => void,
  ): StudioSavedSessionStorage {
    const data = new Map(Object.entries(initial));
    return {
      getItem: (name: string) => data.get(name) ?? null,
      setItem: (name: string, value: string) => {
        onSet?.(value);
        data.set(name, value);
      },
    };
  }

  it("reports an absent cache rather than pretending it is empty", () => {
    expect(readStudioSessionsResult(null)).toEqual({
      sessions: [],
      status: "unavailable",
      discarded: 0,
    });
  });

  it("reports a corrupt payload instead of silently losing the sessions", () => {
    const result = readStudioSessionsResult(storage({ [key]: "{not json" }));
    expect(result.status).toBe("corrupt");
    expect(result.sessions).toEqual([]);
  });

  it("reports how many entries it had to discard", () => {
    const good = { name: "a", savedAt: 1, state: {} };
    const result = readStudioSessionsResult(
      storage({ [key]: JSON.stringify([good, { rubbish: true }]) }),
    );
    expect(result.status).toBe("partial");
    expect(result.discarded).toBe(1);
    expect(result.sessions).toHaveLength(1);
  });

  it("distinguishes an empty cache from a populated one", () => {
    expect(readStudioSessionsResult(storage({ [key]: "[]" })).status).toBe("empty");
    const good = { name: "a", savedAt: 1, state: {} };
    expect(readStudioSessionsResult(storage({ [key]: JSON.stringify([good]) })).status).toBe("ok");
  });

  it("turns a denied quota into a message instead of an exception", () => {
    const denying = storage({}, () => {
      throw new DOMException("out of space", "QuotaExceededError");
    });

    const result = writeStoredStudioSessions([], denying);

    expect(result.status).toBe("quota-exceeded");
    expect(result.message).toContain("local storage is full");
    // The server-side workspace is the copy that matters, and the message says so.
    expect(result.message).toContain("server");
  });

  it("reports any other write failure without throwing", () => {
    const failing = storage({}, () => {
      throw new Error("disk on fire");
    });

    const result = writeStoredStudioSessions([], failing);

    expect(result.status).toBe("failed");
    expect(result.message).toContain("disk on fire");
  });

  it("says when there is no storage to write to", () => {
    expect(writeStoredStudioSessions([], null).status).toBe("unavailable");
  });

  it("succeeds quietly when the browser cooperates", () => {
    const store = storage();
    expect(writeStoredStudioSessions([{ name: "a", state: {} }], store)).toEqual({
      status: "ok",
      message: "",
    });
    expect(readStudioSessionsResult(store).sessions).toHaveLength(1);
  });
});
