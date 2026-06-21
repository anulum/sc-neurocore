// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio saved-session persistence tests

import { describe, expect, it } from "vitest";

import {
  STUDIO_SAVED_SESSIONS_KEY,
  readStoredStudioSessions,
  removeStudioSavedSession,
  upsertStudioSavedSession,
  writeStoredStudioSessions,
  type StudioSavedSession,
  type StudioSavedSessionStorage,
} from "./studioSavedSessions";

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

  it("removes saved sessions by name", () => {
    const existing: StudioSavedSession = { name: "other", state: { dt: 0.1 } };

    expect(removeStudioSavedSession([demoSession, existing], "demo")).toEqual([existing]);
  });
});
