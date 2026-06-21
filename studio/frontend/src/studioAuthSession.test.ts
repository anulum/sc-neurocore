// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio browser auth token persistence tests

import { describe, expect, it } from "vitest";

import {
  STUDIO_AUTH_STORAGE_KEY,
  clearStoredStudioAuthToken,
  readStoredStudioAuthToken,
  storeStudioAuthToken,
  syncStoredStudioAuthToken,
  type StudioAuthTokenStorage,
} from "./studioAuthSession";

function createStorage(initialToken: string | null = null): StudioAuthTokenStorage {
  const values = new Map<string, string>();
  if (initialToken !== null) {
    values.set(STUDIO_AUTH_STORAGE_KEY, initialToken);
  }
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
    removeItem: (key: string) => {
      values.delete(key);
    },
  };
}

describe("Studio auth token persistence", () => {
  it("treats unavailable browser storage as an unauthenticated session", () => {
    const syncedTokens: Array<string | null> = [];
    const setToken = (token: string | null): void => {
      syncedTokens.push(token);
    };

    expect(readStoredStudioAuthToken(null)).toBeNull();
    expect(syncStoredStudioAuthToken(setToken, null)).toBeNull();
    storeStudioAuthToken("session-token", null);
    clearStoredStudioAuthToken(null);

    expect(syncedTokens).toEqual([null]);
  });

  it("reads, stores, and clears the Studio bearer token", () => {
    const storage = createStorage();

    expect(readStoredStudioAuthToken(storage)).toBeNull();

    storeStudioAuthToken("session-token", storage);
    expect(readStoredStudioAuthToken(storage)).toBe("session-token");

    clearStoredStudioAuthToken(storage);
    expect(readStoredStudioAuthToken(storage)).toBeNull();
  });

  it("syncs the persisted browser token into the API client state", () => {
    const storage = createStorage("persisted-token");
    const syncedTokens: Array<string | null> = [];
    const setToken = (token: string | null): void => {
      syncedTokens.push(token);
    };

    expect(syncStoredStudioAuthToken(setToken, storage)).toBe("persisted-token");

    expect(syncedTokens).toEqual(["persisted-token"]);
  });
});
