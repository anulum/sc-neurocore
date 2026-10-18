// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public saved-session action persistence contracts
// @vitest-environment happy-dom

import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./studio";
import { STUDIO_SAVED_SESSIONS_KEY, readStoredStudioSessions } from "../studioSavedSessions";
import { stringifyStudioJson } from "../studioJson";

const initial = useStudioStore.getState();
const previous = { name: "previous", state: { sourceMode: "ode", seed: 424242 } };

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem(STUDIO_SAVED_SESSIONS_KEY, stringifyStudioJson([previous]));
  useStudioStore.setState({
    ...initial, savedSessions: [previous], sessionWriteResult: null,
    sourceMode: "ode", seed: 9223372036854775807n,
    error: "An unrelated simulation failed",
  }, true);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  localStorage.clear();
  useStudioStore.setState(initial, true);
});

it("returns and records a real successful write with exact seed custody and name replacement", () => {
  const result = useStudioStore.getState().saveSession("current");
  expect(result).toEqual({ status: "ok", message: "" });
  expect(useStudioStore.getState().sessionWriteResult).toBe(result);
  expect(localStorage.getItem(STUDIO_SAVED_SESSIONS_KEY)).toContain('"seed":9223372036854775807');
  expect(readStoredStudioSessions().map((session) => session.name)).toEqual(["current", "previous"]);
  useStudioStore.setState({ current: 12 });
  useStudioStore.getState().saveSession("current");
  expect(readStoredStudioSessions().map((session) => session.name)).toEqual(["current", "previous"]);
  expect(readStoredStudioSessions()[0]?.state.current).toBe(12);
  expect(useStudioStore.getState().error).toBe("An unrelated simulation failed");
});

it("returns and records successful deletion while preserving another snapshot", () => {
  useStudioStore.getState().saveSession("current");
  const result = useStudioStore.getState().deleteSession("previous");
  expect(result).toEqual({ status: "ok", message: "" });
  expect(useStudioStore.getState().sessionWriteResult).toBe(result);
  expect(readStoredStudioSessions().map((session) => session.name)).toEqual(["current"]);
  expect(readStoredStudioSessions()[0]?.state.seed).toBe(9223372036854775807n);
});

it.each(["saveSession", "deleteSession"] as const)(
  "%s keeps page changes and the old stored bytes when the browser refuses quota",
  (action) => {
    const stored = localStorage.getItem(STUDIO_SAVED_SESSIONS_KEY);
    const persistent = localStorage;
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => persistent.getItem(key),
      setItem: () => { throw new DOMException("full", "QuotaExceededError"); },
    });
    const result = useStudioStore.getState()[action](action === "saveSession" ? "current" : "previous");
    expect(result.status).toBe("quota-exceeded");
    expect(result.message).toContain("local storage is full");
    expect(useStudioStore.getState().sessionWriteResult).toBe(result);
    expect(useStudioStore.getState().savedSessions.map((session) => session.name))
      .toEqual(action === "saveSession" ? ["current", "previous"] : []);
    expect(localStorage.getItem(STUDIO_SAVED_SESSIONS_KEY)).toBe(stored);
    expect(useStudioStore.getState().error).toBe("An unrelated simulation failed");
  },
);

it.each(["saveSession", "deleteSession"] as const)(
  "%s exposes unavailable storage without discarding page changes",
  (action) => {
    vi.stubGlobal("localStorage", undefined);
    const result = useStudioStore.getState()[action](action === "saveSession" ? "current" : "previous");
    expect(result.status).toBe("unavailable");
    expect(result.message).toContain("unavailable");
    expect(useStudioStore.getState().sessionWriteResult).toBe(result);
    expect(useStudioStore.getState().savedSessions.map((session) => session.name))
      .toEqual(action === "saveSession" ? ["current", "previous"] : []);
  },
);

it.each(["saveSession", "deleteSession"] as const)(
  "%s exposes a non-quota storage failure and retains the previous persistent list",
  (action) => {
    const stored = localStorage.getItem(STUDIO_SAVED_SESSIONS_KEY);
    const persistent = localStorage;
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => persistent.getItem(key),
      setItem: () => { throw new Error("write denied"); },
    });
    const result = useStudioStore.getState()[action](action === "saveSession" ? "current" : "previous");
    expect(result.status).toBe("failed");
    expect(useStudioStore.getState().sessionWriteResult).toBe(result);
    expect(localStorage.getItem(STUDIO_SAVED_SESSIONS_KEY)).toBe(stored);
  },
);

it("retries all retained page changes after a failed write and clears the failure result", () => {
  const persistent = localStorage;
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => persistent.getItem(key),
    setItem: () => { throw new DOMException("full", "QuotaExceededError"); },
  });
  useStudioStore.getState().saveSession("first");
  expect(useStudioStore.getState().sessionWriteResult?.status).toBe("quota-exceeded");
  vi.unstubAllGlobals();
  const result = useStudioStore.getState().saveSession("second");
  expect(result.status).toBe("ok");
  expect(useStudioStore.getState().sessionWriteResult).toBe(result);
  expect(readStoredStudioSessions().map((session) => session.name)).toEqual(["second", "first", "previous"]);
  expect(readStoredStudioSessions()[1]?.state.seed).toBe(9223372036854775807n);
});
