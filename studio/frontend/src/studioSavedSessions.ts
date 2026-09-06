// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio saved-session persistence helpers

export const STUDIO_SAVED_SESSIONS_KEY = "sc-studio-sessions";

export interface StudioSavedSession {
  name: string;
  state: Record<string, unknown>;
}

export type StudioSavedSessionSourceMode = "model" | "ode";

export interface StudioSavedSessionInput {
  sourceMode: StudioSavedSessionSourceMode;
  equations: string[];
  threshold: string;
  reset: string;
  odeParams: Record<string, number>;
  odeInit: Record<string, number>;
  selectedModelName: string;
  modelParams: Record<string, number>;
  dt: number;
  duration: number;
  current: number;
  protocol: string;
  frequencyHz: number;
  seed: number | null;
  trial: StudioSavedSessionTrial;
}

export type StudioSavedSessionTrial = "replay" | "fresh";

export interface StudioSavedSessionRestoreState extends StudioSavedSessionInput {}

export interface StudioSavedSessionsStatePatch {
  savedSessions: StudioSavedSession[];
}

export interface StudioSavedSessionStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStudioSavedSession(value: unknown): value is StudioSavedSession {
  if (!isRecord(value)) {
    return false;
  }
  return typeof value.name === "string" && isRecord(value.state);
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function sourceModeValue(value: unknown): StudioSavedSessionSourceMode {
  return value === "ode" ? "ode" : "model";
}

function stringArrayValue(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function nonZeroFiniteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value !== 0
    ? value
    : fallback;
}

function finiteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function positiveFiniteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

function seedValue(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : null;
}

function trialValue(value: unknown): StudioSavedSessionTrial {
  return value === "fresh" ? "fresh" : "replay";
}

function numberRecordValue(value: unknown): Record<string, number> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1])),
  );
}

export function browserSavedSessionStorage(): StudioSavedSessionStorage | null {
  return typeof localStorage === "undefined" ? null : localStorage;
}

export function studioSavedSessionState(input: StudioSavedSessionInput): Record<string, unknown> {
  return {
    sourceMode: input.sourceMode,
    equations: input.equations,
    threshold: input.threshold,
    reset: input.reset,
    odeParams: input.odeParams,
    odeInit: input.odeInit,
    selectedModelName: input.selectedModelName,
    modelParams: input.modelParams,
    dt: input.dt,
    duration: input.duration,
    current: input.current,
    protocol: input.protocol,
    frequencyHz: input.frequencyHz,
    seed: input.seed,
    trial: input.trial,
  };
}

export function studioSavedSessionRestoreState(
  state: Record<string, unknown>,
): StudioSavedSessionRestoreState {
  return {
    sourceMode: sourceModeValue(state.sourceMode),
    equations: stringArrayValue(state.equations),
    threshold: stringValue(state.threshold, ""),
    reset: stringValue(state.reset, ""),
    odeParams: numberRecordValue(state.odeParams),
    odeInit: numberRecordValue(state.odeInit),
    selectedModelName: stringValue(state.selectedModelName, ""),
    modelParams: numberRecordValue(state.modelParams),
    dt: nonZeroFiniteNumberValue(state.dt, 0.1),
    duration: nonZeroFiniteNumberValue(state.duration, 100),
    current: finiteNumberValue(state.current, 10),
    protocol: stringValue(state.protocol, "constant"),
    frequencyHz: positiveFiniteNumberValue(state.frequencyHz, 10),
    seed: seedValue(state.seed),
    trial: trialValue(state.trial),
  };
}

/**
 * What reading the browser cache actually found.
 *
 * `corrupt` and `partial` are distinct from `empty` on purpose: a payload the
 * browser mangled used to be reported as "you have no saved sessions", which
 * looks exactly like data loss and tells the user nothing.
 */
export type StudioSessionReadStatus = "ok" | "empty" | "unavailable" | "corrupt" | "partial";

export interface StudioSessionReadResult {
  sessions: StudioSavedSession[];
  status: StudioSessionReadStatus;
  /** Entries that were present but unreadable; zero unless `partial`. */
  discarded: number;
}

/** What writing to the browser cache achieved. */
export type StudioSessionWriteStatus = "ok" | "unavailable" | "quota-exceeded" | "failed";

export interface StudioSessionWriteResult {
  status: StudioSessionWriteStatus;
  /** Operator-facing reason; empty when the write succeeded. */
  message: string;
}

function isQuotaError(error: unknown): boolean {
  if (typeof DOMException !== "undefined" && error instanceof DOMException) {
    return error.name === "QuotaExceededError" || error.name === "NS_ERROR_DOM_QUOTA_REACHED";
  }
  return error instanceof Error && /quota/i.test(error.message);
}

/**
 * Read the cached sessions and say what state the cache was in.
 *
 * The browser cache is a convenience copy of workspaces the server holds; it
 * is never the record of truth. Reporting corruption instead of silently
 * returning an empty list is what lets the caller say so.
 */
export function readStudioSessionsResult(
  storage: StudioSavedSessionStorage | null = browserSavedSessionStorage(),
): StudioSessionReadResult {
  if (storage === null) {
    return { sessions: [], status: "unavailable", discarded: 0 };
  }
  let storedSessions: string | null | undefined;
  try {
    storedSessions = storage.getItem(STUDIO_SAVED_SESSIONS_KEY);
  } catch {
    return { sessions: [], status: "unavailable", discarded: 0 };
  }
  if (storedSessions === undefined || storedSessions === null) {
    return { sessions: [], status: "empty", discarded: 0 };
  }
  let parsedSessions: unknown;
  try {
    parsedSessions = JSON.parse(storedSessions);
  } catch {
    return { sessions: [], status: "corrupt", discarded: 0 };
  }
  if (!Array.isArray(parsedSessions)) {
    return { sessions: [], status: "corrupt", discarded: 0 };
  }
  const sessions = parsedSessions.filter(isStudioSavedSession);
  const discarded = parsedSessions.length - sessions.length;
  if (discarded > 0) {
    return { sessions, status: "partial", discarded };
  }
  return { sessions, status: sessions.length === 0 ? "empty" : "ok", discarded: 0 };
}

export function readStoredStudioSessions(
  storage: StudioSavedSessionStorage | null = browserSavedSessionStorage(),
): StudioSavedSession[] {
  return readStudioSessionsResult(storage).sessions;
}

/**
 * Write the cached sessions and say whether it worked.
 *
 * A denied quota used to throw out of this call. The caller now gets a status
 * it can show, and the server-side workspace revision remains the copy that
 * matters.
 */
export function writeStoredStudioSessions(
  sessions: readonly StudioSavedSession[],
  storage: StudioSavedSessionStorage | null = browserSavedSessionStorage(),
): StudioSessionWriteResult {
  if (storage === null) {
    return {
      status: "unavailable",
      message: "This browser has no local storage; sessions are kept on the server only.",
    };
  }
  try {
    storage.setItem(STUDIO_SAVED_SESSIONS_KEY, JSON.stringify(sessions));
    return { status: "ok", message: "" };
  } catch (error) {
    if (isQuotaError(error)) {
      return {
        status: "quota-exceeded",
        message:
          "The browser refused to cache this session: local storage is full. " +
          "Saved workspaces on the server are unaffected.",
      };
    }
    return {
      status: "failed",
      message:
        error instanceof Error
          ? `The browser could not cache this session: ${error.message}`
          : "The browser could not cache this session.",
    };
  }
}

export function upsertStudioSavedSession(
  sessions: readonly StudioSavedSession[],
  nextSession: StudioSavedSession,
): StudioSavedSession[] {
  return [
    nextSession,
    ...sessions.filter((session) => session.name !== nextSession.name),
  ];
}

export function studioSavedSessionUpsertState(
  sessions: readonly StudioSavedSession[],
  nextSession: StudioSavedSession,
): StudioSavedSessionsStatePatch {
  return {
    savedSessions: upsertStudioSavedSession(sessions, nextSession),
  };
}

export function removeStudioSavedSession(
  sessions: readonly StudioSavedSession[],
  name: string,
): StudioSavedSession[] {
  return sessions.filter((session) => session.name !== name);
}

export function studioSavedSessionRemovedState(
  sessions: readonly StudioSavedSession[],
  name: string,
): StudioSavedSessionsStatePatch {
  return {
    savedSessions: removeStudioSavedSession(sessions, name),
  };
}
