// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio saved-session persistence helpers

/**
 * Keeping the reader's workspaces in the browser, and reading them back.
 *
 * **The browser copy is a convenience, never the record.** Workspaces live on
 * the server; this is a local cache so a reload does not lose what someone was
 * in the middle of. Every failure here is therefore reported rather than
 * thrown: a full quota, a storage the browser refuses to open, a payload that
 * was mangled. The caller shows the reason and the server copy is unaffected.
 *
 * Reading is asymmetric with writing on purpose. A session is written as a
 * typed snapshot; it is read back as a document that may predate any field it
 * is missing, so every value goes through an accessor with a stated fallback
 * and an unreadable entry is dropped rather than trusted. The count of dropped
 * entries is reported, because "some of your sessions were unreadable" and
 * "you have no sessions" must not look alike.
 */

/** Where the cache lives in the browser's storage. */
export const STUDIO_SAVED_SESSIONS_KEY = "sc-studio-sessions";

/** One saved workspace: the name the reader gave it, and its state. */
export interface StudioSavedSession {
  name: string;
  state: Record<string, unknown>;
}

/** Whether a workspace was driven by a catalogue model or by an ODE. */
export type StudioSavedSessionSourceMode = "model" | "ode";

/** Everything a workspace remembers, as the panel holds it. */
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

/**
 * Whether a restored run repeats the original seed or draws a fresh one.
 * `replay` is the default everywhere, because a workspace reopened is expected
 * to reproduce what it showed.
 */
export type StudioSavedSessionTrial = "replay" | "fresh";

/**
 * What a restore produces. Identical to the input by construction: a workspace
 * must restore to the shape it was saved from, and stating that as the same
 * type is what keeps the two from drifting.
 */
export type StudioSavedSessionRestoreState = StudioSavedSessionInput;

/** The patch that replaces the store's list of saved workspaces. */
export interface StudioSavedSessionsStatePatch {
  savedSessions: StudioSavedSession[];
}

/**
 * The two storage calls this module makes. Declared rather than taking
 * `Storage`, so a test can supply one and so the cache cannot quietly grow a
 * dependency on the rest of the browser's storage API.
 */
export interface StudioSavedSessionStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/**
 * Whether a value is a plain object.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Whether a stored entry is readable as a saved workspace.
 *
 * @param value - The entry, as it was parsed.
 * @returns Whether it has a name and a state.
 */
function isStudioSavedSession(value: unknown): value is StudioSavedSession {
  if (!isRecord(value)) {
    return false;
  }
  return typeof value.name === "string" && isRecord(value.state);
}

/**
 * Read a string field, falling back when it is anything else.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The string.
 */
function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

/**
 * Read the source mode.
 *
 * @param value - The stored value.
 * @returns `ode` only when it says so; a workspace whose mode was lost opens
 *   as a model, which is the mode most of them are.
 */
function sourceModeValue(value: unknown): StudioSavedSessionSourceMode {
  return value === "ode" ? "ode" : "model";
}

/**
 * Read a list of strings, keeping only the entries that are strings.
 *
 * @param value - The stored value.
 * @returns The strings, or an empty list.
 */
function stringArrayValue(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

/**
 * Read a number that must not be zero, such as a step or a duration.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The number. Zero falls back, because a zero step or duration makes
 *   a run that cannot start.
 */
function nonZeroFiniteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value !== 0
    ? value
    : fallback;
}

/**
 * Read a number that may be zero, such as an injected current.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The number.
 */
function finiteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Read a number that must be above zero, such as a frequency.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The number.
 */
function positiveFiniteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

/**
 * Read a seed.
 *
 * @param value - The stored value.
 * @returns The seed, or `null` for no seed. A seed must be a whole number of
 *   zero or more; anything else is read as no seed rather than rounded,
 *   because a rounded seed reproduces a different run.
 */
function seedValue(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : null;
}

/**
 * Read whether the restore should replay or draw fresh.
 *
 * @param value - The stored value.
 * @returns `fresh` only when it says so.
 */
function trialValue(value: unknown): StudioSavedSessionTrial {
  return value === "fresh" ? "fresh" : "replay";
}

/**
 * Read a map of parameters, dropping every entry that is not a number.
 *
 * @param value - The stored value.
 * @returns The parameters that survived.
 */
function numberRecordValue(value: unknown): Record<string, number> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1])),
  );
}

/**
 * The browser's own storage, if this context has one.
 *
 * @returns The storage, or `null` where there is none -- a server render, or a
 *   browser configured to refuse it.
 */
export function browserSavedSessionStorage(): StudioSavedSessionStorage | null {
  return typeof localStorage === "undefined" ? null : localStorage;
}

/**
 * Take the snapshot that gets written.
 *
 * @param input - The workspace as the panel holds it.
 * @returns The document to store.
 */
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

/**
 * Read a stored document back into a workspace.
 *
 * Every field goes through an accessor with a stated fallback, because the
 * document may have been written by an older build that did not have it.
 *
 * @param state - The stored document.
 * @returns The workspace, complete whatever the document was missing.
 */
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

/** What was read, in what state the cache was, and how much was dropped. */
export interface StudioSessionReadResult {
  sessions: StudioSavedSession[];
  status: StudioSessionReadStatus;
  /** Entries that were present but unreadable; zero unless `partial`. */
  discarded: number;
}

/** What writing to the browser cache achieved. */
export type StudioSessionWriteStatus = "ok" | "unavailable" | "quota-exceeded" | "failed";

/** Whether the write landed, and what to tell the reader when it did not. */
export interface StudioSessionWriteResult {
  status: StudioSessionWriteStatus;
  /** Operator-facing reason; empty when the write succeeded. */
  message: string;
}

/**
 * Whether a storage failure was the browser running out of room.
 *
 * Quota is worth naming because it is the one failure the reader can act on,
 * and browsers report it under two different exception names.
 *
 * @param error - What was thrown.
 * @returns Whether it was a quota failure.
 */
function isQuotaError(error: unknown): boolean {
  if (typeof DOMException !== "undefined" && error instanceof DOMException) {
    return error.name === "QuotaExceededError" || error.name === "NS_ERROR_DOM_QUOTA_REACHED";
  }
  return error instanceof Error && /quota/i.test(error.message);
}

/**
 * Read the cached workspaces and say what state the cache was in.
 *
 * Reporting corruption instead of returning an empty list is the point: a
 * mangled payload read as "you have no saved sessions" looks exactly like data
 * loss and tells the reader nothing.
 *
 * @param storage - The storage; the browser's own by default.
 * @returns The workspaces, the cache's state, and how many entries were
 *   present but unreadable.
 */
export function readStudioSessionsResult(
  storage: StudioSavedSessionStorage | null = browserSavedSessionStorage(),
): StudioSessionReadResult {
  if (storage === null) {
    return { sessions: [], status: "unavailable", discarded: 0 };
  }
  let storedSessions: string | null;
  try {
    storedSessions = storage.getItem(STUDIO_SAVED_SESSIONS_KEY);
  } catch {
    return { sessions: [], status: "unavailable", discarded: 0 };
  }
  if (storedSessions === null) {
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

/**
 * Read the cached workspaces, ignoring what state the cache was in.
 *
 * @param storage - The storage; the browser's own by default.
 * @returns The workspaces that were readable.
 */
export function readStoredStudioSessions(
  storage: StudioSavedSessionStorage | null = browserSavedSessionStorage(),
): StudioSavedSession[] {
  return readStudioSessionsResult(storage).sessions;
}

/**
 * Write the cached workspaces and say whether it worked.
 *
 * A denied quota used to throw out of this call. It is a status now, with the
 * sentence to show, because the server copy is unaffected and the reader needs
 * to know that rather than see a crash.
 *
 * @param sessions - The workspaces to cache.
 * @param storage - The storage; the browser's own by default.
 * @returns Whether it landed, and why not when it did not.
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

/**
 * Add a workspace, or replace the one with the same name.
 *
 * The new entry goes first, so the list the reader sees is ordered by when
 * they last saved rather than by when they first did.
 *
 * @param sessions - The list as it stands.
 * @param nextSession - The workspace to add.
 * @returns The new list.
 */
export function upsertStudioSavedSession(
  sessions: readonly StudioSavedSession[],
  nextSession: StudioSavedSession,
): StudioSavedSession[] {
  return [
    nextSession,
    ...sessions.filter((session) => session.name !== nextSession.name),
  ];
}

/**
 * The patch that adds or replaces a workspace.
 *
 * @param sessions - The list as it stands.
 * @param nextSession - The workspace to add.
 * @returns The patch.
 */
export function studioSavedSessionUpsertState(
  sessions: readonly StudioSavedSession[],
  nextSession: StudioSavedSession,
): StudioSavedSessionsStatePatch {
  return {
    savedSessions: upsertStudioSavedSession(sessions, nextSession),
  };
}

/**
 * Remove a workspace by name.
 *
 * @param sessions - The list as it stands.
 * @param name - The workspace to remove.
 * @returns The new list.
 */
export function removeStudioSavedSession(
  sessions: readonly StudioSavedSession[],
  name: string,
): StudioSavedSession[] {
  return sessions.filter((session) => session.name !== name);
}

/**
 * The patch that removes a workspace.
 *
 * @param sessions - The list as it stands.
 * @param name - The workspace to remove.
 * @returns The patch.
 */
export function studioSavedSessionRemovedState(
  sessions: readonly StudioSavedSession[],
  name: string,
): StudioSavedSessionsStatePatch {
  return {
    savedSessions: removeStudioSavedSession(sessions, name),
  };
}
