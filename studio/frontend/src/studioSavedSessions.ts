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
}

export interface StudioSavedSessionRestoreState extends StudioSavedSessionInput {}

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

function finiteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value !== 0
    ? value
    : fallback;
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
    dt: finiteNumberValue(state.dt, 0.1),
    duration: finiteNumberValue(state.duration, 100),
    current: finiteNumberValue(state.current, 10),
    protocol: stringValue(state.protocol, "constant"),
  };
}

export function readStoredStudioSessions(
  storage: StudioSavedSessionStorage | null = browserSavedSessionStorage(),
): StudioSavedSession[] {
  const storedSessions = storage?.getItem(STUDIO_SAVED_SESSIONS_KEY);
  if (storedSessions === undefined || storedSessions === null) {
    return [];
  }
  try {
    const parsedSessions: unknown = JSON.parse(storedSessions);
    if (!Array.isArray(parsedSessions)) {
      return [];
    }
    return parsedSessions.filter(isStudioSavedSession);
  } catch {
    return [];
  }
}

export function writeStoredStudioSessions(
  sessions: readonly StudioSavedSession[],
  storage: StudioSavedSessionStorage | null = browserSavedSessionStorage(),
): void {
  storage?.setItem(STUDIO_SAVED_SESSIONS_KEY, JSON.stringify(sessions));
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

export function removeStudioSavedSession(
  sessions: readonly StudioSavedSession[],
  name: string,
): StudioSavedSession[] {
  return sessions.filter((session) => session.name !== name);
}
