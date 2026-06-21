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

export function browserSavedSessionStorage(): StudioSavedSessionStorage | null {
  return typeof localStorage === "undefined" ? null : localStorage;
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
