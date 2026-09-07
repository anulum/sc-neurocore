// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio browser auth token persistence helpers

/**
 * Where the reader's Studio token lives, and the patches that follow it.
 *
 * The token is kept in **session** storage, not local storage. That is the
 * whole security posture of this file: closing the tab ends the session, and a
 * token does not outlive the browsing context that obtained it.
 *
 * Every storage call goes through an optional chain, because a browser may
 * refuse storage entirely. A reader in that browser is simply unauthenticated
 * rather than met with a crash, and every failure path here produces the
 * unauthenticated session rather than leaving the previous one in place.
 */

import type { StudioAuthSession } from "./api/client";

/** Where the token lives in the browser's session storage. */
export const STUDIO_AUTH_STORAGE_KEY = "sc-neurocore-studio-auth-token";

/**
 * The three storage calls this module makes. Declared rather than taking
 * `Storage`, so a test can supply one.
 */
export interface StudioAuthTokenStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/**
 * A patch from an authentication step. Each field is optional because the
 * steps differ in what they know: a logout failure has a message and no new
 * session, and a completed logout has a session and no message.
 */
export interface StudioAuthStatePatch {
  authError?: string | null;
  authLoading?: boolean;
  authSession?: StudioAuthSession;
}

/**
 * Read a thrown value's message, or fall back.
 *
 * @param error - What was thrown, which need not be an `Error`.
 * @param fallback - What to say when it carries no message.
 * @returns The message to show.
 */
function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.length > 0 ? error.message : fallback;
}

/**
 * The browser's session storage, if this context has one.
 *
 * @returns The storage, or `null` where there is none.
 */
export function browserSessionTokenStorage(): StudioAuthTokenStorage | null {
  return typeof sessionStorage === "undefined" ? null : sessionStorage;
}

/**
 * The session of someone who is not signed in.
 *
 * @returns The session: not authenticated, no principal, no roles.
 */
export function unauthenticatedStudioAuthSession(): StudioAuthSession {
  return {
    authenticated: false,
    principal_id: null,
    roles: [],
  };
}

/**
 * An authentication request has started.
 *
 * @returns The patch.
 */
export function studioAuthLoadingState(): StudioAuthStatePatch {
  return {
    authError: null,
    authLoading: true,
  };
}

/**
 * A session arrived.
 *
 * @param authSession - The session the server reported.
 * @returns The patch.
 */
export function studioAuthSessionLoadedState(
  authSession: StudioAuthSession,
): StudioAuthStatePatch {
  return {
    authError: null,
    authLoading: false,
    authSession,
  };
}

/**
 * The server said nobody is signed in.
 *
 * This is not an error and carries no message: not being signed in is an
 * ordinary state, and showing an error for it would be wrong.
 *
 * @returns The patch.
 */
export function studioAuthUnauthenticatedState(): StudioAuthStatePatch {
  return {
    authSession: unauthenticatedStudioAuthSession(),
  };
}

/**
 * An authentication request failed.
 *
 * The session is replaced with the unauthenticated one rather than left as
 * it was: after a failed check, the previous session is no longer
 * something this build can vouch for.
 *
 * @param error - What was thrown.
 * @param fallback - What to say when it carries no message.
 * @returns The patch.
 */
export function studioAuthFailureState(
  error: unknown,
  fallback: string,
): StudioAuthStatePatch {
  return {
    authError: errorMessage(error, fallback),
    authLoading: false,
    authSession: unauthenticatedStudioAuthSession(),
  };
}

/**
 * A logout request failed.
 *
 * The session is deliberately left alone: the server may still hold it,
 * and showing the reader as signed out while they are not would be a lie
 * in the direction that matters.
 *
 * @param error - What was thrown.
 * @returns The patch.
 */
export function studioAuthLogoutFailureState(error: unknown): StudioAuthStatePatch {
  return {
    authError: errorMessage(error, "Logout failed"),
  };
}

/**
 * A logout succeeded.
 *
 * @returns The patch.
 */
export function studioAuthLogoutCompleteState(): StudioAuthStatePatch {
  return {
    authLoading: false,
    authSession: unauthenticatedStudioAuthSession(),
  };
}

/**
 * Read the stored token.
 *
 * @param storage - The storage; the browser's session storage by default.
 * @returns The token, or `null` when there is none or no storage.
 */
export function readStoredStudioAuthToken(
  storage: StudioAuthTokenStorage | null = browserSessionTokenStorage(),
): string | null {
  return storage?.getItem(STUDIO_AUTH_STORAGE_KEY) ?? null;
}

/**
 * Store a token for the rest of this browsing session.
 *
 * @param token - The token.
 * @param storage - The storage; the browser's session storage by default.
 */
export function storeStudioAuthToken(
  token: string,
  storage: StudioAuthTokenStorage | null = browserSessionTokenStorage(),
): void {
  storage?.setItem(STUDIO_AUTH_STORAGE_KEY, token);
}

/**
 * Forget the stored token.
 *
 * @param storage - The storage; the browser's session storage by default.
 */
export function clearStoredStudioAuthToken(
  storage: StudioAuthTokenStorage | null = browserSessionTokenStorage(),
): void {
  storage?.removeItem(STUDIO_AUTH_STORAGE_KEY);
}

/**
 * Hand the stored token to the API client, whatever it is.
 *
 * The absence of a token is synced too: the client must be told there is
 * none, or it keeps sending the one from before.
 *
 * @param setToken - How to tell the client.
 * @param storage - The storage; the browser's session storage by default.
 * @returns The token that was synced, or `null`.
 */
export function syncStoredStudioAuthToken(
  setToken: (token: string | null) => void,
  storage: StudioAuthTokenStorage | null = browserSessionTokenStorage(),
): string | null {
  const token = readStoredStudioAuthToken(storage);
  setToken(token);
  return token;
}
