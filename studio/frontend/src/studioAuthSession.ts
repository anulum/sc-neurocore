// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio browser auth token persistence helpers

import type { StudioAuthSession } from "./api/client";

export const STUDIO_AUTH_STORAGE_KEY = "sc-neurocore-studio-auth-token";

export interface StudioAuthTokenStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface StudioAuthStatePatch {
  authError?: string | null;
  authLoading?: boolean;
  authSession?: StudioAuthSession;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.length > 0 ? error.message : fallback;
}

export function browserSessionTokenStorage(): StudioAuthTokenStorage | null {
  return typeof sessionStorage === "undefined" ? null : sessionStorage;
}

export function unauthenticatedStudioAuthSession(): StudioAuthSession {
  return {
    authenticated: false,
    principal_id: null,
    roles: [],
  };
}

export function studioAuthLoadingState(): StudioAuthStatePatch {
  return {
    authError: null,
    authLoading: true,
  };
}

export function studioAuthSessionLoadedState(
  authSession: StudioAuthSession,
): StudioAuthStatePatch {
  return {
    authError: null,
    authLoading: false,
    authSession,
  };
}

export function studioAuthUnauthenticatedState(): StudioAuthStatePatch {
  return {
    authSession: unauthenticatedStudioAuthSession(),
  };
}

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

export function studioAuthLogoutCompleteState(): StudioAuthStatePatch {
  return {
    authLoading: false,
    authSession: unauthenticatedStudioAuthSession(),
  };
}

export function readStoredStudioAuthToken(
  storage: StudioAuthTokenStorage | null = browserSessionTokenStorage(),
): string | null {
  return storage?.getItem(STUDIO_AUTH_STORAGE_KEY) ?? null;
}

export function storeStudioAuthToken(
  token: string,
  storage: StudioAuthTokenStorage | null = browserSessionTokenStorage(),
): void {
  storage?.setItem(STUDIO_AUTH_STORAGE_KEY, token);
}

export function clearStoredStudioAuthToken(
  storage: StudioAuthTokenStorage | null = browserSessionTokenStorage(),
): void {
  storage?.removeItem(STUDIO_AUTH_STORAGE_KEY);
}

export function syncStoredStudioAuthToken(
  setToken: (token: string | null) => void,
  storage: StudioAuthTokenStorage | null = browserSessionTokenStorage(),
): string | null {
  const token = readStoredStudioAuthToken(storage);
  setToken(token);
  return token;
}
