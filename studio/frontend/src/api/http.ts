// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// HTTP transport for Studio /api (auth token, JSON helpers).

const BASE = "/api";
let studioAuthToken: string | null = null;

export function setStudioAuthToken(token: string | null): void {
  studioAuthToken = token;
}

export function progressWebSocketProtocols(token: string | null = studioAuthToken): string[] | undefined {
  if (token === null) {
    return undefined;
  }
  return ["studio-auth", `studio-bearer.${token}`];
}

export function authHeaders(): Record<string, string> {
  return studioAuthToken === null ? {} : { Authorization: `Bearer ${studioAuthToken}` };
}

/**
 * A failed Studio request, carrying the status and the structured detail.
 *
 * FastAPI returns `detail` as an object for errors that have more to say than
 * a sentence — a save conflict names the revision that is actually current.
 * `new Error(detail)` would stringify that to "[object Object]", so the
 * message is read out of the object and the object itself is kept for callers
 * that need to act on it.
 */
export class StudioRequestError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "StudioRequestError";
    this.status = status;
    this.detail = detail;
  }
}

function errorMessage(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.length > 0) {
    return detail;
  }
  if (typeof detail === "object" && detail !== null) {
    const record = detail as Record<string, unknown>;
    for (const key of ["reason", "message", "error"]) {
      const value = record[key];
      if (typeof value === "string" && value.length > 0) {
        return value;
      }
    }
  }
  return `${status}`;
}

async function requestError(r: Response): Promise<StudioRequestError> {
  const body = await r.json().catch(() => ({ detail: r.statusText }));
  const detail = (body as { detail?: unknown }).detail;
  return new StudioRequestError(errorMessage(detail, r.status), r.status, detail);
}

export async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    throw await requestError(r);
  }
  return r.json();
}

export function post<T>(path: string, body: unknown): Promise<T> {
  return fetch(`${BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  }).then((r) => json<T>(r));
}

export function patch<T>(path: string, body: unknown): Promise<T> {
  return fetch(`${BASE}${path}`, {
    method: "PATCH", headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  }).then((r) => json<T>(r));
}

export function get<T>(path: string): Promise<T> {
  return fetch(`${BASE}${path}`, { headers: authHeaders() }).then((r) => json<T>(r));
}

export async function blob(r: Response): Promise<Blob> {
  if (!r.ok) {
    throw await requestError(r);
  }
  return r.blob();
}

export function getBlob(path: string): Promise<Blob> {
  return fetch(`${BASE}${path}`, { headers: authHeaders() }).then((r) => blob(r));
}

export function encodeArtifactPath(path: string): string {
  return path.split("/").map((segment) => encodeURIComponent(segment)).join("/");
}
