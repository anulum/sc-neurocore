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

export async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `${r.status}`);
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
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `${r.status}`);
  }
  return r.blob();
}

export function getBlob(path: string): Promise<Blob> {
  return fetch(`${BASE}${path}`, { headers: authHeaders() }).then((r) => blob(r));
}

export function encodeArtifactPath(path: string): string {
  return path.split("/").map((segment) => encodeURIComponent(segment)).join("/");
}
