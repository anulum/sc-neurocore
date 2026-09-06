// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// HTTP transport for Studio /api (auth token, JSON helpers).

/** Prefix every Studio route shares. */
const BASE = "/api";

/** The bearer token the Studio was last signed in with, if any. */
let studioAuthToken: string | null = null;

/**
 * Set or clear the token every later request is sent with.
 *
 * @param token - The bearer token, or `null` to sign out.
 */
export function setStudioAuthToken(token: string | null): void {
  studioAuthToken = token;
}

/**
 * Carry the token into a WebSocket handshake, which has no headers.
 *
 * The browser's WebSocket API sends no `Authorization` header, so the token
 * travels as a subprotocol the server reads back. An unauthenticated socket
 * sends no subprotocol at all rather than an empty one, which a server is
 * entitled to reject.
 *
 * @param token - The token to send; defaults to the one currently set.
 * @returns The subprotocols to open the socket with, or `undefined` when there
 *   is no token.
 */
export function progressWebSocketProtocols(token: string | null = studioAuthToken): string[] | undefined {
  if (token === null) {
    return undefined;
  }
  return ["studio-auth", `studio-bearer.${token}`];
}

/**
 * The authorisation header for a request, empty when signed out.
 *
 * @returns Headers to spread into a request, which is empty rather than
 *   carrying an empty token.
 */
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
  /** HTTP status the server answered with. */
  readonly status: number;
  /** The `detail` the server sent, whatever shape it was in. */
  readonly detail: unknown;

  /**
   * Build the error from what the server actually said.
   *
   * @param message - The sentence read out of the detail, for display.
   * @param status - The HTTP status.
   * @param detail - The detail itself, kept for callers that act on it.
   */
  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "StudioRequestError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Read a displayable sentence out of a server's `detail`.
 *
 * @param detail - The `detail` field, of any shape.
 * @param status - The HTTP status, used when the detail says nothing usable.
 * @returns A sentence to show, never an empty string.
 */
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

/**
 * Turn a failed response into the error the callers expect.
 *
 * A body that is not JSON at all falls back to the status text, so a proxy
 * error page does not become an unhandled parse failure.
 *
 * @param r - The failed response.
 * @returns The error to throw.
 */
async function requestError(r: Response): Promise<StudioRequestError> {
  const body: unknown = await r.json().catch(() => ({ detail: r.statusText }));
  const detail =
    typeof body === "object" && body !== null ? (body as { detail?: unknown }).detail : undefined;
  return new StudioRequestError(errorMessage(detail, r.status), r.status, detail);
}

/**
 * Read a successful response's JSON, or throw what the server said.
 *
 * The caller names `T`; nothing here can check that the server agreed, which
 * is why every route module states the type it expects rather than leaving it
 * to be inferred at the call site.
 *
 * @param r - The response.
 * @returns The parsed body, as the caller's `T`.
 * @throws {StudioRequestError} When the response is not a success.
 */
export async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    throw await requestError(r);
  }
  return (await r.json()) as T;
}

/**
 * POST JSON to a Studio route.
 *
 * @param path - Route below `/api`, beginning with a slash.
 * @param body - The value to send as JSON.
 * @returns The parsed response body.
 */
export function post<T>(path: string, body: unknown): Promise<T> {
  return fetch(`${BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  }).then((r) => json<T>(r));
}

/**
 * PATCH JSON to a Studio route.
 *
 * @param path - Route below `/api`, beginning with a slash.
 * @param body - The value to send as JSON.
 * @returns The parsed response body.
 */
export function patch<T>(path: string, body: unknown): Promise<T> {
  return fetch(`${BASE}${path}`, {
    method: "PATCH", headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  }).then((r) => json<T>(r));
}

/**
 * GET JSON from a Studio route.
 *
 * @param path - Route below `/api`, beginning with a slash.
 * @returns The parsed response body.
 */
export function get<T>(path: string): Promise<T> {
  return fetch(`${BASE}${path}`, { headers: authHeaders() }).then((r) => json<T>(r));
}

/**
 * Read a successful response's bytes, or throw what the server said.
 *
 * @param r - The response.
 * @returns The body as a blob.
 * @throws {StudioRequestError} When the response is not a success.
 */
export async function blob(r: Response): Promise<Blob> {
  if (!r.ok) {
    throw await requestError(r);
  }
  return r.blob();
}

/**
 * GET a file from a Studio route.
 *
 * @param path - Route below `/api`, beginning with a slash.
 * @returns The body as a blob.
 */
export function getBlob(path: string): Promise<Blob> {
  return fetch(`${BASE}${path}`, { headers: authHeaders() }).then((r) => blob(r));
}

/**
 * Encode an artefact path for a URL without losing its separators.
 *
 * `encodeURIComponent` on the whole path would escape the slashes too, which
 * turns a nested artefact into a single segment the server cannot find; each
 * segment is encoded on its own instead.
 *
 * @param path - A slash-separated artefact path.
 * @returns The same path with each segment percent-encoded.
 */
export function encodeArtifactPath(path: string): string {
  return path.split("/").map((segment) => encodeURIComponent(segment)).join("/");
}
