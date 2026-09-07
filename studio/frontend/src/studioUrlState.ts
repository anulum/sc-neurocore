// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio share URL state codec

/**
 * Putting a workspace in a link, and reading one back out of a link.
 *
 * The state is base64 in the URL **fragment**, which is the part a browser
 * never sends to a server. A shared link therefore carries the reader's setup
 * without it appearing in anyone's access log.
 *
 * Reading is deliberately narrow. A link is an untrusted document from an
 * unknown sender, so decoding refuses anything malformed by returning `null`
 * -- never a partly-applied state -- and restores only four fields: the model,
 * the current, the duration and the protocol. The payload carries more, and a
 * later build may widen this, but a link should not be able to replace
 * equations or parameters in a workspace someone is already using.
 *
 * The keys are one and two characters because the whole payload has to fit in
 * a URL that survives being pasted into a chat client.
 */

/** Whether the shared workspace was driven by a model or by an ODE. */
export type StudioUrlSourceMode = "model" | "ode";

/** The workspace being shared, as the panel holds it. */
export interface StudioShareUrlInput {
  sourceMode: StudioUrlSourceMode;
  selectedModelName: string;
  equations: string[];
  threshold: string;
  reset: string;
  modelParams: Record<string, number>;
  odeParams: Record<string, number>;
  odeInit: Record<string, number>;
  dt: number;
  duration: number;
  current: number;
  protocol: string;
}

/**
 * The parts of the address a share link is built from. Declared rather than
 * taking `Location`, so a test can supply one.
 */
export interface StudioShareUrlLocation {
  origin: string;
  pathname: string;
}

/** The one clipboard call this module makes. */
export interface StudioShareUrlClipboard {
  writeText(text: string): Promise<void>;
}

/**
 * The workspace as it travels in the fragment. The names are short because the
 * whole payload has to survive being pasted into a chat client.
 */
export interface StudioShareUrlPayload {
  m: StudioUrlSourceMode;
  mn: string;
  eq: string[];
  th: string;
  rs: string;
  p: Record<string, number>;
  i: Record<string, number>;
  dt: number;
  d: number;
  c: number;
  pr: string;
}

/**
 * What a link is allowed to set on startup: the model, the current, the
 * duration and the protocol. The payload carries more; a link deliberately
 * cannot replace equations or parameters.
 */
export interface StudioStartupHashState {
  selectedModelName: string;
  current: number;
  duration: number;
  protocol: string;
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
 * Read a number that must not be zero.
 *
 * @param value - The decoded value.
 * @param fallback - What to use instead.
 * @returns The number. Zero falls back, because a zero current or duration
 *   makes a run that cannot start.
 */
function finiteNumberOrDefault(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value !== 0
    ? value
    : fallback;
}

/**
 * Reduce a workspace to what travels in the link.
 *
 * Only the active parameter set is carried: sharing both would double the
 * payload to describe a run that used one of them.
 *
 * @param input - The workspace.
 * @returns The payload.
 */
export function studioShareUrlPayload(input: StudioShareUrlInput): StudioShareUrlPayload {
  return {
    m: input.sourceMode,
    mn: input.selectedModelName,
    eq: input.equations,
    th: input.threshold,
    rs: input.reset,
    p: input.sourceMode === "model" ? input.modelParams : input.odeParams,
    i: input.odeInit,
    dt: input.dt,
    d: input.duration,
    c: input.current,
    pr: input.protocol,
  };
}

/**
 * Encode a payload for the fragment.
 *
 * @param payload - The payload.
 * @param encodeBase64 - The encoder; the browser's `btoa` by default.
 * @returns The encoded text.
 */
export function encodeStudioSharePayload(
  payload: StudioShareUrlPayload,
  encodeBase64: (payload: string) => string = btoa,
): string {
  return encodeBase64(JSON.stringify(payload));
}

/**
 * Build the link that shares a workspace.
 *
 * @param input - The workspace.
 * @param location - The address to build it against.
 * @param encodeBase64 - The encoder; the browser's `btoa` by default.
 * @returns The link.
 */
export function buildStudioShareUrl(
  input: StudioShareUrlInput,
  location: StudioShareUrlLocation,
  encodeBase64: (payload: string) => string = btoa,
): string {
  const encodedState = encodeStudioSharePayload(studioShareUrlPayload(input), encodeBase64);
  return `${location.origin}${location.pathname}#${encodedState}`;
}

/**
 * Build the link and put it on the clipboard.
 *
 * The clipboard write is not awaited: the link is returned immediately so
 * the panel can show it whether or not the copy is permitted.
 *
 * @param input - The workspace.
 * @param location - The address to build it against.
 * @param clipboard - Where to write it.
 * @param encodeBase64 - The encoder; the browser's `btoa` by default.
 * @returns The link.
 */
export function copyStudioShareUrl(
  input: StudioShareUrlInput,
  location: StudioShareUrlLocation,
  clipboard: StudioShareUrlClipboard,
  encodeBase64: (payload: string) => string = btoa,
): string {
  const url = buildStudioShareUrl(input, location, encodeBase64);
  void clipboard.writeText(url);
  return url;
}

/**
 * Read a workspace out of a link's fragment.
 *
 * Anything malformed returns `null` rather than a partly-applied state: a
 * link is an untrusted document, and half of one is worse than none.
 *
 * @param hash - The fragment, with or without its leading `#`.
 * @param decodeBase64 - The decoder; the browser's `atob` by default.
 * @returns The four fields a link may set, or `null`.
 */
export function decodeStudioStartupHash(
  hash: string,
  decodeBase64: (payload: string) => string = atob,
): StudioStartupHashState | null {
  const encodedState = hash.startsWith("#") ? hash.slice(1) : hash;
  if (encodedState.length === 0) {
    return null;
  }
  try {
    const decodedState: unknown = JSON.parse(decodeBase64(encodedState));
    if (!isRecord(decodedState)) {
      return null;
    }
    if (
      (decodedState.m !== "model" && decodedState.m !== "ode")
      || typeof decodedState.mn !== "string"
      || decodedState.mn.length === 0
    ) {
      return null;
    }
    return {
      selectedModelName: decodedState.mn,
      current: finiteNumberOrDefault(decodedState.c, 10),
      duration: finiteNumberOrDefault(decodedState.d, 100),
      protocol: typeof decodedState.pr === "string" && decodedState.pr.length > 0
        ? decodedState.pr
        : "constant",
    };
  } catch {
    return null;
  }
}
