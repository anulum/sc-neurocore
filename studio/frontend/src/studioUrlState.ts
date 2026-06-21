// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio share URL state codec

export type StudioUrlSourceMode = "model" | "ode";

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

export interface StudioShareUrlLocation {
  origin: string;
  pathname: string;
}

export interface StudioShareUrlClipboard {
  writeText(text: string): Promise<void>;
}

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

export interface StudioStartupHashState {
  selectedModelName: string;
  current: number;
  duration: number;
  protocol: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumberOrDefault(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value !== 0
    ? value
    : fallback;
}

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

export function encodeStudioSharePayload(
  payload: StudioShareUrlPayload,
  encodeBase64: (payload: string) => string = btoa,
): string {
  return encodeBase64(JSON.stringify(payload));
}

export function buildStudioShareUrl(
  input: StudioShareUrlInput,
  location: StudioShareUrlLocation,
  encodeBase64: (payload: string) => string = btoa,
): string {
  const encodedState = encodeStudioSharePayload(studioShareUrlPayload(input), encodeBase64);
  return `${location.origin}${location.pathname}#${encodedState}`;
}

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
