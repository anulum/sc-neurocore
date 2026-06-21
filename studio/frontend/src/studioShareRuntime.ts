// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio share URL browser runtime

import {
  buildStudioShareUrl,
  type StudioShareUrlClipboard,
  type StudioShareUrlInput,
  type StudioShareUrlLocation,
} from "./studioUrlState";

export interface StudioShareRuntime {
  clipboard: StudioShareUrlClipboard | null;
  location: StudioShareUrlLocation;
}

export type StudioShareRuntimeResult =
  | { ok: true; url: string }
  | { ok: false; message: string };

export function browserStudioShareRuntime(): StudioShareRuntime | null {
  if (typeof window === "undefined") {
    return null;
  }
  return {
    clipboard: typeof navigator === "undefined" ? null : navigator.clipboard ?? null,
    location: window.location,
  };
}

export async function copyStudioShareUrlInRuntime(
  input: StudioShareUrlInput,
  runtime: StudioShareRuntime | null = browserStudioShareRuntime(),
  encodeBase64: (payload: string) => string = btoa,
): Promise<StudioShareRuntimeResult> {
  if (runtime === null) {
    return { ok: false, message: "Share URL is available only in a browser session." };
  }
  if (runtime.clipboard === null) {
    return { ok: false, message: "Clipboard access is unavailable in this browser session." };
  }
  const url = buildStudioShareUrl(input, runtime.location, encodeBase64);
  try {
    await runtime.clipboard.writeText(url);
  } catch (error: unknown) {
    return {
      ok: false,
      message: error instanceof Error ? error.message : "Clipboard write failed.",
    };
  }
  return { ok: true, url };
}
