// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio share URL browser runtime

/**
 * Copying a share link, in a browser that may not let you.
 *
 * Every way this can fail produces a sentence rather than a throw: no browser
 * at all, no clipboard in this context, or a clipboard that refused. The panel
 * shows one line either way, and the link itself is still built, so a reader
 * who cannot copy can still select it.
 *
 * The status clears itself after a couple of seconds through an injectable
 * scheduler, so a test can advance it without waiting.
 */

import {
  buildModelLinkUrl,
  buildStudioShareUrl,
  type StudioShareUrlClipboard,
  type StudioShareUrlInput,
  type StudioShareUrlLocation,
} from "./studioUrlState";

/** How long the copy confirmation stays on screen. */
export const STUDIO_SHARE_STATUS_CLEAR_DELAY_MS = 2000;

/** What copying needs from the browser: a clipboard, and an address. */
export interface StudioShareRuntime {
  clipboard: StudioShareUrlClipboard | null;
  location: StudioShareUrlLocation;
}

/** The link that was copied, or the sentence explaining why it was not. */
export type StudioShareRuntimeResult =
  | { ok: true; url: string }
  | { ok: false; message: string };

/** The patch that puts the copy status on screen. */
export interface StudioShareStatusStatePatch {
  error: string;
}

/** The patch that takes it off again. */
export interface StudioShareStatusClearedStatePatch {
  error: null;
}

/** The handle a scheduled clear returns, so a caller can cancel it. */
export type StudioShareStatusClearTimer = ReturnType<typeof setTimeout>;

/**
 * How the clear is scheduled. Declared rather than calling `setTimeout`
 * directly, so a test can run the delay without waiting for it.
 */
export interface StudioShareStatusClearScheduler {
  setTimeout(callback: () => void, delayMs: number): StudioShareStatusClearTimer;
}

/**
 * Read the clipboard the browser is actually offering.
 *
 * `lib.dom` declares `navigator.clipboard` as always present, and it is not:
 * outside a secure context the property is missing, and a page served over
 * plain HTTP would throw on the first copy instead of showing the fallback.
 * The check is structural rather than a comparison against `undefined`,
 * because a comparison is what the type declaration says can never fail.
 *
 * @param nav - The navigator.
 * @returns The clipboard, or `null` when this context has none.
 */
function presentClipboard(nav: Navigator): Clipboard | null {
  const candidate: unknown = nav.clipboard;
  if (typeof candidate !== "object" || candidate === null || !("writeText" in candidate)) {
    return null;
  }
  const { writeText } = candidate;
  return typeof writeText === "function" ? nav.clipboard : null;
}

/**
 * Read what this browser offers for sharing.
 *
 * @returns The runtime, or `null` outside a browser.
 */
export function browserStudioShareRuntime(): StudioShareRuntime | null {
  if (typeof window === "undefined") {
    return null;
  }
  return {
    clipboard: typeof navigator === "undefined" ? null : presentClipboard(navigator),
    location: window.location,
  };
}

/**
 * The browser's own timer.
 *
 * @returns The scheduler.
 */
export function browserStudioShareStatusClearScheduler(): StudioShareStatusClearScheduler {
  return {
    setTimeout: (callback, delayMs) => setTimeout(callback, delayMs),
  };
}

/**
 * Turn a copy attempt into the line the panel shows.
 *
 * Success and failure both produce a line: silence after a click reads as
 * a broken button.
 *
 * @param result - What the attempt produced.
 * @returns The patch.
 */
export function studioShareStatusState(
  result: StudioShareRuntimeResult,
): StudioShareStatusStatePatch {
  return { error: result.ok ? "URL copied to clipboard" : result.message };
}

/**
 * Clear the copy status.
 *
 * @returns The patch.
 */
export function studioShareStatusClearedState(): StudioShareStatusClearedStatePatch {
  return { error: null };
}

/**
 * Schedule the status to clear itself.
 *
 * @param clearStatus - What to call when the delay elapses.
 * @param scheduler - The scheduler; the browser's own by default.
 * @param delayMs - How long to wait.
 * @returns The handle, so a caller unmounting can cancel it.
 */
export function scheduleStudioShareStatusClear(
  clearStatus: () => void,
  scheduler: StudioShareStatusClearScheduler = browserStudioShareStatusClearScheduler(),
  delayMs: number = STUDIO_SHARE_STATUS_CLEAR_DELAY_MS,
): StudioShareStatusClearTimer {
  return scheduler.setTimeout(clearStatus, delayMs);
}

/**
 * Build the share link and copy it.
 *
 * @param input - The workspace to share.
 * @param runtime - The browser's clipboard and address; read by default.
 * @param encodeBase64 - The encoder; the browser's `btoa` by default.
 * @returns The link, or the sentence explaining why it was not copied.
 */
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

/**
 * Build the link that opens one model and copy it.
 *
 * @param modelName - The catalogue identity.
 * @param runtime - The browser's clipboard and address; read by default.
 * @returns The link, or the sentence explaining why it was not copied.
 */
export async function copyModelLinkInRuntime(
  modelName: string,
  runtime: StudioShareRuntime | null = browserStudioShareRuntime(),
): Promise<StudioShareRuntimeResult> {
  if (runtime === null) {
    return { ok: false, message: "A model link is available only in a browser session." };
  }
  if (runtime.clipboard === null) {
    return { ok: false, message: "Clipboard access is unavailable in this browser session." };
  }
  const url = buildModelLinkUrl(modelName, runtime.location);
  try {
    await runtime.clipboard.writeText(url);
  } catch (error: unknown) {
    return { ok: false, message: error instanceof Error ? error.message : "Clipboard write failed." };
  }
  return { ok: true, url };
}
