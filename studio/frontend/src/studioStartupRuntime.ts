// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio startup hash browser runtime

/**
 * Reading the share fragment the page was opened with.
 *
 * Separated from the decoder so the decoding is testable without a browser and
 * this file holds the one thing that needs one: `window.location.hash`.
 */

import {
  decodeStudioStartupHash,
  type StudioStartupHashState,
} from "./studioUrlState";

/** The one thing startup needs from the browser: the address fragment. */
export interface StudioStartupRuntime {
  hash: string;
}

/**
 * Read the fragment this page was opened with.
 *
 * @returns The runtime, or `null` outside a browser.
 */
export function browserStudioStartupRuntime(): StudioStartupRuntime | null {
  return typeof window === "undefined" ? null : { hash: window.location.hash };
}

/**
 * Read the workspace a share link was opened with.
 *
 * @param runtime - The fragment; read from the browser by default.
 * @param decodeBase64 - The decoder; the browser's `atob` by default.
 * @returns What the link sets, or `null` when there is no link or it is
 *   not readable.
 */
export function readStudioStartupHashState(
  runtime: StudioStartupRuntime | null = browserStudioStartupRuntime(),
  decodeBase64: (payload: string) => string = atob,
): StudioStartupHashState | null {
  return runtime === null ? null : decodeStudioStartupHash(runtime.hash, decodeBase64);
}
