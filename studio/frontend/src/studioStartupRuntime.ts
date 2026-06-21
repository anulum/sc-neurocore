// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio startup hash browser runtime

import {
  decodeStudioStartupHash,
  type StudioStartupHashState,
} from "./studioUrlState";

export interface StudioStartupRuntime {
  hash: string;
}

export function browserStudioStartupRuntime(): StudioStartupRuntime | null {
  return typeof window === "undefined" ? null : { hash: window.location.hash };
}

export function readStudioStartupHashState(
  runtime: StudioStartupRuntime | null = browserStudioStartupRuntime(),
  decodeBase64: (payload: string) => string = atob,
): StudioStartupHashState | null {
  return runtime === null ? null : decodeStudioStartupHash(runtime.hash, decodeBase64);
}
