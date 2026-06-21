// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio network NIR export helper

import type { NIRFormat } from "./api/client";

export const NETWORK_NIR_EXPORT_FILENAME = "network.nir.json";

export function networkNirJson(nir: NIRFormat): string {
  return JSON.stringify(nir, null, 2);
}

export function networkNirBlob(nir: NIRFormat): Blob {
  return new Blob([networkNirJson(nir)], { type: "application/json" });
}
