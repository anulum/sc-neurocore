// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio network NIR export helper

import type { NIRFormat } from "./api/client";
import { downloadBrowserArtefact } from "./browserArtefactDownload";

export const NETWORK_NIR_EXPORT_FILENAME = "network.nir.json";

export type NetworkNirExportDownloader = (payload: Blob, filename: string) => void;

export interface NetworkNirExportArtefact {
  blob: Blob;
  filename: string;
}

export interface NetworkNirExportPlan {
  artefact: NetworkNirExportArtefact;
  writeArtefact: (downloader?: NetworkNirExportDownloader) => void;
}

export function networkNirJson(nir: NIRFormat): string {
  return JSON.stringify(nir, null, 2);
}

export function networkNirBlob(nir: NIRFormat): Blob {
  return new Blob([networkNirJson(nir)], { type: "application/json" });
}

export function networkNirExport(nir: NIRFormat): NetworkNirExportArtefact {
  return {
    blob: networkNirBlob(nir),
    filename: NETWORK_NIR_EXPORT_FILENAME,
  };
}

export function networkNirExportPlan(nir: NIRFormat): NetworkNirExportPlan {
  const artefact = networkNirExport(nir);
  return {
    artefact,
    writeArtefact: (downloader = downloadBrowserArtefact) => {
      downloader(artefact.blob, artefact.filename);
    },
  };
}
