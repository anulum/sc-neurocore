// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio network NIR export helper

/**
 * Exporting the canvas network as NIR.
 *
 * Like the other exports in this Studio, this is a plan rather than an act:
 * the caller gets the bytes, the filename and a `writeArtefact` it may never
 * call, so a view can name the file it would write and a test can read the
 * bytes with no download starting.
 */

import type { NIRFormat } from "./api/client";
import { downloadBrowserArtefact } from "./browserArtefactDownload";

/** What an exported network is saved as. */
export const NETWORK_NIR_EXPORT_FILENAME = "network.nir.json";

/** How the export reaches disk. Injected so a test can watch without one happening. */
export type NetworkNirExportDownloader = (payload: Blob, filename: string) => void;

/** The bytes of the export and the name to save them under. */
export interface NetworkNirExportArtefact {
  blob: Blob;
  filename: string;
}

/** The export, with the call that writes it. */
export interface NetworkNirExportPlan {
  artefact: NetworkNirExportArtefact;
  writeArtefact: (downloader?: NetworkNirExportDownloader) => void;
}

/**
 * Render a network as indented NIR JSON.
 *
 * Indented because the file is read and diffed by people, not only by
 * machines.
 *
 * @param nir - The network.
 * @returns Its JSON text.
 */
export function networkNirJson(nir: NIRFormat): string {
  return JSON.stringify(nir, null, 2);
}

/**
 * Render a network as a JSON blob.
 *
 * @param nir - The network.
 * @returns The blob.
 */
export function networkNirBlob(nir: NIRFormat): Blob {
  return new Blob([networkNirJson(nir)], { type: "application/json" });
}

/**
 * Build the export for a network.
 *
 * @param nir - The network.
 * @returns Its bytes and filename.
 */
export function networkNirExport(nir: NIRFormat): NetworkNirExportArtefact {
  return {
    blob: networkNirBlob(nir),
    filename: NETWORK_NIR_EXPORT_FILENAME,
  };
}

/**
 * Build the plan that writes a network export.
 *
 * @param nir - The network.
 * @returns The plan. Its `writeArtefact` uses the browser download unless
 *   the caller passes another downloader.
 */
export function networkNirExportPlan(nir: NIRFormat): NetworkNirExportPlan {
  const artefact = networkNirExport(nir);
  return {
    artefact,
    writeArtefact: (downloader = downloadBrowserArtefact) => {
      downloader(artefact.blob, artefact.filename);
    },
  };
}
