// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio network NIR export and import helpers

/**
 * Exporting the canvas network as a NIR file, and reading one back.
 *
 * Like the other exports in this Studio, the export is a plan rather than an
 * act: the caller gets the bytes, the filename and a `writeArtefact` it may
 * never call, so a view can name the file it would write and a test can read
 * the bytes with no download starting.
 */

import type {
  GraphEnvelope,
  NIRExportResult,
  NIRImportRequest,
  NIRImportResult,
} from "./api/client";
import { downloadBrowserArtefact } from "./browserArtefactDownload";

/** What an exported network is saved as: a NIR graph in HDF5. */
export const NETWORK_NIR_EXPORT_FILENAME = "network.nir";

/** The media type of a NIR file. */
export const NETWORK_NIR_MEDIA_TYPE = "application/x-hdf5";

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
 * Decode base64 text into bytes.
 *
 * @param text - The base64 text.
 * @returns The bytes.
 */
export function base64Bytes(text: string): Uint8Array<ArrayBuffer> {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

/**
 * Encode bytes as base64 text.
 *
 * Built in chunks, because spreading a large file into one call exceeds the
 * engine's argument limit.
 *
 * @param bytes - The bytes.
 * @returns The base64 text.
 */
export function bytesBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunk) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunk));
  }
  return btoa(binary);
}

/**
 * Build the export for a network.
 *
 * @param exported - What the export route returned.
 * @returns The file's bytes and filename.
 */
export function networkNirExport(exported: NIRExportResult): NetworkNirExportArtefact {
  return {
    blob: new Blob([base64Bytes(exported.content_base64)], { type: NETWORK_NIR_MEDIA_TYPE }),
    filename: NETWORK_NIR_EXPORT_FILENAME,
  };
}

/**
 * Build the plan that writes a network export.
 *
 * @param exported - What the export route returned.
 * @returns The plan. Its `writeArtefact` uses the browser download unless
 *   the caller passes another downloader.
 */
export function networkNirExportPlan(exported: NIRExportResult): NetworkNirExportPlan {
  const artefact = networkNirExport(exported);
  return {
    artefact,
    writeArtefact: (downloader = downloadBrowserArtefact) => {
      downloader(artefact.blob, artefact.filename);
    },
  };
}

/**
 * Turn a chosen file into the request the import route takes.
 *
 * A `.json` file is a graph envelope an earlier build saved; anything else is
 * sent as the bytes of a NIR file, which the server reads and refuses if it is
 * not one.
 *
 * @param file - The chosen file.
 * @returns The import request.
 */
export async function networkNirImportRequest(file: Blob & { name: string }): Promise<NIRImportRequest> {
  if (file.name.toLowerCase().endsWith(".json")) {
    return JSON.parse(await file.text()) as GraphEnvelope;
  }
  return { content_base64: bytesBase64(new Uint8Array(await file.arrayBuffer())) };
}

/**
 * Say in one notice what an export wrote and what it could not carry.
 *
 * @param exported - What the export route returned.
 * @returns The notice.
 */
export function networkNirExportNotice(exported: NIRExportResult): string {
  return `Exported ${NETWORK_NIR_EXPORT_FILENAME} (NIR ${exported.nir_version}). `
    + `Not carried exactly: ${exported.notes.join("; ")}.`;
}

/**
 * Say in one notice where an imported network came from and what reading it assumed.
 *
 * @param imported - What the import route returned.
 * @returns The notice.
 */
export function networkNirImportNotice(imported: NIRImportResult): string {
  const source = {
    studio: "a NIR file this Studio wrote; every tensor matched its recorded network",
    foreign: "a NIR file from another tool",
    "studio-envelope": "a saved graph envelope",
  }[imported.origin];
  const assumed = imported.notes.length === 0 ? "" : ` Assumed: ${imported.notes.join("; ")}.`;
  return `Imported ${source}.${assumed}`;
}
