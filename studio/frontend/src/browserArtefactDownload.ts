// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio browser artefact download helper

/**
 * Handing a file to the browser to save.
 *
 * The object URL is revoked in a `finally`, so a click handler that throws
 * still releases the blob rather than leaking it for the life of the document.
 *
 * Every browser call goes through an injectable target. That is what lets the
 * download path be tested without a real DOM, and it is why this file has no
 * direct `document` or `URL` reference outside `browserDownloadTarget`.
 */

/** The three anchor properties a download needs. */
export interface BrowserDownloadAnchor {
  download: string;
  href: string;
  click: () => void;
}

/** Everything the download takes from the browser, so a test can supply it. */
export interface BrowserDownloadTarget {
  createAnchor: () => BrowserDownloadAnchor;
  createObjectUrl: (payload: Blob) => string;
  revokeObjectUrl: (url: string) => void;
}

/**
 * Name the file a server artefact is saved as.
 *
 * @param relativePath - The artefact's path on the server.
 * @returns Its last path segment, or a generic name when the path has no
 *   segments -- a file must be saved under some name.
 */
export function artefactDownloadName(relativePath: string): string {
  return relativePath.split("/").filter(Boolean).pop() ?? "studio-artefact";
}

/**
 * The browser's own download machinery.
 *
 * @returns The target.
 */
export function browserDownloadTarget(): BrowserDownloadTarget {
  return {
    createAnchor: () => document.createElement("a"),
    createObjectUrl: (payload) => URL.createObjectURL(payload),
    revokeObjectUrl: (url) => {
      URL.revokeObjectURL(url);
    },
  };
}

/**
 * Save a payload to the reader's disk.
 *
 * @param payload - The bytes.
 * @param relativePath - The artefact's path, which names the file.
 * @param target - The browser calls to use; the real ones by default.
 */
export function downloadBrowserArtefact(
  payload: Blob,
  relativePath: string,
  target: BrowserDownloadTarget = browserDownloadTarget(),
): void {
  const url = target.createObjectUrl(payload);
  try {
    const anchor = target.createAnchor();
    anchor.href = url;
    anchor.download = artefactDownloadName(relativePath);
    anchor.click();
  } finally {
    target.revokeObjectUrl(url);
  }
}
