// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio browser artefact download helper

export interface BrowserDownloadAnchor {
  download: string;
  href: string;
  click: () => void;
}

export interface BrowserDownloadTarget {
  createAnchor: () => BrowserDownloadAnchor;
  createObjectUrl: (payload: Blob) => string;
  revokeObjectUrl: (url: string) => void;
}

export function artefactDownloadName(relativePath: string): string {
  return relativePath.split("/").filter(Boolean).pop() ?? "studio-artefact";
}

export function browserDownloadTarget(): BrowserDownloadTarget {
  return {
    createAnchor: () => document.createElement("a"),
    createObjectUrl: (payload) => URL.createObjectURL(payload),
    revokeObjectUrl: (url) => URL.revokeObjectURL(url),
  };
}

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
