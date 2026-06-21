// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio browser canvas export helper

import type { BrowserDownloadAnchor } from "./browserArtefactDownload";

export const STUDIO_CANVAS_PNG_FILENAME = "sc_neurocore_plot.png";

export interface BrowserCanvasExportCanvas {
  toDataURL(type?: string, quality?: unknown): string;
}

export interface BrowserCanvasExportTarget {
  createAnchor: () => BrowserDownloadAnchor;
  selectCanvas: () => BrowserCanvasExportCanvas | null;
}

export function browserCanvasExportTarget(): BrowserCanvasExportTarget {
  return {
    createAnchor: () => document.createElement("a"),
    selectCanvas: () => document.querySelector("canvas"),
  };
}

export function downloadCanvasPng(
  target: BrowserCanvasExportTarget = browserCanvasExportTarget(),
): boolean {
  const canvas = target.selectCanvas();
  if (canvas === null) {
    return false;
  }
  const anchor = target.createAnchor();
  anchor.href = canvas.toDataURL("image/png", 1.0);
  anchor.download = STUDIO_CANVAS_PNG_FILENAME;
  anchor.click();
  return true;
}
