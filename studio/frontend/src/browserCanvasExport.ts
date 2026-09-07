// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio browser canvas export helper

/**
 * Saving the plot on screen as an image.
 *
 * The canvas is selected from the document rather than passed in, because the
 * plot components do not hold a ref to it and adding one to reach this call
 * would put an export concern into every plot. That is a trade, and the cost is
 * that this only works when exactly one canvas is on screen -- which is the
 * Studio's layout, and why `selectCanvas` is injectable so a test can say so.
 */

import type { BrowserDownloadAnchor } from "./browserArtefactDownload";

/** What the exported image is saved as. */
export const STUDIO_CANVAS_PNG_FILENAME = "sc_neurocore_plot.png";

/** The one canvas call an export makes. */
export interface BrowserCanvasExportCanvas {
  toDataURL(type?: string, quality?: unknown): string;
}

/** Everything the export takes from the browser, so a test can supply it. */
export interface BrowserCanvasExportTarget {
  createAnchor: () => BrowserDownloadAnchor;
  selectCanvas: () => BrowserCanvasExportCanvas | null;
}

/**
 * The browser's own canvas and anchor.
 *
 * @returns The target.
 */
export function browserCanvasExportTarget(): BrowserCanvasExportTarget {
  return {
    createAnchor: () => document.createElement("a"),
    selectCanvas: () => document.querySelector("canvas"),
  };
}

/**
 * Save the plot on screen as a PNG.
 *
 * @param target - The browser calls to use; the real ones by default.
 * @returns Whether there was a canvas to save. `false` is not an error:
 *   a panel with no plot yet has nothing to export, and the caller shows
 *   that rather than a failure.
 */
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
