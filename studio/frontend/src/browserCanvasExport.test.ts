// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio browser canvas export helper tests

import { describe, expect, it } from "vitest";

import type { BrowserDownloadAnchor } from "./browserArtefactDownload";
import {
  STUDIO_CANVAS_PNG_FILENAME,
  downloadCanvasPng,
  type BrowserCanvasExportCanvas,
  type BrowserCanvasExportTarget,
} from "./browserCanvasExport";

class FakeAnchor implements BrowserDownloadAnchor {
  download = "";
  href = "";
  clicked = false;

  click(): void {
    this.clicked = true;
  }
}

class FakeCanvas implements BrowserCanvasExportCanvas {
  requestedType: string | undefined;
  requestedQuality: unknown;

  toDataURL(type?: string, quality?: unknown): string {
    this.requestedType = type;
    this.requestedQuality = quality;
    return "data:image/png;base64,plot";
  }
}

describe("browser canvas export", () => {
  it("returns false when no canvas exists", () => {
    const target: BrowserCanvasExportTarget = {
      createAnchor: () => new FakeAnchor(),
      selectCanvas: () => null,
    };

    expect(downloadCanvasPng(target)).toBe(false);
  });

  it("exports the selected canvas as a PNG download", () => {
    const anchor = new FakeAnchor();
    const canvas = new FakeCanvas();
    const target: BrowserCanvasExportTarget = {
      createAnchor: () => anchor,
      selectCanvas: () => canvas,
    };

    expect(downloadCanvasPng(target)).toBe(true);

    expect(canvas.requestedType).toBe("image/png");
    expect(canvas.requestedQuality).toBe(1.0);
    expect(anchor.href).toBe("data:image/png;base64,plot");
    expect(anchor.download).toBe(STUDIO_CANVAS_PNG_FILENAME);
    expect(anchor.clicked).toBe(true);
  });
});
