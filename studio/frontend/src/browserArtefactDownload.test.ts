// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio browser artefact download helper tests
import { describe, expect, it } from "vitest";

import {
  artefactDownloadName,
  downloadBrowserArtefact,
  type BrowserDownloadAnchor,
  type BrowserDownloadTarget,
} from "./browserArtefactDownload";

class FakeAnchor implements BrowserDownloadAnchor {
  download = "";
  href = "";
  clicked = false;

  click(): void {
    this.clicked = true;
  }
}

class FakeDownloadTarget implements BrowserDownloadTarget {
  readonly anchor = new FakeAnchor();
  createdPayload: Blob | null = null;
  revokedUrl: string | null = null;

  createAnchor(): BrowserDownloadAnchor {
    return this.anchor;
  }

  createObjectUrl(payload: Blob): string {
    this.createdPayload = payload;
    return "blob:studio-test";
  }

  revokeObjectUrl(url: string): void {
    this.revokedUrl = url;
  }
}

describe("browser artefact downloads", () => {
  it("uses the last path segment as the browser download name", () => {
    expect(artefactDownloadName("evidence/jobs/sj/artifacts/result.json")).toBe("result.json");
    expect(artefactDownloadName("/evidence/manifest.json")).toBe("manifest.json");
    expect(artefactDownloadName("")).toBe("studio-artefact");
  });

  it("downloads a blob through the supplied browser target and revokes the object URL", () => {
    const target = new FakeDownloadTarget();
    const payload = new Blob(["{}"], { type: "application/json" });

    downloadBrowserArtefact(payload, "evidence/manifest.json", target);

    expect(target.createdPayload).toBe(payload);
    expect(target.anchor.href).toBe("blob:studio-test");
    expect(target.anchor.download).toBe("manifest.json");
    expect(target.anchor.clicked).toBe(true);
    expect(target.revokedUrl).toBe("blob:studio-test");
  });

  it("revokes the object URL when anchor creation fails", () => {
    const target: BrowserDownloadTarget = {
      createAnchor: () => {
        throw new Error("anchor failed");
      },
      createObjectUrl: () => "blob:studio-test",
      revokeObjectUrl: (url) => {
        expect(url).toBe("blob:studio-test");
      },
    };

    expect(() => downloadBrowserArtefact(new Blob(["{}"]), "evidence/manifest.json", target))
      .toThrow("anchor failed");
  });
});
