// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Development-preview labelling tests
import { describe, expect, it } from "vitest";

import {
  DEVELOPMENT_PREVIEW_LABEL,
  developmentPreviewBannerModel,
  shouldShowDevelopmentPreviewBanner,
} from "./developmentPreview";

describe("development preview labelling", () => {
  it("exposes the canonical Development preview label string", () => {
    expect(DEVELOPMENT_PREVIEW_LABEL).toBe("Development preview");
  });

  it("shows the banner for development and production until release-validated", () => {
    expect(shouldShowDevelopmentPreviewBanner("development")).toBe(true);
    expect(shouldShowDevelopmentPreviewBanner("production")).toBe(true);
    expect(shouldShowDevelopmentPreviewBanner(null)).toBe(true);
    expect(shouldShowDevelopmentPreviewBanner("production", { releaseValidated: true })).toBe(
      false,
    );
  });

  it("builds visible banner copy including the required label", () => {
    const model = developmentPreviewBannerModel("development");
    expect(model.visible).toBe(true);
    expect(model.label).toBe("Development preview");
    expect(model.detail.toLowerCase()).toContain("not a production-validated");
  });
});
