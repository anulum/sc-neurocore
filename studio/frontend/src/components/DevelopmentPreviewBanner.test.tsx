// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Development preview banner tests
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import DevelopmentPreviewBanner from "./DevelopmentPreviewBanner";

describe("DevelopmentPreviewBanner", () => {
  it("renders the Development preview label on the main shell surface", () => {
    const html = renderToStaticMarkup(
      <DevelopmentPreviewBanner deploymentProfile="development" />,
    );
    expect(html).toContain("data-testid=\"development-preview-banner\"");
    expect(html).toContain("Development preview");
    expect(html).toContain("not a production-validated");
  });

  it("hides when releaseValidated is true", () => {
    const html = renderToStaticMarkup(
      <DevelopmentPreviewBanner deploymentProfile="production" releaseValidated />,
    );
    expect(html).toBe("");
  });
});
