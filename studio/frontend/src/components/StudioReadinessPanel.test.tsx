// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio readiness panel tests
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { StudioReadinessModel } from "../studioReadiness";
import StudioReadinessPanel from "./StudioReadinessPanel";

const readinessModel: StudioReadinessModel = {
  actionLabel: "Enable route policies",
  blockingCount: 2,
  headline: "Readiness blocked",
  items: [
    {
      action: "Enable route policies",
      key: "routes",
      label: "Route policies",
      status: "blocked",
      value: "disabled",
    },
    {
      action: "Resolve unavailable capability",
      key: "capabilities",
      label: "Capabilities",
      status: "warning",
      value: "9/10 healthy",
    },
  ],
  posture: "blocked",
  readyCount: 0,
  subhead: "0/2 checks ready, 2 blocked",
  warningCount: 1,
};

describe("StudioReadinessPanel", () => {
  it("renders blocking readiness actions without local path leakage", () => {
    const html = renderToStaticMarkup(
      <StudioReadinessPanel
        model={readinessModel}
        onOpenAdmin={() => undefined}
        onRefresh={() => undefined}
      />,
    );

    expect(html).toContain("Readiness");
    expect(html).toContain("Readiness blocked");
    expect(html).toContain("2 blockers / 1 warnings");
    expect(html).toContain("Route policies");
    expect(html).toContain("Enable route policies");
    expect(html).toContain("Capabilities");
    expect(html).toContain("Resolve unavailable capability");
    expect(html).toContain("Refresh");
    expect(html).toContain("Open admin");
    expect(html).not.toContain("/tmp/");
  });

  it("allows Admin to relabel the primary action", () => {
    const html = renderToStaticMarkup(
      <StudioReadinessPanel
        model={readinessModel}
        onOpenAdmin={() => undefined}
        onRefresh={() => undefined}
        primaryActionLabel="Refresh status"
      />,
    );

    expect(html).toContain("Refresh status");
    expect(html).not.toContain("Open admin");
  });
});
