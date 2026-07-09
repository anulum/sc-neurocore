// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import EvidenceSummaryStrip from "./EvidenceSummaryStrip";

describe("EvidenceSummaryStrip", () => {
  it("renders path-free evidence label pairs", () => {
    const html = renderToStaticMarkup(
      <EvidenceSummaryStrip
        variant="grid"
        items={[
          { label: "class", value: "compile" },
          { label: "artifact", value: "pipeline/evidence.json" },
        ]}
      />,
    );

    expect(html).toContain("class compile");
    expect(html).toContain("artifact pipeline/evidence.json");
  });

  it("renders overlay evidence without changing label content", () => {
    const html = renderToStaticMarkup(
      <EvidenceSummaryStrip
        variant="overlay"
        items={[
          { label: "class", value: "simulation" },
          { label: "out", value: "aaaaaaaaaa" },
        ]}
      />,
    );

    expect(html).toContain("class simulation");
    expect(html).toContain("out aaaaaaaaaa");
  });
});
