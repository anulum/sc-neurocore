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
});
