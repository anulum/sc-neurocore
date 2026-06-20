import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import ProjectEvidenceStrip from "./ProjectEvidenceStrip";

describe("ProjectEvidenceStrip", () => {
  it("renders path-free project save evidence", () => {
    const html = renderToStaticMarkup(
      <ProjectEvidenceStrip
        evidence={{
          classification: "project_workspace",
          name: "saved-network",
          projectDigest: "aaaaaaaaaaaa",
          schemaVersion: "studio.project-save.v1",
          stateDigest: "bbbbbbbbbbbb",
        }}
      />,
    );

    expect(html).toContain("project_workspace");
    expect(html).toContain("saved-network");
    expect(html).toContain("state sha");
    expect(html).toContain("bbbbbbbbbbbb");
    expect(html).toContain("project sha");
    expect(html).toContain("aaaaaaaaaaaa");
    expect(html).toContain("studio.project-save.v1");
  });
});
