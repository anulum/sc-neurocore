// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import ProjectEvidenceStrip from "./ProjectEvidenceStrip";

describe("ProjectEvidenceStrip", () => {
  it("renders path-free project save evidence", () => {
    const html = renderToStaticMarkup(
      <ProjectEvidenceStrip
        artifacts={[
          {
            relative_path: "evidence/projects/saved-network.json",
            sha256: "c".repeat(64),
            size_bytes: 1536,
          },
        ]}
        evidence={{
          classification: "project_workspace",
          name: "saved-network",
          projectDigest: "aaaaaaaaaaaa",
          schemaVersion: "studio.project-save.v1",
          stateDigest: "bbbbbbbbbbbb",
        }}
        exportBundleId="seb_project"
        exportError={null}
        exportJobId="sj_project"
        loading={false}
        onDownloadArtifact={() => undefined}
        onExportBundle={() => undefined}
      />,
    );

    expect(html).toContain("project_workspace");
    expect(html).toContain("saved-network");
    expect(html).toContain("state sha");
    expect(html).toContain("bbbbbbbbbbbb");
    expect(html).toContain("project sha");
    expect(html).toContain("aaaaaaaaaaaa");
    expect(html).toContain("studio.project-save.v1");
    expect(html).toContain("Export saved-network project evidence bundle");
    expect(html).toContain("Project evidence bundle artifacts");
    expect(html).toContain("evidence/projects/saved-network.json");
    expect(html).toContain("1.5 KiB - sha cccccccccccc");
    expect(html).toContain("Download project evidence artifact evidence/projects/saved-network.json");
    expect(html).toContain("seb_project");
    expect(html).toContain("sj_project");
  });
});
