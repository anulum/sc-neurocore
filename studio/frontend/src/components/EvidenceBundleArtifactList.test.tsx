import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import EvidenceBundleArtifactList from "./EvidenceBundleArtifactList";

describe("EvidenceBundleArtifactList", () => {
  it("renders artifact metadata and download labels", () => {
    const html = renderToStaticMarkup(
      <EvidenceBundleArtifactList
        ariaLabel="Compile evidence bundle artifacts"
        artifacts={[
          {
            relative_path: "evidence/replay.json",
            sha256: "c".repeat(64),
            size_bytes: 128,
          },
          {
            relative_path: "evidence/large.json",
            sha256: "d".repeat(64),
            size_bytes: 2048,
          },
        ]}
        downloadLabelPrefix="Download compile evidence artifact"
        loading={false}
        onDownloadArtifact={() => undefined}
      />,
    );

    expect(html).toContain("Compile evidence bundle artifacts");
    expect(html).toContain("evidence/replay.json");
    expect(html).toContain("128 B - sha cccccccccccc");
    expect(html).toContain("Download compile evidence artifact evidence/replay.json");
    expect(html).toContain("evidence/large.json");
    expect(html).toContain("2.0 KiB - sha dddddddddddd");
  });

  it("renders nothing without artifacts", () => {
    const html = renderToStaticMarkup(
      <EvidenceBundleArtifactList
        ariaLabel="Empty evidence bundle artifacts"
        artifacts={[]}
        downloadLabelPrefix="Download empty evidence artifact"
        loading={false}
        onDownloadArtifact={() => undefined}
      />,
    );

    expect(html).toBe("");
  });
});
