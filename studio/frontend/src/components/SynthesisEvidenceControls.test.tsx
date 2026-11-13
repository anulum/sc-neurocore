// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — exporting a synthesis run's evidence bundle
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { StudioEvidenceBundleResponse } from "../api/client";
import SynthesisEvidenceControls, { type SynthesisEvidenceControlsProps } from "./SynthesisEvidenceControls";

let root: Root | undefined;
let container: HTMLDivElement;
const onDownloadArtifact = vi.fn();
const onExport = vi.fn();

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  onDownloadArtifact.mockClear();
  onExport.mockClear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/**
 * A bundle of a finished synthesis run.
 *
 * @param paths - The artefacts it gathered.
 * @returns The bundle.
 */
function bundle(paths: string[]): StudioEvidenceBundleResponse {
  return {
    artifact_paths: paths,
    artifacts: paths.map((relative_path, index) => ({ relative_path, sha256: String(index).repeat(64), size_bytes: 2048 })),
    bundle_id: "eb_42",
    job_id: "sj_7",
    manifest: {},
    schema_version: "studio.evidence-bundle.v1",
    summary: {
      artifact_path_count: paths.length,
      entry_count: paths.length,
      entry_type_counts: {},
      evidence_classification_counts: {},
      source_job_count: 1,
      source_job_kind_counts: {},
      source_job_owner_counts: {},
    },
  };
}

/**
 * Render the controls.
 *
 * @param overrides - Props that differ from a finished run with no bundle yet.
 */
async function render(overrides: Partial<SynthesisEvidenceControlsProps> = {}): Promise<void> {
  const props: SynthesisEvidenceControlsProps = {
    bundle: null, error: null, jobId: "sj_7", loading: false, onDownloadArtifact, onExport, ...overrides,
  };
  await act(async () => { root?.render(<SynthesisEvidenceControls {...props} />); await Promise.resolve(); });
}

/** The export button. */
function exportButton(): HTMLButtonElement {
  const button = container.querySelector<HTMLButtonElement>('button[aria-label="Export synthesis evidence bundle"]');
  if (button === null) throw new Error("no export button");
  return button;
}

describe("SynthesisEvidenceControls", () => {
  it("waits for a job before it offers an export", async () => {
    await render({ jobId: null });
    expect(container.textContent).toContain("job pending");
    expect(exportButton().disabled).toBe(true);
    expect(exportButton().style.cursor).toBe("not-allowed");
  });

  it("exports a finished run's evidence", async () => {
    await render();
    expect(container.textContent).toContain("job sj_7");
    expect(exportButton().disabled).toBe(false);
    expect(exportButton().style.cursor).toBe("pointer");
    await act(async () => { exportButton().click(); await Promise.resolve(); });
    expect(onExport).toHaveBeenCalledTimes(1);
  });

  it("refuses a second export while one is on its way", async () => {
    await render({ loading: true });
    expect(exportButton().disabled).toBe(true);
    expect(exportButton().style.cursor).toBe("not-allowed");
  });

  it("names the bundle and downloads each of its artefacts", async () => {
    await render({ bundle: bundle(["netlist.json", "report.txt"]) });
    expect(container.textContent).toContain("bundle eb_42");
    const list = container.querySelector('[aria-label="Synthesis evidence bundle artefacts"]');
    const downloads = [...(list?.querySelectorAll("button") ?? [])];
    expect(downloads.map((button) => button.getAttribute("aria-label"))).toEqual([
      "Download synthesis evidence artefact netlist.json",
      "Download synthesis evidence artefact report.txt",
    ]);
    await act(async () => { downloads[1]?.click(); await Promise.resolve(); });
    expect(onDownloadArtifact).toHaveBeenCalledWith("report.txt");
  });

  it("names an empty bundle without listing artefacts", async () => {
    await render({ bundle: bundle([]) });
    expect(container.textContent).toContain("bundle eb_42");
    expect(container.querySelector('[aria-label="Synthesis evidence bundle artefacts"]')).toBeNull();
  });

  it("shows why an export failed, and nothing when it did not", async () => {
    await render();
    expect(container.textContent).not.toContain("refused");
    await render({ error: "the bundle was refused" });
    expect(container.textContent).toContain("the bundle was refused");
  });
});
