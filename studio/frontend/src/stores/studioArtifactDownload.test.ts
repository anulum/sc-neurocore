// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Verified evidence artifact download workflow

import { afterEach, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";
import { useStudioStore } from "./studio";

const initial = useStudioStore.getState();
afterEach(() => { useStudioStore.setState(initial, true); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it.each(["manifest", "tampered", "short"])("verifies artifact bytes before the browser save: %s", async (contents) => {
  useStudioStore.setState({ projectEvidenceBundle: {
    artifact_paths: ["evidence/manifest.json"], bundle_id: "seb_controlled", job_id: "sj_exact",
    manifest: {}, schema_version: "studio.evidence-bundle.v1",
    artifacts: [{ relative_path: "evidence/manifest.json", size_bytes: 8,
      sha256: createHash("sha256").update("manifest").digest("hex") }],
    summary: { artifact_path_count: 1, entry_count: 0, entry_type_counts: {},
      evidence_classification_counts: {}, source_job_count: 0,
      source_job_kind_counts: {}, source_job_owner_counts: {} },
  } });
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(contents));
  vi.stubGlobal("fetch", fetch);
  const click = vi.fn();
  vi.stubGlobal("document", { createElement: () => ({ href: "", download: "", click }) });
  const createUrl = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:controlled");
  const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
  await useStudioStore.getState().downloadEvidenceBundleArtifactForSurface("project", "evidence/manifest.json");
  const requestedUrl = fetch.mock.calls[0]?.[0];
  if (typeof requestedUrl !== "string") throw new Error("Expected string artifact URL");
  expect(requestedUrl).toContain("sj_exact");
  if (contents === "manifest") {
    expect(click).toHaveBeenCalledOnce();
    expect(revoke).toHaveBeenCalledWith("blob:controlled");
    expect(useStudioStore.getState().projectEvidenceBundleError).toBeNull();
  } else {
    expect(click).not.toHaveBeenCalled();
    expect(createUrl).not.toHaveBeenCalled();
    expect(useStudioStore.getState().projectEvidenceBundleError).toContain("mismatch");
  }
  expect(useStudioStore.getState().evidenceBundleError).toBeNull();
});
