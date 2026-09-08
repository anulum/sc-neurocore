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
import { evidenceBundleDownloadSelection, type EvidenceBundleSurface } from "../evidenceBundles";

const initial = useStudioStore.getState();
afterEach(() => { useStudioStore.setState(initial, true); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

/**
 * Build a declared artifact with independently computed byte digest.
 *
 * @returns A complete evidence bundle fixture.
 */
function bundle() {
  return {
    artifact_paths: ["evidence/manifest.json"], bundle_id: "seb_controlled", job_id: "sj_exact",
    manifest: {}, schema_version: "studio.evidence-bundle.v1",
    artifacts: [{ relative_path: "evidence/manifest.json", size_bytes: 8,
      sha256: createHash("sha256").update("manifest").digest("hex") }],
    summary: { artifact_path_count: 1, entry_count: 0, entry_type_counts: {},
      evidence_classification_counts: {}, source_job_count: 0,
      source_job_kind_counts: {}, source_job_owner_counts: {} },
  };
}

it.each(["manifest", "tampered", "short"])("verifies artifact bytes before the browser save: %s", async (contents) => {
  useStudioStore.setState({ projectEvidenceBundle: bundle() });
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

it.each(["admin", "project", "compile", "synthesis"] as const)("keeps late %s download errors out of a replacement bundle", async (surface: EvidenceBundleSurface) => {
  const field = surface === "admin" ? "evidenceBundle" : `${surface}EvidenceBundle` as const;
  useStudioStore.setState({ [field]: bundle() });
  let rejectFetch: (error: Error) => void = () => { throw new Error("Fetch did not start"); };
  vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockImplementation(() => new Promise((_resolve, reject) => { rejectFetch = reject; })));
  const download = useStudioStore.getState().downloadEvidenceBundleArtifactForSurface(surface, "evidence/manifest.json");
  const replacement = { ...bundle(), bundle_id: "seb_new", job_id: "sj_new" };
  const errorKey = evidenceBundleDownloadSelection(surface, useStudioStore.getState()).error;
  useStudioStore.setState({ [field]: replacement, [errorKey]: "New export diagnostic" });
  rejectFetch(new Error("Old download failed"));
  await download;
  expect(useStudioStore.getState()[errorKey]).toBe("New export diagnostic");
  expect(evidenceBundleDownloadSelection(surface, useStudioStore.getState()).bundle).toBe(replacement);
});

it("does not replace a newer download's diagnostic with an older failure", async () => {
  useStudioStore.setState({ projectEvidenceBundle: bundle() });
  let rejectOld: (error: Error) => void = () => { throw new Error("Fetch did not start"); };
  vi.stubGlobal("fetch", vi.fn<typeof fetch>()
    .mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectOld = reject; }))
    .mockRejectedValueOnce(new Error("New download failed")));
  const old = useStudioStore.getState().downloadEvidenceBundleArtifactForSurface("project", "evidence/manifest.json");
  await useStudioStore.getState().downloadEvidenceBundleArtifactForSurface("project", "evidence/manifest.json");
  const diagnostic = useStudioStore.getState().projectEvidenceBundleError;
  expect(diagnostic).toContain("New download failed");
  rejectOld(new Error("Old download failed"));
  await old;
  expect(useStudioStore.getState().projectEvidenceBundleError).toBe(diagnostic);
});

it("finishes a verified requested download without mutating the replacement bundle", async () => {
  useStudioStore.setState({ projectEvidenceBundle: bundle() });
  let resolveFetch: (response: Response) => void = () => { throw new Error("Fetch did not start"); };
  vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockImplementation(() => new Promise((resolve) => { resolveFetch = resolve; })));
  const click = vi.fn();
  vi.stubGlobal("document", { createElement: () => ({ href: "", download: "", click }) });
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:historical");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
  const download = useStudioStore.getState().downloadEvidenceBundleArtifactForSurface("project", "evidence/manifest.json");
  const replacement = { ...bundle(), job_id: "sj_new", bundle_id: "seb_new" };
  useStudioStore.setState({ projectEvidenceBundle: replacement, projectEvidenceBundleError: "New diagnostic" });
  resolveFetch(new Response("manifest"));
  await download;
  expect(click).toHaveBeenCalledOnce();
  expect(useStudioStore.getState().projectEvidenceBundle).toBe(replacement);
  expect(useStudioStore.getState().projectEvidenceBundleError).toBe("New diagnostic");
});

it("keeps simultaneous panel diagnostics independent", async () => {
  useStudioStore.setState({ projectEvidenceBundle: bundle(), compileEvidenceBundle: bundle() });
  let rejectProject: (error: Error) => void = () => { throw new Error("Fetch did not start"); };
  vi.stubGlobal("fetch", vi.fn<typeof fetch>()
    .mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectProject = reject; }))
    .mockRejectedValueOnce(new Error("Compile download failed")));
  const project = useStudioStore.getState().downloadEvidenceBundleArtifactForSurface("project", "evidence/manifest.json");
  await useStudioStore.getState().downloadEvidenceBundleArtifactForSurface("compile", "evidence/manifest.json");
  rejectProject(new Error("Project download failed"));
  await project;
  expect(useStudioStore.getState().projectEvidenceBundleError).toBe("Project download failed");
  expect(useStudioStore.getState().compileEvidenceBundleError).toBe("Compile download failed");
  expect(useStudioStore.getState().evidenceBundleError).toBeNull();
});
