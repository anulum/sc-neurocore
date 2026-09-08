// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Evidence download diagnostic ownership

import { fetchStudioJobArtifact } from "../api/client";
import { evidenceBundleArtifactDownloadPlan, evidenceBundleDownloadSelection,
  type EvidenceBundleSurface } from "../evidenceBundles";
import type { StudioState } from "./studioTypes";

/**
 * Create a store-local downloader with independent per-surface diagnostics.
 *
 * Each invocation owns its captured artifact download, even if a replacement
 * bundle arrives. Only the latest invocation for the unchanged bundle may
 * publish an error. Byte verification remains mandatory before browser save.
 *
 * @param get - Read the live bundle slots after asynchronous work.
 * @param set - Apply only this download's owned diagnostic patch.
 * @returns The production store action; request tokens never cross stores.
 */
export function createStoreArtifactDownloader(
  get: () => StudioState, set: (patch: Partial<StudioState>) => void,
): StudioState["downloadEvidenceBundleArtifactForSurface"] {
  const latest = new Map<EvidenceBundleSurface, symbol>();
  return async (surface, relativePath) => {
    const token = Symbol();
    latest.set(surface, token);
    const bundle = evidenceBundleDownloadSelection(surface, get()).bundle;
    const plan = evidenceBundleArtifactDownloadPlan(surface, relativePath, get());
    if (!plan.available) { set(plan.statePatch); return; }
    set(plan.startState);
    try {
      const payload = await fetchStudioJobArtifact(plan.jobId, plan.relativePath);
      await plan.writePayload(payload);
    } catch (error: unknown) {
      if (latest.get(surface) === token
        && evidenceBundleDownloadSelection(surface, get()).bundle === bundle) {
        set(plan.failureState(error));
      }
    }
  };
}
