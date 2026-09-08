// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio evidence cart orchestration (outside App.tsx)

/**
 * Deciding what belongs in the evidence cart, and what would be a duplicate.
 *
 * The rule this layer exists for is **identity by digest, not by shape**. Two
 * runs of the same model with the same parameters produce results of identical
 * shape and different values; comparing anything but the server's own
 * `result_sha256` would treat the second as unchanged and drop it. So a run
 * whose digest matches what was there before it started is skipped, and a run
 * whose digest cannot be read is skipped too -- failing closed, because
 * queueing an artefact that cannot be identified puts an unverifiable entry
 * into a ledger whose whole purpose is verifiability.
 *
 * Every decision is returned with a stable reason, so the panel can say why a
 * successful run did not reach the cart. Nothing here mutates the cart or
 * touches the store; `App.tsx` stays composition only.
 */

import type { SimulateResponse } from "./api/client";
import {
  analysisCartDraft,
  buildEvidenceCartExport,
  enqueueEvidenceCartArtefact,
  evidenceCartExportFilename,
  evidenceCartExportToBlob,
  simulationCartDraft,
  verifyEvidenceCartExportRoundTrip,
  type EvidenceCart,
  type EvidenceCartExportBundle,
  type EvidenceCartItemKind,
} from "./evidenceCart";
import { canonicalSealText } from "./evidenceSeal";
export { analysisResultIdentity } from "./evidenceCartIdentity";

/** What the store reports after a simulation run. */
export interface SimulationQueueInput {
  /** True when the store reports the last run completed successfully. */
  runSucceeded: boolean;
  sourceMode: "model" | "ode";
  selectedModelName: string;
  result: SimulateResponse | null;
  /** Snapshot of result identity before the run started (detect stale leave-behind). */
  resultIdentityBefore: string | null;
}

/** What the store reports after an analysis run. */
export interface AnalysisQueueInput {
  runSucceeded: boolean;
  sourceMode: "model" | "ode";
  selectedModelName: string;
  /** Only the analysis kind that just succeeded. */
  analysisKind: "fi_curve" | "bifurcation" | "sensitivity" | "heatmap" | "other";
  /** Exact successful analysis payload (not a bag of all store fields). */
  analysisResult: unknown;
  resultIdentityBefore: string | null;
  resultIdentityAfter: string | null;
}

/** The cart with the artefact queued, or the reason it was not. */
export type QueueDecision =
  | { action: "enqueue"; cart: EvidenceCart; kind: EvidenceCartItemKind }
  | { action: "skip"; reason: string; cart: EvidenceCart };

/**
 * Identify a simulation result by what it contains.
 *
 * The server's own `result_sha256` is the identity. Two traces of the same
 * shape with different values must not collide, and only a digest over the
 * values distinguishes them.
 *
 * @param result - The run's result, or `null` when there is none.
 * @returns The digest, or `null` when the result carries none. A result
 *   without a digest is unidentifiable, and the callers treat that as a reason
 *   to skip rather than a reason to queue.
 */
export function simulationResultIdentity(result: SimulateResponse | null): string | null {
  if (result === null) {
    return null;
  }
  const digest = result.run_metadata.result_sha256;
  if (typeof digest !== "string" || digest.length === 0) {
    return null;
  }
  return digest;
}

/**
 * Decide whether a finished simulation belongs in the cart.
 *
 * @param cart - The cart as it stands.
 * @param input - What the store reports about the run, including the result
 *   identity from before it started.
 * @returns The new cart, or the reason nothing was queued.
 */
export function decideSimulationEnqueue(
  cart: EvidenceCart,
  input: SimulationQueueInput,
): QueueDecision {
  if (!input.runSucceeded || input.result === null) {
    return { action: "skip", reason: "simulation_run_failed", cart };
  }
  const afterId = simulationResultIdentity(input.result);
  if (afterId === null || afterId === input.resultIdentityBefore) {
    return { action: "skip", reason: "simulation_result_unchanged", cart };
  }
  const sourceName =
    input.sourceMode === "ode"
      ? "ode"
      : input.selectedModelName.trim() || "unknown-model";
  const queued = enqueueEvidenceCartArtefact(
    cart,
    simulationCartDraft(sourceName, {
      ...input.result,
      model_name: input.result.model_name ?? sourceName,
      source_mode: input.sourceMode,
    }),
  );
  if (!queued.ok) {
    return { action: "skip", reason: queued.error, cart };
  }
  return { action: "enqueue", cart: queued.cart, kind: "simulation" };
}

/**
 * Decide whether a finished analysis belongs in the cart.
 *
 * The payload queued is the analysis that just succeeded, not a snapshot of
 * every analysis field the store happens to hold: a cart entry should be one
 * result, identifiable by its own digest.
 *
 * @param cart - The cart as it stands.
 * @param input - What the store reports about the run, with the result
 *   identity from before and after it.
 * @returns The new cart, or the reason nothing was queued. An analysis whose
 *   identity cannot be read is skipped rather than queued unidentified.
 */
export function decideAnalysisEnqueue(
  cart: EvidenceCart,
  input: AnalysisQueueInput,
): QueueDecision {
  if (!input.runSucceeded || input.analysisResult === null || input.analysisResult === undefined) {
    return { action: "skip", reason: "analysis_run_failed", cart };
  }
  if (input.resultIdentityAfter === null) {
    return { action: "skip", reason: "analysis_result_identity_invalid", cart };
  }
  if (input.resultIdentityAfter === input.resultIdentityBefore) {
    return { action: "skip", reason: "analysis_result_unchanged", cart };
  }
  const sourceName =
    input.sourceMode === "ode"
      ? "ode"
      : input.selectedModelName.trim() || "unknown-model";
  const queued = enqueueEvidenceCartArtefact(
    cart,
    analysisCartDraft(sourceName, {
      analysis_kind: input.analysisKind,
      result: input.analysisResult,
      source_mode: input.sourceMode,
      source_name: sourceName,
    }),
  );
  if (!queued.ok) {
    return { action: "skip", reason: queued.error, cart };
  }
  return { action: "enqueue", cart: queued.cart, kind: "analysis" };
}

/**
 * Whether the guided flow's export step is satisfied by what has been exported.
 *
 * Counts, ordered identities, metadata and payloads must match the exported
 * snapshot. Same-count replacements are not already exported. This is a local
 * freshness check; cryptographic verification remains in the export operation.
 *
 * @param cart - The cart as it stands.
 * @param lastExport - The last bundle built, if any.
 * @param exportItemCount - How many items the cart held when it was built.
 * @returns Whether the step is satisfied.
 */
export function evidenceCartExportSatisfiesGuided(
  cart: EvidenceCart,
  lastExport: EvidenceCartExportBundle | null,
  exportItemCount: number | null,
): boolean {
  if (lastExport === null || exportItemCount === null) {
    return false;
  }
  if (cart.items.length === 0 || exportItemCount !== cart.items.length
    || lastExport.entry_count !== cart.items.length || lastExport.entries.length !== cart.items.length) return false;
  try {
    return cart.items.every((item, index) => {
      const entry = lastExport.entries[index];
      return entry?.id === item.id && item.kind === entry.kind
        && item.classification === entry.classification && item.label === entry.label
        && item.queuedAtUtc === entry.queued_at_utc && (item.sourceName ?? null) === entry.source_name
        && canonicalSealText(item.payload) === canonicalSealText(entry.payload);
    });
  } catch { return false; }
}

/**
 * Build the export and check it before handing it over.
 *
 * The verification is not ceremony: the bundle is what someone else will check
 * later, and a bundle that fails its own round-trip should never reach a
 * downloads folder where it looks like evidence.
 *
 * @param cart - The cart to export.
 * @param options - An export timestamp, for reproducible tests.
 * @returns The bundle with its bytes and filename, or the reason there is
 *   none.
 */
export async function exportEvidenceCartWithVerification(
  cart: EvidenceCart,
  options: { exportedAtUtc?: string } = {},
): Promise<
  | { ok: true; bundle: EvidenceCartExportBundle; blob: Blob; filename: string }
  | { ok: false; error: string }
> {
  const bundle = await buildEvidenceCartExport(cart, options);
  if ("error" in bundle) {
    return { ok: false, error: bundle.error };
  }
  const verified = await verifyEvidenceCartExportRoundTrip(bundle);
  if (!verified.ok) {
    return { ok: false, error: verified.error };
  }
  return {
    ok: true,
    bundle,
    blob: evidenceCartExportToBlob(bundle),
    filename: evidenceCartExportFilename(bundle),
  };
}
