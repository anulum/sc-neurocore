// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio evidence cart orchestration (outside App.tsx)

/**
 * Pure controller helpers for session evidence-cart enqueue/export decisions.
 * Keeps App.tsx composition-only: no queue policy lives in the shell file.
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

export interface SimulationQueueInput {
  /** True when the store reports the last run completed successfully. */
  runSucceeded: boolean;
  sourceMode: "model" | "ode";
  selectedModelName: string;
  result: SimulateResponse | null;
  /** Snapshot of result identity before the run started (detect stale leave-behind). */
  resultIdentityBefore: string | null;
}

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

export type QueueDecision =
  | { action: "enqueue"; cart: EvidenceCart; kind: EvidenceCartItemKind }
  | { action: "skip"; reason: string; cart: EvidenceCart };

/**
 * Build a content-faithful identity for a simulation result snapshot.
 *
 * Uses the authoritative ``run_metadata.result_sha256`` so two same-shape
 * traces with different values cannot collide and be treated as unchanged.
 */
export function simulationResultIdentity(result: SimulateResponse | null): string | null {
  if (result === null) {
    return null;
  }
  const digest = result.run_metadata?.result_sha256;
  if (typeof digest !== "string" || digest.length === 0) {
    return null;
  }
  return digest;
}

/**
 * Decide whether to enqueue a simulation artefact after a store run.
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
      current_trace: input.result.current_trace,
      dt: input.result.dt,
      model_name: input.result.model_name ?? sourceName,
      n_steps: input.result.n_steps,
      source_mode: input.sourceMode,
      spike_count: input.result.spike_count,
      spikes: input.result.spikes,
      states: input.result.states,
      stats: input.result.stats,
      time: input.result.time,
    }),
  );
  if (!queued.ok) {
    return { action: "skip", reason: queued.error, cart };
  }
  return { action: "enqueue", cart: queued.cart, kind: "simulation" };
}

/**
 * Decide whether to enqueue the exact analysis artefact that just succeeded.
 */
export function decideAnalysisEnqueue(
  cart: EvidenceCart,
  input: AnalysisQueueInput,
): QueueDecision {
  if (!input.runSucceeded || input.analysisResult === null || input.analysisResult === undefined) {
    return { action: "skip", reason: "analysis_run_failed", cart };
  }
  if (
    input.resultIdentityAfter !== null
    && input.resultIdentityBefore !== null
    && input.resultIdentityAfter === input.resultIdentityBefore
    && input.analysisKind === "other"
  ) {
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
 * True when guided-flow may treat evidence as exported for the current cart.
 *
 * A previous export must not satisfy the export step after new items are queued.
 */
export function evidenceCartExportSatisfiesGuided(
  cart: EvidenceCart,
  lastExport: EvidenceCartExportBundle | null,
  exportItemCount: number | null,
): boolean {
  if (lastExport === null || exportItemCount === null) {
    return false;
  }
  return exportItemCount === cart.items.length && lastExport.entry_count === cart.items.length;
}

/**
 * Build export blob with payload-bearing entries and verified digests.
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
