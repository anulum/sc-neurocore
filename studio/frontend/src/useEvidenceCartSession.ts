// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Session evidence-cart React wiring (outside App.tsx)

/**
 * Owns evidence-cart React state and the exact success-only enqueue/export
 * handlers used by guided flow and the operator workbench.
 */

import { useCallback, useRef, useState } from "react";

import { downloadBrowserArtefact } from "./browserArtefactDownload";
import {
  emptyEvidenceCart,
  type EvidenceCart,
  type EvidenceCartExportBundle,
} from "./evidenceCart";
import {
  analysisResultIdentity,
  decideAnalysisEnqueue,
  decideSimulationEnqueue,
  evidenceCartExportSatisfiesGuided,
  exportEvidenceCartWithVerification,
  simulationResultIdentity,
  type QueueDecision,
} from "./evidenceCartController";
import { useStudioStore } from "./stores/studio";

/** What a panel needs: the cart, the last export, and the actions. */
export interface EvidenceCartSession {
  cart: EvidenceCart;
  error: string | null;
  exportBundle: EvidenceCartExportBundle | null;
  exportSatisfiesGuided: boolean;
  exportSessionCart: () => Promise<void>;
  runAnalysisIntoCart: () => Promise<void>;
  runSimulationIntoCart: () => Promise<void>;
}

/**
 * Hold the evidence cart for as long as the reader's session lasts.
 *
 * The actions here run a job **and then decide** whether its result belongs in
 * the cart, which is why they are one call rather than two: the identity of
 * the result before the run is what tells a genuinely new result from a repeat,
 * and only the caller that started the run holds it.
 *
 * @returns The cart, the last export, whether the guided flow's export step is
 *   satisfied, and the three actions a panel offers.
 */
export function useEvidenceCartSession(): EvidenceCartSession {
  const [cart, setCart] = useState<EvidenceCart>(() => emptyEvidenceCart());
  // Single synchronous decision cursor; React state is its rendered projection.
  // Never enqueue or call setters inside a replayable functional state updater.
  const cartRef = useRef(cart);
  const [exportBundle, setExportBundle] = useState<EvidenceCartExportBundle | null>(null);
  const [exportItemCount, setExportItemCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const applyDecision = useCallback((decision: QueueDecision) => {
    if (decision.action === "skip") {
      if (
        decision.reason !== "simulation_result_unchanged"
        && decision.reason !== "analysis_result_unchanged"
        && decision.reason !== "simulation_run_failed"
        && decision.reason !== "analysis_run_failed"
      ) {
        setError(decision.reason);
      }
      return;
    }
    cartRef.current = decision.cart;
    setCart(decision.cart);
    setError(null);
  }, []);

  const exportSessionCart = useCallback(async () => {
    const result = await exportEvidenceCartWithVerification(cart);
    if (!result.ok) {
      setError(result.error);
      throw new Error(result.error);
    }
    downloadBrowserArtefact(result.blob, result.filename);
    setExportBundle(result.bundle);
    setExportItemCount(cart.items.length);
    setError(null);
  }, [cart]);

  const runSimulationIntoCart = useCallback(async () => {
    const beforeState = useStudioStore.getState();
    if (beforeState.isSimulating) return;
    const { selectedModelName, sourceMode } = beforeState;
    const beforeId = simulationResultIdentity(beforeState.result);
    await beforeState.runSimulation();
    const afterState = useStudioStore.getState();
    applyDecision(decideSimulationEnqueue(cartRef.current, {
        result: afterState.result,
        resultIdentityBefore: beforeId,
        runSucceeded: afterState.result !== null
          && simulationResultIdentity(afterState.result) !== beforeId,
        selectedModelName,
        sourceMode,
    }));
  }, [applyDecision]);

  const runAnalysisIntoCart = useCallback(async () => {
    // W12-G: snapshot identity before the async analysis job; re-read the store
    // only after runFICurve resolves (store path is runStudioAnalysisJob).
    const beforeState = useStudioStore.getState();
    if (beforeState.isSimulating) return;
    const { selectedModelName, sourceMode } = beforeState;
    const beforeId = analysisResultIdentity(beforeState.fiResult);
    setError(null);
    await beforeState.runFICurve();
    const afterState = useStudioStore.getState();
    const after = afterState.fiResult;
    const afterId = analysisResultIdentity(after);
    const runSucceeded = afterId !== null && afterId !== beforeId;
    applyDecision(decideAnalysisEnqueue(cartRef.current, {
        analysisKind: "fi_curve",
        analysisResult: after,
        resultIdentityAfter: afterId,
        resultIdentityBefore: beforeId,
        runSucceeded,
        selectedModelName,
        sourceMode,
    }));
  }, [applyDecision]);

  return {
    cart,
    error,
    exportBundle,
    exportSatisfiesGuided: evidenceCartExportSatisfiesGuided(
      cart,
      exportBundle,
      exportItemCount,
    ),
    exportSessionCart,
    runAnalysisIntoCart,
    runSimulationIntoCart,
  };
}
