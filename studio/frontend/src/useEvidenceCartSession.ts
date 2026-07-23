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

import { useCallback, useState } from "react";

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
 * Session-scoped evidence cart state and success-only queue/export actions.
 */
export function useEvidenceCartSession(): EvidenceCartSession {
  const [cart, setCart] = useState<EvidenceCart>(() => emptyEvidenceCart());
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
    const beforeId = simulationResultIdentity(useStudioStore.getState().result);
    await useStudioStore.getState().runSimulation();
    const afterState = useStudioStore.getState();
    setCart((current) => {
      const decision = decideSimulationEnqueue(current, {
        result: afterState.result,
        resultIdentityBefore: beforeId,
        runSucceeded: afterState.result !== null
          && simulationResultIdentity(afterState.result) !== beforeId,
        selectedModelName: afterState.selectedModelName,
        sourceMode: afterState.sourceMode,
      });
      applyDecision(decision);
      return decision.action === "enqueue" ? decision.cart : current;
    });
  }, [applyDecision]);

  const runAnalysisIntoCart = useCallback(async () => {
    // W12-G: snapshot identity before the async analysis job; re-read the store
    // only after runFICurve resolves (store path is runStudioAnalysisJob).
    const beforeId = analysisResultIdentity(useStudioStore.getState().fiResult);
    setError(null);
    await useStudioStore.getState().runFICurve();
    const afterState = useStudioStore.getState();
    const after = afterState.fiResult;
    const afterId = analysisResultIdentity(after);
    const runSucceeded = afterId !== null && afterId !== beforeId;
    setCart((current) => {
      const decision = decideAnalysisEnqueue(current, {
        analysisKind: "fi_curve",
        analysisResult: after,
        resultIdentityAfter: afterId,
        resultIdentityBefore: beforeId,
        runSucceeded,
        selectedModelName: afterState.selectedModelName,
        sourceMode: afterState.sourceMode,
      });
      applyDecision(decision);
      return decision.action === "enqueue" ? decision.cart : current;
    });
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
