// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided default-flow panel
import { useState } from "react";

import type { GuidedRunController } from "../guidedRunController";
import type { GuidedFlowState, GuidedFlowStepStatus } from "../guidedFlowState";

export interface GuidedFlowPanelProps {
  controller?: GuidedRunController;
  state: GuidedFlowState;
}

const STATUS_LABEL: Record<GuidedFlowStepStatus, string> = {
  completed: "done",
  current: "next",
  available: "ready",
  blocked: "blocked",
};

const STATUS_COLOR: Record<GuidedFlowStepStatus, string> = {
  completed: "#7bc67b",
  current: "#7bb4ff",
  available: "var(--text-secondary)",
  blocked: "#c98a8a",
};

export default function GuidedFlowPanel({ controller, state }: GuidedFlowPanelProps) {
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionRunning, setActionRunning] = useState(false);
  const actionDisabled = controller === undefined
    || actionRunning
    || controller.nextActionKey === "blocked"
    || controller.nextActionKey === "complete";
  const completedEvidence = controller?.completedEvidence ?? [];
  const runNextStep = async () => {
    if (controller === undefined || actionDisabled) return;
    setActionRunning(true);
    setActionError(null);
    const result = await controller.runNextStep();
    setActionRunning(false);
    if (!result.ok) {
      setActionError(result.error ?? "Guided run action failed");
    }
  };

  return (
    <div
      className="guided-flow-panel"
      aria-label="Guided flow"
      style={{
        padding: 8,
        background: "var(--bg-secondary)",
        borderRadius: 4,
        fontSize: 11,
        color: "var(--text-secondary)",
      }}
    >
      <div
        className="guided-flow-header"
        style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}
      >
        <span>Guided flow</span>
        <span aria-label="Guided flow progress">
          {state.completedCount}/{state.totalCount}
        </span>
      </div>
      {controller !== undefined && (
        <div className="guided-run-controller" style={{ marginBottom: 8 }}>
          <button
            aria-label="Run next guided step"
            disabled={actionDisabled}
            onClick={() => { void runNextStep(); }}
            style={{
              background: actionDisabled ? "var(--bg-tertiary)" : "var(--accent)",
              border: "1px solid var(--border)",
              color: actionDisabled ? "var(--text-muted)" : "var(--bg-primary)",
              cursor: actionDisabled ? "not-allowed" : "pointer",
              fontSize: 10,
              padding: "2px 6px",
              width: "100%",
            }}
            type="button"
          >
            {actionRunning ? "Running" : `Run next step: ${controller.nextActionLabel}`}
          </button>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
            <span>{controller.exportReady ? "Evidence ready" : "Evidence pending"}</span>
            <span>{completedEvidence.length} completed</span>
          </div>
          {completedEvidence.length > 0 && (
            <div title={completedEvidence.join(", ")}>
              Completed evidence: {completedEvidence.join(", ")}
            </div>
          )}
          {controller.blockerReason !== null && (
            <div style={{ color: "#c98a8a" }}>{controller.blockerReason}</div>
          )}
          {actionError !== null && <div style={{ color: "#c98a8a" }}>{actionError}</div>}
        </div>
      )}
      <ol style={{ listStyle: "none", margin: 0, padding: 0 }}>
        {state.steps.map((step) => (
          <li
            key={step.key}
            data-step={step.key}
            data-status={step.status}
            aria-current={step.status === "current" ? "step" : undefined}
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: 8,
              padding: "2px 0",
              fontWeight: step.status === "current" ? 600 : 400,
            }}
          >
            <span>
              {step.title}
              {step.optional ? " (optional)" : ""}
            </span>
            <span style={{ color: STATUS_COLOR[step.status] }} title={step.blockedReason ?? ""}>
              {step.status === "blocked" && step.blockedReason
                ? step.blockedReason
                : STATUS_LABEL[step.status]}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
