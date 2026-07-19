// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Presentational async analysis job control

/**
 * Pure view over W08/W09 analysis-job surfaces. No session, hook, poller,
 * request builder, store, or API calls.
 */

import type { AnalysisJobRequestBody } from "../api/client";
import {
  analysisJobPhaseLabel,
  type AnalysisJobViewState,
} from "../analysisJob";
import type { AnalysisJobRequestBuildResult } from "../analysisJobRequest";

export interface AnalysisJobControlProps {
  busy: boolean;
  canSubmit: boolean;
  request: AnalysisJobRequestBuildResult;
  selectedAnalysisLabel: string;
  startJob: (request: AnalysisJobRequestBody) => void;
  state: AnalysisJobViewState;
}

export type AnalysisJobControlStartDecision =
  | "started"
  | "blocked_invalid"
  | "blocked_busy";

/**
 * Guarded start helper for the submit control (no side effects beyond startJob).
 */
export function decideAnalysisJobControlStart(input: {
  busy: boolean;
  canSubmit: boolean;
  request: AnalysisJobRequestBuildResult;
  startJob: (request: AnalysisJobRequestBody) => void;
}): AnalysisJobControlStartDecision {
  if (!input.request.ok) {
    return "blocked_invalid";
  }
  if (!input.canSubmit || input.busy) {
    return "blocked_busy";
  }
  input.startJob(input.request.value);
  return "started";
}

/**
 * Whether the submit button may be enabled.
 */
export function isAnalysisJobControlSubmitEnabled(input: {
  busy: boolean;
  canSubmit: boolean;
  request: AnalysisJobRequestBuildResult;
}): boolean {
  return input.request.ok && input.canSubmit && !input.busy;
}

function completedMetadataSummary(
  state: AnalysisJobViewState,
): { analysisType: string; classification: string; schema: string; status: string } | null {
  if (state.phase !== "completed" || state.result === null) {
    return null;
  }
  const meta = state.result.analysis_metadata;
  return {
    analysisType: meta.analysis_type,
    classification: meta.evidence_classification,
    schema: meta.schema_version,
    status: meta.status,
  };
}

/**
 * Focused presentational control for async analysis job submission/status.
 */
export default function AnalysisJobControl({
  busy,
  canSubmit,
  request,
  selectedAnalysisLabel,
  startJob,
  state,
}: AnalysisJobControlProps) {
  const phaseLabel = analysisJobPhaseLabel(state.phase);
  const submitEnabled = isAnalysisJobControlSubmitEnabled({
    busy,
    canSubmit,
    request,
  });
  const requestError = request.ok ? null : request.error;
  const publicError = state.error;
  const completed = completedMetadataSummary(state);

  return (
    <div
      className="analysis-job-control"
      data-testid="analysis-job-control"
    >
      <div className="panel-header">Async analysis</div>
      <div data-testid="analysis-job-control-selection">
        Selected: {selectedAnalysisLabel}
      </div>
      <div
        aria-live="polite"
        data-testid="analysis-job-control-phase"
        data-phase={state.phase}
      >
        Phase: {phaseLabel}
      </div>
      {requestError !== null && (
        <div
          role="alert"
          data-testid="analysis-job-control-request-error"
        >
          {requestError}
        </div>
      )}
      {publicError !== null && (
        <div
          role="alert"
          aria-live="assertive"
          data-testid="analysis-job-control-error"
        >
          {publicError}
        </div>
      )}
      {completed !== null && (
        <div data-testid="analysis-job-control-completed-summary">
          <span data-testid="analysis-job-meta-type">{completed.analysisType}</span>
          {" · "}
          <span data-testid="analysis-job-meta-class">{completed.classification}</span>
          {" · "}
          <span data-testid="analysis-job-meta-schema">{completed.schema}</span>
          {" · "}
          <span data-testid="analysis-job-meta-status">{completed.status}</span>
        </div>
      )}
      <button
        type="button"
        className="btn-simulate"
        data-testid="analysis-job-control-submit"
        disabled={!submitEnabled}
        aria-busy={busy}
        onClick={() => {
          decideAnalysisJobControlStart({
            busy,
            canSubmit,
            request,
            startJob,
          });
        }}
      >
        {busy ? phaseLabel : "Run async analysis"}
      </button>
    </div>
  );
}
