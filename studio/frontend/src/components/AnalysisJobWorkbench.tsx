// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Async analysis job workbench composition

/**
 * Thin composition of W07–W10 surfaces: builds a W09 request from typed
 * inputs, obtains runtime state/actions from W08, and renders W10 control.
 * No second validator, reducer, poller, store, API call, or phase mapping.
 */

import { useMemo } from "react";

import {
  buildAnalysisJobRequest,
  type AnalysisJobSelection,
} from "../analysisJobRequest";
import type { StudioSimulationConfigInput } from "../studioSimulationConfig";
import {
  useAnalysisJob,
  type UseAnalysisJobOptions,
} from "../useAnalysisJob";
import AnalysisJobControl from "./AnalysisJobControl";

export interface AnalysisJobWorkbenchProps {
  simulationInput: StudioSimulationConfigInput;
  selection: AnalysisJobSelection;
  selectedAnalysisLabel: string;
  hookOptions?: UseAnalysisJobOptions;
}

/**
 * Compose request policy + session hook + presentational control.
 */
export default function AnalysisJobWorkbench({
  simulationInput,
  selection,
  selectedAnalysisLabel,
  hookOptions,
}: AnalysisJobWorkbenchProps) {
  const request = useMemo(
    () => buildAnalysisJobRequest(simulationInput, selection),
    [simulationInput, selection],
  );
  const { busy, canSubmit, startJob, state } = useAnalysisJob(
    hookOptions ?? {},
  );

  return (
    <div
      className="analysis-job-workbench"
      data-testid="analysis-job-workbench"
    >
      <AnalysisJobControl
        busy={busy}
        canSubmit={canSubmit}
        request={request}
        selectedAnalysisLabel={selectedAnalysisLabel}
        startJob={startJob}
        state={state}
      />
    </div>
  );
}
