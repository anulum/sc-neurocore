// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — React wiring for job-backed model catalogue scan

/**
 * Thin React adapter over {@link createModelScanJobSession}. Polling policy
 * stays in modelScanJob; this hook only holds view state and lifecycle.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  fetchStudioJobAtStatusRoute,
  submitModelScanJob,
} from "./api/client";
import {
  canSubmitModelScanJob,
  createModelScanJobSession,
  initialModelScanJobState,
  isModelScanJobBusy,
  modelScanJobPhaseLabel,
  type ModelScanJobApi,
  type ModelScanJobSession,
  type ModelScanJobViewState,
} from "./modelScanJob";

/** What the hook may be given instead of its defaults. */
export interface UseModelScanJobOptions {
  api?: ModelScanJobApi;
  pollIntervalMs?: number;
}

/** What the control needs: where the scan is, and how to start another. */
export interface UseModelScanJobResult {
  busy: boolean;
  canSubmit: boolean;
  phaseLabel: string;
  startScan: () => void;
  state: ModelScanJobViewState;
}

/** The real Studio routes, used unless a caller passes its own. */
const defaultApi: ModelScanJobApi = {
  fetchJob: fetchStudioJobAtStatusRoute,
  submit: submitModelScanJob,
};

/**
 * Run a catalogue scan for as long as the control is on screen.
 *
 * @param options - The API and the poll interval.
 * @returns The scan's state and the calls the control needs: whether it is
 *   busy, whether another may be started, its phase as a label, and how to
 *   start one.
 */
export function useModelScanJob(
  options: UseModelScanJobOptions = {},
): UseModelScanJobResult {
  const [state, setState] = useState<ModelScanJobViewState>(() =>
    initialModelScanJobState(),
  );
  const sessionRef = useRef<ModelScanJobSession | null>(null);
  const api = options.api ?? defaultApi;
  const pollIntervalMs = options.pollIntervalMs ?? 500;

  useEffect(() => {
    const session = createModelScanJobSession({
      api,
      onChange: setState,
      pollIntervalMs,
    });
    sessionRef.current = session;
    return () => {
      session.dispose();
      sessionRef.current = null;
    };
  }, [api, pollIntervalMs]);

  const startScan = useCallback(() => {
    void sessionRef.current?.startScan();
  }, []);

  return {
    busy: isModelScanJobBusy(state.phase),
    canSubmit: canSubmitModelScanJob(state),
    phaseLabel: modelScanJobPhaseLabel(state.phase),
    startScan,
    state,
  };
}
