// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — React adapter for W07 analysis-job session

/**
 * Thin React wiring over {@link createAnalysisJobSession}. Does not reimplement
 * validation, reducers, polling, or timers.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchStudioJobAtStatusRoute,
  submitAnalysisJob,
  type AnalysisJobRequestBody,
} from "./api/client";
import {
  canSubmitAnalysisJob,
  createAnalysisJobSession,
  initialAnalysisJobState,
  isAnalysisJobBusy,
  type AnalysisJobApi,
  type AnalysisJobSession,
  type AnalysisJobSessionOptions,
  type AnalysisJobViewState,
} from "./analysisJob";

export interface UseAnalysisJobOptions {
  api?: AnalysisJobApi;
  /** Test seam only; production uses createAnalysisJobSession. */
  createSession?: (
    options: AnalysisJobSessionOptions,
  ) => AnalysisJobSession;
  pollIntervalMs?: number;
}

export interface UseAnalysisJobResult {
  busy: boolean;
  canSubmit: boolean;
  startJob: (request: AnalysisJobRequestBody) => void;
  state: AnalysisJobViewState;
}

export interface AnalysisJobReactBinding {
  dispose: () => void;
  getState: () => AnalysisJobViewState;
  startJob: (request: AnalysisJobRequestBody) => void;
}

const defaultApi: AnalysisJobApi = {
  fetchJob: fetchStudioJobAtStatusRoute,
  submit: submitAnalysisJob,
};

/**
 * Pure React-free binding used by {@link useAnalysisJob} and unit tests.
 *
 * Applies session onChange only while the binding is live (mounted).
 */
export function attachAnalysisJobReactBinding(options: {
  api?: AnalysisJobApi;
  createSession?: (
    options: AnalysisJobSessionOptions,
  ) => AnalysisJobSession;
  onState?: (state: AnalysisJobViewState) => void;
  pollIntervalMs?: number;
}): AnalysisJobReactBinding {
  let live = true;
  const api = options.api ?? defaultApi;
  const createSession = options.createSession ?? createAnalysisJobSession;
  const session = createSession({
    api,
    onChange: (next) => {
      if (live) {
        options.onState?.(next);
      }
    },
    pollIntervalMs: options.pollIntervalMs ?? 500,
  });
  options.onState?.(session.getState());
  return {
    dispose: () => {
      live = false;
      session.dispose();
    },
    getState: () => session.getState(),
    startJob: (request) => {
      void session.startJob(request);
    },
  };
}

/**
 * Session-scoped analysis job state for React surfaces.
 */
export function useAnalysisJob(
  options: UseAnalysisJobOptions = {},
): UseAnalysisJobResult {
  const [state, setState] = useState<AnalysisJobViewState>(() =>
    initialAnalysisJobState(),
  );
  const bindingRef = useRef<AnalysisJobReactBinding | null>(null);
  const api = options.api ?? defaultApi;
  const pollIntervalMs = options.pollIntervalMs ?? 500;
  const createSession = options.createSession ?? createAnalysisJobSession;

  useEffect(() => {
    const binding = attachAnalysisJobReactBinding({
      api,
      createSession,
      onState: setState,
      pollIntervalMs,
    });
    bindingRef.current = binding;
    return () => {
      binding.dispose();
      bindingRef.current = null;
    };
  }, [api, createSession, pollIntervalMs]);

  const startJob = useCallback((request: AnalysisJobRequestBody) => {
    bindingRef.current?.startJob(request);
  }, []);

  return useMemo(
    () => ({
      busy: isAnalysisJobBusy(state.phase),
      canSubmit: canSubmitAnalysisJob(state),
      startJob,
      state,
    }),
    [startJob, state],
  );
}
