// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — React adapter for W07 analysis-job session

/**
 * Holding an analysis job for as long as a React surface is on screen.
 *
 * The binding underneath is deliberately React-free: it owns the session and a
 * `live` flag, and drops every state change once disposed. That is what stops
 * a job that outlives its panel from setting state on an unmounted component,
 * and it is why the same binding can be driven by a test with no React at all.
 *
 * Nothing here re-implements validation, reducing or polling. It is wiring.
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

/** What the hook may be given instead of its defaults. */
export interface UseAnalysisJobOptions {
  api?: AnalysisJobApi;
  /**
   * How a session is created. Replaced in tests to control timers; production
   * leaves it unset.
   */
  createSession?: (
    options: AnalysisJobSessionOptions,
  ) => AnalysisJobSession;
  pollIntervalMs?: number;
}

/** What a view needs: where the job is, and how to start another. */
export interface UseAnalysisJobResult {
  busy: boolean;
  canSubmit: boolean;
  startJob: (request: AnalysisJobRequestBody) => void;
  state: AnalysisJobViewState;
}

/**
 * The React-free half of the hook: a session, a way to read it, and a dispose
 * that stops it reporting.
 */
export interface AnalysisJobReactBinding {
  dispose: () => void;
  getState: () => AnalysisJobViewState;
  startJob: (request: AnalysisJobRequestBody) => void;
}

/** The real Studio routes, used unless a caller passes its own. */
const defaultApi: AnalysisJobApi = {
  fetchJob: fetchStudioJobAtStatusRoute,
  submit: submitAnalysisJob,
};

/**
 * Attach a session and report its state until disposed.
 *
 * The current state is reported once immediately, so a caller does not have to
 * wait for the first change to know where it is starting from.
 *
 * @param options - The API, the session factory, the poll interval, and where
 *   to report state.
 * @returns The binding. Its `startJob` is fire-and-forget: the job continues
 *   through polls and its progress arrives through `onState`.
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
 * Track one analysis job for the lifetime of a component.
 *
 * @param options - The API, the session factory and the poll interval.
 * @returns The job's state and the calls a view needs: whether it is busy,
 *   whether another may be started, and how to start one.
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
