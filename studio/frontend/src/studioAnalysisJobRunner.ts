// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Imperative async analysis job runner (no UI)

/**
 * Builds a fail-closed analysis-job request, runs W07/W08 session submit+poll
 * to a terminal phase, sinks a completed result into store patches, and always
 * disposes the session. No React tree, App, or Zustand store imports.
 */

import type { AnalysisJobKind, AnalysisJobRequestBody } from "./api/client";
import {
  buildAnalysisJobRequest,
  type AnalysisJobSelection,
} from "./analysisJobRequest";
import {
  initialAnalysisJobState,
  isAnalysisJobBusy,
  type AnalysisJobApi,
  type AnalysisJobPhase,
  type AnalysisJobSession,
  type AnalysisJobSessionOptions,
  type AnalysisJobViewState,
} from "./analysisJob";
import {
  studioAnalysisFailureState,
  studioAnalysisStartState,
  type StudioAnalysisFailureStatePatch,
  type StudioAnalysisStartStatePatch,
} from "./studioAnalysisState";
import {
  studioAnalysisResultSink,
  studioAnalysisResultViewTab,
  type StudioAnalysisResultSinkPatch,
} from "./studioAnalysisResultSink";
import type { StudioSimulationConfigInput } from "./studioSimulationConfig";
import {
  attachAnalysisJobReactBinding,
  type AnalysisJobReactBinding,
} from "./useAnalysisJob";

const TERMINAL_PHASES: ReadonlySet<AnalysisJobPhase> = new Set([
  "completed",
  "failed",
  "cancelled",
  "timed_out",
  "malformed",
]);

export function isAnalysisJobTerminalPhase(phase: AnalysisJobPhase): boolean {
  return TERMINAL_PHASES.has(phase);
}

export interface StudioAnalysisJobRunnerInput {
  simulation: StudioSimulationConfigInput;
  selection: AnalysisJobSelection;
}

export interface StudioAnalysisJobRunnerOptions {
  api?: AnalysisJobApi;
  /** Test seam; production uses createAnalysisJobSession via the binding. */
  createSession?: (
    options: AnalysisJobSessionOptions,
  ) => AnalysisJobSession;
  pollIntervalMs?: number;
  onState?: (state: AnalysisJobViewState) => void;
  /** Optional store write-back for start/failure/success patches (W12-D). */
  applyPatch?: (
    patch:
      | StudioAnalysisStartStatePatch
      | StudioAnalysisFailureStatePatch
      | StudioAnalysisResultSinkPatch,
  ) => void;
}

export type StudioAnalysisJobRunnerResult =
  | {
      ok: true;
      kind: AnalysisJobKind;
      patch: StudioAnalysisResultSinkPatch;
      state: AnalysisJobViewState;
      startPatch: StudioAnalysisStartStatePatch;
    }
  | {
      ok: false;
      error: string;
      stage: "request" | "session";
      kind: AnalysisJobKind | null;
      state: AnalysisJobViewState | null;
      startPatch: StudioAnalysisStartStatePatch | null;
      failurePatch: StudioAnalysisFailureStatePatch | null;
    };

function selectionKind(selection: AnalysisJobSelection): AnalysisJobKind {
  return selection.analysis;
}

/**
 * Run one analysis job to terminal and map a completed result through the sink.
 *
 * Always disposes the underlying session/binding. Does not invent cancel,
 * progress %, or ETA. Invalid request building never starts a session.
 */
export async function runStudioAnalysisJob(
  input: StudioAnalysisJobRunnerInput,
  options: StudioAnalysisJobRunnerOptions = {},
): Promise<StudioAnalysisJobRunnerResult> {
  const kind = selectionKind(input.selection);
  const built = buildAnalysisJobRequest(input.simulation, input.selection);
  if (!built.ok) {
    return {
      ok: false,
      error: built.error,
      stage: "request",
      kind,
      state: null,
      startPatch: null,
      failurePatch: null,
    };
  }

  const request: AnalysisJobRequestBody = built.value;
  const viewTab = studioAnalysisResultViewTab(request.analysis);
  const startPatch = studioAnalysisStartState(viewTab);
  options.applyPatch?.(startPatch);

  let live = true;
  let resolveTerminal: ((state: AnalysisJobViewState) => void) | null = null;
  const terminalPromise = new Promise<AnalysisJobViewState>((resolve) => {
    resolveTerminal = resolve;
  });

  const onState = (state: AnalysisJobViewState) => {
    if (!live) {
      return;
    }
    options.onState?.(state);
    if (isAnalysisJobTerminalPhase(state.phase) && resolveTerminal !== null) {
      resolveTerminal(state);
      resolveTerminal = null;
    }
  };

  const binding: AnalysisJobReactBinding = attachAnalysisJobReactBinding({
    api: options.api,
    createSession: options.createSession,
    pollIntervalMs: options.pollIntervalMs,
    onState,
  });

  try {
    const startPromise = Promise.resolve(binding.startJob(request)).then(() => {
      // startJob is fire-and-forget on the binding; session work continues via polls.
      return undefined;
    });
    await startPromise;

    let state = binding.getState();
    if (!isAnalysisJobTerminalPhase(state.phase)) {
      state = await terminalPromise;
    }

    if (state.phase === "completed" && state.result !== null) {
      const sunk = studioAnalysisResultSink(request.analysis, state.result);
      if (!sunk.ok) {
        const failurePatch = studioAnalysisFailureState(sunk.error);
        options.applyPatch?.(failurePatch);
        return {
          ok: false,
          error: sunk.error,
          stage: "session",
          kind: request.analysis,
          state,
          startPatch,
          failurePatch,
        };
      }
      options.applyPatch?.(sunk.patch);
      return {
        ok: true,
        kind: request.analysis,
        patch: sunk.patch,
        state,
        startPatch,
      };
    }

    const error =
      state.error
      ?? `analysis_runner_terminal_${state.phase}`;
    const failurePatch = studioAnalysisFailureState(error);
    options.applyPatch?.(failurePatch);
    return {
      ok: false,
      error,
      stage: "session",
      kind: request.analysis,
      state,
      startPatch,
      failurePatch,
    };
  } finally {
    live = false;
    binding.dispose();
  }
}

/**
 * True when the runner should refuse a new start because a view state is busy.
 * Thin re-export of session policy for store delegates (W12-D).
 */
export function canStartStudioAnalysisJob(
  state: AnalysisJobViewState = initialAnalysisJobState(),
): boolean {
  return !isAnalysisJobBusy(state.phase) && state.phase !== "submitting";
}
