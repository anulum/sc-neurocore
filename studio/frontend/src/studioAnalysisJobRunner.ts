// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Imperative async analysis job runner (no UI)

/**
 * Running one analysis job from a form's inputs to a patch the store can apply.
 *
 * This is the whole path in one call: build the request, refuse it if it is
 * not sound, submit, poll until the job stops, check what came back, and turn
 * it into the patch that displays it. The session is disposed in a `finally`,
 * so a job abandoned halfway leaves no timer polling a server for a result
 * nobody will read.
 *
 * A request that fails to build **never starts a session**. The distinction is
 * carried in the result's `stage`: `request` means nothing was submitted and
 * nothing needs cleaning up, `session` means a job ran and this is its verdict.
 *
 * Nothing here imports React or the store. The store is written through the
 * optional `applyPatch`, which is what lets the same runner drive a panel, a
 * test, and a scripted run.
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

/** The phases in which a job has stopped, however it stopped. */
const TERMINAL_PHASES: ReadonlySet<AnalysisJobPhase> = new Set([
  "completed",
  "failed",
  "cancelled",
  "timed_out",
  "interrupted",
  "malformed",
]);

/**
 * Whether a job in this phase has stopped.
 *
 * @param phase - The phase.
 * @returns Whether it is terminal.
 */
export function isAnalysisJobTerminalPhase(phase: AnalysisJobPhase): boolean {
  return TERMINAL_PHASES.has(phase);
}

/** What to run: the simulation to run it on, and which analysis. */
export interface StudioAnalysisJobRunnerInput {
  simulation: StudioSimulationConfigInput;
  selection: AnalysisJobSelection;
}

/**
 * Everything the runner may be given instead of its defaults: the API, the
 * poll interval, the session factory, and where to write patches.
 */
export interface StudioAnalysisJobRunnerOptions {
  api?: AnalysisJobApi;
  /**
   * How a session is created. Replaced in tests to control timers; production
   * leaves it unset and the binding uses `createAnalysisJobSession`.
   */
  createSession?: (
    options: AnalysisJobSessionOptions,
  ) => AnalysisJobSession;
  pollIntervalMs?: number;
  onState?: (state: AnalysisJobViewState) => void;
  /** Where to write the start, failure and result patches, if anywhere. */
  applyPatch?: (
    patch:
      | StudioAnalysisStartStatePatch
      | StudioAnalysisFailureStatePatch
      | StudioAnalysisResultSinkPatch,
  ) => void;
}

/**
 * What the run produced. A failure names the stage it happened at, because a
 * request refused before submission and a job that ran and failed need
 * different things from the caller.
 */
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

/**
 * Name the analysis a selection asks for.
 *
 * @param selection - The selection.
 * @returns Its analysis kind.
 */
function selectionKind(selection: AnalysisJobSelection): AnalysisJobKind {
  return selection.analysis;
}

/**
 * Run one analysis job and turn its result into a patch.
 *
 * Progress is not invented: there is no percentage and no estimate, because
 * the job route reports neither, and a made-up one would be read as measured.
 *
 * @param input - The simulation configuration and the analysis to run.
 * @param options - The API, the poll interval, and the seams a caller may
 *   replace. `applyPatch` is called with the start patch before submitting and
 *   with the result or failure patch at the end, so a store follows the job
 *   without this module importing one.
 * @returns The patch and the terminal state, or the refusal with the stage it
 *   happened at.
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
    // `startJob` is fire-and-forget by contract -- it returns void and the
    // session continues through polls -- so it is called as a statement. The
    // two microtask ticks that follow are the ones the promise chain this
    // replaced took, and they matter: the submit it began must have run before
    // the state below is read.
    binding.startJob(request);
    await Promise.resolve().then(() => undefined);

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
 * Whether a new job may be started from this state.
 *
 * The same policy as `canSubmitAnalysisJob`, exposed here so a store delegate
 * can ask the runner rather than reach into the session module.
 *
 * @param state - The current state; an absent one counts as idle.
 * @returns Whether a start is allowed.
 */
export function canStartStudioAnalysisJob(
  state: AnalysisJobViewState = initialAnalysisJobState(),
): boolean {
  return !isAnalysisJobBusy(state.phase) && state.phase !== "submitting";
}
