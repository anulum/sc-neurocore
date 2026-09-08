// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — React integration for async analysis jobs (W12-E)

/**
 * The one hook a panel needs to run an analysis job.
 *
 * It joins three things that are separately testable: resolving what the
 * reader selected, holding the session that runs it, and turning a completed
 * result into a store patch. The join is the only part that needs React, so it
 * is the only part that lives in a hook -- `resolveStudioAnalysisJobIntegration`
 * and `applyCompletedAnalysisJobResult` are plain functions, and the tests use
 * them directly.
 *
 * Submission is gated three ways and all three must agree: the session must be
 * free, the capability must be enabled, and the request must have built. A
 * disabled submit button therefore always has a stated reason available.
 *
 * The store is written only through `applyPatch`, which a caller may omit.
 */

import { useEffect, useMemo, useRef } from "react";
import { studioExperimentKey } from "./studioExperimentKey";
import { canonicalSealText } from "./evidenceSeal";

import type { AnalysisJobKind, AnalysisJobRequestBody } from "./api/client";
import { isAnalysisJobBusy, type AnalysisJobViewState } from "./analysisJob";
import {
  buildAnalysisJobRequest,
  type AnalysisJobRequestBuildResult,
  type AnalysisJobSelection,
} from "./analysisJobRequest";
import type { AnalysisJobWorkbenchProps } from "./components/AnalysisJobWorkbench";
import {
  studioAnalysisFailureState,
  type StudioAnalysisFailureStatePatch,
} from "./studioAnalysisState";
import { buildStudioAnalysisJobSelection } from "./studioAnalysisJobSelection";
import {
  studioAnalysisResultSink,
  type StudioAnalysisResultSinkPatch,
} from "./studioAnalysisResultSink";
import type { StudioSimulationConfigInput } from "./studioSimulationConfig";
import {
  useAnalysisJob,
  type UseAnalysisJobOptions,
  type UseAnalysisJobResult,
} from "./useAnalysisJob";

/** What this hook writes to the store: a result, or a failure. */
export type StudioAnalysisJobIntegrationPatch =
  | ((StudioAnalysisFailureStatePatch | StudioAnalysisResultSinkPatch) & {
    analysisExperimentKey?: string | null; heatmapExperimentKey?: string | null;
  })
  | { analysisExperimentKey: null; heatmapExperimentKey?: null };

/** What the panel has selected, as the reader left it. */
export interface StudioAnalysisJobIntegrationInput {
  simulation: StudioSimulationConfigInput;
  analysis: AnalysisJobKind;
  sweepParam: string;
  sweepParamY: string;
}

/** The gates, the patch sink, and anything the session needs. */
export interface UseStudioAnalysisJobIntegrationOptions {
  disabled?: boolean;
  /**
   * Whether the capability governing this panel is enabled. Omitted counts as
   * enabled, so a caller that does not consult the registry is not silently
   * locked out.
   */
  capabilityEnabled?: boolean;
  applyPatch?: (patch: StudioAnalysisJobIntegrationPatch) => void;
  hookOptions?: UseAnalysisJobOptions;
}

/** What the panel shows before anything is submitted. */
export interface StudioAnalysisJobIntegrationResolved {
  selection: AnalysisJobSelection | null;
  selectionError: string | null;
  selectedAnalysisLabel: string | null;
  request: AnalysisJobRequestBuildResult;
  disabled: boolean;
  workbenchProps: AnalysisJobWorkbenchProps | null;
}

/**
 * Work out what the panel should show and whether it may submit.
 *
 * @param input - The simulation configuration, the analysis, and the sweep
 *   names as typed.
 * @param options - Whether the panel is disabled, and whether the capability
 *   governing it is enabled. An omitted capability counts as enabled.
 * @returns The resolved selection, the built request, whether submit is
 *   disabled, and the workbench's props -- `null` when the selection did not
 *   resolve, so the workbench is not rendered against a selection that failed.
 */
export function resolveStudioAnalysisJobIntegration(
  input: StudioAnalysisJobIntegrationInput,
  options: Pick<
    UseStudioAnalysisJobIntegrationOptions,
    "disabled" | "capabilityEnabled"
  > = {},
): StudioAnalysisJobIntegrationResolved {
  const selectionResult = buildStudioAnalysisJobSelection({
    analysis: input.analysis,
    sourceMode: input.simulation.sourceMode,
    modelParams: input.simulation.modelParams,
    odeParams: input.simulation.odeParams,
    sweepParam: input.sweepParam,
    sweepParamY: input.sweepParamY,
  });
  if (!selectionResult.ok) {
    return {
      selection: null,
      selectionError: selectionResult.error,
      selectedAnalysisLabel: null,
      request: { ok: false, error: selectionResult.error },
      disabled: true,
      workbenchProps: null,
    };
  }
  const request = buildAnalysisJobRequest(
    input.simulation,
    selectionResult.selection,
  );
  const capabilityEnabled = options.capabilityEnabled !== false;
  const disabled =
    Boolean(options.disabled) || !capabilityEnabled || !request.ok;
  return {
    selection: selectionResult.selection,
    selectionError: null,
    selectedAnalysisLabel: selectionResult.label,
    request,
    disabled,
    workbenchProps: {
      simulationInput: input.simulation,
      selection: selectionResult.selection,
      selectedAnalysisLabel: selectionResult.label,
    },
  };
}

/**
 * Whether submit is allowed, given all three gates.
 *
 * @param input - Whether the session is free, whether the panel is disabled,
 *   and whether the request built.
 * @returns Whether the reader may submit.
 */
export function studioAnalysisJobIntegrationCanSubmit(input: {
  sessionCanSubmit: boolean;
  disabled: boolean;
  requestOk: boolean;
}): boolean {
  return input.sessionCanSubmit && !input.disabled && input.requestOk;
}

/**
 * Turn a completed job's result into a patch, if it is one.
 *
 * A job that has not completed is not an error here: the answer is simply that
 * nothing was applied. A job that completed and returned an unreadable result
 * is an error, and its failure patch is applied so the panel says so.
 *
 * @param input - The analysis that was run, the terminal state, and where to
 *   write the patch.
 * @returns Whether a result was applied, and the refusal when one was read and
 *   rejected.
 */
export function applyCompletedAnalysisJobResult(input: {
  kind: AnalysisJobKind;
  state: AnalysisJobViewState;
  applyPatch?: (patch: StudioAnalysisFailureStatePatch | StudioAnalysisResultSinkPatch) => void;
}): { applied: boolean; error: string | null } {
  if (input.state.phase !== "completed" || input.state.result === null) {
    return { applied: false, error: null };
  }
  const sunk = studioAnalysisResultSink(input.kind, input.state.result);
  if (!sunk.ok) {
    input.applyPatch?.(studioAnalysisFailureState(sunk.error));
    return { applied: false, error: sunk.error };
  }
  input.applyPatch?.(sunk.patch);
  return { applied: true, error: null };
}

/**
 * Everything the panel renders from: the resolved selection, the running
 * job's state, the gates, and the call that starts one.
 */
export interface UseStudioAnalysisJobIntegrationResult {
  selection: AnalysisJobSelection | null;
  selectionError: string | null;
  selectedAnalysisLabel: string | null;
  request: AnalysisJobRequestBuildResult;
  disabled: boolean;
  busy: boolean;
  canSubmit: boolean;
  state: AnalysisJobViewState;
  startJob: (request: AnalysisJobRequestBody) => void;
  workbenchProps: AnalysisJobWorkbenchProps | null;
  session: UseAnalysisJobResult;
}

/**
 * Run an analysis job from a panel.
 *
 * @param input - The simulation configuration, the analysis, and the sweep
 *   names as typed.
 * @param options - Whether the panel is disabled, the capability gate, where
 *   to write patches, and anything the session needs.
 * @returns Everything the panel renders from, and `startJob` to begin.
 */
export function useStudioAnalysisJobIntegration(
  input: StudioAnalysisJobIntegrationInput,
  options: UseStudioAnalysisJobIntegrationOptions = {},
): UseStudioAnalysisJobIntegrationResult {
  const resolved = useMemo(
    () =>
      resolveStudioAnalysisJobIntegration(input, {
        disabled: options.disabled,
        capabilityEnabled: options.capabilityEnabled,
      }),
    [
      input.analysis,
      input.simulation,
      input.sweepParam,
      input.sweepParamY,
      options.disabled,
      options.capabilityEnabled,
    ],
  );

  const workbenchProps = useMemo((): AnalysisJobWorkbenchProps | null => {
    if (resolved.workbenchProps === null) return null;
    return { ...resolved.workbenchProps, hookOptions: options.hookOptions };
  }, [resolved.workbenchProps, options.hookOptions]);

  const session = useAnalysisJob(options.hookOptions ?? {});
  const appliedKeyRef = useRef<string | null>(null);
  const submittedRef = useRef<{ experiment: string; kind: AnalysisJobKind; pending: boolean } | null>(null);
  const applyPatch = options.applyPatch;

  useEffect(() => {
    if (
      session.state.phase === "idle"
      || session.state.phase === "submitting"
      || session.state.phase === "pending"
    ) {
      appliedKeyRef.current = null;
    }
    const submitted = submittedRef.current;
    if (session.state.phase !== "idle" && !isAnalysisJobBusy(session.state.phase)) {
      if (submitted) submitted.pending = false;
    }
    if (submitted === null) return;
    if (session.state.phase !== "completed" || session.state.result === null) {
      return;
    }
    const key =
      `${session.state.jobId ?? "none"}:${submitted.kind}`;
    if (appliedKeyRef.current === key) return;
    submitted.pending = false;
    appliedKeyRef.current = key;
    try {
      if (studioExperimentKey(input.simulation) !== submitted.experiment
        || session.state.analysis !== submitted.kind) return;
    } catch { return; }
    const outcome = applyCompletedAnalysisJobResult({
      kind: submitted.kind,
      state: session.state,
      applyPatch: (patch) => {
        const completed = "error" in patch && patch.error === null;
        applyPatch?.({ ...patch, analysisExperimentKey: completed ? submitted.experiment : null,
          ...(submitted.kind === "heatmap" ? { heatmapExperimentKey: completed ? submitted.experiment : null } : {}) });
      },
    });
    if (outcome.applied || outcome.error !== null) {
      appliedKeyRef.current = key;
    }
  }, [applyPatch, input.simulation, session.state]);

  const canSubmit = studioAnalysisJobIntegrationCanSubmit({
    sessionCanSubmit: session.canSubmit,
    disabled: resolved.disabled,
    requestOk: resolved.request.ok,
  });

  return {
    selection: resolved.selection,
    selectionError: resolved.selectionError,
    selectedAnalysisLabel: resolved.selectedAnalysisLabel,
    request: resolved.request,
    disabled: resolved.disabled,
    busy: session.busy,
    canSubmit,
    state: session.state,
    startJob: (request) => {
      if (!canSubmit || submittedRef.current?.pending || !resolved.request.ok) return;
      try {
        if (canonicalSealText(request) !== canonicalSealText(resolved.request.value)) return;
        submittedRef.current = { experiment: studioExperimentKey(input.simulation), kind: request.analysis, pending: true };
        appliedKeyRef.current = null;
        applyPatch?.({ analysisExperimentKey: null,
          ...(request.analysis === "heatmap" ? { heatmapExperimentKey: null } : {}) });
        session.startJob(request);
      } catch (error: unknown) {
        submittedRef.current = null;
        applyPatch?.(studioAnalysisFailureState(error));
      }
    },
    workbenchProps,
    session,
  };
}
