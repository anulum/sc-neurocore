// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — React integration for async analysis jobs (W12-E)

/**
 * Binds W12-A selection, W08 session, and W12-B sink for App hosts.
 * Patches only via optional applyPatch; returns memoised control/workbench props.
 */

import { useEffect, useMemo, useRef } from "react";

import type { AnalysisJobKind, AnalysisJobRequestBody } from "./api/client";
import type { AnalysisJobViewState } from "./analysisJob";
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

export type StudioAnalysisJobIntegrationPatch =
  | StudioAnalysisFailureStatePatch
  | StudioAnalysisResultSinkPatch;

export interface StudioAnalysisJobIntegrationInput {
  simulation: StudioSimulationConfigInput;
  analysis: AnalysisJobKind;
  sweepParam: string;
  sweepParamY: string;
}

export interface UseStudioAnalysisJobIntegrationOptions {
  disabled?: boolean;
  /** Capability gate; false disables submit. Omitted defaults to enabled. */
  capabilityEnabled?: boolean;
  applyPatch?: (patch: StudioAnalysisJobIntegrationPatch) => void;
  hookOptions?: UseAnalysisJobOptions;
}

export interface StudioAnalysisJobIntegrationResolved {
  selection: AnalysisJobSelection | null;
  selectionError: string | null;
  selectedAnalysisLabel: string | null;
  request: AnalysisJobRequestBuildResult;
  disabled: boolean;
  workbenchProps: AnalysisJobWorkbenchProps | null;
}

/** Pure resolve: selection + request + disabled + workbench props. */
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

/** Effective canSubmit (session ∧ capability ∧ request). */
export function studioAnalysisJobIntegrationCanSubmit(input: {
  sessionCanSubmit: boolean;
  disabled: boolean;
  requestOk: boolean;
}): boolean {
  return input.sessionCanSubmit && !input.disabled && input.requestOk;
}

/** Apply a completed session result through the W12-B sink. */
export function applyCompletedAnalysisJobResult(input: {
  kind: AnalysisJobKind;
  state: AnalysisJobViewState;
  applyPatch?: (patch: StudioAnalysisJobIntegrationPatch) => void;
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

/** Memoised resolve + W08 session + sink on completed. */
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
  const applyPatch = options.applyPatch;

  useEffect(() => {
    if (
      session.state.phase === "idle"
      || session.state.phase === "submitting"
      || session.state.phase === "pending"
    ) {
      appliedKeyRef.current = null;
    }
    if (resolved.selection === null) return;
    if (session.state.phase !== "completed" || session.state.result === null) {
      return;
    }
    const key =
      `${session.state.jobId ?? "none"}:${resolved.selection.analysis}`;
    if (appliedKeyRef.current === key) return;
    const outcome = applyCompletedAnalysisJobResult({
      kind: resolved.selection.analysis,
      state: session.state,
      applyPatch,
    });
    if (outcome.applied || outcome.error !== null) {
      appliedKeyRef.current = key;
    }
  }, [applyPatch, resolved.selection, session.state]);

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
    startJob: session.startJob,
    workbenchProps,
    session,
  };
}
