// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — What the store says the guided workflow has accomplished

import { studioBundleIsCurrent } from "./studioBundleContext";

/**
 * Deciding what the guided workflow may call done.
 *
 * Every rule here answers the same question in a different place: *is this
 * step's evidence about the experiment currently configured, and did the run
 * that produced it actually succeed?* Two answers used to be taken on trust
 * and both overstated:
 *
 * - **An epoch marked training complete.** A run that emitted one epoch and
 *   then failed, or was stopped, or is still going, satisfied the step.
 *   Completion now comes from the terminal status, and `stopped` is terminal
 *   but not complete -- the reader ended that run early.
 * - **A non-null synthesis result marked the ODE branch successful.** A
 *   synthesis that reported failure ticked the step. Both branches now require
 *   the run to say it succeeded.
 *
 * The third rule is about time rather than truthfulness: a trace or an
 * analysis is evidence of the experiment that produced it, so results carry
 * the experiment they were run under and stop counting once the reader changes
 * the model, a parameter, the step, or the protocol. They are still shown --
 * the reader's work is not thrown away -- they simply stop standing in for
 * work that has not been done.
 *
 * Nothing here invents applicability: co-simulation applies to catalogue
 * models only, and that is a property of the source mode rather than of any
 * result.
 *
 * Failure is kept apart from "not yet done". A stage whose latest attempt for
 * the current experiment failed is reported with that attempt's message, from
 * the store's record of it or from the result itself: a training run that
 * failed or was interrupted, a co-simulation of the current RTL that is not
 * bit-exact, a synthesis that reported failure. A failure recorded under an
 * earlier experiment is not shown -- it was about inputs the reader has since
 * changed.
 */

import type { GuidedFlowInputs, GuidedFlowStepKey } from "./guidedFlowState";
import {
  studioExperimentKey,
  studioPrecisionKey,
  studioResultIsCurrent,
  studioTrainingKey,
} from "./studioExperimentKey";
import { studioSimulationConfigInput } from "./studioSimulationConfigInput";
import type { StudioGuidedFlowSource } from "./studioGuidedFlowSource";

/**
 * What the reader has decided, which no result can tell us.
 */
export interface StudioGuidedFlowDecisions {
  /** Whether the reader has chosen to skip the optional training step. */
  trainingSkipped: boolean;
  /** Whether the session's own evidence cart satisfies the export step. */
  evidenceExportSatisfied: boolean;
}

/**
 * Read what the store says the guided workflow has accomplished.
 *
 * @param state - The store's current state.
 * @param decisions - What the reader has decided rather than produced.
 * @returns The inputs the flow's state machine is computed from.
 */
export function studioGuidedFlowInputs(
  state: StudioGuidedFlowSource,
  decisions: StudioGuidedFlowDecisions,
): GuidedFlowInputs {
  const currentExperimentKey = studioExperimentKey(studioSimulationConfigInput(state));
  const analysisResults = [
    state.fiResult,
    state.bifResult,
    state.sensResult,
    state.heatmapResult,
    state.compareResult,
    state.nullclineResult,
    state.freqResult,
    state.charResult,
  ];
  return {
    analysisComplete: (analysisResults.some((analysis) => analysis !== null)
      && studioResultIsCurrent(state.analysisExperimentKey, currentExperimentKey))
      || (state.precResult !== null && studioResultIsCurrent(
        state.analysisExperimentKey,
        studioPrecisionKey(studioSimulationConfigInput(state), state.modelQFormat),
      )),
    compileComplete: state.compileTraceability !== null,
    cosimApplicable: state.sourceMode === "model",
    cosimComplete: cosimMatchesCompile(state),
    evidenceExported: studioBundleIsCurrent("project", state)
      || studioBundleIsCurrent("compile", state)
      || studioBundleIsCurrent("synthesis", state)
      || decisions.evidenceExportSatisfied,
    modelSelected: state.sourceMode === "ode"
      ? state.equations.length > 0
      : state.selectedModelName.length > 0,
    simulationComplete: state.result !== null
      && studioResultIsCurrent(state.resultExperimentKey, currentExperimentKey),
    synthesisComplete: studioSynthesisComplete(state),
    trainingComplete: state.trainingStatus === "completed"
      && studioResultIsCurrent(
        state.trainingExperimentKey,
        studioTrainingKey(state.trainingConfig),
      ),
    trainingSkipped: decisions.trainingSkipped,
    failures: studioGuidedFlowFailures(state, currentExperimentKey),
  };
}

/**
 * Why each stage's latest attempt for the current experiment failed.
 *
 * @param state - The store's current state.
 * @param currentExperimentKey - The experiment the store now describes.
 * @returns The failure message per failed stage; a recorded failure comes
 *   first, a failure read from a result fills a stage it left empty.
 */
export function studioGuidedFlowFailures(
  state: StudioGuidedFlowSource,
  currentExperimentKey: string,
): Partial<Record<GuidedFlowStepKey, string>> {
  const failures: Partial<Record<GuidedFlowStepKey, string>> = {};
  const recorded = state.stageFailure;
  if (recorded !== null && recorded.experimentKey === currentExperimentKey) {
    failures[recorded.stage] = recorded.message;
  }
  if ((state.trainingStatus === "failed" || state.trainingStatus === "interrupted")
    && studioResultIsCurrent(state.trainingExperimentKey, studioTrainingKey(state.trainingConfig))) {
    failures.train ??= `Training run ${state.trainingStatus}`;
  }
  if (state.cosimResult?.bit_exact === false
    && state.compileTraceability !== null
    && state.cosimResult.rtl.source_sha256 === state.compileTraceability.output.rtl_sha256) {
    failures.cosim ??= "RTL co-simulation of the compiled design is not bit-exact";
  }
  const synthesis = studioSynthesisFailure(state);
  if (synthesis !== null) {
    failures.synthesise ??= synthesis;
  }
  return failures;
}

/**
 * Why the synthesis on record failed, if it did.
 *
 * @param state - The store's current state.
 * @returns The failure message, or `null` when synthesis succeeded or has
 *   not reported an outcome.
 */
export function studioSynthesisFailure(state: StudioGuidedFlowSource): string | null {
  if (studioSynthesisComplete(state)) {
    return null;
  }
  if (state.sourceMode === "model") {
    const terminal = state.synthResult?.silicon_terminal;
    return terminal === undefined
      ? null
      : terminal.place_and_route?.error ?? "Synthesis/PnR terminal did not complete";
  }
  if (state.synthResult !== null) {
    return state.synthResult.error ?? "Synthesis did not complete";
  }
  const targets = Object.keys(state.multiTargetResult?.targets ?? {});
  return targets.length > 0 ? "No synthesis target succeeded" : null;
}

/**
 * Whether a synthesis run reported success.
 *
 * A synthesis result is an outcome, not a verdict. The ODE branch used to
 * complete the step on a non-null result, so a synthesis that reported failure
 * ticked it; both branches require the run to say it succeeded, and a
 * multi-target run succeeds when at least one target did.
 *
 * @param state - The store's current state.
 * @returns Whether the synthesis step is complete.
 */
export function studioSynthesisComplete(state: StudioGuidedFlowSource): boolean {
  if (state.sourceMode === "model") {
    return state.synthResult?.silicon_terminal?.success === true;
  }
  return state.synthResult?.success === true
    || Object.values(state.multiTargetResult?.targets ?? {}).some((target) => target.success);
}

/**
 * Whether the co-simulation on record is a parity report for the RTL on record.
 *
 * A parity report for RTL that has since been recompiled says nothing about
 * what would be synthesised now, so the digests must agree.
 *
 * @param state - The store's current state.
 * @returns Whether co-simulation parity holds for the current compile.
 */
function cosimMatchesCompile(state: StudioGuidedFlowSource): boolean {
  return state.cosimResult?.bit_exact === true
    && state.compileTraceability !== null
    && state.cosimResult.rtl.source_sha256 === state.compileTraceability.output.rtl_sha256;
}
