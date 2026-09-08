// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — What the store says the guided workflow has accomplished

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
 */

import type { GuidedFlowInputs } from "./guidedFlowState";
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
    evidenceExported: state.evidenceBundle !== null
      || state.projectEvidenceBundle !== null
      || state.compileEvidenceBundle !== null
      || state.synthesisEvidenceBundle !== null
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
  };
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
