// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided operator run controller

/**
 * What the guided panel's single button does next.
 *
 * The whole controller exists to answer one question -- what should happen when
 * the reader presses the one button -- and to say why when the answer is
 * nothing. A blocked flow reports the first blocker the reader can actually
 * act on: a step blocked merely because its predecessor is not done is not
 * actionable, so it is skipped in favour of one that names a real obstacle,
 * and a capability's own message is preferred over the flow's generic one
 * because it says what the deployment is missing.
 *
 * Every action is awaited and its failure returned rather than thrown. The
 * caller is a click handler, and a rejected promise there is an unhandled
 * rejection with nothing naming the step that failed.
 */
import type { GuidedFlowState, GuidedFlowStepKey } from "./guidedFlowState";

/** What the button will do, including doing nothing and saying why. */
export type GuidedRunActionKey =
  | "blocked"
  | "complete"
  | "export-evidence"
  | "run-analysis"
  | "run-compile"
  | "run-cosim"
  | "run-simulation"
  | "run-synthesis"
  | "skip-training";

/** The work each step performs, supplied by whoever owns it. */
export interface GuidedRunActions {
  exportEvidence: () => Promise<void>;
  runAnalysis: () => Promise<void>;
  runCompile: () => Promise<void>;
  runCosim: () => Promise<void>;
  runSimulation: () => Promise<void>;
  runSynthesis: () => Promise<void>;
  skipTraining: () => Promise<void>;
}

/** The flow, and what the deployment can currently do. */
export interface GuidedRunControllerInputs {
  capabilityMessages?: Partial<Record<GuidedFlowStepKey, string>>;
  compileConfigured?: boolean;
  cosimConfigured?: boolean;
  exportReady: boolean;
  flow: GuidedFlowState;
  sourceMode: "model" | "ode";
}

/** Whether the step ran, and the message when it did not. */
export interface GuidedRunResult {
  error?: string;
  ok: boolean;
}

/** What the panel renders, and the call behind its button. */
export interface GuidedRunController {
  blockerReason: string | null;
  completedEvidence: string[];
  exportReady: boolean;
  nextActionKey: GuidedRunActionKey;
  nextActionLabel: string;
  runNextStep: () => Promise<GuidedRunResult>;
}

/** The decided action: its key, its label, and any blocker. */
interface GuidedRunPlan {
  blockerReason: string | null;
  key: GuidedRunActionKey;
  label: string;
}

/** What each completed step is called in the evidence summary. */
const STEP_EVIDENCE_LABELS: Record<GuidedFlowStepKey, string> = {
  analyse: "Analyse",
  compile: "Compile",
  cosim: "Co-sim parity",
  design: "Design",
  export: "Export evidence",
  simulate: "Simulate",
  synthesise: "Synthesise",
  train: "Train",
};

/**
 * Decide what the guided panel should offer next.
 *
 * @param inputs - The flow and what the deployment can do.
 * @param actions - The work each step performs.
 * @returns The controller. `runNextStep` performs whatever the plan chose,
 *   and returns its failure rather than throwing it.
 */
export function buildGuidedRunController(
  inputs: GuidedRunControllerInputs,
  actions: GuidedRunActions,
): GuidedRunController {
  const plan = guidedRunPlan(inputs);
  return {
    blockerReason: plan.blockerReason,
    completedEvidence: inputs.flow.steps
      .filter((step) => step.status === "completed")
      .map((step) => STEP_EVIDENCE_LABELS[step.key]),
    exportReady: inputs.exportReady,
    nextActionKey: plan.key,
    nextActionLabel: plan.label,
    runNextStep: () => runPlannedAction(plan, actions),
  };
}

/**
 * Choose the action for the flow as it stands.
 *
 * @param inputs - The flow and what the deployment can do.
 * @returns The plan: the current step's action, an actionable blocker, or
 *   completion.
 */
function guidedRunPlan(inputs: GuidedRunControllerInputs): GuidedRunPlan {
  const current = inputs.flow.steps.find((step) => step.status === "current") ?? null;
  if (current !== null) {
    return currentStepPlan(current.key, inputs);
  }
  const blocker = firstActionableBlocker(inputs);
  if (blocker !== null) {
    return { blockerReason: blocker, key: "blocked", label: "Resolve blocker" };
  }
  return { blockerReason: null, key: "complete", label: "Workflow complete" };
}

/**
 * Choose the action for a step that is current.
 *
 * @param stepKey - The current step.
 * @param inputs - The flow and what the deployment can do.
 * @returns The plan for it, which may still be blocked when the step is
 *   current but its prerequisites in the deployment are not configured.
 */
function currentStepPlan(
  stepKey: GuidedFlowStepKey,
  inputs: GuidedRunControllerInputs,
): GuidedRunPlan {
  switch (stepKey) {
    case "design":
      return { blockerReason: "Choose or enter a design before running.", key: "blocked", label: "Choose design" };
    case "simulate":
      return { blockerReason: null, key: "run-simulation", label: "Run simulation" };
    case "analyse":
      return { blockerReason: null, key: "run-analysis", label: "Run f-I analysis" };
    case "train":
      return { blockerReason: null, key: "skip-training", label: "Skip training" };
    case "compile":
      if (inputs.compileConfigured === false) {
        return {
          blockerReason: "Selected model has no canonical schema-backed RTL path.",
          key: "blocked",
          label: "Resolve blocker",
        };
      }
      return { blockerReason: null, key: "run-compile", label: "Compile RTL" };
    case "cosim":
      if (inputs.cosimConfigured === false) {
        return {
          blockerReason: "Selected integrator has no bit-exact co-simulation path.",
          key: "blocked",
          label: "Resolve blocker",
        };
      }
      return { blockerReason: null, key: "run-cosim", label: "Run RTL co-sim" };
    case "synthesise":
      return { blockerReason: null, key: "run-synthesis", label: "Run synthesis" };
    case "export":
      if (!inputs.exportReady) {
        return {
          blockerReason: "Evidence export is not ready yet.",
          key: "blocked",
          label: "Resolve blocker",
        };
      }
      return { blockerReason: null, key: "export-evidence", label: "Export evidence" };
  }
}

/**
 * Find the first blocker the reader can do something about.
 *
 * A step blocked only because its predecessor is unfinished is not
 * actionable and is skipped: telling someone their workflow is blocked
 * because they have not done the previous step is not help.
 *
 * @param inputs - The flow and what the deployment can do.
 * @returns The reason, preferring a capability's own message, or `null`.
 */
function firstActionableBlocker(inputs: GuidedRunControllerInputs): string | null {
  const firstBlocked = inputs.flow.steps.find(
    (step) =>
      step.status === "blocked"
      && step.blockedReason !== null
      && !step.blockedReason.startsWith("Requires "),
  );
  if (firstBlocked === undefined) {
    return null;
  }
  return inputs.capabilityMessages?.[firstBlocked.key] ?? firstBlocked.blockedReason;
}

/**
 * Perform the planned action.
 *
 * @param plan - What was chosen.
 * @param actions - The work each step performs.
 * @returns Whether it succeeded, with the message when it did not. A
 *   failure is returned, never thrown: the caller is a click handler.
 */
async function runPlannedAction(
  plan: GuidedRunPlan,
  actions: GuidedRunActions,
): Promise<GuidedRunResult> {
  try {
    switch (plan.key) {
      case "run-analysis":
        await actions.runAnalysis();
        return { ok: true };
      case "run-compile":
        await actions.runCompile();
        return { ok: true };
      case "run-cosim":
        await actions.runCosim();
        return { ok: true };
      case "run-simulation":
        await actions.runSimulation();
        return { ok: true };
      case "run-synthesis":
        await actions.runSynthesis();
        return { ok: true };
      case "skip-training":
        await actions.skipTraining();
        return { ok: true };
      case "export-evidence":
        await actions.exportEvidence();
        return { ok: true };
      case "blocked":
        return { error: plan.blockerReason ?? "Guided run is blocked.", ok: false };
      case "complete":
        return { ok: true };
    }
  } catch (error: unknown) {
    return {
      error: error instanceof Error && error.message.length > 0 ? error.message : String(error),
      ok: false,
    };
  }
}
