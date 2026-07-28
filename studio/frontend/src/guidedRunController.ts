// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided operator run controller
import type { GuidedFlowState, GuidedFlowStepKey } from "./guidedFlowState";

export type GuidedRunActionKey =
  | "blocked"
  | "complete"
  | "export-evidence"
  | "run-analysis"
  | "run-compile"
  | "run-simulation"
  | "run-synthesis"
  | "skip-training";

export interface GuidedRunActions {
  exportEvidence: () => Promise<void>;
  runAnalysis: () => Promise<void>;
  runCompile: () => Promise<void>;
  runSimulation: () => Promise<void>;
  runSynthesis: () => Promise<void>;
  skipTraining: () => Promise<void>;
}

export interface GuidedRunControllerInputs {
  capabilityMessages?: Partial<Record<GuidedFlowStepKey, string>>;
  compileConfigured?: boolean;
  exportReady: boolean;
  flow: GuidedFlowState;
  sourceMode: "model" | "ode";
}

export interface GuidedRunResult {
  error?: string;
  ok: boolean;
}

export interface GuidedRunController {
  blockerReason: string | null;
  completedEvidence: string[];
  exportReady: boolean;
  nextActionKey: GuidedRunActionKey;
  nextActionLabel: string;
  runNextStep: () => Promise<GuidedRunResult>;
}

interface GuidedRunPlan {
  blockerReason: string | null;
  key: GuidedRunActionKey;
  label: string;
}

const STEP_EVIDENCE_LABELS: Record<GuidedFlowStepKey, string> = {
  analyse: "Analyse",
  compile: "Compile",
  design: "Design",
  export: "Export evidence",
  simulate: "Simulate",
  synthesise: "Synthesise",
  train: "Train",
};

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
